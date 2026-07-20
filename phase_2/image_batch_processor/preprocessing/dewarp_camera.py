"""Standalone camera-model page dewarping (Option B) — a SEPARATE, OPTIONAL step.

This module is **not** part of the preprocessing pipeline (it is not a
``PreprocessingStage``, is not registered in the factory, and has no
``StageConfig``). It is a self-contained, optionally-parallel batch process you
run *on its own* — typically over the pipeline's ``output/`` directory — to
rectify page geometry using the full camera-reconstruction technique described
in ``docs/dewarp-option-b.md``.

Why a separate process
----------------------
Unlike the pipeline's Option A dewarp (a fast vertical text-line remap), Option
B recovers a parametric 3D scene model and is comparatively heavy: it runs a
nonlinear least-squares optimisation (``scipy.optimize.minimize``) per page. So
it lives outside the pipeline and can be run selectively, in parallel across
CPU cores, when the lighter correction is not enough.

The model (Matt Zucker's ``page_dewarp`` family)
------------------------------------------------
The source of the distortion is the camera's position/orientation relative to
the page. We model the page as a 3D surface that is flat vertically but curls
horizontally as a **cubic** (the spine curl), viewed by a pinhole **camera**
with unknown rotation (``rvec``), translation (``tvec``) and a fixed normalised
focal length. Detected text-span points are the observations; we solve for the
model parameters that best reproject them (least squares), then remap the image
off the recovered flat surface — correcting perspective (both axes) and
curvature together.

Pipeline of one page:

1. Detect text as ordered point sequences per line ("spans").
2. Estimate a page x/y axis (PCA over spans) and four corner extents.
3. ``solvePnP`` for an initial camera pose; cubic slopes start at zero.
4. ``scipy.optimize.minimize`` (Powell) refines pose + cubic + per-span
   coordinates so projected keypoints match detected ones.
5. Recover page dimensions and remap the full-resolution image.

If too few spans are found or the optimisation degenerates, the page is skipped
(or copied through unchanged, depending on ``--on-failure``) and logged, so a
run over a mixed folder never crashes on decorative/photo pages.

Usage::

    uv run python dewarp_camera.py --source output --output output_dewarped
    uv run python dewarp_camera.py --source output --output out --serial --limit 4

Dependencies: numpy, opencv, scipy, pillow (all already in this project's venv).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# --- parameter-vector layout (mirrors page_dewarp) -------------------------
# pvec = [ rvec(3) | tvec(3) | cubic slopes(2) | ycoords(nspans) | xcoords(N) ]
_RVEC = slice(0, 3)
_TVEC = slice(3, 6)
_CUBIC = slice(6, 8)
_N_HEAD = 8  # rvec + tvec + cubic

# Minimum word-blobs a chain must have to count as a text-line span.
_MIN_SPAN_BLOBS = 3

_SUPPORTED_MODES = frozenset({"L", "RGB"})
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})


@dataclass
class CameraDewarpConfig:
    """Tuning parameters for the standalone camera dewarp (all picklable)."""

    # Downscale width used for text-span detection (speed).
    max_detect_width: int = 1200
    # Minimum detected spans required to attempt the model; below this the page
    # is skipped/passed through (a decorative or too-sparse page).
    min_text_lines: int = 4
    # A span must cover at least this fraction of the detection width.
    min_line_width_ratio: float = 0.15
    # Normalised camera focal length (page_dewarp default).
    focal_length: float = 1.2
    # Fraction of the detection frame ignored as a margin when estimating the
    # page corner extents.
    page_margin_ratio: float = 0.02
    # Cap on optimiser iterations (Powell). Higher = better fit, slower.
    opt_max_iter: int = 2000


# --- coordinate transforms -------------------------------------------------

def _camera_matrix(focal_length: float) -> np.ndarray:
    """Intrinsic matrix K for a centred pinhole camera with the given focal length."""
    return np.array(
        [[focal_length, 0.0, 0.0], [0.0, focal_length, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pix2norm(shape: Tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """Map pixel coords (..., 1, 2) to a centred, roughly [-1, 1] normalised frame."""
    height, width = shape[:2]
    scale = 2.0 / max(height, width)
    offset = np.array([width, height], dtype=np.float64).reshape((-1, 1, 2)) * 0.5
    return (pts - offset) * scale


def _norm2pix(
    shape: Tuple[int, int], pts: np.ndarray, as_integer: bool = False
) -> np.ndarray:
    """Inverse of :func:`_pix2norm`: normalised coords back to pixel coords."""
    height, width = shape[:2]
    scale = max(height, width) * 0.5
    offset = np.array([0.5 * width, 0.5 * height], dtype=np.float64).reshape((-1, 1, 2))
    rval = pts * scale + offset
    return (rval + 0.5).astype(int) if as_integer else rval


# --- cubic-sheet + camera projection ---------------------------------------

def _project_xy(xy_coords: np.ndarray, pvec: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project flat-page (x, y) points through the cubic sheet + camera model.

    The page z-height is a cubic in x with boundary conditions f(0)=0, f'(0)=alpha,
    f(1)=0, f'(1)=beta (the two ``_CUBIC`` slopes); the resulting 3D points are
    projected with the estimated ``rvec``/``tvec`` and intrinsics ``K``.
    """
    alpha, beta = pvec[_CUBIC]
    # Clamp cubic slopes to a safe range (prevents runaway stretching).
    alpha = float(np.clip(alpha, -0.5, 0.5))
    beta = float(np.clip(beta, -0.5, 0.5))
    poly = np.array([alpha + beta, -2.0 * alpha - beta, alpha, 0.0])

    xy = xy_coords.reshape((-1, 2))
    z = np.polyval(poly, xy[:, 0])
    objpoints = np.hstack((xy, z.reshape((-1, 1)))).astype(np.float64)
    image_points, _ = cv2.projectPoints(
        objpoints, pvec[_RVEC], pvec[_TVEC], K, np.zeros(5)
    )
    return image_points


def _make_keypoint_index(span_counts: List[int]) -> np.ndarray:
    """Index array mapping each keypoint row to its (xcoord, ycoord) params.

    Faithful to page_dewarp's scheme: row 0 is the reserved origin keypoint;
    column 1 selects the span's shared ycoord parameter and column 0 selects the
    point's own xcoord parameter.
    """
    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts + 1, 2), dtype=int)
    start = 1
    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start : start + end, 1] = _N_HEAD + i
        start = end
    keypoint_index[1:, 0] = np.arange(npts) + _N_HEAD + nspans
    return keypoint_index


def _project_keypoints(
    pvec: np.ndarray, keypoint_index: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Gather the keypoints' (x, y) params from ``pvec`` and project them."""
    xy_coords = pvec[keypoint_index].copy()
    xy_coords[0, :] = 0
    return _project_xy(xy_coords, pvec, K)


# --- text-span detection (self-contained) ----------------------------------

def _group_blobs_into_spans(
    blobs: List[Tuple[float, float, int, int]]
) -> List[List[Tuple[float, float, int, int]]]:
    """Greedily chain word blobs (cx, cy, w, h) into text-line spans."""
    median_h = float(np.median([b[3] for b in blobs]))
    gap_limit = median_h * 8.0
    y_tol = median_h * 0.7

    order = sorted(range(len(blobs)), key=lambda k: blobs[k][0])
    parent = list(range(len(blobs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for oi, i in enumerate(order):
        cx_i, cy_i = blobs[i][0], blobs[i][1]
        for j in order[oi + 1 :]:
            cx_j, cy_j = blobs[j][0], blobs[j][1]
            gap = cx_j - cx_i
            if gap > gap_limit:
                break
            if gap > 0 and abs(cy_j - cy_i) <= y_tol:
                union(i, j)

    groups: Dict[int, list] = defaultdict(list)
    for i in range(len(blobs)):
        groups[find(i)].append(blobs[i])
    return list(groups.values())


def detect_text_spans(
    gray: np.ndarray, max_detect_width: int, min_line_width_ratio: float
) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """Detect text lines as ordered (x, y) point sequences ("spans").

    Returns ``(spans, scale, (det_h, det_w))`` where each span is an (M, 2)
    array of blob centroids in detection-pixel coords, sorted left-to-right, and
    the span list is sorted top-to-bottom. ``scale`` is detection/full-res.
    """
    height, width = gray.shape
    scale = 1.0
    if width > max_detect_width:
        scale = max_detect_width / width
        gray = cv2.resize(
            gray, (max_detect_width, max(1, int(round(height * scale))))
        )

    det_h, det_w = gray.shape
    block = max(3, (det_w // 30) | 1)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15
    )
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)), 1)
    bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), 1)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    max_blob_h = det_h * 0.05
    blobs: List[Tuple[float, float, int, int]] = []
    for i in range(1, count):
        _x, _y, w, h, area = stats[i]
        if 4 <= h <= max_blob_h and w >= 6 and area >= 20:
            blobs.append((float(centroids[i][0]), float(centroids[i][1]), w, h))

    if len(blobs) < _MIN_SPAN_BLOBS:
        return [], scale, (det_h, det_w)

    min_span_width = min_line_width_ratio * det_w
    spans: List[np.ndarray] = []
    for span in _group_blobs_into_spans(blobs):
        if len(span) < _MIN_SPAN_BLOBS:
            continue
        xs = np.array([b[0] for b in span], dtype=np.float64)
        ys = np.array([b[1] for b in span], dtype=np.float64)
        if float(xs.max() - xs.min()) < min_span_width:
            continue
        srt = np.argsort(xs)
        spans.append(np.column_stack([xs[srt], ys[srt]]))

    spans.sort(key=lambda p: float(p[:, 1].mean()))
    return spans, scale, (det_h, det_w)


# --- model assembly & solve -------------------------------------------------

def _keypoints_from_samples(
    span_points: List[np.ndarray], page_outline_norm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Estimate page x/y axes (PCA), four corner extents, and per-span coords."""
    all_evecs = np.zeros((1, 2))
    all_weights = 0.0
    for points in span_points:
        _, evec = cv2.PCACompute(
            points.reshape((-1, 2)).astype(np.float32), mean=None, maxComponents=1
        )
        weight = float(np.linalg.norm(points[-1] - points[0]))
        all_evecs += evec * weight
        all_weights += weight
    evec = all_evecs / all_weights
    x_dir = evec.flatten()
    if x_dir[0] < 0:
        x_dir = -x_dir
    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline_norm.astype(np.float32)).reshape((-1, 2))
    px = pagecoords @ x_dir
    py = pagecoords @ y_dir
    px0, px1 = float(px.min()), float(px.max())
    py0, py1 = float(py.min()), float(py.max())
    x_dir_coeffs = np.pad([px0, px1], 2, mode="symmetric")[2:].reshape(-1, 1)
    y_dir_coeffs = np.repeat([py0, py1], 2).reshape(-1, 1)
    corners = np.expand_dims((x_dir_coeffs * x_dir) + (y_dir_coeffs * y_dir), 1)

    xcoords: List[np.ndarray] = []
    ycoords: List[float] = []
    basis = np.transpose([x_dir, y_dir])
    for points in span_points:
        pts = points.reshape((-1, 2))
        proj = pts @ basis
        xcoords.append(proj[:, 0] - px0)
        ycoords.append(float(proj[:, 1].mean() - py0))
    return corners, np.array(ycoords), xcoords


def _default_params(
    corners: np.ndarray,
    ycoords: np.ndarray,
    xcoords: List[np.ndarray],
    K: np.ndarray,
) -> Tuple[Tuple[float, float], List[int], np.ndarray]:
    """Initial parameter vector: solvePnP pose, zero cubic slopes, span coords."""
    page_width = float(np.linalg.norm(corners[1] - corners[0]))
    page_height = float(np.linalg.norm(corners[-1] - corners[0]))
    corners_object3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [page_width, 0.0, 0.0],
            [page_width, page_height, 0.0],
            [0.0, page_height, 0.0],
        ]
    )
    _, rvec, tvec = cv2.solvePnP(
        corners_object3d, corners.astype(np.float64), K, np.zeros(5)
    )
    span_counts = [len(xc) for xc in xcoords]
    params = np.hstack(
        (
            np.array(rvec).flatten(),
            np.array(tvec).flatten(),
            np.array([0.0, 0.0]),
            ycoords.flatten(),
        )
        + tuple(xcoords)
    )
    return (page_width, page_height), span_counts, params


def _optimise(
    dstpoints: np.ndarray,
    keypoint_index: np.ndarray,
    params: np.ndarray,
    K: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Least-squares refine all params so projected keypoints match detected."""

    def objective(pvec: np.ndarray) -> float:
        projected = _project_keypoints(pvec, keypoint_index, K)
        return float(np.sum((dstpoints - projected) ** 2))

    result = minimize(
        objective,
        params,
        method="Powell",
        options={"maxiter": max_iter, "maxfev": max_iter * 100},
    )
    return result.x


def _optimise_page_dims(
    corners: np.ndarray,
    rough_dims: Tuple[float, float],
    params: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Refine (width, height) of the flat page so its far corner reprojects well."""
    dst_br = corners[2].flatten()

    def objective(dims_local: np.ndarray) -> float:
        proj_br = _project_xy(np.asarray(dims_local).reshape(1, 2), params, K)
        return float(np.sum((dst_br - proj_br.flatten()) ** 2))

    result = minimize(objective, np.array(rough_dims), method="Powell")
    return result.x


def _remap(
    full_image: np.ndarray,
    page_dims: np.ndarray,
    params: np.ndarray,
    K: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Sample the recovered flat page back through the model to rectify the image."""
    page_x_range = np.linspace(0, page_dims[0], out_w)
    page_y_range = np.linspace(0, page_dims[1], out_h)
    px, py = np.meshgrid(page_x_range, page_y_range)
    page_xy = np.hstack(
        (px.flatten().reshape((-1, 1)), py.flatten().reshape((-1, 1)))
    ).astype(np.float32)

    image_points = _project_xy(page_xy, params, K)
    image_points = _norm2pix(full_image.shape, image_points, as_integer=False)
    map_x = image_points[:, 0, 0].reshape(px.shape).astype(np.float32)
    map_y = image_points[:, 0, 1].reshape(py.shape).astype(np.float32)

    border = (255, 255, 255) if full_image.ndim == 3 else 255
    return cv2.remap(
        full_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


# --- top-level per-image entry point ---------------------------------------

def dewarp_image(
    image: Image.Image, config: CameraDewarpConfig
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Rectify one page with the camera model.

    Returns ``(result, None)`` on success, or ``(None, reason)`` when the page
    should be skipped (too few spans, degenerate solve). Never mutates ``image``.
    Output dimensions match the input.
    """
    source = image if image.mode in _SUPPORTED_MODES else image.convert("RGB")
    array = np.asarray(source)
    gray = array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    spans_det, _scale, det_shape = detect_text_spans(
        gray, config.max_detect_width, config.min_line_width_ratio
    )
    if len(spans_det) < config.min_text_lines:
        return None, (
            f"only {len(spans_det)} text span(s) detected "
            f"(need >= {config.min_text_lines})"
        )

    det_h, det_w = det_shape
    span_points = [
        _pix2norm(det_shape, pts.reshape((-1, 1, 2)).astype(np.float64))
        for pts in spans_det
    ]

    margin_x = det_w * config.page_margin_ratio
    margin_y = det_h * config.page_margin_ratio
    outline = np.array(
        [
            [margin_x, margin_y],
            [margin_x, det_h - margin_y],
            [det_w - margin_x, det_h - margin_y],
            [det_w - margin_x, margin_y],
        ],
        dtype=np.float64,
    ).reshape((-1, 1, 2))
    outline_norm = _pix2norm(det_shape, outline).reshape((-1, 2))

    K = _camera_matrix(config.focal_length)
    corners, ycoords, xcoords = _keypoints_from_samples(span_points, outline_norm)
    rough_dims, span_counts, params = _default_params(corners, ycoords, xcoords, K)

    dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))
    keypoint_index = _make_keypoint_index(span_counts)
    params = _optimise(dstpoints, keypoint_index, params, K, config.opt_max_iter)

    page_dims = _optimise_page_dims(corners, rough_dims, params, K)
    if not np.all(np.isfinite(page_dims)) or np.any(page_dims <= 0):
        page_dims = np.array(rough_dims)
    if not np.all(np.isfinite(page_dims)) or np.any(page_dims <= 0):
        return None, "degenerate page dimensions"

    height, width = array.shape[:2]
    result = _remap(array, page_dims, params, K, height, width)
    return Image.fromarray(result, mode=source.mode), None


# --- batch runner -----------------------------------------------------------

def _iter_source_images(source_dir: Path) -> List[Path]:
    """Return sorted image files directly in ``source_dir``."""
    return sorted(
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _process_file(
    task: Tuple[Path, Path, CameraDewarpConfig, str]
) -> Tuple[str, str, Optional[str]]:
    """Worker: dewarp one file. Returns (name, status, detail). Never raises."""
    src_path, out_path, config, on_failure = task
    try:
        with Image.open(src_path) as img:
            img.load()
            result, reason = dewarp_image(img, config)
        if result is None:
            if on_failure == "copy":
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.open(src_path).save(out_path)
                return (src_path.name, "copied", reason)
            return (src_path.name, "skipped", reason)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        return (src_path.name, "ok", None)
    except Exception as exc:  # keep the batch going on any single-page failure
        return (src_path.name, "error", str(exc))


def run_batch(
    source_dir: Path,
    output_dir: Path,
    config: CameraDewarpConfig,
    workers: Optional[int],
    on_failure: str,
    limit: Optional[int],
    serial: bool,
) -> int:
    """Dewarp every image in ``source_dir`` into ``output_dir``.

    Returns a process exit code (0 on completion). Individual page failures are
    logged and counted, not fatal.
    """
    sources = _iter_source_images(source_dir)
    if limit is not None:
        sources = sources[:limit]
    if not sources:
        logger.error("No images found in %s", source_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (src, output_dir / f"{src.stem}.png", config, on_failure) for src in sources
    ]

    logger.info(
        "Camera dewarp: %d image(s) from %s -> %s (%s)",
        len(tasks),
        source_dir,
        output_dir,
        "serial" if serial else f"{workers or 'auto'} workers",
    )

    counts: Dict[str, int] = defaultdict(int)
    if serial:
        results = (_process_file(t) for t in tasks)
        _tally(results, counts)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            _tally(pool.map(_process_file, tasks), counts)

    logger.info(
        "Done: %d ok, %d skipped, %d copied, %d error(s)",
        counts["ok"],
        counts["skipped"],
        counts["copied"],
        counts["error"],
    )
    return 0


def _tally(results, counts: Dict[str, int]) -> None:
    for name, status, detail in results:
        counts[status] += 1
        if status == "ok":
            logger.info("  %s: dewarped", name)
        elif status == "error":
            logger.warning("  %s: ERROR - %s", name, detail)
        else:  # skipped / copied
            logger.info("  %s: %s (%s)", name, status, detail)


def _build_arg_parser() -> argparse.ArgumentParser:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Standalone camera-model page dewarping (Option B). Runs separately "
            "from the preprocessing pipeline, optionally in parallel."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=module_dir / "output",
        help="Directory of input images (default: ./output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=module_dir / "output_dewarped",
        help="Directory to write rectified images (default: ./output_dewarped).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: auto / CPU count).",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run single-process (easier to debug a failing page).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (quick preview).",
    )
    parser.add_argument(
        "--on-failure",
        choices=("skip", "copy"),
        default="skip",
        help="When a page can't be modeled: skip it, or copy the original through.",
    )
    parser.add_argument(
        "--min-text-lines",
        type=int,
        default=CameraDewarpConfig.min_text_lines,
        help="Minimum detected spans required to attempt the model.",
    )
    parser.add_argument(
        "--max-detect-width",
        type=int,
        default=CameraDewarpConfig.max_detect_width,
        help="Downscale width used for text-span detection.",
    )
    parser.add_argument(
        "--opt-max-iter",
        type=int,
        default=CameraDewarpConfig.opt_max_iter,
        help="Cap on optimiser iterations (higher = better fit, slower).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = _build_arg_parser().parse_args(argv)
    config = CameraDewarpConfig(
        max_detect_width=args.max_detect_width,
        min_text_lines=args.min_text_lines,
        opt_max_iter=args.opt_max_iter,
    )
    return run_batch(
        source_dir=args.source,
        output_dir=args.output,
        config=config,
        workers=args.workers,
        on_failure=args.on_failure,
        limit=args.limit,
        serial=args.serial,
    )


if __name__ == "__main__":
    sys.exit(main())
