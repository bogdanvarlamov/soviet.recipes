"""Dewarp stage (1 to 1) — text-line-based geometric flattening.

Book pages photographed at an angle near the spine suffer two coupled
distortions that, on a text-only page, can only be measured from the text
itself (there are no page edges or ruled lines to detect):

- **Curvature** — text lines bow instead of running straight.
- **Perspective foreshortening** — lines bunch closer together toward the
  spine, and each half-page is distorted in the opposite direction.

This stage corrects both, per page, using a text-line grid remap (Option A;
see ``docs/dewarp-option-b.md`` for the heavier full-camera-model alternative).

Text-line detection follows the established page-dewarp / Leptonica approach —
detect many small text contours, then group them into line "spans" — rather
than forcing one wide blob per line (which fails on curved or sparse text):

1. Downscale to grayscale for fast detection.
2. Adaptive-threshold to isolate text; dilate horizontally to merge characters
   into word blobs, then erode vertically to drop thin blips.
3. Connected-component analysis, keeping only text-sized blobs (rejecting the
   too-tall/too-large blobs that come from photos or graphics).
4. Greedily chain neighbouring blobs (close in x, similar y) into line spans,
   and keep spans with enough blobs spanning enough width.
5. Fit a low-order polynomial ``p_i(x)`` to each span (how that line bows).

The correction is then a grid remap:

6. Choose evenly spaced *target* rows ``Y_i`` for the lines (equal spacing
   between the first and last removes the foreshortening compression).
7. At a grid of columns, map each line's actual position ``p_i(x)`` to its
   target ``Y_i`` and interpolate between lines to build a smooth vertical
   displacement field, then remap with ``cv2.remap`` so every line becomes
   straight and evenly spaced.

Because the correction is estimated from each page's own text, the left and
right half-pages naturally get opposite corrections. This models the dominant
**vertical** geometry; horizontal compression near the spine is not corrected
(rarely matters for OCR — that is Option B's domain).

This stage is intended to run **last** in the pipeline, on the final
tonally-adjusted/sharpened image (also cleaner to detect text on). If fewer
than ``min_text_lines`` usable line spans are found (e.g. a photo/decorative
page) or the modeled correction is negligible, the image is passed through
unchanged and the pass-through is logged so it can be reviewed (safe fallback).
The stage performs no file I/O and does not mutate its input in place; output
dimensions match the input.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..config.settings import DewarpConfig
from ..core.models import WorkingImage
from ..exceptions import ConfigurationError, StageError
from .base import PreprocessingStage

logger = logging.getLogger(__name__)

_STAGE_TYPE = "dewarp"

_SUPPORTED_MODES = frozenset({"L", "RGB"})

# Grid resolution for the displacement mesh (columns x rows). The mesh is
# smoothly upscaled to full resolution, so a coarse grid is enough and fast.
_MESH_COLS = 64
_MESH_ROWS = 64
# Below this many pixels of maximum full-resolution vertical displacement,
# there is no meaningful distortion and the stage passes through.
_MIN_CORRECTION_PX = 1.5
# Minimum blobs a chain must have to be treated as a text-line span.
_MIN_SPAN_BLOBS = 3


@dataclass
class _TextLine:
    """One detected text line: its bow polynomial, x-extent, and mean row."""

    coeffs: np.ndarray
    x_min: float
    x_max: float
    mean_y: float

    def eval_at(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the line's fitted y at columns ``x``, clamped to its extent."""
        clamped = np.clip(x, self.x_min, self.x_max)
        return np.polyval(self.coeffs, clamped)


class DewarpStage(PreprocessingStage):
    """A 1 to 1 stage that flattens spine-distorted text lines (Option A)."""

    def __init__(self, config: DewarpConfig):
        self._config = config

    @property
    def stage_type(self) -> str:
        return _STAGE_TYPE

    def validate_config(self) -> bool:
        if not isinstance(self._config, DewarpConfig):
            raise ConfigurationError(
                "dewarp stage requires a DewarpConfig, "
                f"got {type(self._config).__name__}"
            )
        return True

    def apply(self, working_set: List[WorkingImage]) -> List[WorkingImage]:
        return [self._process_one(item) for item in working_set]

    # -- internal helpers -------------------------------------------------

    def _process_one(self, item: WorkingImage) -> WorkingImage:
        image = item.image
        if not isinstance(image, Image.Image):
            raise StageError(
                f"expected an in-memory Pillow image, got {type(image).__name__}",
                stage_name=_STAGE_TYPE,
            )

        try:
            source = image if image.mode in _SUPPORTED_MODES else image.convert("RGB")
            array = np.asarray(source)
            lines, scale, det_shape = self._detect_lines(array)
            if len(lines) < self._config.min_text_lines:
                # Safe fallback: likely a photo/decorative page (or too few
                # lines to model). Log so it can be reviewed/fact-checked.
                logger.info(
                    "dewarp: %s passed through unchanged - only %d text line(s) "
                    "detected (need >= %d); likely an image/decorative page",
                    item.source_name,
                    len(lines),
                    self._config.min_text_lines,
                )
                result_array = array
            else:
                result_array = self._remap_from_lines(array, lines, scale, det_shape)
            result = Image.fromarray(result_array, mode=source.mode)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                f"failed to dewarp a {image.mode} image: {exc}",
                stage_name=_STAGE_TYPE,
            ) from exc

        return WorkingImage(
            source_name=item.source_name,
            image=result,
            width=result.width,
            height=result.height,
            lineage=list(item.lineage),
        )

    def _detect_lines(
        self, array: np.ndarray
    ) -> Tuple[List[_TextLine], float, Tuple[int, int]]:
        """Detect text-line spans at a downscaled resolution.

        Returns ``(lines, scale, (det_h, det_w))`` where ``lines`` are fitted
        :class:`_TextLine` records in detection-pixel coordinates and ``scale``
        is the detection/full-resolution factor.
        """
        gray = array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

        height, width = gray.shape
        scale = 1.0
        if width > self._config.max_detect_width:
            scale = self._config.max_detect_width / width
            gray = cv2.resize(
                gray, (self._config.max_detect_width, max(1, int(round(height * scale))))
            )

        det_h, det_w = gray.shape
        block = max(3, (det_w // 30) | 1)  # odd, scaled to image size
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15
        )
        # Merge characters into word blobs, then drop thin vertical blips.
        bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)), 1)
        bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), 1)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            bw, connectivity=8
        )

        # Keep only text-sized blobs; reject the tall/huge blobs from photos or
        # graphics (Requirement: image/decorative pages should find no lines).
        max_blob_h = det_h * 0.05
        blobs: List[Tuple[float, float, int, int]] = []  # (cx, cy, w, h)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if 4 <= h <= max_blob_h and w >= 6 and area >= 20:
                blobs.append((float(centroids[i][0]), float(centroids[i][1]), w, h))

        if len(blobs) < _MIN_SPAN_BLOBS:
            return [], scale, (det_h, det_w)

        spans = self._group_blobs_into_spans(blobs, det_w)

        min_span_width = self._config.min_line_width_ratio * det_w
        order = min(self._config.poly_order, 4)
        lines: List[_TextLine] = []
        for span in spans:
            if len(span) < _MIN_SPAN_BLOBS:
                continue
            xs = np.array([b[0] for b in span], dtype=np.float64)
            ys = np.array([b[1] for b in span], dtype=np.float64)
            if float(xs.max() - xs.min()) < min_span_width:
                continue
            sort = np.argsort(xs)
            xs, ys = xs[sort], ys[sort]
            fit_order = min(order, xs.size - 1)
            lines.append(
                _TextLine(
                    coeffs=np.polyfit(xs, ys, fit_order),
                    x_min=float(xs.min()),
                    x_max=float(xs.max()),
                    mean_y=float(ys.mean()),
                )
            )

        lines.sort(key=lambda ln: ln.mean_y)
        return lines, scale, (det_h, det_w)

    @staticmethod
    def _group_blobs_into_spans(
        blobs: List[Tuple[float, float, int, int]], det_w: int
    ) -> List[List[Tuple[float, float, int, int]]]:
        """Greedily chain word blobs into text-line spans.

        Blobs close in x and similar in y (within a fraction of the median blob
        height) are unioned into the same span, so a curved line's blobs still
        chain together locally. Mirrors the page-dewarp span-assembly step.
        """
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
                    break  # order is by x; nothing further will be close enough
                if gap > 0 and abs(cy_j - cy_i) <= y_tol:
                    union(i, j)

        groups: dict = defaultdict(list)
        for i in range(len(blobs)):
            groups[find(i)].append(blobs[i])
        return list(groups.values())

    def _remap_from_lines(
        self,
        array: np.ndarray,
        lines: List[_TextLine],
        scale: float,
        det_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Build and apply the vertical remap that straightens the text grid."""
        det_h, det_w = det_shape
        n = len(lines)

        # Target (output) row for each line: evenly spaced between the first and
        # last line's mean positions. Equal spacing undoes the perspective
        # foreshortening that bunches lines toward the spine.
        means = np.array([ln.mean_y for ln in lines], dtype=np.float64)
        first, last = means[0], means[-1]
        if last > first:
            targets = first + np.arange(n) * ((last - first) / (n - 1))
        else:  # degenerate (all lines at one row): keep as-is
            targets = means

        # Coarse displacement mesh in detection coordinates.
        grid_cols = np.linspace(0, det_w - 1, _MESH_COLS)
        grid_rows = np.linspace(0, det_h - 1, _MESH_ROWS)
        disp = np.empty((_MESH_ROWS, _MESH_COLS), dtype=np.float64)

        for c, xg in enumerate(grid_cols):
            src_anchors = np.array([ln.eval_at(np.array([xg]))[0] for ln in lines])
            source_rows = _interp_extrap(grid_rows, targets, src_anchors)
            disp[:, c] = source_rows - grid_rows

        # Upscale the smooth mesh to full resolution and convert detection-pixel
        # displacement to full-resolution pixels (1 det px = 1/scale full px).
        height, width = array.shape[:2]
        disp_full = cv2.resize(
            disp.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
        disp_full /= scale

        if float(np.abs(disp_full).max()) < _MIN_CORRECTION_PX:
            return array  # negligible correction; pass through unchanged

        cols = np.arange(width, dtype=np.float32)
        rows = np.arange(height, dtype=np.float32)[:, None]
        map_x = np.broadcast_to(cols, (height, width)).astype(np.float32)
        map_y = (rows + disp_full).astype(np.float32)

        border = (255, 255, 255) if array.ndim == 3 else 255
        return cv2.remap(
            array,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )


def _interp_extrap(query: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of ``fp`` over increasing ``xp``.

    Beyond the ends of ``xp``, extrapolates linearly using the slope of the
    nearest segment (rather than clamping), so page margins above the first line
    and below the last line map sensibly instead of being squashed.
    """
    result = np.interp(query, xp, fp)
    if xp.size >= 2:
        below = query < xp[0]
        if below.any():
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            result[below] = fp[0] + slope * (query[below] - xp[0])
        above = query > xp[-1]
        if above.any():
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            result[above] = fp[-1] + slope * (query[above] - xp[-1])
    return result
