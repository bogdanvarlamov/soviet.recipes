"""Text-wise consensus comparison across processing engine runs.

Scans the processing_results directory for engine runs (any subfolder that
contains a ``text/`` directory of ``.txt`` outputs), then builds a consensus
table: one row per source image (page half), one column per engine method,
plus pairwise text similarity and a flag for which method conflicts.

Quick + simple. Standard library only. Run with:

    uv run python phase_2/processing_results/compare_consensus.py

Status per row:
    MATCH    - every method produced identical normalized text.
    MISMATCH - all methods have text but it differs (see similarity columns).
    MISSING  - some methods found text while others were blank.
    BLANK    - no method found real text (empty / photo-only page).

Optional args:
    --results-dir  Directory to scan (default: this file's folder).
    --output       Output CSV path (default: <results-dir>/consensus.csv).
"""

from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

# Runs to skip even if they look like engine outputs.
EXCLUDE_RUNS = {"docling_easyocr_dualpage"}

# Key pattern shared across runs, e.g. "pages-104-a" (ignores the NNN- prefix).
KEY_RE = re.compile(r"(pages-\d+-[ab])", re.IGNORECASE)

# Markdown / docling noise stripped before comparing.
IMAGE_TAG_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop image tags / markdown punctuation, collapse whitespace."""
    text = IMAGE_TAG_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip().lower()


# Real page content is Russian (Cyrillic). Output with essentially no Cyrillic
# is either an empty easyOCR file or a VLM English "blank/empty page" caption.
CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
MIN_CYRILLIC = 3


def is_blank(norm_text: str) -> bool:
    """True when the output carries no real (Cyrillic) transcribed text."""
    return len(CYRILLIC_RE.findall(norm_text)) < MIN_CYRILLIC


def word_set(text: str) -> set[str]:
    """Unique words in normalized text (order-independent)."""
    return set(text.split())


def jaccard(a: str, b: str) -> float:
    """Word-set overlap: |A n B| / |A u B|. Robust to block reordering."""
    words_a, words_b = word_set(a), word_set(b)
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def discover_methods(results_dir: Path) -> dict[str, Path]:
    """Return {method_name: text_dir} for each run that has a text/ folder."""
    methods: dict[str, Path] = {}
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in EXCLUDE_RUNS:
            continue
        text_dir = child / "text"
        if text_dir.is_dir() and any(text_dir.glob("*.txt")):
            methods[child.name] = text_dir
    return methods


def index_files(text_dir: Path) -> dict[str, Path]:
    """Map shared key (pages-N-a/b) -> file path for one method."""
    index: dict[str, Path] = {}
    for path in text_dir.glob("*.txt"):
        match = KEY_RE.search(path.stem)
        if match:
            index[match.group(1).lower()] = path
    return index


def load_text(path: Path | None) -> str:
    if path is None:
        return ""
    return normalize(path.read_text(encoding="utf-8", errors="replace"))


def sort_key(image_key: str) -> tuple[int, str]:
    match = re.search(r"pages-(\d+)-([ab])", image_key)
    if match:
        return int(match.group(1)), match.group(2)
    return (0, image_key)


def analyze(image_key: str, texts: dict[str, str]) -> dict:
    """Build one result row for a single image across all methods."""
    methods = list(texts)
    blank = {m: is_blank(texts[m]) for m in methods}
    content = [m for m in methods if not blank[m]]

    # Pairwise similarity over method pairs where both sides have content.
    # char_sim: character-sequence ratio (sensitive to reordering / OCR noise).
    # word_sim: word-set Jaccard (order-independent content overlap).
    char_sim: dict[tuple[str, str], float] = {}
    word_sim: dict[tuple[str, str], float] = {}
    for a, b in combinations(methods, 2):
        if not blank[a] and not blank[b]:
            char_sim[(a, b)] = SequenceMatcher(None, texts[a], texts[b]).ratio()
            word_sim[(a, b)] = jaccard(texts[a], texts[b])
        else:
            char_sim[(a, b)] = 0.0
            word_sim[(a, b)] = 0.0

    blanks = [m for m in methods if blank[m]]

    # Status:
    #   BLANK    - no method found real text (empty page / photo-only page).
    #   MISSING  - some methods found text, others were blank (discrepancy).
    #   MATCH    - every method produced identical normalized text.
    #   MISMATCH - all have text but it differs; conflict flags the outlier.
    conflict = ""
    if len(blanks) == len(methods):
        status = "BLANK"
    elif blanks:
        status = "MISSING"
        conflict = ";".join(blanks)
    elif all(char_sim[pair] == 1.0 for pair in char_sim):
        status = "MATCH"
    else:
        status = "MISMATCH"
        if len(content) >= 3:
            # Outlier = method with the lowest average Jaccard to the rest.
            avg = {
                m: sum(word_sim[tuple(sorted((m, o)))] for o in content if o != m)
                / (len(content) - 1)
                for m in content
            }
            conflict = min(avg, key=avg.get)
        else:
            # Two methods: they simply disagree with each other.
            conflict = ";".join(content)

    char_vals = list(char_sim.values())
    word_vals = list(word_sim.values())
    row = {"image": image_key, "status": status, "conflict_method": conflict}
    for m in methods:
        row[f"{m}__chars"] = len(texts[m])
    for (a, b) in char_sim:
        row[f"charsim__{a}__vs__{b}"] = round(char_sim[(a, b)], 4)
        row[f"jaccard__{a}__vs__{b}"] = round(word_sim[(a, b)], 4)
    row["min_charsim"] = round(min(char_vals), 4) if char_vals else 0.0
    row["mean_charsim"] = round(sum(char_vals) / len(char_vals), 4) if char_vals else 0.0
    row["min_jaccard"] = round(min(word_vals), 4) if word_vals else 0.0
    row["mean_jaccard"] = round(sum(word_vals) / len(word_vals), 4) if word_vals else 0.0
    return row


def main() -> None:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_dir)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    output_path: Path = args.output or (results_dir / "consensus.csv")

    methods = discover_methods(results_dir)
    if not methods:
        raise SystemExit(f"No engine runs with a text/ folder found in {results_dir}")

    print(f"Comparing {len(methods)} method(s): {', '.join(methods)}")

    indexes = {name: index_files(text_dir) for name, text_dir in methods.items()}
    all_keys = sorted(
        {key for idx in indexes.values() for key in idx}, key=sort_key
    )

    rows = []
    for image_key in all_keys:
        texts = {
            name: load_text(indexes[name].get(image_key)) for name in methods
        }
        rows.append(analyze(image_key, texts))

    # Stable, readable column order.
    method_names = list(methods)
    fieldnames = ["image", "status", "conflict_method"]
    fieldnames += [f"{m}__chars" for m in method_names]
    for a, b in combinations(method_names, 2):
        fieldnames += [f"charsim__{a}__vs__{b}", f"jaccard__{a}__vs__{b}"]
    fieldnames += ["min_charsim", "mean_charsim", "min_jaccard", "mean_jaccard"]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Console summary.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote {len(rows)} rows -> {output_path}")
    print("Status breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
