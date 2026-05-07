from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image

# ============================================================
# LOOKUP TABLES (ImageMagick heuristic constants)
# ============================================================

HASH_1 = [...]
SUMS_1 = [...]

HASH_2 = [...]
SUMS_2 = [...]

# (keep full arrays as-is — omitted here for readability)


# ============================================================
# FEATURE EXTRACTION
# ============================================================


def extract_quantization_features(qtables: Dict[int, List[int]]) -> Tuple[int, int]:
    """
    Convert JPEG quantization tables into two heuristic features.

    Returns:
        (qsum, qvalue)
    """

    # total energy in quantization tables
    qsum = sum(sum(table) for table in qtables.values())

    # primary structural signature
    first = qtables.get(0, [])
    second = qtables.get(1, [])

    qvalue = 0

    if len(first) >= 54:
        qvalue += first[2] + first[53]

    # chroma component improves classification stability
    if len(second) > 0:
        qvalue += second[0] + second[-1]

    return qsum, qvalue


# ============================================================
# QUALITY CLASSIFIER (lookup logic)
# ============================================================


def classify_jpeg_quality(qsum: int, qvalue: int) -> int:
    """
    Map extracted features to JPEG quality using heuristic thresholds.
    """

    # choose model depending on presence of chroma info
    use_chroma_model = qvalue > 0

    hash_table = HASH_2 if use_chroma_model else HASH_1
    sum_table = SUMS_2 if use_chroma_model else SUMS_1

    for quality in range(100):
        if (qvalue < hash_table[quality]) and (qsum < sum_table[quality]):  # type: ignore
            continue

        if (qvalue <= hash_table[quality]) and (qsum <= sum_table[quality]) or quality >= 50:  # type: ignore
            return quality + 1

        break

    return -1


# ============================================================
# PUBLIC API
# ============================================================


def get_jpg_quality(image: Image.Image) -> int:
    """
    Estimate JPEG quality using ImageMagick heuristic model.

    Args:
        image: PIL Image (must contain quantization info)

    Returns:
        Estimated quality in range 1–100, or -1 if unknown.
    """

    qtables = getattr(image, "quantization", None)

    if not qtables:
        return -1

    qsum, qvalue = extract_quantization_features(qtables)

    return classify_jpeg_quality(qsum, qvalue)
