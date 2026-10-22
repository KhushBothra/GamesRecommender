"""Assign positional weights to ordered tags using the fitted exponential model.

Copy this file beside a notebook, then use:

    from tag_weights import weight_tags

    weighted_tags = weight_tags(["Action", "Adventure", "Indie"])
    # Returns [(tag, weight), ...] in the original order.

Requires only Python's standard library. No dataset or model file is needed.
Coefficients are the all-data refit saved by the Steam tag decay study in
results/generalization/selected_length_aware_model.json.
"""

from __future__ import annotations

from collections.abc import Sequence
import math


DECAY_A = 2.249150856857026
DECAY_B = 0.0946333263878718

__all__ = ["weight_tags"]


def weight_tags(tags: Sequence[str]) -> list[tuple[str, float]]:
    """Return an ordered list of (tag, weight) pairs.

    Tags must already be ordered from most to least important. Names do not
    affect weights, and this function does not sort, deduplicate, or modify the
    input. Duplicate tags are preserved at their respective positions.

    For k tags and 1-based rank r:
        rate(k) = A * (1 - exp(-B * (k - 1)))
        weight(r, k) = exp(-rate(k) * (r - 1) / (k - 1))

    An empty input returns []; a singleton returns [(tag, 1.0)]. All nonempty
    lists start at weight 1, and subsequent weights are non-increasing within
    [0, 1]. Weights are not probabilities and are not normalized to sum to 1.

    Calibration used lengths 2 through 21. Longer lists use the same formula
    without truncation; these weights are extrapolations.

    Args:
        tags: A list or other sequence of tag strings (not a single string).

    Raises:
        TypeError: If tags is not a sequence of strings.
    """
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise TypeError("tags must be a sequence of strings, such as ['Action', 'RPG'].")
    if any(not isinstance(tag, str) for tag in tags):
        raise TypeError("Every tag must be a string.")

    k = len(tags)
    if k == 0:
        return []
    if k == 1:
        return [(tags[0], 1.0)]

    decay_rate = -DECAY_A * math.expm1(-DECAY_B * (k - 1))
    return [(tag, math.exp(-decay_rate * index / (k - 1)))
            for index, tag in enumerate(tags)]
