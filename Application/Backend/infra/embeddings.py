"""Build query vectors in the same weighted descriptor space as the games."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from Backend.infra.config import settings

# Fitted shared decay coefficients from Tag Weight Study's selected
# length-aware model. Keep aligned with Tag Weight Study/tag_weights.py.
TAG_DECAY_A = 2.249150856857026
TAG_DECAY_B = 0.0946333263878718


def weight_ranked_tags(tags: Sequence[str]) -> list[tuple[str, float]]:
    """Assign fitted position weights to tags ordered by user preference."""
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise TypeError("tags must be a sequence of strings")
    if any(not isinstance(tag, str) for tag in tags):
        raise TypeError("each tag must be a string")

    # Deduplicate before assigning ranks so repeated LLM output cannot distort
    # the decay positions.
    unique_tags: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        label = tag.strip()
        key = label.casefold()
        if label and key not in seen:
            unique_tags.append(label)
            seen.add(key)

    k = len(unique_tags)
    if k == 0:
        return []
    if k == 1:
        return [(unique_tags[0], 1.0)]

    decay_rate = -TAG_DECAY_A * math.expm1(-TAG_DECAY_B * (k - 1))
    return [
        (tag, math.exp(-decay_rate * rank / (k - 1)))
        for rank, tag in enumerate(unique_tags)
    ]


class DescriptorVectorizer:
    """Map ordered query descriptors into the catalogue's fixed column order.

    Tags follow the fitted preference-rank decay. Genres and Categories receive
    value 1. Exact overlaps take the maximum value, matching Data_Prep's rule.
    Unknown names raise an error rather than silently producing a weaker query.
    """

    def __init__(self, vocabulary: Sequence[str]) -> None:
        clean_vocab = [str(item).strip() for item in vocabulary]
        if not clean_vocab or any(not item for item in clean_vocab):
            raise ValueError("descriptor vocabulary must contain non-empty names")
        self.vocabulary = clean_vocab
        self._exact_index = {name: i for i, name in enumerate(clean_vocab)}
        folded: dict[str, list[int]] = {}
        for i, name in enumerate(clean_vocab):
            folded.setdefault(name.casefold(), []).append(i)
        self._folded_index = {
            key: indexes[0] for key, indexes in folded.items() if len(indexes) == 1
        }
        self._ambiguous_folded_names = {
            key for key, indexes in folded.items() if len(indexes) > 1
        }

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> "DescriptorVectorizer":
        """Load descriptor order from the header of games_descriptors.csv."""
        source = Path(path) if path is not None else settings.games_descriptors_path
        if not source.is_file():
            raise FileNotFoundError(f"Descriptor matrix not found: {source}")
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            headers = next(csv.reader(stream), None)
        if not headers or headers[0].casefold() != "appid":
            raise ValueError("Expected games_descriptors.csv to start with an AppID column")
        return cls(headers[1:])

    @property
    def dimension(self) -> int:
        return len(self.vocabulary)

    def encode_query(
        self,
        *,
        tags: Sequence[str],
        genres: Iterable[str] = (),
        categories: Iterable[str] = (),
    ) -> list[float]:
        """Return a dense vector aligned exactly to the game's descriptor columns."""
        vector = [0.0] * self.dimension

        def set_value(label: str, value: float) -> None:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("descriptor labels must be non-empty strings")
            canonical = label.strip()
            column = self._exact_index.get(canonical)
            if column is None:
                normalized = canonical.casefold()
                if normalized in self._ambiguous_folded_names:
                    raise ValueError(
                        f"Descriptor {label!r} is case-ambiguous; use its exact catalogue spelling."
                    )
                column = self._folded_index.get(normalized)
            if column is None:
                raise ValueError(f"Unknown descriptor from LLM: {label!r}")
            vector[column] = max(vector[column], value)

        for tag, weight in weight_ranked_tags(tags):
            set_value(tag, weight)
        for genre in genres:
            set_value(genre, 1.0)
        for category in categories:
            set_value(category, 1.0)

        if not any(vector):
            raise ValueError("Query contains no descriptors in the catalogue vocabulary")
        return vector
