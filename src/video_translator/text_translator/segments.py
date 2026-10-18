from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Any


DEFAULT_MAX_GAP_SECONDS = 0.75
DEFAULT_MAX_MERGED_DURATION_SECONDS = 12.0
DEFAULT_MAX_MERGED_CHARACTERS = 500
DEFAULT_SHORT_SEGMENT_WORDS = 3


class TextSegmentError(Exception):
    """Raised when text segment validation fails."""


@dataclass(frozen=True)
class TextSegment:
    """A timestamped text segment."""

    start: float
    end: float
    text: str
    speaker_id: str | None = None


@dataclass(frozen=True)
class TranslatedTextSegment:
    """A translated text segment preserving the original timing."""

    start: float
    end: float
    source_text: str
    translated_text: str
    speaker_id: str | None = None

    @property
    def text(self) -> str:
        """Return translated text."""
        return self.translated_text


@dataclass(frozen=True)
class TranslationResult:
    """Collection of translated segments and language metadata."""

    source_language: str
    target_language: str
    segments: tuple[TranslatedTextSegment, ...]

    @property
    def text(self) -> str:
        """Return the complete translated transcript."""
        return " ".join(
            segment.translated_text
            for segment in self.segments
            if segment.translated_text
        )


def _coerce_segment(segment: Any) -> TextSegment:
    """Convert a compatible segment object into TextSegment."""
    try:
        start = float(segment.start)
        end = float(segment.end)
        text = segment.text
        speaker_id = getattr(segment, "speaker_id", None)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TextSegmentError(
            "Invalid text segment: expected start, end, and text."
        ) from exc

    if not math.isfinite(start) or not math.isfinite(end):
        raise TextSegmentError(
            "Segment timestamps must be finite."
        )

    if start < 0 or end <= start:
        raise TextSegmentError(
            "Segment timestamps must satisfy 0 <= start < end."
        )

    if not isinstance(text, str):
        raise TextSegmentError(
            "Segment text must be a string."
        )

    if speaker_id is not None and not isinstance(speaker_id, str):
        raise TextSegmentError(
            "speaker_id must be a string or None."
        )

    return TextSegment(
        start=start,
        end=end,
        text=text.strip(),
        speaker_id=speaker_id,
    )


def _ends_sentence(text: str) -> bool:
    """Check whether the text ends with terminal punctuation."""
    return bool(
        re.search(
            r"""[.!?…。！？]["'”’)\]]*$""",
            text.strip(),
        )
    )


def _word_count(text: str) -> int:
    return len(text.split())


def normalize_text_segments(
    segments: Iterable[Any],
) -> list[TextSegment]:
    """Validate, normalize, and sort segments without merging them."""
    normalized = [
        _coerce_segment(segment)
        for segment in segments
    ]

    normalized = [
        segment
        for segment in normalized
        if segment.text
    ]

    normalized.sort(
        key=lambda segment: (segment.start, segment.end)
    )

    return normalized


def merge_text_segments(
    segments: Iterable[Any],
    *,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    max_duration_seconds: float = DEFAULT_MAX_MERGED_DURATION_SECONDS,
    max_characters: int = DEFAULT_MAX_MERGED_CHARACTERS,
    short_segment_words: int = DEFAULT_SHORT_SEGMENT_WORDS,
) -> list[TextSegment]:
    """Merge choppy transcript segments into more natural text groups.

    Segments may be merged when:
    - The temporal gap is within the configured threshold.
    - The speaker does not change.
    - The merged duration and text length stay within their limits.
    - The previous segment is incomplete or sufficiently short.

    Timestamps preserve the first segment's start and the last
    segment's end.
    """
    if (
        not math.isfinite(max_gap_seconds)
        or max_gap_seconds < 0
    ):
        raise ValueError(
            "max_gap_seconds must be finite and non-negative."
        )

    if (
        not math.isfinite(max_duration_seconds)
        or max_duration_seconds <= 0
    ):
        raise ValueError(
            "max_duration_seconds must be finite and positive."
        )

    if max_characters <= 0:
        raise ValueError("max_characters must be positive.")

    if short_segment_words < 0:
        raise ValueError("short_segment_words must be non-negative.")

    normalized = normalize_text_segments(segments)

    if not normalized:
        return []

    merged: list[TextSegment] = []
    current = normalized[0]

    for following in normalized[1:]:
        gap = max(0.0, following.start - current.end)

        combined_text = (
            f"{current.text.rstrip()} "
            f"{following.text.lstrip()}"
        ).strip()

        speaker_changed = (
            current.speaker_id is not None
            and following.speaker_id is not None
            and current.speaker_id != following.speaker_id
        )

        sentence_complete = (
            _ends_sentence(current.text)
            and _word_count(current.text) > short_segment_words
        )

        can_merge = (
            gap <= max_gap_seconds
            and not speaker_changed
            and not sentence_complete
            and following.end - current.start
            <= max_duration_seconds
            and len(combined_text) <= max_characters
        )

        if can_merge:
            current = TextSegment(
                start=current.start,
                end=max(current.end, following.end),
                text=combined_text,
                speaker_id=(
                    current.speaker_id or following.speaker_id
                ),
            )
        else:
            merged.append(current)
            current = following

    merged.append(current)

    return merged
