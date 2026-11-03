"""Speaker diarization module for identifying speakers in audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CLUSTERING_THRESHOLD = 0.25
DEFAULT_MIN_CLUSTER_SIZE = 0.5


class DiarizationError(Exception):
    """Base exception for diarization errors."""


class InvalidAudioError(DiarizationError):
    """Raised when input audio is invalid."""


@dataclass(frozen=True)
class SpeakerSegment:
    """A segment of audio attributed to a speaker."""

    speaker_id: str
    start_time: float
    end_time: float
    confidence: float


@dataclass(frozen=True)
class SpeakerTurn:
    """A turn spoken by a single speaker."""

    speaker_id: str
    start_time: float
    end_time: float
    text: str


class PyannoteSpeakerDiarizer:
    """Speaker diarization using PyAnnote model."""

    def __init__(
        self,
        *,
        model_name: str = "pyannote/voice-activity-detection",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        clustering_threshold: float | None = DEFAULT_CLUSTERING_THRESHOLD,
        min_cluster_size: float | None = DEFAULT_MIN_CLUSTER_SIZE,
    ) -> None:
        if num_speakers is not None and (
            min_speakers is not None or max_speakers is not None
        ):
            raise ValueError("Cannot specify both num_speakers and min/max_speakers.")

        self.model_name = model_name
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.clustering_threshold = clustering_threshold
        self.min_cluster_size = min_cluster_size
        self.pipeline = None  # Would load model here

    def detect_speakers(
        self,
        audio_path: str | Path,
    ) -> dict[str, list[tuple[float, float]]]:
        """Detect speakers in audio and return their time segments.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dictionary mapping speaker IDs to lists of (start, end) time tuples.

        Raises:
            DiarizationError: If diarization fails.
        """
        # Implementation would use PyAnnote here
        # Placeholder implementation
        return {}

    def _apply_clustering_parameters(
        self,
        pipeline: Any,
    ) -> None:
        """Apply clustering parameters to the diarization pipeline."""
        if (
            self.clustering_threshold is not None or self.min_cluster_size is not None
        ) and not self._clustering_parameters_applied:
            clustering_parameters: dict[str, float | int] = {}
            if self.clustering_threshold is not None:
                clustering_parameters["threshold"] = self.clustering_threshold
            if self.min_cluster_size is not None:
                clustering_parameters["min_cluster_size"] = self.min_cluster_size
            try:
                pipeline.instantiate({"clustering": clustering_parameters})
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise DiarizationError(
                    "Unable to apply agglomerative clustering "
                    f"parameters: {clustering_parameters}."
                ) from exc
            self._clustering_parameters_applied = True

    _clustering_parameters_applied: bool = False
