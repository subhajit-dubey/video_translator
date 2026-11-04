"""Speaker diarization module for identifying speakers in audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CLUSTERING_THRESHOLD = 0.25
DEFAULT_MIN_CLUSTER_SIZE = 0.5


class DiarizationError(Exception):
    """Base exception for diarization errors."""


class InvalidAudioError(DiarizationError):
    """Raised when input audio is invalid."""


class PyannoteNotFoundError(DiarizationError):
    """Raised when PyAnnote is not installed."""


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


@dataclass(frozen=True)
class SpeakerProfile:
    """Profile for a speaker's voice characteristics."""

    speaker_id: str
    pitch_hz: float
    duration_seconds: float
    reference_wav: Path
    turns: tuple[SpeakerTurn, ...]


class PyannoteSpeakerDiarizer:
    """Speaker diarization using PyAnnote model."""

    def __init__(
        self,
        *,
        model_name: str = "pyannote/speaker-diarization-3.1",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        clustering_threshold: float | None = None,
        min_cluster_size: float | None = None,
        device: str | None = None,
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline: Any = None
        self._clustering_applied = False

    def _load_pipeline(self) -> Any:
        """Load and cache the PyAnnote diarization pipeline."""
        if self.pipeline is not None:
            return self.pipeline

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            msg = "PyAnnote is not installed. Install with: pip install pyannote.audio"
            raise PyannoteNotFoundError(msg) from exc

        try:
            self.pipeline = Pipeline.from_pretrained(self.model_name)
            self.pipeline.to(torch.device(self.device))
        except Exception as exc:
            msg = f"Unable to load PyAnnote model: {self.model_name}"
            raise DiarizationError(msg) from exc

        return self.pipeline

    def _apply_clustering_if_needed(self, pipeline: Any) -> None:
        """Apply clustering parameters only if explicitly provided."""
        if self._clustering_applied:
            return

        # Only apply if user explicitly provided clustering parameters
        if self.clustering_threshold is not None or self.min_cluster_size is not None:
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
            self._clustering_applied = True

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
            InvalidAudioError: If the input audio is invalid.
        """
        path = Path(audio_path)
        if not path.exists():
            msg = f"Input audio file does not exist: {path}"
            raise InvalidAudioError(msg)

        if not path.is_file():
            msg = f"Input path is not a file: {path}"
            raise InvalidAudioError(msg)

        if path.stat().st_size == 0:
            msg = f"Input audio file is empty: {path}"
            raise InvalidAudioError(msg)

        pipeline = self._load_pipeline()

        # Apply clustering parameters only if explicitly provided
        self._apply_clustering_if_needed(pipeline)

        # Run diarization
        try:
            diarization = pipeline(
                str(path),
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )
        except Exception as exc:
            msg = f"Diarization failed for: {path}"
            raise DiarizationError(msg) from exc

        # Extract speaker segments from EXCLUSIVE diarization (non-overlapping)
        result: dict[str, list[tuple[float, float]]] = {}
        annotation = diarization.exclusive_speaker_diarization

        for segment, _, label in annotation.itertracks(yield_label=True):
            if label not in result:
                result[label] = []
            result[label].append((float(segment.start), float(segment.end)))

        # Sort segments by start time for each speaker
        for speaker_segments in result.values():
            speaker_segments.sort(key=lambda x: x[0])

        return result

    def extract_profiles(
        self,
        audio_path: str | Path,
        turns: list[SpeakerTurn],
        output_dir: Path,
        max_duration_seconds: float = 30.0,
    ) -> dict[str, SpeakerProfile]:
        """Extract reference audio profiles for each speaker.

        Args:
            audio_path: Path to the source audio file.
            turns: List of speaker turns with timing information.
            output_dir: Directory to save reference WAV files.
            max_duration_seconds: Maximum duration of reference audio per speaker.

        Returns:
            Dictionary mapping speaker IDs to SpeakerProfile objects.

        Raises:
            DiarizationError: If profile extraction fails.
        """
        import soundfile as sf  # type: ignore[import-untyped]

        path = Path(audio_path)
        if not path.exists():
            msg = f"Input audio file does not exist: {path}"
            raise InvalidAudioError(msg)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Group turns by speaker
        speaker_turns: dict[str, list[SpeakerTurn]] = {}
        for turn in turns:
            if turn.speaker_id not in speaker_turns:
                speaker_turns[turn.speaker_id] = []
            speaker_turns[turn.speaker_id].append(turn)

        # Load the full audio
        audio_data, sample_rate = sf.read(str(path))
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono

        profiles: dict[str, SpeakerProfile] = {}

        for speaker_id, turns_list in speaker_turns.items():
            # Sort turns by start time
            turns_list.sort(key=lambda t: t.start_time)

            # Calculate total duration needed
            total_duration = sum(t.end_time - t.start_time for t in turns_list)
            target_duration = min(total_duration, max_duration_seconds)

            # Collect audio segments for this speaker
            segments: list[tuple[int, int]] = []
            accumulated = 0.0

            for turn in turns_list:
                if accumulated >= target_duration:
                    break

                start_sample = int(turn.start_time * sample_rate)
                end_sample = int(turn.end_time * sample_rate)
                segment_duration = (end_sample - start_sample) / sample_rate

                if accumulated + segment_duration > target_duration:
                    # Trim the last segment
                    remaining_samples = int(
                        (target_duration - accumulated) * sample_rate
                    )
                    end_sample = start_sample + remaining_samples
                    segment_duration = remaining_samples / sample_rate

                segments.append((start_sample, end_sample))
                accumulated += segment_duration

            if not segments:
                continue

            # Extract and concatenate segments
            reference_audio_parts = []
            for start_sample, end_sample in segments:
                reference_audio_parts.append(audio_data[start_sample:end_sample])

            if reference_audio_parts:
                reference_audio = np.concatenate(reference_audio_parts)

                # Save reference WAV
                reference_wav = output_dir / f"{speaker_id}_reference.wav"
                sf.write(str(reference_wav), reference_audio, sample_rate)

                # Estimate pitch (simple method using zero-crossing rate)
                pitch_hz = self._estimate_pitch(reference_audio, sample_rate)

                profiles[speaker_id] = SpeakerProfile(
                    speaker_id=speaker_id,
                    pitch_hz=pitch_hz,
                    duration_seconds=accumulated,
                    reference_wav=reference_wav,
                    turns=tuple(turns_list),
                )

        return profiles

    def _estimate_pitch(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate fundamental frequency using autocorrelation."""
        # Simple autocorrelation-based pitch estimation
        # Use a window of at least 20ms at the sample rate
        window_size = min(len(audio), int(0.04 * sample_rate))  # 40ms window
        if window_size < 10:
            return 150.0  # Default fallback

        window = audio[:window_size]
        window = window - np.mean(window)

        # Autocorrelation
        autocorr = np.correlate(window, window, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Find first peak after the zero lag
        min_period = int(sample_rate / 400)  # Max 400 Hz
        max_period = int(sample_rate / 60)  # Min 60 Hz

        if max_period >= len(autocorr):
            max_period = len(autocorr) - 1

        if min_period >= max_period:
            return 150.0

        peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period

        if peak_idx <= 0:
            return 150.0

        pitch = sample_rate / peak_idx
        return float(np.clip(pitch, 60, 400))
