from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_SIZE = "small"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "default"
DEFAULT_BEAM_SIZE = 5


class SpeechRecognizerError(Exception):
    """Base exception for speech-recognition errors."""


class FasterWhisperNotFoundError(SpeechRecognizerError):
    """Raised when Faster-Whisper is not installed."""


class InvalidAudioError(SpeechRecognizerError):
    """Raised when the input audio is missing or invalid."""


class ModelLoadError(SpeechRecognizerError):
    """Raised when the speech-recognition model cannot be loaded."""


class TranscriptionError(SpeechRecognizerError):
    """Raised when transcription fails."""


@dataclass(frozen=True)
class TranscriptSegment:
    """A transcribed speech segment with timestamps in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Result returned by the speech recognizer."""

    audio_path: Path
    language: str | None
    language_probability: float | None
    duration_seconds: float
    segments: tuple[TranscriptSegment, ...]

    @property
    def text(self) -> str:
        """Return the complete transcript."""
        return " ".join(segment.text for segment in self.segments)


class SpeechRecognizer:
    """Transcribe audio using Faster-Whisper."""

    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        beam_size: int = DEFAULT_BEAM_SIZE,
        vad_filter: bool = False,
        vad_parameters: dict[str, Any] | None = None,
        word_timestamps: bool = False,
        condition_on_previous_text: bool = True,
        model_instance: Any | None = None,
    ) -> None:
        if not model_size or not model_size.strip():
            raise ValueError("model_size must not be empty")

        if not device or not device.strip():
            raise ValueError("device must not be empty")

        if not compute_type or not compute_type.strip():
            raise ValueError("compute_type must not be empty")

        if beam_size < 1:
            raise ValueError("beam_size must be greater than zero")

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = (
            dict(vad_parameters)
            if vad_parameters is not None
            else None
        )
        self.word_timestamps = word_timestamps
        self.condition_on_previous_text = condition_on_previous_text

        self._model = model_instance

    @staticmethod
    def _validate_input(path: Path) -> None:
        if not path.exists():
            raise InvalidAudioError(
                f"Input audio file does not exist: {path}"
            )

        if not path.is_file():
            raise InvalidAudioError(
                f"Input path is not a file: {path}"
            )

        if path.stat().st_size == 0:
            raise InvalidAudioError(
                f"Input audio file is empty: {path}"
            )

    def _load_model(self) -> Any:
        """Load and cache the Faster-Whisper model."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise FasterWhisperNotFoundError(
                "Faster-Whisper is not installed. "
                "Install it using: poetry add faster-whisper"
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Unable to load Faster-Whisper model: "
                f"{self.model_size}"
            ) from exc

        return self._model

    @staticmethod
    def _parse_segment(
        segment: Any,
    ) -> TranscriptSegment:
        """Convert a Faster-Whisper segment into a public data object."""
        try:
            start = float(segment.start)
            end = float(segment.end)
            raw_text = segment.text

            if not isinstance(raw_text, str):
                raise TranscriptionError(
                    "Faster-Whisper returned an invalid segment text."
                )

            text = raw_text.strip()

        except TranscriptionError:
            raise

        except (AttributeError, TypeError, ValueError) as exc:
            raise TranscriptionError(
                "Faster-Whisper returned an invalid transcript segment."
            ) from exc

        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise TranscriptionError(
                "Faster-Whisper returned invalid segment timestamps."
            )

        return TranscriptSegment(
            start=start,
            end=end,
            text=text,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file into timestamped text segments.

        Args:
            audio_path: Path to the input audio file.
            language: Optional source language code, such as 'en' or 'hi'.
                If omitted, Faster-Whisper detects the language.

        Returns:
            A TranscriptionResult containing timestamped segments
            and detected language information.

        Raises:
            InvalidAudioError: If the input file is missing or empty.
            FasterWhisperNotFoundError: If Faster-Whisper is unavailable.
            ModelLoadError: If the model cannot be loaded.
            TranscriptionError: If transcription fails.
        """
        path = Path(audio_path)
        self._validate_input(path)

        if language is not None and not language.strip():
            raise ValueError("language must not be empty")

        model = self._load_model()

        try:
            segments_iterator, info = model.transcribe(
                str(path),
                language=language,
                task="transcribe",
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters,
                word_timestamps=self.word_timestamps,
                condition_on_previous_text=self.condition_on_previous_text,
            )

            segments = tuple(
                self._parse_segment(segment)
                for segment in segments_iterator
            )

            segments = tuple(
                segment
                for segment in segments
                if segment.text
            )

        except SpeechRecognizerError:
            raise

        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed for: {path}"
            ) from exc

        language_code = getattr(info, "language", None)
        language_probability = getattr(
            info,
            "language_probability",
            None,
        )

        if language_probability is not None:
            try:
                language_probability = float(language_probability)
            except (TypeError, ValueError) as exc:
                raise TranscriptionError(
                    "Faster-Whisper returned an invalid language probability."
                ) from exc

            if (
                not math.isfinite(language_probability)
                or not 0.0 <= language_probability <= 1.0
            ):
                raise TranscriptionError(
                    "Faster-Whisper returned an invalid language probability."
                )

        duration_value = getattr(info, "duration", None)

        try:
            duration = float(duration_value)
        except (TypeError, ValueError):
            duration = 0.0

        if not math.isfinite(duration) or duration < 0:
            duration = 0.0

        if segments:
            duration = max(
                duration,
                max(segment.end for segment in segments),
            )

        return TranscriptionResult(
            audio_path=path,
            language=language_code,
            language_probability=language_probability,
            duration_seconds=duration,
            segments=segments,
        )