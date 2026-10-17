from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SPEECH_DURATION_MS = 250
DEFAULT_MIN_SILENCE_DURATION_MS = 100
DEFAULT_SPEECH_PAD_MS = 30


class SpeechDetectorError(Exception):
    """Base exception for speech-detection errors."""


class SileroVADNotFoundError(SpeechDetectorError):
    """Raised when Silero VAD is unavailable."""


class InvalidAudioError(SpeechDetectorError):
    """Raised when the input audio is missing or invalid."""


@dataclass(frozen=True)
class SpeechRegion:
    """A detected speech interval, measured in seconds."""

    start: float
    end: float


class SpeechDetector:
    """Detect speech regions using Silero VAD."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
        speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        model: Any | None = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

        if min_speech_duration_ms < 0:
            raise ValueError(
                "min_speech_duration_ms must not be negative"
            )

        if min_silence_duration_ms < 0:
            raise ValueError(
                "min_silence_duration_ms must not be negative"
            )

        if speech_pad_ms < 0:
            raise ValueError("speech_pad_ms must not be negative")

        if sample_rate not in {8_000, 16_000}:
            raise ValueError("sample_rate must be 8000 or 16000")

        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        self._model = model

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

    def _read_audio(self, path: Path) -> tuple[np.ndarray, int]:
        """Read audio, convert it to mono, and validate its samples."""
        try:
            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidAudioError(
                f"Unable to read audio file: {path}"
            ) from exc

        if audio.size == 0 or audio.shape[0] == 0:
            raise InvalidAudioError(
                f"Audio file contains no samples: {path}"
            )

        if sample_rate <= 0:
            raise InvalidAudioError(
                f"Invalid audio sample rate: {sample_rate}"
            )

        if not np.isfinite(audio).all():
            raise InvalidAudioError(
                f"Audio contains invalid samples: {path}"
            )

        mono_audio = audio.mean(axis=1, dtype=np.float32)

        return mono_audio, int(sample_rate)

    def _prepare_audio(self, path: Path) -> np.ndarray:
        """Convert audio to mono float32 at the configured sample rate."""
        audio, original_sample_rate = self._read_audio(path)

        if original_sample_rate != self.sample_rate:
            divisor = math.gcd(
                original_sample_rate,
                self.sample_rate,
            )

            audio = resample_poly(
                audio,
                self.sample_rate // divisor,
                original_sample_rate // divisor,
            ).astype(np.float32, copy=False)

        if audio.size == 0:
            raise InvalidAudioError(
                f"Audio contains no usable samples: {path}"
            )

        return np.ascontiguousarray(audio, dtype=np.float32)

    def _load_model(self) -> Any:
        """Load and cache the Silero VAD ONNX model."""
        if self._model is not None:
            return self._model

        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise SileroVADNotFoundError(
                "Silero VAD is not installed. "
                "Install it using: poetry add silero-vad onnxruntime"
            ) from exc

        try:
            self._model = load_silero_vad(onnx=True)
        except Exception as exc:
            raise SpeechDetectorError(
                "Unable to load the Silero VAD ONNX model."
            ) from exc

        return self._model

    def _run_inference(
        self,
        audio: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run Silero VAD and return speech timestamps in seconds."""
        try:
            import torch
            from silero_vad import get_speech_timestamps
        except ImportError as exc:
            raise SileroVADNotFoundError(
                "Silero VAD and PyTorch must be installed."
            ) from exc

        model = self._load_model()
        waveform = torch.from_numpy(audio)

        try:
            with torch.inference_mode():
                timestamps = get_speech_timestamps(
                    waveform,
                    model,
                    threshold=self.threshold,
                    sampling_rate=self.sample_rate,
                    min_speech_duration_ms=(
                        self.min_speech_duration_ms
                    ),
                    min_silence_duration_ms=(
                        self.min_silence_duration_ms
                    ),
                    speech_pad_ms=self.speech_pad_ms,
                    return_seconds=True,
                )
        except Exception as exc:
            raise SpeechDetectorError(
                "Silero VAD inference failed."
            ) from exc

        return timestamps

    @staticmethod
    def _parse_regions(
        timestamps: list[dict[str, Any]],
        duration_seconds: float,
    ) -> list[SpeechRegion]:
        """Validate and convert model timestamps to speech regions."""
        if not isinstance(timestamps, list):
            raise SpeechDetectorError(
                "Silero VAD returned an invalid timestamp collection."
            )

        regions: list[SpeechRegion] = []

        for timestamp in timestamps:
            if not isinstance(timestamp, dict):
                raise SpeechDetectorError(
                    "Silero VAD returned an invalid speech interval."
                )

            try:
                start = float(timestamp["start"])
                end = float(timestamp["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SpeechDetectorError(
                    "Silero VAD returned an invalid speech interval."
                ) from exc

            if not math.isfinite(start) or not math.isfinite(end):
                raise SpeechDetectorError(
                    "Silero VAD returned non-finite timestamps."
                )

            if start < 0 or end <= start:
                raise SpeechDetectorError(
                    "Silero VAD returned invalid interval boundaries."
                )

            start = min(start, duration_seconds)
            end = min(end, duration_seconds)

            if end <= start:
                continue

            regions.append(
                SpeechRegion(
                    start=start,
                    end=end,
                )
            )

        return sorted(regions, key=lambda region: region.start)

    def detect(
        self,
        audio_path: str | Path,
    ) -> list[SpeechRegion]:
        """Detect speech intervals in an audio file.

        Args:
            audio_path: Path to the input audio.

        Returns:
            Speech regions with start and end timestamps in seconds.

        Raises:
            InvalidAudioError: If the input audio is missing or invalid.
            SileroVADNotFoundError: If Silero VAD is unavailable.
            SpeechDetectorError: If inference or timestamp processing fails.
        """
        path = Path(audio_path)
        self._validate_input(path)

        audio = self._prepare_audio(path)
        duration_seconds = len(audio) / self.sample_rate

        timestamps = self._run_inference(audio)

        return self._parse_regions(
            timestamps,
            duration_seconds,
        )