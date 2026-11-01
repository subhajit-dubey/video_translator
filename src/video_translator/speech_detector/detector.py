from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-not-found,import-untyped]
import soundfile as sf  # type: ignore[import-not-found,import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-not-found,import-untyped]

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


@dataclass(frozen=True)
class SpeechDetectorConfig:
    """Configuration for SpeechDetector."""

    threshold: float = DEFAULT_THRESHOLD
    min_speech_duration_ms: int = DEFAULT_MIN_SPEECH_DURATION_MS
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS
    speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS
    sample_rate: int = DEFAULT_SAMPLE_RATE
    model: Any | None = None


class SpeechDetector:
    """Detect speech regions using Silero VAD."""

    def __init__(self, *, config: SpeechDetectorConfig | None = None) -> None:
        cfg = config or SpeechDetectorConfig()

        if not 0.0 < cfg.threshold < 1.0:
            msg = "threshold must be between 0 and 1"
            raise ValueError(msg)

        if cfg.min_speech_duration_ms < 0:
            msg = "min_speech_duration_ms must not be negative"
            raise ValueError(msg)

        if cfg.min_silence_duration_ms < 0:
            msg = "min_silence_duration_ms must not be negative"
            raise ValueError(msg)

        if cfg.speech_pad_ms < 0:
            msg = "speech_pad_ms must not be negative"
            raise ValueError(msg)

        if cfg.sample_rate not in {8_000, 16_000}:
            msg = "sample_rate must be 8000 or 16000"
            raise ValueError(msg)

        self.threshold = cfg.threshold
        self.min_speech_duration_ms = cfg.min_speech_duration_ms
        self.min_silence_duration_ms = cfg.min_silence_duration_ms
        self.speech_pad_ms = cfg.speech_pad_ms
        self.sample_rate = cfg.sample_rate
        self._model = cfg.model

    @staticmethod
    def _validate_input(path: Path) -> None:
        if not path.exists():
            msg = f"Input audio file does not exist: {path}"
            raise InvalidAudioError(msg)

        if not path.is_file():
            msg = f"Input path is not a file: {path}"
            raise InvalidAudioError(msg)

        if path.stat().st_size == 0:
            msg = f"Input audio file is empty: {path}"
            raise InvalidAudioError(msg)

    def _read_audio(self, path: Path) -> tuple[np.ndarray, int]:
        """Read audio, convert it to mono, and validate its samples."""
        try:
            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            msg = f"Unable to read audio file: {path}"
            raise InvalidAudioError(msg) from exc

        if audio.size == 0 or audio.shape[0] == 0:
            msg = f"Audio file contains no samples: {path}"
            raise InvalidAudioError(msg)

        if sample_rate <= 0:
            msg = f"Invalid audio sample rate: {sample_rate}"
            raise InvalidAudioError(msg)

        if not np.isfinite(audio).all():
            msg = f"Audio contains invalid samples: {path}"
            raise InvalidAudioError(msg)

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
            msg = f"Audio contains no usable samples: {path}"
            raise InvalidAudioError(msg)

        return np.ascontiguousarray(audio, dtype=np.float32)

    def _load_model(self) -> object:
        """Load and cache the Silero VAD ONNX model."""
        if self._model is not None:
            return self._model

        try:
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = (
                "Silero VAD is not installed. "
                "Install it using: poetry add silero-vad onnxruntime"
            )
            raise SileroVADNotFoundError(msg) from exc

        try:
            self._model = load_silero_vad(onnx=True)
        except Exception as exc:
            msg = "Unable to load the Silero VAD ONNX model."
            raise SpeechDetectorError(msg) from exc

        return self._model

    def _run_inference(
        self,
        audio: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run Silero VAD and return speech timestamps in seconds."""
        import torch  # type: ignore[import-untyped]
        from silero_vad import get_speech_timestamps

        model = self._load_model()
        waveform = torch.from_numpy(audio)

        try:
            with torch.inference_mode():
                timestamps: list[dict[str, Any]] = get_speech_timestamps(
                    waveform,
                    model,
                    threshold=self.threshold,
                    sampling_rate=self.sample_rate,
                    min_speech_duration_ms=(self.min_speech_duration_ms),
                    min_silence_duration_ms=(self.min_silence_duration_ms),
                    speech_pad_ms=self.speech_pad_ms,
                    return_seconds=True,
                )
        except Exception as exc:
            msg = "Silero VAD inference failed."
            raise SpeechDetectorError(msg) from exc

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
                msg = "Silero VAD returned an invalid speech interval."
                raise SpeechDetectorError(msg) from exc

            if not math.isfinite(start) or not math.isfinite(end):
                msg = "Silero VAD returned non-finite timestamps."
                raise SpeechDetectorError(msg)

            if start < 0 or end <= start:
                msg = "Silero VAD returned invalid interval boundaries."
                raise SpeechDetectorError(msg)

            start = min(start, duration_seconds)
            end = min(end, duration_seconds)

            if end <= start:
                continue

            regions.append(
                SpeechRegion(
                    start=start,
                    end=end,
                ),
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
