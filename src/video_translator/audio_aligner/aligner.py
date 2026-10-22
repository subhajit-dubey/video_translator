from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyrubberband as pyrb
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt
from math import gcd


class AudioAlignmentError(Exception):
    """Raised when audio alignment or post-processing fails."""


@dataclass(frozen=True)
class AudioAlignerConfig:
    """Configuration for timeline alignment and audio post-processing."""

    sample_rate: int = 24000
    min_stretch_rate: float = 0.67
    max_stretch_rate: float = 1.50
    eq_strength: float = 0.35
    max_eq_gain_db: float = 6.0
    fade_seconds: float = 0.005
    peak_limit: float = 0.98

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        if (
            not math.isfinite(self.min_stretch_rate)
            or not math.isfinite(self.max_stretch_rate)
            or self.min_stretch_rate <= 0
            or self.max_stretch_rate < self.min_stretch_rate
        ):
            raise ValueError("Invalid time-stretch rate limits.")

        if not math.isfinite(self.eq_strength) or not 0 <= self.eq_strength <= 1:
            raise ValueError("eq_strength must be between 0 and 1.")

        if (
            not math.isfinite(self.max_eq_gain_db)
            or self.max_eq_gain_db < 0
        ):
            raise ValueError("max_eq_gain_db must be non-negative.")

        if not math.isfinite(self.fade_seconds) or self.fade_seconds < 0:
            raise ValueError("fade_seconds must be non-negative.")

        if not math.isfinite(self.peak_limit) or not 0 < self.peak_limit < 1:
            raise ValueError("peak_limit must be between 0 and 1.")


@dataclass(frozen=True)
class AlignedSpeechSegment:
    """A synthesized segment adjusted to its target timeline interval."""

    start: float
    end: float
    text: str
    speaker_id: str
    source_audio_path: Path
    audio_path: Path
    source_duration_seconds: float
    target_duration_seconds: float
    stretch_rate: float
    eq_gains_db: dict[str, float]

    @property
    def duration_seconds(self) -> float:
        return self.target_duration_seconds


@dataclass(frozen=True)
class AudioAlignmentResult:
    """Paths and metadata produced by the audio alignment pipeline."""

    output_dir: Path
    aligned_audio_path: Path
    manifest_path: Path
    sample_rate: int
    duration_seconds: float
    mix_gain: float
    segments: tuple[AlignedSpeechSegment, ...]


class AudioAligner:
    """Align synthesized speech to the original timeline.

    Responsibilities:
        1. Adjust speech duration using pitch-preserving Rubber Band stretching.
        2. Match broad frequency-band balance to the original audio.
        3. Apply short fades to reduce boundary clicks.
        4. Place each segment at its intended timeline position.
        5. Prevent clipping when overlapping segments are mixed.
        6. Save aligned segments, a timeline WAV, and a JSON manifest.
    """

    def __init__(
        self,
        *,
        config: AudioAlignerConfig | None = None,
    ) -> None:
        self.config = config or AudioAlignerConfig()

    @staticmethod
    def _read_mono(path: Path) -> tuple[np.ndarray, int]:
        """Read an audio file and convert it to mono float32."""
        if not path.is_file() or path.stat().st_size == 0:
            raise AudioAlignmentError(
                f"Audio file is missing or empty: {path}"
            )

        try:
            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise AudioAlignmentError(
                f"Unable to read audio file: {path}"
            ) from exc

        if audio.shape[0] == 0 or sample_rate <= 0:
            raise AudioAlignmentError(
                f"Audio file contains no usable samples: {path}"
            )

        audio = np.mean(audio, axis=1, dtype=np.float32)

        if not np.isfinite(audio).all():
            raise AudioAlignmentError(
                f"Audio contains non-finite samples: {path}"
            )

        return audio, int(sample_rate)

    @staticmethod
    def _resample(
        audio: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Resample audio without changing its intended duration."""
        if source_rate == target_rate:
            return np.asarray(audio, dtype=np.float32)

        common = gcd(source_rate, target_rate)

        try:
            result = resample_poly(
                audio,
                target_rate // common,
                source_rate // common,
            )
        except Exception as exc:
            raise AudioAlignmentError(
                f"Unable to resample audio from {source_rate} "
                f"Hz to {target_rate} Hz."
            ) from exc

        return np.asarray(result, dtype=np.float32)

    @staticmethod
    def _fit_length(
        audio: np.ndarray,
        target_samples: int,
    ) -> np.ndarray:
        """Trim or zero-pad audio to an exact sample count."""
        if target_samples <= 0:
            raise AudioAlignmentError(
                "Target audio length must be positive."
            )

        if len(audio) > target_samples:
            return audio[:target_samples].copy()

        if len(audio) < target_samples:
            return np.pad(
                audio,
                (0, target_samples - len(audio)),
                mode="constant",
            ).astype(np.float32)

        return audio.astype(np.float32, copy=True)

    def _stretch(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_samples: int,
    ) -> tuple[np.ndarray, float]:
        """Time-stretch speech to its target duration without changing pitch."""
        if len(audio) == 0:
            raise AudioAlignmentError(
                "Cannot stretch an empty audio segment."
            )

        rate = len(audio) / target_samples

        if not math.isfinite(rate) or rate <= 0:
            raise AudioAlignmentError(
                "Calculated time-stretch rate is invalid."
            )

        if not (
            self.config.min_stretch_rate
            <= rate
            <= self.config.max_stretch_rate
        ):
            raise AudioAlignmentError(
                f"Required stretch rate {rate:.3f} is outside the supported "
                f"range [{self.config.min_stretch_rate:.3f}, "
                f"{self.config.max_stretch_rate:.3f}]. "
                "Adjust the generated segment or stretch-rate limits."
            )

        # [Rubber Band] A larger rate means faster playback and shorter audio.
        if abs(rate - 1.0) > 0.01:
            try:
                audio = pyrb.time_stretch(
                    np.asarray(audio, dtype=np.float32),
                    sample_rate,
                    rate,
                )
            except Exception as exc:
                raise AudioAlignmentError(
                    "Rubber Band time stretching failed. Ensure "
                    "rubberband-cli is installed and available on PATH."
                ) from exc

        # Correct small frame-count differences introduced by processing.
        audio = self._fit_length(
            np.asarray(audio, dtype=np.float32),
            target_samples,
        )

        return audio, rate

    @staticmethod
    def _filter_band(
        audio: np.ndarray,
        sos: np.ndarray,
    ) -> np.ndarray:
        """Apply a zero-phase filter, with a fallback for short recordings."""
        if len(audio) < 64:
            return sosfilt(sos, audio).astype(np.float32)

        try:
            filtered = sosfiltfilt(sos, audio)
        except ValueError:
            filtered = sosfilt(sos, audio)

        return np.asarray(filtered, dtype=np.float32)

    def _match_eq(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Match the broad low, mid, and high frequency balance.

        This is a conservative three-band EQ approximation, not full
        frequency-dependent spectral-envelope matching.
        """
        names = ("low", "mid", "high")
        gains = {name: 0.0 for name in names}

        if self.config.eq_strength == 0 or len(reference) < 64:
            return generated, gains

        generated_rms = float(np.sqrt(np.mean(np.square(generated))))
        reference_rms = float(np.sqrt(np.mean(np.square(reference))))

        if generated_rms < 1e-5 or reference_rms < 1e-5:
            return generated, gains

        nyquist = sample_rate / 2.0
        low_cut = min(250.0, nyquist * 0.20)
        high_cut = min(3000.0, nyquist * 0.75)

        if high_cut <= low_cut:
            return generated, gains

        filters = (
            butter(
                4,
                low_cut,
                btype="lowpass",
                fs=sample_rate,
                output="sos",
            ),
            butter(
                4,
                [low_cut, high_cut],
                btype="bandpass",
                fs=sample_rate,
                output="sos",
            ),
            butter(
                4,
                high_cut,
                btype="highpass",
                fs=sample_rate,
                output="sos",
            ),
        )

        processed = np.zeros_like(generated, dtype=np.float32)

        for name, sos in zip(names, filters, strict=True):
            generated_band = self._filter_band(generated, sos)
            reference_band = self._filter_band(reference, sos)

            generated_band_rms = float(
                np.sqrt(np.mean(np.square(generated_band)))
            )
            reference_band_rms = float(
                np.sqrt(np.mean(np.square(reference_band)))
            )

            gain_db = 0.0

            if generated_band_rms >= 1e-5 and reference_band_rms >= 1e-5:
                gain_db = 20.0 * math.log10(
                    reference_band_rms / generated_band_rms
                )

                gain_db = float(
                    np.clip(
                        gain_db,
                        -self.config.max_eq_gain_db,
                        self.config.max_eq_gain_db,
                    )
                )

                gain_db *= self.config.eq_strength

            gains[name] = gain_db

            processed += generated_band * np.float32(
                10.0 ** (gain_db / 20.0)
            )

        if not np.isfinite(processed).all():
            raise AudioAlignmentError(
                "EQ matching produced non-finite audio samples."
            )

        return processed, gains

    def _fade(self, audio: np.ndarray) -> np.ndarray:
        """Apply short raised-cosine fades at both segment boundaries."""
        audio = np.asarray(audio, dtype=np.float32).copy()

        fade_samples = min(
            int(round(self.config.fade_seconds * self.config.sample_rate)),
            len(audio) // 2,
        )

        if fade_samples <= 1:
            return audio

        ramp = np.sin(
            np.linspace(
                0.0,
                math.pi / 2.0,
                fade_samples,
                endpoint=True,
            )
        ) ** 2

        audio[:fade_samples] *= ramp.astype(np.float32)
        audio[-fade_samples:] *= ramp[::-1].astype(np.float32)

        return audio

    def align(
        self,
        generation_result: Any,
        source_audio_path: str | Path,
        output_dir: str | Path,
        *,
        output_filename: str = "aligned_dubbing.wav",
        manifest_filename: str = "audio_alignment_manifest.json",
    ) -> AudioAlignmentResult:
        """Align VoiceGenerationResult segments to the source timeline."""
        source_path = Path(source_audio_path)
        destination = Path(output_dir)

        destination.mkdir(parents=True, exist_ok=True)

        source_audio, source_rate = self._read_mono(source_path)
        source_audio = self._resample(
            source_audio,
            source_rate,
            self.config.sample_rate,
        )

        sample_rate = self.config.sample_rate
        source_duration = len(source_audio) / sample_rate

        segments = tuple(generation_result.segments)

        for segment in segments:
            if (
                not math.isfinite(float(segment.start))
                or not math.isfinite(float(segment.end))
                or segment.start < 0
                or segment.end <= segment.start
            ):
                raise AudioAlignmentError(
                    f"Invalid segment timestamps: "
                    f"{segment.start} - {segment.end}"
                )

        timeline_duration = max(
            [source_duration]
            + [float(segment.end) for segment in segments]
        )

        timeline_samples = max(
            1,
            int(round(timeline_duration * sample_rate)),
        )

        timeline_mix = np.zeros(
            timeline_samples,
            dtype=np.float32,
        )

        aligned_dir = destination / "aligned_segments"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        aligned_segments: list[AlignedSpeechSegment] = []

        for index, segment in enumerate(segments, start=1):
            input_path = Path(segment.audio_path)
            generated_audio, generated_rate = self._read_mono(input_path)

            generated_audio = self._resample(
                generated_audio,
                generated_rate,
                sample_rate,
            )

            source_duration_seconds = len(generated_audio) / sample_rate

            start_frame = int(round(float(segment.start) * sample_rate))
            end_frame = int(round(float(segment.end) * sample_rate))
            target_samples = end_frame - start_frame

            if target_samples <= 0:
                raise AudioAlignmentError(
                    f"Segment {index} has no usable target samples."
                )

            target_duration_seconds = target_samples / sample_rate

            stretched_audio, stretch_rate = self._stretch(
                generated_audio,
                sample_rate,
                target_samples,
            )

            # Use the corresponding original source interval as the EQ reference.
            reference_audio = source_audio[
                min(start_frame, len(source_audio)):
                min(end_frame, len(source_audio))
            ]

            eq_audio, eq_gains = self._match_eq(
                stretched_audio,
                reference_audio,
                sample_rate,
            )

            eq_audio = self._fade(
                self._fit_length(eq_audio, target_samples)
            )

            aligned_path = aligned_dir / f"segment_{index:04d}.wav"

            try:
                sf.write(
                    str(aligned_path),
                    eq_audio,
                    sample_rate,
                    subtype="PCM_24",
                )
            except Exception as exc:
                raise AudioAlignmentError(
                    f"Unable to save aligned segment: {aligned_path}"
                ) from exc

            mix_end = min(
                start_frame + len(eq_audio),
                len(timeline_mix),
            )
            mix_length = mix_end - start_frame

            if mix_length > 0:
                timeline_mix[start_frame:mix_end] += eq_audio[:mix_length]

            aligned_segments.append(
                AlignedSpeechSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text),
                    speaker_id=str(segment.speaker_id),
                    source_audio_path=input_path,
                    audio_path=aligned_path,
                    source_duration_seconds=source_duration_seconds,
                    target_duration_seconds=target_duration_seconds,
                    stretch_rate=stretch_rate,
                    eq_gains_db=eq_gains,
                )
            )

        # Apply one global gain reduction if overlapping segments would clip.
        peak = float(np.max(np.abs(timeline_mix))) if timeline_mix.size else 0.0

        mix_gain = (
            self.config.peak_limit / peak
            if peak > self.config.peak_limit
            else 1.0
        )

        if mix_gain < 1.0:
            timeline_mix *= np.float32(mix_gain)

        aligned_audio_path = destination / output_filename
        manifest_path = destination / manifest_filename

        try:
            sf.write(
                str(aligned_audio_path),
                timeline_mix,
                sample_rate,
                subtype="PCM_24",
            )
        except Exception as exc:
            raise AudioAlignmentError(
                f"Unable to save aligned timeline: {aligned_audio_path}"
            ) from exc

        manifest = {
            "source_audio_path": str(source_path.resolve()),
            "aligned_audio_path": str(aligned_audio_path.resolve()),
            "sample_rate": sample_rate,
            "duration_seconds": len(timeline_mix) / sample_rate,
            "mix_gain": mix_gain,
            "configuration": asdict(self.config),
            "segments": [
                {
                    "start": item.start,
                    "end": item.end,
                    "text": item.text,
                    "speaker_id": item.speaker_id,
                    "source_audio_path": str(item.source_audio_path.resolve()),
                    "audio_path": str(item.audio_path.resolve()),
                    "source_duration_seconds": (
                        item.source_duration_seconds
                    ),
                    "target_duration_seconds": (
                        item.target_duration_seconds
                    ),
                    "stretch_rate": item.stretch_rate,
                    "eq_gains_db": item.eq_gains_db,
                }
                for item in aligned_segments
            ],
        }

        try:
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise AudioAlignmentError(
                f"Unable to write alignment manifest: {manifest_path}"
            ) from exc

        return AudioAlignmentResult(
            output_dir=destination,
            aligned_audio_path=aligned_audio_path,
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            duration_seconds=len(timeline_mix) / sample_rate,
            mix_gain=mix_gain,
            segments=tuple(aligned_segments),
        )