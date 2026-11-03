"""Audio alignment module for matching generated speech to original timeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FADER_LENGTH_SECONDS = 0.5


class AudioAlignmentError(Exception):
    """Base exception for audio alignment errors."""


class InvalidAudioError(AudioAlignmentError):
    """Raised when input audio is invalid."""


class OutputAlreadyExistsError(AudioAlignmentError):
    """Raised when output file already exists."""


@dataclass(frozen=True)
class AlignedSpeechSegment:
    """A speech segment aligned to its original timeline position."""

    start: float
    end: float
    text: str
    speaker_id: str
    audio_path: Path
    duration_seconds: float
    target_duration_seconds: float


@dataclass(frozen=True)
class AudioAlignmentResult:
    """Result of aligning generated speech to original timeline."""

    output_dir: Path
    manifest_path: Path
    segments: tuple[AlignedSpeechSegment, ...]


class AudioAlignerConfig:
    """Configuration for audio alignment operations."""

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        fader_length_seconds: float = DEFAULT_FADER_LENGTH_SECONDS,
    ) -> None:
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite.")

        if not math.isfinite(fader_length_seconds) or fader_length_seconds < 0:
            raise ValueError("fader_length_seconds must be non-negative.")

        self.sample_rate = sample_rate
        self.fader_length_seconds = fader_length_seconds


class AudioAligner:
    """Align generated speech to original timeline with crossfade transitions."""

    def __init__(self, config: AudioAlignerConfig | None = None) -> None:
        self.config = config or AudioAlignerConfig()

    @staticmethod
    def _validate_audio(path: Path) -> None:
        """Validate that an audio file exists and is non-empty."""
        if not path.exists():
            raise InvalidAudioError(f"Audio file does not exist: {path}")

        if not path.is_file():
            raise InvalidAudioError(f"Audio path is not a file: {path}")

        if path.stat().st_size == 0:
            raise InvalidAudioError(f"Audio file is empty: {path}")

    def _fade(
        self,
        audio: np.ndarray,
        fade_length_samples: int,
        *,
        reverse: bool = False,
    ) -> np.ndarray:
        """Apply a linear fade-in or fade-out to audio."""
        if fade_length_samples <= 0:
            return audio

        fade_samples = min(fade_length_samples, len(audio))
        fade_array = np.linspace(
            1.0 if reverse else 0.0,
            0.0 if reverse else 1.0,
            fade_samples,
            dtype=np.float32,
        )

        if reverse:
            return audio[:fade_samples] * fade_array
        else:
            return audio[-fade_samples:] * fade_array

    def _match_eq(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray,
    ) -> np.ndarray:
        """Match the EQ of generated audio to reference audio using spectral subtraction."""
        if len(generated_audio) == 0 or len(reference_audio) == 0:
            return generated_audio

        # Compute FFT
        n_fft = min(2048, len(reference_audio))
        gen_fft = np.fft.rfft(generated_audio[:n_fft])
        ref_fft = np.fft.rfft(reference_audio[:n_fft])

        # Compute magnitude spectra
        ref_mag = np.abs(ref_fft)
        # Avoid division by zero
        ref_mag_safe = np.maximum(ref_mag, 1e-8)

        # Compute gain to match reference spectrum
        gain = ref_mag / ref_mag_safe

        # Apply gain to generated audio spectrum
        corrected_fft = gen_fft * gain

        # Inverse FFT
        corrected_audio = np.fft.irfft(corrected_fft, n=n_fft)

        # Normalize to original length if needed
        if len(corrected_audio) < len(generated_audio):
            corrected_audio = np.pad(
                corrected_audio,
                (0, len(generated_audio) - len(corrected_audio)),
                mode="constant",
            )

        return corrected_audio

    def align(
        self,
        generated_path: str | Path,
        reference_path: str | Path,
        output_path: str | Path,
        *,
        crossfade_duration_seconds: float | None = None,
    ) -> AlignedSpeechSegment:
        """Align a generated speech segment to its original timeline position.

        Args:
            generated_path: Path to the generated speech WAV file.
            reference_path: Path to the original source audio for EQ matching.
            output_path: Path where the aligned output will be saved.
            crossfade_duration_seconds: Optional duration for fade transitions.
                If None, uses config.fader_length_seconds.

        Returns:
            AlignedSpeechSegment with metadata about the aligned segment.

        Raises:
            InvalidAudioError: If input audio files are invalid.
            OutputAlreadyExistsError: If output file already exists.
        """
        generated = Path(generated_path)
        reference = Path(reference_path)
        destination = Path(output_path)

        # Validate inputs
        self._validate_audio(generated)
        self._validate_audio(reference)

        # Check if output already exists
        if destination.exists() and not destination.is_symlink():
            raise OutputAlreadyExistsError(
                f"Output file already exists: {destination}",
            )

        # Read audio files
        try:
            generated_audio, gen_sample_rate = sf.read(
                str(generated),
                dtype="float32",
                always_2d=True,
            )
            reference_audio, ref_sample_rate = sf.read(
                str(reference),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise InvalidAudioError(
                "Unable to read audio files for alignment.",
            ) from exc

        # Use reference sample rate
        sample_rate = ref_sample_rate

        # Convert to mono if stereo
        if generated_audio.shape[0] > 1:
            generated_audio = np.mean(generated_audio, axis=0)
        if reference_audio.shape[0] > 1:
            reference_audio = np.mean(reference_audio, axis=0)

        # Match EQ to reference audio
        aligned_audio = self._match_eq(generated_audio, reference_audio)

        # Apply crossfade if needed
        fade_duration = crossfade_duration_seconds or self.config.fader_length_seconds
        fade_samples = int(fade_duration * sample_rate)

        if fade_samples > 0:
            # Fade out at the end of generated audio
            faded_generated = np.concatenate(
                [
                    generated_audio[:-fade_samples],
                    self._fade(generated_audio, fade_samples, reverse=True),
                ]
            )

            # Fade in from reference audio
            faded_reference = np.concatenate(
                [
                    self._fade(reference_audio, fade_samples, reverse=True),
                    reference_audio[fade_samples:],
                ]
            )

            # Blend the two
            blend_length = min(len(faded_generated), len(faded_reference))
            if blend_length > 0:
                blend_mask = np.linspace(
                    0.5,
                    1.0,
                    blend_length,
                    dtype=np.float32,
                )
                blended = faded_generated[:blend_length] * blend_mask + faded_reference[
                    :blend_length
                ] * (1.0 - blend_mask)

                # Pad or trim to match original duration
                if len(blended) < len(reference_audio):
                    blended = np.pad(
                        blended,
                        (0, len(reference_audio) - len(blended)),
                        mode="constant",
                    )
                else:
                    blended = blended[: len(reference_audio)]

                aligned_audio = blended

        # Write output
        try:
            sf.write(
                str(destination),
                aligned_audio,
                sample_rate,
                subtype="PCM_16",
            )
        except Exception as exc:
            raise InvalidAudioError(
                f"Unable to write aligned audio: {destination}",
            ) from exc

        # Calculate duration
        try:
            info = sf.info(str(destination))
            duration = float(info.duration)
        except Exception:
            duration = len(aligned_audio) / sample_rate

        return AlignedSpeechSegment(
            start=0.0,  # Would need timeline metadata to set properly
            end=duration,
            text="",  # Would need text metadata
            speaker_id="",  # Would need speaker ID metadata
            audio_path=destination,
            duration_seconds=duration,
            target_duration_seconds=duration,
        )
