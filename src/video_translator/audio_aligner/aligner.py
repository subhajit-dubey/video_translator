"""Audio alignment module for matching generated speech to original timeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    aligned_audio_path: Path
    duration_seconds: float


class AudioAlignerConfig:
    """Configuration for audio alignment operations."""

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        fader_length_seconds: float = DEFAULT_FADER_LENGTH_SECONDS,
        # Backward compatibility parameters (not used in new API)
        eq_strength: float = 0.35,
        max_eq_gain_db: float = 6.0,
        fade_seconds: float | None = None,
        min_stretch_rate: float = 0.67,
        max_stretch_rate: float = 1.50,
    ) -> None:
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be positive and finite.")

        if not math.isfinite(fader_length_seconds) or fader_length_seconds < 0:
            raise ValueError("fader_length_seconds must be non-negative.")

        self.sample_rate = sample_rate
        self.fader_length_seconds = fader_length_seconds
        # Store for backward compatibility
        self.eq_strength = eq_strength
        self.max_eq_gain_db = max_eq_gain_db
        self.fade_seconds = (
            fade_seconds if fade_seconds is not None else fader_length_seconds
        )
        self.min_stretch_rate = min_stretch_rate
        self.max_stretch_rate = max_stretch_rate


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

        # Use the full length of generated audio for FFT so no samples are lost
        n_fft = len(generated_audio)

        # Compute FFT on the full generated audio
        gen_fft = np.fft.rfft(generated_audio)

        # Use reference audio truncated to the same length as generated audio
        ref_segment = (
            reference_audio[:n_fft]
            if len(reference_audio) >= n_fft
            else np.pad(reference_audio, (0, n_fft - len(reference_audio)))
        )
        ref_fft = np.fft.rfft(ref_segment)

        # Compute magnitude spectra
        ref_mag = np.abs(ref_fft)
        # Avoid division by zero
        ref_mag_safe = np.maximum(ref_mag, 1e-8)

        # Compute gain to match reference spectrum
        gain = ref_mag / ref_mag_safe

        # Apply gain to generated audio spectrum
        corrected_fft = gen_fft * gain

        # Inverse FFT — full-length output, no truncation
        corrected_audio = np.fft.irfft(corrected_fft, n=n_fft)

        # Ensure output length matches generated audio length
        if len(corrected_audio) < len(generated_audio):
            corrected_audio = np.pad(
                corrected_audio,
                (0, len(generated_audio) - len(corrected_audio)),
                mode="constant",
            )
        elif len(corrected_audio) > len(generated_audio):
            corrected_audio = corrected_audio[: len(generated_audio)]

        return corrected_audio

    def align(
        self,
        generated_path: str | Path | None = None,
        reference_path: str | Path | None = None,
        output_path: str | Path | None = None,
        *,
        crossfade_duration_seconds: float | None = None,
        # Batch mode parameters (for backward compatibility)
        generation_result: Any = None,
        source_audio_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        max_duration_seconds: float | None = None,
    ) -> AlignedSpeechSegment | AudioAlignmentResult:
        """Align a generated speech segment (single) or batch of segments.

        Single-segment mode (new API):
            align(generated_path, reference_path, output_path, crossfade_duration_seconds?)

        Batch mode (legacy API):
            align(generation_result=..., source_audio_path=..., output_dir=..., max_duration_seconds?)

        Args:
            generated_path: Path to the generated speech WAV file (single mode).
            reference_path: Path to the original source audio for EQ matching (single mode).
            output_path: Path where the aligned output will be saved (single mode).
            crossfade_duration_seconds: Optional duration for fade transitions (single mode).
            generation_result: Object with 'segments' attribute (batch mode).
            source_audio_path: Path to the original source audio (batch mode).
            output_dir: Directory where aligned segments will be saved (batch mode).
            max_duration_seconds: Optional maximum duration cap (batch mode).

        Returns:
            AlignedSpeechSegment (single mode) or AudioAlignmentResult (batch mode).

        Raises:
            InvalidAudioError: If input audio files are invalid.
            OutputAlreadyExistsError: If output file already exists.
            AudioAlignmentError: If alignment fails.
        """
        # Detect batch mode
        if generation_result is not None:
            return self._align_batch(
                generation_result=generation_result,
                source_audio_path=source_audio_path,
                output_dir=output_dir,
                max_duration_seconds=max_duration_seconds,
            )

        # Single-segment mode (new API)
        if generated_path is None:
            raise AudioAlignmentError(
                "generated_path is required for single-segment mode"
            )
        if reference_path is None:
            raise AudioAlignmentError(
                "reference_path is required for single-segment mode"
            )
        if output_path is None:
            raise AudioAlignmentError("output_path is required for single-segment mode")
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
            generated_audio, _gen_sample_rate = sf.read(
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
        except (RuntimeError, OSError):
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

    def _align_batch(
        self,
        generation_result: Any,
        source_audio_path: str | Path | None,
        output_dir: str | Path | None,
        *,
        max_duration_seconds: float | None = None,
    ) -> AudioAlignmentResult:
        """Align all generated speech segments to original timeline (batch mode).

        This is the legacy batch processing API that processes all segments
        and stitches them into a single timeline audio file.

        Args:
            generation_result: Object with 'segments' attribute containing
                segment objects with start, end, text, speaker_id, audio_path.
            source_audio_path: Path to the original source audio for reference.
            output_dir: Directory where aligned segments and timeline will be saved.
            max_duration_seconds: Optional maximum duration cap for output.

        Returns:
            AudioAlignmentResult with aligned timeline audio and segment metadata.

        Raises:
            AudioAlignmentError: If alignment fails.
            InvalidAudioError: If input audio files are invalid.
        """
        if source_audio_path is None:
            raise AudioAlignmentError("source_audio_path is required for batch mode")
        if output_dir is None:
            raise AudioAlignmentError("output_dir is required for batch mode")
        source_path = Path(source_audio_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._validate_audio(source_path)

        # Validate segments
        segments = getattr(generation_result, "segments", [])
        if not segments:
            raise AudioAlignmentError("No segments to align.")

        # Check if any segment exceeds max duration
        if max_duration_seconds is not None:
            for segment in segments:
                if segment.end > max_duration_seconds:
                    raise AudioAlignmentError(
                        f"Segment ends at {segment.end:.2f}s, beyond the "
                        f"{max_duration_seconds:.2f}s maximum duration."
                    )

        # Read source audio for reference
        source_audio, sample_rate = sf.read(str(source_path), dtype="float32")
        if source_audio.ndim > 1:
            source_audio = np.mean(source_audio, axis=0)

        # Determine output duration
        output_duration = max_duration_seconds or max(s.end for s in segments)
        if max_duration_seconds is None:
            output_duration = max(output_duration, len(source_audio) / sample_rate)
        output_samples = int(output_duration * sample_rate)

        # Initialize output timeline with silence
        timeline_audio = np.zeros(output_samples, dtype=np.float32)

        aligned_segments = []

        for segment in segments:
            # Validate segment
            if not hasattr(segment, "audio_path") or not segment.audio_path:
                raise AudioAlignmentError(f"Segment missing audio_path: {segment}")

            generated_path = Path(segment.audio_path)
            self._validate_audio(generated_path)

            # Read generated audio
            generated_audio, gen_sr = sf.read(str(generated_path), dtype="float32")
            if generated_audio.ndim > 1:
                generated_audio = np.mean(generated_audio, axis=0)

            # Resample if needed
            if gen_sr != sample_rate:
                # Simple linear interpolation resampling
                ratio = sample_rate / gen_sr
                new_length = int(len(generated_audio) * ratio)
                indices = np.linspace(0, len(generated_audio) - 1, new_length)
                generated_audio = np.interp(
                    indices, np.arange(len(generated_audio)), generated_audio
                )

            # Match EQ to source audio
            aligned_audio = self._match_eq(generated_audio, source_audio)

            # Calculate position in timeline
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            segment_length = end_sample - start_sample

            # Trim or pad aligned audio to fit segment duration
            if len(aligned_audio) > segment_length:
                aligned_audio = aligned_audio[:segment_length]
            elif len(aligned_audio) < segment_length:
                aligned_audio = np.pad(
                    aligned_audio,
                    (0, segment_length - len(aligned_audio)),
                    mode="constant",
                )

            # Apply crossfade at boundaries
            fade_samples = int(self.config.fader_length_seconds * sample_rate)
            if fade_samples > 0 and len(aligned_audio) > fade_samples:
                # Fade in
                fade_in = np.linspace(0.0, 1.0, min(fade_samples, len(aligned_audio)))
                aligned_audio[: len(fade_in)] *= fade_in
                # Fade out
                fade_out = np.linspace(1.0, 0.0, min(fade_samples, len(aligned_audio)))
                aligned_audio[-len(fade_out) :] *= fade_out

            # Place in timeline (overlay/mix)
            timeline_end = min(start_sample + len(aligned_audio), output_samples)
            timeline_start = start_sample
            audio_end = timeline_end - timeline_start

            if timeline_start < output_samples and audio_end > 0:
                timeline_audio[timeline_start:timeline_end] += aligned_audio[:audio_end]

            # Save individual aligned segment
            segment_output = (
                output_path
                / f"aligned_{segment.speaker_id}_{segment.start:.3f}_{segment.end:.3f}.wav"
            )
            try:
                sf.write(
                    str(segment_output),
                    aligned_audio[:audio_end],
                    sample_rate,
                    subtype="PCM_16",
                )
            except Exception as exc:
                raise AudioAlignmentError(
                    f"Unable to write aligned segment: {segment_output}",
                ) from exc

            aligned_segments.append(
                AlignedSpeechSegment(
                    start=segment.start,
                    end=segment.end,
                    text=getattr(segment, "text", ""),
                    speaker_id=getattr(segment, "speaker_id", ""),
                    audio_path=segment_output,
                    duration_seconds=float(audio_end) / sample_rate,
                    target_duration_seconds=segment.end - segment.start,
                )
            )

        # Write final timeline audio
        timeline_output = output_path / "aligned_timeline.wav"
        try:
            # Normalize to prevent clipping
            max_val = np.max(np.abs(timeline_audio))
            if max_val > 1.0:
                timeline_audio = timeline_audio / max_val * 0.95

            sf.write(
                str(timeline_output),
                timeline_audio,
                sample_rate,
                subtype="PCM_16",
            )
        except Exception as exc:
            raise AudioAlignmentError(
                f"Unable to write aligned timeline: {timeline_output}",
            ) from exc

        # Write manifest
        manifest_path = output_path / "alignment_manifest.json"
        manifest_data = {
            "source_audio": str(source_path),
            "output_audio": str(timeline_output),
            "sample_rate": sample_rate,
            "duration_seconds": output_duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker_id": s.speaker_id,
                    "audio_path": str(s.audio_path),
                    "duration_seconds": s.duration_seconds,
                    "target_duration_seconds": s.target_duration_seconds,
                }
                for s in aligned_segments
            ],
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        return AudioAlignmentResult(
            output_dir=output_path,
            manifest_path=manifest_path,
            segments=tuple(aligned_segments),
            aligned_audio_path=timeline_output,
            duration_seconds=output_duration,
        )
