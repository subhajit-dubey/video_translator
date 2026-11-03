"""Voice generator module for synthesizing speech from text translations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore[import-untyped]

DEFAULT_TTS_DEVICE = (
    "cuda"
    if hasattr(np, "__version__")
    and "cuda"
    in str(
        np.__version__,
    )
    else "cpu"
)
DEFAULT_TTS_MODEL = "piper"
DEFAULT_HINDI_TTS_MODEL = "hinditts"
DEFAULT_PIPER_MALE_VOICE = "male"
DEFAULT_PIPER_FEMALE_VOICE = "female"
DEFAULT_TTS_VOCAB_FILE = "vocab.json"
DEFAULT_TTS_CKPT_FILE = "model.pt"
DEFAULT_NUM_SPEAKER_PROFILE_SECONDS = 3.0
DEFAULT_PIPER_VOICE_PITCH_THRESHOLD_HZ = 250
DEFAULT_DIARIZATION_MODEL = "pyannote/voice-activity-detection"


class VoiceGeneratorError(Exception):
    """Base exception for voice generation errors."""


class VoiceSynthesisError(VoiceGeneratorError):
    """Raised when TTS synthesis fails."""


class UnsupportedTTSConfigurationError(VoiceGeneratorError):
    """Raised when TTS configuration is not supported."""


class HuggingFaceTokenError(VoiceGeneratorError):
    """Raised when Hugging Face token is invalid or missing."""


class F5TTSNotFoundError(VoiceGeneratorError):
    """Raised when F5-TTS is not installed."""


class F5TTSModelLoadError(VoiceGeneratorError):
    """Raised when F5-TTS model fails to load."""


class NoSpeakersDetectedError(VoiceGeneratorError):
    """Raised when no speakers are detected in the audio."""


class DiarizationError(VoiceGeneratorError):
    """Raised when speaker diarization fails."""


class PyannoteNotFoundError(VoiceGeneratorError):
    """Raised when PyAnnote is not installed."""


class InvalidTranslationFileError(VoiceGeneratorError):
    """Raised when translation file is invalid."""


class PiperTTSSynthesizer:
    """Piper TTS synthesizer for generating speech from text."""

    def __init__(
        self,
        *,
        device: str = DEFAULT_TTS_DEVICE,
        model_name: str = DEFAULT_HINDI_TTS_MODEL,
        voice: str = DEFAULT_PIPER_FEMALE_VOICE,
        vocab_file: str = DEFAULT_TTS_VOCAB_FILE,
        ckpt_file: str = DEFAULT_TTS_CKPT_FILE,
    ) -> None:
        if not model_name or not ckpt_file or not vocab_file:
            raise UnsupportedTTSConfigurationError(
                "Piper TTS requires model name, checkpoint file, and vocabulary file.",
            )

        self.device = device
        self.model_name = model_name
        self.voice = voice
        self.vocab_file = vocab_file
        self.ckpt_file = ckpt_file

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
    ) -> None:
        """Synthesize speech from text and save to WAV file.

        Args:
            text: Text to synthesize.
            output_path: Path where the generated WAV file will be saved.

        Raises:
            VoiceSynthesisError: If synthesis fails.
        """
        # Implementation would use Piper TTS library here
        # This is a placeholder - actual implementation depends on Piper TTS


class F5TTSSynthesizer:
    """F5-TTS synthesizer for high-quality speech generation."""

    def __init__(self) -> None:
        self.model = None  # Would load model here

    def synthesize(
        self,
        text: str,
        reference_audio_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Synthesize speech from text using F5-TTS.

        Args:
            text: Text to synthesize.
            reference_audio_path: Reference audio for voice cloning.
            output_path: Path where the generated WAV file will be saved.

        Raises:
            VoiceSynthesisError: If synthesis fails.
        """
        # Implementation would use F5-TTS library here


class PyannoteSpeakerDiarizer:
    """Speaker diarization using PyAnnote."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DIARIZATION_MODEL,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        clustering_threshold: float | None = None,
        min_cluster_size: float | None = None,
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
        return {}


@dataclass(frozen=True)
class SpeakerProfile:
    """Profile for a speaker's voice characteristics."""

    speaker_id: str
    pitch_hz: float
    duration_seconds: float
    audio_path: Path | None = None


@dataclass(frozen=True)
class SpeakerTurn:
    """A turn spoken by a single speaker."""

    speaker_id: str
    start_time: float
    end_time: float
    text: str


@dataclass(frozen=True)
class TranslatedSegment:
    """A translated segment with voice assignment."""

    speaker_id: str
    text: str
    audio_path: Path | None = None


@dataclass(frozen=True)
class VoiceGenerationResult:
    """Result of generating voices for all speakers."""

    output_dir: Path
    manifest_path: Path
    segments: tuple[GeneratedSpeechSegment, ...]


@dataclass(frozen=True)
class GeneratedSpeechSegment:
    """A generated speech segment with voice information."""

    speaker_id: str
    text: str
    audio_path: Path
    duration_seconds: float
    target_duration_seconds: float


class VoiceGenerator:
    """Generate voices for all speakers in a translated script."""

    def __init__(
        self,
        *,
        device: str = DEFAULT_TTS_DEVICE,
        model_name: str = DEFAULT_HINDI_TTS_MODEL,
        voice: str = DEFAULT_PIPER_FEMALE_VOICE,
        vocab_file: str = DEFAULT_TTS_VOCAB_FILE,
        ckpt_file: str = DEFAULT_TTS_CKPT_FILE,
        num_speaker_profile_seconds: float = DEFAULT_NUM_SPEAKER_PROFILE_SECONDS,
        piper_voice_pitch_threshold_hz: int = DEFAULT_PIPER_VOICE_PITCH_THRESHOLD_HZ,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.voice = voice
        self.vocab_file = vocab_file
        self.ckpt_file = ckpt_file
        self.num_speaker_profile_seconds = num_speaker_profile_seconds
        self.piper_voice_pitch_threshold_hz = piper_voice_pitch_threshold_hz

    def generate(
        self,
        translated_path: str | Path,
        output_dir: str | Path,
        *,
        reference_audio_path: str | Path | None = None,
    ) -> VoiceGenerationResult:
        """Generate voices for all speakers in a translated script.

        Args:
            translated_path: Path to the translated text file.
            output_dir: Directory where generated audio files will be saved.
            reference_audio_path: Optional reference audio for voice cloning.

        Returns:
            VoiceGenerationResult with generated audio segments.

        Raises:
            VoiceSynthesisError: If voice generation fails.
            UnsupportedTTSConfigurationError: If TTS configuration is invalid.
        """
        translated = Path(translated_path)
        destination = Path(output_dir)

        # Validate inputs
        if not translated.exists():
            raise VoiceSynthesisError(f"Translated file does not exist: {translated}")

        if not destination.exists():
            destination.mkdir(parents=True, exist_ok=True)

        # Read translated text
        try:
            with open(translated, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            raise VoiceSynthesisError(
                f"Unable to read translated file: {translated}",
            ) from exc

        # Parse translated text into segments
        segments = self._parse_translated_text(content)

        # Generate voices for each segment
        generated_segments: dict[str, GeneratedSpeechSegment] = {}
        for speaker_id, text in segments.items():
            try:
                audio_path = self._synthesize_speaker_voice(
                    speaker_id=speaker_id,
                    text=text,
                    output_dir=destination,
                    reference_audio_path=reference_audio_path,
                )
                duration = self._get_audio_duration(audio_path)
                generated_segments[speaker_id] = GeneratedSpeechSegment(
                    speaker_id=speaker_id,
                    text=text,
                    audio_path=audio_path,
                    duration_seconds=duration,
                    target_duration_seconds=duration,
                )
            except Exception as exc:
                raise VoiceSynthesisError(
                    f"Failed to generate voice for speaker {speaker_id}."
                ) from exc

        # Create manifest
        manifest_path = destination / "segments.json"
        self._write_manifest(manifest_path, generated_segments)

        return VoiceGenerationResult(
            output_dir=destination,
            manifest_path=manifest_path,
            segments=tuple(generated_segments.values()),
        )

    def _parse_translated_text(self, content: str) -> dict[str, str]:
        """Parse translated text into speaker segments.

        Args:
            content: Raw translated text content.

        Returns:
            Dictionary mapping speaker IDs to their text.
        """
        # Implementation would parse the translated text
        # This is a placeholder - actual implementation depends on format
        return {}

    def _synthesize_speaker_voice(
        self,
        speaker_id: str,
        text: str,
        output_dir: Path,
        reference_audio_path: str | Path | None = None,
    ) -> Path:
        """Synthesize voice for a speaker.

        Args:
            speaker_id: Speaker identifier.
            text: Text to synthesize.
            output_dir: Output directory for generated audio.
            reference_audio_path: Optional reference audio for voice cloning.

        Returns:
            Path to the generated audio file.

        Raises:
            VoiceSynthesisError: If synthesis fails.
        """
        # Implementation would use TTS synthesizer here
        output_path = output_dir / f"{speaker_id}.wav"

        # Check if output already exists
        if output_path.exists():
            return output_path

        # Synthesize audio (placeholder)
        # Actual implementation would use Piper or F5-TTS
        # Create an empty file as placeholder
        output_path.write_bytes(b"")
        return output_path

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds.
        """
        # Implementation would use soundfile or similar
        import soundfile as sf  # type: ignore[import-untyped]

        try:
            info = sf.info(str(audio_path))
            return float(info.duration)
        except (RuntimeError, OSError):
            return 0.0

    def _write_manifest(
        self,
        manifest_path: Path,
        segments: dict[str, GeneratedSpeechSegment],
    ) -> None:
        """Write manifest file with segment information.

        Args:
            manifest_path: Path to the manifest file.
            segments: Dictionary of generated segments.
        """
        # Implementation would write JSON manifest here
        import json

        data = {
            speaker_id: {
                "speaker_id": segment.speaker_id,
                "text": segment.text,
                "audio_path": str(segment.audio_path),
                "duration_seconds": segment.duration_seconds,
                "target_duration_seconds": segment.target_duration_seconds,
            }
            for speaker_id, segment in segments.items()
        }
        manifest_path.write_text(json.dumps(data, indent=2))
