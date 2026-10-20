from __future__ import annotations

import ctypes
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import soundfile as sf

from video_translator.voice_generator.diarization import (
    InvalidAudioError,
    PyannoteSpeakerDiarizer,
    SpeakerProfile,
    SpeakerTurn,
)


DEFAULT_TTS_MODEL = "F5TTS_v1_Base"
DEFAULT_HINDI_TTS_MODEL = "F5-Hindi-Small"
_HINDI_TTS_CONFIG_MODEL = "F5TTS_Small"
_HINDI_TEXT_NORMALIZATION_LOCK = RLock()
DEFAULT_TTS_DEVICE = "cuda"
DEFAULT_NUM_SPEAKER_PROFILE_SECONDS = 10.0
DEFAULT_TTS_CKPT_FILE = ""
DEFAULT_TTS_VOCAB_FILE = ""


class VoiceGeneratorError(Exception):
    """Base exception for voice-generation errors."""


class F5TTSNotFoundError(VoiceGeneratorError):
    """Raised when F5-TTS is unavailable."""


class F5TTSModelLoadError(VoiceGeneratorError):
    """Raised when the F5-TTS model cannot be loaded."""


class UnsupportedTTSConfigurationError(VoiceGeneratorError):
    """Raised when the F5-TTS configuration is invalid."""


class InvalidTranslationFileError(VoiceGeneratorError):
    """Raised when the translated JSON file is invalid."""


class VoiceSynthesisError(VoiceGeneratorError):
    """Raised when speech synthesis fails."""


def _preload_cuda_nvrtc(torch_module: Any) -> None:
    """Make the bundled CUDA NVRTC library visible to TorchCodec."""
    if sys.platform != "linux" or not torch_module.version.cuda:
        return

    cuda_major = torch_module.version.cuda.split(".", maxsplit=1)[0]
    site_packages = Path(torch_module.__file__).resolve().parent.parent
    nvrtc_path = (
        site_packages
        / "nvidia"
        / "cuda_nvrtc"
        / "lib"
        / f"libnvrtc.so.{cuda_major}"
    )

    if not nvrtc_path.is_file():
        raise VoiceGeneratorError(
            f"CUDA NVRTC runtime library for CUDA {cuda_major} "
            f"was not found: {nvrtc_path}"
        )

    try:
        ctypes.CDLL(str(nvrtc_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise VoiceGeneratorError(
            f"Unable to load CUDA NVRTC runtime library: {nvrtc_path}"
        ) from exc


def _preserve_hindi_text(text_list: list[str]) -> list[str]:
    return text_list


@contextmanager
def _hindi_text_normalization():
    """Disable F5-TTS's Chinese-only Pinyin conversion for Hindi text."""
    from f5_tts.infer import utils_infer

    with _HINDI_TEXT_NORMALIZATION_LOCK:
        original_converter = utils_infer.convert_char_to_pinyin
        utils_infer.convert_char_to_pinyin = _preserve_hindi_text
        try:
            yield
        finally:
            utils_infer.convert_char_to_pinyin = original_converter


@dataclass(frozen=True)
class TranslatedSegment:
    """A translated segment with original-timeline timestamps."""

    start: float
    end: float
    text: str
    speaker_id: str | None = None


@dataclass(frozen=True)
class GeneratedSpeechSegment:
    """Synthesized speech and its intended original-timeline interval."""

    start: float
    end: float
    text: str
    speaker_id: str
    audio_path: Path
    duration_seconds: float

    @property
    def target_duration_seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VoiceGenerationResult:
    """Output paths and generated speech segments."""

    output_dir: Path
    manifest_path: Path
    source_language: str
    target_language: str
    speaker_profiles: dict[str, SpeakerProfile]
    segments: tuple[GeneratedSpeechSegment, ...]


class F5TTSSynthesizer:
    """Synthesize speech using F5-TTS and a speaker reference recording."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_TTS_MODEL,
        device: str = DEFAULT_TTS_DEVICE,
        ckpt_file: str = DEFAULT_TTS_CKPT_FILE,
        vocab_file: str = DEFAULT_TTS_VOCAB_FILE,
        model_instance: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")

        if model_name == DEFAULT_HINDI_TTS_MODEL and (
            not ckpt_file.strip() or not vocab_file.strip()
        ):
            raise ValueError(
                f"{DEFAULT_HINDI_TTS_MODEL} requires both ckpt_file "
                "and vocab_file."
            )

        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(
                "device must be 'cpu', 'cuda', or 'cuda:<index>'."
            )

        self.model_name = model_name
        self.device = device
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self._tts = model_instance

    def _load_model(self) -> Any:
        """Load and cache the F5-TTS model."""
        if self._tts is not None:
            return self._tts

        try:
            import torch
        except ImportError as exc:
            raise F5TTSNotFoundError(
                "F5-TTS is not installed. "
                "Install it using: poetry add f5-tts"
            ) from exc

        _preload_cuda_nvrtc(torch)

        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise F5TTSNotFoundError(
                "F5-TTS is not installed. "
                "Install it using: poetry add f5-tts"
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise VoiceGeneratorError(
                "CUDA was requested but is unavailable."
            )

        try:
            model_config = (
                _HINDI_TTS_CONFIG_MODEL
                if self.model_name == DEFAULT_HINDI_TTS_MODEL
                else self.model_name
            )
            self._tts = F5TTS(
                model=model_config,
                ckpt_file=self.ckpt_file,
                vocab_file=self.vocab_file,
                device=self.device,
            )

        except Exception as exc:
            raise F5TTSModelLoadError(
                f"Unable to load F5-TTS model configuration: "
                f"{self.model_name}"
            ) from exc

        return self._tts

    def synthesize(
        self,
        text: str,
        speaker_wav: str | Path,
        output_path: str | Path,
        *,
        reference_text: str = "",
    ) -> Path:
        """Generate a WAV file conditioned on the speaker reference.

        If reference_text is empty, F5-TTS automatically transcribes
        the reference recording. Providing its transcript is preferred.
        """
        # [Existing validation] Reject empty generation text.
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be empty.")

        # [Regression fix] Check missing audio before checking its transcript.
        if speaker_wav is None or (
            isinstance(speaker_wav, str) and not speaker_wav.strip()
        ):
            raise VoiceSynthesisError(
                "Speaker reference audio is required for F5-TTS synthesis."
            )

        reference = Path(speaker_wav)

        # [Regression fix] A nonexistent or empty audio file must produce
        # an audio-specific error even when reference_text is also empty.
        if not reference.is_file() or reference.stat().st_size == 0:
            raise VoiceSynthesisError(
                f"Speaker reference audio is missing or empty: {reference}"
            )

        # [Existing validation] Preserve the reference transcript type check.
        if not isinstance(reference_text, str):
            raise TypeError("reference_text must be a string.")

        # [Existing validation] Reject an empty transcript after audio validation.
        if not reference_text.strip():
            raise VoiceSynthesisError(
                "reference_text cannot be empty. Provide the original-language "
                "transcript or let VoiceGenerator transcribe the speaker profile "
                "using Faster-Whisper."
            )

        destination = Path(output_path)

        destination.parent.mkdir(parents=True, exist_ok=True)

        tts = self._load_model()

        try:
            normalization = (
                _hindi_text_normalization()
                if self.model_name == DEFAULT_HINDI_TTS_MODEL
                else nullcontext()
            )
            with normalization:
                tts.infer(
                    ref_file=str(reference),
                    ref_text=reference_text.strip(),
                    gen_text=text.strip(),
                    file_wave=str(destination),
                    seed=None,
                )

        except Exception as exc:
            destination.unlink(missing_ok=True)

            raise VoiceSynthesisError(
                f"F5-TTS synthesis failed for: {text[:80]!r}"
            ) from exc

        if (
            not destination.is_file()
            or destination.stat().st_size == 0
        ):
            destination.unlink(missing_ok=True)

            raise VoiceSynthesisError(
                f"F5-TTS did not generate a valid audio file: "
                f"{destination}"
            )

        try:
            info = sf.info(str(destination))

            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError("Invalid generated audio metadata.")

        except Exception as exc:
            destination.unlink(missing_ok=True)

            raise VoiceSynthesisError(
                f"F5-TTS generated an invalid WAV file: {destination}"
            ) from exc

        return destination


class VoiceGenerator:
    """Generate speaker-conditioned translated speech.

    Timeline alignment, duration adjustment, and EQ matching are
    intentionally delegated to the Audio Aligner component.
    """

    def __init__(
        self,
        *,
        diarizer: PyannoteSpeakerDiarizer | None = None,
        synthesizer: F5TTSSynthesizer | None = None,
        diarization_device: str = "cuda",
        tts_device: str = "cuda",
        auth_token: str | None = None,
        tts_model: str = DEFAULT_TTS_MODEL,
        tts_ckpt_file: str = DEFAULT_TTS_CKPT_FILE,
        tts_vocab_file: str = DEFAULT_TTS_VOCAB_FILE,
        max_profile_duration_seconds: float = (
            DEFAULT_NUM_SPEAKER_PROFILE_SECONDS
        ),
        reference_transcriber: Any | None = None,
    ) -> None:
        if (
            not math.isfinite(max_profile_duration_seconds)
            or max_profile_duration_seconds <= 0
        ):
            raise ValueError(
                "max_profile_duration_seconds must be positive and finite."
            )

        self.diarizer = diarizer or PyannoteSpeakerDiarizer(
            device=diarization_device,
            auth_token=auth_token,
        )

        self.synthesizer = synthesizer or F5TTSSynthesizer(
            model_name=tts_model,
            device=tts_device,
            ckpt_file=tts_ckpt_file,
            vocab_file=tts_vocab_file,
        )

        self.max_profile_duration_seconds = (
            max_profile_duration_seconds
        )

        self.reference_transcriber = reference_transcriber

    @staticmethod
    def _load_translation_file(
        path: Path,
    ) -> tuple[str, str, list[TranslatedSegment]]:
        """Load and validate the TextTranslator JSON output."""
        if not path.is_file():
            raise InvalidTranslationFileError(
                f"Translation JSON file does not exist: {path}"
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidTranslationFileError(
                f"Unable to read translation JSON: {path}"
            ) from exc

        if not isinstance(data, dict):
            raise InvalidTranslationFileError(
                "Translation JSON must contain an object."
            )

        source_language = data.get("source_language")
        target_language = data.get("target_language")
        raw_segments = data.get("segments")

        if (
            not isinstance(source_language, str)
            or not source_language.strip()
        ):
            raise InvalidTranslationFileError(
                "source_language must be a non-empty string."
            )

        if (
            not isinstance(target_language, str)
            or not target_language.strip()
        ):
            raise InvalidTranslationFileError(
                "target_language must be a non-empty string."
            )

        if not isinstance(raw_segments, list):
            raise InvalidTranslationFileError(
                "segments must be a list."
            )

        segments: list[TranslatedSegment] = []

        for index, item in enumerate(raw_segments):
            if not isinstance(item, dict):
                raise InvalidTranslationFileError(
                    f"Segment {index} must be a JSON object."
                )

            try:
                start = float(item["start"])
                end = float(item["end"])
                text = item["text"]
                speaker_id = item.get("speaker_id")
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidTranslationFileError(
                    f"Segment {index} has invalid fields."
                ) from exc

            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise InvalidTranslationFileError(
                    f"Segment {index} has invalid timestamps."
                )

            if not isinstance(text, str):
                raise InvalidTranslationFileError(
                    f"Segment {index} text must be a string."
                )

            if (
                speaker_id is not None
                and not isinstance(speaker_id, str)
            ):
                raise InvalidTranslationFileError(
                    f"Segment {index} speaker_id must be a string or null."
                )

            if text.strip():
                segments.append(
                    TranslatedSegment(
                        start=start,
                        end=end,
                        text=text.strip(),
                        speaker_id=speaker_id,
                    )
                )

        segments.sort(key=lambda segment: (segment.start, segment.end))

        return source_language, target_language, segments

    def _get_reference_text(
        self,
        speaker_id: str,
        profile: SpeakerProfile,
        reference_texts: dict[str, str] | None,
    ) -> str:
        """Obtain the original-language transcript for a speaker profile."""

        provided_text = (
            reference_texts.get(speaker_id, "").strip()
            if reference_texts
            else ""
        )

        if provided_text:
            return provided_text

        # [Regression fix] Transcribe the matching speaker's reference audio
        # when the caller has not supplied an original-language transcript.
        if self.reference_transcriber is None:
            return ""

        transcription = self.reference_transcriber.transcribe(
            profile.reference_wav
        )

        # Support transcribers that return either text or an object with .text.
        transcribed_text = (
            transcription
            if isinstance(transcription, str)
            else getattr(transcription, "text", "")
        )

        return (
            transcribed_text.strip()
            if isinstance(transcribed_text, str)
            else ""
        )

    @staticmethod
    def _select_speaker(
        segment: TranslatedSegment,
        turns: list[SpeakerTurn],
        profiles: dict[str, SpeakerProfile],
    ) -> str:
        """Assign a segment to the speaker with greatest time overlap."""
        if segment.speaker_id is not None:
            if segment.speaker_id not in profiles:
                raise VoiceGeneratorError(
                    f"No reference profile exists for speaker "
                    f"{segment.speaker_id!r}."
                )

            return segment.speaker_id

        overlaps: dict[str, float] = {}

        for turn in turns:
            overlap = max(
                0.0,
                min(segment.end, turn.end)
                - max(segment.start, turn.start),
            )

            if overlap > 0:
                overlaps[turn.speaker_id] = (
                    overlaps.get(turn.speaker_id, 0.0) + overlap
                )

        if overlaps:
            selected = max(overlaps, key=overlaps.__getitem__)

            if selected in profiles:
                return selected

        if not turns:
            raise VoiceGeneratorError(
                "No diarization turns are available for speaker assignment."
            )

        midpoint = (segment.start + segment.end) / 2.0

        def distance(turn: SpeakerTurn) -> float:
            if turn.start <= midpoint <= turn.end:
                return 0.0

            return min(
                abs(midpoint - turn.start),
                abs(midpoint - turn.end),
            )

        nearest = min(turns, key=distance)

        if nearest.speaker_id not in profiles:
            raise VoiceGeneratorError(
                f"No reference profile exists for speaker "
                f"{nearest.speaker_id!r}."
            )

        return nearest.speaker_id

    def generate(
        self,
        translation_json_path: str | Path,
        source_audio_path: str | Path,
        output_dir: str | Path,
        *,
        reference_texts: dict[str, str] | None = None,
    ) -> VoiceGenerationResult:
        """Create speaker profiles and synthesize each translated segment.

        Args:
            translation_json_path: TextTranslator JSON output.
            source_audio_path: Original source-language audio.
            output_dir: Directory in which results are saved.
            reference_texts: Optional source-language transcript per speaker,
                keyed by diarization speaker ID. Missing entries are
                automatically transcribed by F5-TTS.
        """
        translation_path = Path(translation_json_path)
        source_audio = Path(source_audio_path)
        destination = Path(output_dir)

        if (
            not source_audio.is_file()
            or source_audio.stat().st_size == 0
        ):
            raise InvalidAudioError(
                f"Source audio file does not exist or is empty: "
                f"{source_audio}"
            )

        if reference_texts is not None:
            if not isinstance(reference_texts, dict):
                raise TypeError("reference_texts must be a dictionary.")

            if any(
                not isinstance(key, str)
                or not isinstance(value, str)
                for key, value in reference_texts.items()
            ):
                raise TypeError(
                    "reference_texts must map speaker IDs to strings."
                )

        (
            source_language,
            target_language,
            segments,
        ) = self._load_translation_file(translation_path)

        destination.mkdir(parents=True, exist_ok=True)

        profiles_dir = destination / "speaker_profiles"
        segments_dir = destination / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        turns = self.diarizer.diarize(source_audio)

        profiles = self.diarizer.extract_profiles(
            source_audio,
            turns,
            profiles_dir,
            max_duration_seconds=self.max_profile_duration_seconds,
        )

        if not profiles:
            raise VoiceGeneratorError(
                "No usable speaker profiles were created."
            )

        generated_segments: list[GeneratedSpeechSegment] = []

        for index, segment in enumerate(segments, start=1):
            speaker_id = self._select_speaker(
                segment,
                turns,
                profiles,
            )

            profile = profiles[speaker_id]
            audio_path = segments_dir / f"segment_{index:04d}.wav"

            reference_text = self._get_reference_text(
                speaker_id=speaker_id,
                profile=profile,
                reference_texts=reference_texts,
            )

            reference_text = (
                reference_text.strip()
                if isinstance(reference_text, str)
                else ""
            )

            self.synthesizer.synthesize(
                text=segment.text,
                speaker_wav=profile.reference_wav,
                output_path=audio_path,
                reference_text=reference_text,
            )

            try:
                audio_info = sf.info(str(audio_path))
                actual_duration = float(audio_info.duration)
            except Exception as exc:
                audio_path.unlink(missing_ok=True)

                raise VoiceGeneratorError(
                    f"Unable to inspect generated audio: {audio_path}"
                ) from exc

            if (
                not math.isfinite(actual_duration)
                or actual_duration <= 0
            ):
                audio_path.unlink(missing_ok=True)

                raise VoiceGeneratorError(
                    f"Generated audio has an invalid duration: "
                    f"{audio_path}"
                )

            generated_segments.append(
                GeneratedSpeechSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker_id=speaker_id,
                    audio_path=audio_path,
                    duration_seconds=actual_duration,
                )
            )

        manifest_path = destination / "voice_manifest.json"

        manifest = {
            "source_language": source_language,
            "target_language": target_language,
            "tts_model": self.synthesizer.model_name,
            "tts_device": self.synthesizer.device,
            "tts_ckpt_file": self.synthesizer.ckpt_file,
            "tts_vocab_file": self.synthesizer.vocab_file,
            "alignment_pending": True,
            "speakers": [
                {
                    "speaker_id": speaker_id,
                    "reference_wav": str(profile.reference_wav),
                    "profile_duration_seconds": profile.duration_seconds,
                    "reference_text_provided": bool(
                        reference_texts
                        and reference_texts.get(speaker_id, "").strip()
                    ),
                    "turns": [
                        {
                            "start": turn.start,
                            "end": turn.end,
                        }
                        for turn in profile.turns
                    ],
                }
                for speaker_id, profile in profiles.items()
            ],
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "target_duration_seconds": (
                        segment.target_duration_seconds
                    ),
                    "generated_duration_seconds": segment.duration_seconds,
                    "speaker_id": segment.speaker_id,
                    "text": segment.text,
                    "audio_path": str(segment.audio_path),
                }
                for segment in generated_segments
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
            raise VoiceGeneratorError(
                f"Unable to write voice-generation manifest: "
                f"{manifest_path}"
            ) from exc

        return VoiceGenerationResult(
            output_dir=destination,
            manifest_path=manifest_path,
            source_language=source_language,
            target_language=target_language,
            speaker_profiles=profiles,
            segments=tuple(generated_segments),
        )