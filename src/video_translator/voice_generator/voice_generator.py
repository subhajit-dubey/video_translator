# src/video_translator/voice_generator/diarization.py

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_MAX_PROFILE_DURATION_SECONDS = 20.0
DEFAULT_MIN_PROFILE_DURATION_SECONDS = 0.5
DEFAULT_PROFILE_PADDING_SECONDS = 0.15


class DiarizationError(Exception):
    """Base exception for speaker diarization errors."""


class PyannoteNotFoundError(DiarizationError):
    """Raised when pyannote.audio is unavailable."""


class HuggingFaceTokenError(DiarizationError):
    """Raised when Hugging Face authentication is unavailable."""


class InvalidAudioError(DiarizationError):
    """Raised when the input audio is invalid."""


class NoSpeakersDetectedError(DiarizationError):
    """Raised when no usable speaker turns are detected."""


@dataclass(frozen=True)
class SpeakerTurn:
    """A speaker's speech interval in the source audio timeline."""

    start: float
    end: float
    speaker_id: str


@dataclass(frozen=True)
class SpeakerProfile:
    """A speaker reference recording and its diarization intervals."""

    speaker_id: str
    reference_wav: Path
    turns: tuple[SpeakerTurn, ...]
    duration_seconds: float


class PyannoteSpeakerDiarizer:
    """Detect speakers and construct reference audio profiles."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DIARIZATION_MODEL,
        device: str = "cuda",
        auth_token: str | None = None,
        pipeline: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")

        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(
                "device must be 'cpu', 'cuda', or 'cuda:<index>'."
            )

        self.model_name = model_name
        self.device = device
        self.auth_token = auth_token
        self._pipeline = pipeline

    @staticmethod
    def _validate_audio(path: Path) -> None:
        if not path.exists():
            raise InvalidAudioError(
                f"Audio file does not exist: {path}"
            )

        if not path.is_file():
            raise InvalidAudioError(
                f"Audio path is not a file: {path}"
            )

        if path.stat().st_size == 0:
            raise InvalidAudioError(
                f"Audio file is empty: {path}"
            )

    def _load_pipeline(self) -> Any:
        """Load the diarization pipeline lazily."""
        if self._pipeline is not None:
            return self._pipeline

        token = self.auth_token or os.getenv("HF_TOKEN")

        if not token:
            raise HuggingFaceTokenError(
                "Set HF_TOKEN or provide auth_token. "
                "Accept the model conditions for "
                "pyannote/speaker-diarization-3.1 and "
                "pyannote/segmentation-3.0 on Hugging Face."
            )

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise PyannoteNotFoundError(
                "Install pyannote.audio to enable diarization."
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise DiarizationError(
                "CUDA was requested but is unavailable."
            )

        try:
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=token,
            )

            if pipeline is None:
                raise RuntimeError(
                    "The diarization pipeline could not be loaded."
                )

            pipeline.to(torch.device(self.device))
            self._pipeline = pipeline

            return pipeline

        except DiarizationError:
            raise

        except Exception as exc:
            raise DiarizationError(
                f"Unable to load diarization model: {self.model_name}"
            ) from exc

    def diarize(
        self,
        audio_path: str | Path,
    ) -> list[SpeakerTurn]:
        """Detect speaker turns and preserve source-audio timestamps."""
        path = Path(audio_path)
        self._validate_audio(path)

        pipeline = self._load_pipeline()

        try:
            annotation = pipeline(str(path))
            turns: list[SpeakerTurn] = []

            for segment, _, speaker_id in annotation.itertracks(
                yield_label=True
            ):
                start = float(segment.start)
                end = float(segment.end)

                if (
                    not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or end <= start
                ):
                    continue

                turns.append(
                    SpeakerTurn(
                        start=start,
                        end=end,
                        speaker_id=str(speaker_id),
                    )
                )

            turns.sort(key=lambda turn: (turn.start, turn.end))

            if not turns:
                raise NoSpeakersDetectedError(
                    f"No speaker turns detected in: {path}"
                )

            return turns

        except DiarizationError:
            raise

        except Exception as exc:
            raise DiarizationError(
                f"Speaker diarization failed for: {path}"
            ) from exc

    def extract_profiles(
        self,
        audio_path: str | Path,
        turns: Iterable[SpeakerTurn],
        output_dir: str | Path,
        *,
        max_duration_seconds: float = (
            DEFAULT_MAX_PROFILE_DURATION_SECONDS
        ),
        min_duration_seconds: float = (
            DEFAULT_MIN_PROFILE_DURATION_SECONDS
        ),
        padding_seconds: float = DEFAULT_PROFILE_PADDING_SECONDS,
    ) -> dict[str, SpeakerProfile]:
        """Create a reference WAV for each speaker from their source turns."""
        path = Path(audio_path)
        destination = Path(output_dir)

        self._validate_audio(path)

        if (
            not math.isfinite(max_duration_seconds)
            or max_duration_seconds <= 0
        ):
            raise ValueError(
                "max_duration_seconds must be positive and finite."
            )

        if (
            not math.isfinite(min_duration_seconds)
            or min_duration_seconds < 0
        ):
            raise ValueError(
                "min_duration_seconds must be non-negative and finite."
            )

        if (
            not math.isfinite(padding_seconds)
            or padding_seconds < 0
        ):
            raise ValueError(
                "padding_seconds must be non-negative and finite."
            )

        try:
            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise InvalidAudioError(
                f"Unable to read source audio: {path}"
            ) from exc

        if audio.shape[0] == 0 or not np.isfinite(audio).all():
            raise InvalidAudioError(
                f"Source audio has no usable samples: {path}"
            )

        mono = audio.mean(axis=1, dtype=np.float32)
        audio_duration = len(mono) / sample_rate
        grouped: dict[str, list[SpeakerTurn]] = {}

        for turn in turns:
            if turn.speaker_id:
                grouped.setdefault(turn.speaker_id, []).append(turn)

        if not grouped:
            raise NoSpeakersDetectedError(
                "No speaker turns are available for profile extraction."
            )

        destination.mkdir(parents=True, exist_ok=True)
        profiles: dict[str, SpeakerProfile] = {}

        max_frames = max(
            1,
            int(max_duration_seconds * sample_rate),
        )
        padding_frames = int(padding_seconds * sample_rate)
        min_frames = int(min_duration_seconds * sample_rate)

        for speaker_id, speaker_turns in grouped.items():
            chunks: list[np.ndarray] = []
            total_frames = 0

            speaker_turns.sort(
                key=lambda turn: (turn.start, turn.end)
            )

            for turn in speaker_turns:
                if total_frames >= max_frames:
                    break

                start_time = max(0.0, min(turn.start, audio_duration))
                end_time = max(0.0, min(turn.end, audio_duration))

                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)

                if end_frame <= start_frame:
                    continue

                available_frames = max_frames - total_frames
                clip = mono[start_frame:end_frame]
                clip = clip[:available_frames]

                if clip.size == 0:
                    continue

                chunks.append(clip)
                total_frames += len(clip)

                if (
                    padding_frames > 0
                    and total_frames < max_frames
                    and len(chunks) < len(speaker_turns)
                ):
                    silence = np.zeros(
                        min(
                            padding_frames,
                            max_frames - total_frames,
                        ),
                        dtype=np.float32,
                    )

                    chunks.append(silence)
                    total_frames += len(silence)

            if not chunks:
                continue

            profile_audio = np.concatenate(chunks)

            if len(profile_audio) < min_frames:
                continue

            safe_speaker_id = "".join(
                character
                if character.isalnum() or character in "-_"
                else "_"
                for character in speaker_id
            )

            profile_path = (
                destination / f"{safe_speaker_id}_reference.wav"
            )

            try:
                sf.write(
                    str(profile_path),
                    profile_audio,
                    sample_rate,
                    subtype="PCM_16",
                )
            except Exception as exc:
                raise DiarizationError(
                    f"Unable to save speaker profile: {speaker_id}"
                ) from exc

            profiles[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                reference_wav=profile_path,
                turns=tuple(speaker_turns),
                duration_seconds=len(profile_audio) / sample_rate,
            )

        if not profiles:
            raise NoSpeakersDetectedError(
                "No speaker profiles could be created from the audio."
            )

        return profiles