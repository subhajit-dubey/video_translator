from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from audio_separator.separator import Separator  # type: ignore[import-untyped]

# Constants for audio processing
MAX_PEAK_THRESHOLD = 0.99
DEFAULT_MODEL = "htdemucs.yaml"

DEMUCS_STEMS = ("Vocals", "Drums", "Bass", "Other")
DEMUCS_6S_STEMS = ("Vocals", "Drums", "Bass", "Other", "Guitar", "Piano")


class AudioSeparatorError(Exception):
    """Base exception for audio-separation errors."""


class AudioSeparatorNotFoundError(AudioSeparatorError):
    """Raised when the audio-separator package is unavailable."""


class InvalidAudioError(AudioSeparatorError):
    """Raised when the input audio is missing or invalid."""


class OutputAlreadyExistsError(AudioSeparatorError):
    """Raised when a separation output already exists."""


@dataclass(frozen=True)
class AudioSeparationResult:
    """Paths to the separated audio stems."""

    input_path: Path
    vocals_path: Path
    instrumental_path: Path
    model: str


class AudioSeparator:
    """Separate vocals from accompaniment using audio-separator."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not model or not model.strip():
            msg = "model must not be empty"
            raise ValueError(msg)

        self._model = model

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

    @staticmethod
    def _is_demucs_model(model: str) -> bool:
        return "demucs" in Path(model).name.lower()

    @staticmethod
    def _get_demucs_stems(model: str) -> tuple[str, ...]:
        if "6s" in Path(model).stem.lower():
            return DEMUCS_6S_STEMS

        return DEMUCS_STEMS

    @staticmethod
    def _create_engine(output_dir: Path) -> Separator:
        from audio_separator.separator import Separator

        try:
            return Separator(
                output_dir=str(output_dir),
                output_format="WAV",
                normalization_threshold=1.0,
                amplification_threshold=0.0,
                use_soundfile=True,
            )
        except ImportError as exc:
            msg = (
                "audio-separator is not installed. "
                "Install it using: poetry add audio-separator"
            )
            raise AudioSeparatorNotFoundError(msg) from exc
        except Exception as exc:
            msg = "Unable to initialize audio-separator."
            raise AudioSeparatorError(msg) from exc

    @staticmethod
    def _remove_partial_outputs(paths: tuple[Path, ...]) -> None:
        for path in paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    @staticmethod
    def _validate_outputs(paths: tuple[Path, ...]) -> None:
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                msg = f"Audio separator did not produce a valid stem: {path}"
                raise AudioSeparatorError(msg)

    @staticmethod
    def _validate_vocals_stem(vocals: np.ndarray) -> None:
        if vocals.shape[0] == 0:
            raise AudioSeparatorError("The vocals stem is empty.")
        if not np.isfinite(vocals).all():
            raise AudioSeparatorError("The vocals stem contains invalid samples.")

    @staticmethod
    def _validate_stem_compatibility(
        stem_name: str,
        audio: np.ndarray,
        stem_sample_rate: int,
        vocals: np.ndarray,
        sample_rate: int,
    ) -> None:
        if stem_sample_rate != sample_rate:
            raise AudioSeparatorError(f"Sample-rate mismatch in the {stem_name} stem.")
        if audio.shape != vocals.shape:
            raise AudioSeparatorError(f"Audio shape mismatch in the {stem_name} stem.")

    @staticmethod
    def _validate_instrumental_stem(instrumental: np.ndarray) -> None:
        if not np.isfinite(instrumental).all():
            raise AudioSeparatorError("The instrumental stem contains invalid samples.")

    @staticmethod
    def _apply_peak_normalization(
        vocals: np.ndarray,
        instrumental: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        peak = max(
            float(np.max(np.abs(vocals))),
            float(np.max(np.abs(instrumental))),
        )

        if peak > MAX_PEAK_THRESHOLD:
            scale = MAX_PEAK_THRESHOLD / peak
            return vocals * scale, instrumental * scale
        return vocals, instrumental

    @staticmethod
    def _combine_demucs_stems(
        stem_paths: dict[str, Path],
        vocals_path: Path,
        instrumental_path: Path,
    ) -> None:
        vocals, sample_rate = sf.read(
            str(vocals_path),
            dtype="float32",
            always_2d=True,
        )

        AudioSeparator._validate_vocals_stem(vocals)

        instrumental = np.zeros_like(vocals)

        for stem_name, stem_path in stem_paths.items():
            if stem_name == "Vocals":
                continue

            audio, stem_sample_rate = sf.read(
                str(stem_path),
                dtype="float32",
                always_2d=True,
            )

            AudioSeparator._validate_stem_compatibility(
                stem_name,
                audio,
                stem_sample_rate,
                vocals,
                sample_rate,
            )

            instrumental += audio

        AudioSeparator._validate_instrumental_stem(instrumental)

        vocals, instrumental = AudioSeparator._apply_peak_normalization(
            vocals,
            instrumental,
        )

        sf.write(
            str(vocals_path),
            vocals,
            sample_rate,
            subtype="PCM_16",
        )

        sf.write(
            str(instrumental_path),
            instrumental,
            sample_rate,
            subtype="PCM_16",
        )

    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        model: str | None = None,
    ) -> AudioSeparationResult:
        """Separate vocals and instrumental audio."""

        source = Path(input_path).resolve()
        destination = Path(output_dir).resolve()
        selected_model = model or self._model

        if not selected_model or not selected_model.strip():
            msg = "model must not be empty"
            raise ValueError(msg)

        self._validate_input(source)

        if destination.exists() and not destination.is_dir():
            msg = f"Output path is not a directory: {destination}"
            raise InvalidAudioError(msg)

        is_demucs = self._is_demucs_model(selected_model)

        vocals_path = destination / f"{source.stem}_vocals.wav"
        instrumental_path = destination / f"{source.stem}_instrumental.wav"

        if is_demucs:
            stem_names = self._get_demucs_stems(selected_model)

            stem_paths = {
                name: destination / f"{source.stem}_{name.lower()}.wav"
                for name in stem_names
            }

            output_names = {
                name: f"{source.stem}_{name.lower()}" for name in stem_names
            }

            generated_paths = (*tuple(stem_paths.values()), instrumental_path)

        else:
            stem_paths = {}

            output_names = {
                "Vocals": f"{source.stem}_vocals",
                "Instrumental": f"{source.stem}_instrumental",
            }

            generated_paths = (
                vocals_path,
                instrumental_path,
            )

        if any(path.exists() for path in generated_paths):
            msg = f"Separation output already exists in: {destination}"
            raise OutputAlreadyExistsError(msg)

        destination.mkdir(parents=True, exist_ok=True)

        try:
            engine = self._create_engine(destination)
            engine.load_model(model_filename=selected_model)

            engine.separate(
                str(source),
                output_names,
            )

            if is_demucs:
                self._validate_outputs(tuple(stem_paths.values()))

                self._combine_demucs_stems(
                    stem_paths=stem_paths,
                    vocals_path=vocals_path,
                    instrumental_path=instrumental_path,
                )

                intermediate_paths = tuple(
                    path for name, path in stem_paths.items() if name != "Vocals"
                )

                self._remove_partial_outputs(intermediate_paths)

            self._validate_outputs((vocals_path, instrumental_path))

        except AudioSeparatorError:
            self._remove_partial_outputs(generated_paths)
            raise

        except Exception as exc:
            self._remove_partial_outputs(generated_paths)
            msg = f"Audio separation failed for: {source}"
            raise AudioSeparatorError(msg) from exc

        return AudioSeparationResult(
            input_path=source,
            vocals_path=vocals_path,
            instrumental_path=instrumental_path,
            model=selected_model,
        )
