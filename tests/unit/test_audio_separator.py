import builtins
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from video_translator.audio_separator import (
    DEFAULT_MODEL,
    AudioSeparationResult,
    AudioSeparator,
    AudioSeparatorError,
    AudioSeparatorNotFoundError,
    InvalidAudioError,
    OutputAlreadyExistsError,
)


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    sf.write(
        str(path),
        np.zeros((4410, 2), dtype=np.float32),
        44100,
        subtype="PCM_16",
    )
    return path


@pytest.fixture
def fake_engine(tmp_path: Path):
    class FakeEngine:
        def __init__(self) -> None:
            self.output_dir = tmp_path
            self.model_filename = None
            self.load_model = Mock(side_effect=self._load_model)
            self.separate = Mock(side_effect=self._separate)

        def _load_model(self, model_filename: str) -> None:
            self.model_filename = model_filename

        def _separate(
            self,
            input_path: str,
            output_names: dict[str, str],
        ) -> list[str]:
            outputs = []

            for index, name in enumerate(output_names.values(), start=1):
                output_path = self.output_dir / f"{name}.wav"

                audio = np.full(
                    (4410, 2),
                    0.05 * index,
                    dtype=np.float32,
                )

                sf.write(
                    str(output_path),
                    audio,
                    44100,
                    subtype="PCM_16",
                )

                outputs.append(str(output_path))

            return outputs

    return FakeEngine()


@pytest.fixture
def separator(fake_engine, monkeypatch):
    instance = AudioSeparator()

    def create_engine(output_dir: Path):
        fake_engine.output_dir = output_dir
        return fake_engine

    monkeypatch.setattr(instance, "_create_engine", create_engine)

    return instance


def test_default_model():
    assert DEFAULT_MODEL == "htdemucs.yaml"
    assert AudioSeparator()._model == DEFAULT_MODEL


def test_custom_model():
    separator = AudioSeparator(model="custom_model.ckpt")

    assert separator._model == "custom_model.ckpt"


def test_empty_model_is_rejected():
    with pytest.raises(ValueError, match="model"):
        AudioSeparator(model="")


def test_demucs_model_detection():
    assert AudioSeparator._is_demucs_model("htdemucs.yaml")
    assert AudioSeparator._is_demucs_model("htdemucs_ft.yaml")
    assert AudioSeparator._is_demucs_model("htdemucs_6s.yaml")
    assert not AudioSeparator._is_demucs_model("UVR_MDXNET_KARA_2.onnx")
    assert not AudioSeparator._is_demucs_model(
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    )


def test_demucs_stem_selection():
    assert AudioSeparator._get_demucs_stems("htdemucs.yaml") == (
        "Vocals",
        "Drums",
        "Bass",
        "Other",
    )

    assert AudioSeparator._get_demucs_stems("htdemucs_6s.yaml") == (
        "Vocals",
        "Drums",
        "Bass",
        "Other",
        "Guitar",
        "Piano",
    )


def test_demucs_separation_returns_vocals_and_instrumental(
    separator,
    fake_engine,
    audio_file,
    tmp_path,
):
    output_dir = tmp_path / "separated"

    result = separator.separate(audio_file, output_dir)

    assert isinstance(result, AudioSeparationResult)
    assert result.input_path == audio_file.resolve()
    assert result.model == DEFAULT_MODEL
    assert result.vocals_path.is_file()
    assert result.instrumental_path.is_file()

    assert result.vocals_path.name == "sample_vocals.wav"
    assert result.instrumental_path.name == "sample_instrumental.wav"

    fake_engine.load_model.assert_called_once_with(
        model_filename="htdemucs.yaml"
    )

    fake_engine.separate.assert_called_once_with(
        str(audio_file.resolve()),
        {
            "Vocals": "sample_vocals",
            "Drums": "sample_drums",
            "Bass": "sample_bass",
            "Other": "sample_other",
        },
    )

    assert not (output_dir / "sample_drums.wav").exists()
    assert not (output_dir / "sample_bass.wav").exists()
    assert not (output_dir / "sample_other.wav").exists()


def test_roformer_separation_uses_instrumental_output(
    separator,
    fake_engine,
    audio_file,
    tmp_path,
):
    result = separator.separate(
        audio_file,
        tmp_path / "separated",
        model="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    )

    assert result.vocals_path.is_file()
    assert result.instrumental_path.is_file()

    fake_engine.separate.assert_called_once_with(
        str(audio_file.resolve()),
        {
            "Vocals": "sample_vocals",
            "Instrumental": "sample_instrumental",
        },
    )


def test_missing_input_raises_error(separator, tmp_path):
    with pytest.raises(InvalidAudioError, match="does not exist"):
        separator.separate(
            tmp_path / "missing.wav",
            tmp_path / "separated",
        )


def test_directory_input_raises_error(separator, tmp_path):
    with pytest.raises(InvalidAudioError, match="not a file"):
        separator.separate(
            tmp_path,
            tmp_path / "separated",
        )


def test_empty_input_raises_error(separator, tmp_path):
    empty_audio = tmp_path / "empty.wav"
    empty_audio.touch()

    with pytest.raises(InvalidAudioError, match="empty"):
        separator.separate(
            empty_audio,
            tmp_path / "separated",
        )


def test_non_directory_output_raises_error(
    separator,
    audio_file,
    tmp_path,
):
    output = tmp_path / "output"
    output.write_text("not a directory")

    with pytest.raises(InvalidAudioError, match="not a directory"):
        separator.separate(audio_file, output)


def test_existing_output_raises_error(
    separator,
    audio_file,
    tmp_path,
):
    output_dir = tmp_path / "separated"
    output_dir.mkdir()

    existing_file = output_dir / "sample_vocals.wav"
    existing_file.write_bytes(b"existing output")

    with pytest.raises(OutputAlreadyExistsError):
        separator.separate(audio_file, output_dir)

    assert existing_file.read_bytes() == b"existing output"


def test_engine_failure_cleans_partial_outputs(
    separator,
    fake_engine,
    audio_file,
    tmp_path,
):
    output_dir = tmp_path / "separated"

    def failing_separation(input_path, output_names):
        partial_output = output_dir / "sample_vocals.wav"

        sf.write(
            str(partial_output),
            np.zeros((100, 2), dtype=np.float32),
            44100,
        )

        raise RuntimeError("Simulated separation failure")

    fake_engine.separate.side_effect = failing_separation

    with pytest.raises(AudioSeparatorError, match="separation failed"):
        separator.separate(audio_file, output_dir)

    assert not (output_dir / "sample_vocals.wav").exists()
    assert not (output_dir / "sample_instrumental.wav").exists()


def test_missing_audio_separator_package(
    tmp_path,
    monkeypatch,
):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "audio_separator.separator":
            raise ModuleNotFoundError(name=name)

        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(AudioSeparatorNotFoundError):
        AudioSeparator._create_engine(tmp_path)


def test_demucs_combines_non_vocal_stems(
    separator,
    fake_engine,
    audio_file,
    tmp_path,
):
    output_dir = tmp_path / "separated"

    result = separator.separate(audio_file, output_dir)

    vocals, vocals_rate = sf.read(
        str(result.vocals_path),
        dtype="float32",
        always_2d=True,
    )

    instrumental, instrumental_rate = sf.read(
        str(result.instrumental_path),
        dtype="float32",
        always_2d=True,
    )

    assert vocals_rate == 44100
    assert instrumental_rate == 44100
    assert vocals.shape == instrumental.shape
    assert vocals.shape[0] > 0
    assert np.any(vocals != 0)
    assert np.any(instrumental != 0)
    assert np.isfinite(vocals).all()
    assert np.isfinite(instrumental).all()