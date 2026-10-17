import builtins
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from video_translator.speech_detector import (
    InvalidAudioError,
    SileroVADNotFoundError,
    SpeechDetector,
    SpeechDetectorError,
    SpeechRegion,
)


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"mock audio content")
    return path


@pytest.fixture
def detector(monkeypatch) -> SpeechDetector:
    instance = SpeechDetector()

    monkeypatch.setattr(
        instance,
        "_prepare_audio",
        Mock(return_value=np.zeros(32_000, dtype=np.float32)),
    )

    monkeypatch.setattr(
        instance,
        "_run_inference",
        Mock(
            return_value=[
                {"start": 0.25, "end": 0.75},
                {"start": 1.0, "end": 1.5},
            ]
        ),
    )

    return instance


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"threshold": 1},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"sample_rate": 44100},
        {"min_speech_duration_ms": -1},
        {"min_silence_duration_ms": -1},
        {"speech_pad_ms": -1},
    ],
)
def test_invalid_configuration_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        SpeechDetector(**kwargs)


def test_detect_returns_speech_regions(
    detector,
    audio_file,
):
    regions = detector.detect(audio_file)

    assert regions == [
        SpeechRegion(start=0.25, end=0.75),
        SpeechRegion(start=1.0, end=1.5),
    ]


def test_detect_passes_prepared_audio_to_inference(
    detector,
    audio_file,
):
    detector.detect(audio_file)

    detector._run_inference.assert_called_once()

    audio = detector._run_inference.call_args.args[0]

    assert audio.dtype == np.float32
    assert len(audio) == 32_000


def test_detect_missing_audio_raises_error(
    detector,
    tmp_path,
):
    with pytest.raises(InvalidAudioError, match="does not exist"):
        detector.detect(tmp_path / "missing.wav")


def test_detect_directory_raises_error(
    detector,
    tmp_path,
):
    with pytest.raises(InvalidAudioError, match="not a file"):
        detector.detect(tmp_path)


def test_detect_empty_file_raises_error(
    detector,
    tmp_path,
):
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()

    with pytest.raises(InvalidAudioError, match="empty"):
        detector.detect(empty_file)


def test_detect_returns_empty_list_when_no_speech(
    detector,
    audio_file,
):
    detector._run_inference.return_value = []

    assert detector.detect(audio_file) == []


def test_detect_resamples_audio(
    audio_file,
    monkeypatch,
):
    instance = SpeechDetector(sample_rate=16000)

    monkeypatch.setattr(
        instance,
        "_read_audio",
        Mock(
            return_value=(
                np.zeros(8_000, dtype=np.float32),
                8_000,
            )
        ),
    )

    captured = {}

    def fake_inference(audio):
        captured["audio"] = audio
        return [{"start": 0.1, "end": 0.5}]

    monkeypatch.setattr(
        instance,
        "_run_inference",
        fake_inference,
    )

    regions = instance.detect(audio_file)

    assert len(captured["audio"]) == 16_000
    assert captured["audio"].dtype == np.float32
    assert regions == [SpeechRegion(start=0.1, end=0.5)]


def test_detect_propagates_inference_error(
    detector,
    audio_file,
):
    detector._run_inference.side_effect = SpeechDetectorError(
        "Inference failed"
    )

    with pytest.raises(SpeechDetectorError, match="Inference failed"):
        detector.detect(audio_file)


@pytest.mark.parametrize(
    "timestamps",
    [
        [{"start": -1.0, "end": 0.5}],
        [{"start": 0.8, "end": 0.5}],
        [{"start": float("nan"), "end": 1.0}],
        [{"start": 0.1}],
        ["invalid"],
    ],
)
def test_invalid_timestamps_raise_error(timestamps):
    with pytest.raises(SpeechDetectorError):
        SpeechDetector._parse_regions(
            timestamps,
            duration_seconds=2.0,
        )


def test_timestamps_are_clamped_to_audio_duration():
    regions = SpeechDetector._parse_regions(
        [
            {"start": 0.5, "end": 10.0},
            {"start": 8.0, "end": 12.0},
        ],
        duration_seconds=10.0,
    )

    assert regions == [
        SpeechRegion(start=0.5, end=10.0),
        SpeechRegion(start=8.0, end=10.0),
    ]


def test_model_load_error_is_wrapped(monkeypatch):
    import sys
    import types

    fake_module = types.ModuleType("silero_vad")
    fake_module.load_silero_vad = Mock(
        side_effect=RuntimeError("Model load failed")
    )

    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)

    instance = SpeechDetector()

    with pytest.raises(
        SpeechDetectorError,
        match="Unable to load",
    ):
        instance._load_model()


def test_missing_silero_package_raises_error(
    monkeypatch,
):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "silero_vad":
            raise ImportError("Silero VAD unavailable")

        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SileroVADNotFoundError):
        SpeechDetector()._load_model()