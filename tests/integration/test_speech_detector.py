import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest

from video_translator.speech_detector import SpeechDetector


@pytest.fixture
def silent_audio(tmp_path: Path) -> Path:
    if importlib.util.find_spec("silero_vad") is None:
        pytest.skip("Silero VAD is not installed")

    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("ONNX Runtime is not installed")

    audio_path = tmp_path / "silence.wav"
    sample_rate = 16_000
    duration_seconds = 2

    samples = np.zeros(
        sample_rate * duration_seconds,
        dtype=np.int16,
    )

    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())

    return audio_path


def test_detect_silence_with_real_silero_model(
    silent_audio: Path,
):
    detector = SpeechDetector()

    regions = detector.detect(silent_audio)

    assert regions == []


def test_detect_returns_valid_regions_for_real_audio_file(
    silent_audio: Path,
):
    detector = SpeechDetector()

    regions = detector.detect(silent_audio)

    for region in regions:
        assert region.start >= 0
        assert region.end > region.start
        assert region.end <= 2.0