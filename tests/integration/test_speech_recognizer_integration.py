import importlib.util
import wave
from pathlib import Path

import pytest

from video_translator.speech_recognizer import SpeechRecognizer


@pytest.fixture
def silent_audio(tmp_path: Path) -> Path:
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("Faster-Whisper is not installed")

    audio_path = tmp_path / "silence.wav"
    sample_rate = 16_000
    duration_seconds = 3

    samples = bytes(sample_rate * duration_seconds * 2)

    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples)

    return audio_path


def test_transcribe_real_audio_with_faster_whisper(
    silent_audio: Path,
):
    recognizer = SpeechRecognizer(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        vad_filter=True,
    )

    result = recognizer.transcribe(silent_audio)

    assert result.audio_path == silent_audio
    assert result.duration_seconds >= 3.0
    assert isinstance(result.segments, tuple)

    for segment in result.segments:
        assert segment.start >= 0
        assert segment.end > segment.start
        assert segment.end <= result.duration_seconds
        assert isinstance(segment.text, str)