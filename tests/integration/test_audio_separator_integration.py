import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from video_translator.audio_separator import (
    AudioSeparator,
    DEFAULT_MODEL,
)


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "sample.wav"
    sample_rate = 44100
    duration_seconds = 8
    frame_count = sample_rate * duration_seconds

    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)

        frames = bytearray()

        for index in range(frame_count):
            time_seconds = index / sample_rate

            left = int(
                12000 * math.sin(2 * math.pi * 220 * time_seconds)
            )
            right = int(
                12000 * math.sin(2 * math.pi * 330 * time_seconds)
            )

            frames.extend(struct.pack("<hh", left, right))

        audio.writeframes(frames)

    return audio_path


def test_separate_real_audio(
    sample_audio: Path,
    tmp_path: Path,
):
    separator = AudioSeparator()
    output_dir = tmp_path / "separated"

    result = separator.separate(
        sample_audio,
        output_dir,
    )

    assert result.model == DEFAULT_MODEL
    assert result.input_path == sample_audio.resolve()
    assert result.vocals_path.is_file()
    assert result.instrumental_path.is_file()

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

    assert vocals_rate == instrumental_rate
    assert vocals.shape == instrumental.shape
    assert vocals.shape[0] > 0
    assert np.isfinite(vocals).all()
    assert np.isfinite(instrumental).all()

    assert not (output_dir / "sample_drums.wav").exists()
    assert not (output_dir / "sample_bass.wav").exists()
    assert not (output_dir / "sample_other.wav").exists()