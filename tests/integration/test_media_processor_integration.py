import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from video_translator.media_processor import MediaProcessor


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg and ffprobe are required")

    video = tmp_path / "sample.mp4"

    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return video


def test_inspect_real_video(sample_video: Path):
    processor = MediaProcessor()

    info = processor.get_info(sample_video)

    assert info.path == sample_video
    assert info.duration_seconds == pytest.approx(2.0, abs=0.2)
    assert info.width == 320
    assert info.height == 240
    assert info.frame_rate == pytest.approx(25.0)
    assert info.has_audio is True


def test_extract_real_audio(
    sample_video: Path,
    tmp_path: Path,
):
    processor = MediaProcessor()
    output = tmp_path / "extracted.wav"

    result = processor.extract_audio(
        sample_video,
        output,
        sample_rate=16_000,
    )

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 44

    with wave.open(str(output), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16_000
        assert audio.getsampwidth() == 2
        assert audio.getnframes() > 0
