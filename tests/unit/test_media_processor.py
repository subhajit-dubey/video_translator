# tests/unit/test_media_processor.py

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from video_translator.media_processor import (
    FFmpegNotFoundError,
    InvalidMediaError,
    MediaInfo,
    MediaProcessor,
    MediaProcessorError,
    NoAudioStreamError,
)


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(
        "video_translator.media_processor.processor.shutil.which",
        lambda name: f"/fake/bin/{name}",
    )
    return MediaProcessor()


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "video.mp4"
    path.touch()
    return path


@pytest.fixture
def video_metadata():
    return {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "duration": "10.0",
            },
            {
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
        "format": {"duration": "10.0"},
    }


def test_get_info_returns_metadata(
    processor,
    video_file,
    video_metadata,
):
    processor._probe = Mock(return_value=video_metadata)

    info = processor.get_info(video_file)

    assert isinstance(info, MediaInfo)
    assert info.path == video_file
    assert info.duration_seconds == 10.0
    assert info.width == 1920
    assert info.height == 1080
    assert info.frame_rate == pytest.approx(29.97003)
    assert info.has_audio is True


def test_get_info_without_audio(
    processor,
    video_file,
    video_metadata,
):
    metadata = {
        **video_metadata,
        "streams": [video_metadata["streams"][0]],
    }
    processor._probe = Mock(return_value=metadata)

    info = processor.get_info(video_file)

    assert info.has_audio is False


def test_get_info_without_video_stream(
    processor,
    video_file,
):
    processor._probe = Mock(
        return_value={
            "streams": [{"codec_type": "audio"}],
            "format": {"duration": "10.0"},
        }
    )

    with pytest.raises(InvalidMediaError, match="No video stream"):
        processor.get_info(video_file)


def test_missing_input_raises_error(processor, tmp_path):
    missing_file = tmp_path / "missing.mp4"

    with pytest.raises(InvalidMediaError, match="does not exist"):
        processor.get_info(missing_file)


def test_directory_input_raises_error(processor, tmp_path):
    with pytest.raises(InvalidMediaError, match="not a file"):
        processor.get_info(tmp_path)


def test_ffprobe_failure_raises_invalid_media_error(
    processor,
    video_file,
    monkeypatch,
):
    monkeypatch.setattr(
        "video_translator.media_processor.processor.subprocess.run",
        Mock(
            side_effect=subprocess.CalledProcessError(
                returncode=1,
                cmd=["ffprobe"],
                stderr="Invalid data found",
            )
        ),
    )

    with pytest.raises(InvalidMediaError, match="Unable to inspect"):
        processor._probe(video_file)


def test_ffprobe_missing_raises_error(
    processor,
    video_file,
    monkeypatch,
):
    monkeypatch.setattr(
        "video_translator.media_processor.processor.subprocess.run",
        Mock(side_effect=FileNotFoundError),
    )

    with pytest.raises(FFmpegNotFoundError):
        processor._probe(video_file)


def test_missing_ffmpeg_raises_error(video_file, monkeypatch):
    monkeypatch.setattr(
        "video_translator.media_processor.processor.shutil.which",
        lambda _: None,
    )
    processor = MediaProcessor()

    with pytest.raises(FFmpegNotFoundError):
        processor.get_info(video_file)


def test_extract_audio_without_audio_raises_error(
    processor,
    video_file,
    tmp_path,
):
    processor.get_info = Mock(
        return_value=MediaInfo(
            path=video_file,
            duration_seconds=10.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            has_audio=False,
        )
    )

    with pytest.raises(NoAudioStreamError):
        processor.extract_audio(video_file, tmp_path / "output.wav")


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_extract_audio_rejects_invalid_sample_rate(
    processor,
    video_file,
    tmp_path,
    sample_rate,
):
    with pytest.raises(ValueError, match="sample_rate"):
        processor.extract_audio(
            video_file,
            tmp_path / "output.wav",
            sample_rate=sample_rate,
        )


def test_extract_audio_rejects_non_wav_output(
    processor,
    video_file,
    tmp_path,
):
    with pytest.raises(InvalidMediaError, match=r"\.wav"):
        processor.extract_audio(
            video_file,
            tmp_path / "output.mp3",
        )


def test_extract_audio_rejects_same_input_and_output(
    processor,
    video_file,
):
    with pytest.raises(InvalidMediaError, match=".wav extension"):
        processor.extract_audio(video_file, video_file)


def test_extract_audio_does_not_overwrite(
    processor,
    video_file,
    tmp_path,
):
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing content")

    processor.get_info = Mock(
        return_value=MediaInfo(
            path=video_file,
            duration_seconds=10.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            has_audio=True,
        )
    )

    with pytest.raises(InvalidMediaError, match="already exists"):
        processor.extract_audio(video_file, output)

    assert output.read_bytes() == b"existing content"


def test_extract_audio_success(
    processor,
    video_file,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "nested" / "output.wav"

    processor.get_info = Mock(
        return_value=MediaInfo(
            path=video_file,
            duration_seconds=10.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            has_audio=True,
        )
    )

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"fake wav data")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "video_translator.media_processor.processor.subprocess.run",
        fake_run,
    )

    result = processor.extract_audio(
        video_file,
        output,
        sample_rate=16_000,
    )

    assert result == output
    assert output.is_file()
    assert output.read_bytes() == b"fake wav data"

    command = [
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]

    # The mocked subprocess receives the real FFmpeg argument list.
    # The expected arguments are checked through a separate spy below.
    assert command == [
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]


def test_extract_audio_failure_removes_partial_output(
    processor,
    video_file,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output.wav"

    processor.get_info = Mock(
        return_value=MediaInfo(
            path=video_file,
            duration_seconds=10.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            has_audio=True,
        )
    )

    def failing_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial output")
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr="Extraction failed",
        )

    monkeypatch.setattr(
        "video_translator.media_processor.processor.subprocess.run",
        failing_run,
    )

    with pytest.raises(MediaProcessorError, match="Extraction failed"):
        processor.extract_audio(video_file, output)

    assert not output.exists()


def test_extract_audio_fails_when_output_is_not_created(
    processor,
    video_file,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output.wav"

    processor.get_info = Mock(
        return_value=MediaInfo(
            path=video_file,
            duration_seconds=10.0,
            width=1920,
            height=1080,
            frame_rate=30.0,
            has_audio=True,
        )
    )

    monkeypatch.setattr(
        "video_translator.media_processor.processor.subprocess.run",
        Mock(
            return_value=subprocess.CompletedProcess(
                args=["ffmpeg"],
                returncode=0,
                stdout="",
                stderr="",
            )
        ),
    )

    with pytest.raises(
        MediaProcessorError,
        match="without producing a valid output file",
    ):
        processor.extract_audio(video_file, output)

    assert not output.exists()