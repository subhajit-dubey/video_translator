from __future__ import annotations

import contextlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MediaProcessorError(Exception):
    """Base exception for media-processing errors."""


class FFmpegNotFoundError(MediaProcessorError):
    """Raised when FFmpeg or ffprobe is unavailable."""


class InvalidMediaError(MediaProcessorError):
    """Raised when media is missing, invalid, or unsupported."""


class NoAudioStreamError(MediaProcessorError):
    """Raised when the input video contains no audio stream."""


@dataclass(frozen=True)
class MediaInfo:
    """Metadata describing a video file."""

    path: Path
    duration_seconds: float
    width: int | None
    height: int | None
    frame_rate: float | None
    has_audio: bool


class MediaProcessor:
    """Inspect video metadata and extract audio using FFmpeg."""

    def __init__(self) -> None:
        self._ffmpeg = shutil.which("ffmpeg")
        self._ffprobe = shutil.which("ffprobe")

    def _require_tools(self) -> None:
        missing = [
            tool for tool in ("ffmpeg", "ffprobe") if not getattr(self, f"_{tool}")
        ]
        if missing:
            msg = (
                f"Required executable(s) not found: {', '.join(missing)}. "
                "Install FFmpeg and ensure it is available on PATH."
            )
            raise FFmpegNotFoundError(msg)

    @staticmethod
    def _validate_input(path: Path) -> None:
        if not path.exists():
            msg = f"Input file does not exist: {path}"
            raise InvalidMediaError(msg)

        if not path.is_file():
            msg = f"Input path is not a file: {path}"
            raise InvalidMediaError(msg)

    @staticmethod
    def _parse_frame_rate(value: str | None) -> float | None:
        if not value or value in {"0/0", "N/A"}:
            return None

        try:
            if "/" in value:
                numerator, denominator = value.split("/", maxsplit=1)
                denominator_value = float(denominator)

                if denominator_value == 0:
                    return None

                frame_rate = float(numerator) / denominator_value
            else:
                frame_rate = float(value)

            if not math.isfinite(frame_rate) or frame_rate <= 0:
                return None

        except (ValueError, ZeroDivisionError):
            return None
        else:
            return frame_rate

    def _probe(self, path: Path) -> dict[str, Any]:
        self._validate_input(path)
        self._require_tools()

        command: list[str] = [
            self._ffprobe,  # type: ignore[list-item]
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            msg = "ffprobe could not be executed."
            raise FFmpegNotFoundError(msg) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or "").strip()
            msg = f"Unable to inspect media file: {path}. {details}"
            raise InvalidMediaError(msg) from exc
        else:
            try:
                metadata = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                msg = f"ffprobe returned invalid metadata for: {path}"
                raise InvalidMediaError(msg) from exc

            if not isinstance(metadata, dict):
                msg = f"Unexpected metadata returned for: {path}"
                raise InvalidMediaError(msg)

            return metadata

    def get_info(self, input_path: str | Path) -> MediaInfo:
        """Read duration, dimensions, frame rate, and audio availability."""
        path = Path(input_path)
        metadata = self._probe(path)

        streams = metadata.get("streams", [])
        format_info = metadata.get("format", {})

        if not isinstance(streams, list) or not isinstance(format_info, dict):
            msg = f"Unexpected media metadata structure: {path}"
            raise InvalidMediaError(msg)

        video_streams = [
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ]

        audio_streams = [
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ]

        if not video_streams:
            msg = f"No video stream found in: {path}"
            raise InvalidMediaError(msg)

        video = video_streams[0]
        duration_value = format_info.get("duration")

        if duration_value is None:
            duration_value = video.get("duration")

        if duration_value is None:
            msg = f"Unable to determine media duration: {path}"
            raise InvalidMediaError(msg)

        try:
            duration = float(duration_value)
        except (TypeError, ValueError) as exc:
            msg = f"Unable to determine media duration: {path}"
            raise InvalidMediaError(msg) from exc

        if not math.isfinite(duration) or duration < 0:
            msg = f"Invalid media duration: {duration_value}"
            raise InvalidMediaError(msg)

        width = video.get("width")
        height = video.get("height")

        if width is not None and (not isinstance(width, int) or width <= 0):
            width = None

        if height is not None and (not isinstance(height, int) or height <= 0):
            height = None

        frame_rate = self._parse_frame_rate(
            video.get("avg_frame_rate") or video.get("r_frame_rate"),
        )

        return MediaInfo(
            path=path,
            duration_seconds=duration,
            width=width,
            height=height,
            frame_rate=frame_rate,
            has_audio=bool(audio_streams),
        )

    @staticmethod
    def _remove_partial_output(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    def extract_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        sample_rate: int = 16_000,
    ) -> Path:
        """Extract mono, 16-bit PCM WAV audio from a video."""
        source = Path(input_path)
        destination = Path(output_path)

        if sample_rate <= 0:
            msg = "sample_rate must be greater than zero"
            raise ValueError(msg)

        if destination.suffix.lower() != ".wav":
            msg = "output_path must have a .wav extension."
            raise InvalidMediaError(msg)

        if source.resolve() == destination.resolve():
            msg = "Input and output paths must be different."
            raise InvalidMediaError(msg)

        info = self.get_info(source)

        if not info.has_audio:
            msg = f"No audio stream found in: {source}"
            raise NoAudioStreamError(msg)

        self._require_tools()

        if destination.exists():
            msg = f"Output file already exists: {destination}"
            raise InvalidMediaError(msg)

        destination.parent.mkdir(parents=True, exist_ok=True)

        command: list[str] = [
            self._ffmpeg,  # type: ignore[list-item]
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]

        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            self._remove_partial_output(destination)
            msg = "FFmpeg could not be executed."
            raise FFmpegNotFoundError(msg) from exc
        except subprocess.CalledProcessError as exc:
            self._remove_partial_output(destination)
            details = (exc.stderr or "").strip()
            msg = f"Audio extraction failed for {source}. {details}"
            raise MediaProcessorError(msg) from exc
        else:
            if not destination.is_file() or destination.stat().st_size == 0:
                self._remove_partial_output(destination)
                msg = "FFmpeg completed without producing a valid output file."
                raise MediaProcessorError(msg)

        return destination
