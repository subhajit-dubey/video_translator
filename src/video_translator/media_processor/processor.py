from __future__ import annotations

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
        missing = []

        if not self._ffmpeg:
            missing.append("ffmpeg")

        if not self._ffprobe:
            missing.append("ffprobe")

        if missing:
            raise FFmpegNotFoundError(
                f"Required executable(s) not found: {', '.join(missing)}. "
                "Install FFmpeg and ensure it is available on PATH."
            )

    @staticmethod
    def _validate_input(path: Path) -> None:
        if not path.exists():
            raise InvalidMediaError(
                f"Input file does not exist: {path}"
            )

        if not path.is_file():
            raise InvalidMediaError(
                f"Input path is not a file: {path}"
            )

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

            return frame_rate

        except (ValueError, ZeroDivisionError):
            return None

    def _probe(self, path: Path) -> dict[str, Any]:
        self._validate_input(path)
        self._require_tools()

        command = [
            self._ffprobe,
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
            metadata = json.loads(result.stdout)

            if not isinstance(metadata, dict):
                raise InvalidMediaError(
                    f"Unexpected metadata returned for: {path}"
                )

            return metadata

        except FileNotFoundError as exc:
            raise FFmpegNotFoundError(
                "ffprobe could not be executed."
            ) from exc

        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or "").strip()
            raise InvalidMediaError(
                f"Unable to inspect media file: {path}. {details}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise InvalidMediaError(
                f"ffprobe returned invalid metadata for: {path}"
            ) from exc

    def get_info(self, input_path: str | Path) -> MediaInfo:
        """Read duration, dimensions, frame rate, and audio availability."""
        path = Path(input_path)
        metadata = self._probe(path)

        streams = metadata.get("streams", [])
        format_info = metadata.get("format", {})

        if not isinstance(streams, list) or not isinstance(format_info, dict):
            raise InvalidMediaError(
                f"Unexpected media metadata structure: {path}"
            )

        video_streams = [
            stream
            for stream in streams
            if isinstance(stream, dict)
            and stream.get("codec_type") == "video"
        ]

        audio_streams = [
            stream
            for stream in streams
            if isinstance(stream, dict)
            and stream.get("codec_type") == "audio"
        ]

        if not video_streams:
            raise InvalidMediaError(
                f"No video stream found in: {path}"
            )

        video = video_streams[0]
        duration_value = format_info.get("duration")

        if duration_value is None:
            duration_value = video.get("duration")

        try:
            duration = float(duration_value)
        except (TypeError, ValueError) as exc:
            raise InvalidMediaError(
                f"Unable to determine media duration: {path}"
            ) from exc

        if not math.isfinite(duration) or duration < 0:
            raise InvalidMediaError(
                f"Invalid media duration: {duration_value}"
            )

        width = video.get("width")
        height = video.get("height")

        if width is not None and (
            not isinstance(width, int) or width <= 0
        ):
            width = None

        if height is not None and (
            not isinstance(height, int) or height <= 0
        ):
            height = None

        frame_rate = self._parse_frame_rate(
            video.get("avg_frame_rate")
            or video.get("r_frame_rate")
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
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

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
            raise ValueError("sample_rate must be greater than zero")

        if destination.suffix.lower() != ".wav":
            raise InvalidMediaError(
                "output_path must have a .wav extension."
            )

        if source.resolve() == destination.resolve():
            raise InvalidMediaError(
                "Input and output paths must be different."
            )

        info = self.get_info(source)

        if not info.has_audio:
            raise NoAudioStreamError(
                f"No audio stream found in: {source}"
            )

        self._require_tools()

        if destination.exists():
            raise InvalidMediaError(
                f"Output file already exists: {destination}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)

        command = [
            self._ffmpeg,
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
            raise FFmpegNotFoundError(
                "FFmpeg could not be executed."
            ) from exc

        except subprocess.CalledProcessError as exc:
            self._remove_partial_output(destination)
            details = (exc.stderr or "").strip()
            raise MediaProcessorError(
                f"Audio extraction failed for {source}. {details}"
            ) from exc

        if (
            not destination.is_file()
            or destination.stat().st_size == 0
        ):
            self._remove_partial_output(destination)
            raise MediaProcessorError(
                "FFmpeg completed without producing a valid output file."
            )

        return destination