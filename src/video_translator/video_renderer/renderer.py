from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VideoRendererError(Exception):
    """Raised when video rendering or audio muxing fails."""


@dataclass(frozen=True)
class VideoRendererConfig:
    """Configuration for rendering the final dubbed video."""

    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_crf: int = 18
    pixel_format: str = "yuv420p"

    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 48000
    audio_channels: int = 2

    # Used only when a separate background audio track is supplied.
    dubbed_audio_volume: float = 1.0
    background_audio_volume: float = 0.25

    ffmpeg_timeout_seconds: float = 3600.0

    def __post_init__(self) -> None:
        if not self.video_codec.strip():
            raise ValueError("video_codec must not be empty.")

        if not self.video_preset.strip():
            raise ValueError("video_preset must not be empty.")

        if not 0 <= self.video_crf <= 51:
            raise ValueError("video_crf must be between 0 and 51.")

        if not self.pixel_format.strip():
            raise ValueError("pixel_format must not be empty.")

        if not self.audio_codec.strip() or not self.audio_bitrate.strip():
            raise ValueError("Audio codec and bitrate must not be empty.")

        if self.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be positive.")

        if self.audio_channels not in (1, 2):
            raise ValueError("audio_channels must be 1 or 2.")

        if not math.isfinite(self.dubbed_audio_volume) or self.dubbed_audio_volume <= 0:
            raise ValueError("dubbed_audio_volume must be positive and finite.")

        if (
            not math.isfinite(self.background_audio_volume)
            or self.background_audio_volume < 0
        ):
            raise ValueError("background_audio_volume must be non-negative and finite.")

        if (
            not math.isfinite(self.ffmpeg_timeout_seconds)
            or self.ffmpeg_timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_timeout_seconds must be positive and finite.")


@dataclass(frozen=True)
class VideoRenderResult:
    """Metadata for a successfully rendered video."""

    source_video_path: Path
    aligned_audio_path: Path
    output_path: Path
    duration_seconds: float
    video_codec: str
    audio_codec: str
    background_audio_mixed: bool
    output_size_bytes: int


class VideoRenderer:
    """Render the final video using the original video and aligned dubbing.

    The original video stream is retained. Its original audio is replaced
    by the aligned dubbing unless a separate background audio track is
    explicitly supplied.

    Rendering requires FFmpeg and FFprobe to be installed and available
    on PATH.

    Example:
        renderer = VideoRenderer()

        result = renderer.render(
            source_video_path="input.mp4",
            aligned_audio_path="alignment/aligned_dubbing.wav",
            output_path="output/final_dubbed_video.mp4",
            background_audio_path="background.wav",
        )
    """

    def __init__(
        self,
        *,
        config: VideoRendererConfig | None = None,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ) -> None:
        self.config = config or VideoRendererConfig()

        self.ffmpeg_path = self._resolve_executable(
            ffmpeg_path,
            "FFmpeg",
        )
        self.ffprobe_path = self._resolve_executable(
            ffprobe_path,
            "FFprobe",
        )

    @staticmethod
    def _resolve_executable(
        executable: str,
        name: str,
    ) -> str:
        """Resolve an executable by name or explicit path."""
        resolved = shutil.which(executable)

        if resolved:
            return resolved

        candidate = Path(executable)

        if candidate.is_file():
            return str(candidate.resolve())

        raise VideoRendererError(
            f"{name} executable was not found: {executable!r}. "
            "Install FFmpeg and ensure both ffmpeg and ffprobe are on PATH."
        )

    @staticmethod
    def _validate_input_file(
        path: str | Path,
        label: str,
    ) -> Path:
        """Validate and resolve an input file."""
        resolved = Path(path).expanduser().resolve()

        if not resolved.is_file():
            raise VideoRendererError(f"{label} does not exist: {resolved}")

        if resolved.stat().st_size == 0:
            raise VideoRendererError(f"{label} is empty: {resolved}")

        return resolved

    def _probe(self, path: Path) -> dict[str, Any]:
        """Read stream and container metadata with FFprobe."""
        command = [
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:"
            "stream=codec_type,codec_name,duration,width,height",
            "-of",
            "json",
            str(path),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.ffmpeg_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VideoRendererError(
                f"FFprobe timed out while inspecting: {path}"
            ) from exc
        except OSError as exc:
            raise VideoRendererError(f"Unable to execute FFprobe for: {path}") from exc

        if completed.returncode != 0:
            raise VideoRendererError(
                f"FFprobe failed for {path}:\n{completed.stderr[-4000:]}"
            )

        try:
            raw_metadata: Any = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise VideoRendererError(
                f"FFprobe returned invalid metadata for: {path}"
            ) from exc

        assert isinstance(raw_metadata, dict), "FFprobe returned non-dict metadata"
        metadata: dict[str, Any] = raw_metadata

        return metadata

    def _get_duration(
        self,
        metadata: dict[str, Any],
        *,
        prefer_video_stream: bool,
    ) -> float:
        """Get a finite, positive media duration."""
        streams = metadata.get("streams", [])

        if prefer_video_stream:
            candidates = [
                stream for stream in streams if stream.get("codec_type") == "video"
            ]

            for stream in candidates:
                try:
                    duration = float(stream.get("duration", ""))
                except (TypeError, ValueError):
                    continue

                if math.isfinite(duration) and duration > 0:
                    return duration

        container = metadata.get("format", {})

        try:
            duration = float(container.get("duration", ""))
        except (AttributeError, TypeError, ValueError):
            duration = 0.0

        if math.isfinite(duration) and duration > 0:
            return duration

        # Last resort: use the longest reported stream duration.
        stream_durations: list[float] = []

        for stream in streams:
            try:
                value = float(stream.get("duration", ""))
            except (TypeError, ValueError):
                continue

            if math.isfinite(value) and value > 0:
                stream_durations.append(value)

        if stream_durations:
            return max(stream_durations)

        raise VideoRendererError("Unable to determine a valid media duration.")

    @staticmethod
    def _require_stream(
        metadata: dict[str, Any],
        stream_type: str,
        path: Path,
    ) -> dict[str, Any]:
        """Return the first stream of a required type."""
        for stream in metadata.get("streams", []):
            if isinstance(stream, dict) and stream.get("codec_type") == stream_type:
                return stream

        raise VideoRendererError(f"No {stream_type} stream was found in: {path}")

    @staticmethod
    def _volume_filter(volume: float) -> str:
        """Format a volume for an FFmpeg audio filter."""
        return f"volume={volume:.6f}"

    def _build_command(
        self,
        *,
        source_video: Path,
        aligned_audio: Path,
        output_path: Path,
        video_duration: float,
        background_audio: Path | None,
        dubbed_volume: float,
        background_volume: float,
    ) -> list[str]:
        """Build the FFmpeg command for replacing or mixing audio."""
        config = self.config

        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source_video),
            "-i",
            str(aligned_audio),
        ]

        if background_audio is not None:
            command.extend(["-i", str(background_audio)])

        command.extend(
            [
                "-map",
                "0:v:0",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
            ]
        )

        if background_audio is None:
            # Replace original audio with aligned dubbing.
            command.extend(
                [
                    "-map",
                    "1:a:0",
                    "-af",
                    (
                        f"aresample={config.audio_sample_rate},"
                        f"apad=whole_dur={video_duration:.6f}"
                    ),
                ]
            )
        else:
            # Mix the optional background track and the translated speech.
            # The source video's original audio is intentionally not included.
            dubbed_filter = (
                f"[1:a:0]"
                f"aresample={config.audio_sample_rate},"
                f"{self._volume_filter(dubbed_volume)},"
                f"apad=whole_dur={video_duration:.6f}"
                f"[dub]"
            )

            background_filter = (
                f"[2:a:0]"
                f"aresample={config.audio_sample_rate},"
                f"{self._volume_filter(background_volume)},"
                f"apad=whole_dur={video_duration:.6f}"
                f"[bg]"
            )

            mix_filter = (
                "[bg][dub]"
                "amix=inputs=2:duration=longest:"
                "dropout_transition=0:normalize=0,"
                "alimiter=limit=0.95:level=false"
                "[aout]"
            )

            command.extend(
                [
                    "-filter_complex",
                    f"{dubbed_filter};{background_filter};{mix_filter}",
                    "-map",
                    "[aout]",
                ]
            )

        command.extend(
            [
                "-t",
                f"{video_duration:.6f}",
                "-c:v",
                config.video_codec,
                "-preset",
                config.video_preset,
                "-crf",
                str(config.video_crf),
                "-pix_fmt",
                config.pixel_format,
                "-c:a",
                config.audio_codec,
                "-b:a",
                config.audio_bitrate,
                "-ar",
                str(config.audio_sample_rate),
                "-ac",
                str(config.audio_channels),
                "-max_muxing_queue_size",
                "2048",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(output_path),
            ]
        )

        return command

    def render(
        self,
        source_video_path: str | Path,
        aligned_audio_path: str | Path,
        output_path: str | Path,
        *,
        background_audio_path: str | Path | None = None,
        dubbed_audio_volume: float | None = None,
        background_audio_volume: float | None = None,
    ) -> VideoRenderResult:
        """Create the final MP4 video with aligned translated speech.

        Args:
            source_video_path:
                Original video whose visual stream must be retained.
            aligned_audio_path:
                Timeline-aligned WAV produced by AudioAligner.
            output_path:
                Destination MP4 file.
            background_audio_path:
                Optional music or background-effects track. When supplied,
                it is mixed with the dubbing; source-video audio is excluded.
            dubbed_audio_volume:
                Optional volume multiplier for the translated speech.
            background_audio_volume:
                Optional volume multiplier for the background track.

        Returns:
            Metadata describing the rendered MP4.

        Raises:
            VideoRendererError:
                If inputs are invalid, FFmpeg fails, or the output is invalid.
        """
        source_video = self._validate_input_file(
            source_video_path,
            "Source video",
        )
        aligned_audio = self._validate_input_file(
            aligned_audio_path,
            "Aligned dubbing audio",
        )

        destination = Path(output_path).expanduser().resolve()

        if destination.suffix.lower() != ".mp4":
            raise VideoRendererError("The output filename must use the .mp4 extension.")

        if destination in (source_video, aligned_audio):
            raise VideoRendererError(
                "The output path must differ from the input paths."
            )

        background_audio: Path | None = None

        if background_audio_path is not None:
            background_audio = self._validate_input_file(
                background_audio_path,
                "Background audio",
            )

            if destination == background_audio:
                raise VideoRendererError(
                    "The output path must differ from the background audio."
                )

        config = self.config

        dubbed_volume = (
            config.dubbed_audio_volume
            if dubbed_audio_volume is None
            else dubbed_audio_volume
        )

        background_volume = (
            config.background_audio_volume
            if background_audio_volume is None
            else background_audio_volume
        )

        if not math.isfinite(dubbed_volume) or dubbed_volume <= 0:
            raise ValueError("dubbed_audio_volume must be positive and finite.")

        if not math.isfinite(background_volume) or background_volume < 0:
            raise ValueError("background_audio_volume must be non-negative and finite.")

        if background_audio is None and background_audio_volume is not None:
            raise ValueError("background_audio_volume requires background_audio_path.")

        source_metadata = self._probe(source_video)
        self._require_stream(
            source_metadata,
            "video",
            source_video,
        )

        aligned_metadata = self._probe(aligned_audio)
        self._require_stream(
            aligned_metadata,
            "audio",
            aligned_audio,
        )

        if background_audio is not None:
            background_metadata = self._probe(background_audio)
            self._require_stream(
                background_metadata,
                "audio",
                background_audio,
            )

        video_duration = self._get_duration(
            source_metadata,
            prefer_video_stream=True,
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Use a temporary MP4 so a failed render cannot leave a partial
        # file masquerading as the requested final output.
        temporary_output = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.tmp.mp4"
        )

        command = self._build_command(
            source_video=source_video,
            aligned_audio=aligned_audio,
            output_path=temporary_output,
            video_duration=video_duration,
            background_audio=background_audio,
            dubbed_volume=dubbed_volume,
            background_volume=background_volume,
        )

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=config.ffmpeg_timeout_seconds,
                check=False,
            )

            if completed.returncode != 0:
                raise VideoRendererError(
                    "FFmpeg rendering failed.\n"
                    f"Command: {' '.join(command)}\n"
                    f"Error output:\n{completed.stderr[-8000:]}"
                )

            if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
                raise VideoRendererError(
                    "FFmpeg completed without producing a valid output file."
                )

            output_metadata = self._probe(temporary_output)

            video_stream = self._require_stream(
                output_metadata,
                "video",
                temporary_output,
            )
            audio_stream = self._require_stream(
                output_metadata,
                "audio",
                temporary_output,
            )

            # Confirm that the output has a usable duration before replacing
            # any existing destination file.
            output_duration = self._get_duration(
                output_metadata,
                prefer_video_stream=True,
            )

            os.replace(temporary_output, destination)

        except subprocess.TimeoutExpired as exc:
            raise VideoRendererError(
                "FFmpeg rendering timed out after "
                f"{config.ffmpeg_timeout_seconds:.0f} seconds."
            ) from exc

        except OSError as exc:
            raise VideoRendererError(
                f"Unable to complete video rendering: {exc}"
            ) from exc

        finally:
            temporary_output.unlink(missing_ok=True)

        return VideoRenderResult(
            source_video_path=source_video,
            aligned_audio_path=aligned_audio,
            output_path=destination,
            duration_seconds=output_duration,
            video_codec=str(video_stream.get("codec_name", "unknown")),
            audio_codec=str(audio_stream.get("codec_name", "unknown")),
            background_audio_mixed=background_audio is not None,
            output_size_bytes=destination.stat().st_size,
        )
