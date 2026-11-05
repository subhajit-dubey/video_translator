from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from video_translator.audio_aligner import (
    AudioAligner,
    AudioAlignerConfig,
    AudioAlignmentError,
)


def test_short_speech_is_padded_without_changing_timeline(tmp_path: Path):
    sample_rate = 16000
    source_audio_path = tmp_path / "source.wav"
    source_audio = np.zeros(sample_rate, dtype=np.float32)
    sf.write(str(source_audio_path), source_audio, sample_rate)

    generated_audio_path = tmp_path / "short.wav"
    generated_audio = np.full(
        int(sample_rate * 0.2),
        0.1,
        dtype=np.float32,
    )
    sf.write(str(generated_audio_path), generated_audio, sample_rate)

    segment = SimpleNamespace(
        start=0.2,
        end=0.8,
        text="नमस्ते",
        speaker_id="SPEAKER_00",
        audio_path=generated_audio_path,
    )
    result = SimpleNamespace(segments=(segment,))

    alignment = AudioAligner(
        config=AudioAlignerConfig(
            sample_rate=sample_rate,
            eq_strength=0,
            fade_seconds=0,
        )
    ).align(
        generation_result=result,
        source_audio_path=source_audio_path,
        output_dir=tmp_path / "alignment",
    )

    aligned_audio, aligned_rate = sf.read(
        str(alignment.aligned_audio_path),
        dtype="float32",
    )
    aligned_segment, segment_rate = sf.read(
        str(alignment.segments[0].audio_path),
        dtype="float32",
    )

    assert aligned_rate == sample_rate
    assert segment_rate == sample_rate
    assert alignment.duration_seconds == 1.0
    assert alignment.segments[0].start == 0.2
    assert alignment.segments[0].end == 0.8
    assert len(aligned_audio) == sample_rate
    assert len(aligned_segment) == int(sample_rate * 0.6)
    assert np.all(aligned_segment[int(sample_rate * 0.3) :] == 0)


def test_alignment_output_is_capped_to_video_duration(tmp_path: Path):
    sample_rate = 16000
    source_audio_path = tmp_path / "source.wav"
    sf.write(
        str(source_audio_path),
        np.zeros(sample_rate, dtype=np.float32),
        sample_rate,
    )
    generated_audio_path = tmp_path / "speech.wav"
    sf.write(
        str(generated_audio_path),
        np.full(int(sample_rate * 0.2), 0.1, dtype=np.float32),
        sample_rate,
    )
    segment = SimpleNamespace(
        start=0.1,
        end=0.7,
        text="नमस्ते",
        speaker_id="SPEAKER_00",
        audio_path=generated_audio_path,
    )

    alignment = AudioAligner(
        config=AudioAlignerConfig(
            sample_rate=sample_rate,
            eq_strength=0,
            fade_seconds=0,
        )
    ).align(
        generation_result=SimpleNamespace(segments=(segment,)),
        source_audio_path=source_audio_path,
        output_dir=tmp_path / "alignment",
        max_duration_seconds=0.75,
    )

    aligned_audio, _ = sf.read(
        str(alignment.aligned_audio_path),
        dtype="float32",
    )
    assert alignment.duration_seconds == pytest.approx(0.75)
    assert len(aligned_audio) == int(sample_rate * 0.75)


def test_alignment_rejects_segments_beyond_video_duration(tmp_path: Path):
    sample_rate = 16000
    source_audio_path = tmp_path / "source.wav"
    sf.write(
        str(source_audio_path),
        np.zeros(sample_rate, dtype=np.float32),
        sample_rate,
    )
    generated_audio_path = tmp_path / "speech.wav"
    sf.write(
        str(generated_audio_path),
        np.full(int(sample_rate * 0.2), 0.1, dtype=np.float32),
        sample_rate,
    )
    segment = SimpleNamespace(
        start=0.1,
        end=0.8,
        text="नमस्ते",
        speaker_id="SPEAKER_00",
        audio_path=generated_audio_path,
    )

    with pytest.raises(AudioAlignmentError, match="beyond the"):
        AudioAligner().align(
            generation_result=SimpleNamespace(segments=(segment,)),
            source_audio_path=source_audio_path,
            output_dir=tmp_path / "alignment",
            max_duration_seconds=0.75,
        )
