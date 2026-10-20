import json
import os
from pathlib import Path

import pytest
import torch
from dotenv import load_dotenv

from video_translator.voice_generator import VoiceGenerator


load_dotenv()


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_VOICE_GENERATOR_INTEGRATION") != "1",
    reason=(
        "Set RUN_VOICE_GENERATOR_INTEGRATION=1 "
        "to run model integration tests."
    ),
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this integration test.",
)
def test_voice_generator_end_to_end(tmp_path: Path):
    source_audio_value = os.getenv("VOICE_GENERATOR_SOURCE_AUDIO")
    translation_json_value = os.getenv("VOICE_GENERATOR_TRANSLATION_JSON")

    if not source_audio_value or not translation_json_value:
        pytest.skip(
            "Set VOICE_GENERATOR_SOURCE_AUDIO and "
            "VOICE_GENERATOR_TRANSLATION_JSON."
        )

    source_audio = Path(source_audio_value)
    translation_json = Path(translation_json_value)

    if not source_audio.is_file():
        pytest.skip(f"Source audio is unavailable: {source_audio}")

    if not translation_json.is_file():
        pytest.skip(
            f"Translation JSON is unavailable: {translation_json}"
        )

    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN is required for pyannote diarization.")

    tts_model = os.getenv(
        "VOICE_GENERATOR_TTS_MODEL",
        "F5-Hindi-Small",
    )
    tts_ckpt_file = os.getenv("VOICE_GENERATOR_TTS_CKPT_FILE", "")
    tts_vocab_file = os.getenv("VOICE_GENERATOR_TTS_VOCAB_FILE", "")

    if tts_model == "F5-Hindi-Small" and not (
        tts_ckpt_file and tts_vocab_file
    ):
        pytest.skip(
            "VOICE_GENERATOR_TTS_CKPT_FILE and "
            "VOICE_GENERATOR_TTS_VOCAB_FILE are required for Hindi TTS."
        )

    with translation_json.open(encoding="utf-8") as handle:
        translation = json.load(handle)

    if not translation.get("segments"):
        pytest.skip("The translation JSON contains no text segments.")

    reference_texts_path = os.getenv(
        "VOICE_GENERATOR_REFERENCE_TEXTS_JSON"
    )

    reference_texts = None

    if reference_texts_path:
        reference_path = Path(reference_texts_path)

        with reference_path.open(encoding="utf-8") as handle:
            reference_texts = json.load(handle)

    generator = VoiceGenerator(
        diarization_device="cuda",
        tts_device="cuda",
        tts_model=tts_model,
        tts_ckpt_file=tts_ckpt_file,
        tts_vocab_file=tts_vocab_file,
    )

    result = generator.generate(
        translation_json,
        source_audio,
        tmp_path / "voice_output",
        reference_texts=reference_texts,
    )

    assert result.manifest_path.is_file()
    assert result.speaker_profiles
    assert result.segments

    manifest = json.loads(
        result.manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["alignment_pending"] is True

    for segment in result.segments:
        assert segment.audio_path.is_file()
        assert segment.duration_seconds > 0