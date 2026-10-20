import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from video_translator.voice_generator import (
    DEFAULT_HINDI_TTS_MODEL,
    DEFAULT_TTS_MODEL,
    F5TTSSynthesizer,
    InvalidAudioError,
    InvalidTranslationFileError,
    PyannoteSpeakerDiarizer,
    SpeakerTurn,
    VoiceGenerator,
    VoiceGeneratorError,
)
from video_translator.voice_generator import generator as voice_generator_module


class FakeReferenceTranscriber:
    """Mock Faster-Whisper reference transcription."""

    def transcribe(self, audio_path):
        return SimpleNamespace(
            text=f"Original speech from {Path(audio_path).stem}."
        )


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        assert yield_label is True

        yield (
            SimpleNamespace(start=0.0, end=1.4),
            "track_1",
            "SPEAKER_00",
        )

        yield (
            SimpleNamespace(start=1.5, end=3.0),
            "track_2",
            "SPEAKER_01",
        )


class FakePipeline:
    def __call__(self, audio_path):
        return FakeAnnotation()


class FakeSynthesizer:
    model_name = "fake-f5-tts"
    device = "cpu"
    ckpt_file = ""
    vocab_file = ""

    def __init__(self):
        self.calls = []

    def synthesize(
        self,
        text,
        speaker_wav,
        output_path,
        *,
        reference_text="",
    ):
        self.calls.append(
            {
                "text": text,
                "speaker_wav": Path(speaker_wav),
                "reference_text": reference_text,
                "output_path": Path(output_path),
            }
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = 22050
        duration = 0.8
        time = np.arange(
            int(sample_rate * duration),
            dtype=np.float32,
        ) / sample_rate

        audio = 0.2 * np.sin(2 * np.pi * 220 * time)

        sf.write(
            str(output),
            audio,
            sample_rate,
            subtype="PCM_16",
        )

        return output


@pytest.fixture
def source_audio(tmp_path):
    sample_rate = 16000
    duration = 3.0

    samples = np.zeros(
        int(sample_rate * duration),
        dtype=np.float32,
    )

    time = np.arange(len(samples)) / sample_rate
    samples += 0.2 * np.sin(2 * np.pi * 180 * time)

    path = tmp_path / "source.wav"

    sf.write(
        str(path),
        samples,
        sample_rate,
        subtype="PCM_16",
    )

    return path


@pytest.fixture
def translation_json(tmp_path):
    path = tmp_path / "translated.json"

    path.write_text(
        json.dumps(
            {
                "source_language": "eng_Latn",
                "target_language": "hin_Deva",
                "segments": [
                    {
                        "start": 0.1,
                        "end": 0.8,
                        "text": "नमस्ते।",
                    },
                    {
                        "start": 1.7,
                        "end": 2.4,
                        "text": "आप कैसे हैं?",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


@pytest.fixture
def fake_diarizer():
    return PyannoteSpeakerDiarizer(
        device="cpu",
        pipeline=FakePipeline(),
    )


@pytest.fixture
def fake_synthesizer():
    return FakeSynthesizer()


@pytest.fixture
def voice_generator(fake_diarizer, fake_synthesizer):
    return VoiceGenerator(
        diarizer=fake_diarizer,
        synthesizer=fake_synthesizer,
        reference_transcriber=FakeReferenceTranscriber(),
    )


def test_default_tts_model():
    assert DEFAULT_TTS_MODEL == "F5TTS_v1_Base"

    synthesizer = F5TTSSynthesizer()

    assert synthesizer.model_name == DEFAULT_TTS_MODEL
    assert synthesizer.device == "cuda"


def test_hindi_tts_model_uses_small_architecture_and_custom_files(
    monkeypatch,
):
    from f5_tts import api

    model = Mock()
    f5_tts = Mock(return_value=model)
    monkeypatch.setattr(api, "F5TTS", f5_tts)

    synthesizer = F5TTSSynthesizer(
        model_name=DEFAULT_HINDI_TTS_MODEL,
        device="cpu",
        ckpt_file="/models/model_2500000.safetensors",
        vocab_file="/models/vocab.txt",
    )

    assert synthesizer._load_model() is model
    f5_tts.assert_called_once_with(
        model="F5TTS_Small",
        ckpt_file="/models/model_2500000.safetensors",
        vocab_file="/models/vocab.txt",
        device="cpu",
    )


@pytest.mark.parametrize(
    ("ckpt_file", "vocab_file"),
    [("", "/models/vocab.txt"), ("/models/model.safetensors", "")],
)
def test_hindi_tts_model_requires_custom_checkpoint_and_vocab(
    ckpt_file,
    vocab_file,
):
    with pytest.raises(
        ValueError,
        match="requires both ckpt_file and vocab_file",
    ):
        F5TTSSynthesizer(
            model_name=DEFAULT_HINDI_TTS_MODEL,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
        )


def test_preload_cuda_nvrtc_loads_matching_runtime_library(
    tmp_path,
    monkeypatch,
):
    site_packages = tmp_path / "site-packages"
    nvrtc_path = (
        site_packages
        / "nvidia"
        / "cuda_nvrtc"
        / "lib"
        / "libnvrtc.so.12"
    )
    nvrtc_path.parent.mkdir(parents=True)
    nvrtc_path.touch()

    torch_module = SimpleNamespace(
        __file__=site_packages / "torch" / "__init__.py",
        version=SimpleNamespace(cuda="12.6"),
    )
    load_library = Mock()
    monkeypatch.setattr(
        voice_generator_module.ctypes,
        "CDLL",
        load_library,
    )
    monkeypatch.setattr(voice_generator_module.sys, "platform", "linux")

    voice_generator_module._preload_cuda_nvrtc(torch_module)

    load_library.assert_called_once_with(
        str(nvrtc_path),
        mode=ctypes.RTLD_GLOBAL,
    )


def test_preload_cuda_nvrtc_reports_missing_runtime_library(tmp_path):
    site_packages = tmp_path / "site-packages"
    torch_module = SimpleNamespace(
        __file__=site_packages / "torch" / "__init__.py",
        version=SimpleNamespace(cuda="12.6"),
    )

    with pytest.raises(
        VoiceGeneratorError,
        match="CUDA NVRTC runtime library for CUDA 12 was not found",
    ):
        voice_generator_module._preload_cuda_nvrtc(torch_module)


def test_hindi_normalization_preserves_devanagari_text():
    from f5_tts.infer import utils_infer

    original_converter = utils_infer.convert_char_to_pinyin

    with voice_generator_module._hindi_text_normalization():
        assert utils_infer.convert_char_to_pinyin(["नमस्ते"]) == ["नमस्ते"]

    assert utils_infer.convert_char_to_pinyin is original_converter


def test_diarize_returns_speaker_turns(
    fake_diarizer,
    source_audio,
):
    turns = fake_diarizer.diarize(source_audio)

    assert turns == [
        SpeakerTurn(0.0, 1.4, "SPEAKER_00"),
        SpeakerTurn(1.5, 3.0, "SPEAKER_01"),
    ]


def test_extract_profiles_creates_one_reference_per_speaker(
    fake_diarizer,
    source_audio,
    tmp_path,
):
    turns = fake_diarizer.diarize(source_audio)

    profiles = fake_diarizer.extract_profiles(
        source_audio,
        turns,
        tmp_path / "profiles",
    )

    assert set(profiles) == {"SPEAKER_00", "SPEAKER_01"}

    for profile in profiles.values():
        assert profile.reference_wav.is_file()
        assert profile.reference_wav.stat().st_size > 0
        assert profile.duration_seconds > 0


def test_load_translation_json(translation_json):
    (
        source_language,
        target_language,
        segments,
    ) = VoiceGenerator._load_translation_file(translation_json)

    assert source_language == "eng_Latn"
    assert target_language == "hin_Deva"
    assert len(segments) == 2
    assert segments[0].text == "नमस्ते।"


def test_generate_missing_source_audio_raises_invalid_audio_error(
    voice_generator,
    translation_json,
    tmp_path,
):
    with pytest.raises(InvalidAudioError, match="Source audio file"):
        voice_generator.generate(
            translation_json,
            tmp_path / "missing.wav",
            tmp_path / "output",
        )


def test_missing_translation_json_raises_error(tmp_path):
    with pytest.raises(InvalidTranslationFileError):
        VoiceGenerator._load_translation_file(
            tmp_path / "missing.json"
        )


def test_generate_creates_audio_for_each_speaker(
    voice_generator,
    source_audio,
    translation_json,
    tmp_path,
):
    output_dir = tmp_path / "voice_output"

    result = voice_generator.generate(
        translation_json,
        source_audio,
        output_dir,
    )

    assert len(result.speaker_profiles) == 2
    assert len(result.segments) == 2

    assert result.segments[0].speaker_id == "SPEAKER_00"
    assert result.segments[1].speaker_id == "SPEAKER_01"

    for segment in result.segments:
        assert segment.audio_path.is_file()
        assert segment.audio_path.stat().st_size > 0
        assert segment.duration_seconds == pytest.approx(0.8, abs=0.01)

    assert result.manifest_path.is_file()

    manifest = json.loads(
        result.manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["target_language"] == "hin_Deva"
    assert manifest["alignment_pending"] is True
    assert len(manifest["speakers"]) == 2
    assert len(manifest["segments"]) == 2


def test_generate_uses_matching_speaker_reference(
    voice_generator,
    fake_synthesizer,
    source_audio,
    translation_json,
    tmp_path,
):
    voice_generator.generate(
        translation_json,
        source_audio,
        tmp_path / "output",
    )

    assert len(fake_synthesizer.calls) == 2

    assert (
        fake_synthesizer.calls[0]["speaker_wav"]
        != fake_synthesizer.calls[1]["speaker_wav"]
    )

    assert all(
        call["reference_text"].strip()
        for call in fake_synthesizer.calls
    )


def test_generate_uses_provided_reference_transcripts(
    voice_generator,
    fake_synthesizer,
    source_audio,
    translation_json,
    tmp_path,
):
    voice_generator.generate(
        translation_json,
        source_audio,
        tmp_path / "output",
        reference_texts={
            "SPEAKER_00": "Hello, how are you?",
            "SPEAKER_01": "Welcome to the presentation.",
        },
    )

    assert [
        call["reference_text"]
        for call in fake_synthesizer.calls
    ] == [
        "Hello, how are you?",
        "Welcome to the presentation.",
    ]


def test_generated_audio_is_not_time_stretched(
    voice_generator,
    source_audio,
    translation_json,
    tmp_path,
):
    result = voice_generator.generate(
        translation_json,
        source_audio,
        tmp_path / "output",
    )

    for segment in result.segments:
        info = sf.info(str(segment.audio_path))

        assert info.duration == pytest.approx(
            segment.duration_seconds,
            abs=0.01,
        )


def test_segment_speaker_selection_by_overlap():
    segment = SimpleNamespace(
        start=1.0,
        end=2.0,
        text="Hello",
        speaker_id=None,
    )

    turns = [
        SpeakerTurn(0.0, 1.3, "SPEAKER_00"),
        SpeakerTurn(1.3, 2.8, "SPEAKER_01"),
    ]

    profiles = {
        "SPEAKER_00": SimpleNamespace(),
        "SPEAKER_01": SimpleNamespace(),
    }

    selected = VoiceGenerator._select_speaker(
        segment,
        turns,
        profiles,
    )

    assert selected == "SPEAKER_01"


def test_explicit_speaker_id_is_respected():
    segment = SimpleNamespace(
        start=0.0,
        end=1.0,
        text="Hello",
        speaker_id="SPEAKER_01",
    )

    turns = [
        SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
    ]

    profiles = {
        "SPEAKER_00": SimpleNamespace(),
        "SPEAKER_01": SimpleNamespace(),
    }

    selected = VoiceGenerator._select_speaker(
        segment,
        turns,
        profiles,
    )

    assert selected == "SPEAKER_01"


def test_invalid_timestamp_in_translation_json(tmp_path):
    path = tmp_path / "invalid.json"

    path.write_text(
        json.dumps(
            {
                "source_language": "en",
                "target_language": "hi",
                "segments": [
                    {
                        "start": 2.0,
                        "end": 1.0,
                        "text": "Invalid segment",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidTranslationFileError):
        VoiceGenerator._load_translation_file(path)


def test_f5_synthesizer_calls_infer_and_saves_audio(
    source_audio,
    tmp_path,
):
    class FakeF5Model:
        def __init__(self):
            self.call_kwargs = None

        def infer(self, **kwargs):
            self.call_kwargs = kwargs

            output = Path(kwargs["file_wave"])
            sample_rate = 22050
            audio = np.zeros(sample_rate, dtype=np.float32)

            sf.write(
                str(output),
                audio,
                sample_rate,
                subtype="PCM_16",
            )

            return audio, sample_rate, None

    fake_model = FakeF5Model()

    synthesizer = F5TTSSynthesizer(
        model_name="F5TTS_v1_Base",
        device="cpu",
        model_instance=fake_model,
    )

    output = tmp_path / "generated.wav"

    result = synthesizer.synthesize(
        text="Hello there.",
        speaker_wav=source_audio,
        output_path=output,
        reference_text="This is the speaker's original speech.",
    )

    assert result == output
    assert output.is_file()

    assert fake_model.call_kwargs["ref_file"] == str(source_audio)
    assert (
        fake_model.call_kwargs["ref_text"]
        == "This is the speaker's original speech."
    )
    assert fake_model.call_kwargs["gen_text"] == "Hello there."
    assert fake_model.call_kwargs["file_wave"] == str(output)


def test_f5_synthesizer_rejects_empty_reference_text(
    source_audio,
    tmp_path,
):
    from video_translator.voice_generator import VoiceSynthesisError

    fake_model = Mock()

    synthesizer = F5TTSSynthesizer(
        model_instance=fake_model,
        device="cpu",
    )

    with pytest.raises(
        VoiceSynthesisError,
        match="reference_text cannot be empty",
    ):
        synthesizer.synthesize(
            text="Generated text.",
            speaker_wav=source_audio,
            output_path=tmp_path / "generated.wav",
        )

    fake_model.infer.assert_not_called()


def test_missing_audio_reference_raises_error(tmp_path):
    from video_translator.voice_generator import VoiceSynthesisError

    synthesizer = F5TTSSynthesizer(
        model_instance=Mock(),
        device="cpu",
    )

    with pytest.raises(VoiceSynthesisError, match="reference audio"):
        synthesizer.synthesize(
            "Hello",
            tmp_path / "missing.wav",
            tmp_path / "generated.wav",
        )


def test_invalid_reference_text_type_raises_error(source_audio, tmp_path):
    synthesizer = F5TTSSynthesizer(
        model_instance=Mock(),
        device="cpu",
    )

    with pytest.raises(TypeError, match="reference_text"):
        synthesizer.synthesize(
            "Hello",
            source_audio,
            tmp_path / "generated.wav",
            reference_text=None,
        )