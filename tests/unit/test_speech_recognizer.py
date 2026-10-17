import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from video_translator.speech_recognizer import (
    DEFAULT_MODEL_SIZE,
    FasterWhisperNotFoundError,
    InvalidAudioError,
    ModelLoadError,
    SpeechRecognizer,
    TranscriptSegment,
    TranscriptionError,
)


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"mock audio content")
    return path


@pytest.fixture
def fake_model() -> Mock:
    model = Mock()

    model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    start=0.25,
                    end=1.75,
                    text=" Hello there.",
                ),
                SimpleNamespace(
                    start=2.0,
                    end=3.5,
                    text=" How are you?",
                ),
            ]
        ),
        SimpleNamespace(
            language="en",
            language_probability=0.98,
            duration=5.0,
        ),
    )

    return model


@pytest.fixture
def recognizer(fake_model) -> SpeechRecognizer:
    return SpeechRecognizer(model_instance=fake_model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_size": ""},
        {"device": ""},
        {"compute_type": ""},
        {"beam_size": 0},
        {"beam_size": -1},
    ],
)
def test_invalid_configuration_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        SpeechRecognizer(**kwargs)


def test_default_configuration():
    recognizer = SpeechRecognizer()

    assert recognizer.model_size == DEFAULT_MODEL_SIZE
    assert recognizer.beam_size == 5
    assert recognizer.vad_filter is False


def test_transcribe_returns_result(
    recognizer,
    fake_model,
    audio_file,
):
    result = recognizer.transcribe(audio_file)

    assert result.audio_path == audio_file
    assert result.language == "en"
    assert result.language_probability == pytest.approx(0.98)
    assert result.duration_seconds == 5.0

    assert result.segments == (
        TranscriptSegment(
            start=0.25,
            end=1.75,
            text="Hello there.",
        ),
        TranscriptSegment(
            start=2.0,
            end=3.5,
            text="How are you?",
        ),
    )

    assert result.text == "Hello there. How are you?"


def test_transcribe_passes_options_to_faster_whisper(
    recognizer,
    fake_model,
    audio_file,
):
    recognizer.transcribe(audio_file, language="en")

    fake_model.transcribe.assert_called_once_with(
        str(audio_file),
        language="en",
        task="transcribe",
        beam_size=5,
        vad_filter=False,
        vad_parameters=None,
        word_timestamps=False,
        condition_on_previous_text=True,
    )


def test_transcribe_supports_vad_and_word_timestamps(
    fake_model,
    audio_file,
):
    recognizer = SpeechRecognizer(
        model_instance=fake_model,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=False,
        beam_size=3,
    )

    recognizer.transcribe(audio_file)

    fake_model.transcribe.assert_called_once_with(
        str(audio_file),
        language=None,
        task="transcribe",
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=False,
    )


def test_transcribe_uses_detected_language(
    recognizer,
    fake_model,
    audio_file,
):
    recognizer.transcribe(audio_file)

    assert fake_model.transcribe.call_args.kwargs["language"] is None


def test_transcribe_with_explicit_language(
    recognizer,
    fake_model,
    audio_file,
):
    recognizer.transcribe(audio_file, language="fr")

    assert (
        fake_model.transcribe.call_args.kwargs["language"]
        == "fr"
    )


def test_transcribe_rejects_empty_language(
    recognizer,
    audio_file,
):
    with pytest.raises(ValueError, match="language"):
        recognizer.transcribe(audio_file, language=" ")


def test_transcribe_missing_file(
    recognizer,
    tmp_path,
):
    with pytest.raises(InvalidAudioError, match="does not exist"):
        recognizer.transcribe(tmp_path / "missing.wav")


def test_transcribe_directory(
    recognizer,
    tmp_path,
):
    with pytest.raises(InvalidAudioError, match="not a file"):
        recognizer.transcribe(tmp_path)


def test_transcribe_empty_file(
    recognizer,
    tmp_path,
):
    path = tmp_path / "empty.wav"
    path.touch()

    with pytest.raises(InvalidAudioError, match="empty"):
        recognizer.transcribe(path)


def test_transcribe_empty_result(
    recognizer,
    fake_model,
    audio_file,
):
    fake_model.transcribe.return_value = (
        iter([]),
        SimpleNamespace(
            language="en",
            language_probability=0.9,
            duration=3.0,
        ),
    )

    result = recognizer.transcribe(audio_file)

    assert result.segments == ()
    assert result.text == ""
    assert result.duration_seconds == 3.0


def test_transcribe_uses_last_segment_when_duration_missing(
    recognizer,
    fake_model,
    audio_file,
):
    fake_model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    start=1.0,
                    end=4.0,
                    text="Test",
                )
            ]
        ),
        SimpleNamespace(
            language="en",
            language_probability=0.9,
            duration=None,
        ),
    )

    result = recognizer.transcribe(audio_file)

    assert result.duration_seconds == 4.0


def test_transcribe_wraps_model_failure(
    recognizer,
    fake_model,
    audio_file,
):
    fake_model.transcribe.side_effect = RuntimeError(
        "Model inference failed"
    )

    with pytest.raises(TranscriptionError, match="Transcription failed"):
        recognizer.transcribe(audio_file)


@pytest.mark.parametrize(
    "segment",
    [
        SimpleNamespace(start=-1.0, end=1.0, text="Invalid"),
        SimpleNamespace(start=2.0, end=1.0, text="Invalid"),
        SimpleNamespace(start=0.0, end=float("nan"), text="Invalid"),
        SimpleNamespace(start=0.0, end=1.0, text=None),
        SimpleNamespace(start=0.0, text="Invalid"),
    ],
)
def test_invalid_segment_raises_error(segment):
    with pytest.raises(TranscriptionError):
        SpeechRecognizer._parse_segment(segment)


def test_transcribe_invalid_language_probability(
    recognizer,
    fake_model,
    audio_file,
):
    fake_model.transcribe.return_value = (
        iter([]),
        SimpleNamespace(
            language="en",
            language_probability=1.5,
            duration=2.0,
        ),
    )

    with pytest.raises(
        TranscriptionError,
        match="language probability",
    ):
        recognizer.transcribe(audio_file)


def test_model_load_is_cached(
    audio_file,
    monkeypatch,
):
    whisper_model = Mock()
    factory = Mock(return_value=whisper_model)

    monkeypatch.setattr(
        "video_translator.speech_recognizer.recognizer.WhisperModel",
        factory,
        raising=False,
    )

    recognizer = SpeechRecognizer(model_instance=whisper_model)

    assert recognizer._load_model() is whisper_model
    assert recognizer._load_model() is whisper_model
    factory.assert_not_called()


def test_model_load_error_is_wrapped(monkeypatch):
    import sys
    import types

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = Mock(
        side_effect=RuntimeError("Model loading failed")
    )

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        fake_module,
    )

    recognizer = SpeechRecognizer()

    with pytest.raises(ModelLoadError, match="Unable to load"):
        recognizer._load_model()


def test_missing_faster_whisper_raises_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("Faster-Whisper unavailable")

        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(FasterWhisperNotFoundError):
        SpeechRecognizer()._load_model()