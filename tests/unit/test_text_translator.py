from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from video_translator.text_translator import (
    CudaUnavailableError,
    DEFAULT_MODEL,
    ModelLoadError,
    TextSegment,
    TextSegmentError,
    TextTranslator,
    TranslatedTextSegment,
    TranslationError,
    UnsupportedLanguageError,
    merge_text_segments,
)


class FakeTokenizer:
    def __init__(self):
        self.src_lang = None
        self.unk_token_id = 0
        self.last_texts = []
        self.decoded_texts = []
        self.token_ids = {
            "eng_Latn": 10,
            "hin_Deva": 11,
            "fra_Latn": 12,
            "spa_Latn": 13,
        }

    def convert_tokens_to_ids(self, token):
        return self.token_ids.get(token, self.unk_token_id)

    def __call__(
        self,
        texts,
        return_tensors,
        padding,
        truncation,
        max_length,
    ):
        self.last_texts = list(texts)

        return {
            "input_ids": torch.ones(
                (len(texts), 4),
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                (len(texts), 4),
                dtype=torch.long,
            ),
        }

    def batch_decode(self, generated, skip_special_tokens):
        return self.decoded_texts[: len(generated)]


class FakeModel:
    def __init__(self):
        self.generate = Mock(side_effect=self._generate)
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def _generate(self, input_ids, attention_mask, **kwargs):
        self.generation_kwargs = kwargs

        return torch.ones(
            (input_ids.shape[0], 3),
            dtype=torch.long,
        )


@pytest.fixture
def fake_components():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    return tokenizer, model


@pytest.fixture
def translator(fake_components):
    tokenizer, model = fake_components

    return TextTranslator(
        device="cpu",
        model_instance=model,
        tokenizer_instance=tokenizer,
    )


def test_default_model_and_device():
    translator = TextTranslator()

    assert translator.model_name == DEFAULT_MODEL
    assert translator.device == "cuda"


def test_resolve_language_codes():
    assert TextTranslator.resolve_language("en") == "eng_Latn"
    assert TextTranslator.resolve_language("English") == "eng_Latn"
    assert TextTranslator.resolve_language("hi") == "hin_Deva"
    assert TextTranslator.resolve_language("Hindi") == "hin_Deva"
    assert TextTranslator.resolve_language("fra_Latn") == "fra_Latn"


@pytest.mark.parametrize(
    "language",
    ["", "unknown", "auto", "xx"],
)
def test_unsupported_language(language):
    with pytest.raises(UnsupportedLanguageError):
        TextTranslator.resolve_language(language)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        TextTranslator(device="mps")

    with pytest.raises(ValueError):
        TextTranslator(num_beams=0)

    with pytest.raises(ValueError):
        TextTranslator(batch_size=0)

    with pytest.raises(ValueError):
        TextTranslator(max_input_tokens=0)


def test_model_and_tokenizer_must_be_supplied_together():
    with pytest.raises(ValueError, match="supplied together"):
        TextTranslator(model_instance=FakeModel())


def test_merge_choppy_segments():
    segments = [
        TextSegment(0.0, 0.5, "The quick"),
        TextSegment(0.6, 1.0, "brown fox"),
        TextSegment(1.1, 1.5, "jumps over"),
        TextSegment(1.6, 2.0, "the dog."),
    ]

    merged = merge_text_segments(segments)

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 2.0
    assert merged[0].text == (
        "The quick brown fox jumps over the dog."
    )


def test_merge_respects_sentence_boundaries():
    segments = [
        TextSegment(0.0, 1.0, "This is a complete sentence."),
        TextSegment(1.1, 2.0, "This is another complete sentence."),
    ]

    merged = merge_text_segments(segments)

    assert len(merged) == 2


def test_merge_allows_short_complete_segments():
    segments = [
        TextSegment(0.0, 0.3, "Hello."),
        TextSegment(0.4, 0.8, "How are you?"),
    ]

    merged = merge_text_segments(segments)

    assert len(merged) == 1
    assert merged[0].text == "Hello. How are you?"


def test_merge_respects_long_pauses():
    segments = [
        TextSegment(0.0, 0.5, "Hello"),
        TextSegment(2.0, 2.5, "there."),
    ]

    merged = merge_text_segments(
        segments,
        max_gap_seconds=0.5,
    )

    assert len(merged) == 2


def test_merge_respects_speaker_changes():
    segments = [
        TextSegment(0.0, 0.5, "Hello", speaker_id="speaker_1"),
        TextSegment(0.6, 1.0, "Hi", speaker_id="speaker_2"),
    ]

    merged = merge_text_segments(segments)

    assert len(merged) == 2


def test_merge_respects_maximum_duration():
    segments = [
        TextSegment(0.0, 5.0, "This is a long fragment"),
        TextSegment(5.1, 11.0, "and this is another fragment"),
    ]

    merged = merge_text_segments(
        segments,
        max_duration_seconds=10.0,
    )

    assert len(merged) == 2


def test_merge_respects_character_limit():
    segments = [
        TextSegment(0.0, 0.5, "One two three"),
        TextSegment(0.6, 1.0, "four five six"),
    ]

    merged = merge_text_segments(
        segments,
        max_characters=15,
    )

    assert len(merged) == 2


def test_merge_sorts_segments_by_start_time():
    segments = [
        TextSegment(1.0, 1.5, "world"),
        TextSegment(0.0, 0.5, "Hello"),
    ]

    merged = merge_text_segments(segments)

    assert merged[0].start == 0.0
    assert merged[0].text == "Hello world"


def test_merge_empty_collection():
    assert merge_text_segments([]) == []


def test_merge_discards_empty_text_segments():
    segments = [
        TextSegment(0.0, 0.2, " "),
        TextSegment(0.3, 0.8, "Hello there."),
    ]

    merged = merge_text_segments(segments)

    assert len(merged) == 1
    assert merged[0].text == "Hello there."


def test_merge_rejects_invalid_segment():
    segments = [
        SimpleNamespace(start=1.0, end=0.0, text="Invalid"),
    ]

    with pytest.raises(TextSegmentError):
        merge_text_segments(segments)


def test_translate_text_uses_nllb(
    translator,
    fake_components,
):
    tokenizer, model = fake_components
    tokenizer.decoded_texts = ["नमस्ते दुनिया"]

    translated = translator.translate_text(
        "Hello world",
        source_language="en",
        target_language="hi",
    )

    assert translated == "नमस्ते दुनिया"
    assert tokenizer.src_lang == "eng_Latn"
    assert tokenizer.last_texts == ["Hello world"]

    assert model.generate.call_count == 1
    assert model.generate.call_args.kwargs[
        "forced_bos_token_id"
    ] == 11


def test_translate_segments_preserves_timestamps(
    translator,
    fake_components,
):
    tokenizer, _ = fake_components
    tokenizer.decoded_texts = ["नमस्ते दुनिया"]

    segments = [
        TextSegment(1.0, 1.5, "Hello"),
        TextSegment(1.6, 2.0, "world."),
    ]

    result = translator.translate_segments(
        segments,
        source_language="en",
        target_language="hi",
    )

    assert result.source_language == "eng_Latn"
    assert result.target_language == "hin_Deva"
    assert len(result.segments) == 1

    translated = result.segments[0]

    assert isinstance(translated, TranslatedTextSegment)
    assert translated.start == 1.0
    assert translated.end == 2.0
    assert translated.source_text == "Hello world."
    assert translated.translated_text == "नमस्ते दुनिया"
    assert result.text == "नमस्ते दुनिया"


def test_translate_segments_preserves_speaker_id(
    translator,
    fake_components,
):
    tokenizer, _ = fake_components
    tokenizer.decoded_texts = ["नमस्ते"]

    result = translator.translate_segments(
        [
            TextSegment(
                0.0,
                1.0,
                "Hello.",
                speaker_id="speaker_1",
            )
        ],
        "en",
        "hi",
    )

    assert result.segments[0].speaker_id == "speaker_1"


def test_translate_segments_supports_speech_recognizer_objects(
    translator,
    fake_components,
):
    tokenizer, _ = fake_components
    tokenizer.decoded_texts = ["नमस्ते"]

    recognizer_segments = [
        SimpleNamespace(
            start=0.0,
            end=1.0,
            text="Hello.",
        )
    ]

    result = translator.translate_segments(
        recognizer_segments,
        "en",
        "hi",
    )

    assert result.segments[0].source_text == "Hello."
    assert result.segments[0].translated_text == "नमस्ते"


def test_translate_segments_batches_requests(
    fake_components,
):
    tokenizer, model = fake_components

    translator = TextTranslator(
        device="cpu",
        batch_size=2,
        merge_segments=False,
        model_instance=model,
        tokenizer_instance=tokenizer,
    )

    tokenizer.batch_decode = Mock(
        side_effect=[
            ["Hola", "Buenos días"],
            ["Adiós"],
        ]
    )

    segments = [
        TextSegment(0.0, 0.5, "Hello"),
        TextSegment(1.0, 1.5, "Good morning"),
        TextSegment(2.0, 2.5, "Goodbye"),
    ]

    result = translator.translate_segments(
        segments,
        "en",
        "es",
    )

    assert [segment.translated_text for segment in result.segments] == [
        "Hola",
        "Buenos días",
        "Adiós",
    ]

    assert model.generate.call_count == 2


def test_same_source_and_target_language_skips_model(
    translator,
    fake_components,
):
    _, model = fake_components

    result = translator.translate_segments(
        [TextSegment(0.0, 1.0, "Hello there.")],
        "en",
        "eng_Latn",
    )

    assert result.segments[0].translated_text == "Hello there."
    model.generate.assert_not_called()


def test_empty_segments_skip_model(
    translator,
    fake_components,
):
    _, model = fake_components

    result = translator.translate_segments([], "en", "hi")

    assert result.segments == ()
    model.generate.assert_not_called()


def test_translate_text_empty_string_skips_model(
    translator,
    fake_components,
):
    _, model = fake_components

    assert translator.translate_text("  ", "en", "hi") == ""
    model.generate.assert_not_called()


def test_generation_error_is_wrapped(
    translator,
    fake_components,
):
    _, model = fake_components
    model.generate.side_effect = RuntimeError("Generation failed")

    with pytest.raises(TranslationError, match="translation failed"):
        translator.translate_text("Hello", "en", "hi")


def test_cuda_unavailable_raises_error(monkeypatch):
    import sys
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = Mock()
    fake_transformers.AutoModelForSeq2SeqLM = Mock()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        fake_transformers,
    )

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    translator = TextTranslator(device="cuda")

    with pytest.raises(CudaUnavailableError):
        translator._load_model()


def test_model_load_failure_is_wrapped(monkeypatch):
    import sys
    import types

    fake_transformers = types.ModuleType("transformers")

    fake_tokenizer_class = Mock()
    fake_tokenizer_class.from_pretrained.side_effect = RuntimeError(
        "Download failed"
    )

    fake_model_class = Mock()

    fake_transformers.AutoTokenizer = fake_tokenizer_class
    fake_transformers.AutoModelForSeq2SeqLM = fake_model_class

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        fake_transformers,
    )

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    translator = TextTranslator(device="cuda")

    with pytest.raises(
        ModelLoadError,
        match="Unable to load",
    ):
        translator._load_model()

    fake_tokenizer_class.from_pretrained.assert_called_once_with(
        "facebook/nllb-200-1.3B"
    )
    fake_model_class.from_pretrained.assert_not_called()