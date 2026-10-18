from __future__ import annotations

import importlib
import math
import re
from typing import Any, Sequence

from video_translator.text_translator.segments import (
    TextSegment,
    TranslatedTextSegment,
    TranslationResult,
    merge_text_segments,
    normalize_text_segments,
)


DEFAULT_MODEL = "facebook/nllb-200-1.3B"
DEFAULT_DEVICE = "cuda"
DEFAULT_NUM_BEAMS = 4
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 512

LANGUAGE_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "english": "eng_Latn",
    "es": "spa_Latn",
    "spanish": "spa_Latn",
    "fr": "fra_Latn",
    "french": "fra_Latn",
    "de": "deu_Latn",
    "german": "deu_Latn",
    "it": "ita_Latn",
    "italian": "ita_Latn",
    "pt": "por_Latn",
    "portuguese": "por_Latn",
    "pl": "pol_Latn",
    "polish": "pol_Latn",
    "tr": "tur_Latn",
    "turkish": "tur_Latn",
    "ru": "rus_Cyrl",
    "russian": "rus_Cyrl",
    "nl": "nld_Latn",
    "dutch": "nld_Latn",
    "cs": "ces_Latn",
    "czech": "ces_Latn",
    "ar": "arb_Arab",
    "arabic": "arb_Arab",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "chinese": "zho_Hans",
    "ja": "jpn_Jpan",
    "japanese": "jpn_Jpan",
    "ko": "kor_Hang",
    "korean": "kor_Hang",
    "hi": "hin_Deva",
    "hindi": "hin_Deva",
    "bn": "ben_Beng",
    "bengali": "ben_Beng",
    "ur": "urd_Arab",
    "urdu": "urd_Arab",
    "ta": "tam_Taml",
    "tamil": "tam_Taml",
    "te": "tel_Telu",
    "telugu": "tel_Telu",
    "mr": "mar_Deva",
    "marathi": "mar_Deva",
    "pa": "pan_Guru",
    "punjabi": "pan_Guru",
}


class TextTranslatorError(Exception):
    """Base exception for text translation errors."""


class TransformersNotFoundError(TextTranslatorError):
    """Raised when Transformers or PyTorch is unavailable."""


class CudaUnavailableError(TextTranslatorError):
    """Raised when CUDA is required but unavailable."""


class ModelLoadError(TextTranslatorError):
    """Raised when the translation model cannot be loaded."""


class UnsupportedLanguageError(TextTranslatorError):
    """Raised when a language has no recognised NLLB language code."""


class TranslationError(TextTranslatorError):
    """Raised when translation fails."""


class TextTranslator:
    """Translate timestamped speech segments using local NLLB on CUDA."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        num_beams: int = DEFAULT_NUM_BEAMS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        merge_segments: bool = True,
        max_gap_seconds: float = 0.75,
        max_merged_duration_seconds: float = 12.0,
        max_merged_characters: int = 500,
        short_segment_words: int = 3,
        model_instance: Any | None = None,
        tokenizer_instance: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")

        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError(
                "device must be 'cuda', 'cuda:0', or 'cpu'."
            )

        if num_beams < 1:
            raise ValueError("num_beams must be positive.")

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("Token limits must be positive.")

        if (model_instance is None) != (tokenizer_instance is None):
            raise ValueError(
                "model_instance and tokenizer_instance "
                "must be supplied together."
            )

        self.model_name = model_name
        self.device = device
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        self.merge_segments = merge_segments
        self.max_gap_seconds = max_gap_seconds
        self.max_merged_duration_seconds = (
            max_merged_duration_seconds
        )
        self.max_merged_characters = max_merged_characters
        self.short_segment_words = short_segment_words

        self._model = model_instance
        self._tokenizer = tokenizer_instance

    @staticmethod
    def resolve_language(language: str) -> str:
        """Resolve a common language name/code to an NLLB FLORES code."""
        if not isinstance(language, str) or not language.strip():
            raise UnsupportedLanguageError(
                "A source and target language must be provided."
            )

        normalized = language.strip()

        if normalized.lower() in LANGUAGE_CODES:
            return LANGUAGE_CODES[normalized.lower()]

        if re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", normalized):
            return normalized

        raise UnsupportedLanguageError(
            f"Unsupported language: {language!r}. "
            "Provide a supported language name, ISO code, "
            "or NLLB FLORES-200 code."
        )

    def _load_model(self) -> tuple[Any, Any]:
        """Load the tokenizer and model once, directly onto the device."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TransformersNotFoundError(
                "Install the translation dependencies using: "
                "poetry add torch transformers sentencepiece"
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise CudaUnavailableError(
                "CUDA is unavailable. Install a CUDA-enabled PyTorch build "
                "and verify that an NVIDIA GPU is accessible."
            )

        try:
            dtype = (
                torch.float16
                if self.device.startswith("cuda")
                else torch.float32
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self.device)

            self._model.eval()

        except Exception as exc:
            self._model = None
            self._tokenizer = None

            raise ModelLoadError(
                f"Unable to load translation model: {self.model_name}"
            ) from exc

        return self._model, self._tokenizer

    def _translate_batch(
        self,
        texts: Sequence[str],
        source_code: str,
        target_code: str,
    ) -> list[str]:
        """Translate a batch of text using the loaded NLLB model."""
        if not texts:
            return []

        model, tokenizer = self._load_model()

        try:
            import torch

            tokenizer.src_lang = source_code

            target_token_id = tokenizer.convert_tokens_to_ids(
                target_code
            )

            unknown_token_id = getattr(
                tokenizer,
                "unk_token_id",
                None,
            )

            if (
                target_token_id is None
                or (
                    unknown_token_id is not None
                    and target_token_id == unknown_token_id
                )
            ):
                raise UnsupportedLanguageError(
                    f"NLLB does not recognise target language "
                    f"code: {target_code}"
                )

            encoded = tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=target_token_id,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                )

            translations = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            if len(translations) != len(texts):
                raise TranslationError(
                    "The translation model returned an unexpected "
                    "number of translations."
                )

            results = []

            for translation in translations:
                if not isinstance(translation, str):
                    raise TranslationError(
                        "The translation model returned non-text output."
                    )

                results.append(translation.strip())

            return results

        except TextTranslatorError:
            raise

        except Exception as exc:
            raise TranslationError(
                "NLLB translation failed."
            ) from exc

    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate a single string."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text.strip():
            return ""

        source_code = self.resolve_language(source_language)
        target_code = self.resolve_language(target_language)

        if source_code == target_code:
            return text.strip()

        translations = self._translate_batch(
            [text.strip()],
            source_code,
            target_code,
        )

        return translations[0]

    def translate_segments(
        self,
        segments: Sequence[Any],
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Merge transcript fragments and translate them while preserving timing.

        Accepts TextSegment objects or compatible objects such as
        TranscriptSegment returned by SpeechRecognizer.
        """
        source_code = self.resolve_language(source_language)
        target_code = self.resolve_language(target_language)

        normalized_segments = [
            TextSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text,
                speaker_id=getattr(segment, "speaker_id", None),
            )
            for segment in segments
        ]

        if self.merge_segments:
            prepared_segments = merge_text_segments(
                segments,
                max_gap_seconds=self.max_gap_seconds,
                max_duration_seconds=self.max_merged_duration_seconds,
                max_characters=self.max_merged_characters,
                short_segment_words=self.short_segment_words,
            )
        else:
            prepared_segments = normalize_text_segments(segments)

        if not prepared_segments:
            return TranslationResult(
                source_language=source_code,
                target_language=target_code,
                segments=(),
            )

        if source_code == target_code:
            translations = [
                segment.text
                for segment in prepared_segments
            ]
        else:
            texts = [
                segment.text
                for segment in prepared_segments
            ]

            translations = []

            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset : offset + self.batch_size]

                translations.extend(
                    self._translate_batch(
                        batch,
                        source_code,
                        target_code,
                    )
                )

        translated_segments = tuple(
            TranslatedTextSegment(
                start=segment.start,
                end=segment.end,
                source_text=segment.text,
                translated_text=translation,
                speaker_id=segment.speaker_id,
            )
            for segment, translation in zip(
                prepared_segments,
                translations,
                strict=True,
            )
        )

        return TranslationResult(
            source_language=source_code,
            target_language=target_code,
            segments=translated_segments,
        )