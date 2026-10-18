import os

import pytest
import torch

from video_translator.text_translator import (
    DEFAULT_MODEL,
    TextSegment,
    TextTranslator,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NLLB_INTEGRATION") != "1",
    reason="Set RUN_NLLB_INTEGRATION=1 to download/load NLLB-200 1.3B.",
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="NLLB integration test requires a CUDA-enabled GPU.",
)
def test_translate_english_to_hindi_on_cuda():
    translator = TextTranslator(
        model_name=DEFAULT_MODEL,
        device="cuda",
        num_beams=2,
        batch_size=2,
    )

    segments = [
        TextSegment(0.0, 0.5, "Hello"),
        TextSegment(0.6, 1.2, "everyone."),
        TextSegment(1.5, 2.5, "Welcome to our presentation."),
    ]

    result = translator.translate_segments(
        segments,
        source_language="en",
        target_language="hi",
    )

    assert result.source_language == "eng_Latn"
    assert result.target_language == "hin_Deva"
    assert len(result.segments) >= 1
    assert result.text.strip()

    for segment in result.segments:
        assert segment.start >= 0
        assert segment.end > segment.start
        assert segment.source_text.strip()
        assert segment.translated_text.strip()