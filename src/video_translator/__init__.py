"""
Video Translator - A pipeline for translating video audio tracks.

This package provides modular components for:
- Media processing (video info, audio extraction)
- Speech detection (Silero VAD)
- Speech recognition (Faster-Whisper)
- Text translation (NLLB via Transformers)
- Audio separation (Demucs via audio-separator)
- Voice generation (F5-TTS, Piper with voice conversion)
- Audio alignment (Rubber Band time-stretching, EQ matching)
- Video rendering (FFmpeg)
"""

from video_translator.audio_aligner import (
    AlignedSpeechSegment,
    AudioAligner,
    AudioAlignerConfig,
    AudioAlignmentError,
    AudioAlignmentResult,
)

from video_translator.audio_separator import (
    AudioSeparationResult,
    AudioSeparator,
    AudioSeparatorError,
    AudioSeparatorNotFoundError,
    InvalidAudioError,
    OutputAlreadyExistsError,
)

from video_translator.media_processor import (
    FFmpegNotFoundError,
    InvalidMediaError,
    MediaInfo,
    MediaProcessor,
    MediaProcessorError,
    NoAudioStreamError,
)

from video_translator.speech_detector import (
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_MIN_SPEECH_DURATION_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEECH_PAD_MS,
    DEFAULT_THRESHOLD,
    InvalidAudioError as SpeechDetectorInvalidAudioError,
    SileroVADNotFoundError,
    SpeechDetector,
    SpeechDetectorError,
    SpeechRegion,
)

from video_translator.speech_recognizer import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_SIZE,
    FasterWhisperNotFoundError,
    InvalidAudioError as SpeechRecognizerInvalidAudioError,
    ModelLoadError,
    SpeechRecognizer,
    SpeechRecognizerError,
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)

from video_translator.text_translator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_MERGED_CHARACTERS,
    DEFAULT_MAX_MERGED_DURATION_SECONDS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_NUM_BEAMS,
    DEFAULT_SHORT_SEGMENT_WORDS,
    CudaUnavailableError,
    ModelLoadError as TextTranslatorModelLoadError,
    TextSegment,
    TextSegmentError,
    TextTranslator,
    TextTranslatorError,
    TranslatedTextSegment,
    TranslationError,
    TranslationResult,
    TransformersNotFoundError,
    UnsupportedLanguageError,
    merge_text_segments,
    normalize_text_segments,
)

from video_translator.video_renderer import (
    VideoRenderer,
    VideoRendererConfig,
    VideoRendererError,
    VideoRenderResult,
)

from video_translator.voice_generator import (
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_HINDI_TTS_MODEL,
    DEFAULT_NUM_SPEAKER_PROFILE_SECONDS,
    DEFAULT_PIPER_FEMALE_VOICE,
    DEFAULT_PIPER_MALE_VOICE,
    DEFAULT_PIPER_VOICE_PITCH_THRESHOLD_HZ,
    DEFAULT_TTS_CKPT_FILE,
    DEFAULT_TTS_DEVICE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOCAB_FILE,
    DiarizationError,
    F5TTSModelLoadError,
    F5TTSNotFoundError,
    F5TTSSynthesizer,
    GeneratedSpeechSegment,
    HuggingFaceTokenError,
    InvalidAudioError as VoiceGeneratorInvalidAudioError,
    InvalidTranslationFileError,
    NoSpeakersDetectedError,
    PiperTTSSynthesizer,
    PyannoteNotFoundError,
    PyannoteSpeakerDiarizer,
    SpeakerProfile,
    SpeakerTurn,
    TranslatedSegment,
    UnsupportedTTSConfigurationError,
    VoiceGenerationResult,
    VoiceGenerator,
    VoiceGeneratorError,
    VoiceSynthesisError,
)

__all__ = [
    # Audio Aligner
    "AlignedSpeechSegment",
    "AudioAligner",
    "AudioAlignerConfig",
    "AudioAlignmentError",
    "AudioAlignmentResult",
    # Audio Separator
    "AudioSeparationResult",
    "AudioSeparator",
    "AudioSeparatorError",
    "AudioSeparatorNotFoundError",
    "InvalidAudioError",
    "OutputAlreadyExistsError",
    # Media Processor
    "FFmpegNotFoundError",
    "InvalidMediaError",
    "MediaInfo",
    "MediaProcessor",
    "MediaProcessorError",
    "NoAudioStreamError",
    # Speech Detector
    "DEFAULT_MIN_SILENCE_DURATION_MS",
    "DEFAULT_MIN_SPEECH_DURATION_MS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SPEECH_PAD_MS",
    "DEFAULT_THRESHOLD",
    "SileroVADNotFoundError",
    "SpeechDetector",
    "SpeechDetectorError",
    "SpeechRegion",
    # Speech Recognizer
    "DEFAULT_BEAM_SIZE",
    "DEFAULT_COMPUTE_TYPE",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_SIZE",
    "FasterWhisperNotFoundError",
    "ModelLoadError",
    "SpeechRecognizer",
    "SpeechRecognizerError",
    "TranscriptSegment",
    "TranscriptionError",
    "TranscriptionResult",
    # Text Translator
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE",
    "DEFAULT_MAX_GAP_SECONDS",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_MERGED_CHARACTERS",
    "DEFAULT_MAX_MERGED_DURATION_SECONDS",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_NUM_BEAMS",
    "DEFAULT_SHORT_SEGMENT_WORDS",
    "CudaUnavailableError",
    "TextSegment",
    "TextSegmentError",
    "TextTranslator",
    "TextTranslatorError",
    "TranslatedTextSegment",
    "TranslationError",
    "TranslationResult",
    "TransformersNotFoundError",
    "UnsupportedLanguageError",
    "merge_text_segments",
    "normalize_text_segments",
    # Video Renderer
    "VideoRenderer",
    "VideoRendererConfig",
    "VideoRendererError",
    "VideoRenderResult",
    # Voice Generator
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_HINDI_TTS_MODEL",
    "DEFAULT_NUM_SPEAKER_PROFILE_SECONDS",
    "DEFAULT_PIPER_FEMALE_VOICE",
    "DEFAULT_PIPER_MALE_VOICE",
    "DEFAULT_PIPER_VOICE_PITCH_THRESHOLD_HZ",
    "DEFAULT_TTS_CKPT_FILE",
    "DEFAULT_TTS_DEVICE",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TTS_VOCAB_FILE",
    "DiarizationError",
    "F5TTSModelLoadError",
    "F5TTSNotFoundError",
    "F5TTSSynthesizer",
    "GeneratedSpeechSegment",
    "HuggingFaceTokenError",
    "NoSpeakersDetectedError",
    "PiperTTSSynthesizer",
    "PyannoteNotFoundError",
    "PyannoteSpeakerDiarizer",
    "SpeakerProfile",
    "SpeakerTurn",
    "TranslatedSegment",
    "UnsupportedTTSConfigurationError",
    "VoiceGenerationResult",
    "VoiceGenerator",
    "VoiceGeneratorError",
    "VoiceSynthesisError",
]
