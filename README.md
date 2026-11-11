## Video Translator

| Sl. no. | Component | Responsibility | Possible Implementation |
| ----- | ----- | ----- | ----- |
| 1 | Media Processor | Extract source audio, inspect media metadata, and separate vocals/instrumental stems | FFmpeg, ffprobe, Demucs / HTDemucs |
| 2 | Speech Recognizer | Detect speech regions, transcribe audio, and diarize speakers | Faster-Whisper + diarization pipeline |
| 3 | Subtitle Validator | Match generated English transcript against source subtitle timings and validate alignment | Custom timestamp validation logic |
| 4 | Text Translator | Translate validated English segments into the target language | NLLB / translation API / local translation model |
| 5 | Text Cleaner | Normalize translated text for language-specific issues such as Hindi cleanup | Custom post-processing / Hindi cleaning model |
| 6 | Voice Generator | Synthesize dubbed speech in the target language and optionally convert voice style | Piper TTS + OpenVoice |
| 7 | Audio Aligner | Adjust generated speech to fit the original timeline and silence gaps | Custom alignment and time-stretching logic |
| 8 | Video Renderer | Mix dubbed audio back into the original video and export the final file | FFmpeg |

## Folder Structure

```text
video_translator/
├── src
│   └── video_translator
│       ├── audio_aligner
│       │   ├── __init__.py
│       │   └── aligner.py
│       ├── audio_separator
│       │   ├── __init__.py
│       │   └── separator.py
│       ├── media_processor
│       │   ├── __init__.py
│       │   └── processor.py
│       ├── speech_detector
│       │   ├── __init__.py
│       │   └── detector.py
│       ├── speech_recognizer
│       │   ├── __init__.py
│       │   └── recognizer.py
│       ├── text_translator
│       │   ├── __init__.py
│       │   ├── segments.py
│       │   └── translator.py
│       ├── video_renderer
│       │   ├── __init__.py
│       │   └── renderer.py
│       ├── voice_generator
│       │   ├── __init__.py
│       │   ├── diarization.py
│       │   └── generator.py
│       └── __init__.py
├── tests
│   ├── integration
│   │   ├── test_audio_separator_integration.py
│   │   ├── test_media_processor_integration.py
│   │   ├── test_speech_detector_integration.py
│   │   ├── test_speech_recognizer_integration.py
│   │   ├── test_text_translator_integration.py
│   │   └── test_voice_generator_integration.py
│   ├── unit
│   │   ├── test_audio_aligner.py
│   │   ├── test_audio_separator.py
│   │   ├── test_media_processor.py
│   │   ├── test_speech_detector.py
│   │   ├── test_speech_recognizer.py
│   │   ├── test_text_translator.py
│   │   └── test_voice_generation.py
│   └── __init__.py
├── .gitignore
├── .pre-commit-config.yaml
├── poetry.lock
├── poetry.toml
├── pyproject.toml
└── README.md
```

## Live Project

A full blown movie translation project is currently under development. Check this [repo](https://github.com/subhajit-dubey/movie_translate_live).
