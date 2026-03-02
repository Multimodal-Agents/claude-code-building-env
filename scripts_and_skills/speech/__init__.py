"""
scripts_and_skills.speech
=========================
Real-time speech-to-speech pipeline:
  Mic → VAD → faster-whisper STT → Claude LLM stream → StreamingChunker
  → edge-tts TTS → (optional) whisper-vits-svc voice conversion → playback

Lazy imports: heavy deps (faster-whisper, sounddevice, edge-tts) are only
loaded when the relevant class is actually instantiated.
"""

__all__ = [
    "SpeechRecognizer",
    "TextChunker",
    "StreamingChunker",
    "TTSEngine",
    "TTSQueue",
    "VoicePipeline",
]

# Graceful availability flags (callers can check before importing)
import importlib.util as _ilu

def _has(pkg: str) -> bool:
    return _ilu.find_spec(pkg) is not None

HAS_SOUNDDEVICE  = _has("sounddevice")
HAS_WEBRTCVAD    = _has("webrtcvad")
HAS_FASTER_WHISPER = _has("faster_whisper")
HAS_EDGE_TTS     = _has("edge_tts")
HAS_ANTHROPIC    = _has("anthropic")
HAS_NUMPY        = _has("numpy")
HAS_SCIPY        = _has("scipy")


def SpeechRecognizer(*args, **kwargs):
    from scripts_and_skills.speech.stt import SpeechRecognizer as _C
    return _C(*args, **kwargs)


def TextChunker(*args, **kwargs):
    from scripts_and_skills.speech.text_chunker import TextChunker as _C
    return _C(*args, **kwargs)


def StreamingChunker(*args, **kwargs):
    from scripts_and_skills.speech.text_chunker import StreamingChunker as _C
    return _C(*args, **kwargs)


def TTSEngine(*args, **kwargs):
    from scripts_and_skills.speech.tts import TTSEngine as _C
    return _C(*args, **kwargs)


def TTSQueue(*args, **kwargs):
    from scripts_and_skills.speech.tts import TTSQueue as _C
    return _C(*args, **kwargs)


def VoicePipeline(*args, **kwargs):
    from scripts_and_skills.speech.voice_pipeline import VoicePipeline as _C
    return _C(*args, **kwargs)
