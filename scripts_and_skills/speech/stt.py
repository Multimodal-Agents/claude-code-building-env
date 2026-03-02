"""
stt.py — Speech-to-Text via faster-whisper.

Public API
----------
class SpeechRecognizer
    transcribe(audio_bytes) -> str
    listen_and_transcribe()  -> str
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Windows UTF-8 fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class SpeechRecognizer:
    """
    Wraps faster-whisper for transcription with optional VAD-based recording.

    Parameters
    ----------
    model_size  : Whisper model size ("tiny", "base", "small", "medium", "large-v3")
    device      : "auto" | "cpu" | "cuda"
    compute_type: "auto" | "int8" | "float16" | "float32"
    language    : ISO language code or None (auto-detect)
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is required — pip install faster-whisper"
            ) from e

        # Resolve "auto" device
        _device = device
        _compute = compute_type
        if device == "auto":
            try:
                import torch
                _device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                _device = "cpu"

        if compute_type == "auto":
            _compute = "int8" if _device == "cpu" else "float16"

        logger.info(
            "Loading faster-whisper model=%s device=%s compute=%s",
            model_size, _device, _compute,
        )
        self._model = WhisperModel(model_size, device=_device, compute_type=_compute)
        self._language = language
        logger.info("faster-whisper ready")

    # ------------------------------------------------------------------
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw PCM int16 bytes (16 kHz, mono).

        Returns the full transcript as a single string.
        """
        from scripts_and_skills.speech.audio_utils import save_temp_wav
        import os

        if not audio_bytes:
            return ""

        wav_path = save_temp_wav(audio_bytes, sample_rate=16000)
        try:
            segments, info = self._model.transcribe(
                wav_path,
                language=self._language,
                beam_size=5,
                vad_filter=True,
            )
            logger.debug(
                "Detected language: %s (%.0f%%)",
                info.language, info.language_probability * 100,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        logger.info("Transcribed: %r", text)
        return text

    # ------------------------------------------------------------------
    def listen_and_transcribe(
        self,
        silence_ms: int = 600,
        vad_aggressiveness: int = 2,
        sample_rate: int = 16000,
    ) -> str:
        """
        Blocking: record mic audio (VAD + silence debounce) then transcribe.

        Returns the transcript string (empty if no speech captured).
        """
        from scripts_and_skills.speech.audio_utils import record_with_vad

        audio_bytes = record_with_vad(
            silence_ms=silence_ms,
            vad_aggressiveness=vad_aggressiveness,
            sample_rate=sample_rate,
        )
        if not audio_bytes:
            return ""
        return self.transcribe(audio_bytes)


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test STT: record + transcribe once")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--silence-ms", type=int, default=600)
    parser.add_argument("--test", action="store_true", help="Run a single capture test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rec = SpeechRecognizer(model_size=args.model)
    print("Speak now (will stop after silence)…")
    result = rec.listen_and_transcribe(silence_ms=args.silence_ms)
    print(f"\nTranscript: {result!r}")
