"""
tts.py — Text-to-Speech engine + threaded playback queue.

Public API
----------
class TTSEngine
    synthesize(text) -> str          # returns wav path

class TTSQueue
    push(text)
    wait_done(timeout) -> bool
    is_idle() -> bool
    interrupt()                      # stop playback, flush queue
    stop()
"""

import asyncio
import logging
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Windows UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Synthesize text to a .wav file.

    Steps
    -----
    1. edge-tts → temp .wav
    2. (optional) whisper-vits-svc inference → voice-converted .wav

    Parameters
    ----------
    voice      : edge-tts voice name, e.g. "en-US-AriaNeural"
    use_svc    : Enable whisper-vits-svc voice conversion (requires setup)
    svc_repo   : Path to cloned whisper-vits-svc repo
    svc_model  : Path to pretrained/fine-tuned .pth model
    svc_speaker: Path to speaker .npy embedding file

    SVC Setup
    ---------
    1. git clone https://github.com/PlayVoice/whisper-vits-svc
    2. cd whisper-vits-svc && pip install -r requirements.txt
    3. Download or train a speaker model (.pth) and speaker embedding (.npy)
    4. Set env vars:
         S2S_SVC_REPO  = /path/to/whisper-vits-svc
         S2S_SVC_MODEL = /path/to/model.pth
         S2S_SVC_SPK   = /path/to/speaker.npy
    5. Run with --svc flag

    Note: whisper-vits-svc converts the voice characteristics of edge-tts output
    to match a target speaker. Pass --svc for the most realistic-sounding voice
    once your speaker model is trained. The model can also handle sung audio if
    the input contains melodic content.
    """

    def __init__(
        self,
        voice: str = "en-US-AriaNeural",
        use_svc: bool = False,
        svc_repo: Optional[str] = None,
        svc_model: Optional[str] = None,
        svc_speaker: Optional[str] = None,
    ) -> None:
        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            raise ImportError("edge-tts is required — pip install edge-tts") from e

        self.voice = voice
        self.use_svc = use_svc
        self.svc_repo = svc_repo or os.getenv("S2S_SVC_REPO", "")
        self.svc_model = svc_model or os.getenv("S2S_SVC_MODEL", "")
        self.svc_speaker = svc_speaker or os.getenv("S2S_SVC_SPK", "")

        if use_svc and not (self.svc_repo and self.svc_model and self.svc_speaker):
            logger.warning(
                "SVC requested but svc_repo/svc_model/svc_speaker not fully set — "
                "falling back to plain edge-tts. "
                "Set S2S_SVC_REPO, S2S_SVC_MODEL, S2S_SVC_SPK to enable."
            )
            self.use_svc = False

    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> str:
        """
        Convert *text* to speech and return the path to a .wav file.
        The caller is responsible for deleting the file when done.
        """
        if not text.strip():
            return ""

        wav_path = self._edge_tts(text)

        if self.use_svc and wav_path:
            wav_path = self._svc_convert(wav_path)

        return wav_path

    def _edge_tts(self, text: str) -> str:
        """Run edge-tts and return a temp .wav path."""
        import edge_tts

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        async def _run() -> None:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(path)

        asyncio.run(_run())

        # edge-tts saves as MP3 when given a .wav extension in older versions.
        path = self._ensure_wav(path)
        return path

    def _ensure_wav(self, path: str) -> str:
        """
        edge-tts may write MP3 bytes regardless of file extension.
        Detect and convert to WAV if needed.
        """
        with open(path, "rb") as f:
            header = f.read(3)

        is_mp3 = (
            header[:3] == b"ID3"
            or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
        )
        if is_mp3:
            try:
                import subprocess
                wav_path = path.replace(".wav", "_conv.wav")
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", path, wav_path],
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    os.unlink(path)
                    return wav_path
                else:
                    logger.warning("ffmpeg conversion failed — trying pydub")
            except FileNotFoundError:
                logger.debug("ffmpeg not found — trying pydub")

            try:
                from pydub import AudioSegment
                sound = AudioSegment.from_mp3(path)
                wav_path = path.replace(".wav", "_conv.wav")
                sound.export(wav_path, format="wav")
                os.unlink(path)
                return wav_path
            except Exception as exc:
                logger.warning("pydub conversion failed: %s — playing as-is", exc)

        return path

    def _svc_convert(self, wav_path: str) -> str:
        """
        Run whisper-vits-svc inference on *wav_path* and return converted path.

        Requires svc_repo, svc_model, svc_speaker to be set.
        The converted audio will sound like the trained speaker rather than
        the edge-tts voice.

        Notes
        -----
        - CLI flag is ``--wave`` (not --wav)
        - Output is always written to ``<svc_repo>/svc_out.wav`` (no --out flag)
        - Subprocess runs with cwd=svc_repo so the output lands there
        """
        import shutil
        import subprocess

        svc_repo = Path(self.svc_repo)
        svc_out = svc_repo / "svc_out.wav"
        out_path = wav_path.replace(".wav", "_svc.wav")

        # Convert input path to absolute so it's valid from cwd=svc_repo
        wav_abs = str(Path(wav_path).resolve())

        cmd = [
            sys.executable,
            "svc_inference.py",
            "--config", str(svc_repo / "configs" / "base.yaml"),
            "--model", self.svc_model,
            "--spk", self.svc_speaker,
            "--wave", wav_abs,
            "--shift", "0",
        ]
        logger.debug("SVC cmd (cwd=%s): %s", svc_repo, " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(svc_repo),
        )
        if result.returncode != 0:
            logger.error("SVC inference failed:\n%s", result.stderr[-1000:])
            return wav_path  # fallback to edge-tts output

        if not svc_out.exists():
            logger.error("SVC ran but svc_out.wav not found in %s", svc_repo)
            return wav_path

        # Move output to temp path alongside original so cleanup works normally
        shutil.move(str(svc_out), out_path)

        try:
            os.unlink(wav_path)
        except OSError:
            pass

        return out_path


# ---------------------------------------------------------------------------
# TTSQueue — threaded synthesis + playback with overlap and interrupt support
# ---------------------------------------------------------------------------

class TTSQueue:
    """
    Threaded queue that synthesizes and plays audio chunks, overlapping
    synthesis of chunk N+1 with playback of chunk N.

    Supports interrupt() to stop current playback and flush all pending items.

    Usage::

        engine = TTSEngine()
        q = TTSQueue(engine)
        q.push("Hello there!")
        q.push("How can I help you?")
        q.wait_done()

        # Interrupt mid-playback:
        q.interrupt()

        q.stop()  # when done with the queue entirely
    """

    _SENTINEL = object()

    def __init__(self, engine: TTSEngine) -> None:
        self._engine = engine
        self._synth_q: queue.Queue = queue.Queue()
        self._play_q: queue.Queue = queue.Queue()

        # Interrupt: stop_event is checked between playback chunks
        self._interrupt = threading.Event()

        # Idle tracking: pending counts items from push() until play completes
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()  # start idle

        self._synth_thread = threading.Thread(
            target=self._synth_worker, daemon=True, name="tts-synth"
        )
        self._play_thread = threading.Thread(
            target=self._play_worker, daemon=True, name="tts-play"
        )
        self._synth_thread.start()
        self._play_thread.start()

    # ------------------------------------------------------------------
    # Public API

    def push(self, text: str) -> None:
        """Enqueue *text* for synthesis + playback."""
        if text.strip():
            with self._pending_lock:
                self._pending += 1
                self._idle.clear()
            self._synth_q.put(text)

    def wait_done(self, timeout: float = 120.0) -> bool:
        """Block until all synthesis and playback is complete. Returns False on timeout."""
        return self._idle.wait(timeout=timeout)

    def is_idle(self) -> bool:
        """True when no synthesis or playback is pending."""
        return self._idle.is_set()

    def interrupt(self) -> None:
        """
        Stop current playback immediately and discard all pending items.

        Safe to call from any thread. After this call, is_idle() returns True.
        """
        self._interrupt.set()

        # Drain synth queue
        while True:
            try:
                self._synth_q.get_nowait()
                self._synth_q.task_done()
            except queue.Empty:
                break

        # Drain play queue (delete temp WAV files)
        while True:
            try:
                item = self._play_q.get_nowait()
                self._play_q.task_done()
                if isinstance(item, str):
                    try:
                        os.unlink(item)
                    except OSError:
                        pass
            except queue.Empty:
                break

        # Force idle
        with self._pending_lock:
            self._pending = 0
        self._idle.set()
        self._interrupt.clear()

    def stop(self) -> None:
        """Shut down worker threads gracefully."""
        self._synth_q.put(self._SENTINEL)
        self._play_q.put(self._SENTINEL)
        self._synth_thread.join(timeout=5)
        self._play_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal

    def _decrement(self) -> None:
        with self._pending_lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()

    def _synth_worker(self) -> None:
        while True:
            item = self._synth_q.get()
            try:
                if item is self._SENTINEL:
                    self._play_q.put(self._SENTINEL)
                    return

                if self._interrupt.is_set():
                    # Interrupted before synthesis — skip
                    self._decrement()
                    continue

                wav_path = self._engine.synthesize(item)

                if not wav_path:
                    # Empty synthesis result — decrement without playback
                    self._decrement()
                elif self._interrupt.is_set():
                    # Interrupted during synthesis — discard wav
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
                    self._decrement()
                else:
                    # Send to playback (decrement happens there)
                    self._play_q.put(wav_path)

            except Exception:
                logger.exception("Synthesis error for %r", item)
                self._decrement()
            finally:
                self._synth_q.task_done()

    def _play_worker(self) -> None:
        from scripts_and_skills.speech.audio_utils import play_wav_interruptible

        while True:
            item = self._play_q.get()
            try:
                if item is self._SENTINEL:
                    return

                if not self._interrupt.is_set():
                    play_wav_interruptible(item, self._interrupt)

                try:
                    os.unlink(item)
                except OSError:
                    pass

            except Exception:
                logger.exception("Playback error for %r", item)
            finally:
                self._play_q.task_done()
                self._decrement()


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test TTS synthesis + playback")
    parser.add_argument("text", nargs="?", default="Hello, this is a test of the voice pipeline.")
    parser.add_argument("--voice", default="en-US-AriaNeural")
    parser.add_argument("--svc", action="store_true", help="Enable SVC voice conversion")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    engine = TTSEngine(
        voice=args.voice,
        use_svc=args.svc,
        svc_repo=os.getenv("S2S_SVC_REPO", ""),
        svc_model=os.getenv("S2S_SVC_MODEL", ""),
        svc_speaker=os.getenv("S2S_SVC_SPK", ""),
    )
    q = TTSQueue(engine)
    q.push(args.text)
    q.wait_done()
    q.stop()
    print("Done.")
