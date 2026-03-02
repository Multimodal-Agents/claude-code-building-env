"""
audio_utils.py — Microphone capture, WebRTC VAD, WAV helpers.

Public API
----------
record_with_vad(silence_ms, vad_aggressiveness, sample_rate) -> bytes
record_with_vad_continue(initial_audio, silence_ms, ...) -> bytes
play_wav(path)
play_wav_interruptible(path, stop_event)
save_temp_wav(audio_bytes, sample_rate) -> str

class SpeechMonitor
    start()
    stop()
    is_speaking() -> bool
    drain() -> bytes
"""

import collections
import logging
import struct
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so the rest of the package loads
# even on machines without audio hardware.
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False
    logger.debug("sounddevice not installed — audio capture/playback unavailable")

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    logger.debug("webrtcvad not installed — VAD unavailable")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    import scipy.io.wavfile as _wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FRAME_DURATION_MS = 30          # webrtcvad works with 10 / 20 / 30 ms frames
_CHANNELS = 1
_DTYPE = "int16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_temp_wav(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Write raw PCM int16 bytes to a temp .wav file and return the path."""
    import os
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    n_channels = _CHANNELS
    sampwidth = 2  # int16 = 2 bytes

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

    logger.debug("Saved temp WAV: %s (%d bytes)", path, len(audio_bytes))
    return path


def play_wav(path: str) -> None:
    """Blocking playback of a .wav file via sounddevice."""
    if not HAS_SD:
        raise RuntimeError("sounddevice is required for playback — pip install sounddevice")
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for WAV reading — pip install scipy")

    import scipy.io.wavfile as wf
    sr, data = wf.read(path)
    sd.play(data, samplerate=sr)
    sd.wait()


def play_wav_interruptible(path: str, stop_event: threading.Event) -> None:
    """
    Play a .wav file, stopping early if stop_event is set.

    Uses sd.OutputStream so we can check stop_event between chunks without
    calling the global sd.stop() (which would kill the SpeechMonitor stream).
    Falls back to blocking play_wav if deps are missing.
    """
    if not HAS_SD or not HAS_SCIPY or not HAS_NP:
        play_wav(path)
        return

    import scipy.io.wavfile as wf
    sr, data = wf.read(path)

    # Normalize to float32 for OutputStream
    if data.dtype == np.int16:
        fdata = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        fdata = data.astype(np.float32) / 2147483648.0
    else:
        fdata = data.astype(np.float32)

    if fdata.ndim == 1:
        fdata = fdata.reshape(-1, 1)
    channels = fdata.shape[1]

    CHUNK = 2048
    pos = 0
    try:
        with sd.OutputStream(samplerate=sr, channels=channels, dtype="float32") as stream:
            while pos < len(fdata):
                if stop_event.is_set():
                    break
                end = min(pos + CHUNK, len(fdata))
                stream.write(fdata[pos:end])
                pos = end
    except Exception as exc:
        logger.warning("Interruptible playback error: %s — falling back to play_wav", exc)
        if not stop_event.is_set():
            play_wav(path)


# ---------------------------------------------------------------------------
# Core recording function
# ---------------------------------------------------------------------------

def record_with_vad(
    silence_ms: int = 800,
    vad_aggressiveness: int = 2,
    sample_rate: int = 16000,
    max_recording_sec: int = 60,
) -> bytes:
    """
    Record audio from the default microphone using WebRTC VAD.

    Strategy
    --------
    1. Continuously read 30 ms frames.
    2. Once speech is detected (VAD returns True for ≥ 3 consecutive frames),
       mark recording as started.
    3. Once ``silence_ms`` of continuous silence is detected AFTER speech onset,
       stop and return the accumulated audio.
    4. Hard-cut at ``max_recording_sec`` to prevent runaway captures.

    Returns
    -------
    bytes  — raw PCM int16 at ``sample_rate`` Hz, mono.
    """
    if not HAS_SD:
        raise RuntimeError("sounddevice is required — pip install sounddevice")
    if not HAS_VAD:
        raise RuntimeError("webrtcvad is required — pip install webrtcvad")
    if not HAS_NP:
        raise RuntimeError("numpy is required — pip install numpy")

    vad = webrtcvad.Vad(vad_aggressiveness)

    frame_samples = int(sample_rate * _FRAME_DURATION_MS / 1000)
    frame_bytes = frame_samples * 2  # int16 = 2 bytes

    silence_frames_needed = int(silence_ms / _FRAME_DURATION_MS)
    max_frames = int(max_recording_sec * 1000 / _FRAME_DURATION_MS)

    print("  [Listening…]", end="", flush=True)

    frames_collected: list[bytes] = []
    speech_started = False
    silent_frames = 0
    speech_onset_count = 0

    with sd.RawInputStream(
        samplerate=sample_rate,
        channels=_CHANNELS,
        dtype=_DTYPE,
        blocksize=frame_samples,
    ) as stream:
        for _ in range(max_frames):
            raw, _ = stream.read(frame_samples)
            frame = bytes(raw)

            # Pad or trim to exact frame size expected by webrtcvad
            if len(frame) < frame_bytes:
                frame = frame + b"\x00" * (frame_bytes - len(frame))
            elif len(frame) > frame_bytes:
                frame = frame[:frame_bytes]

            is_speech = vad.is_speech(frame, sample_rate)

            if not speech_started:
                if is_speech:
                    speech_onset_count += 1
                    frames_collected.append(frame)
                    if speech_onset_count >= 3:
                        speech_started = True
                        print(" [Recording]", end="", flush=True)
                else:
                    speech_onset_count = 0
            else:
                frames_collected.append(frame)
                if is_speech:
                    silent_frames = 0
                else:
                    silent_frames += 1
                    if silent_frames >= silence_frames_needed:
                        break

    print()  # newline after inline status

    if not frames_collected:
        logger.warning("No speech detected in recording window")
        return b""

    audio_bytes = b"".join(frames_collected)
    logger.debug(
        "Recorded %d frames → %d bytes (%.1f s)",
        len(frames_collected),
        len(audio_bytes),
        len(frames_collected) * _FRAME_DURATION_MS / 1000,
    )
    return audio_bytes


def record_with_vad_continue(
    initial_audio: bytes,
    silence_ms: int = 800,
    vad_aggressiveness: int = 2,
    sample_rate: int = 16000,
    max_additional_sec: int = 30,
) -> bytes:
    """
    Continue VAD recording after speech onset is already established.

    Used after an interrupt: the SpeechMonitor has already captured audio
    from speech onset; this function continues recording until silence.

    Parameters
    ----------
    initial_audio : bytes  — audio already buffered by SpeechMonitor
    silence_ms    : int    — ms of silence before stopping (after initial_audio)

    Returns
    -------
    bytes — initial_audio + additional recorded bytes
    """
    if not HAS_SD or not HAS_VAD or not HAS_NP:
        return initial_audio

    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_samples = int(sample_rate * _FRAME_DURATION_MS / 1000)
    frame_bytes = frame_samples * 2
    silence_frames_needed = int(silence_ms / _FRAME_DURATION_MS)
    max_frames = int(max_additional_sec * 1000 / _FRAME_DURATION_MS)

    accumulated: list[bytes] = [initial_audio]
    silent_frames = 0

    with sd.RawInputStream(
        samplerate=sample_rate,
        channels=_CHANNELS,
        dtype=_DTYPE,
        blocksize=frame_samples,
    ) as stream:
        for _ in range(max_frames):
            raw, _ = stream.read(frame_samples)
            frame = bytes(raw)
            if len(frame) < frame_bytes:
                frame += b"\x00" * (frame_bytes - len(frame))
            elif len(frame) > frame_bytes:
                frame = frame[:frame_bytes]

            accumulated.append(frame)

            try:
                is_speech = vad.is_speech(frame, sample_rate)
            except Exception:
                continue

            if is_speech:
                silent_frames = 0
            else:
                silent_frames += 1
                if silent_frames >= silence_frames_needed:
                    break

    return b"".join(accumulated)


# ---------------------------------------------------------------------------
# SpeechMonitor — background VAD via sd.InputStream callback
# ---------------------------------------------------------------------------

class SpeechMonitor:
    """
    Continuously monitors the microphone in the background using WebRTC VAD.

    Designed to run while TTS is playing so we can detect if the user starts
    speaking (interrupt intent). Uses sd.InputStream callback — a separate
    stream from TTS playback so they don't interfere.

    Usage::

        monitor = SpeechMonitor()
        monitor.start()

        while tts_is_playing:
            if monitor.is_speaking():
                tts.interrupt()
                audio = monitor.drain()  # get buffered audio from speech onset
                # continue recording + transcribe
                break

        monitor.stop()
    """

    _ONSET_FRAMES = 3   # consecutive VAD=True frames before "speaking" fires

    def __init__(
        self,
        vad_aggressiveness: int = 2,
        sample_rate: int = 16000,
        pre_buffer_frames: int = 5,
    ) -> None:
        if not HAS_VAD:
            raise RuntimeError("webrtcvad is required — pip install webrtcvad")
        if not HAS_SD:
            raise RuntimeError("sounddevice is required — pip install sounddevice")
        if not HAS_NP:
            raise RuntimeError("numpy is required — pip install numpy")

        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._sr = sample_rate
        self._frame_samples = int(sample_rate * _FRAME_DURATION_MS / 1000)
        self._frame_bytes = self._frame_samples * 2

        self._lock = threading.Lock()
        self._speaking = threading.Event()
        self._pre_buf: collections.deque = collections.deque(maxlen=pre_buffer_frames)
        self._audio_buf: list[bytes] = []
        self._onset_count = 0
        self._stream: Optional["sd.InputStream"] = None

    def start(self) -> None:
        """Open the background monitoring stream."""
        self._reset()
        try:
            self._stream = sd.InputStream(
                samplerate=self._sr,
                channels=1,
                dtype="int16",
                blocksize=self._frame_samples,
                callback=self._callback,
            )
            self._stream.start()
            logger.debug("SpeechMonitor stream started")
        except Exception as exc:
            logger.warning("SpeechMonitor could not open stream: %s", exc)
            self._stream = None

    def stop(self) -> None:
        """Close the monitoring stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            logger.debug("SpeechMonitor stream stopped")

    def is_speaking(self) -> bool:
        """True once speech onset (≥3 VAD frames) has been confirmed."""
        return self._speaking.is_set()

    def drain(self) -> bytes:
        """
        Return all buffered audio from just before speech onset and reset state.
        Call this after detecting is_speaking() to get audio for transcription.
        """
        with self._lock:
            data = b"".join(self._pre_buf) + b"".join(self._audio_buf)
            self._pre_buf.clear()
            self._audio_buf.clear()
            self._onset_count = 0
            self._speaking.clear()
        return data

    def _reset(self) -> None:
        self._speaking.clear()
        with self._lock:
            self._pre_buf.clear()
            self._audio_buf.clear()
            self._onset_count = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        frame = bytes(indata)
        # Ensure exact frame byte length expected by webrtcvad
        if len(frame) < self._frame_bytes:
            frame += b"\x00" * (self._frame_bytes - len(frame))
        elif len(frame) > self._frame_bytes:
            frame = frame[: self._frame_bytes]

        try:
            is_speech = self._vad.is_speech(frame, self._sr)
        except Exception:
            return

        with self._lock:
            if not self._speaking.is_set():
                self._pre_buf.append(frame)
                if is_speech:
                    self._onset_count += 1
                    if self._onset_count >= self._ONSET_FRAMES:
                        self._speaking.set()
                        self._audio_buf.extend(list(self._pre_buf))
                        self._pre_buf.clear()
                else:
                    self._onset_count = max(0, self._onset_count - 1)
            else:
                self._audio_buf.append(frame)
