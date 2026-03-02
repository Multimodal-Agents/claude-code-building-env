"""
voice_pipeline.py — Full real-time speech-to-speech pipeline.

Flow
----
Mic → VAD → faster-whisper → [confirm?] → LLM stream → StreamingChunker → TTSQueue → speaker
                                              ↑ interrupt if user talks over TTS

Backends
--------
  --anthropic    Anthropic API  (default; uses ANTHROPIC_API_KEY)
  --ollama       Local Ollama   (http://localhost:11434, model: gpt-oss:20b)
  --claude-code  Claude CLI subprocess (no separate API key needed, unsets CLAUDECODE)

Entry points
------------
  python -m scripts_and_skills.speech.voice_pipeline
  python -m scripts_and_skills.speech.voice_pipeline --ollama
  python -m scripts_and_skills.speech.voice_pipeline --claude-code
  python -m scripts_and_skills.speech.voice_pipeline --svc
  python -m scripts_and_skills.speech.voice_pipeline --confirm

Interrupt while speaking
------------------------
While the assistant is talking, just start speaking — the pipeline detects your
voice, stops the TTS mid-sentence, and processes your new input immediately.
Requires sounddevice to support simultaneous input + output streams (usually works
on Windows WASAPI / macOS CoreAudio). Disabled automatically if not supported.

Confirm mode
------------
After each transcript, the assistant reads back the first few words and waits for
"yes" or "no" before executing. Useful for hands-off environments or when you want
to review what was transcribed before it runs.

SVC (realistic voice)
---------------------
Pass --svc to enable whisper-vits-svc voice conversion. Requires:
  S2S_SVC_REPO  = /path/to/whisper-vits-svc  (cloned repo)
  S2S_SVC_MODEL = /path/to/model.pth
  S2S_SVC_SPK   = /path/to/speaker.npy

Env vars (all optional)
-----------------------
  S2S_BACKEND   anthropic | ollama       (default: anthropic)
  S2S_MODEL     model id
  S2S_VOICE     edge-tts voice name      (default: en-US-AriaNeural)
  S2S_SVC       1 to enable SVC
  S2S_SVC_REPO  path to whisper-vits-svc
  S2S_SVC_MODEL path to .pth model
  S2S_SVC_SPK   path to speaker .npy
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
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
# Default config from env
# ---------------------------------------------------------------------------
_BACKEND  = os.getenv("S2S_BACKEND",  "anthropic")
_MODEL_AP = os.getenv("S2S_MODEL",    "claude-sonnet-4-6")
_MODEL_OL = os.getenv("S2S_MODEL",    "gpt-oss:20b")
_VOICE    = os.getenv("S2S_VOICE",    "en-US-AriaNeural")
_USE_SVC  = os.getenv("S2S_SVC",      "0") == "1"
_SVC_REPO = os.getenv("S2S_SVC_REPO", "")
_SVC_MDL  = os.getenv("S2S_SVC_MODEL","")
_SVC_SPK  = os.getenv("S2S_SVC_SPK",  "")

_DEFAULT_SYSTEM = (
    "You are a concise, conversational voice assistant. "
    "Respond in plain spoken English — no markdown, no bullet points, no code blocks. "
    "Keep answers brief (1-3 sentences) unless the user explicitly asks for detail."
)

# Prepended to every voice prompt when using claude-code backend
_CLAUDE_CODE_PREFIX = (
    "[Voice input — respond in plain spoken English, "
    "no markdown, no bullet points, 1-3 sentences max]: "
)

_DEFAULT_STOP_PHRASE = "goodbye claude"

# Words that count as "yes" / "no" in confirm mode
_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "go", "confirm", "affirmative"}
_NO_WORDS  = {"no", "nope", "cancel", "negative", "stop", "abort", "wrong", "incorrect", "dont", "nah"}

_RE_STRIP_PUNCT = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Internal sentinel for breaking out of nested loops
# ---------------------------------------------------------------------------
class _StopVoice(Exception):
    pass


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """
    Orchestrates the full speech-to-speech loop.

    Parameters
    ----------
    backend       : "anthropic" | "ollama" | "claude-code"
    model         : LLM model id (backend-specific default if None)
    system_prompt : LLM system prompt
    stt           : pre-built SpeechRecognizer (created if None)
    tts_engine    : pre-built TTSEngine (created if None)
    use_svc       : enable SVC voice conversion
    stop_phrase   : phrase to exit voice mode (default: "goodbye claude")
    confirm_mode  : if True, ask yes/no before executing each prompt
    silence_ms    : ms of silence to end a recording (default: 800)
    min_words     : discard transcripts shorter than this many words (noise filter)
    """

    def __init__(
        self,
        backend: str = _BACKEND,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stt=None,
        tts_engine=None,
        use_svc: bool = _USE_SVC,
        stop_phrase: str = _DEFAULT_STOP_PHRASE,
        confirm_mode: bool = False,
        silence_ms: int = 800,
        min_words: int = 2,
    ) -> None:
        self.backend = backend.lower()
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM
        self.use_svc = use_svc
        self.confirm_mode = confirm_mode
        self.silence_ms = silence_ms
        self.min_words = min_words
        self.stop_phrase = _RE_STRIP_PUNCT.sub("", stop_phrase.lower()).strip()

        if model:
            self.model = model
        elif self.backend == "anthropic":
            self.model = _MODEL_AP
        else:
            self.model = _MODEL_OL

        # STT
        if stt is None:
            from scripts_and_skills.speech.stt import SpeechRecognizer
            self._stt = SpeechRecognizer()
        else:
            self._stt = stt

        # TTS
        if tts_engine is None:
            from scripts_and_skills.speech.tts import TTSEngine
            self._tts_engine = TTSEngine(
                voice=_VOICE,
                use_svc=use_svc,
                svc_repo=_SVC_REPO,
                svc_model=_SVC_MDL,
                svc_speaker=_SVC_SPK,
            )
        else:
            self._tts_engine = tts_engine

        self._history: list[dict] = []

        logger.info(
            "VoicePipeline ready | backend=%s model=%s svc=%s confirm=%s",
            self.backend, self.model, self.use_svc, self.confirm_mode,
        )

    # ------------------------------------------------------------------
    # Main loop

    def run(self) -> None:
        """Blocking speech-to-speech loop. Press Ctrl+C or say stop phrase to exit."""
        from scripts_and_skills.speech.tts import TTSQueue

        tts_queue = TTSQueue(self._tts_engine)

        # Try to set up the interrupt monitor (may fail on some audio drivers)
        monitor = self._try_build_monitor()

        print(f"\n[Voice mode | backend={self.backend} | model={self.model}]")
        if self.use_svc:
            print("[SVC voice conversion enabled]")
        if self.confirm_mode:
            print("[Confirm mode ON — say yes/no after each prompt]")
        print(f'Say "{self.stop_phrase}" or Ctrl+C to exit.\n')

        try:
            while True:
                self._one_turn(tts_queue, monitor)

        except _StopVoice:
            print("\n[Voice mode ended — back to text chat]")
        except KeyboardInterrupt:
            print("\n[Exiting voice mode]")
        finally:
            tts_queue.stop()
            if monitor is not None:
                monitor.stop()

    def _one_turn(self, tts_queue, monitor) -> None:
        """Handle one listen → respond cycle."""
        # 1. Listen
        user_text = self._stt.listen_and_transcribe(silence_ms=self.silence_ms)
        if not user_text:
            return

        # 2. Filter very short transcripts (VAD noise / silence artifacts)
        if len(user_text.split()) < self.min_words:
            logger.debug("Short transcript filtered: %r", user_text)
            return

        print(f"\nYou: {user_text}")

        # 3. Stop phrase check
        self._check_stop(user_text, tts_queue)

        # 4. Confirm mode
        if self.confirm_mode:
            if not self._confirm_prompt(user_text, tts_queue):
                print("  [Cancelled]")
                return

        # 5. Stream LLM response → chunk → TTS
        self._stream_and_speak(user_text, tts_queue)

        # 6. Wait for TTS, watching for interrupt from user speaking
        self._wait_with_interrupt(tts_queue, monitor)

        print()  # blank line between turns

    def _check_stop(self, user_text: str, tts_queue) -> None:
        """Raise _StopVoice if stop phrase detected in user_text."""
        normalized = _RE_STRIP_PUNCT.sub("", user_text.lower()).strip()
        if self.stop_phrase and self.stop_phrase in normalized:
            farewell = "Goodbye! Returning to text chat."
            print(f"\n{farewell}")
            tts_queue.push(farewell)
            tts_queue.wait_done()
            raise _StopVoice()

    def _wait_with_interrupt(self, tts_queue, monitor) -> None:
        """
        Wait for TTS to finish. If monitor detects speech, interrupt TTS,
        record the new utterance, and process it.
        """
        from scripts_and_skills.speech.tts import TTSQueue

        if monitor is None:
            tts_queue.wait_done()
            return

        monitor.start()
        try:
            while not tts_queue.is_idle():
                time.sleep(0.03)

                if monitor.is_speaking():
                    tts_queue.interrupt()
                    buffered = monitor.drain()
                    monitor.stop()

                    print("\n  [Interrupt — listening]", flush=True)

                    # Continue recording from where the monitor left off
                    from scripts_and_skills.speech.audio_utils import record_with_vad_continue
                    all_audio = record_with_vad_continue(
                        buffered, silence_ms=self.silence_ms
                    )
                    interrupt_text = self._stt.transcribe(all_audio)

                    if interrupt_text and len(interrupt_text.split()) >= self.min_words:
                        print(f"You: {interrupt_text}")
                        self._check_stop(interrupt_text, tts_queue)

                        if self.confirm_mode and not self._confirm_prompt(interrupt_text, tts_queue):
                            print("  [Cancelled]")
                            monitor.start()
                            continue

                        self._stream_and_speak(interrupt_text, tts_queue)
                        monitor.start()
                    else:
                        # Brief noise — restart monitor
                        monitor.start()
        finally:
            monitor.stop()

    # ------------------------------------------------------------------
    # Confirm mode

    def _confirm_prompt(self, prompt_text: str, tts_queue) -> bool:
        """
        TTS reads back the first few words of the prompt and listens for yes/no.
        Returns True if confirmed (or after 2 unclear responses).
        """
        words = prompt_text.split()
        preview = " ".join(words[:8]) + ("..." if len(words) > 8 else "")

        for attempt in range(2):
            cue = (
                f"Confirm: {preview}? Say yes or no."
                if attempt == 0
                else "I didn't catch that — say yes or no."
            )
            tts_queue.push(cue)
            tts_queue.wait_done()

            response = self._stt.listen_and_transcribe(silence_ms=700)
            if not response:
                continue

            normalized_words = set(_RE_STRIP_PUNCT.sub("", response.lower()).split())

            if normalized_words & _YES_WORDS:
                return True
            if normalized_words & _NO_WORDS:
                tts_queue.push("Cancelled.")
                tts_queue.wait_done()
                return False

        # Default: execute after 2 unclear responses
        return True

    # ------------------------------------------------------------------
    # LLM streaming

    def _stream_and_speak(self, user_text: str, tts_queue) -> None:
        """Stream LLM response, chunk it, and enqueue TTS."""
        from scripts_and_skills.speech.text_chunker import StreamingChunker

        self._history.append({"role": "user", "content": user_text})

        chunker = StreamingChunker()
        full_response: list[str] = []

        print("Assistant: ", end="", flush=True)

        try:
            if self.backend == "claude-code":
                self._stream_claude_code(user_text, chunker, tts_queue, full_response)
                return  # history not maintained (new session each turn)
            elif self.backend == "anthropic":
                self._stream_anthropic(chunker, tts_queue, full_response)
            else:
                self._stream_ollama(chunker, tts_queue, full_response)
        except Exception as exc:
            logger.error("LLM streaming error: %s", exc, exc_info=True)
            error_msg = "I encountered an error processing that request."
            tts_queue.push(error_msg)
            print(error_msg)
            return

        tail = chunker.flush()
        if tail:
            tts_queue.push(tail)
            print(tail, end="", flush=True)

        print()

        assistant_text = "".join(full_response)
        self._history.append({"role": "assistant", "content": assistant_text})

    def _stream_anthropic(self, chunker, tts_queue, full_response: list) -> None:
        import anthropic

        client = anthropic.Anthropic()
        messages = self._build_messages()

        with client.messages.stream(
            model=self.model,
            max_tokens=512,
            system=self.system_prompt,
            messages=messages,
        ) as stream:
            for text_delta in stream.text_stream:
                print(text_delta, end="", flush=True)
                full_response.append(text_delta)
                for chunk in chunker.feed(text_delta):
                    tts_queue.push(chunk)

    def _stream_ollama(self, chunker, tts_queue, full_response: list) -> None:
        import httpx

        url = "http://localhost:11434/api/chat"
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._build_messages())

        payload = {"model": self.model, "messages": messages, "stream": True}

        with httpx.stream("POST", url, json=payload, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_response.append(token)
                    for chunk in chunker.feed(token):
                        tts_queue.push(chunk)
                if data.get("done"):
                    break

    def _stream_claude_code(
        self, user_text: str, chunker, tts_queue, full_response: list
    ) -> None:
        """
        Stream response via `claude --print --output-format stream-json`.

        Key fix: unsets CLAUDECODE env var so the subprocess can launch even
        when called from within a running Claude Code session. Uses
        --no-session-persistence to avoid interfering with the parent session.

        Conversation context is embedded in the prompt text (last 3 turns)
        since we don't reuse sessions.
        """
        # Build prompt with context
        prompt_parts: list[str] = []
        if self._history:
            for msg in self._history[-6:]:  # last 3 user+assistant turns
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt_parts.append(f"{role}: {msg['content']}")
        prompt_parts.append(
            f"User: {_CLAUDE_CODE_PREFIX}{user_text}"
            "\n\nRespond as the assistant — plain spoken English only."
        )
        full_prompt = "\n".join(prompt_parts)

        # Remove CLAUDECODE so we can nest claude CLI under a running session
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        cmd = [
            "claude",
            "--print",
            "--no-session-persistence",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            full_prompt,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError("claude CLI not found — ensure 'claude' is on PATH")

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Plain text fallback
                if line:
                    print(line, end="", flush=True)
                    full_response.append(line)
                    for chunk in chunker.feed(line):
                        tts_queue.push(chunk)
                continue

            event_type = event.get("type", "")

            if event_type == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        token = block.get("text", "")
                        if token:
                            print(token, end="", flush=True)
                            full_response.append(token)
                            for chunk in chunker.feed(token):
                                tts_queue.push(chunk)

            elif event_type == "content_block_delta":
                token = event.get("delta", {}).get("text", "")
                if token:
                    print(token, end="", flush=True)
                    full_response.append(token)
                    for chunk in chunker.feed(token):
                        tts_queue.push(chunk)

            elif event_type == "result":
                if event.get("subtype") == "error":
                    logger.error("claude CLI error: %s", event.get("error", "unknown"))
                # Fallback: if no streaming text arrived, use the result field
                if not full_response:
                    result_text = event.get("result", "")
                    if result_text:
                        print(result_text, end="", flush=True)
                        full_response.append(result_text)
                        for chunk in chunker.feed(result_text):
                            tts_queue.push(chunk)

        proc.wait()
        stderr_out = proc.stderr.read()
        if stderr_out.strip():
            logger.warning("claude CLI stderr: %s", stderr_out[:800].strip())

    # ------------------------------------------------------------------
    # Helpers

    def _build_messages(self) -> list[dict]:
        """Return last N turns of history as a messages list (without system)."""
        return self._history[-20:]

    def _try_build_monitor(self):
        """
        Attempt to create a SpeechMonitor for interrupt detection.
        Returns None (and logs a note) if the audio driver can't support it.
        """
        try:
            from scripts_and_skills.speech.audio_utils import SpeechMonitor
            monitor = SpeechMonitor()
            logger.info("Interrupt-while-speaking enabled (SpeechMonitor ready)")
            return monitor
        except Exception as exc:
            logger.info(
                "Interrupt-while-speaking disabled — SpeechMonitor unavailable: %s", exc
            )
            return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Speech-to-speech voice pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts_and_skills.speech.voice_pipeline
  python -m scripts_and_skills.speech.voice_pipeline --ollama
  python -m scripts_and_skills.speech.voice_pipeline --claude-code
  python -m scripts_and_skills.speech.voice_pipeline --svc --voice en-US-GuyNeural
  python -m scripts_and_skills.speech.voice_pipeline --confirm
  python -m scripts_and_skills.speech.voice_pipeline --silence-ms 1000 --min-words 3

SVC voice setup:
  Set S2S_SVC_REPO, S2S_SVC_MODEL, S2S_SVC_SPK then pass --svc.
  See scripts_and_skills/claude_skills/blues_skills/skills/speech-to-speech/SKILL.md
        """,
    )

    # Backend
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--ollama", action="store_true",
        help="Use local Ollama backend (default model: gpt-oss:20b)"
    )
    backend_group.add_argument(
        "--claude-code", action="store_true",
        help="Use Claude CLI subprocess as backend (unsets CLAUDECODE for nesting)"
    )

    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument(
        "--voice", default=_VOICE,
        help=f"edge-tts voice name (default: {_VOICE})"
    )
    parser.add_argument(
        "--svc", action="store_true",
        help="Enable whisper-vits-svc voice conversion (needs S2S_SVC_* env vars)"
    )
    parser.add_argument("--system", default=None, help="Custom system prompt")
    parser.add_argument(
        "--stop-phrase", default=_DEFAULT_STOP_PHRASE,
        help=f'Spoken phrase that exits voice mode (default: "{_DEFAULT_STOP_PHRASE}")'
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Require voice yes/no confirmation before executing each prompt"
    )
    parser.add_argument(
        "--silence-ms", type=int, default=800,
        help="Milliseconds of silence to end a recording (default: 800)"
    )
    parser.add_argument(
        "--min-words", type=int, default=2,
        help="Minimum word count — shorter transcripts are treated as noise (default: 2)"
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.claude_code:
        backend = "claude-code"
    elif args.ollama:
        backend = "ollama"
    else:
        backend = "anthropic"

    os.environ["S2S_VOICE"] = args.voice

    pipeline = VoicePipeline(
        backend=backend,
        model=args.model,
        system_prompt=args.system,
        use_svc=args.svc,
        stop_phrase=args.stop_phrase,
        confirm_mode=args.confirm,
        silence_ms=args.silence_ms,
        min_words=args.min_words,
    )
    pipeline.run()
