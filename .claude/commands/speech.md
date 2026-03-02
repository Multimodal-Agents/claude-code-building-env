# /speech — Start real-time speech-to-speech voice mode

Launches the full voice pipeline: **Mic → VAD → Whisper → LLM → edge-tts → speaker**

**Features:**
- **Interrupt while speaking** — just start talking and the assistant stops mid-sentence
- **Confirm mode** — voice yes/no confirmation before each prompt executes
- **Stop phrase** — say "goodbye claude" to return to text chat
- **SVC voice** — whisper-vits-svc converts edge-tts to a realistic trained voice

---

## Usage

```
/speech [OPTIONS]
```

**Backend (pick one):**
- _(none)_ — Anthropic API (default; uses `ANTHROPIC_API_KEY`)
- `--claude-code` — Claude CLI subprocess (works inside Claude Code, unsets `CLAUDECODE` automatically)
- `--ollama` — local Ollama (`gpt-oss:20b`)

**All options:**
- `--voice NAME` — edge-tts voice. Common: `en-US-GuyNeural`, `en-US-JennyNeural`, `en-GB-SoniaNeural`
- `--svc` — enable whisper-vits-svc voice conversion (requires `S2S_SVC_*` env vars)
- `--confirm` — require voice yes/no confirmation before executing each prompt
- `--stop-phrase "PHRASE"` — custom exit phrase (default: `"goodbye claude"`)
- `--model NAME` — override LLM model
- `--system PROMPT` — custom system prompt
- `--silence-ms N` — ms of silence to end a recording (default: 800)
- `--min-words N` — discard transcripts shorter than N words, prevents noise (default: 2)
- `--log-level LEVEL` — DEBUG | INFO | WARNING | ERROR

---

## Instructions for Claude

When this command is invoked:

1. **Parse arguments** from `$ARGUMENTS`. Extract and pass through any flags.

2. **Tell the user** the stop phrase and any active modes before launching.

3. **Run the pipeline**:
   ```bash
   cd "M:/claude_code_building_env" && python -m scripts_and_skills.speech.voice_pipeline [flags]
   ```

4. **Default behaviour** (no args):
   ```bash
   cd "M:/claude_code_building_env" && python -m scripts_and_skills.speech.voice_pipeline
   ```

5. Once the pipeline exits, inform the user they're back in text chat.

---

## Examples

```bash
# Claude Code backend (inside a Claude Code session)
/speech --claude-code

# Default — Anthropic, AriaNeural
/speech

# Local Ollama model
/speech --ollama

# Different voice
/speech --voice en-US-JennyNeural

# With voice confirmation before each prompt
/speech --confirm

# SVC realistic voice (requires S2S_SVC_* setup)
/speech --svc

# Custom stop phrase
/speech --stop-phrase "exit voice"

# Full featured
/speech --claude-code --voice en-US-GuyNeural --confirm --svc

# Slower talker — extend silence window
/speech --silence-ms 1200
```

---

## Interrupt While Speaking

While the assistant is talking, just start speaking. The pipeline will:
1. Detect your voice via background VAD
2. Stop TTS playback immediately
3. Record your new utterance
4. Process it as the next prompt

Works automatically if your audio driver supports simultaneous input + output streams
(Windows WASAPI and macOS CoreAudio both do). Disabled automatically if not supported.

---

## Confirm Mode

With `--confirm`, after each transcript the assistant reads back the first few words
and waits for "yes" or "no" before executing.

**Accepted yes words:** yes, yeah, yep, yup, sure, correct, right, go, confirm, affirmative
**Accepted no words:** no, nope, cancel, negative, stop, abort, wrong, incorrect, don't, nah

---

## Stop Phrase Matching

The stop phrase check is fuzzy — punctuation is stripped and it matches anywhere in the sentence.
So `"goodbye claude"` triggers on:
- "Goodbye, Claude."
- "Okay goodbye Claude!"
- "I think goodbye Claude is what I want"

Choose a phrase unlikely to appear in normal conversation.
