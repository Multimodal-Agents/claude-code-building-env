---
name: speech-to-speech
description: >
  Real-time voice conversation pipeline. Triggers when the user mentions:
  voice, speech, speak, listen, microphone, talk to Claude, voice pipeline,
  speech-to-speech, S2S, TTS, STT, whisper, edge-tts, SVC, voice conversion.
---

# Speech-to-Speech Skill

Real-time pipeline: **Mic → Whisper STT → Claude → edge-tts → speaker**
Optional: whisper-vits-svc voice conversion layer between TTS and playback.

---

## Pipeline Architecture

```
[Mic] → [VAD + 800ms silence debounce] → [faster-whisper small]
                                                    ↓ text
                                         [optional confirm mode: TTS readback + yes/no]
                                                    ↓ confirmed text
                                         [Anthropic SDK / Ollama / claude CLI stream]
                                                    ↓ token stream
                                         [StreamingChunker: buffer → sentence chunks]
                                                    ↓ chunk queue
                                  ┌─────────────────┴──────────────────────┐
                          [edge-tts → .wav]                    [playing previous chunk]
                                  ↓                            (interruptible)
                    [svc_inference.py → converted .wav]  (optional, skipped if not set up)
                                  ↓
                             [play audio]  ←── [SpeechMonitor VAD background thread]
                                                        ↓ speech detected mid-playback
                                                   [interrupt TTS → drain buffer → re-record]
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r scripts_and_skills/speech/requirements.txt
```

On Windows, `webrtcvad` may need the Visual C++ build tools:
```bash
pip install webrtcvad-wheels  # pre-built wheels for Windows
```

### 2. Verify components individually

```bash
# STT test (record + transcribe once)
python -m scripts_and_skills.speech.stt --test

# TTS test (synthesize + play a sentence)
python -m scripts_and_skills.speech.tts "Hello, this is a test." --voice en-US-AriaNeural

# Chunker smoke test
python -c "
from scripts_and_skills.speech.text_chunker import TextChunker
chunks = TextChunker().chunk('Hello! How are you? This is sentence three.')
print(chunks)
"
```

### 3. Run the full pipeline

```bash
# Anthropic backend (default)
python -m scripts_and_skills.speech.voice_pipeline

# Local Ollama backend
python -m scripts_and_skills.speech.voice_pipeline --ollama

# With SVC voice conversion (after setup below)
python -m scripts_and_skills.speech.voice_pipeline --svc
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `S2S_BACKEND` | `anthropic` | LLM backend: `anthropic` or `ollama` |
| `S2S_MODEL` | `claude-sonnet-4-6` / `gpt-oss:20b` | Model override |
| `S2S_VOICE` | `en-US-AriaNeural` | edge-tts voice name |
| `S2S_SVC` | `0` | Set to `1` to enable SVC |
| `S2S_SVC_REPO` | _(empty)_ | Path to whisper-vits-svc repo |
| `S2S_SVC_MODEL` | _(empty)_ | Path to `.pth` model file |
| `S2S_SVC_SPK` | _(empty)_ | Path to speaker `.npy` embedding |

---

## CLI Options

```
python -m scripts_and_skills.speech.voice_pipeline [OPTIONS]

Backend (mutually exclusive):
  --ollama           Use local Ollama (gpt-oss:20b)
  --claude-code      Use claude CLI subprocess — works inside Claude Code sessions

Model / voice:
  --model NAME       Override model
  --voice NAME       edge-tts voice (default: en-US-AriaNeural)
  --svc              Enable whisper-vits-svc voice conversion

Behaviour:
  --confirm          Require voice yes/no confirmation before each prompt
  --stop-phrase PHR  Exit phrase (default: "goodbye claude")
  --silence-ms N     ms of silence to end recording (default: 800)
  --min-words N      Discard transcripts shorter than N words (default: 2)
  --system PROMPT    Custom system prompt
  --log-level LVL    DEBUG | INFO | WARNING | ERROR (default: WARNING)
```

### Available edge-tts voices (sample)

```bash
python -c "import asyncio, edge_tts; asyncio.run(edge_tts.list_voices())" | grep en-US
```

Common choices:
- `en-US-AriaNeural` — female, conversational
- `en-US-GuyNeural` — male, conversational
- `en-US-JennyNeural` — female, warm
- `en-GB-SoniaNeural` — British female

---

## whisper-vits-svc Setup (Optional)

Skip this section unless you want custom voice conversion.

### Step 1: Clone the repo

```bash
git clone https://github.com/PlayVoice/whisper-vits-svc.git
cd whisper-vits-svc
pip install -r requirements.txt
```

### Step 2: Download pretrained models

Five files are required in the repo root / `pretrain/` folder:

| File | Source |
|------|--------|
| `sovits5.0.pth` | [HuggingFace PlayVoice/vits-svc-5.0](https://huggingface.co/PlayVoice/vits-svc-5.0) |
| `whisper-small.pt` | OpenAI Whisper (auto-downloads on first use) |
| `hubert-soft-0d54a1f4.pt` | [bshall/hubert-soft](https://github.com/bshall/hubert-soft/releases) |
| `checkpoint_best_legacy_500.pt` | [contentvec](https://github.com/auspicious3000/contentvec) |
| `singer.spk.npy` | Pretrained speaker or your fine-tuned speaker embedding |

### Step 3: Test SVC inference

```bash
cd whisper-vits-svc
python svc_inference.py \
  --config configs/base.yaml \
  --model sovits5.0.pth \
  --spk singer.spk.npy \
  --wav test_input.wav \
  --out test_output.wav
```

---

## Training Your Own Voice (Optional)

### Step 1: Record & clean raw audio

- Record 30–60 minutes of clean speech (same speaker, quiet room)
- Use [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) to remove background noise

### Step 2: Slice into segments

```bash
# Install slicer
pip install audio-slicer

# Slice into 5–15 second clips
python slicer2.py \
  --input raw_audio/ \
  --output dataset_raw/my_voice/ \
  --min_interval 100 --hop_size 10
```

### Step 3: Organize dataset structure

```
whisper-vits-svc/
└── dataset_raw/
    └── my_voice/           # speaker name (used as --spk value)
        ├── clip_001.wav
        ├── clip_002.wav
        └── ...
```

### Step 4: Preprocess

```bash
python preprocess.py -c configs/base.yaml -d dataset_raw
```

### Step 5: Train

```bash
python train.py -c configs/base.yaml -m my_voice
```

Fine-tuning from pretrained (recommended, much faster):
```bash
python train.py -c configs/base.yaml -m my_voice \
  --pretrain sovits5.0.pth
```

### Step 6: Export speaker embedding

```bash
python spk_extract.py \
  --config configs/base.yaml \
  --model logs/my_voice/G_latest.pth \
  --out my_voice.spk.npy
```

### Step 7: Run with your voice

```bash
S2S_SVC_REPO=/path/to/whisper-vits-svc \
S2S_SVC_MODEL=/path/to/whisper-vits-svc/logs/my_voice/G_latest.pth \
S2S_SVC_SPK=/path/to/my_voice.spk.npy \
python -m scripts_and_skills.speech.voice_pipeline --svc
```

---

## Text Chunker Rules

The `StreamingChunker` splits the LLM token stream into TTS-friendly phrases:

**Split boundaries:**
- After `.`, `!`, `?` followed by whitespace
- Double newline (paragraph break)
- Before numbered list items (`1. `, `2. ` etc.)

**Stripped characters** (markdown noise):
```
* # ` _ ~ [ ] { } | \ < > @
```

**Merge / split limits:**
- Chunks < 15 chars are merged with the next chunk
- Chunks > 250 chars are hard-split at the nearest word boundary

---

## Module Reference

| Module | Key Classes/Functions |
|--------|-----------------------|
| `audio_utils` | `record_with_vad()`, `record_with_vad_continue()`, `play_wav()`, `play_wav_interruptible()`, `save_temp_wav()`, `SpeechMonitor` |
| `stt` | `SpeechRecognizer.transcribe()`, `.listen_and_transcribe()` |
| `text_chunker` | `TextChunker.chunk()`, `StreamingChunker.feed()/.flush()` |
| `tts` | `TTSEngine.synthesize()`, `TTSQueue.push()/.wait_done()/.is_idle()/.interrupt()` |
| `voice_pipeline` | `VoicePipeline.run()` |

---

## Troubleshooting

**`webrtcvad` install fails on Windows**
```bash
pip install webrtcvad-wheels
```

**No audio device found**
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**edge-tts outputs MP3 instead of WAV**
- Install ffmpeg (recommended): `winget install ffmpeg`
- Or install pydub fallback: `pip install pydub`

**faster-whisper CUDA errors**
```bash
# Force CPU mode
python -m scripts_and_skills.speech.voice_pipeline --log-level DEBUG
# Or set env: CUDA_VISIBLE_DEVICES=-1
```

**SVC model not found**
- Verify `S2S_SVC_REPO`, `S2S_SVC_MODEL`, `S2S_SVC_SPK` are all set
- Check paths exist before starting the pipeline

**`--claude-code` shows "Assistant:" with empty response**
- Root cause: Claude Code blocks nested CLI launches via `CLAUDECODE` env var
- Fix: already applied in `_stream_claude_code()` — it unsets `CLAUDECODE` from the subprocess env and adds `--no-session-persistence`
- If still empty, run with `--log-level DEBUG` to see stderr from the subprocess

**Interrupt-while-speaking not working**
- Some Windows audio drivers (MME, DirectSound) don't allow simultaneous input+output streams
- Switch to WASAPI exclusive in Windows Sound settings, or use ASIO
- The pipeline disables interrupt automatically if `SpeechMonitor` fails to open — normal listening still works
