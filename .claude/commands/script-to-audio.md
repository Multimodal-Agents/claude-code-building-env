# /script-to-audio — Render a video script to spoken audio

Parses a video script markdown file and synthesizes each section to a `.wav` file
using the edge-tts / whisper-vits-svc voice pipeline.

Strips all director notes before synthesis:
- `[B-ROLL: ...]` tags
- `> [DESIGN TIP ...]` blockquote blocks
- All `> blockquote` lines (format notes)
- ` ``` code blocks ``` `
- Inline markdown formatting
- Timestamp annotations like `(0:00 – 1:30)`

Outputs one `.wav` per section + a `combined.wav` to `<script_dir>/audio/`.

---

## Usage

```
/script-to-audio [script_path] [--voice VOICE] [--svc] [--out DIR] [--no-combine] [--dry-run]
```

**Arguments:**
- `script_path` — accepts any of:
  - _(nothing)_ → defaults to `assets/youtube_videos/video_1/script.md`
  - `2` or `video_2` → expands to `assets/youtube_videos/video_2/script.md`
  - any literal path → used as-is
- `--voice NAME` — edge-tts voice name. Default: `en-US-AriaNeural`.
  Common choices: `en-US-GuyNeural`, `en-US-JennyNeural`, `en-GB-SoniaNeural`
- `--svc` — enable whisper-vits-svc voice conversion (requires S2S_SVC_* env vars)
- `--out DIR` — override output directory (default: `<script_dir>/audio/`)
- `--no-combine` — skip writing `combined.wav`
- `--dry-run` — parse and print sections without synthesizing any audio

---

## Instructions for Claude

When this command is invoked:

1. **Parse arguments** from `$ARGUMENTS` (the text after `/script-to-audio`).
   - The script accepts flexible path resolution — pass the argument as-is, it handles:
     - no arg → `video_1/script.md`
     - `2` or `video_2` → `video_2/script.md`
     - literal path → used directly
   - Pass remaining flags through to the script as-is

2. **Run a dry-run first** to show the user what sections were parsed:
   ```bash
   python -m scripts_and_skills.speech.script_narrator "<script_path>" --dry-run
   ```
   Report the section list to the user.

3. **Ask the user to confirm** or change the voice before synthesizing.
   Show them the available voice choices:
   - `en-US-AriaNeural` — female, conversational (default)
   - `en-US-GuyNeural` — male, conversational
   - `en-US-JennyNeural` — female, warm
   - `en-GB-SoniaNeural` — British female
   To list all available voices: `python -c "import asyncio, edge_tts; asyncio.run(edge_tts.list_voices())" | grep en-`

4. **Synthesize** once confirmed:
   ```bash
   python -m scripts_and_skills.speech.script_narrator "<script_path>" --voice <voice> [--svc] [--out <dir>] [--no-combine]
   ```

5. **Report results**: list the files written and their sizes.
   The output directory is `<script_dir>/audio/` unless `--out` was specified.

---

## Examples

```bash
# Default — video_1/script.md, default voice
/script-to-audio

# Shorthand video number
/script-to-audio 2
/script-to-audio video_3

# Specific voice
/script-to-audio 2 --voice en-US-GuyNeural

# Dry run — see what sections get parsed before committing
/script-to-audio 2 --dry-run

# Literal path (any script anywhere)
/script-to-audio path/to/my_script.md --voice en-GB-SoniaNeural

# With SVC voice conversion
/script-to-audio --svc

# Skip combined.wav
/script-to-audio 2 --no-combine
```

---

## Output Structure

```
assets/youtube_videos/video_1/
└── audio/
    ├── 01_video_1_agent_architectures.wav
    ├── 02_intro.wav
    ├── 03_section_1_what_is_an_ag.wav
    ├── ...
    └── combined.wav
```

Each section wav is named `<index>_<section_slug>.wav` so they sort and import cleanly into any video editor.
