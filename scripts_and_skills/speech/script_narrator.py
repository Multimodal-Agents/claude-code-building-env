"""
script_narrator.py — Parse a video script markdown file and render it to audio.

Strips director notes ([B-ROLL: ...], [DESIGN TIP] blocks, code blocks, blockquotes)
and synthesizes the spoken text section-by-section using TTSEngine.

Output
------
  <out_dir>/
    01_intro.wav
    02_section_1.wav
    ...
    combined.wav   (all sections concatenated with 1s silence between)

Usage
-----
  python -m scripts_and_skills.speech.script_narrator <script.md> [options]

  Options:
    --voice NAME      edge-tts voice (default: en-US-AriaNeural)
    --svc             Enable whisper-vits-svc voice conversion
    --out DIR         Output directory (default: <script_dir>/audio)
    --no-combine      Skip writing combined.wav
    --dry-run         Print parsed sections without synthesizing
"""

import argparse
import logging
import os
import re
import shutil
import sys
import wave
from dataclasses import dataclass, field
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
# Script Parser
# ---------------------------------------------------------------------------

@dataclass
class Section:
    name: str          # e.g. "INTRO", "SECTION_1"
    slug: str          # filesystem-safe name
    text: str          # cleaned spoken text


class ScriptParser:
    """
    Parse a video script markdown file into speakable sections.

    Strips:
      - [B-ROLL: ...] lines and inline tags
      - > [DESIGN TIP ...] blockquote blocks
      - All other > blockquotes (format guide notes)
      - ``` ... ``` code blocks
      - ` inline code `
      - Markdown formatting: #, *, _, ~, |, etc.
      - --- dividers
      - Timestamp annotations like "(0:00 – 1:30)"
      - Empty sections

    Keeps:
      - Section headings (used as section names)
      - All spoken paragraph text
    """

    # Patterns
    _RE_BROLL_LINE    = re.compile(r"^\s*\[B-ROLL[^\]]*\]\s*$", re.IGNORECASE)
    _RE_BROLL_INLINE  = re.compile(r"\[B-ROLL[^\]]*\]", re.IGNORECASE)
    _RE_HEADING       = re.compile(r"^#{1,6}\s+(.*)")
    _RE_TIMESTAMP     = re.compile(r"\(\d+:\d+\s*[–—-]+\s*\d+:\d+\)")
    _RE_DIVIDER       = re.compile(r"^\s*-{3,}\s*$")
    _RE_CLEAN_MD      = re.compile(r"[*_~`|\\<>]")
    _RE_WHITESPACE    = re.compile(r"[ \t]+")

    def parse(self, path: str | Path) -> list[Section]:
        text = Path(path).read_text(encoding="utf-8")
        lines = text.splitlines()

        sections: list[Section] = []
        current_name = "intro"
        current_lines: list[str] = []

        in_code_block = False
        in_blockquote_skip = False  # for multi-line blockquote blocks

        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Code block fence ---
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                i += 1
                continue
            if in_code_block:
                i += 1
                continue

            # --- Blockquotes (> ...) — skip entirely ---
            # This catches: format guide notes, [DESIGN TIP] blocks
            if line.startswith(">"):
                i += 1
                continue

            # --- Dividers ---
            if self._RE_DIVIDER.match(line):
                i += 1
                continue

            # --- [B-ROLL: ...] standalone lines ---
            if self._RE_BROLL_LINE.match(line):
                i += 1
                continue

            # --- Section headings ---
            m = self._RE_HEADING.match(line)
            if m:
                # Save previous section
                section = self._make_section(current_name, current_lines)
                if section:
                    sections.append(section)
                current_name = m.group(1).strip()
                current_lines = []
                i += 1
                continue

            # --- Regular content line ---
            cleaned = self._clean_line(line)
            if cleaned:
                current_lines.append(cleaned)

            i += 1

        # Final section
        section = self._make_section(current_name, current_lines)
        if section:
            sections.append(section)

        return sections

    def _clean_line(self, line: str) -> str:
        # Strip inline [B-ROLL: ...] tags
        line = self._RE_BROLL_INLINE.sub("", line)
        # Strip timestamps like (0:00 – 1:30)
        line = self._RE_TIMESTAMP.sub("", line)
        # Strip markdown formatting characters
        line = self._RE_CLEAN_MD.sub("", line)
        # Collapse whitespace
        line = self._RE_WHITESPACE.sub(" ", line).strip()
        return line

    def _make_section(self, name: str, lines: list[str]) -> Optional[Section]:
        text = " ".join(l for l in lines if l).strip()
        if not text:
            return None
        slug = re.sub(r"[^\w]+", "_", name.lower()).strip("_")[:40]
        return Section(name=name, slug=slug, text=text)


# ---------------------------------------------------------------------------
# Script Narrator
# ---------------------------------------------------------------------------

class ScriptNarrator:
    """
    Synthesize a list of Section objects to audio files.

    Parameters
    ----------
    voice      : edge-tts voice name
    use_svc    : enable whisper-vits-svc conversion
    out_dir    : directory to write wav files into
    combine    : whether to write a combined.wav
    """

    SILENCE_BETWEEN_SECTIONS_MS = 1000  # 1 second gap

    def __init__(
        self,
        voice: str = "en-US-AriaNeural",
        use_svc: bool = False,
        out_dir: str | Path = "audio",
        combine: bool = True,
    ) -> None:
        from scripts_and_skills.speech.tts import TTSEngine

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.combine = combine
        self.engine = TTSEngine(
            voice=voice,
            use_svc=use_svc,
            svc_repo=os.getenv("S2S_SVC_REPO", ""),
            svc_model=os.getenv("S2S_SVC_MODEL", ""),
            svc_speaker=os.getenv("S2S_SVC_SPK", ""),
        )

    def narrate(self, sections: list[Section]) -> list[Path]:
        """Synthesize all sections. Returns list of written wav paths."""
        written: list[Path] = []

        for idx, section in enumerate(sections, start=1):
            out_path = self.out_dir / f"{idx:02d}_{section.slug}.wav"
            print(f"  [{idx:02d}/{len(sections)}] {section.name} → {out_path.name}")
            logger.info("Synthesizing section %d: %s (%d chars)",
                        idx, section.name, len(section.text))

            try:
                tmp = self.engine.synthesize(section.text)
                if tmp:
                    shutil.move(tmp, out_path)
                    written.append(out_path)
                    print(f"         ✓ saved ({out_path.stat().st_size // 1024} KB)")
                else:
                    print(f"         ⚠ empty synthesis for section: {section.name}")
            except Exception as exc:
                logger.error("Failed to synthesize section %d: %s", idx, exc)
                print(f"         ✗ error: {exc}")

        if self.combine and written:
            combined_path = self._combine(written)
            if combined_path:
                print(f"\n  Combined → {combined_path}")
                written.append(combined_path)

        return written

    def _combine(self, wav_paths: list[Path]) -> Optional[Path]:
        """Concatenate wav files with silence gaps into combined.wav."""
        out_path = self.out_dir / "combined.wav"
        try:
            # Read first file to get params
            with wave.open(str(wav_paths[0]), "rb") as w:
                params = w.getparams()
                rate = params.framerate
                width = params.sampwidth
                channels = params.nchannels

            silence_frames = int(rate * self.SILENCE_BETWEEN_SECTIONS_MS / 1000)
            silence_bytes = b"\x00" * (silence_frames * width * channels)

            with wave.open(str(out_path), "wb") as out_wav:
                out_wav.setparams(params)
                for i, path in enumerate(wav_paths):
                    with wave.open(str(path), "rb") as src:
                        # Re-open to verify compatible params
                        out_wav.writeframes(src.readframes(src.getnframes()))
                    if i < len(wav_paths) - 1:
                        out_wav.writeframes(silence_bytes)

            return out_path

        except Exception as exc:
            logger.error("Failed to combine wav files: %s", exc)
            print(f"  ⚠ Could not combine: {exc}")
            print("    Tip: ensure all section wavs have identical sample rate/channels.")
            return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VIDEOS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "youtube_videos"


def _resolve_script(raw: str | None) -> Path:
    """
    Resolve the script path from a variety of shorthand inputs:

      None / ""        → assets/youtube_videos/video_1/script.md  (default)
      "2" or "video_2" → assets/youtube_videos/video_2/script.md
      any/path.md      → treated as a literal path
    """
    if not raw:
        return _VIDEOS_ROOT / "video_1" / "script.md"

    # Pure integer shorthand: "2" → video_2
    if raw.isdigit():
        return _VIDEOS_ROOT / f"video_{raw}" / "script.md"

    # "video_N" shorthand (with or without leading assets/... prefix)
    m = re.fullmatch(r"video[_-]?(\d+)", raw, re.IGNORECASE)
    if m:
        return _VIDEOS_ROOT / f"video_{m.group(1)}" / "script.md"

    # Literal path
    return Path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a video script markdown file to audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
script argument accepts:
  (none)       default — video_1/script.md
  2            shorthand — video_2/script.md
  video_3      shorthand — video_3/script.md
  path/to/script.md   literal path
        """,
    )
    parser.add_argument(
        "script", nargs="?", default=None,
        help="Script path, video number (e.g. 2), or video name (e.g. video_3). "
             "Default: video_1/script.md"
    )
    parser.add_argument(
        "--voice", default=os.getenv("S2S_VOICE", "en-US-AriaNeural"),
        help="edge-tts voice name (default: en-US-AriaNeural)"
    )
    parser.add_argument("--svc", action="store_true", help="Enable SVC voice conversion")
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: <script_dir>/audio)"
    )
    parser.add_argument("--no-combine", action="store_true", help="Skip combined.wav")
    parser.add_argument("--dry-run", action="store_true", help="Print sections, no audio")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s"
    )

    script_path = _resolve_script(args.script)
    if not script_path.exists():
        print(f"Error: script not found: {script_path}", file=sys.stderr)
        print(f"  Videos root: {_VIDEOS_ROOT}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else script_path.parent / "audio"

    # --- Parse ---
    print(f"\nParsing script: {script_path}")
    sections = ScriptParser().parse(script_path)
    print(f"Found {len(sections)} speakable sections:\n")
    for i, s in enumerate(sections, 1):
        preview = s.text[:80].replace("\n", " ")
        print(f"  {i:02d}. {s.name:<35} {len(s.text):>5} chars  \"{preview}...\"")

    if args.dry_run:
        print("\nDry run — no audio synthesized.")
        return

    # --- Narrate ---
    print(f"\nOutput → {out_dir}/")
    narrator = ScriptNarrator(
        voice=args.voice,
        use_svc=args.svc,
        out_dir=out_dir,
        combine=not args.no_combine,
    )
    written = narrator.narrate(sections)
    print(f"\nDone. {len(written)} file(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
