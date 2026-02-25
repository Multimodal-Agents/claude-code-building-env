import sys
import json
import argparse
# Force UTF-8 output — Windows terminals default to cp1252 which can't encode
# many characters that LLMs commonly generate (em-dashes, smart quotes, etc.)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts_and_skills.data.prompt_store import PromptStore

parser = argparse.ArgumentParser(description="View dataset conversations")
parser.add_argument("dataset", nargs="?", default="ollama-interface-training",
                    help="Dataset name to display")
parser.add_argument("--row", type=int, default=None,
                    help="Show only this row index (0-based)")
parser.add_argument("--max", type=int, default=None,
                    help="Max rows to display")
args = parser.parse_args()

store = PromptStore()
rows = store.load(args.dataset)

if rows.empty:
    print(f"No rows found in dataset '{args.dataset}'")
else:
    subset = rows.iloc[[args.row]] if args.row is not None else rows
    if args.max:
        subset = subset.head(args.max)
    for i, row in subset.iterrows():
        print(f"\n{'='*60}")
        print(f"Row {i+1}/{len(rows)} | {row.get('description', '')}")
        print(f"Source: {row.get('source', '')}")
        print(f"{'='*60}")
        convs_raw = row.get("conversations", "")
        if convs_raw and convs_raw not in ("", "null"):
            convs = json.loads(convs_raw)
            for msg in convs:
                speaker = msg["from"].upper()
                print(f"\n[{speaker}]\n{msg['value']}\n")
        else:
            print(f"[INPUT]\n{row.get('input','')}\n")
            if row.get("output"):
                print(f"[OUTPUT]\n{row['output']}\n")
