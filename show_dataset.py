import json
from scripts_and_skills.data.prompt_store import PromptStore

store = PromptStore()
rows = store.load("ollama-interface-training")
for i, row in rows.iterrows():
    print(f"{'='*60}")
    print(f"Row {i+1} | {row['description']}")
    print(f"{'='*60}")
    convs = json.loads(row["conversations"])
    for msg in convs:
        speaker = msg["from"].upper()
        print(f"\n[{speaker}]\n{msg['value']}\n")
