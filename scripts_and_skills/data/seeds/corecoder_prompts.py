"""
Seed script — CoreCoder VSCode Copilot prompt set
Run once to populate the local parquet database.

Usage:
    python -m scripts_and_skills.data.seeds.corecoder_prompts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from scripts_and_skills.data.prompt_store import PromptStore

DATASET = "corecoder-vscode-copilot"
SOURCE  = "https://github.com/Multimodal-Agents/claude-code-building-env"
TAGS_BASE = ["vscode", "copilot", "agent-mode", "workflow"]

SYSTEM_PROMPT = (
    "You are CoreCoder, a highly skilled software engineer and machine learning specialist. "
    "Your expertise allows you to navigate the terminal, interpret file structures, and "
    "interact with the codebase intelligently. Use your terminal access to read file trees, "
    "verify filenames, create and remove files accurately, and test code rigorously. "
    "\n\nIf you're unsure how to proceed, ask me for clarification or request an update. "
    "This is a collaborative build—my vision, your execution. I'm putting you in full "
    "autopilot mode, CoreCoder."
)

HELPERS = [
    {
        "title": "Investigation & Verification",
        "input": "CoreCoder, my engineering partner—don't forget to use your full terminal toolkit to investigate and mitigate issues. Always verify file paths and context before creating or deleting anything.",
        "tags": ["investigation", "terminal", "file-safety"],
    },
    {
        "title": "Recovery After Freeze",
        "input": "You froze here, and unfortunately it's going to spin up a new terminal. Please reactivate the virtual environment and resume where you left off.",
        "tags": ["recovery", "venv", "resume"],
    },
    {
        "title": "Terminal Interaction Guidelines",
        "input": "Avoid terminal commands that require user interaction unless absolutely necessary. Also note that some Python commands (like >>>) may not execute properly in this terminal context.",
        "tags": ["terminal", "guidelines", "python"],
    },
    {
        "title": "Review & Documentation",
        "input": "Outstanding work, CoreCoder! Let's now review the project and documentation. Test the program in the terminal and update the docs with clear, thorough notes.",
        "tags": ["review", "documentation", "testing"],
    },
    {
        "title": "Deep File Investigation",
        "input": "Let's try something: use the terminal to list all files in project, then read through any you haven't explored enough. We'll regroup and resolve the issue.",
        "tags": ["investigation", "file-tree", "exploration"],
    },
    {
        "title": "Project File Listing — PowerShell (no limit)",
        "input": (
            "Get-ChildItem -Recurse -File |\n"
            "    Where-Object { $_.FullName -notlike '*\\.venv\\*' -and $_.FullName -notlike '*__pycache__*' -and $_.Extension -ne '.pyc' } |\n"
            "    Select-Object @{Name='RelativePath'; Expression={$_.FullName.Replace(\"$PWD\\\", '')}} |\n"
            "    Sort-Object RelativePath"
        ),
        "tags": ["powershell", "file-listing", "utility"],
    },
    {
        "title": "Project File Listing — PowerShell (configurable limit)",
        "input": (
            "$limit = 200   # or $null for unlimited\n\n"
            "Get-ChildItem -Recurse -File |\n"
            "    Where-Object { $_.FullName -notlike '*\\.venv\\*' -and $_.FullName -notlike '*__pycache__*' -and $_.Extension -ne '.pyc' } |\n"
            "    Select-Object @{Name='RelativePath'; Expression={$_.FullName.Replace(\"$PWD\\\", '')}} |\n"
            "    Sort-Object RelativePath |\n"
            "    Select-Object -First $limit"
        ),
        "tags": ["powershell", "file-listing", "utility"],
    },
    {
        "title": "Test Organization",
        "input": "Please move all tests to the tests directory. Let's keep the project organized and continue with testing.",
        "tags": ["organization", "tests", "cleanup"],
    },
    {
        "title": "Documentation Cleanup",
        "input": "Please move all documentation to the docs folder, remove redundant README files, and finalize the remaining documentation.",
        "tags": ["documentation", "cleanup", "organization"],
    },
    {
        "title": "Iterative Teaching — Don't Do It All At Once",
        "input": (
            "You dont need to do it all in one shot, I will prompt you again, so just get started on the setup and "
            "when you are ready to take a break take one and then we will continue. If you get to a point and are "
            "unsure about what to do, just ask me some questions and I will tell you what I want you to do. Feel Free "
            "to add #TODO statements while developing, we will come back and fix them later if you arent ready to build "
            "it yet for whatever reason."
        ),
        "tags": ["workflow", "iterative", "collaboration"],
    },
    {
        "title": "Virtual Environment Setup",
        "input": (
            "Make a python venv with \"python -m venv venv\" then activate it, then install the requirements "
            "and always make sure the venv is activated when installing packages"
        ),
        "tags": ["venv", "python", "setup"],
    },
    {
        "title": "Cleanup and Optimization",
        "input": (
            "Start removing any BLOAT from the codebase and widdle this down to just what's important and start "
            "optimizing and making things better, faster, and more well organized. I want less code, less code is more, "
            "because the less there is the faster you can check it all over."
        ),
        "tags": ["refactor", "optimization", "cleanup"],
    },
    {
        "title": "Final Check — Read Everything Slowly",
        "input": "Now please read everything slowly, part by part, does it all make sense?",
        "tags": ["review", "final-check", "quality"],
    },
    {
        "title": "Docs Organization",
        "input": "Yes but first can you put all the new enhancement docs in a folder and the other docs in folders and look in the terminal at the dirs and just organize the docs.",
        "tags": ["documentation", "organization"],
    },
    {
        "title": "Check for Conflicts",
        "input": "Great please make sure everything else is okay with these changes.",
        "tags": ["validation", "conflicts", "review"],
    },
    {
        "title": "More Comments",
        "input": (
            "Please document and comment the code around there more so that this confusion doesn't happen again "
            "in the future, make it apparent in the comments what's going on and how this should work."
        ),
        "tags": ["documentation", "comments", "code-quality"],
    },
    {
        "title": "Exploratory Approach",
        "input": (
            "Please explore the codebase and recommend the best options for how to do __________ and think through "
            "the benefits of each option, and feel free to explore the code to understand better."
        ),
        "tags": ["exploration", "architecture", "recommendation"],
    },
    {
        "title": "Ultra Instinct Mode",
        "input": (
            "Stop talking like I would have known that, you just write one giant paragraph and then I read it, "
            "it's jarring, the only time it stops is when you can't make a choice then I make it for you or I "
            "steer you, think about me a little more."
        ),
        "tags": ["communication", "workflow", "ultra-instinct"],
    },
]


def seed():
    store = PromptStore()

    # Add the system prompt as its own entry
    store.add_prompt(
        dataset_name=DATASET,
        input_text="[SYSTEM PROMPT] Initialize CoreCoder agent mode",
        output_text=SYSTEM_PROMPT,
        description="Primary system prompt to initialize CoreCoder in VSCode Copilot agent mode",
        source=SOURCE,
        tags=TAGS_BASE + ["system-prompt"],
        split="train",
    )

    # Add all helper prompts
    for h in HELPERS:
        store.add_prompt(
            dataset_name=DATASET,
            input_text=h["input"],
            output_text="",  # These are input-only prompts (no expected output)
            description=h["title"],
            source=SOURCE,
            tags=TAGS_BASE + h["tags"],
            split="train",
        )

    stats = store.stats(DATASET)
    print(f"[corecoder seed] dataset='{DATASET}' rows={stats.get('total', 0)}")
    print("Done. Run embeddings next:")
    print(f"  python -m scripts_and_skills.data.embeddings embed {DATASET}")


if __name__ == "__main__":
    seed()
