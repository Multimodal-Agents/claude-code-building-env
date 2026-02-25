# Skills Directory

This directory will contain your custom Blues skills.

## Creating a New Skill

1. Create a folder with your skill name (lowercase-with-hyphens)
2. Add a `SKILL.md` file inside
3. Update `../.claude-plugin/marketplace.json` to include your skill

Example structure:
```
skills/
├── my-first-skill/
│   ├── SKILL.md
│   ├── scripts/       (optional)
│   └── templates/     (optional)
└── my-second-skill/
    └── SKILL.md
```

## Skill Template

```markdown
---
name: my-skill-name
description: What this skill does and when Claude should use it
---

# My Skill Name

Your instructions for Claude...
```
