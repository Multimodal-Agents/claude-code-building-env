# Blues Skills

Custom skill marketplace for Blues-specific workflows and capabilities.

## Structure

```
blues_skills/
├── .claude-plugin/
│   └── marketplace.json          # Marketplace configuration
├── skills/
│   └── (your custom skills here)
└── README.md                     # This file
```

## Installing Blues Skills

### Local Installation
```powershell
claude plugin marketplace add D:\PowerShell_Scripts\claude_skills\blues_skills
claude plugin install blues-core@blues-skills
```

### During Development
```powershell
# Load specific skill directly while developing
claude --plugin-dir D:\PowerShell_Scripts\claude_skills\blues_skills\skills\skill-name
```

## Skills Included

### blues-terminal-execution
Expert terminal command execution with PowerShell mastery. Provides intelligent command suggestions, safety checks, and shortcuts for common operations like navigation, file management, system status, and development tasks. Gives Claude full autonomy to execute commands efficiently while maintaining safety.

## Creating New Skills

1. Create a new folder under `skills/` with your skill name
2. Add a `SKILL.md` file with the skill definition:
   ```markdown
   ---
   name: your-skill-name
   description: What this skill does
   ---
   
   # Skill Instructions
   
   Your instructions here...
   ```
3. Add the skill path to `marketplace.json` under the `skills` array:
   ```json
   "skills": [
     "./skills/your-skill-name"
   ]
   ```

## Quick Reference

### Skill Template
```markdown
---
name: skill-name
description: Clear description of what the skill does and when to use it
---

# Skill Name

## Overview
What this skill does

## Instructions
Step-by-step guidance

## Examples
Concrete examples

## Guidelines
Best practices
```

### Adding Scripts
- Place Python scripts in `skills/your-skill/scripts/`
- Place templates in `skills/your-skill/templates/`
- Reference them in your SKILL.md

### Testing Your Skills
```powershell
# Test a single skill
claude --plugin-dir D:\PowerShell_Scripts\claude_skills\blues_skills\skills\your-skill

# Ask Claude to use it
# Then inside Claude session, just mention the skill naturally
```

## Learn More

- [Agent Skills Spec](https://agentskills.io/specification)
- [Template Skills](../template_skills/) - Use this as a reference
- [Anthropic Skills Examples](../skills/) - More examples
