# Custom Skills Marketplace Template

This is a template repository for creating custom Claude Code skill marketplaces.

## Structure

```
template_skills/
├── .claude-plugin/
│   └── marketplace.json          # Marketplace configuration
├── skills/
│   └── example-skill/
│       └── SKILL.md              # Skill definition
└── README.md                     # This file
```

## How to Use This Template

1. **Clone or copy this directory** to create your own skill marketplace
2. **Edit marketplace.json**:
   - Change `name` to your marketplace name (lowercase-with-hyphens)
   - Update `owner` information
   - Update `metadata.description`
   - Add/modify plugins and their skills

3. **Create your skills**:
   - Each skill needs its own folder under `skills/`
   - Each skill folder must contain a `SKILL.md` file
   - Follow the YAML frontmatter format:
     ```markdown
     ---
     name: my-skill-name
     description: Clear description of what the skill does
     ---
     
     # Skill Instructions
     
     Your instructions here...
     ```

4. **Add skills to marketplace.json**:
   - List each skill path in the `skills` array
   - Group related skills into plugins

## Installing Your Marketplace

### From Local Directory
```powershell
claude plugin marketplace add D:\path\to\your-skills
claude plugin install plugin-name@your-marketplace-name
```

### From GitHub (After Publishing)
```powershell
claude plugin marketplace add username/repo-name
claude plugin install plugin-name@your-marketplace-name
```

## Skill Structure Reference

### Minimal SKILL.md
```markdown
---
name: skill-name
description: What this skill does and when to use it
---

# Instructions for Claude

Your detailed instructions here...
```

### With Optional Fields
```markdown
---
name: skill-name
description: What this skill does and when to use it
license: Complete terms in LICENSE.txt
version: 1.0.0
---

# Your Skill Content

Instructions, examples, guidelines...
```

## Tips for Creating Good Skills

1. **Clear Description**: Make the description specific about when Claude should use this skill
2. **Detailed Instructions**: Provide step-by-step guidance for Claude
3. **Examples**: Include concrete examples of how to use the skill
4. **Resources**: Add any scripts, templates, or reference materials in the skill folder
5. **Test Thoroughly**: Test your skills before sharing

## Learn More

- [Agent Skills Spec](https://agentskills.io/specification)
- [Anthropic Skills Examples](https://github.com/anthropics/skills)
- [Creating Custom Skills Guide](https://support.claude.com/en/articles/12512198-creating-custom-skills)
