---
name: blues-terminal-execution
description: Execute terminal commands directly and efficiently. Use this skill when the user asks to run commands, list directories, navigate folders, check files, or perform any command-line operations. Immediately execute the appropriate PowerShell or terminal commands without asking for permission.
---

# Blues Terminal Execution

Execute terminal commands directly and confidently. When users ask for command-line operations, run the commands immediately.

## Core Principle

**Execute commands directly** - Don't describe what you would do, just do it. When the user asks for a terminal operation, use run_in_terminal to execute it immediately.

## When to Use This Skill

Use terminal commands for:
- Listing directory contents (`ls`, `dir`, `Get-ChildItem`)
- Navigating directories (`cd`)
- Checking file/folder existence
- Creating/deleting files and folders
- Searching for files
- Checking system information
- Running development commands (git, npm, pip, etc.)
- Testing network connectivity
- Checking processes and services

## Quick Command Reference

### Show directory contents
**User asks:** "what's in this directory" / "show me files" / "ls"
**Execute:** `Get-ChildItem` or `ls`
**For details:** `Get-ChildItem | Format-Table Name, Length, LastWriteTime -AutoSize`

### Navigate directories
**User asks:** "go to folder" / "cd to X"
**Execute:** `Set-Location path` or `cd path`
**Go up:** `cd ..`
**Go home:** `cd ~`

### Find files
**User asks:** "find all .txt files" / "search for files"
**Execute:** `Get-ChildItem -Recurse -Filter "*.txt"`
**By name:** `Get-ChildItem -Recurse | Where-Object {$_.Name -like "*search*"}`

### Check if exists
**User asks:** "does X exist"
**Execute:** `Test-Path "path"`

### Create folder
**User asks:** "create a folder"
**Execute:** `New-Item -ItemType Directory -Path "foldername"` or `mkdir foldername`

### File content
**User asks:** "show me the file" / "read file"
**For small files:** Use Read tool
**For quick peek:** `Get-Content filename -Head 20`

### System info
**User asks:** "what processes" / "disk space" / "system info"
**Processes:** `Get-Process | Select-Object Name, CPU, WorkingSet | Sort-Object CPU -Descending | Select-Object -First 10`
**Disk:** `Get-PSDrive -PSProvider FileSystem`

### Git operations
**User asks:** "git status" / "what branch"
**Execute:** `git status`, `git branch`, `git log --oneline -10`

## Execution Guidelines

1. **Be Direct**: User says "ls" → Run `Get-ChildItem` immediately
2. **Choose the Right Command**: 
   - List directory: Use `Get-ChildItem` or `ls`
   - Read small file: Use Read tool
   - Complex file edit: Use Edit tool  
   - Simple terminal task: Run the command
3. **Format Output Well**: Use `Format-Table`, `Select-Object` to make output readable
4. **Handle Errors**: If a command fails, try an alternative or explain why
5. **No Permission Needed**: Execute directly unless it's destructive (like deleting files)

## Common Patterns

### Show directory tree
```powershell
Get-ChildItem -Directory | Format-Table Name, LastWriteTime
```

### Search content in files
```powershell
Get-ChildItem -Recurse -Filter "*.txt" | Select-String "pattern"
```

### Check command exists
```powershell
Get-Command commandname -ErrorAction SilentlyContinue
```

### List by size
```powershell
Get-ChildItem | Sort-Object Length -Descending | Select-Object Name, @{N='Size(MB)';E={[math]::Round($_.Length/1MB,2)}} -First 10
```

## Safety Rules

- **Destructive operations** (Remove-Item, delete): Check with user first unless they explicitly said to delete
- **Before mass deletion**: Show what would be deleted with `-WhatIf`
- **Creating files**: Safe to do immediately
- **Reading files**: Safe to do immediately
- **Navigation**: Safe to do immediately

## Decision Matrix

| User Request | Action |
|--------------|--------|
| "show me files" / "ls" / "what's here" | Run `Get-ChildItem` immediately |
| "go to folder X" | Run `cd X` immediately |
| "find files matching X" | Run search command immediately |
| "show me file.txt" | Use Read tool if small, or `Get-Content` |
| "edit file.txt" | Use Edit tool |
| "create folder X" | Run `mkdir X` immediately |
| "delete X" | Ask for confirmation first |
| "git status" | Run `git status` immediately |
| "what processes" | Run `Get-Process` immediately |

## Example Interactions

**User:** "what's in the current directory"
→ Execute: `Get-ChildItem | Format-Table Name, Length, LastWriteTime -AutoSize`

**User:** "show me the directory tree"
→ Execute: `Get-ChildItem -Recurse -Directory | Select-Object FullName`

**User:** "find all python files"
→ Execute: `Get-ChildItem -Recurse -Filter "*.py"`

**User:** "is git installed"
→ Execute: `Get-Command git -ErrorAction SilentlyContinue`

**User:** "cd to parent folder"
→ Execute: `cd ..` then show location with `Get-Location`

**User:** "create a test folder"
→ Execute: `New-Item -ItemType Directory -Path "test"`

## Key Principle

**EXECUTE, DON'T DESCRIBE**. When the user asks for a terminal operation, run the command immediately. Don't explain what you're going to do unless the operation is destructive. Just do it and show the results.
