# /blender-robot — Generate Blender Python scripts for robot design

Generates complete `bpy` scripts you paste into **Blender's Scripting workspace**
and run with **Alt+P**. No plugins or pip installs required. Output is
3D-print-ready geometry organised into collections, with optional STL export.

---

## What you get

- A **self-contained Python script** (paste into Text Editor → Run Script)
- Parametric: all key dimensions are variables at the top — edit and re-run
- Geometry scaled in **millimetres** (`scale_length = 0.001`)
- 3D-print-aware: wall thickness, overhang notes, joint clearances
- Structured into **Blender collections** (BODY, ARM, LEG, JOINTS, etc.)
- Optional per-part **STL export** (set `EXPORT_STL = True`)

---

## How to run the generated script in Blender

1. Open Blender 4.x
2. Click the **Scripting** tab at the top (or press `Shift+F11` for Text Editor)
3. Click **New** to create a blank text block
4. Paste the entire generated script
5. Press **Alt+P** or click the **▶ Run Script** button
6. The scene is cleared and the robot is rebuilt from scratch (safe to re-run)

---

## Usage

```
/blender-robot [DESCRIPTION]
```

If `DESCRIPTION` is provided, Claude skips intake questions already answered.

---

## Examples

```
/blender-robot 30cm humanoid robot, FDM printing, articulated arms and legs
/blender-robot 6-DOF desk robot arm, 20cm reach, functional revolute joints
/blender-robot hexapod spider, 6 legs (3 segments each), 15cm body, FDM
/blender-robot tank chassis with rotating turret, 200mm long, static (no joints)
/blender-robot just the joint library — show me all 4 joint types
```

---

## Instructions for Claude

When this command is invoked:

1. **Parse `$ARGUMENTS`** — extract any description and options.

2. **Run intake** (skip questions already answered in `$ARGUMENTS`):

   **Tier 1 — Always ask if not provided:**
   - Robot type: humanoid / robot arm / spider-hexapod / tank-wheeled / custom
   - Size: overall height or reach (e.g. "30 cm tall", "20 cm arm reach")
   - Scene: empty .blend or existing geometry? If existing: describe what's there
   - Printing: FDM (nozzle size, default 0.4 mm) or resin? Bed size?

   **Tier 2 — Ask only if ambiguous:**
   - Detail level: block-out / medium / high (with functional joints)
   - Joints: articulated (real clearance gaps) or static (fused, single print)?
   - Aesthetic: industrial / sci-fi sleek / chunky mech / minimalist

3. **Print a Design Brief** (one paragraph confirming the interpretation).

4. **Generate the script** using the templates and patterns from:
   ```
   scripts_and_skills/claude_skills/blues_skills/skills/blender-robot/
   ├── SKILL.md               ← full API reference and rules
   └── templates/
       ├── scene_setup.py     ← utility library
       ├── humanoid_robot.py  ← humanoid template
       ├── robot_arm.py       ← 6-DOF arm template
       └── joint_library.py   ← 4 joint types
   ```

5. **Wrap the script** in a code block:
   ````
   ```python
   # script here
   ```
   ````

6. **After the script**, print a **Next Steps** section:
   - How to run it (Scripting tab → paste → Alt+P)
   - What to check first (proportions, then tell Claude what to change)
   - How to export (set `EXPORT_STL = True`, update `EXPORT_DIR`)
   - 3D print check: Edit Mode → Mesh → Cleanup → Fill Holes; 3D Print Toolbox → Check All

7. **Iteration mode** — if the user says "change X" or "the Y is too big":
   - Generate a **delta script** that only rebuilds the modified objects
   - Start the delta script with a comment: `# DELTA: rebuilds [object names]`
   - Include deletion of the old objects before rebuilding:
     ```python
     for name in ['torso', 'head']:
         if name in bpy.data.objects:
             bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
     ```

---

## Template reference

| Template | Use for |
|----------|---------|
| `scene_setup.py` | Utility belt — paste at top of any custom script |
| `humanoid_robot.py` | Bipedal robot, full body, proportional constants |
| `robot_arm.py` | 6-DOF desk arm with gripper |
| `joint_library.py` | Ball/socket, pin, snap-fit, living hinge |

For spider/hexapod, tank, or custom types — adapt the nearest template and
apply the anatomy constants from SKILL.md Section 5.
