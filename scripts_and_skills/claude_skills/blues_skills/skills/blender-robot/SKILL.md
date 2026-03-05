---
name: blender-robot
triggers:
  - "blender"
  - "bpy"
  - "robot design"
  - "3d print robot"
  - "robot model"
  - "blender script"
  - "STL export"
  - "robot arm"
  - "humanoid robot"
  - "hexapod"
  - "FDM robot"
  - "blender python"
description: >
  Blender Python (bpy) scripting skill for robot design and 3D-print-ready geometry.
  Triggers when the user mentions Blender, bpy, robot design, 3D printing robots,
  STL export, or asks to generate robot geometry. Produces complete bpy scripts the
  user pastes into Blender's Text Editor and runs. Handles scene setup, robot anatomy,
  joint design, collections, manifold geometry checks, and STL export.
---

# Blender Robot Design Skill

You generate self-contained Python scripts the user pastes into Blender's **Scripting
workspace** (Text Editor) and runs with **Alt+P**. No plugins, no pip installs — only
the standard `bpy`, `bmesh`, and `mathutils` modules that ship with Blender.

Target: **Blender 4.x** (API stable since 3.6 LTS; note 4.2 removed `export_mesh.stl`).
All output geometry is **3D-print-ready**: manifold, scaled in millimeters, and designed
to the user's stated printer type (FDM or resin).

---

## Phase 0 — Intake

Run this intake before writing any code. Skip any question the user has already answered.

### Tier 1 — Always ask (blocking)

1. **Robot type**: humanoid / robot arm / spider-hexapod / tank-wheeled / custom
2. **Size**: overall height or reach (e.g. "30 cm tall", "20 cm arm reach")
3. **Scene state**: empty .blend or existing geometry? If existing: what objects are present?
4. **Printing**: FDM (nozzle size? default 0.4 mm) or resin? Printer bed size?

### Tier 2 — Ask if ambiguous

5. **Detail level**: block-out (rough proportions) / medium (panel lines, shaped limbs) / high (functional joints, fine detail)
6. **Joints**: articulated (clearance gaps for real movement) or static (fused, single print)?
7. **Aesthetic**: industrial / sci-fi sleek / chunky mech / organic-mechanical / minimalist

### Tier 3 — Optional

8. **Emphasis**: any part to focus on? ("detailed gripper", "elaborate head sensor array")
9. **Export**: STL for slicing / OBJ for rendering / keep as .blend

After intake, print a one-paragraph **Design Brief** confirming the interpretation,
then write the script.

---

## Phase 1 — Scene Setup Rules

Every generated script begins with these operations (inline, not imported):

```python
import bpy, bmesh, math, os
from mathutils import Vector, Matrix, Euler

# ── UNITS: 1 Blender Unit = 1 mm ─────────────────────────────────────────────
scene = bpy.context.scene
scene.unit_settings.system       = 'METRIC'
scene.unit_settings.scale_length = 0.001   # critical for correct STL export scale
scene.unit_settings.length_unit  = 'MILLIMETERS'

# ── CLEAR SCENE ───────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for block in bpy.data.meshes:
    if block.users == 0:
        bpy.data.meshes.remove(block)
```

`scale_length = 0.001` is mandatory. Without it, a 30 mm cube appears as a 30-meter
structure in every slicer.

---

## Phase 2 — Collection Management

All robot parts go into a named collection hierarchy. Never leave objects in the
scene root.

```python
def get_or_create_collection(name: str,
                              parent: bpy.types.Collection = None
                              ) -> bpy.types.Collection:
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col

def link_to(obj: bpy.types.Object, col: bpy.types.Collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
```

Standard hierarchy for a humanoid:
```
ROBOT_<name>/
├── BODY/       — torso, pelvis, head, neck
├── LEFT_ARM/   — upper arm, forearm, hand
├── RIGHT_ARM/
├── LEFT_LEG/   — thigh, shin, foot
├── RIGHT_LEG/
└── JOINTS/     — all ball/pin/socket pieces
```

For a robot arm: `BASE / LINK_1 / LINK_2 ... / WRIST / GRIPPER`.
For a hexapod: `BODY / LEG_0 ... LEG_5` with `COXA / FEMUR / TIBIA` inside each.

---

## Phase 3 — Primitive Helpers

Always define these at the top of each generated script so it is self-contained:

```python
def add_box(name, w, h, d, loc=(0,0,0), rot=(0,0,0)):
    """w=width(X), h=height(Z), d=depth(Y). Dimensions in mm."""
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = (w/2, d/2, h/2)     # default cube is 2x2x2 BU
    bpy.ops.object.transform_apply(scale=True)
    return o

def add_cyl(name, radius, depth, verts=32, loc=(0,0,0), rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, vertices=verts,
        location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    return o

def add_sphere(name, radius, segs=32, rings=16, loc=(0,0,0)):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, segments=segs, ring_count=rings, location=loc)
    o = bpy.context.active_object
    o.name = name
    return o

def bool_diff(base, cutter, delete_cutter=True):
    """Subtract cutter from base. Returns base."""
    mod = base.modifiers.new('BoolDiff', 'BOOLEAN')
    mod.operation  = 'DIFFERENCE'
    mod.solver     = 'EXACT'
    mod.object     = cutter
    bpy.context.view_layer.objects.active = base
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if delete_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return base

def bool_union(a, b, result_name=None):
    mod = a.modifiers.new('BoolUnion', 'BOOLEAN')
    mod.operation = 'UNION'
    mod.solver    = 'EXACT'
    mod.object    = b
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(b, do_unlink=True)
    if result_name:
        a.name = result_name
    return a

def cleanup_mesh(obj):
    """Remove doubles and fix normals — call after booleans."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
```

---

## Phase 4 — 3D Print Rules (Non-Negotiable)

These constraints must be reflected in every dimension constant. If a user-specified
size would violate them, emit a `# WARNING:` comment in the script.

### Wall Thickness

| Printer | Minimum | Structural | Load-bearing |
|---------|---------|-----------|--------------|
| FDM 0.4 mm nozzle | 1.2 mm | 2.0 mm | 3.0+ mm |
| FDM 0.6 mm nozzle | 1.8 mm | 2.4 mm | 4.0+ mm |
| Resin | 0.8 mm | 1.5 mm | 2.5+ mm |

Define `FDM_MIN_WALL = 1.2` (or `RESIN_MIN_WALL = 0.8`) at the top of each script
and reference it in dimension constants: `SHELL_THICKNESS = max(2.0, FDM_MIN_WALL)`.

### Overhangs

- FDM: no face angled more than 45° from vertical without support or chamfer.
- Design chamfers (45°) into the model rather than relying on slicer supports wherever
  possible. Note print orientation in comments at the end of the script.

### Clearance for Moving Joints

```python
JOINT_CLEARANCE   = 0.3   # mm radial, for ball-and-socket
PIN_CLEARANCE     = 0.2   # mm diametric (bore = pin_dia + PIN_CLEARANCE)
SNAP_CLEARANCE    = 0.15  # mm, tight snap-fit
SLIDE_CLEARANCE   = 0.2   # mm each face, prismatic joint
```

### Manifold Output

After every boolean operation, call `cleanup_mesh(obj)`. Always check for open edges
before export. Tell the user to run: **Edit Mode → Mesh → Cleanup → Fill Holes**
and the **3D Print Toolbox addon** (bundled with Blender) → **Check All**.

### Minimum Feature Size

- FDM: pins/lugs minimum 0.8 mm (2 nozzle passes). Do not generate smaller detail.
- Resin: 0.3 mm for feature detail, 0.5 mm for structural features.

---

## Phase 5 — Robot Anatomy Catalog

### 5A — Humanoid (reference: 30 cm total height)

```python
TOTAL_H   = 300.0
HEAD_H    = TOTAL_H * 0.13   # 39 mm
NECK_H    = TOTAL_H * 0.04   # 12 mm
TORSO_H   = TOTAL_H * 0.30   # 90 mm
PELVIS_H  = TOTAL_H * 0.08   # 24 mm
UPPER_LEG = TOTAL_H * 0.23   # 69 mm
LOWER_LEG = TOTAL_H * 0.22   # 66 mm
FOOT_L    = TOTAL_H * 0.14   # 42 mm
SHLD_W    = TOTAL_H * 0.28   # 84 mm, shoulder-to-shoulder
UPPER_ARM = TOTAL_H * 0.18   # 54 mm
FOREARM   = TOTAL_H * 0.16   # 48 mm
HAND_L    = TOTAL_H * 0.12   # 36 mm
```

Build order: pelvis → torso → head → right arm → mirror left arm → right leg → mirror left leg → joints.

Mirror pattern (apply after building the right side):
```python
def mirror_object(src_obj, name_prefix='L_'):
    bpy.ops.object.select_all(action='DESELECT')
    src_obj.select_set(True)
    bpy.context.view_layer.objects.active = src_obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.name = name_prefix + src_obj.name
    dup.scale.x *= -1
    bpy.ops.object.transform_apply(scale=True)
    # Recalculate normals after mirroring (they flip with scale.x=-1)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    return dup
```

### 5B — Robot Arm (6-DOF, desk-mounted)

```python
BASE_RADIUS  = 40.0
BASE_H       = 20.0
LINK_LENGTHS = [120.0, 100.0, 60.0]   # shoulder→elbow, elbow→wrist, wrist→tool
JOINT_R      = 22.0
WRIST_R      = 15.0
GRIPPER_OPEN = 50.0    # jaw spread at max
GRIPPER_STRK = 25.0    # travel per jaw
```

Chain links vertically: each link's base center = previous link's tip center.
```python
z = BASE_H
for length in LINK_LENGTHS:
    link = add_box(f"link_{i}", w=18, h=length, d=14, loc=(0, 0, z + length/2))
    z += length
```

### 5C — Spider / Hexapod

```python
BODY_L      = 120.0
BODY_W      = 80.0
BODY_H      = 30.0
LEG_COUNT   = 6
COXA_L      = 25.0     # hip segment
FEMUR_L     = 60.0     # thigh
TIBIA_L     = 80.0     # shin
COXA_R      = 8.0
FEMUR_R     = 7.0
TIBIA_R     = 5.0
LEG_SPREAD  = 30.0     # degrees outward from body
```

Symmetric leg placement:
```python
for i in range(LEG_COUNT):
    angle = math.radians(i * (360 / LEG_COUNT))
    mount_x = math.cos(angle) * (BODY_L / 2 * 0.8)
    mount_y = math.sin(angle) * (BODY_W / 2 * 0.8)
    build_leg(index=i, mount_loc=(mount_x, mount_y, 0), base_angle=angle)
```

### 5D — Tank / Wheeled

```python
CHASSIS_L    = 200.0
CHASSIS_W    = 120.0
CHASSIS_H    = 40.0
TRACK_W      = 25.0
WHEEL_R      = 30.0
WHEEL_THICK  = 20.0
ROAD_WHEELS  = 4        # per side
TURRET_R     = 45.0
BARREL_R     = 8.0
BARREL_L     = 80.0
```

---

## Phase 6 — Joint Library

### Type 1: Ball and Socket

```python
def build_ball_joint(center, ball_r, clearance=0.3):
    """
    Returns (ball_obj, socket_obj).
    Ball prints flat-side-down. Socket prints open-side-up.
    """
    # Ball: full sphere with flat base cut
    ball = add_sphere('ball', radius=ball_r, loc=center)
    cutter = add_box('_cut', w=ball_r*3, h=ball_r, d=ball_r*3,
                     loc=(center[0], center[1], center[2] - ball_r/2))
    ball = bool_diff(ball, cutter)

    # Socket: hemisphere + retention lip, slightly larger than ball
    sock_r = ball_r + clearance
    socket = add_sphere('socket_outer', radius=sock_r, loc=center)
    inner  = add_sphere('socket_inner', radius=ball_r, loc=center)
    socket = bool_diff(socket, inner, delete_cutter=True)
    # Keep only bottom hemisphere
    top_cut = add_box('_topcut', w=sock_r*3, h=sock_r, d=sock_r*3,
                      loc=(center[0], center[1], center[2] + sock_r/2))
    socket = bool_diff(socket, top_cut)
    socket.name = 'socket'

    cleanup_mesh(ball)
    cleanup_mesh(socket)
    return ball, socket
```

**Print notes:** Ball prints flat base down (no supports). Socket prints open side up.
Assembly: press ball into socket; retention lip snaps over equator.

### Type 2: Revolute (Pin Joint)

```python
def build_pin_joint(center, pin_r, depth, clearance=0.2):
    """
    Returns (pin_obj, bore dimensions dict).
    Bore must be cut into the receiving part with bore_r = pin_r + clearance/2.
    """
    bore_r = pin_r + clearance / 2
    pin = add_cyl('pin', radius=pin_r, depth=depth,
                  loc=center, rot=(math.pi/2, 0, 0))
    # Chamfer ends for self-supporting print
    # (manual: bevel the two end edge loops in Edit Mode, or add bevel modifier)
    bevel = pin.modifiers.new('Bevel', 'BEVEL')
    bevel.width        = pin_r * 0.25
    bevel.segments     = 2
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit  = math.radians(80)
    return pin, {'bore_radius': bore_r, 'bore_depth': depth + 0.5}
```

**Use case:** elbow, knee, wrist — single-axis rotation.

### Type 3: Snap-Fit Clip

```python
SNAP_CLEARANCE = 0.15   # tight; use 0.3 for loose

def snap_fit_tab(name, width, height, thickness, clearance=SNAP_CLEARANCE):
    """Flexible clip tab. Only suitable for PETG or TPU (PLA too brittle)."""
    # Rectangular tab with angled lead-in (30 deg) for self-release
    tab = add_box(name, w=width, h=height, d=thickness)
    # Lead-in wedge: cut a triangle from the tip
    wedge = add_box(f'_{name}_wedge', w=width*1.2, h=height*0.4, d=thickness*0.6,
                    loc=(0, 0, height*0.3))
    # Rotate wedge 30 deg around X to make the angled lead-in
    wedge.rotation_euler.x = math.radians(30)
    bpy.ops.object.transform_apply(rotation=True)
    tab = bool_diff(tab, wedge)
    return tab
```

### Type 4: Living Hinge (Flexure)

```python
def living_hinge(name, width, panel_length, flex_thickness=0.6):
    """
    Two rectangular panels connected by a thin flex strip.
    ONLY use PETG or TPU. Print flat (hinge horizontal, parallel to bed).
    flex_thickness MUST be >= 0.6 mm for FDM; layer height <= 0.15 mm recommended.
    """
    if flex_thickness < 0.6:
        print(f"WARNING: flex_thickness {flex_thickness} mm is below FDM minimum 0.6 mm")
    panel_a = add_box(f'{name}_panel_a', w=width, h=panel_length, d=2.0,
                      loc=(0, -panel_length/2 - flex_thickness/2, 0))
    flex    = add_box(f'{name}_flex',    w=width, h=flex_thickness, d=2.0,
                      loc=(0, 0, 0))
    panel_b = add_box(f'{name}_panel_b', w=width, h=panel_length, d=2.0,
                      loc=(0, panel_length/2 + flex_thickness/2, 0))
    # Union into one piece
    result = bool_union(panel_a, flex)
    result = bool_union(result, panel_b, result_name=name)
    return result
```

---

## Phase 7 — STL Export Pattern

Always guard export behind a flag so the script is safe to run repeatedly:

```python
EXPORT_STL  = False   # ← set True to export
EXPORT_DIR  = "C:/Users/YourName/Desktop/robot_stls/"

def export_all_meshes(export_dir):
    os.makedirs(export_dir, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.name.startswith('_'):
            continue

        # Evaluate with modifiers applied
        obj_eval  = obj.evaluated_get(depsgraph)
        mesh_copy = bpy.data.meshes.new_from_object(obj_eval)
        temp      = bpy.data.objects.new(obj.name + '_exp', mesh_copy)
        bpy.context.scene.collection.objects.link(temp)

        bpy.ops.object.select_all(action='DESELECT')
        temp.select_set(True)
        bpy.context.view_layer.objects.active = temp

        fp = os.path.join(export_dir, obj.name + '.stl')
        try:
            # Blender 4.0+ API
            bpy.ops.wm.stl_export(
                filepath=fp,
                ascii_format=False,
                apply_modifiers=True,
                export_selected_objects=True,
                global_scale=1.0,
            )
        except AttributeError:
            # Blender 3.x fallback
            bpy.ops.export_mesh.stl(
                filepath=fp, use_selection=True, global_scale=1.0)

        bpy.data.objects.remove(temp, do_unlink=True)
        bpy.data.meshes.remove(mesh_copy)
        print(f"Exported: {fp}")

if EXPORT_STL:
    export_all_meshes(EXPORT_DIR)
```

---

## Phase 8 — Script Output Format

Every generated script must start with this header:

```python
"""
ROBOT SCRIPT: <descriptive name>
Skill: blender-robot | Target: Blender 4.x | Units: millimeters

HOW TO RUN:
  1. Open Blender 4.x
  2. Open Scripting workspace (top menu: Scripting tab, or Shift+F11)
  3. New text → paste this entire script
  4. Press Alt+P or click the ▶ Run Script button
  5. Script clears the scene and rebuilds from scratch (safe to re-run)

PARAMETERS — edit here, then re-run:
"""

# ══ PARAMETERS ════════════════════════════════════════════════════════════════
TOTAL_HEIGHT_MM  = 300.0
JOINT_CLEARANCE  = 0.3
FDM_MIN_WALL     = 1.2
EXPORT_STL       = False
EXPORT_DIR       = "C:/Users/YourName/Desktop/robot_stls/"
# ══════════════════════════════════════════════════════════════════════════════
```

All tunable values go in the `PARAMETERS` block. No magic numbers buried in functions.

---

## Phase 9 — Iteration Protocol

### Round 1 — Block-out
Generate a minimal script: simple boxes/cylinders at correct proportions. No joints, no detail.
Tell the user: *"Run this, look at the proportions, tell me what to change."*

### Round 2 — Refinement
Generate a **delta script**: only rebuilds the objects that need changing. Starts with:
```python
# DELTA: rebuilds [torso, head] — run after the Round 1 script
# Delete only the objects we're replacing:
for name in ['torso', 'head']:
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
```

### Round 3 — Joints
Add articulation: ball joints, pin joints, or static fillets. Ask the user to check
clearances in Blender before printing.

### Round 4 — Print Prep
Apply all modifiers, run cleanup, export per-part STLs. Append a part manifest:
```python
"""
PART MANIFEST:
  torso.stl         — print upright, no supports needed
  head.stl          — print face-up, supports for visor recess
  upper_arm_R.stl   — print vertically, shoulder joint socket face-up
  ball_shoulder_R.stl — print flat side down, no supports
  ...
"""
```

---

## Common Mistakes to Avoid

| Mistake | Fix |
|---------|-----|
| Using `bpy.ops.transform.resize()` without selecting | Always `obj.select_set(True)` + set active before any op |
| Boolean on non-manifold mesh → garbage result | Call `cleanup_mesh()` before every boolean |
| Exporting without modifier evaluation | Use `evaluated_get(depsgraph)` pattern |
| Wall dimension below FDM_MIN_WALL | Clamp: `w = max(dim, FDM_MIN_WALL)` |
| `bpy.ops.export_mesh.stl` in Blender 4.2+ | Use `wm.stl_export` with try/except fallback |
| Not setting `scale_length = 0.001` | Slicer sees the model as meters — always include scene setup |
| Hemisphere as a mesh cap → open edge | Use boolean to cut sphere, not a plane-based cap |
| Mirroring without recalculating normals | After `scale.x *= -1`, run `normals_make_consistent()` |
| Applying booleans before mirror modifier | Apply Mirror first, then booleans |
| Joint clearance hardcoded | Always use `JOINT_CLEARANCE` variable, defined once at top |
