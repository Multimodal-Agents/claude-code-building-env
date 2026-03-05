"""
ROBOT SCRIPT: Parametric Humanoid Robot
Skill: blender-robot | Target: Blender 4.x | Units: millimeters

HOW TO RUN:
  1. Open Blender 4.x
  2. Open Scripting workspace (Scripting tab at top, or Shift+F11)
  3. New text block → paste this entire script
  4. Press Alt+P (or ▶ Run Script button)
  5. Robot is built in the scene. Safe to re-run — clears scene each time.

WHAT YOU GET:
  A blocky humanoid robot with these parts, each in its own collection:
    ROBOT/BODY/   — head, neck, torso, pelvis
    ROBOT/L_ARM/  — upper arm, forearm, hand (left)
    ROBOT/R_ARM/  — upper arm, forearm, hand (right)
    ROBOT/L_LEG/  — thigh, shin, foot (left)
    ROBOT/R_LEG/  — thigh, shin, foot (right)

AFTER RUNNING:
  - Check proportions in the 3D viewport
  - Tell Claude what to change; it will generate a delta script
  - To 3D print: set EXPORT_STL = True and update EXPORT_DIR

PRINT NOTES (FDM, per part):
  torso       — print upright. No supports needed.
  pelvis      — print upright.
  head        — print face-up. Supports for visor recess.
  upper_arm_* — print vertically (shoulder up).
  forearm_*   — print vertically.
  hand_*      — print flat (palm down).
  thigh_*     — print vertically.
  shin_*      — print vertically.
  foot_*      — print flat (sole down).
"""

# ══ PARAMETERS — edit here, then re-run ══════════════════════════════════════
ROBOT_NAME      = "Robot_01"
TOTAL_HEIGHT_MM = 300.0    # overall height from foot to crown
STYLE           = "blocky"  # "blocky" | "slim" — controls aspect ratios
JOINT_CLEARANCE = 0.3      # mm, articulated joint gap
FDM_MIN_WALL    = 1.2      # mm, hard minimum wall thickness
EXPORT_STL      = False    # set True to export each part as a separate STL
EXPORT_DIR      = "C:/Users/YourName/Desktop/robot_stls/"
# ══════════════════════════════════════════════════════════════════════════════

import bpy, bmesh, math, os
from mathutils import Vector

# ─── Proportions (derived from TOTAL_HEIGHT_MM) ───────────────────────────────
H = TOTAL_HEIGHT_MM
W_FACTOR = 1.15 if STYLE == "blocky" else 0.85   # overall width multiplier

HEAD_H    = H * 0.13
HEAD_W    = H * 0.11 * W_FACTOR
HEAD_D    = H * 0.10 * W_FACTOR
NECK_H    = H * 0.04
NECK_R    = H * 0.025

TORSO_H   = H * 0.30
TORSO_W   = H * 0.22 * W_FACTOR
TORSO_D   = H * 0.14 * W_FACTOR
PELVIS_H  = H * 0.08
PELVIS_W  = H * 0.20 * W_FACTOR
PELVIS_D  = H * 0.14 * W_FACTOR

SHLD_W    = H * 0.28 * W_FACTOR  # shoulder-to-shoulder (X span)
UARM_H    = H * 0.18
UARM_W    = H * 0.07 * W_FACTOR
UARM_D    = H * 0.06 * W_FACTOR
FARM_H    = H * 0.16
FARM_W    = H * 0.055 * W_FACTOR
FARM_D    = H * 0.050 * W_FACTOR
HAND_H    = H * 0.045
HAND_W    = H * 0.065 * W_FACTOR
HAND_D    = H * 0.030

THIGH_H   = H * 0.23
THIGH_W   = H * 0.075 * W_FACTOR
THIGH_D   = H * 0.070 * W_FACTOR
SHIN_H    = H * 0.22
SHIN_W    = H * 0.060 * W_FACTOR
SHIN_D    = H * 0.055 * W_FACTOR
FOOT_H    = H * 0.040
FOOT_W    = H * 0.070 * W_FACTOR
FOOT_L    = H * 0.140     # foot length (Y axis)

HIP_X     = PELVIS_W / 2 * 0.75   # distance of hip joint from centre
LEG_GAP   = 0.5                   # mm gap between thigh and pelvis


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_collection(name, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col

def link_to(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)

def add_box(name, w, h, d, loc=(0,0,0), rot=(0,0,0)):
    """All dims in mm."""
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = (max(w, FDM_MIN_WALL)/2, max(d, FDM_MIN_WALL)/2, max(h, FDM_MIN_WALL)/2)
    bpy.ops.object.transform_apply(scale=True)
    return o

def add_cyl(name, radius, depth, verts=32, loc=(0,0,0), rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=max(radius, FDM_MIN_WALL/2),
                                        depth=depth, vertices=verts,
                                        location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    return o

def cleanup_mesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

def mirror_r_to_l(obj, col_l, prefix='L_'):
    """Mirror a right-side object to create the left-side counterpart."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.name = prefix + obj.name.replace('R_', '')
    dup.scale.x *= -1
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    link_to(dup, col_l)
    return dup


# ─────────────────────────────────────────────────────────────────────────────
#  SCENE SETUP
# ─────────────────────────────────────────────────────────────────────────────

scene = bpy.context.scene
scene.unit_settings.system       = 'METRIC'
scene.unit_settings.scale_length = 0.001
scene.unit_settings.length_unit  = 'MILLIMETERS'

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for block in bpy.data.meshes:
    if block.users == 0:
        bpy.data.meshes.remove(block)

# Collections
col_root  = get_or_create_collection(ROBOT_NAME)
col_body  = get_or_create_collection('BODY',  col_root)
col_r_arm = get_or_create_collection('R_ARM', col_root)
col_l_arm = get_or_create_collection('L_ARM', col_root)
col_r_leg = get_or_create_collection('R_LEG', col_root)
col_l_leg = get_or_create_collection('L_LEG', col_root)


# ─────────────────────────────────────────────────────────────────────────────
#  BODY
# ─────────────────────────────────────────────────────────────────────────────

# Stack from bottom: foot_bottom → shin → thigh → pelvis → torso → neck → head
# Z=0 is the ground plane (sole of foot).

FOOT_Z    = FOOT_H / 2
SHIN_Z    = FOOT_H + SHIN_H / 2
THIGH_Z   = FOOT_H + SHIN_H + THIGH_H / 2
PELVIS_Z  = FOOT_H + SHIN_H + THIGH_H + PELVIS_H / 2
TORSO_Z   = PELVIS_Z + PELVIS_H / 2 + TORSO_H / 2
NECK_Z    = TORSO_Z + TORSO_H / 2 + NECK_H / 2
HEAD_Z    = NECK_Z  + NECK_H  / 2 + HEAD_H / 2

# Torso
torso = add_box('torso', TORSO_W, TORSO_H, TORSO_D, loc=(0, 0, TORSO_Z))
# Chest panel cutout detail
panel_cutter = add_box('_chest_cut', TORSO_W * 0.55, TORSO_H * 0.35, 1.5,
                        loc=(0, TORSO_D/2, TORSO_Z + TORSO_H * 0.05))
bpy.ops.object.select_all(action='DESELECT')
torso.select_set(True)
bpy.context.view_layer.objects.active = torso
mod = torso.modifiers.new('BoolChest', 'BOOLEAN')
mod.operation = 'DIFFERENCE'; mod.solver = 'EXACT'; mod.object = panel_cutter
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(panel_cutter, do_unlink=True)
cleanup_mesh(torso)
link_to(torso, col_body)

# Pelvis
pelvis = add_box('pelvis', PELVIS_W, PELVIS_H, PELVIS_D, loc=(0, 0, PELVIS_Z))
link_to(pelvis, col_body)

# Neck
neck = add_cyl('neck', NECK_R, NECK_H, loc=(0, 0, NECK_Z))
link_to(neck, col_body)

# Head (slightly tapered box with visor recess)
head = add_box('head', HEAD_W, HEAD_H, HEAD_D, loc=(0, 0, HEAD_Z))
visor = add_box('_visor_cut', HEAD_W * 0.70, HEAD_H * 0.30, 2.0,
                loc=(0, HEAD_D/2, HEAD_Z + HEAD_H * 0.05))
mod = head.modifiers.new('BoolVisor', 'BOOLEAN')
mod.operation = 'DIFFERENCE'; mod.solver = 'EXACT'; mod.object = visor
bpy.context.view_layer.objects.active = head
bpy.ops.object.modifier_apply(modifier=mod.name)
bpy.data.objects.remove(visor, do_unlink=True)
cleanup_mesh(head)
link_to(head, col_body)


# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT ARM  (positive X side; mirrored to left afterwards)
# ─────────────────────────────────────────────────────────────────────────────

ARM_X    = SHLD_W / 2                     # X offset from centre
UARM_Z   = TORSO_Z + TORSO_H/2 - UARM_H/2 - 2   # just below shoulder top
FARM_Z   = UARM_Z  - UARM_H/2 - FARM_H/2
HAND_Z   = FARM_Z  - FARM_H/2 - HAND_H/2

r_upper_arm = add_box('R_upper_arm', UARM_W, UARM_H, UARM_D,
                       loc=(ARM_X, 0, UARM_Z))
link_to(r_upper_arm, col_r_arm)

r_forearm = add_box('R_forearm', FARM_W, FARM_H, FARM_D,
                    loc=(ARM_X, 0, FARM_Z))
link_to(r_forearm, col_r_arm)

r_hand = add_box('R_hand', HAND_W, HAND_H, HAND_D,
                 loc=(ARM_X, 0, HAND_Z))
link_to(r_hand, col_r_arm)

# Mirror to left arm
mirror_r_to_l(r_upper_arm, col_l_arm)
mirror_r_to_l(r_forearm,   col_l_arm)
mirror_r_to_l(r_hand,      col_l_arm)


# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT LEG  (positive X side; mirrored to left afterwards)
# ─────────────────────────────────────────────────────────────────────────────

# Thigh sits under pelvis, offset to the hip position
r_thigh = add_box('R_thigh', THIGH_W, THIGH_H, THIGH_D,
                  loc=(HIP_X, 0, THIGH_Z))
link_to(r_thigh, col_r_leg)

r_shin = add_box('R_shin', SHIN_W, SHIN_H, SHIN_D,
                 loc=(HIP_X, 0, SHIN_Z))
link_to(r_shin, col_r_leg)

# Foot offset slightly forward (Y) to give realistic stance
r_foot = add_box('R_foot', FOOT_W, FOOT_H, FOOT_L,
                 loc=(HIP_X, FOOT_L * 0.15, FOOT_Z))
link_to(r_foot, col_r_leg)

# Mirror to left leg
mirror_r_to_l(r_thigh, col_l_leg)
mirror_r_to_l(r_shin,  col_l_leg)
mirror_r_to_l(r_foot,  col_l_leg)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
print(f"  {ROBOT_NAME}  built successfully")
print(f"  Total height: {TOTAL_HEIGHT_MM:.0f} mm")
print(f"  Style: {STYLE}")
print(f"  Joint clearance: {JOINT_CLEARANCE} mm")
print("═" * 60)
print("\nNEXT STEPS:")
print("  1. Inspect proportions in the 3D viewport.")
print("  2. Tell Claude what to change; a delta script will update only those parts.")
print("  3. To export STL: set EXPORT_STL = True and update EXPORT_DIR, then re-run.")
print("\nPRINT CHECK — run these in Blender:")
print("  Edit Mode → Mesh → Cleanup → Fill Holes")
print("  3D Print Toolbox (addon) → Check All")


# ─────────────────────────────────────────────────────────────────────────────
#  STL EXPORT  (runs only if EXPORT_STL = True)
# ─────────────────────────────────────────────────────────────────────────────

def export_all_meshes(export_dir):
    os.makedirs(export_dir, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in list(bpy.data.objects):
        if obj.type != 'MESH' or obj.name.startswith('_'):
            continue
        obj_eval  = obj.evaluated_get(depsgraph)
        mesh_copy = bpy.data.meshes.new_from_object(obj_eval)
        temp      = bpy.data.objects.new(obj.name + '__exp', mesh_copy)
        bpy.context.scene.collection.objects.link(temp)
        bpy.ops.object.select_all(action='DESELECT')
        temp.select_set(True)
        bpy.context.view_layer.objects.active = temp
        fp = os.path.join(export_dir, obj.name + '.stl')
        try:
            bpy.ops.wm.stl_export(filepath=fp, ascii_format=False,
                                   apply_modifiers=True,
                                   export_selected_objects=True, global_scale=1.0)
        except AttributeError:
            bpy.ops.export_mesh.stl(filepath=fp, use_selection=True, global_scale=1.0)
        bpy.data.objects.remove(temp, do_unlink=True)
        bpy.data.meshes.remove(mesh_copy)
        print(f"  Exported: {fp}")

if EXPORT_STL:
    print(f"\nExporting STL files to: {EXPORT_DIR}")
    export_all_meshes(EXPORT_DIR)
