"""
ROBOT SCRIPT: Parametric 6-DOF Robot Arm
Skill: blender-robot | Target: Blender 4.x | Units: millimeters

HOW TO RUN:
  1. Open Blender 4.x → Scripting workspace
  2. New text block → paste this entire script
  3. Press Alt+P (or ▶ Run Script)

WHAT YOU GET:
  A desk-mounted 6-DOF robot arm with:
    BASE       — circular mounting flange + motor housing
    LINK_1     — shoulder-to-elbow (largest link)
    LINK_2     — elbow-to-wrist
    LINK_3     — wrist extension
    WRIST      — wrist roll cylinder
    GRIPPER    — two-jaw parallel gripper

  Each part is in its own collection. Links are stacked vertically
  (arm pointing up) for easy proportions check. Rotate joints in
  Blender after reviewing.

PRINT NOTES (FDM):
  base        — print flat (bottom face down). No supports.
  link_1/2/3  — print vertically (long axis up). Supports for bore holes.
  wrist       — print upright.
  jaw_L/R     — print flat (rail face down).

ARTICULATION:
  Joint 1 — base rotation (yaw)       — revolute, vertical axis
  Joint 2 — shoulder pitch             — revolute, horizontal axis
  Joint 3 — elbow pitch                — revolute, horizontal axis
  Joint 4 — wrist pitch                — revolute, horizontal axis
  Joint 5 — wrist roll                 — revolute, Z axis of wrist
  Joint 6 — gripper (prismatic)        — linear, jaw open/close
"""

# ══ PARAMETERS — edit here, then re-run ══════════════════════════════════════
ARM_NAME        = "Arm_6DOF"
BASE_RADIUS     = 40.0     # mm, mounting flange
BASE_H          = 22.0     # mm
MOTOR_RADIUS    = 28.0     # mm, motor housing cylinder
MOTOR_H         = 30.0     # mm

LINK_WIDTHS     = [18.0, 15.0, 12.0]   # mm, XY cross-section width per link
LINK_DEPTHS     = [14.0, 12.0, 10.0]   # mm, XY cross-section depth per link
LINK_LENGTHS    = [120.0, 100.0, 60.0] # mm, length along Z per link

JOINT_R         = 12.0     # mm, revolute joint bore radius (pin hole)
JOINT_CLEARANCE = 0.2      # mm diametric, bore = pin_dia + JOINT_CLEARANCE
JOINT_WALL      = 2.0      # mm, wall thickness around bore holes

WRIST_R         = 15.0     # mm, wrist cylinder radius
WRIST_H         = 28.0     # mm

GRIPPER_BASE_W  = 30.0     # mm, gripper mounting block width
GRIPPER_BASE_H  = 18.0     # mm
GRIPPER_BASE_D  = 20.0     # mm
JAW_W           = 12.0     # mm, each jaw width
JAW_H           = 35.0     # mm, jaw length (reach)
JAW_D           = 8.0      # mm, jaw thickness
JAW_OFFSET      = 12.0     # mm, jaw X offset from centreline (half-open position)
JAW_CLEARANCE   = 0.25     # mm, prismatic slide clearance each face

FDM_MIN_WALL    = 1.2      # mm
EXPORT_STL      = False
EXPORT_DIR      = "C:/Users/YourName/Desktop/robot_stls/"
# ══════════════════════════════════════════════════════════════════════════════

import bpy, bmesh, math, os
from mathutils import Vector


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
    w = max(w, FDM_MIN_WALL); h = max(h, FDM_MIN_WALL); d = max(d, FDM_MIN_WALL)
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = (w/2, d/2, h/2)
    bpy.ops.object.transform_apply(scale=True)
    return o

def add_cyl(name, radius, depth, verts=32, loc=(0,0,0), rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=max(radius, FDM_MIN_WALL/2),
                                        depth=depth, vertices=verts,
                                        location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    return o

def bool_diff(base, cutter, delete_cutter=True):
    mod = base.modifiers.new('BoolDiff', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'; mod.solver = 'EXACT'; mod.object = cutter
    bpy.context.view_layer.objects.active = base
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if delete_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return base

def bool_union(a, b, result_name=None):
    mod = a.modifiers.new('BoolUnion', 'BOOLEAN')
    mod.operation = 'UNION'; mod.solver = 'EXACT'; mod.object = b
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(b, do_unlink=True)
    if result_name: a.name = result_name
    return a

def cleanup_mesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data); bm.free(); obj.data.update()

def add_bore(obj, bore_r, bore_depth, axis='X', centre=(0,0,0)):
    """Cut a cylindrical bore hole through obj along the given axis."""
    rot = {'X': (0, math.pi/2, 0), 'Y': (math.pi/2, 0, 0), 'Z': (0,0,0)}[axis]
    cutter = add_cyl('_bore', bore_r + JOINT_CLEARANCE/2, bore_depth + 1.0,
                     loc=centre, rot=rot)
    return bool_diff(obj, cutter)


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

col_root    = get_or_create_collection(ARM_NAME)
col_base    = get_or_create_collection('BASE',    col_root)
col_links   = [get_or_create_collection(f'LINK_{i+1}', col_root) for i in range(3)]
col_wrist   = get_or_create_collection('WRIST',   col_root)
col_gripper = get_or_create_collection('GRIPPER', col_root)


# ─────────────────────────────────────────────────────────────────────────────
#  BASE
# ─────────────────────────────────────────────────────────────────────────────

BASE_TOP_Z = BASE_H + MOTOR_H   # Z of the base/link-1 interface

# Flange disc
flange = add_cyl('base_flange', BASE_RADIUS, BASE_H, loc=(0, 0, BASE_H/2))
# Motor housing cylinder (smaller, sits on top of flange)
motor  = add_cyl('_motor_housing', MOTOR_RADIUS, MOTOR_H,
                 loc=(0, 0, BASE_H + MOTOR_H/2))
base   = bool_union(flange, motor, result_name='base')
# Centre bore for cable pass-through
cable_bore = add_cyl('_cable_bore', MOTOR_RADIUS * 0.35, BASE_H + MOTOR_H + 2,
                     loc=(0, 0, BASE_TOP_Z/2))
base = bool_diff(base, cable_bore)
cleanup_mesh(base)
link_to(base, col_base)


# ─────────────────────────────────────────────────────────────────────────────
#  LINKS 1–3
# ─────────────────────────────────────────────────────────────────────────────

# Links stack on top of each other (arm pointing straight up for block-out)
z_cursor = BASE_TOP_Z
links = []

for i, (lw, ld, ll) in enumerate(zip(LINK_WIDTHS, LINK_DEPTHS, LINK_LENGTHS)):
    link_z = z_cursor + ll / 2
    name   = f'link_{i+1}'
    link   = add_box(name, lw, ll, ld, loc=(0, 0, link_z))

    # Bore holes at top and bottom for revolute joints
    bore_r = JOINT_R + JOINT_CLEARANCE / 2
    # Bottom bore (connects to previous joint / base)
    add_bore(link, bore_r, ld + 2, axis='Y',
             centre=(0, 0, z_cursor + JOINT_WALL + bore_r))
    # Top bore (connects to next joint)
    add_bore(link, bore_r, ld + 2, axis='Y',
             centre=(0, 0, z_cursor + ll - JOINT_WALL - bore_r))

    cleanup_mesh(link)
    link_to(link, col_links[i])
    links.append(link)
    z_cursor += ll


# ─────────────────────────────────────────────────────────────────────────────
#  WRIST
# ─────────────────────────────────────────────────────────────────────────────

wrist_z = z_cursor + WRIST_H / 2
wrist   = add_cyl('wrist', WRIST_R, WRIST_H, loc=(0, 0, wrist_z))
# Flatten into a short disc with a square cross-section block for tool mount
tool_mount = add_box('_tool_block', WRIST_R * 1.4, WRIST_R, WRIST_R * 1.4,
                     loc=(0, 0, wrist_z + WRIST_H / 2))
wrist = bool_union(wrist, tool_mount, result_name='wrist')
cleanup_mesh(wrist)
link_to(wrist, col_wrist)
z_cursor += WRIST_H


# ─────────────────────────────────────────────────────────────────────────────
#  GRIPPER
# ─────────────────────────────────────────────────────────────────────────────

grip_z = z_cursor + GRIPPER_BASE_H / 2

# Gripper body
grip_body = add_box('gripper_body', GRIPPER_BASE_W, GRIPPER_BASE_H,
                    GRIPPER_BASE_D, loc=(0, 0, grip_z))
link_to(grip_body, col_gripper)

# Jaws (symmetric, ±X)
jaw_z = grip_z + GRIPPER_BASE_H/2 + JAW_H/2
for side, sign in (('jaw_R', 1), ('jaw_L', -1)):
    jaw = add_box(side, JAW_W, JAW_H, JAW_D,
                  loc=(sign * JAW_OFFSET, 0, jaw_z))
    # Finger tip chamfer — cut a wedge from the jaw tip for a pinch geometry
    tip_cutter = add_box(f'_{side}_tip', JAW_W * 1.2, JAW_W * 0.5, JAW_D * 1.2,
                         loc=(sign * JAW_OFFSET, 0, jaw_z + JAW_H/2))
    tip_cutter.rotation_euler.x = math.radians(45)
    bpy.ops.object.select_all(action='DESELECT')
    tip_cutter.select_set(True)
    bpy.context.view_layer.objects.active = tip_cutter
    bpy.ops.object.transform_apply(rotation=True)
    jaw = bool_diff(jaw, tip_cutter)
    cleanup_mesh(jaw)
    link_to(jaw, col_gripper)


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

total_reach = BASE_TOP_Z + sum(LINK_LENGTHS) + WRIST_H + GRIPPER_BASE_H + JAW_H
print("\n" + "═" * 60)
print(f"  {ARM_NAME}  built successfully")
print(f"  Total height (arm extended): {total_reach:.0f} mm")
print(f"  Joint bore diameter: {(JOINT_R + JOINT_CLEARANCE/2)*2:.2f} mm")
print(f"  Jaw half-open offset: ±{JAW_OFFSET:.0f} mm")
print("═" * 60)
print("\nNEXT STEPS:")
print("  • Rotate links to a natural pose in the 3D viewport")
print("  • For functional joints: see joint_library.py for pin+bore patterns")
print("  • To export: set EXPORT_STL = True and update EXPORT_DIR")
print("\nPRINT CHECK:")
print("  Edit Mode → Mesh → Cleanup → Fill Holes")
print("  3D Print Toolbox → Check All")


# ─────────────────────────────────────────────────────────────────────────────
#  STL EXPORT
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
