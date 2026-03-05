"""
JOINT LIBRARY — Printable Robot Joints for Blender 4.x
Skill: blender-robot | Target: Blender 4.x | Units: millimeters

This file is a REFERENCE LIBRARY, not a standalone script.
Copy individual functions into your robot script as needed,
or paste the whole file above your script and call the functions.

JOINT TYPES:
  1. Ball and Socket   — multi-axis (shoulder, hip, neck, wrist)
  2. Revolute (Pin)    — single-axis (elbow, knee, ankle)
  3. Snap-Fit Tab      — one-direction clip (panel covers, battery doors)
  4. Living Hinge      — flexible flexure (finger, small flap)

MATERIAL NOTES:
  Ball/Revolute — PLA, PETG, ABS all work
  Snap-Fit      — PETG preferred; PLA works for low-cycle clips; avoid ABS (brittle)
  Living Hinge  — PETG or TPU ONLY; PLA is too brittle at 0.6 mm
"""

# ══ GLOBAL JOINT PARAMETERS — adjust once here ════════════════════════════════
JOINT_CLEARANCE = 0.30   # mm radial, ball-and-socket
PIN_CLEARANCE   = 0.20   # mm diametric (bore_r = pin_r + PIN_CLEARANCE/2)
SNAP_CLEARANCE  = 0.15   # mm, tight snap-fit
SLIDE_CLEARANCE = 0.20   # mm per face, prismatic/dovetail slides
FDM_MIN_WALL    = 1.20   # mm, hard floor for FDM 0.4 mm nozzle
# ═════════════════════════════════════════════════════════════════════════════

import bpy
import bmesh
import math
from mathutils import Vector


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _add_cyl(name, radius, depth, verts=32, loc=(0,0,0), rot=(0,0,0)):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, vertices=verts,
        location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    return o

def _add_sphere(name, radius, segs=32, rings=16, loc=(0,0,0)):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, segments=segs, ring_count=rings, location=loc)
    o = bpy.context.active_object
    o.name = name
    return o

def _add_box(name, w, h, d, loc=(0,0,0), rot=(0,0,0)):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = (max(w,0.01)/2, max(d,0.01)/2, max(h,0.01)/2)
    bpy.ops.object.transform_apply(scale=True)
    return o

def _bool_diff(base, cutter, delete_cutter=True):
    mod = base.modifiers.new('BoolDiff', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'; mod.solver = 'EXACT'; mod.object = cutter
    bpy.context.view_layer.objects.active = base
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if delete_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return base

def _bool_union(a, b, result_name=None):
    mod = a.modifiers.new('BoolUnion', 'BOOLEAN')
    mod.operation = 'UNION'; mod.solver = 'EXACT'; mod.object = b
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(b, do_unlink=True)
    if result_name:
        a.name = result_name
    return a

def _cleanup(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data); bm.free(); obj.data.update()


# ─────────────────────────────────────────────────────────────────────────────
#  TYPE 1 — BALL AND SOCKET JOINT
# ─────────────────────────────────────────────────────────────────────────────

def build_ball_joint(center=(0,0,0), ball_r=8.0,
                     clearance=JOINT_CLEARANCE,
                     lip_fraction=0.15):
    """
    Create a matched ball-and-socket pair.

    Returns:
        (ball_obj, socket_obj)

    Parameters:
        center       — world position of the joint centre
        ball_r       — radius of the ball in mm
        clearance    — radial gap between ball and socket (mm)
        lip_fraction — fraction of ball_r used for the retention lip depth

    PRINT INSTRUCTIONS:
        ball   — print flat side down (equatorial flat face on bed). No supports.
        socket — print open side up. Supports inside the socket cavity.

    ASSEMBLY:
        Press ball into socket from the open side. The retention lip snaps
        over the ball equator. Use a small chamfer on the socket rim to
        ease insertion: run Edit Mode → Bevel on the inner rim edge loop.

    CLEARANCE GUIDE:
        0.20 mm — very tight, needs post-print reaming
        0.30 mm — normal fit, snaps in with light force
        0.40 mm — loose, free rotation, minimal resistance
    """
    cx, cy, cz = center

    # ── Ball: full sphere, cut flat at the equator ──────────────────────────
    ball = _add_sphere('ball', radius=ball_r, loc=center)
    # Cut away the lower hemisphere to get a flat printing base
    equator_cutter = _add_box('_eq_cut',
                              w=ball_r * 3, h=ball_r * 1.02, d=ball_r * 3,
                              loc=(cx, cy, cz - ball_r * 1.01 / 2))
    ball = _bool_diff(ball, equator_cutter)
    _cleanup(ball)

    # ── Socket: outer sphere - inner ball cavity - top half removed ─────────
    sock_r = ball_r + clearance

    outer = _add_sphere('_sock_outer', radius=sock_r + lip_fraction * ball_r,
                        loc=center)
    inner = _add_sphere('_sock_inner', radius=sock_r, loc=center)
    socket = _bool_diff(outer, inner, delete_cutter=True)   # hollow shell

    # Remove top hemisphere so the ball can be inserted
    top_cutter = _add_box('_top_cut',
                          w=sock_r * 3, h=sock_r * 1.02, d=sock_r * 3,
                          loc=(cx, cy, cz + sock_r * 1.01 / 2))
    socket = _bool_diff(socket, top_cutter)

    # Retention lip: thin torus-like ring at the equatorial opening
    # Approximated as a hollow cylinder ring around the equator
    lip_outer = _add_cyl('_lip_outer', radius=sock_r + lip_fraction * ball_r * 0.6,
                          depth=ball_r * 0.18, loc=(cx, cy, cz))
    lip_inner = _add_cyl('_lip_inner', radius=sock_r - clearance * 0.5,
                          depth=ball_r * 0.20, loc=(cx, cy, cz))
    lip = _bool_diff(lip_outer, lip_inner, delete_cutter=True)
    socket = _bool_union(socket, lip, result_name='socket')
    _cleanup(socket)

    print(f"ball_joint: ball_r={ball_r}mm, sock_r={sock_r}mm, clearance={clearance}mm")
    return ball, socket


# ─────────────────────────────────────────────────────────────────────────────
#  TYPE 2 — REVOLUTE (PIN) JOINT
# ─────────────────────────────────────────────────────────────────────────────

def build_pin_joint(center=(0,0,0), pin_r=5.0, total_span=20.0,
                    clearance=PIN_CLEARANCE, axis='Y'):
    """
    Create a pin-and-bore pair for a single-axis revolute joint.

    Returns:
        (pin_obj, bore_info_dict)

    bore_info_dict has keys:
        'bore_radius'  — mm, use this for holes in the receiving parts
        'bore_depth'   — mm, depth to drill per side (span / 2 + 1 mm)
        'axis'         — same axis string passed in

    Parameters:
        center     — world position of pin midpoint
        pin_r      — pin shaft radius in mm
        total_span — total pin length (should span both sides of joint)
        clearance  — diametric clearance (bore = pin_dia + clearance)
        axis       — rotation axis: 'X', 'Y', or 'Z'

    PRINT INSTRUCTIONS:
        pin — print along its length axis. The chamfered ends are
              self-supporting. No supports needed.

    ASSEMBLY:
        1. Print two joint arms with bore holes (bore_radius).
        2. Press or slide pin through. For a retained joint, add a
           C-clip groove: call add_cclip_groove() on the pin.

    BORE USAGE EXAMPLE:
        bore_r, bore_d = info['bore_radius'], info['bore_depth']
        bore_cyl = add_cyl('bore_cut', bore_r, bore_d, loc=joint_centre)
        arm_part = bool_diff(arm_part, bore_cyl)
    """
    rot_map = {'X': (0, math.pi/2, 0), 'Y': (math.pi/2, 0, 0), 'Z': (0, 0, 0)}
    rot = rot_map.get(axis, (0, 0, 0))

    bore_r = pin_r + clearance / 2

    # Pin shaft
    pin = _add_cyl('pin', radius=pin_r, depth=total_span, loc=center, rot=rot)

    # Chamfer both ends (self-supporting 45° taper)
    bevel = pin.modifiers.new('Bevel', 'BEVEL')
    bevel.width        = pin_r * 0.35
    bevel.segments     = 2
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit  = math.radians(80)
    bpy.context.view_layer.objects.active = pin
    bpy.ops.object.modifier_apply(modifier=bevel.name)

    bore_info = {
        'bore_radius': bore_r,
        'bore_depth':  total_span / 2 + 1.0,
        'axis':        axis,
    }
    print(f"pin_joint: pin_r={pin_r}mm, bore_r={bore_r:.3f}mm, span={total_span}mm")
    return pin, bore_info


def add_cclip_groove(pin_obj, groove_pos_z: float,
                     groove_width=1.5, groove_depth=0.8):
    """
    Cut a C-clip groove ring into pin_obj at groove_pos_z (local Z along pin axis).
    groove_width and groove_depth in mm.
    Groove is cut as a thin cylinder boolean.
    """
    # The pin's local Z maps to the axis it was created on.
    # This function assumes the pin was created along global Z. Rotate first if needed.
    groove_cyl = _add_cyl('_cclip_groove',
                           radius=pin_obj.dimensions.x / 2,   # match pin outer radius
                           depth=groove_width,
                           loc=(pin_obj.location.x,
                                pin_obj.location.y,
                                groove_pos_z))
    # Scale radially inward by groove_depth to cut a ring channel
    # Use a torus-like approach: big cyl - (big-groove) cyl
    inner_r = pin_obj.dimensions.x / 2 - groove_depth
    inner   = _add_cyl('_cclip_inner', radius=inner_r, depth=groove_width + 1,
                        loc=groove_cyl.location)
    channel = _bool_diff(groove_cyl, inner, delete_cutter=True)
    pin_obj = _bool_diff(pin_obj, channel, delete_cutter=True)
    _cleanup(pin_obj)
    return pin_obj


# ─────────────────────────────────────────────────────────────────────────────
#  TYPE 3 — SNAP-FIT TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_snap_fit(name='snap_tab', width=8.0, height=12.0,
                   thickness=2.0, clearance=SNAP_CLEARANCE,
                   lead_angle_deg=30.0):
    """
    Create a cantilevered snap-fit tab.

    Returns:
        (tab_obj, receptor_info_dict)

    receptor_info_dict has keys:
        'slot_width'  — slot width to cut in the receiving part
        'slot_depth'  — slot depth to cut
        'slot_height' — slot height to cut

    Parameters:
        width          — mm, tab width (perpendicular to snap direction)
        height         — mm, tab length (cantilever reach)
        thickness      — mm, tab thickness (must be >= 1.5 for PETG)
        clearance      — mm, fit clearance for the receptor slot
        lead_angle_deg — degrees, lead-in angle on tab tip (30° = self-releasing)

    MATERIAL WARNING:
        Tab thickness < 1.5 mm is fragile in PLA. Use PETG for repeated cycling.
        For PETG at 2.0 mm thick, deflection limit is approx. height * 0.08 mm.

    PRINT INSTRUCTIONS:
        Print tab flat (base on bed, cantilever extending up).
        The lead-in wedge at the tip is self-supporting.
    """
    if thickness < 1.5:
        print(f"WARNING: snap tab thickness {thickness}mm is below 1.5mm PETG minimum")

    # Main tab body
    tab = _add_box(name, w=width, h=height, d=thickness)

    # Lead-in wedge at tip — cut angled face to ease engagement
    wedge_h = height * 0.30
    wedge = _add_box(f'_{name}_wedge', w=width * 1.1, h=wedge_h, d=thickness * 0.7,
                     loc=(0, 0, height/2 - wedge_h/2 + height * 0.05))
    wedge.rotation_euler.x = math.radians(lead_angle_deg)
    bpy.ops.object.select_all(action='DESELECT')
    wedge.select_set(True)
    bpy.context.view_layer.objects.active = wedge
    bpy.ops.object.transform_apply(rotation=True)
    tab = _bool_diff(tab, wedge)
    _cleanup(tab)

    receptor = {
        'slot_width':  width  + clearance * 2,
        'slot_depth':  thickness + clearance * 2,
        'slot_height': height * 1.1,
    }
    print(f"snap_fit: {width}×{height}×{thickness}mm, slot {receptor['slot_width']:.2f}×{receptor['slot_depth']:.2f}mm")
    return tab, receptor


# ─────────────────────────────────────────────────────────────────────────────
#  TYPE 4 — LIVING HINGE (FLEXURE)
# ─────────────────────────────────────────────────────────────────────────────

def build_living_hinge(name='hinge', panel_width=20.0, panel_length=15.0,
                       flex_thickness=0.6, flex_width=None, panel_depth=2.0):
    """
    Create two panels joined by a thin flex strip — prints as ONE piece.

    Returns:
        hinge_obj (single merged mesh)

    Parameters:
        panel_width    — mm, both panels share this width (X axis)
        panel_length   — mm, each panel's length (Y axis, each side of hinge)
        flex_thickness — mm, MINIMUM 0.6 mm for FDM. Print at 0.15 mm layer height.
        flex_width     — mm, width of flex strip (defaults to panel_width * 0.8)
        panel_depth    — mm, panel thickness (Z axis). Must be >= FDM_MIN_WALL.

    MATERIAL WARNING:
        PLA will crack after 5–10 flex cycles at 0.6 mm.
        PETG: ~200 cycles. TPU: thousands of cycles.

    PRINT INSTRUCTIONS:
        Print FLAT — hinge horizontal, parallel to print bed.
        The flex strip must run parallel to layer lines (not across them).
        Use 0.15 mm layer height and 3 perimeters minimum.
        Do NOT use a brim over the flex strip (hard to remove).

    FLEX RANGE:
        0.6 mm thick × 20 mm wide: ~70° usable flex angle
        1.0 mm thick × 20 mm wide: ~45° usable flex angle
    """
    if flex_thickness < 0.6:
        print(f"WARNING: flex_thickness {flex_thickness}mm below FDM minimum 0.6mm")
    if panel_depth < FDM_MIN_WALL:
        print(f"WARNING: panel_depth {panel_depth}mm below FDM_MIN_WALL {FDM_MIN_WALL}mm")

    fw = flex_width or panel_width * 0.80
    gap = flex_thickness  # the flex strip sits between the two panels

    # Panel A (negative Y)
    pa = _add_box(f'{name}_panel_A', w=panel_width, h=panel_length, d=panel_depth,
                  loc=(0, -(panel_length/2 + gap/2), 0))

    # Flex strip (centre)
    fl = _add_box(f'_{name}_flex', w=fw, h=gap, d=panel_depth,
                  loc=(0, 0, 0))

    # Panel B (positive Y)
    pb = _add_box(f'{name}_panel_B', w=panel_width, h=panel_length, d=panel_depth,
                  loc=(0, panel_length/2 + gap/2, 0))

    result = _bool_union(pa, fl)
    result = _bool_union(result, pb, result_name=name)
    _cleanup(result)

    print(f"living_hinge: panels {panel_width}×{panel_length}mm, "
          f"flex {fw}×{flex_thickness}mm thick")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  JOINT USAGE GUIDE
# ─────────────────────────────────────────────────────────────────────────────

JOINT_USAGE_GUIDE = {
    # Humanoid
    'shoulder':    'ball_joint(ball_r=10)',
    'elbow':       'pin_joint(pin_r=4, total_span=16)',
    'wrist':       'ball_joint(ball_r=7)',
    'hip':         'ball_joint(ball_r=12)',
    'knee':        'pin_joint(pin_r=5, total_span=18)',
    'ankle':       'pin_joint(pin_r=4, total_span=14)',
    'neck':        'ball_joint(ball_r=8)',
    'finger_knuckle': 'pin_joint(pin_r=1.5, total_span=8)',
    # Robot arm
    'arm_joint_1': 'pin_joint(pin_r=6, total_span=24)',
    'arm_joint_2': 'pin_joint(pin_r=5, total_span=20)',
    'gripper_jaw': 'build_snap_fit for a locking position, or pin for free-slide',
    # Hexapod
    'coxa_joint':  'pin_joint(pin_r=3.5, total_span=14)',
    'femur_joint': 'pin_joint(pin_r=3, total_span=12)',
    'tibia_joint': 'pin_joint(pin_r=2.5, total_span=10)',
    # Panels / covers
    'battery_door': 'build_snap_fit(width=8, height=10, thickness=1.8)',
    'chest_panel':  'build_living_hinge OR build_snap_fit',
}


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK DEMO  (runs when executed as standalone script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__' or True:
    # Setup scene units
    scene = bpy.context.scene
    scene.unit_settings.system       = 'METRIC'
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit  = 'MILLIMETERS'
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Demo: one of each joint type, laid out in a row
    print("Building joint demo...")
    ball, socket = build_ball_joint(center=(-60, 0, 0), ball_r=10)
    ball.location.z   += 15   # lift ball above socket for display
    socket.location.z -= 5

    pin, bore_info = build_pin_joint(center=(0, 0, 0), pin_r=4, total_span=18)

    tab, receptor = build_snap_fit('demo_snap', width=8, height=12, thickness=2.0,
                                   loc_offset=(60, 0, 0))

    hinge = build_living_hinge('demo_hinge', panel_width=20, panel_length=15,
                               flex_thickness=0.8)
    hinge.location = (120, 0, 0)

    print("\nJoint demo built. Parts (left to right):")
    print("  -60mm : ball and socket")
    print("    0mm : revolute pin joint")
    print("   60mm : snap-fit tab")
    print("  120mm : living hinge")
    print("\nInspect the Outliner for part names.")
