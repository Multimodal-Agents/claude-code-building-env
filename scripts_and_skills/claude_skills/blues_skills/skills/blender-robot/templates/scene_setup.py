"""
ROBOT SCRIPT: Scene Setup & Utility Library
Skill: blender-robot | Target: Blender 4.x | Units: millimeters

HOW TO RUN:
  Paste this at the top of any robot script (or run standalone to clear the scene).
  Safe to run multiple times — it clears and reinitialises each time.

PARAMETERS — edit here, then re-run:
"""

# ══ PARAMETERS ════════════════════════════════════════════════════════════════
FDM_MIN_WALL    = 1.2    # mm, minimum wall for 0.4 mm FDM nozzle
RESIN_MIN_WALL  = 0.8    # mm, minimum wall for resin SLA/MSLA
JOINT_CLEARANCE = 0.3    # mm radial, ball-and-socket
PIN_CLEARANCE   = 0.2    # mm diametric (bore = pin_dia + PIN_CLEARANCE)
SNAP_CLEARANCE  = 0.15   # mm, tight snap-fit
EXPORT_STL      = False  # set True to export all mesh objects to STL
EXPORT_DIR      = "C:/Users/YourName/Desktop/robot_stls/"
# ══════════════════════════════════════════════════════════════════════════════

import bpy
import bmesh
import math
import os
from mathutils import Vector, Matrix, Euler


# ─────────────────────────────────────────────────────────────────────────────
#  SCENE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def setup_scene_units():
    """Set scene to millimeters. Call before creating any geometry."""
    scene = bpy.context.scene
    scene.unit_settings.system       = 'METRIC'
    scene.unit_settings.scale_length = 0.001   # 1 BU = 1 mm
    scene.unit_settings.length_unit  = 'MILLIMETERS'
    print("Units: 1 Blender Unit = 1 mm")


def clear_scene():
    """Delete all mesh objects and purge orphan data blocks."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.curves:
        if block.users == 0:
            bpy.data.curves.remove(block)
    print("Scene cleared.")


# ─────────────────────────────────────────────────────────────────────────────
#  COLLECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_collection(name: str,
                              parent: bpy.types.Collection = None
                              ) -> bpy.types.Collection:
    """Get existing collection by name or create it under parent."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def link_to(obj: bpy.types.Object, col: bpy.types.Collection):
    """Move object to collection, removing it from all others."""
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMITIVE HELPERS  (all dims in mm, assuming scale_length=0.001)
# ─────────────────────────────────────────────────────────────────────────────

def add_box(name: str, w: float, h: float, d: float,
            loc=(0, 0, 0), rot=(0, 0, 0)) -> bpy.types.Object:
    """
    Create a box.  w=width(X)  h=height(Z)  d=depth(Y)
    All dimensions in mm.
    """
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = (w / 2, d / 2, h / 2)   # default cube is 2x2x2 BU
    bpy.ops.object.transform_apply(scale=True)
    return o


def add_cyl(name: str, radius: float, depth: float,
            verts: int = 32, loc=(0, 0, 0), rot=(0, 0, 0)) -> bpy.types.Object:
    """Create a cylinder. All dims in mm."""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, vertices=verts,
        location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    return o


def add_sphere(name: str, radius: float,
               segs: int = 32, rings: int = 16,
               loc=(0, 0, 0)) -> bpy.types.Object:
    """Create a UV sphere. Radius in mm."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, segments=segs, ring_count=rings, location=loc)
    o = bpy.context.active_object
    o.name = name
    return o


def add_cone(name: str, r_base: float, r_top: float, depth: float,
             verts: int = 32, loc=(0, 0, 0)) -> bpy.types.Object:
    """Create a frustum (truncated cone). r_top=0 gives a full cone."""
    bpy.ops.mesh.primitive_cone_add(
        radius1=r_base, radius2=r_top, depth=depth,
        vertices=verts, location=loc)
    o = bpy.context.active_object
    o.name = name
    return o


# ─────────────────────────────────────────────────────────────────────────────
#  BOOLEAN OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def bool_diff(base: bpy.types.Object, cutter: bpy.types.Object,
              delete_cutter: bool = True) -> bpy.types.Object:
    """Subtract cutter from base. Returns base."""
    mod = base.modifiers.new('BoolDiff', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver    = 'EXACT'
    mod.object    = cutter
    bpy.context.view_layer.objects.active = base
    bpy.ops.object.modifier_apply(modifier=mod.name)
    if delete_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return base


def bool_union(a: bpy.types.Object, b: bpy.types.Object,
               result_name: str = None) -> bpy.types.Object:
    """Union b into a. Returns a (b is removed)."""
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


# ─────────────────────────────────────────────────────────────────────────────
#  MODIFIER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_bevel(obj: bpy.types.Object,
              width: float = 0.5,
              segments: int = 3,
              angle_deg: float = 60.0) -> bpy.types.Modifier:
    """Bevel all edges sharper than angle_deg. width in mm."""
    mod = obj.modifiers.new('Bevel', 'BEVEL')
    mod.width        = width
    mod.segments     = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit  = math.radians(angle_deg)
    return mod


def add_solidify(obj: bpy.types.Object,
                 thickness: float = 2.0,
                 offset: float = -1.0) -> bpy.types.Modifier:
    """Give a surface shell mesh a wall thickness. thickness in mm."""
    mod = obj.modifiers.new('Solidify', 'SOLIDIFY')
    mod.thickness      = thickness
    mod.offset         = offset     # -1 = grow inward, 0 = centred, 1 = outward
    mod.use_even_offset = True
    return mod


def add_array(obj: bpy.types.Object,
              count: int = 4,
              offset: tuple = (0.0, 0.0, 10.0)) -> bpy.types.Modifier:
    """Repeat obj `count` times with `offset` (mm) between copies."""
    mod = obj.modifiers.new('Array', 'ARRAY')
    mod.count = count
    mod.use_relative_offset  = False
    mod.use_constant_offset  = True
    mod.constant_offset_displace = offset
    return mod


def apply_all_modifiers(obj: bpy.types.Object):
    """Apply every modifier on obj in order."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)


# ─────────────────────────────────────────────────────────────────────────────
#  MESH CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_mesh(obj: bpy.types.Object, merge_dist: float = 0.001):
    """
    Remove doubles and fix face normals.
    Call after every boolean operation to maintain manifold geometry.
    merge_dist in mm.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


# ─────────────────────────────────────────────────────────────────────────────
#  MIRROR HELPER
# ─────────────────────────────────────────────────────────────────────────────

def mirror_object(src: bpy.types.Object, name: str,
                  axis: str = 'X') -> bpy.types.Object:
    """
    Duplicate src and mirror it across the given axis.
    Normals are recalculated so they point outward after the flip.
    """
    bpy.ops.object.select_all(action='DESELECT')
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.name = name
    if axis == 'X':
        dup.scale.x *= -1
    elif axis == 'Y':
        dup.scale.y *= -1
    elif axis == 'Z':
        dup.scale.z *= -1
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    return dup


# ─────────────────────────────────────────────────────────────────────────────
#  STL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_all_meshes(export_dir: str):
    """
    Export every visible mesh object to its own STL file.
    Objects whose names start with '_' are treated as helpers and skipped.
    Modifiers are evaluated (applied to a copy) before export.
    """
    os.makedirs(export_dir, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    exported  = []

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
            bpy.ops.wm.stl_export(              # Blender 4.0+
                filepath=fp,
                ascii_format=False,
                apply_modifiers=True,
                export_selected_objects=True,
                global_scale=1.0,
            )
        except AttributeError:
            bpy.ops.export_mesh.stl(            # Blender 3.x fallback
                filepath=fp, use_selection=True, global_scale=1.0)

        bpy.data.objects.remove(temp, do_unlink=True)
        bpy.data.meshes.remove(mesh_copy)
        exported.append(fp)
        print(f"  Exported: {fp}")

    print(f"\n{len(exported)} STL file(s) written to: {export_dir}")
    return exported


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRYPOINT (runs when executed as a standalone script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__' or True:   # bpy scripts always run at module level
    setup_scene_units()
    clear_scene()
    print("scene_setup.py loaded. Utilities available.")
    print("Set EXPORT_STL = True and EXPORT_DIR before exporting.")

    if EXPORT_STL:
        export_all_meshes(EXPORT_DIR)
