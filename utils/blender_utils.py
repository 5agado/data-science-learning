import bpy
import bmesh

is_blender_28 = bpy.app.version[1] >= 80


def delete_all(obj_type: str='MESH'):
    """Delete all objects of the given type from the current scene"""

    for obj in bpy.data.objects:
        if is_blender_28:
            pass
        else:
            obj.hide = False
            obj.select = obj.type == obj_type

    bpy.ops.object.delete(use_global=True)


def create_line(p0: tuple, p1: tuple, name: str = 'line'):
    mesh = bpy.data.meshes.new(name)

    mesh.from_pydata([p0, p1], [(0, 1)], [])
    mesh.update()

    return _add_mesh_to_scene(mesh, name)


def create_circle(radius: float, segments: int=32, name: str = 'circle'):
    bm = bmesh.new()

    if is_blender_28:
        bmesh.ops.create_circle(bm, cap_ends=False, radius=radius, segments=segments)
    else:
        bmesh.ops.create_circle(bm, cap_ends=False, diameter=radius, segments=segments)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()  # TODO necessary??

    return _add_mesh_to_scene(mesh, name)


def create_grid(x_segments: int, y_segments: int, size: int, name: str = 'grid'):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=x_segments, y_segments=y_segments, size=size)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)

    return _add_mesh_to_scene(mesh, name)


def _add_mesh_to_scene(mesh, obj_name: str):
    scene = bpy.context.scene
    obj = bpy.data.objects.new(obj_name, mesh)
    if is_blender_28:
        scene.collection.objects.link(obj)
    else:
        scene.objects.link(obj)
    #scene.update()
    return obj


def add_text(text: str, location: tuple = (0, 0, 0)):
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.scene.objects.active
    text_obj.data.body = text
    return text_obj


"""
# Get object
bpy.context.scene.objects.active
bpy.context.scene.objects['object_name']
bpy.data.objects['Camera'] ??difference with context

# Frame handlers
bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(lambda scene : scene.frame_current)

# Frames setting
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = NUM_FRAMES
bpy.context.scene.frame_set(frame)

# Keyframing
target.location = (0, 0, 0)
target.keyframe_insert("location")
target.rotation_euler = (0, 0, 0)
target.keyframe_insert("rotation_euler")

target.matrix_world.to_translation()  # better to use cause modifiers can not affect the previous

# Create vertex group
vg = target.vertex_groups.new(name="vg_name")
vg.add([1], 1.0, 'ADD')

# Particles system
target.modifiers.new("name", type='PARTICLE_SYSTEM')
ps = target.particle_systems[0]
ps.settings.emit_from = 'VERT'
ps.vertex_group_density = "emitter" 
"""


###################################
###           Greasy Pencil
###################################
def init_greasy_pencil():

    # Create grease pencil object if none exists
    if 'GPencil' not in bpy.context.scene.objects:
        bpy.ops.object.gpencil_add(view_align=False, location=(0., 0., 0.), type='EMPTY')

    g_pencil = bpy.context.scene.objects['GPencil']

    # Reference grease pencil layer or create one if none exists
    if g_pencil.data.layers:
        gp_layer = g_pencil.data.layers[0]
    else:
        gp_layer = g_pencil.data.layers.new('gpl', set_active=True)

    bpy.ops.gpencil.paintmode_toggle()  # need to trigger otherwise there is no frame
    gp_layer.clear()  # clear all previous layer data

    return gp_layer


"""
# Frames and Strokes
gp_frame = gp_layer.frames.new(i) # notice that index in the frames list does not match frame number in the timeline
gp_stroke = gp_frame.strokes.new()
gp_stroke.points.add(count=4)
gp_stroke.points[0].co = (0, 0, 0)
"""


"""
# Blender import system clutter
import bpy
import bmesh
from mathutils import Vector
import numpy as np

import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "graphics/agents"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))

import importlib
import <cls_example>
import utils.blender_utils
importlib.reload(<cls_example>)
importlib.reload(utils.blender_utils)
from <cls_example> import <cls_example>
from utils.blender_utils import delete_all
"""