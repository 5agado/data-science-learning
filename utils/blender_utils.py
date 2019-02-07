import bpy
import bmesh
import numpy as np

is_blender_28 = bpy.app.version[1] >= 80
assert is_blender_28

"""
# Blender import-system clutter
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


# TODO validate
def delete_all(obj_type: str='MESH'):
    """Delete all objects of the given type from the current scene"""

    for obj in bpy.data.objects:
        obj.hide_select = False
        obj.select_set(obj.type == obj_type)

    bpy.ops.object.delete(use_global=True)


def create_line(p0: tuple, p1: tuple, name: str = 'line'):
    mesh = bpy.data.meshes.new(name)

    mesh.from_pydata([p0, p1], [(0, 1)], [])
    mesh.update()

    return _add_mesh_to_scene(mesh, name)


def create_circle(radius: float, segments: int=32, name: str = 'circle'):
    bm = bmesh.new()

    bmesh.ops.create_circle(bm, cap_ends=False, radius=radius, segments=segments)

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
    scene.collection.objects.link(obj)
    #scene.update()
    return obj


def add_text(text: str, location: tuple = (0, 0, 0)):
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.view_layer.objects.active
    text_obj.data.body = text
    return text_obj


"""
# Get object
bpy.context.view_layer.objects.active
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
###           Grease Pencil
###################################
from bpy.types import GPencilFrame
from math import sin, cos, pi
import itertools


def get_grease_pencil(gpencil_obj_name='GPencil') -> bpy.types.GreasePencil:
    """
    Return the grease-pencil object with the given name. Initialize one if not already present.
    :param gpencil_obj_name: name/key of the grease pencil object in the scene
    """

    # If not present already, create grease pencil object
    if gpencil_obj_name not in bpy.context.scene.objects:
        bpy.ops.object.gpencil_add(view_align=False, location=(0, 0, 0), type='EMPTY')

    # Get grease pencil object
    gpencil = bpy.context.scene.objects[gpencil_obj_name]

    return gpencil


def get_grease_pencil_layer(gpencil: bpy.types.GreasePencil, gpencil_layer_name='GP_Layer',
                            clear_layer=False) -> bpy.types.GPencilLayer:
    """
    Return the grease-pencil layer with the given name. Create one if not already present.
    :param gpencil: grease-pencil object for the layer data
    :param gpencil_layer_name: name/key of the grease pencil layer
    :param clear_layer: whether to clear all previous layer data
    """

    # Get grease pencil layer or create one if none exists
    if gpencil.data.layers and gpencil_layer_name in gpencil.data.layers:
        gpencil_layer = gpencil.data.layers[gpencil_layer_name]
    else:
        gpencil_layer = gpencil.data.layers.new(gpencil_layer_name, set_active=True)

    if clear_layer:
        gpencil_layer.clear()  # clear all previous layer data

    # bpy.ops.gpencil.paintmode_toggle()  # need to trigger otherwise there is no frame

    return gpencil_layer


# Util for default behavior merging previous two methods
def init_grease_pencil(gpencil_obj_name='GPencil', gpencil_layer_name='GP_Layer',
                       clear_layer=True) -> bpy.types.GPencilLayer:
    gpencil = get_grease_pencil(gpencil_obj_name)
    gpencil_layer = get_grease_pencil_layer(gpencil, gpencil_layer_name, clear_layer=clear_layer)
    return gpencil_layer


"""
# Frames and Strokes
# https://docs.blender.org/api/blender2.8/bpy.types.GPencilStroke.html#bpy.types.GPencilStroke
gp_frame = gp_layer.frames.new(frame_number) 
# notice that index in the frames list does not necessarily match frame number in the timeline
go_frame = gp_layer.frames[1]
gp_frame.frame_number
gp_frame = gp_layer.frames.copy(gp_layer.frames[-1])
gp_stroke = gp_frame.strokes.new()
gp_stroke.line_width = 10
gp_stroke.material_index
gp_stroke.display_mode = '3DSPACE'  # allows for editing
gp_stroke.points.add(count=4)
gp_stroke.points[0].co = (0, 0, 0)
gp_stroke.points[0].pressure
gp_stroke.points[0].strength

bpy.context.object.active_material_index = 1
"""


def draw_line(gp_frame: GPencilFrame, p0: tuple, p1: tuple, material_index=0):
    # Init new stroke
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.display_mode = '3DSPACE'  # allows for editing
    gp_stroke.material_index = material_index

    # Define stroke geometry
    gp_stroke.points.add(count=2)
    gp_stroke.points[0].co = p0
    gp_stroke.points[1].co = p1
    return gp_stroke


def draw_square(gp_frame: GPencilFrame, center: tuple, size: int, material_index=0):
    # Init new stroke
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.display_mode = '3DSPACE'  # allows for editing
    gp_stroke.draw_cyclic = True        # closes the stroke
    gp_stroke.material_index = material_index

    # Define stroke geometry
    radius = size / 2
    gp_stroke.points.add(count=4)
    gp_stroke.points[0].co = (center[0] - radius, center[1] - radius, center[2])
    gp_stroke.points[1].co = (center[0] - radius, center[1] + radius, center[2])
    gp_stroke.points[2].co = (center[0] + radius, center[1] + radius, center[2])
    gp_stroke.points[3].co = (center[0] + radius, center[1] - radius, center[2])
    return gp_stroke


def draw_circle(gp_frame: GPencilFrame, center: tuple, radius: float, segments: int, material_index=0):
    # Init new stroke
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.display_mode = '3DSPACE'  # allows for editing
    gp_stroke.draw_cyclic = True        # closes the stroke
    gp_stroke.material_index = material_index

    # Define stroke geometry
    angle = 2*pi/segments  # angle in radians
    gp_stroke.points.add(count=segments)
    for i in range(segments):
        x = center[0] + radius*cos(angle*i)
        y = center[1] + radius*sin(angle*i)
        z = center[2]
        gp_stroke.points[i].co = (x, y, z)

    return gp_stroke


def draw_cube(gp_frame: GPencilFrame, center: tuple, size: float, material_index=0):
    # Init new stroke
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.display_mode = '3DSPACE'  # allows for editing
    gp_stroke.draw_cyclic = True  # closes the stroke
    gp_stroke.material_index = material_index

    # Define stroke geometry
    radius = size/2
    offsets = list(itertools.product([1, -1], repeat=3))  # vertices offset-product from the center
    points = [(center[0] + radius * offset[0],
               center[1] + radius * offset[1],
               center[2] + radius * offset[2]) for offset in offsets]
    stroke_idx = [0, 4, 6, 2, 0, 1, 5, 7, 3, 1, 5, 4, 6, 7, 3, 2]

    gp_stroke.points.add(count=len(stroke_idx))
    for i, idx in enumerate(stroke_idx):
        gp_stroke.points[i].co = points[idx]

    return gp_stroke


# Draw sphere by rotating a circle
def draw_sphere(gp_frame: GPencilFrame, center: tuple, radius: int, circles: int, material_index=0):
    angle = pi / circles
    for i in range(circles):
        circle = draw_circle(gp_frame, (0, 0, 0), radius, 4, material_index=material_index)
        rotate_stroke(circle, angle*i, 'x')
        translate_stroke(circle, center)


def rotate_stroke(stroke, angle, axis='z'):
    # Define rotation matrix based on axis
    if axis.lower() == 'x':
        transform_matrix = np.array([[1, 0, 0],
                                     [0, cos(angle), -sin(angle)],
                                     [0, sin(angle), cos(angle)]])
    elif axis.lower() == 'y':
        transform_matrix = np.array([[cos(angle), 0, -sin(angle)],
                                     [0, 1, 0],
                                     [sin(angle), 0, cos(angle)]])
    # default on z
    else:
        transform_matrix = np.array([[cos(angle), -sin(angle), 0],
                                     [sin(angle), cos(angle), 0],
                                     [0, 0, 1]])

    # Apply rotation matrix to each point
    for i, p in enumerate(stroke.points):
        p.co = transform_matrix @ np.array(p.co).reshape(3, 1)


# Simplistic method. Could rely instead on transformation matrices.
def translate_stroke(stroke, vector):
    for i, p in enumerate(stroke.points):
        p.co = np.array(p.co) + vector