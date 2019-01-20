# Blender import system clutter
import bpy
from bpy.types import GPencilFrame
import bmesh
from mathutils import Vector
import math
from math import sin, cos
import numpy as np
import itertools

import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import importlib
import utils.blender_utils
importlib.reload(utils.blender_utils)
from utils.blender_utils import init_grease_pencil


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


def draw_square(gp_frame: GPencilFrame, size: int, center: tuple, material_index=0):
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
    angle = 2*math.pi/segments  # angle in radians
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
    offsets = list(itertools.product([1, -1], repeat=3))  # vertices offset-    product from the center
    points = [(center[0] + radius * offset[0],
               center[1] + radius * offset[1],
               center[2] + radius * offset[2]) for offset in offsets]
    stroke_idx = [0, 4, 6, 2, 0, 1, 5, 7, 3, 1, 5, 4, 6, 7, 3, 2]

    gp_stroke.points.add(count=len(stroke_idx))
    for i, idx in enumerate(stroke_idx):
        gp_stroke.points[i].co = points[idx]

    return gp_stroke


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
        print(type(p.co))
        p.co = transform_matrix @ np.array(p.co).reshape(3, 1)


# Simplistic method. Could rely instead on transformation matrices.
def translate_stroke(stroke, vector):
    for i, p in enumerate(stroke.points):
        p.co = np.array(p.co) + vector


# Draw sphere by rotating a circle
def draw_sphere(gp_frame: GPencilFrame, radius: int, circles: int, material_index=0):
    angle = math.pi / circles
    for i in range(circles):
        circle = draw_circle(gp_frame, (0, 0, 0), radius, 32, material_index=material_index)
        rotate_stroke(circle, angle*i, 'x')


gp_layer = init_grease_pencil()
#if gp_layer.frames:
#    gp_layer.frames.remove(gp_layer.frames[1])  # first frame is there by default. Remove as it makes loops cleaner
gp_layer.clear()  # clear all previous layer data

# notice that index in the frames list does not match frame number in the timeline
gp_frame = gp_layer.frames.new(0)

draw_line(gp_frame, (0, 0, 0), (1, 1, 0))
draw_square(gp_frame, 10, (0, 0, 0))
draw_circle(gp_frame, (0, 0, 0), 2, 32)
draw_cube(gp_frame, (3, 3, 3), 2)
draw_sphere(gp_frame, 3, 32)

NUM_FRAMES = 30
FRAMES_SPACING = 1
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = NUM_FRAMES*FRAMES_SPACING
