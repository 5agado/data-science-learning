# Blender import system clutter
import bpy
from bpy.types import GPencilFrame
import bmesh
from mathutils import Vector
import math
from math import sin, cos
import numpy as np

import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import importlib
import ds_utils.blender_utils
importlib.reload(ds_utils.blender_utils)

from ds_utils.blender_utils import *


###################################
###           Static
###################################

gp_layer = init_grease_pencil(clear_layer=True, gpencil_layer_name='static')

gp_frame = gp_layer.frames.new(0)
# draw_line(gp_frame, (0, 0, 0), (1, 1, 0))
# draw_square(gp_frame, (0, 0, 0), 10)
# draw_circle(gp_frame, (0, 0, 0), 2, 32)
# draw_cube(gp_frame, (3, 3, 3), 1)
# draw_sphere(gp_frame, (1, 1, 1), 3, 32)

NUM_FRAMES = 30
FRAMES_SPACING = 1
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = NUM_FRAMES*FRAMES_SPACING


def draw_wave_clock(gp_frame: GPencilFrame, nb_circles: int, center: tuple):
    for i in range(nb_circles):
        radius = np.random.randint(0, 5) + np.random.rand()
        for axis in ['x', 'y', 'z']:
            angle = np.random.randint(0, 3) + np.random.rand()
            steps = np.random.randint(0, 10)
            for j in range(steps):
                circle = draw_circle(gp_frame, (0, 0, 0), radius, 64, material_index=j)
                rotate_stroke(circle, (angle/steps)*j, axis=axis)
                translate_stroke(circle, center)


#draw_wave_clock(gp_frame, 10, (0, 0, 0))


def squares_grid(gp_frame: GPencilFrame, nb_rows: int, nb_cols: int,
                 rand_size=False, rand_rotation=False, material_index=0):
    for x in range(nb_cols):
        for y in range(nb_rows):
            center = (x, y, 0)
            if rand_size:
                radius = (x % (nb_cols/2) * y % (nb_rows/2))/((nb_cols/2)*(nb_rows/2)) + np.random.rand()/2
            else:
                radius = 1
            gp_stroke = draw_square(gp_frame, center, radius, material_index=material_index)
            draw_cube(gp_frame, center, radius)
            if rand_rotation:
                rotate_stroke(gp_stroke, np.random.rand())


#squares_grid(gp_frame, 10, 15, rand_size=True, rand_rotation=False, material_index=1)


def polygon_stairs(gp_frame, center: tuple, polygon_sides: int, side_len: float, nb_steps: int,
                   rotation_angle=0.5, step_size=0.5):
    for step in range(nb_steps):
        # draw polygon
        stroke = draw_circle(gp_frame, (0, 0, step*step_size), side_len-0.1*step, polygon_sides, step)
        # rotate polygon
        rotate_stroke(stroke, rotation_angle * step)
        translate_stroke(stroke, np.array(center))


#for i in range(10):
#    for j in range(10):
#        polygon_stairs(gp_frame, (i*7, j*7, 0), i+1, 3, 3*(j+1), rotation_angle=0.2, step_size=1)


###################################
###         Animations
###################################

gp_layer = init_grease_pencil(clear_layer=True, gpencil_layer_name='anim')

NUM_FRAMES = 100
FRAMES_SPACING = 1
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = NUM_FRAMES*FRAMES_SPACING


def draw_multiple_circles_animated(gp_layer):
    for frame in range(20):
        gp_frame = gp_layer.frames.new(frame)
        for i in range(15):
            radius = np.random.randint(1, 7) + np.random.rand()
            draw_circle(gp_frame, (0, 0, 0), radius, 80)


#draw_multiple_circles_animated(gp_layer)


def kinetic_rotation_polygon(gp_layer, center: tuple, nb_polygons: int, nb_sides: int,
                    min_radius: float, max_radius: float,
                    nb_frames: int):
    radiuses = np.linspace(min_radius, max_radius, nb_polygons)
    #radiuses = np.random.rand(nb_polygons)*(max_radius - min_radius) + min_radius  # randomized radiuses
    main_angle = (2*pi)/nb_frames

    # Animate polygons across frames
    for frame in range(nb_frames):
        gp_frame = gp_layer.frames.new(frame)
        for i in range(nb_polygons):
            polygon = draw_circle(gp_frame, (0, 0, 0), radiuses[i], nb_sides, i)
            #cur_angle = ((len(radiuses) - i) * (2 * pi)) / nb_frames
            cur_angle = ((len(radiuses) - i) // 2 * (2 * pi)) / nb_frames
            for axis in ['x']:
                rotate_stroke(polygon, cur_angle*frame, axis=axis)
            translate_stroke(polygon, center)



#kinetic_rotation_polygon(gp_layer, (0, 0, 0), nb_polygons=20, nb_sides=4, min_radius=3, max_radius=10,
#                         nb_frames=NUM_FRAMES)


def animate_square_sliding(gp_layer):
    main_size = 4
    positions = np.linspace(-main_size / 2, main_size / 2, num=NUM_FRAMES)
    for frame in range(1, NUM_FRAMES):
        gp_frame = gp_layer.frames.new(frame*FRAMES_SPACING)
        _ = draw_square(gp_frame, (0, 0, 0), main_size)

        draw_square(gp_frame, (main_size/2+0.5, positions[frame], 0), 1)


#animate_square_sliding(gp_layer)


def _get_midpoint(p0: tuple, p1:tuple):
    return (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2


def polygon_recursive(gp_frame, polygon, step=0, max_steps=3):
    # Init new stroke
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.display_mode = '3DSPACE'  # allows for editing
    gp_stroke.draw_cyclic = True  # closes the stroke
    gp_stroke.material_index = step
    # Define stroke geometry
    gp_stroke.points.add(count=len(polygon))
    for i, p in enumerate(polygon):
        gp_stroke.points[i].co = p
    if step >= max_steps:
        return
    else:
        new_polygon = []
        midpoints = []
        for i in range(len(polygon)):
            p0 = polygon[i]
            p1 = polygon[0] if i == len(polygon)-1 else polygon[i+1]
            opposite_point = (0, 0, 0)
            midpoint = _get_midpoint(p0, p1)
            new_point = _get_midpoint(opposite_point, midpoint)
            for i in range(step):
                new_point = _get_midpoint(new_point, midpoint)
            new_polygon.append(new_point)
            midpoints.append(midpoint)
        polygon_recursive(gp_frame, new_polygon, step+1, max_steps)
        for i in range(len(polygon)):
            other_polygon = [polygon[i], midpoints[i-1], new_polygon[i-1], new_polygon[i], midpoints[i]]
            polygon_recursive(gp_frame, other_polygon, step + 1, max_steps)


def polygon_recursive_2(gp_layer, center, radius, sides, step=0, max_steps=3):
    #Init new stroke
    if len(gp_layer.frames) > step:
        gp_frame = gp_layer.frames[1]
    else:
        gp_frame = gp_layer.frames.new(step)
    #gp_frame = gp_layer.frames.new(0)
    draw_sphere(gp_frame, center, radius, 5, step)
    cube = draw_circle(gp_frame, center, radius, 5)
    if step >= max_steps:
        return
    else:
        polygon_recursive_2(gp_layer, center, radius/2, sides, step+1, max_steps=max_steps)
        new_radius = radius/2
        for center in cube.points:
            polygon_recursive_2(gp_layer, center.co, new_radius/2, sides, step + 1, max_steps=max_steps)


def draw_polygon_fractal(gp_frame, polygon_sides: int):
    # Define base polygon
    angle = 2*math.pi/polygon_sides  # angle in radians
    polygon = []
    for i in range(polygon_sides):
        x = 3*cos(angle*i)
        y = 3*sin(angle*i)
        z = 3
        polygon.append((x, y, z))
    polygon_recursive(gp_frame, polygon, max_steps=5)


#draw_polygon_fractal(gp_frame, 6)
#polygon_recursive_2(gp_layer, (0, 0, 0), 10, 4, 0, max_steps=3)