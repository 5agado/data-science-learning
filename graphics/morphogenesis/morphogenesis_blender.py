# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "graphics/morphogenesis"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))

import Morphogenesis
import importlib
importlib.reload(Morphogenesis)
from Morphogenesis import Morphogenesis

import bpy
from math import cos, sin, pi
import numpy as np
import itertools

from utils.blender_utils import init_grease_pencil, draw_line


def draw_segment(gp_frame, points, material_index=0, draw_cyclic=False):
    # Init new stroke
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.display_mode = '3DSPACE'  # allows for editing
    gp_stroke.draw_cyclic = draw_cyclic
    gp_stroke.material_index = material_index

    # Define stroke geometry
    gp_stroke.points.add(count=len(points))
    for i, p in enumerate(points):
        gp_stroke.points[i].co = p
    return gp_stroke


def morphogenesis(morphogenesis_config, nb_nodes, is_circle, draw_debug, draw_progress,
                  num_frames, name):
    # Circle
    if is_circle:
        center = (0, 0, 0)
        radius = 1
        angle = 2 * pi / nb_nodes

        nodes = []
        for i in range(nb_nodes):
            x = center[0] + radius * cos(angle * i)
            y = center[1] + radius * sin(angle * i)
            z = center[2]
            nodes.append(np.array([x, y, z]))
    # Line
    else:
        nodes = [(np.array((x, 0, 0)) + (0.5 - np.random.rand(3))) * np.array([1., 1., 0.])
                 for x in np.linspace(1, 11, nb_nodes)]

    morphogenesis = Morphogenesis(nodes, closed=is_circle, config=morphogenesis_config)

    if draw_debug:
        draw_force_fun = lambda p0, p1, mat_idx: draw_line(gp_frame, p0, p1, material_index=mat_idx)
        draw_segment_fun = lambda nodes, mat_idx: draw_segment(gp_frame, nodes, material_index=mat_idx,
                                                               draw_cyclic=is_circle)
    else:
        draw_force_fun = draw_segment_fun = None

    # Run and Draw Simulation
    gp_layer = init_grease_pencil(gpencil_obj_name=name, clear_layer=True)
    gp_layer.frames.new(-1)
    for frame in range(num_frames):
        print("Updating frame {}".format(frame))
        if draw_progress:
            gp_frame = gp_layer.frames.copy(gp_layer.frames[-1])
            gp_frame.frame_number = frame
        else:
            gp_frame = gp_layer.frames.new(frame)
        morphogenesis.update(draw_force=draw_force_fun, draw_segment=draw_segment_fun)
        draw_segment(gp_frame, morphogenesis.nodes, draw_cyclic=is_circle)


def main():
    print("###################################")

    nb_nodes = 5
    num_frames = 40
    bpy.context.scene.frame_end = num_frames
    draw_progress = True
    draw_debug = True

    is_circle = True
    morphogenesis_config = {
        'VISIBILITY_RADIUS': 0.4,
        'REPULSION_FAC': 1 / 20,
        'ATTRACTION_FAC': 1 / 20,
        'SPLIT_DIST_THRESHOLD': 0.2,
        'RAND_OPTIMIZATION_FAC': 0,
    }

    N = 5
    visibility_radiuses = np.linspace(0.1, 0.5, N)

    for i in range(10):
        morphogenesis_config['VISIBILITY_RADIUS'] = visibility_radiuses[i]
        morphogenesis(morphogenesis_config, nb_nodes, is_circle, draw_debug, draw_progress,
                      num_frames, "exp_{}".format(i))

main()

