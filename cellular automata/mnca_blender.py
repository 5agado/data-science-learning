import bpy
import numpy as np
import subprocess
from collections import namedtuple
from itertools import starmap, product

# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata"
sys.path.append(str(SRC_PATH))
sys.path.append(str(UTILS_PATH))

import Automaton
import mnca_utils
import automata_blender_utils
import ds_utils.blender_utils
import importlib
importlib.reload(Automaton)
importlib.reload(mnca_utils)
importlib.reload(automata_blender_utils)
importlib.reload(ds_utils.blender_utils)
from Automaton import MultipleNeighborhoodAutomaton, get_kernel_2d_square
from mnca_utils import get_circle_grid


# Current setup works only with preset Blender file.
# To use script need to add needed components to the scene and use same names


# update Blender image with the given content
def update_img(img_name: str, system):
    content = system.grid
    image = bpy.data.images[img_name]

    # flatten content and append alpha (1)
    alpha = np.ones([content.shape[0], content.shape[1], 1])
    content = np.expand_dims(content, -1)
    pixels = np.concatenate([content, content, content, alpha], axis=-1).flatten()

    # assign pixels
    image.pixels = pixels

    # TOFIX
    # needed to update displacement texture to match the new image
    #bpy.data.objects['Plane'].modifiers['Displace'].strength = bpy.data.objects['Plane'].modifiers['Displace'].strength
    bpy.data.node_groups["Geometry Nodes"].nodes["Image Texture"].inputs[2].default_value = 0


# handler called at every frame change
def frame_handler(scene, automaton, num_frames_change):
    frame = scene.frame_current
    # When reaching final frame, clear handlers
    if frame >= bpy.context.scene.frame_end:
        bpy.app.handlers.frame_change_pre.clear()
    elif (frame % num_frames_change) == 0:
        automaton.update()
        update_img(img_name, automaton)


if __name__ == "__main__":
    NUM_FRAMES = 240
    NUM_FRAMES_CHANGE = 1

    img_name = 'canvas'
    out_path = Path.home() / 'Documents/graphics/generative_art_output/reaction_diffusion/2020_04'

    configs = [
        {'neighbours_count_born': [0.190, 0.220],
         'neighbours_maxcount_survive': [0.350, 0.500],
         'neighbours_mincount_survive': [0.750, 0.850],
         },
        {'neighbours_count_born': [0.410, 0.550],
         'neighbours_maxcount_survive': [0.100, 0.280],
         'neighbours_mincount_survive': [0.120, 0.150],
         },
    ]

    kernels = [
        get_circle_grid(17, 17, radius_minmax=[5, 7]),
        get_circle_grid(9, 9, radius_minmax=[1, 3]),

    ]

    # system config
    size = 400

    # remove and create image to start from a clean state and match change in canvas size
    assert img_name in bpy.data.images, 'Target image not present'
    bpy.data.images[img_name].scale(size, size)

    bpy.context.scene.frame_set(0)
    bpy.context.scene.frame_end = NUM_FRAMES
    bpy.app.handlers.frame_change_pre.clear()

    # allow to simply run in blender at each frame change

    # init reaction diffusion system
    mnca = MultipleNeighborhoodAutomaton((size, size), configs=configs, kernels=kernels)
    grid = get_circle_grid(mnca.shape[0], mnca.shape[1], radius_minmax=[0, 90])
    mnca.grid = grid

    #randomize_material()

    # set frame handler
    bpy.app.handlers.frame_change_pre.append(lambda x: frame_handler(x, mnca, NUM_FRAMES_CHANGE))