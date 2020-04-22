import bpy
import numpy as np

# Blender import system clutter
import sys
from pathlib import Path

SRC_PATH = Path.home() / "Documents/python_workspace/data-science-learning/graphics/reaction_diffusion"
UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(SRC_PATH))
sys.path.append(str(UTILS_PATH))

import ReactionDiffusionSystem
import ds_utils.blender_utils
import importlib
importlib.reload(ReactionDiffusionSystem)
importlib.reload(ds_utils.blender_utils)
from ReactionDiffusionSystem import ReactionDiffusionSystem, get_init_state
from ReactionDiffusionSystem import SYSTEM_CORAL_CONFIG, SYSTEM_BACTERIA_CONFIG, SYSTEM_SPIRALS_CONFIG, SYSTEM_ZEBRA_CONFIG


def update_img(img_name: str, content):
    image = bpy.data.images[img_name]

    # flatten content and append alpha (1)
    alpha = np.ones([content.shape[0], content.shape[1], 1])
    content = np.expand_dims(content, -1)
    pixels = np.concatenate([content, content, content, alpha], axis=-1).flatten()

    # assign pixels
    image.pixels = pixels


# handler called at every frame change
def frame_handler(scene, rf_system, steps, num_frames_change):
    frame = scene.frame_current
    # When reaching final frame, clear handlers
    if frame >= bpy.context.scene.frame_end:
        bpy.app.handlers.frame_change_pre.clear()
    elif (frame % num_frames_change) == 0:
        rf_system.run_simulation(steps)
        update_img(img_name, rf_system.B)
        # needed to update displacement texture to match the new image
        bpy.data.objects['Plane'].modifiers["Displace"].strength = 0.3


if __name__ == "__main__":
    NUM_FRAMES = 300
    NUM_FRAMES_CHANGE = 1
    bpy.context.scene.frame_set(1)
    bpy.context.scene.frame_end = NUM_FRAMES

    img_name = 'canvas'

    size = 200
    steps = 30
    config = SYSTEM_SPIRALS_CONFIG.copy()
    config['COEFF_A'] = 1.
    config['COEFF_B'] = 0.5
    #config['FEED_RATE'] = 0.055
    #config['KILL_RATE'] = 0.062

    # create image if it doesn't exist
    if img_name not in bpy.data.images:
        image = bpy.data.images.new(img_name, width=size, height=size)

    # init reaction diffusion system
    rf_system = ReactionDiffusionSystem((size, size), config, lambda shape: get_init_state(shape, 'CENTER', random_influence=0))

    # set frame handler
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(lambda x: frame_handler(x, rf_system, steps, NUM_FRAMES_CHANGE))