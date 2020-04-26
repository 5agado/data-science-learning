import bpy
import cv2
import numpy as np
import subprocess
from collections import namedtuple
from itertools import starmap, product

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
from ReactionDiffusionSystem import ReactionDiffusionSystem, get_init_state, get_polygon_mask, ReactionDiffusionException
from ReactionDiffusionSystem import SYSTEM_CORAL_CONFIG, SYSTEM_BACTERIA_CONFIG, SYSTEM_SPIRALS_CONFIG, SYSTEM_ZEBRA_CONFIG
from ds_utils.blender_utils import render
from ds_utils.video_utils import generate_video


# update Blender image with the given content
def update_img(img_name: str, system: ReactionDiffusionSystem):
    content = system.B
    image = bpy.data.images[img_name]

    # flatten content and append alpha (1)
    alpha = np.ones([content.shape[0], content.shape[1], 1])
    content = np.expand_dims(content, -1)
    pixels = np.concatenate([content, content, content, alpha], axis=-1).flatten()

    # assign pixels
    image.pixels = pixels

    # TOFIX
    # needed to update displacement texture to match the new image
    bpy.data.objects['Plane'].modifiers['Displace'].strength = bpy.data.objects['Plane'].modifiers['Displace'].strength


# randomize the material used by the displaced plane
def randomize_material():
    material = bpy.data.materials['Material']

    material.node_tree.nodes["metallic"].outputs[0].default_value = np.random.uniform(low=.1, high=0.9)  # metallic
    material.node_tree.nodes["roughness"].outputs[0].default_value = np.random.uniform(low=.1, high=0.7)  # roughness

    hue = np.random.uniform(low=0., high=1.)
    saturation = np.random.uniform(low=0.1, high=0.95)
    value = np.random.uniform(low=0.5, high=0.95)
    material.node_tree.nodes["Mix"].inputs[2].default_value = (hue, saturation, value, 1.)


# handler called at every frame change
def frame_handler(scene, rf_system, steps, num_frames_change):
    frame = scene.frame_current
    # When reaching final frame, clear handlers
    if frame >= bpy.context.scene.frame_end:
        bpy.app.handlers.frame_change_pre.clear()
    elif (frame % num_frames_change) == 0:
        try:
            rf_system.run_simulation(steps)
        except ReactionDiffusionException as e:
            print(f'System throw exception at frame {frame}')
            bpy.app.handlers.frame_change_pre.clear()
        update_img(img_name, rf_system)


def get_grid_search_configs(nb_vals=2):
    def named_configs(items):
        Config = namedtuple('Config', items.keys())
        return starmap(Config, product(*items.values()))

    grid_search_params = {
        'COEFF_A': np.linspace(0.18, 0.18, 1),
        'COEFF_B': np.linspace(0.09, 0.09, 1),
        'FEED_RATE': np.linspace(0.03, 0.0625, nb_vals),
        'KILL_RATE': np.linspace(0.06, 0.066, nb_vals),
    }
    configs = list(named_configs(grid_search_params))
    return configs


if __name__ == "__main__":
    NUM_FRAMES = 240
    NUM_FRAMES_CHANGE = 1
    min_frames = 40
    do_render = True

    img_name = 'canvas'
    out_path = Path.home() / 'Documents/graphics/generative_art_output/reaction_diffusion/2020_04'

    # system config
    size = 200
    system_shape = tuple([size]*2)
    nb_configs_gs_param = 3
    config = SYSTEM_ZEBRA_CONFIG.copy()
    config['steps'] = 30
    config['random_influence'] = 0.
    config['validate_change_threshold'] = 0.0001
    #config['COEFF_A'] = 1.
    #config['COEFF_B'] = 0.5
    #config['FEED_RATE'] = 0.055
    #config['KILL_RATE'] = 0.062

    system_init_fun = lambda shape: get_init_state(shape, random_influence=config['random_influence'], mask=mask)

    # mask
    mask_name = 'square2'
    mask = get_polygon_mask(system_shape, 4, system_shape[0] // 10, np.array(system_shape) // 2)

    # remove and create image to start from a clean state and match change in canvas size
    assert img_name in bpy.data.images, 'Target image not present'
    bpy.data.images[img_name].scale(size, size)

    bpy.context.scene.frame_set(0)
    bpy.context.scene.frame_end = NUM_FRAMES
    bpy.app.handlers.frame_change_pre.clear()

    # allow to simply run in blender at each frame change
    if not do_render:
        # init reaction diffusion system
        rf_system = ReactionDiffusionSystem(system_shape, config, system_init_fun,
                                            validate_change_threshold=config['validate_change_threshold'])

        randomize_material()

        # set frame handler
        bpy.app.handlers.frame_change_pre.append(lambda x: frame_handler(x, rf_system, config['steps'], NUM_FRAMES_CHANGE))
    # otherwise proceed to render all cofigs
    else:
        render_dir = out_path / f'{NUM_FRAMES}_{size}_{mask_name}'
        render_dir.mkdir(exist_ok=False, parents=True)

        configs = get_grid_search_configs(nb_configs_gs_param)
        with open(str(render_dir / 'logs.txt'), 'w+') as f:
            for run, param_config in enumerate(configs):
                imgs = []
                config.update(param_config._asdict())

                # init reaction diffusion system
                rf_system = ReactionDiffusionSystem(system_shape, config, system_init_fun,
                                                    validate_change_threshold=config['validate_change_threshold'])

                randomize_material()

                # run and render
                for i in range(NUM_FRAMES):
                    try:
                        rf_system.run_simulation(config['steps'])
                    except ReactionDiffusionException as e:
                        print(f'System throw exception at frame {i}')
                        break
                    imgs.append(cv2.normalize(rf_system.B, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
                    update_img(img_name, rf_system)
                    render(str(render_dir / f'still_f_{i:04}.png'))
                    bpy.context.scene.frame_set(i+1)

                # write out config
                config['nb_frames'] = len(imgs)
                f.write(str(config) + '\n')

                # discard if too short
                if len(imgs) < min_frames:
                    continue

                # convert stills from imgs via ffmpeg
                ffmpeg_in = render_dir / f'still_f_%04d.png'
                ffmpeg_out = render_dir / f'run_{run}.mp4'
                subprocess.call(f'ffmpeg -i "{ffmpeg_in}" -pix_fmt yuv420p -vframes {len(imgs)} "{ffmpeg_out}"', shell=True)

                # export maps via cv2
                video_path = render_dir / f'vid_{run}.mp4'
                generate_video(str(video_path), system_shape,
                               frame_gen_fun=lambda i: imgs[i], nb_frames=len(imgs), is_color=False)