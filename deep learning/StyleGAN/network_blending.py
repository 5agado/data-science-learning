# Network blending**: idea of interpolating the weights between different networks.
# Layer-swapping**: more complex approach to blending, "control independently which model you got low and high resolution features from"
# See [StyleGAN network blending](https://www.justinpinkney.com/stylegan-network-blending/)
# Code adapted from https://github.com/justinpinkney/stylegan2 and


import tensorflow as tf
import sys, getopt, os

# Add StyleGAN2 Repo to sys path
# use the original or one of the forks
# I mostly rely on my fork of the stylegan2encoder
# https://github.com/5agado/stylegan2encoder
sys.path.append(os.path.join(*[os.pardir]*3, 'stylegan2encoder'))

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil
from dnnlib.tflib.autosummary import autosummary
import math
import numpy as np

from training import dataset
from training import misc
import pickle

from pathlib import Path
import typer
from typing import Optional


def extract_conv_names(model):
    # layers are G_synthesis/{res}x{res}/...
    # make a list of (name, resolution, level, position)
    # Currently assuming square(?)

    model_names = list(model.trainables.keys())
    conv_names = []

    resolutions = [4 * 2 ** x for x in range(9)]

    level_names = [["Conv0_up", "Const"],
                   ["Conv1", "ToRGB"]]

    position = 0
    # option not to split levels
    for res in resolutions:
        root_name = f"G_synthesis/{res}x{res}/"
        for level, level_suffixes in enumerate(level_names):
            for suffix in level_suffixes:
                search_name = root_name + suffix
                matched_names = [x for x in model_names if x.startswith(search_name)]
                to_add = [(name, f"{res}x{res}", level, position) for name in matched_names]
                conv_names.extend(to_add)
            position += 1

    return conv_names


def blend_models(model_1, model_2, resolution, level, blend_width=None, verbose=False):
    # y is the blending amount which y = 0 means all model 1, y = 1 means all model_2

    # TODO add small x offset for smoother blend animations
    resolution = f"{resolution}x{resolution}"

    model_1_names = extract_conv_names(model_1)
    model_2_names = extract_conv_names(model_2)

    assert all((x == y for x, y in zip(model_1_names, model_2_names)))

    model_out = model_1.clone()

    short_names = [(x[1:3]) for x in model_1_names]
    full_names = [(x[0]) for x in model_1_names]
    mid_point_idx = short_names.index((resolution, level))
    mid_point_pos = model_1_names[mid_point_idx][3]

    ys = []
    for name, resolution, level, position in model_1_names:
        # low to high (res)
        x = position - mid_point_pos
        if blend_width:
            exponent = -x / blend_width
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1 if x > 1 else 0

        ys.append(y)
        if verbose:
            print(f"Blending {name} by {y}")

    tfutil.set_vars(
        tfutil.run(
            {model_out.vars[name]: (model_2.vars[name] * y + model_1.vars[name] * (1 - y))
             for name, y
             in zip(full_names, ys)}
        )
    )

    return model_out


def main(low_res_pkl: Path,  # Pickle file from which to take low res layers
         high_res_pkl: Path,  # Pickle file from which to take high res layers
         resolution: int,  # Resolution level at which to switch between models
         level: int = 0,  # Switch at Conv block 0 or 1?
         blend_width: Optional[float] = None,  # None = hard switch, float = smooth switch (logistic) with given width
         output_grid: Optional[Path] = "blended.jpg",  # Path of image file to save example grid (None = don't save)
         seed: int = 0,  # seed for random grid
         output_pkl: Optional[Path] = None,  # Output path of pickle (None = don't save)
         verbose: bool = False,  # Print out the exact blending fraction
         ):
    grid_size = (3, 3)

    tflib.init_tf()

    with tf.Session() as sess, tf.device('/gpu:0'):
        low_res_G, low_res_D, low_res_Gs = misc.load_pkl(low_res_pkl)
        high_res_G, high_res_D, high_res_Gs = misc.load_pkl(high_res_pkl)

        out = blend_models(low_res_Gs, high_res_Gs, resolution, level, blend_width=blend_width, verbose=verbose)

        if output_grid:
            rnd = np.random.RandomState(seed)
            grid_latents = rnd.randn(np.prod(grid_size), *out.input_shape[1:])
            grid_fakes = out.run(grid_latents, None, is_validation=True, minibatch_size=1)
            misc.save_image_grid(grid_fakes, output_grid, drange=[-1, 1], grid_size=grid_size)

        # TODO modify all the networks
        if output_pkl:
            misc.save_pkl((low_res_G, low_res_D, out), output_pkl)


if __name__ == '__main__':
    typer.run(main)
