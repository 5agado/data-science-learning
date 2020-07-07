import sys
import os
import numpy as np
import pickle
import PIL.Image
from pathlib import Path
from PIL import Image, ImageDraw
import imageio

# Add StyleGAN2 Repo to sys path
# use the original or one of the forks
# I mostly rely on my fork of the stylegan2encoder
# https://github.com/5agado/stylegan2encoder
sys.path.append(os.path.join(*[os.pardir]*3, 'stylegan2encoder'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib

#####################################
#            Version 2
#####################################


def load_network(network_pkl):
    print(f'Loading networks from {network_pkl}...')
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    return Gs, Gs_kwargs, noise_vars


# generate image from z-latents (uses mapping network)
def gen_image_fun(Gs, z_latents, Gs_kwargs, noise_vars, truncation_psi=1.0):
    tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

    Gs_kwargs.truncation_psi = truncation_psi

    images = Gs.run(z_latents, None, **Gs_kwargs)  # [minibatch, height, width, channel]
    return images[0]


# synthesize image from dlatents
def synth_image_fun(Gs, dlatens, Gs_kwargs, randomize_noise=False):
    Gs_kwargs.randomize_noise = randomize_noise
    images = Gs.components.synthesis.run(dlatens, **Gs_kwargs)
    return images[0]


def map_latents(Gs, z_latents, truncation_psi=1.0):
    """
    Map the given latents (z) to intermediate latents (w)
    :return:
    """
    w_avg = Gs.get_var('dlatent_avg')  # [component]
    w = Gs.components.mapping.run(z_latents, None) # TODO ?? truncation_psi=1.
    w = w_avg + (w - w_avg) * truncation_psi
    return w


#####################################
#            Version 1
#####################################

def load_network_v1(network_pkl):
    tflib.init_tf()

    print(f'Loading networks from {network_pkl}...')
    with open(network_pkl, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    Gs_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
                     minibatch_size=8)

    return Gs, Gs_kwargs, None


# generate image from z-latents (uses mapping network)
def gen_image_fun_v1(Gs, z_latents, Gs_kwargs, randomize_noise=False, truncation_psi=1.0):
    images = Gs.run(z_latents, None,
                    randomize_noise=randomize_noise,
                    truncation_psi=truncation_psi,
                    **Gs_kwargs)
    return images[0]


# synthesize image from dlatents
def synth_image_fun_v1(Gs, dlatents, Gs_kwargs, randomize_noise=False):
    images = Gs.components.synthesis.run(dlatents,
                                         randomize_noise=randomize_noise,
                                         **Gs_kwargs)
    return images[0]


#####################################
#            Generic
#####################################

# Create video for projection progress
def create_video(input_dir, out_path):
    imgs = sorted(input_dir.glob("*step*.png"))

    target_imgs = sorted(input_dir.glob("*target*.png"))
    assert len(target_imgs) == 1, "More than one target found?"
    target_img = imageio.imread(target_imgs[0])

    with imageio.get_writer(str(out_path), mode='I') as writer:
        for filename in imgs:
            image = imageio.imread(filename)

            # Concatenate images with original target image
            w,h = image.shape[0:2]
            canvas = PIL.Image.new('RGBA', (w*2,h), 'white')
            canvas.paste(Image.fromarray(target_img), (0, 0))
            canvas.paste(Image.fromarray(image), (w, 0))

            writer.append_data(np.array(canvas))


def load_latents(latents):
    # If not already numpy array, load the latents
    if type(latents) is not np.ndarray:
        latents = np.load(latents)

    # TMP fix for when saved latents as [1, 16, 512]
    if len(latents.shape) == 3:
        assert latents.shape[0] == 1
        latents = latents[0]

    return latents


def load_directions(directions_dir: Path, is_ganspace=False):
    directions = {}

    if is_ganspace:
        for dir_path in directions_dir.glob('*.pkl'):
            with open(str(dir_path), 'rb') as f:
                lat_comp = pickle.load(f)['lat_comp']
            directions[dir_path.stem.split('-')[1]] = lat_comp
    else:
        directions = {e.stem: np.load(directions_dir / f'{e.stem}.npy') for e in directions_dir.glob('*.npy')}

    return directions


#####################################
#         Interactive Plots
#####################################

import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox, Tab


def get_ipywidget_elements(observe_fun, button_fun, entries, entries2,
                           directions,
                           nb_layers=18,
                           min_alpha=0.5, max_alpha=1.5):

    # Main controls
    button = Button(description="Savefig")
    button_reset = Button(description="Reset")

    entries1_dropdown = widgets.Dropdown(
        options=entries,
        value=entries[0],
        description='Content:')

    entries2_dropdown = widgets.Dropdown(
        options=entries2,
        value=entries2[0],
        description='Style:')

    layers = widgets.IntRangeSlider(
        value=[0, nb_layers],
        min=0,
        max=nb_layers,
        step=1,
        description='Layers:',
        continuous_update=False,
        readout_format='d',
    )

    alpha = widgets.FloatSlider(
        value=0.,
        min=-min_alpha,
        max=max_alpha,
        step=0.1,
        description='Alpha:',
        continuous_update=False,
        readout_format='.1f',
    )

    entries_hbox = HBox([entries1_dropdown, entries2_dropdown])
    layers_hbox = HBox([layers, alpha])
    button_hbox = HBox([button, button_reset])

    main_controls = VBox([entries_hbox, layers_hbox, button_hbox])

    # Direction Controls
    directions_controls = []
    directions_coeffs = {}
    directions_layers = {}

    def update_coeffs(key, value):
        directions_coeffs[key] = value

    def update_coeffs_layers(key, value):
        directions_layers[key] = np.arange(value[0], value[1])

    for direction_name in directions:
        directions_coeffs[direction_name] = 0.
        directions_layers[direction_name] = np.arange(0, nb_layers)
        d_layers = widgets.IntRangeSlider(
            value=[0, nb_layers],
            min=0,
            max=nb_layers,
            step=1,
            description='Layers:',
            continuous_update=False,
            readout_format='d',
        )

        d_coeff = widgets.FloatSlider(
            value=0.,
            min=-10,
            max=10,
            step=0.1,
            description='Coeff:' + direction_name,
            continuous_update=False,
            readout_format='.1f',
        )

        d_layers.observe(lambda change, key=direction_name: update_coeffs_layers(key, change['new']), names='value')
        d_layers.observe(lambda change: observe_fun(entries1_dropdown.value,
                                            entries2_dropdown.value,
                                            layers.value, alpha.value,
                                            directions_coeffs, directions_layers), names='value')

        d_coeff.observe(lambda change, key=direction_name: update_coeffs(key, change['new']), names='value')
        d_coeff.observe(lambda change: observe_fun(entries1_dropdown.value,
                                            entries2_dropdown.value,
                                            layers.value, alpha.value,
                                            directions_coeffs, directions_layers), names='value')

        d_controls = VBox([d_layers, d_coeff])
        directions_controls.append(d_controls)

    # Create main tab + directions tabs
    tab = Tab(children=[main_controls, *directions_controls])
    tab.set_title(0, 'main')
    for i, direction_name in enumerate(directions):
        tab.set_title(i+1, direction_name)

    # add observe events
    observe_elements = [entries1_dropdown, entries2_dropdown, layers, alpha]
    # for each widget, add observe event to update image
    for e in observe_elements:
        e.observe(lambda change: observe_fun(entries1_dropdown.value, entries2_dropdown.value,
                                            layers.value, alpha.value,
                                            directions_coeffs, directions_layers),
                 names='value')

    button.on_click(button_fun)

    def reset(_):
        for d_control in directions_controls:
            d_control.children[0].value = [0, nb_layers]
            d_control.children[1].value = 0.

    button_reset.on_click(reset)

    return tab
