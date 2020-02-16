import sys
import os
import numpy as np
import pickle
import PIL.Image
from PIL import Image, ImageDraw
import imageio

# Add StyleGAN2 Repo to sys path
# use the original or one of the forks
# I mostly rely on https://github.com/rolux/stylegan2encoder
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
    tflib.set_vars({var: np.random.rand(*var.shape.as_list()) for var in noise_vars})  # [height, width]

    Gs_kwargs.truncation_psi = truncation_psi

    images = Gs.run(z_latents, None, **Gs_kwargs)  # [minibatch, height, width, channel]
    return images[0]


# synthesize image from dlatents
def synth_image_fun(Gs, dlatens, Gs_kwargs, randomize_noise=False):
    Gs_kwargs.randomize_noise = randomize_noise
    images = Gs.components.synthesis.run(dlatens, **Gs_kwargs)
    return images[0]


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