import sys
import os
import numpy as np
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

def load_network(network_pkl):
    print(f'Loading networks from {network_pkl}...')
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    return Gs, Gs_kwargs, noise_vars


# generate image from z-latents (uses mapping network
def gen_image_fun(Gs, z_latents, noise_vars, Gs_kwargs):
    tflib.set_vars({var: np.random.rand(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    images = Gs.run(z_latents, None, **Gs_kwargs) # [minibatch, height, width, channel]
    return images[0]

# synthesize image from dlatents
def synth_image_fun(Gs, dlatens, randomize_noise=False):
    images = Gs.components.synthesis.run(dlatens,
                                      randomize_noise=randomize_noise,
                                      output_transform=dict(func=tflib.convert_images_to_uint8,
                                                            nchw_to_nhwc=True))
    return images[0]


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