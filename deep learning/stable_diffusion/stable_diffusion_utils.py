# adapted from the following resources:
# https://github.com/deforum/stable-diffusion

import os, random, sys, gc
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from einops import rearrange, repeat
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from pathlib import Path

projects_dir = Path.home() / 'Documents/python_workspace'
sys.path.append(str(projects_dir / 'data-science-learning'))
sys.path.append(str(projects_dir / 'CLIP'))
sys.path.append(str(projects_dir / 'k-diffusion'))
#sys.path.append(str(projects_dir / 'stable-diffusion'))  # https://github.com/CompVis/stable-diffusion
sys.path.append(str(projects_dir / 'stablediffusion'))    # https://github.com/Stability-AI/stablediffusion
#sys.path.append(str(projects_dir / 'InvokeAI'))          # https://github.com/invoke-ai/InvokeAI
sys.path.append(str(projects_dir / 'taming-transformers'))

# dependent repos utils
from ldm.util import instantiate_from_config
#from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from k_diffusion.external import CompVisDenoiser

# local utils
from image_utils import load_img, prepare_mask
from k_samplers import sampler_fn
from ddim import DDIMSampler

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_conv_padding_type(model, is_seamless):
    # allow for seamless generation if specified
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.padding_mode = 'circular' if is_seamless else m._orig_padding_mode


def load_model_from_config(config, ckpt, full_precision= False):
    print(f'>> Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if full_precision:
        print('Using slower but more accurate full-precision math')
    else:
        print('Using half precision math')
        model.half()

    model = model.to(device)
    model.eval()

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m._orig_padding_mode = m.padding_mode

    return model


class config():
    def __init__(self):
        self.ckpt = ''
        self.config = ''
        self.ddim_eta = 0.0  # The DDIM sampling eta constant. If equal to 0 makes the sampling process deterministic
        self.n_steps = 100
        self.sampler = 'klms'
        self.scale = 7.5  # classifier-free guidance, how strongly to match the prompt (at the cost of image quality or diversity)
        self.seed = 42
        self.strength = 0.75  # strength for noising/unnoising. how much the init image is used
        self.init_strength = None
        self.fixed_code = True # ???
        self.init_img = None
        self.init_latent = None
        self.mask_path = None
        self.mask_contrast_adjust = 1.
        self.mask_brightness_adjust = 1.
        self.invert_mask = False
        self.dynamic_threshold = None
        self.static_threshold = None
        self.seamless = False
        self.n_iter = 1
        self.n_samples = 1 # not exposed, you can do 2 or more based on GPU ram
        self.outdir = ''
        self.prompt = ''
        self.H = 512
        self.W = 512
        self.C = 4  # number of channels in the images
        self.f = 8  # image to latent space resolution reduction
        self.precision = 'autocast'
        self.full_precision = False

    def process_config(self):
        if self.seed == -1:
            self.seed = random.randint(0, 2 ** 32)

        self.init_strength = self.strength
        self.strength = max(0.0, min(1.0, 1.0 - self.strength))
        self.W, self.H = map(lambda x: x - x % 64, (self.W, self.H))  # resize to integer multiple of 64

        if self.init_img is not None and self.init_img == '':
            self.init_img = None

        if self.mask_path is not None and self.mask_path == '':
            self.mask_path = None

        # if self.init_img is not None and self.init_img != '':
        #     self.sampler = 'ddim'
        #
        # if self.sampler != 'ddim':
        #     self.ddim_eta = 0.0


def make_callback(sampler_name, dynamic_threshold=None, static_threshold=None, mask=None, init_latent=None, sigmas=None,
                  sampler=None, masked_noise_modifier=1.0):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image at each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(args_dict):
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1 * static_threshold, static_threshold)
        if mask is not None:
            init_noise = init_latent + noise * args_dict['sigma']
            is_masked = torch.logical_and(mask >= mask_schedule[args_dict['i']], mask != 0)
            new_img = init_noise * torch.where(is_masked, 1, 0) + args_dict['x'] * torch.where(is_masked, 0, 1)
            args_dict['x'].copy_(new_img)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1 * static_threshold, static_threshold)
        if mask is not None:
            i_inv = len(sigmas) - i - 1
            init_noise = sampler.stochastic_encode(init_latent, torch.tensor([i_inv] * batch_size).to(device),
                                                   noise=noise)
            is_masked = torch.logical_and(mask >= mask_schedule[i], mask != 0)
            new_img = init_noise * torch.where(is_masked, 1, 0) + img * torch.where(is_masked, 0, 1)
            img.copy_(new_img)

    if init_latent is not None:
        noise = torch.randn_like(init_latent, device=device) * masked_noise_modifier
    if sigmas is not None and len(sigmas) > 0:
        mask_schedule, _ = torch.sort(sigmas / torch.max(sigmas))
    elif len(sigmas) == 0:
        mask = None  # no mask needed if no steps (usually happens because strength==1.0)

    if sampler_name in ["plms", "ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        if mask is not None:
            assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
            batch_size = init_latent.shape[0]

        callback = img_callback_
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback_

    return callback


def samples_to_images(model, samples):
    images = []
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    for x_sample in x_samples:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        images.append(Image.fromarray(x_sample.astype(np.uint8)))
    return images


def generate(opt, model, batch_name, batch_idx, sample_idx):
    seed_everything(opt.seed)
    os.makedirs(opt.outdir, exist_ok=True)

    set_conv_padding_type(model, opt.seamless)
    model_wrap = CompVisDenoiser(model)
    batch_size = opt.n_samples
    data = [batch_size * [opt.prompt]]

    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.n_steps, ddim_eta=opt.ddim_eta, verbose=False)

    # Get init latent code
    # take the one provided directly, or extract from init-image by encoding it (find its latent representation)
    # if neither it will be later initialized to random values
    if opt.init_latent is not None:
        init_latent = opt.init_latent
    elif opt.init_img is not None:
        init_image = load_img(opt.init_img, shape=(opt.W, opt.H)).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    else:
        init_latent = None

    # Mask functions
    if opt.mask_path is not None:
        assert init_latent is not None, 'A latent init image is required for a mask'
        mask = prepare_mask(opt.mask_path, init_latent.shape, opt.mask_contrast_adjust, opt.mask_brightness_adjust,
                            invert_mask=opt.invert_mask)
        mask = mask.to(device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    t_enc = int(opt.strength * opt.n_steps)
    #t_enc = int((1.0 - opt.strength) * opt.n_steps)  # correct because we flip strength in config?
    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(opt.n_steps)
    k_sigmas = k_sigmas[len(k_sigmas) - t_enc - 1:]

    callback = make_callback(sampler_name=opt.sampler,
                             dynamic_threshold=opt.dynamic_threshold, static_threshold=opt.static_threshold,
                             mask=mask, init_latent=init_latent, sigmas=k_sigmas, sampler=sampler)

    precision_scope = autocast if opt.precision == 'autocast' else nullcontext
    all_images = []
    with torch.no_grad():
        with precision_scope('cuda'), model.ema_scope():
            for n in range(opt.n_iter):
                for prompts in data:
                    # In unconditional scaling is not 1 get the embeddings for empty prompts (no conditioning)
                    un_cond = None if opt.scale == 1 else model.get_learned_conditioning(batch_size * [''])
                    cond = model.get_learned_conditioning(prompts)

                    if opt.sampler in ['klms', 'dpm2', 'dpm2_ancestral', 'heun', 'euler', 'euler_ancestral']:
                        samples = sampler_fn(c=cond, uc=un_cond, args=opt, model_wrap=model_wrap,
                                             init_latent=init_latent, t_enc=t_enc, device=device, cb=callback)
                    else:
                        if init_latent is not None:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                        else:
                            z_enc = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

                        if opt.sampler == 'ddim':
                            samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=un_cond, img_callback=callback,
                                                     mask=mask, init_latent=init_latent)
                        elif opt.sampler == 'plms':  # no "decode" function in plms, so use "sample"
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples, _ = sampler.sample(S=opt.n_steps, conditioning=cond,
                                                        batch_size=opt.n_samples,
                                                        shape=shape, verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=un_cond,
                                                        eta=opt.ddim_eta, x_T=z_enc,
                                                        img_callback=callback)
                        else:
                            raise Exception(f'Sampler {opt.sampler} not recognised.')

                    # export samples to images
                    images = samples_to_images(model, samples)
                    all_images.append(images)
                    for image in images:
                        filepath = os.path.join(opt.outdir, f'{batch_name}_{batch_idx}_{sample_idx:04}.png')
                        if sample_idx == 0:
                            print(f'Saving to {filepath}')
                        image.save(filepath)
                        sample_idx += 1
    return all_images
