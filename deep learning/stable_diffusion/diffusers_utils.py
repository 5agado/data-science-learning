# utils for https://github.com/huggingface/diffusers
import torch
import random
import json, os, sys
import cv2
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

import diffusers

diffusers.logging.set_verbosity_info()

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers import LMSDiscreteScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler


sys.path.append(str(Path(__file__).parents[2]))

from ds_utils.video_utils import imageio_generate_video
from image_utils import _load_img, load_mask_img


class Config:
    def __init__(self):
        self.model_path = None
        self.n_steps = 50
        self.scheduler = 'ddim'
        self.scale = 7.5  # classifier-free guidance, how strongly to match the prompt (at the cost of image quality or diversity)
        self.seed = 42
        self.strength = 0.75  # from 0.0 to 1.0, strength of noise applied on the init image. 0 implies unchanged init image
        self.init_img = None
        self.mask_path = None
        self.invert_mask = False
        self.seamless = False
        self.num_images_per_prompt = 1
        self.num_batch_images = 1
        self.outdir = ''
        self.prompt = ''
        self.negative_prompt = ''
        self.H = 512
        self.W = 512
        self.full_precision = False
        self.skip_frames = 1

    def process_config(self):
        if self.seed == -1:
            self.seed = random.randint(0, 2 ** 32)

        self.W, self.H = map(lambda x: x - x % 64, (self.W, self.H))  # resize to integer multiple of 64

        if self.init_img is not None and self.init_img == '':
            self.init_img = None

        if self.mask_path is not None and self.mask_path == '':
            self.mask_path = None

        if self.init_img is None:
            self.strength = 0.


def save_config_to_file(opt: Config, batch_name, batch_idx):
    settings = vars(opt)
    os.makedirs(opt.outdir, exist_ok=True)
    while os.path.isfile(f"{opt.outdir}/{batch_name}_{batch_idx}_settings.txt"):
        batch_idx += 1
    with open(f"{opt.outdir}/{batch_name}_{batch_idx}_settings.txt", "w+", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
    return batch_idx


def set_conv_padding_type(targets, is_seamless):
    # allow for seamless generation if specified
    for target in targets:
        for module in target.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                module.padding_mode = 'circular' if is_seamless else module.orig_padding_mode


def load_pipelines(opt: Config, device):
    if opt.scheduler == 'ddim':
        scheduler = DDIMScheduler.from_pretrained(opt.model_path, subfolder="scheduler")
    elif opt.scheduler == 'lmsd': # better than DDIM
        scheduler = LMSDiscreteScheduler.from_pretrained(opt.model_path, subfolder="scheduler")
    elif opt.scheduler == 'euler': # can generate high quality results with as little as 30 steps
        scheduler = EulerDiscreteScheduler.from_pretrained(opt.model_path, subfolder="scheduler")
    elif opt.scheduler == 'dpms': # best speed/quality trade-off and can be run with as little as 20 steps
        scheduler = DPMSolverMultistepScheduler.from_pretrained(opt.model_path, subfolder="scheduler")
    else:
        print(f'No such scheduler: {opt.scheduler}. Defaulting to DDIM')
        scheduler = DDIMScheduler.from_pretrained(opt.model_path, subfolder="scheduler")

    sd_pipe = StableDiffusionPipeline.from_pretrained(opt.model_path,
                                                      torch_dtype=torch.float32 if opt.full_precision else torch.float16,
                                                      scheduler=scheduler,
                                                      safety_checker=None, requires_safety_checker=False).to(device)

    # store original padding type for conv2d modules (used for seamless img generation)
    for target in [sd_pipe.vae, sd_pipe.text_encoder, sd_pipe.unet]:
        for module in target.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                module.orig_padding_mode = module.padding_mode

    img2img_pipe = StableDiffusionImg2ImgPipeline(unet=sd_pipe.unet, vae=sd_pipe.vae, scheduler=sd_pipe.scheduler,
                                                  feature_extractor=sd_pipe.feature_extractor,
                                                  text_encoder=sd_pipe.text_encoder, tokenizer=sd_pipe.tokenizer,
                                                  safety_checker=None, requires_safety_checker=False).to(device)
    inpaint_pipe = StableDiffusionInpaintPipeline(unet=sd_pipe.unet, vae=sd_pipe.vae, scheduler=sd_pipe.scheduler,
                                                  feature_extractor=sd_pipe.feature_extractor,
                                                  text_encoder=sd_pipe.text_encoder, tokenizer=sd_pipe.tokenizer,
                                                  safety_checker=None, requires_safety_checker=False).to(device)

    return sd_pipe, img2img_pipe, inpaint_pipe


def generate(opt: Config, pipe, batch_name, batch_idx, sample_idx):
    generator = torch.Generator(device=pipe.device).manual_seed(opt.seed)
    pipe_dict = {
        'prompt': opt.prompt,
        'generator': generator,
        'num_inference_steps': opt.n_steps,
        'guidance_scale': opt.scale,
        'num_images_per_prompt': opt.num_images_per_prompt,
        'negative_prompt': opt.negative_prompt
    }

    set_conv_padding_type([pipe.vae, pipe.text_encoder, pipe.unet], opt.seamless)

    if opt.init_img is not None:
        pipe_dict['image'] = _load_img(opt.init_img, (opt.W, opt.H))
        if opt.mask_path is None:
            pipe_dict['strength'] = opt.strength
        else:
            pipe_dict['mask_image'] = _load_img(opt.init_img, (opt.W, opt.H))
    else:
        pipe_dict['height'] = opt.H
        pipe_dict['width'] = opt.W

    images = pipe(**pipe_dict).images

    for image in images:
        filepath = os.path.join(opt.outdir, f'{batch_name}_{batch_idx}_{sample_idx:04}.png')
        if sample_idx == 0:
            print(f'Saving to {filepath}')
        image.save(filepath)
        sample_idx += 1
    return images

def diffusion_process_run(opt: Config, pipe, batch_name, batch_idx):
    batch_idx = save_config_to_file(opt, batch_name, batch_idx)
    sample_idx = 0
    video_extensions = ['.mp4', '.webm']
    # process video
    if opt.init_img is not None and Path(opt.init_img).suffix in video_extensions:
        if opt.mask_path is not None:
            assert Path(opt.mask_path).suffix in video_extensions, 'Video mask must be a video itself.'
            mask_video = cv2.VideoCapture(opt.mask_path)
        print('Run on video')
        video = cv2.VideoCapture(opt.init_img)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS) / opt.skip_frames

        frames_out = []
        for frame_index in tqdm(range(frame_count), desc='Running on video', position=0):
            ret, frame = video.read()
            if opt.mask_path is not None:
                _, mask_frame = mask_video.read()
            if (frame_index % opt.skip_frames) != 0:
                continue
            if ret:
                opt.init_img = frame
                if opt.mask_path is not None:
                    opt.mask_path = mask_frame
                images = generate(opt, pipe, batch_name, batch_idx, sample_idx)
                sample_idx += 1
                frames_out.append(images[0])
        # generate video
        imageio_generate_video(str(f'{opt.outdir}/{batch_name}_{batch_idx}.mp4'),
                               [np.array(img) for img in frames_out], fps=fps, format="mp4")
        # remove generated frames
        for filename in Path(opt.outdir).glob(f'{batch_name}_{batch_idx}_*.png'):
            os.remove(filename)
            # process image
    else:
        for i in tqdm(range(opt.num_batch_images)):
            _ = generate(opt, pipe, batch_name, batch_idx, sample_idx)
            opt.seed += 1
            sample_idx += 1
