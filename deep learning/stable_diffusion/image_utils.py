import os
import random
import requests
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.exposure import match_histograms

import torch
import torchvision.transforms.functional as TF


def _load_img(img_path, shape):
    if img_path.startswith('http://') or img_path.startswith('https://'):
        image = Image.open(requests.get(img_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)
    return image


def load_img(img_path, shape):
    image = _load_img(img_path, shape)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_mask_img(img_path, shape):
    mask = _load_img(img_path, (shape[-1], shape[-2]))
    mask = mask.convert("L")
    return mask


def prepare_mask(mask_path, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0,
                 invert_mask=False):
    """

    :param mask_path: path to the mask image
    :param mask_shape: shape of the image to match, usually latent_image.shape
    :param mask_brightness_adjust: amount to adjust brightness of the iamge, 0 is black, 1 is no adjustment, >1 is brighter
    :param mask_contrast_adjust: amount to adjust contrast of the image, 0 is a flat grey image, 1 is no adjustment, >1 is more contrast
    :return:
    """

    mask = load_mask_img(mask_path, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask)

    if invert_mask:
        mask = ((mask - 0.5) * -1) + 0.5

    mask = np.clip(mask, 0, 1)
    return mask


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)