import math, os, pathlib, shutil, subprocess, sys, time
import cv2
import numpy as np
import torch, torchvision
import torchvision.transforms as T
from einops import rearrange, repeat
from PIL import Image

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def download_depth_models():
    def wget(url, outputdir):
        print(subprocess.run(['wget', url, '-P', outputdir], stdout=subprocess.PIPE).stdout.decode('utf-8'))

    if not os.path.exists(os.path.join(models_path, 'dpt_large-midas-2f21e586.pt')):
        print("Downloading dpt_large-midas-2f21e586.pt...")
        wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", models_path)
    if not os.path.exists('pretrained/AdaBins_nyu.pt'):
        print("Downloading AdaBins_nyu.pt...")
        os.makedirs('pretrained', exist_ok=True)
        wget("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt",
             'pretrained')


def load_depth_model(optimize=True, device=None):
    midas_model = DPTDepthModel(
        path=f"{models_path}/dpt_large-midas-2f21e586.pt",
        backbone="vitl16_384",
        non_negative=True,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    midas_transform = T.Compose([
        Resize(
            384, 384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet()
    ])

    midas_model.eval()
    if optimize:
        if device == torch.device("cuda"):
            midas_model = midas_model.to(memory_format=torch.channels_last)
            midas_model = midas_model.half()
    midas_model.to(device)

    return midas_model, midas_transform


@torch.no_grad()
def transform_image_3d(prev_img_cv2, adabins_helper, midas_model, midas_transform, rot_mat, translate, anim_args):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion

    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    # predict depth with AdaBins
    use_adabins = anim_args.midas_weight < 1.0 and adabins_helper is not None
    if use_adabins:
        print(f"Estimating depth of {w}x{h} image with AdaBins...")
        MAX_ADABINS_AREA = 500000
        MIN_ADABINS_AREA = 448 * 448

        # resize image if too large or too small
        img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2, cv2.COLOR_RGB2BGR))
        image_pil_area = w * h
        resized = True
        if image_pil_area > MAX_ADABINS_AREA:
            scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w * scale), int(h * scale)),
                                         Image.LANCZOS)  # LANCZOS is good for downsampling
            print(f"  resized to {depth_input.width}x{depth_input.height}")
        elif image_pil_area < MIN_ADABINS_AREA:
            scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            print(f"  resized to {depth_input.width}x{depth_input.height}")
        else:
            depth_input = img_pil
            resized = False

        # predict depth and resize back to original dimensions
        try:
            _, adabins_depth = adabins_helper.predict_pil(depth_input)
            if resized:
                adabins_depth = torchvision.transforms.functional.resize(
                    torch.from_numpy(adabins_depth),
                    torch.Size([h, w]),
                    interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC
                )
            adabins_depth = adabins_depth.squeeze()
        except:
            print(f"  exception encountered, falling back to pure MiDaS")
            use_adabins = False
        torch.cuda.empty_cache()

    if midas_model is not None:
        # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
        img_midas = prev_img_cv2.astype(np.float32) / 255.0
        img_midas_input = midas_transform({"image": img_midas})["image"]

        # MiDaS depth estimation implementation
        print(f"Estimating depth of {w}x{h} image with MiDaS...")
        sample = torch.from_numpy(img_midas_input).float().to(device).unsqueeze(0)
        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        midas_depth = midas_model.forward(sample)
        midas_depth = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        midas_depth = midas_depth.cpu().numpy()
        torch.cuda.empty_cache()

        # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
        midas_depth = np.subtract(50.0, midas_depth)
        midas_depth = midas_depth / 19.0

        # blend between MiDaS and AdaBins predictions
        if use_adabins:
            depth_map = midas_depth * anim_args.midas_weight + adabins_depth * (1.0 - anim_args.midas_weight)
        else:
            depth_map = midas_depth

        depth_map = np.expand_dims(depth_map, axis=0)
        depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)
    else:
        depth_tensor = torch.ones((h, w), device=device)

    pixel_aspect = 1.0  # aspect of an individual pixel (so usually 1.0)
    near, far, fov_deg = anim_args.near_plane, anim_args.far_plane, anim_args.fov
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, R=rot_mat,
                                              T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y, x = torch.meshgrid(torch.linspace(-1., 1., h, dtype=torch.float32, device=device),
                          torch.linspace(-1., 1., w, dtype=torch.float32, device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1, 1, h, w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

    image_tensor = torchvision.transforms.functional.to_tensor(Image.fromarray(prev_img_cv2)).to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1 / 512 - 0.0001).unsqueeze(0),
        offset_coords_2d,
        mode=anim_args.sampling_mode,
        padding_mode=anim_args.padding_mode,
        align_corners=False
    )

    # convert back to cv2 style numpy array 0->255 uint8
    result = rearrange(
        new_image.squeeze().clamp(0, 1) * 255.0,
        'c h w -> h w c'
    ).cpu().numpy().astype(np.uint8)
    return result
