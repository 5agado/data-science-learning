import numpy as np
from math import pi, cos, sin


def get_image_init_grid(image_path, shape, threshold=0.5):
    from PIL import Image as IMG
    init_image = IMG.open(image_path).convert("L")
    init_image = init_image.resize(shape)
    grid = np.array(init_image) / 255
    grid = grid > threshold
    return grid.astype(int)


def get_circle_grid(h, w, radius_minmax, center=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    grid = np.zeros_like(dist_from_center)
    mask = ((dist_from_center >= radius_minmax[0]) & (dist_from_center < radius_minmax[1]))
    grid[mask] = 1
    return grid.astype(int)


def get_polygon_mask(h, w, segments: int, radius: int, center=None, outline=0, fill=1):
    from PIL import Image, ImageDraw

    if center is None:  # use the middle of the image
        center = (int(h // 2), int(w // 2))

    # build polygon
    angle = 2 * pi / segments  # angle in radians
    polygon = []
    for i in range(segments):
        x = center[0] + radius * cos(angle * i)
        y = center[1] + radius * sin(angle * i)
        polygon.append((x, y))

    img = Image.new('L', (h, w), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=outline, fill=fill)
    mask = np.array(img, dtype=int)
    return mask


def get_perlin_grid(shape, res=1, seed=None):
    from ds_utils.noise_utils import generate_perlin_noise_2d
    noise = generate_perlin_noise_2d(shape, [res, res], seed=seed)
    grid = noise > 0.
    return grid.astype(int)