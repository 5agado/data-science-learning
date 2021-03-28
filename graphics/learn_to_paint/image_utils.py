import sys
import cv2
import math
from skimage import draw
import numpy as np
from typing import List
from pathlib import Path
import logging
import argparse


def alpha_blur(img, alpha_mask, kernel_size=10):
    """
    Blur image proportional to the given mask
    :param img:
    :param alpha_mask:
    :param kernel_size:
    :return:
    """
    # apply morphology open to smooth the outline
    # kernel_size = max(2, (10 // (i + 1)))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # blurred_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # kernel_size = max(20, (150 // (i + 1)))

    #blurred_img = cv2.bilateralFilter(img, 9, 5, 5)
    blurred_img = cv2.GaussianBlur(img, (21, 21), 11)

    if alpha_mask is not None:
        if alpha_mask.ndim == 3 and alpha_mask.shape[-1] == 3:
            alpha = alpha_mask / 255.0
        else:
            alpha = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR) / 255.0
        blurred_img = cv2.convertScaleAbs(blurred_img * (1 - alpha) + img * alpha)

    return blurred_img


def sample_color(img, x, y, neighbor_size):
    # sample color from image => converges faster.
    color = img[max(0, y - neighbor_size):y + neighbor_size,
               max(0, x - neighbor_size):x + neighbor_size].mean(axis=(0,1))

    return color


def get_phase_and_magnitude(img, sobel_kernel_size=7, magnitude_power=0.3):
    """
    Calculate phase/rotation angle from image gradient
    :param img: image to compute phase from
    :param sobel_kernel_size:
    :return: phase in float32 radian
    """
    # grayify
    img_gray = img.astype('float32') #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')

    # gradient (along x and y axis)
    xg = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    yg = - cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)

    # calculates the rotation angle of the 2D vectors gradients
    phase = cv2.phase(xg, yg)

    # calculates the magnitude of the 2D vectors gradients
    magnitude = cv2.magnitude(xg, yg)
    magnitude = magnitude / magnitude.max()  # normalize to [0, 1] range

    # make magnitude more uniform
    magnitude = np.power(magnitude, magnitude_power)

    return phase, magnitude


def get_point_angle_and_magnitude(x, y, phase_map, magnitude_map,
                                  phase_neighbor_size: int, magnitude_neighbor_size: int):
    # get gradient orientation info from phase map (phase should be between [0,2pi))
    # compute an average phase around the point, for an area proportional to brush size
    phase = phase_map[max(0, y - phase_neighbor_size):y + phase_neighbor_size,
            max(0, x - phase_neighbor_size):x + phase_neighbor_size].mean()

    # choose direction perpendicular to gradient
    angle = (((phase / math.pi) * 180) + 90) % 360

    magnitude = magnitude_map[max(0, y - magnitude_neighbor_size):y + magnitude_neighbor_size,
                max(0, x - magnitude_neighbor_size):x + magnitude_neighbor_size].mean()

    return angle, magnitude


def get_edges(img, img_blur_size=5, min_hyst_val=100, max_hyst_val=200, edges_blur_size=5):
    """
    Detect image edges
    :param img:
    :param img_blur_size: kernel for gaussian-blur on image
    :param min_hyst_val: hysteresis min threshold (canny edge detection)
    :param max_hyst_val: hysteresis max threshold (canny edge detection)
    :param edges_blur_size: blur size applied to edge results
    :return: norm blurred edges and uint8 original edges images
    """
    # remove noise to improve edge detection results
    blurred_img = cv2.GaussianBlur(img, (img_blur_size, img_blur_size), 0)

    # canny edge detection
    edges = cv2.Canny((blurred_img * 255).astype('uint8'), min_hyst_val, max_hyst_val)

    # blur edges
    blurred_edges = cv2.blur(edges, (edges_blur_size, edges_blur_size)).astype('float32') / 255

    norm_edges = blurred_edges / blurred_edges.sum()  # normalize to probabilities

    return norm_edges, edges


def get_distance_map(src_img):
    """
    Get distance values for given image. Distance is the closest zero pixel for each pixel of the source image.
    :param src_img: grayscale image
    :return: distance map
    """
    # use simple euclidean distance and a 3×3 mask for a fast, coarse distance estimation
    dist = cv2.distanceTransform(255 - src_img, cv2.DIST_L2, 3)

    # normalize distance image between 0.0 and 1.0
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    return dist


def get_min_radius_to_edge(dist_img, start_pos, end_pos, dist_threshold=0.01):
    # get lines coordinates
    line = np.transpose(np.array(draw.line(start_pos[0], start_pos[1], end_pos[0], end_pos[1])))
    line = np.array([[x, y] for [x, y] in line if (0 <= x < dist_img.shape[1] and 0<= y < dist_img.shape[0])])
    # get dist values overlapping the line
    data = dist_img[line[:, 1], line[:, 0]]

    # find first index below threshold
    radius = np.argmax(data < dist_threshold)

    return radius


def add_border_to_img(img, border_size: int):
    border_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                                    value=[255] * 3)
    border_img = border_img.astype('float32') / 255  # convert to float32

    img = cv2.copyMakeBorder(img, border_size // 2, border_size // 2, border_size // 2, border_size // 2,
                             cv2.BORDER_CONSTANT, value=[255] * 3)
    img = cv2.resize(img, border_img.shape[:2][::-1])
    img = img.astype('float32') / 255  # convert to float32

    return img, border_img


def combine_salience_images(input_dir: Path, salience_paths: List[Path], weights: List[float], output_dir: Path):
    main_imgs = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))

    output_dir.mkdir(exist_ok=True, parents=True)

    # for each image in out target input folder, compute composed salience image
    for img_path in main_imgs:
        main_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype('float32') / 255
        composed_salience_img = np.zeros(main_img.shape, dtype='float32')

        salience_img_name = img_path.stem
        for i, salience_path in enumerate(salience_paths):
            salience_img_ext = None
            for ext in ['.jpg', '.png']:
                if (salience_path / (salience_img_name + ext)).exists():
                    salience_img_ext = ext
                    break
            if salience_img_ext is None:
                continue

            salience_img_path = str(salience_path / (salience_img_name + ext))
            salience_img = cv2.imread(str(salience_img_path), cv2.IMREAD_GRAYSCALE)
            salience_img = salience_img.astype('float32') / 255  # convert to float32
            salience_img = salience_img.clip(0.)
            composed_salience_img = composed_salience_img + (weights[i] * salience_img)

        composed_salience_img = composed_salience_img / composed_salience_img.max()

        cv2.imwrite(str(output_dir / f'{salience_img_name}.png'), composed_salience_img * 255)


def main(_=None):
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Image Utils')

    parser.add_argument('-i', '--input-path', required=True)
    parser.add_argument('-o', '--output-path', required=True)
    parser.add_argument('-s', '--salience-paths', type=str, nargs='+',)

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    salience_paths = [Path(p) for p in args.salience_paths]
    weights = [1. for p in salience_paths]

    combine_salience_images(input_path, salience_paths=salience_paths, weights=weights, output_dir=output_path)


if __name__ == "__main__":
    main(sys.argv[1:])