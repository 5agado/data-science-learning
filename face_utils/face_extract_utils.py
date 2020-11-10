import logging

import cv2
import PIL
from PIL import Image
import scipy
import scipy.ndimage
import numpy as np

from face_utils.Face import Face

from skimage.transform._geometric import _umeyama


def get_face_mask(face: Face, mask_type,
                  erosion_size=None,
                  dilation_kernel=None,
                  blur_size: int = None):
    """
    Return mask of mask_type for the given face.
    :param face:
    :param mask_type:
    :param erosion_size:
    :param dilation_kernel:
    :param blur_size:
    :return:
    """
    if mask_type == 'hull':
        # we can rotate the hull mask obtained from original image
        # or re-detect face from aligned image, and get mask then
        mask = get_hull_mask(face, 255)
    elif mask_type == 'rect':
        face_img = face.get_face_img()
        mask = np.zeros(face_img.shape, dtype=face_img.dtype)+255
    elif mask_type == 'landmarks':
        mask = get_landmarks_mask(face, 255)
    else:
        logging.error("No such mask type: {}".format(mask_type))
        raise Exception("No such mask type: {}".format(mask_type))

    # apply mask modifiers
    if erosion_size:
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erosion_size)
        mask = cv2.erode(mask, erosion_kernel, iterations=1)
    if dilation_kernel:
        mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    if blur_size:
        mask = cv2.blur(mask, (blur_size, blur_size))

    return mask


def get_hull_mask(from_face: Face, fill_val=1):
    """

    :param from_face:
    :param fill_val: generally 1 or 255
    :return:
    """
    mask = np.zeros(from_face.img.shape, dtype=from_face.img.dtype)

    hull = cv2.convexHull(np.array(from_face.landmarks).reshape((-1, 2)).astype(int)).flatten().reshape((
        -1, 2))
    hull = [(p[0], p[1]) for p in hull]

    cv2.fillConvexPoly(mask, np.int32(hull), (fill_val, fill_val, fill_val))

    return mask


def get_landmarks_mask(face: Face, fill_val=1):
    """
    Return the mask for target landmarks, filling their respective areas with the given value
    :param face:
    :param fill_val:
    :return:
    """
    mask = np.zeros(face.img.shape, dtype=face.img.dtype)

    target_landmarks = list(face.get_eyes())
    target_landmarks.extend([face.landmarks[Face.nose_points]])
    target_landmarks.extend([face.landmarks[Face.mouth]])

    for landmarks in target_landmarks:
        landmark_hull = cv2.convexHull(np.array(landmarks).reshape((-1, 2)).astype(int)).flatten().reshape((-1, 2))
        hull = [(p[0], p[1]) for p in landmark_hull]

        cv2.fillConvexPoly(mask, np.int32(hull), (fill_val, fill_val, fill_val))

    return mask

#################################
#           ALIGNMENT           #
#################################

mean_face_x = np.array([
                        0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                        0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                        0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                        0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                        0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                        0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                        0.553364, 0.490127, 0.42689])
mean_face_y = np.array([
                        0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                        0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                        0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                        0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                        0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                        0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                        0.784792, 0.824182, 0.831803, 0.824182])
default_landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)


# other implementation option see
# https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
def align_face(face, boundary_resize_factor=None, invert=False, img=None):

    if img is None:
        face_img = face.get_face_img(boundary_resize_factor=boundary_resize_factor)
    else:
        face_img = img
    src_landmarks = np.array([(x - face.rect.left, y - face.rect.top) for (x, y) in face.landmarks])

    # need to resize default ones to match given head size
    (w, h) = face.get_face_size()
    translation = None
    if boundary_resize_factor:
        img_w, img_h = face_img.shape[:2][::-1]
        translation = (img_w - w, img_h - h)
        #w += translation[0]
        #h += translation[1]
    # w/1.5 h/1.5
    scaled_default_landmarks = np.array([(int(x * w), int(y * h)) for (x, y) in default_landmarks_2D])
    # default aligned face has only 51 landmarks, so we remove
    # first 17 from the given one in order to align
    src_landmarks = src_landmarks[17:]
    target_landmarks = scaled_default_landmarks

    if invert:
        align_matrix = get_align_matrix(target_landmarks, src_landmarks, translation)
    else:
        align_matrix = get_align_matrix(src_landmarks, target_landmarks, translation)

    aligned_img = cv2.warpAffine(face_img,
                                 align_matrix,
                                 (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)

    return aligned_img, align_matrix


def get_align_matrix(src_landmarks, target_landmarks, translation: tuple = None):
    align_matrix = _umeyama(src_landmarks, target_landmarks, True)[:2]

    if translation:
        align_matrix[0, 2] -= translation[0]//2
        align_matrix[1, 2] -= translation[1]//2

    return align_matrix


# Align function from FFHQ dataset pre-processing step
# https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
def ffhq_align(face, output_size=1024, transform_size=4096, enable_padding=True,
               boundary_resize_factor=None, img=None):
    if img is None:
        face_img = face.get_face_img(boundary_resize_factor=boundary_resize_factor)
    else:
        face_img = img
    face_landmarks = np.array([(x - face.rect.left, y - face.rect.top) for (x, y) in face.landmarks])

    lm = np.array(face_landmarks)
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = Image.fromarray(np.uint8(face_img))

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                        PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return np.array(img)