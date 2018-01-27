import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os
from os.path import join
import sys

from functools import reduce
import matplotlib.patches as patches
import glob

import dlib
import cv2

class FaceSwapException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class LandmarkDetector:
    # face contour points
    jaw_points = np.arange(0, 17)
    eyebrow_dx_points = np.arange(17, 22)
    eyebrow_sx_points = np.arange(22, 27)
    nose_points = np.arange(27, 36)
    nosecenter_points = np.array([30, 33])

    def __init__(self, predictor_model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_model_path)

    def detect_landmark(self, img):
        rects = self.detector(img, 1)
        # continue only if we detected exactly one face
        if len(rects) == 0:
            raise FaceSwapException("No face detected. Exiting")
            #print("No face detected. Exiting")
        elif len(rects) > 1:
            raise FaceSwapException("More than one face detected. Exiting")
            #print("More than one face detected. Exiting")
        shape = self.predictor(img, rects[0])
        return shape

    def get_contour(self, img):
        shape = self.detect_landmark(img)
        return self.get_contour_points(shape)

    def get_face_horizontal_orientation(self, img):
        shape = self.detect_landmark(img)
        points = np.array([p.x for p in shape.parts()])
        nose_center = points[LandmarkDetector.nosecenter_points].mean()
        #print(nose_center)
        #print(shape.rect.center().x)
        if nose_center < shape.rect.center().x:
            return "left"
        else:
            return "right"

    def get_contour_points(self, shape):
        # shape to numpy
        points = np.array([(p.x, p.y) for p in shape.parts()])
        face_boundary = points[np.concatenate([LandmarkDetector.jaw_points,
                                              LandmarkDetector.eyebrow_dx_points,
                                              LandmarkDetector.eyebrow_sx_points])]
        # TODO can simply set shape points to numpy list??
        return face_boundary, shape.rect

def get_convex_hull(points):
    hull_idxs = cv2.convexHull(points, returnPoints=False)


# TODO why rect is smaller than landmark detected?
# for now using image
def delaunay_triangulation(img, rect, hull_idx, face_boundary):
    rect = (0, 0, img.shape[1], img.shape[0])
    # rect = (rect.left(), rect.top(), rect.right(), rect.bottom())
    # print(rect)
    subdiv = cv2.Subdiv2D(rect)
    points = []
    for idx in hull_idx:
        p = tuple(face_boundary[idx[0]])
        # print(p)
        points.append(p)
        subdiv.insert(p)
    triangles = subdiv.getTriangleList()
    return triangles.reshape(triangles.shape[0], 3, 2)

def clean_triangles(rect, triangles):
    # delaunayTri = []
    #
    # pt = []
    #
    # count = 0
    #
    # for t in triangles:
    #     pt.append((t[0], t[1]))
    #     pt.append((t[2], t[3]))
    #     pt.append((t[4], t[5]))
    #
    #     pt1 = (t[0], t[1])
    #     pt2 = (t[2], t[3])
    #     pt3 = (t[4], t[5])
    #
    #     if is_triangle_in(rect, t):
    #         count = count + 1
    #         ind = []
    #         for j in range(0, 3):
    #             for k in range(0, len(points)):
    #                 if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
    #                     ind.append(k)
    #         if len(ind) == 3:
    #             delaunayTri.append((ind[0], ind[1], ind[2]))
    #
    #     pt = []
    #rect = (rect.left(), rect.top(), rect.right(), rect.bottom())
    return list(filter(lambda x: is_triangle_in(rect, x), triangles))


# TODO why rect is smaller than landmark detected?
# for now using image
def get_triangles_indexes(img, rect, hull_idx, face_boundary):
    rect = (0, 0, img.shape[1], img.shape[0])
    # rect = (rect.left(), rect.top(), rect.right(), rect.bottom())
    subdiv = cv2.Subdiv2D(rect)
    points = []
    for idx in hull_idx:
        p = tuple(face_boundary[idx[0]])
        # print(p)
        points.append(p)
        subdiv.insert(p)
    triangles = subdiv.getTriangleList()
    # return triangles
    delaunayTri = []

    pt = []

    count = 0

    for t in triangles:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if is_triangle_in(rect, [pt1, pt2, pt3]):
            count = count + 1
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri

# check the given rectangle contain all triangle points
def is_triangle_in(rect, triangle):
    return reduce(lambda x, y: x and y, [is_point_in(rect, p) for p in triangle])

def is_point_in(rect, point):
    return (rect[0] <= point[0] <= rect[2] and
            rect[1] <= point[1] <= rect[3])

def image_affine_warp(hull_from, hull_to, triangles_to_idxs, img_from, img_to):
    img_to = img_to.copy()

    for h_idx in triangles_to_idxs:
        tri1 = hull_from[np.array(h_idx)].reshape(3, 2)
        tri2 = hull_to[np.array(h_idx)].reshape(3, 2)
        affine_warp(tri1, tri2, img_from, img_to)

    return img_to

def affine_warp(tri1, tri2, img_from, img_to):


    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)

    # Offset points by left top corner of the
    # respective rectangles

    tri1Cropped = []
    tri2Cropped = []

    for i in range(0, 3):
        tri1Cropped.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
        tri2Cropped.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))

    # Apply warpImage to small rectangular patches
    img1Cropped = img_from[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

    # Apply the Affine Transform just found to the src image
    img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[2], r2[3]),
                                 None,
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

    # Apply mask to cropped region
    img2Cropped = img2Cropped * mask

    # Copy triangular region of the rectangular patch to the output image
    img_to[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img_to[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
    (1.0, 1.0, 1.0) - mask)

    img_to[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img_to[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Cropped

def seamless_cloning(hull_to, to_face, img_res):
    # Calculate Mask
    hull8U = [(p[0], p[1]) for p in hull_to]

    mask = np.zeros(to_face.shape, dtype=to_face.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull_to]))

    center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img_res), to_face, mask, center, cv2.NORMAL_CLONE)
    return output

def plot_triangulation(img, triangles):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    # Create a triangular patch for each triangle and add to the axes
    for t in triangles:
        triangle = patches.Polygon(t, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(triangle)

    plt.show()

def swap_faces(from_face_path, to_face_path, landmark_detector):
    from_face = cv2.cvtColor(cv2.imread(from_face_path), cv2.COLOR_BGR2RGB)
    to_face = cv2.cvtColor(cv2.imread(to_face_path), cv2.COLOR_BGR2RGB)

    from_face_orient = landmark_detector.get_face_horizontal_orientation(from_face)
    to_face_orient = landmark_detector.get_face_horizontal_orientation(to_face)

    if from_face_orient != to_face_orient:
        to_face = cv2.flip(to_face, 1)

    # get face boundary points and containing rectangles
    # for both faces
    face_boundary_from, rect_from = landmark_detector.get_contour(from_face)
    face_boundary_to, rect_to = landmark_detector.get_contour(to_face)

    hull_idx_to = cv2.convexHull(face_boundary_to, returnPoints=False)

    hull_from = np.array([face_boundary_from[hull_idx] for hull_idx in hull_idx_to])
    hull_to = np.array([face_boundary_to[hull_idx] for hull_idx in hull_idx_to])

    triangles_to_idxs = get_triangles_indexes(to_face, rect_to, hull_idx_to, face_boundary_to)

    img_res = image_affine_warp(hull_from,
                                      hull_to,
                                      triangles_to_idxs,
                                      from_face,
                                      to_face.copy())

    output = seamless_cloning(hull_to.reshape(hull_to.shape[0], 2), to_face, img_res)

    return output

def main(_):
    parser = argparse.ArgumentParser(description='Face Swap. Iteration over folders')

    parser.add_argument('-f', metavar='fromPath', dest='fromPath')
    parser.add_argument('-t', metavar='toPath', dest='toPath')
    parser.add_argument('-o', metavar='outputPath', dest='outputPath')
    parser.add_argument('-i', metavar='dataPath', dest='dataPath')

    args = parser.parse_args()
    fromPath = args.fromPath
    toPath = args.toPath
    outputPath = args.outputPath
    dataPath = args.dataPath

    landmark_detector = LandmarkDetector(join(dataPath, 'shape_predictor_68_face_landmarks.dat'))

    count = 0
    img_regexp = "*jpg"
    for from_face in glob.glob(join(fromPath, img_regexp)):
        for to_face in glob.glob(join(toPath, img_regexp)):
            try:
                from_filename = os.path.basename(from_face)
                to_filename = os.path.basename(to_face)
                if from_filename == to_filename:
                    print("Skipping {}, same name".format(from_filename))
                    continue
                print("## {} - From {} to {}".format(count, from_filename, to_filename))
                results = swap_faces(join(fromPath, from_filename),
                                    join(toPath, to_filename),
                                    landmark_detector)
                res_path = join(outputPath, '{}_{}.png'.format(from_filename.split('.')[0],
                                                                to_filename.split('.')[0]))
                cv2.imwrite(res_path, cv2.cvtColor(results, cv2.COLOR_RGB2BGR))
                count += 1
            except FaceSwapException as e:
                print(e)
            #except Exception as e:
            #    print(e.)

if __name__ == "__main__":
    main(sys.argv[1:])
