import logging
from ast import literal_eval
import cv2
import numpy as np

from insightface.app import FaceAnalysis

from face_utils import face_extract_utils as utils
from face_utils.Face import Face


class FaceDetector:
    def __init__(self, config, allowed_modules):
        self.config = config
        # init insightface model
        self.detector = FaceAnalysis(name=config['model_name'], root=config['model_dir'],
                                     allowed_modules=allowed_modules)
        self.detector.prepare(ctx_id=0, # ctx_id=-1 to use CPU
                              det_thresh=config['detection_threshold'], det_size=(640, 640))

    def detect_faces(self, img: np.array, min_res=0):
        faces = [Face(img, Face.Rectangle(int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3]), int(f.bbox[0])),
                      embedding=f.embedding) for f in self.detector.get(img)]

        # if specified, keep only faces where both width and height are above min_res
        if min_res:
            faces = [f for f in faces if all(map(lambda x: x > min_res, f.get_face_size()))]

        return faces

    def extract_face(self, face: Face):
        """Utility method which uses directly the current detector configuration for the generic extraction operation
        """
        # size is a tuple, so need to eval from string representation in config
        size = literal_eval(self.config['extract']['size'])
        border_expand = literal_eval(self.config['extract']['border_expand'])
        align = self.config['extract']['align']
        maintain_proportion = self.config['extract']['maintain_proportion']
        masked = self.config['extract']['masked']

        return self._extract_face(face, size, border_expand=border_expand, align=align,
                                  maintain_proportion=maintain_proportion, masked=masked)

    def _extract_face(self, face: Face, out_size=None, border_expand=(0., 0.), align=False,
                      maintain_proportion=False, masked=False):
        face_size = face.get_face_size()
        border_expand = (int(border_expand[0]*face_size[0]), int(border_expand[1]*face_size[1]))

        # if not specified otherwise, we want extracted face size to be exactly as input face size
        if not out_size:
            out_size = face_size

        face.landmarks = self.get_landmarks(face)
        if masked:
            mask = utils.get_face_mask(face, 'hull',
                                       erosion_size=literal_eval(self.config['extract'].get('dilation_kernel', 'None')),
                                       dilation_kernel=literal_eval(self.config['extract'].get('dilation_kernel',
                                                                                               'None')),
                                       blur_size=int(self.config['extract']['blur_size']))
            # black all pixels outside the mask
            face.img = cv2.bitwise_and(face.img, face.img, mask=mask[:, :, 1])

        # keep proportions of original image (rect) for extracted image, otherwise resize might stretch the content
        if maintain_proportion:
            border_delta = self._get_maintain_proportion_delta(face_size, out_size)
            border_expand = (border_expand[0] + int(border_delta[0]//2), border_expand[1] + int(border_delta[1]//2))

        if align:
            cut_face = utils.ffhq_align(face, output_size=out_size[0], boundary_resize_factor=border_expand)
            #cut_face, _ = utils.align_face(face, boundary_resize_factor=border_expand)
            #cut_face = utils._align_face(face, size=out_size)
        else:
            cut_face = cv2.resize(face.get_face_img(), out_size, interpolation=cv2.INTER_CUBIC)

        return cut_face

    def _get_maintain_proportion_delta(self, src_size, dest_size):
        """
        Return delta amount to maintain destination proportion given source size.
        Tuples order is (w, h)
        :param base_border:
        :param src_size:
        :param dest_size:
        :return:
        """
        dest_ratio = max(dest_size) / min(dest_size)
        delta_h = delta_w = 0
        w, h = src_size
        if w > h:
            delta_h = w * dest_ratio - h
        else:
            delta_w = h * dest_ratio - w
        return delta_w, delta_h
