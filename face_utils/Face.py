import copy


class Face:
    def __init__(self, img, rect=None):
        """
        Utility class for a face
        :param img: image containing the face
        :param rect: face rectangle
        """
        self.img = img
        self.rect = rect
        self.landmarks = None

    def get_face_center(self, absolute=True):
        """
        Return center coordinates of the face. Coordinates are rounded to closest int
        :param absolute: if True, center is absolute to whole image, otherwise is relative to face img
        :return: (x, y)
        """
        if self.rect:
            x, y = self.rect.get_center()
            if absolute:
                x += self.rect.left
                y += self.rect.top
            return x, y

    def get_face_img(self, boundary_resize_factor: tuple=None):
        """
        Return image bounded to target face (boundary is defined by rect attribute)
        :return:
        """
        if boundary_resize_factor:
            target_rect = self.rect.resize(boundary_resize_factor)
            self.rect = target_rect
        else:
            target_rect = self.rect
        top, right, bottom, left = target_rect.get_coords()
        face_img = self.img[top:bottom, left:right]
        return face_img

    def get_face_size(self):
        """
        Return size of face as (width, height)
        :return: (w, h)
        """
        #w, h = self.rect.get_size()
        # Consider image cause rect might exceed actual image boundaries
        face_img = self.get_face_img()
        w, h = face_img.shape[:2][::-1]
        return w, h

    def __copy__(self):
        face_copy = Face(self.img.copy(), copy.copy(self.rect))
        face_copy.landmarks = self.landmarks.copy()
        return face_copy


    class Rectangle:
        def __init__(self, top, right, bottom, left):
            """
            Utility class to hold information about face position/boundaries in an image
            :param top:
            :param right:
            :param bottom:
            :param left:
            """
            self.top = top
            self.right = right
            self.bottom = bottom
            self.left = left

        def get_coords(self):
            return self.top, self.right, self.bottom, self.left

        def get_center(self):
            x = (self.right - self.left)//2
            y = (self.bottom - self.top)//2
            return x, y

        def get_size(self):
            w = self.right - self.left
            h = self.bottom - self.top
            return w, h

        def resize(self, resize_factor: tuple):
            """
            Return new resized rectangle
            :return:
            """
            w, h = self.get_size()
            # if float given, consider as expansion ratio and obtain equivalent int values
            if type(resize_factor[0]) == float:
                resize_factor = (int(resize_factor[0] * w),
                                 int(resize_factor[1] * h))

            # divide by two as we add the border on each side
            resize_factor = (resize_factor[0] // 2, resize_factor[1] // 2)

            # compute new rectangle coords
            return Face.Rectangle(top=max(0, self.top - resize_factor[1]),
                                  right=self.right + resize_factor[0],
                                  left=max(0, self.left - resize_factor[0]),
                                  bottom=self.bottom + resize_factor[1])

        def __copy__(self):
            return Face.Rectangle(self.top, self.right, self.bottom, self.left)

        def __str__(self):
            return "top: {}, left: {}, bottom: {}, right: {}".format(
                self.top, self.left, self.bottom, self.right)

