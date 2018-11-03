import numpy as np


class Spirograph:
    def __init__(self, origin, R, r, d, angle, theta):
        self.origin = origin
        self.R = R
        self.r = r
        self.d = d
        self.angle = angle
        self.theta = theta

    def update(self):
        self.angle += self.theta

    def get_inner_circle_center(self):
        # position of circle on the outer circle circumference based on current angle
        pos_on_outer_circle = np.array([np.cos(self.angle), np.sin(self.angle), 0])
        # shift by origin and diff between radiuses
        center = self.origin + (self.R - self.r) * pos_on_outer_circle

        return center

    def get_hypotrochoid_angle(self):
        return ((self.R - self.r) / self.r) * self.angle

    def get_hypotrochoid_loc(self):
        # FIXME notice this is most likely calculated two times when rendering both circle and line
        inner_circle_center = self.get_inner_circle_center()

        hypotrochoid_angle = self.get_hypotrochoid_angle()

        loc_x = inner_circle_center[0] + self.d * np.cos(hypotrochoid_angle)
        loc_y = inner_circle_center[1] - self.d * np.sin(hypotrochoid_angle)
        return np.array([loc_x, loc_y, 0])
