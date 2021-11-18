"""
Author: Dikshant Gupta
Time: 23.08.21 21:52
"""

import numpy as np


class PerceivedRisk:
    def __init__(self):
        # TODO: Check the value for wheel base
        # TODO: Check for the coordinates when calculating final risk(costmap is shifted by min_x, min_y)
        self.wheel_base = 0.1
        self.tla = 3.5  # Look ahead time[s]
        self.par1 = 0.0064  # Steepness of the parabola
        self.kexp1 = 0. * 0.07275  # inside circle
        self.kexp2 = 19. * 0.07275  # outside circle
        self.mcexp = 0.001  # m
        self.cexp = 0.5  # c : ego car width / 4

    def get_risk(self, path, player, costmap):
        return self.tla

    def pointwise_risk(self, x, y, vehicle_x, vehicle_y, velocity, delta, phi):
        """
        vehicle_x = x coordinate of vehicle (m)
        vehicle_y = y coordinate of vehicle (m)
        velocity = velocity of vehicle (m/s)
        delta = steering angle of the vehicle (degrees)
        phi = orientation of the vehicle (degrees)
        """
        # Convert angles to radians
        delta = delta / np.pi
        phi = phi / np.pi

        dla = self.tla * np.sqrt(velocity.x * velocity.x + velocity.y * velocity.y)
        if dla < 1:
            dla = 1
        r = self.wheel_base / np.tan(delta)

        if delta > 0:
            phil = phi + np.pi / 2
        else:
            phil = phi - np.pi / 2

        xc = r * np.cos(phil) + vehicle_x
        yc = r * np.sin(phil) + vehicle_y
        arc_len = self.get_arc_len(x, y, vehicle_x, vehicle_y, delta, xc, yc, r)
        a = self.get_a(arc_len, dla)

        mexp1 = self.mcexp + self.kexp1 * abs(delta)
        sigma1 = mexp1 * arc_len + self.cexp
        mexp2 = self.mcexp + self.kexp2 * abs(delta)
        sigma2 = mexp2 * arc_len + self.cexp

        dist_r = np.sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc))
        a_inside = (1 - np.sign(dist_r - r)) / 2
        a_outside = (1 + np.sign(dist_r - r)) / 2
        num = -(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r) ** 2
        den1 = 2 * sigma1 * sigma1
        z1 = a * a_inside * np.exp(num / den1)
        den2 = 2 * sigma2 * sigma2
        z2 = a * a_outside * np.exp(num / den2)

        z = z1 + z2
        return z

    @staticmethod
    def get_arc_len(x, y, xv, yv, delta, xc, yc, r):
        mag_u = np.sqrt((xv - xc) * (xv - xc) + (yv - yc) * (yv - yc))
        mag_v = np.sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc))
        dot_pro = (xv - xc) * (x - xc) + (yv - yc) * (y - yc)
        costheta = dot_pro / (mag_u * mag_v)
        theta_abs = abs(np.acos(costheta))
        sign_theta = np.sign((xv - xc) * (y - yc) - (x - xc) * (yv - yc))
        theta_pos_neg = np.sign(delta) * sign_theta * theta_abs
        theta = (2. * np.pi + theta_pos_neg) % (2. * np.pi)
        arc_len = r * theta

        return arc_len

    def get_a(self, arc_len, dla):
        a_par = self.par1 * np.power((arc_len - dla), 2)
        a_par_sign1 = np.sign(dla - arc_len)
        a_par_sign2 = np.sign(a_par)
        a_par_sign3 = np.sign(arc_len)
        a = a_par_sign1 * a_par_sign2 * a_par_sign3 * a_par

        return a

