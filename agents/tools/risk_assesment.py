"""
Author: Dikshant Gupta
Time: 23.08.21 21:52
"""

import numpy as np
import matplotlib.pyplot as plt


class PerceivedRisk:
    def __init__(self):
        # TODO: Check the value for wheel base
        self.wheel_base = 1.9887
        self.tla = 1  # Look ahead time[s]
        self.par1 = 0.0064  # Steepness of the parabola
        self.kexp1 = 0. * 0.07275  # inside circle
        self.kexp2 = 5. * 0.07275  # outside circle
        self.mcexp = 0.001  # m
        self.cexp = 0.5  # c : ego car width / 4

        self.grid_cost = np.ones((110, 310)) * 1000.0
        self.minx = -10
        self.miny = -10
        # Road Network
        self.grid_cost[7:13, 13:] = 1.0
        self.grid_cost[97:103, 13:] = 1.0
        self.grid_cost[7:, 7:13] = 1.0
        self.grid_cost = self.grid_cost / 10000.0

    def get_risk(self, path, player, steering_angle):
        risk = 0
        drf = np.zeros(self.grid_cost.shape)
        for i in range(self.grid_cost.shape[0]):
            for j in range(self.grid_cost.shape[1]):
                risk_field = self.pointwise_risk(i, j, player.get_location().x, player.get_location().y,
                                                 steering_angle, player.get_rotation().yaw)
                risk += self.grid_cost[i, j] * risk_field
                drf[i, j] = risk_field
        return risk, drf

    def get_risk_dummy(self, path, player, steering_angle):
        risk = 0
        drf = np.zeros(self.grid_cost.shape)
        for i in range(self.grid_cost.shape[0]):
            for j in range(self.grid_cost.shape[1]):
                risk_field = self.pointwise_risk(i, j, player[0] - self.minx, player[1] - self.miny,
                                                 player[2], 0.5 * steering_angle, player[3])
                risk += self.grid_cost[i, j] * risk_field
                drf[i, j] = risk_field
        return risk, drf

    def get_phi(self, phi):
        return abs(phi % (2 * np.pi))

    def pointwise_risk(self, x, y, vehicle_x, vehicle_y, velocity, delta, phi):
        """
        vehicle_x = x coordinate of vehicle (m)
        vehicle_y = y coordinate of vehicle (m)
        velocity = velocity of vehicle (m/s)
        delta = steering angle of the vehicle (degrees)
        phi = orientation of the vehicle (degrees)
        """
        # Convert angles to radians
        phi = np.pi * phi / 180
        phi = self.get_phi(phi)
        delta = np.pi * delta / 180
        if abs(delta) < 1e-8:
            delta = 1e-8

        # dla = self.tla * np.sqrt(velocity.x * velocity.x + velocity.y * velocity.y)
        dla = self.tla * velocity
        if dla < 1:
            dla = 1
        r = abs(self.wheel_base / np.tan(delta))

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
        a_inside = (1 - np.sign(dist_r - r)) / 2.0
        a_outside = (1 + np.sign(dist_r - r)) / 2.0
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
        costheta = dot_pro / (mag_u * mag_v + 1e-6)
        theta_abs = abs(np.arccos(costheta))
        sign_theta = np.sign((xv - xc) * (y - yc) - (x - xc) * (yv - yc))
        theta_pos_neg = np.sign(delta) * sign_theta * theta_abs
        theta = (2. * np.pi + theta_pos_neg) % (2. * np.pi)
        arc_len = r * theta

        return arc_len

    def get_a(self, arc_len, dla):
        a_par = self.par1 * np.power((arc_len - dla), 2)
        a_par_sign1 = (np.sign(dla - arc_len) + 1) / 2.0
        a_par_sign2 = (np.sign(a_par) + 1) / 2.0
        a_par_sign3 = (np.sign(arc_len) + 1) / 2.0
        a = a_par_sign1 * a_par_sign2 * a_par_sign3 * a_par

        return a


if __name__ == "__main__":
    pr = PerceivedRisk()
    risk, drf = pr.get_risk_dummy([], [2.0, 240.0, 13, -90], 0)
    print(risk, drf)
    plt.imshow(drf.T)
    plt.show()
