"""
Author: Dikshant Gupta
Time: 27.05.22 13:18
"""

import numpy as np
import skimage  # install scikit-image
import skimage.draw
import matplotlib.pyplot as plt
import skimage.transform


# PORTS
PORT = 1246
HOST = ''

# Setting the training parameters
y = .990  # Discount factor on the target Q-values
path = "HybridCheckpoints"  # The path to save our model to.
h_size = 256  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
var_end_size = 256
time_per_step = 1  # Length of each step used in gif creation
num_pedestrians = 1
num_angles = 5
num_actions = 3  # acceleration_type
image_input_size = 100 * 100 * 3
history_size = 2 * 128
observation_size = 6
load_model = True
enableTraining = False
learning_rate = 1e-4
decay = 0.99
momentum = 0.0
epsilon = 0.1
l2_decay = 0.0005

factor = 0.20
snippedSize = 100
multiplyer = factor * 5
halfSize = snippedSize // 2
car_length = 4.5
car_width = 2.0
adapted_length = car_length * multiplyer
adapted_width = car_width * multiplyer
# costMapOriginal = plt.imread('../LearningAssets/combinedmapSimple.png')
# costMapRescaled = skimage.transform.rescale(costMapOriginal, factor, multichannel=True, anti_aliasing=True,
#                                             mode='reflect')
# costMap = costMapRescaled

grid_cost = np.zeros((110, 310))
# Road Network
road_cost = 200.0
grid_cost[7:13, 13:] = road_cost
grid_cost[97:103, 13:] = road_cost
grid_cost[7:, 7:13] = road_cost
# Sidewalk Network
sidewalk_cost = 100.0
grid_cost[4:7, 4:] = sidewalk_cost
grid_cost[:, 4:7] = sidewalk_cost
grid_cost[13:16, 13:] = sidewalk_cost
grid_cost[94:97, 13:] = sidewalk_cost
grid_cost[103:106, 13:] = sidewalk_cost
grid_cost[13:16, 16:94] = sidewalk_cost
costMap = grid_cost / 255.0
costMap = np.repeat(costMap[:, :, np.newaxis], 3, axis=2)

points = np.empty([4, 2])
points[0, :] = (- adapted_length / 2.0, - adapted_width / 2.0)
points[1, :] = (+ adapted_length / 2.0, - adapted_width / 2.0)
points[2, :] = (+ adapted_length / 2.0, + adapted_width / 2.0)
points[3, :] = (- adapted_length / 2.0, + adapted_width / 2.0)


class SinCosLookupTable:

    def __init__(self):
        self.discretization = 0.005
        self.cosT = [None] * int(np.ceil((2 * np.pi) / self.discretization))
        self.sinT = [None] * int(np.ceil((2 * np.pi) / self.discretization))

        i = 0
        while i * self.discretization < 2 * np.pi:
            self.cosT[i] = np.cos(i * self.discretization)
            self.sinT[i] = np.sin(i * self.discretization)
            i += 1

    def sin(self, radAngle):
        return self.sinT[int(radAngle / self.discretization)]

    def cos(self, radAngle):
        return self.cosT[int(radAngle / self.discretization)]


def getCornerPositions(centerX, centerZ, theta):
    if theta < 0:
        theta = theta + 2 * np.pi
    # lookupTable = SinCosLookupTable()
    # tmp = points - (centerX, centerZ)
    # cos = lookupTable.cos(theta)
    # sin = lookupTable.sin(theta)
    #
    # tmp1 = np.empty(points.shape)
    # tmp1[:, 0] = tmp[:, 0] * cos - tmp[:, 1] * sin
    # tmp1[:, 1] = tmp[:, 0] * sin + tmp[:, 1] * cos
    #
    # tmp1 += (centerX, centerZ)

    tmp1 = np.empty(points.shape)
    # TOP RIGHT VERTEX:
    top_right_x = centerX + ((car_width / 2) * np.sin(theta)) + ((car_length / 2) * np.cos(theta))
    top_right_y = centerZ - ((car_width / 2) * np.cos(theta)) + ((car_length / 2) * np.sin(theta))
    tmp1[0, :] = top_right_x, top_right_y

    # TOP LEFT VERTEX:
    top_left_x = centerX - ((car_width / 2) * np.sin(theta)) + ((car_length / 2) * np.cos(theta))
    top_left_y = centerZ + ((car_width / 2) * np.cos(theta)) + ((car_length / 2) * np.sin(theta))
    tmp1[1, :] = top_left_x, top_left_y

    # BOTTOM LEFT VERTEX:
    bot_left_x = centerX - ((car_width / 2) * np.sin(theta)) - ((car_length / 2) * np.cos(theta))
    bot_left_y = centerZ + ((car_width / 2) * np.cos(theta)) - ((car_length / 2) * np.sin(theta))
    tmp1[2, :] = bot_left_x, bot_left_y

    # BOTTOM RIGHT VERTEX:
    bot_right_x = centerX + ((car_width / 2) * np.sin(theta)) - ((car_length / 2) * np.cos(theta))
    bot_right_y = centerZ - ((car_width / 2) * np.cos(theta)) - ((car_length / 2) * np.sin(theta))
    tmp1[3, :] = bot_right_x, bot_right_y

    return tmp1


def getCornerPositionsSimple(theta):
    # shape: num_samples x 4 x 2
    repeatedPoints = np.repeat(points[np.newaxis, ...], theta.shape[0], axis=0)

    cos = np.cos(theta)
    sin = np.sin(theta)

    cos = np.tile(cos,(4,1)).transpose()
    sin = np.tile(sin,(4,1)).transpose()

    tmp1 = np.empty(repeatedPoints.shape)

    tmp1[:, :, 0] = repeatedPoints[:,:, 0] * cos - repeatedPoints[:,:, 1] * sin + halfSize
    tmp1[:, :, 1] = repeatedPoints[:,:, 0] * sin + repeatedPoints[:,:, 1] * cos + halfSize

    return tmp1


def getObsParallel(image, allObsData):
    samples = allObsData.shape[0]
    allObsData *= multiplyer
    allObsData[:, 2:4] /= multiplyer

    result = np.empty([samples, 110, 310, 3])
    # image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    for iterator in range(samples):
        result[iterator, :, :, :] = image

    for iterator in range(samples):
        xx, yy, dd = allObsData[iterator, 0], allObsData[iterator, 1], np.deg2rad(allObsData[iterator, 2])
        cornerPositions = getCornerPositions(xx + 10, yy + 10, dd)
        rr, cc = skimage.draw.polygon(cornerPositions[:, 0], cornerPositions[:, 1], shape=[110, 310, 3])
        result[iterator, rr, cc, :] = (1.0, 0.0, 0.0)

        for i in range(num_pedestrians):
            xx, yy = allObsData[iterator, 4 + (2 * i)], allObsData[iterator, 4 + (2 * i) + 1]
            if xx != 0 or yy != 0:
                rr, cc = skimage.draw.ellipse(xx + 10, yy + 10, 1, 1, shape=[110, 310, 3])
                result[iterator, rr, cc, :] = (0, 0, 1.0)

    """
    plt.imshow(result[0,:,:])
    plt.show()
    """

    result.shape = (samples, 110 * 310 * 3)

    return result


def getObs(image, obsData):
    resizedX = obsData[0] * multiplyer
    resizedZ = obsData[1] * multiplyer

    x = int(round(max(resizedX - halfSize, 0)))
    z = int(round(max(resizedZ - halfSize, 0)))

    result = np.array(image[z:z + snippedSize, x:x + snippedSize])
    cornerPositions = getCornerPositions(0, 0, obsData[2]) + halfSize

    rr, cc = skimage.draw.polygon(cornerPositions[:, 1], cornerPositions[:, 0], shape=result.shape)
    result[rr, cc, :] = (1.0, 0, 0)

    """
    plt.imshow(result)
    plt.show()

    g2.draw(new Line2D.Float(f.get(0)[0] / scalingFactor, f.get(0)[1] / scalingFactor, f.get(1)[0] / scalingFactor, f.get(1)[1] / scalingFactor));
    g2.draw(new Line2D.Float(f.get(1)[0] / scalingFactor, f.get(1)[1] / scalingFactor, f.get(2)[0] / scalingFactor, f.get(2)[1] / scalingFactor));
    g2.draw(new Line2D.Float(f.get(2)[0] / scalingFactor, f.get(2)[1] / scalingFactor, f.get(3)[0] / scalingFactor, f.get(3)[1] / scalingFactor));
    g2.draw(new Line2D.Float(f.get(3)[0] / scalingFactor, f.get(3)[1] / scalingFactor, f.get(0)[0] / scalingFactor, f.get(0)[1] / scalingFactor));

    """

    for i in range(num_pedestrians):
        if obsData[4 + (2 * i)] != 0 or obsData[4 + (2 * i) + 1] != 0:
            rr, cc = skimage.draw.circle(obsData[4 + (2 * i) + 1] * multiplyer - resizedZ + halfSize,
                                         obsData[4 + (2 * i)] * multiplyer - resizedX + halfSize, 2, shape=result.shape)
            result[rr, cc, :] = (0, 0, 1.0)

    """
    global new_im
    if new_im:
        plt.imshow(result)
        plt.show()
        new_im = False
    """

    result.shape = (100 * 100 * 3,)

    return result
