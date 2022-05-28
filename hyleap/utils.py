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
num_pedestrians = 4
num_angles = 5
num_actions = 3  # acceleration_type
image_input_size = 100 * 100 * 3
history_size = 2 * 128
observation_size = 12
load_model = True
enableTraining = False
learning_rate = 0.00015
decay = 0.99
momentum = 0.0
epsilon = 0.1

factor = 0.20
snippedSize = 100
multiplyer = factor * 5
halfSize = snippedSize // 2
car_length = 4.25
car_width = 1.7
adapted_length = car_length * multiplyer
adapted_width = car_width * multiplyer
# costMapOriginal = plt.imread('../LearningAssets/combinedmapSimple.png')
# costMapRescaled = skimage.transform.rescale(costMapOriginal, factor, multichannel=True, anti_aliasing=True,
#                                             mode='reflect')
# costMap = costMapRescaled

grid_cost = np.ones((110, 310)) * 1000.0
# Road Network
grid_cost[7:13, 13:] = 1.0
grid_cost[97:103, 13:] = 1.0
grid_cost[7:, 7:13] = 1.0
# Sidewalk Network
grid_cost[4:7, 4:] = 50.0
grid_cost[:, 4:7] = 50.0
grid_cost[13:16, 13:] = 50.0
grid_cost[94:97, 13:] = 50.0
grid_cost[103:106, 13:] = 50.0
grid_cost[13:16, 16:94] = 50.0
costMap = grid_cost

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
    lookupTable = SinCosLookupTable()
    tmp = points - (centerX, centerZ)
    cos = lookupTable.cos(theta)
    sin = lookupTable.sin(theta)

    tmp1 = np.empty(points.shape)
    tmp1[:, 0] = tmp[:, 0] * cos - tmp[:, 1] * sin
    tmp1[:, 1] = tmp[:, 0] * sin + tmp[:, 1] * cos

    tmp1 += (centerX, centerZ)

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

    coordinate = (allObsData[:,0:2] - halfSize).round().astype(int)
    coordinate[coordinate < 0] = 0

    coordinateMax = coordinate + snippedSize

    result = np.empty([samples, 100, 100, 3])

    for iterator in range(samples):
        result[iterator, :, :] = image[coordinate[iterator, 1] : coordinateMax[iterator, 1], coordinate[iterator, 0]: coordinateMax[iterator, 0]]

    cornerPositions = getCornerPositionsSimple(allObsData[:, 2])

    for iterator in range(samples):
        rr, cc = skimage.draw.polygon(cornerPositions[iterator, :, 1], cornerPositions[iterator, :, 0], shape=[100, 100, 3])
        result[iterator, rr, cc, :] = (1.0, 0, 0)

        for i in range(num_pedestrians):
            if allObsData[iterator, 4 + (2 * i)] != 0 or allObsData[iterator, 4 + (2 * i) + 1] != 0:
                rr, cc = skimage.draw.circle(allObsData[iterator, 4 + (2 * i) + 1] - allObsData[iterator, 1] + halfSize,
                                             allObsData[iterator, 4 + (2 * i)] - allObsData[iterator, 0] + halfSize, 2, shape=[100, 100, 3])
                result[iterator, rr, cc, :] = (0, 0, 1.0)

    """
    plt.imshow(result[0,:,:])
    plt.show()
    """

    result.shape = (samples, 100 * 100 * 3)

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
