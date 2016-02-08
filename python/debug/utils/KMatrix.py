# To help build an intrinsic camera matrix
import numpy as np


class KMatrix:

    def __init__(self, focalLength=1, skew=0, principalPoint=(0, 0)):
        self.focalLength = focalLength
        self.skew = skew
        self.principalPoint = principalPoint
        self.matrix = np.array([[focalLength, skew, principalPoint[0]], [0, focalLength, principalPoint[1]], [0, 0, 1]])
