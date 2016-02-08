import os
import sys
import subprocess
import cv2
import math
import numpy as np
import utils.debug as Debug
import utils.CVFuncs as CVFuncs
from utils import KMatrix, Image
from triangulate import *


def main():
    points = triangulateTwoImages("images/chapel2.jpg", "images/chapel3.jpg")
    r = getArtificalR(-20)
    t = getArtificialTranslation(5)
    points2 = []
    for point in points:
        points2.append((r.dot(point) + t.transpose())[0])

    Debug.writePointsToFile(points, "points1.ply")
    Debug.writePointsToFile(points2, "points2.ply")


if __name__ == '__main__':
    main()
