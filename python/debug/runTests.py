import os
import sys
import subprocess
import cv2
import math
import numpy as np
import utils.debug as Debug
import utils.CVFuncs as CVFuncs
from utils import KMatrix, Image, line
from triangulate import *


def main():

    l1 = line.Line([0, 1, 0], [1, 0, 0])
    l2 = line.Line([1, 0, 0], [1, 0, 1])

    # print "BLAR", CVFuncs.triangulateFromLines([[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 1]])
    print "Triangulation Attempt", CVFuncs.triangulateFromLines(l1, l2)
    exit(0)


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
