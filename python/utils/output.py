import numpy as np
from CVFuncs import *
from KMatrix import *
from line import *


def printRandTPossibilities(possibilities):
    for i, possibility in enumerate(possibilities):
        print "Possibility", i + 1
        print "R:"
        print possibility[0]
        print "T:"
        print possibility[1]
        print "---------------"


def writePointsToFile(points, filename, planar=False):
    points = points[:]
    if planar:
        for i, point in enumerate(points):
            points[i] = [point[0], point[1], 1]

    with open(filename, 'w') as f:
        emitHeader(f)
        emitPoints(points, f)


def ptsToFileColor(pts, filename, image1, kps1, image2, kps2):
    with open(filename, 'w') as f:
        emitColorHeader(f, pts.shape[0])

        for row_num in range(pts.shape[0]):
            row = pts[row_num][0]
            line = "%f %f %f" % (row[0], row[1], row[2])
            color1 = getKeypointColor(image1, kps1, row_num)
            color2 = getKeypointColor(image2, kps2, row_num)
            red = (color1[0] + color2[0]) / 2
            green = (color1[1] + color2[1]) / 2
            blue = (color1[2] + color2[1]) / 2
            line2 = "%i %i %i" % (red, green, blue)
            writeline(f, line + " " + line2)


def getKeypointColor(img, kps, kpindex):
    red = []
    green = []
    blue = []

    xcoord = int(kps[kpindex].pt[0])
    ycoord = int(kps[kpindex].pt[1])

    for y in range(ycoord - 5, ycoord + 5):
        for x in range(xcoord - 5, xcoord + 5):
            if y >= 0 and y < len(img) and x >= 0 and x < len(img[0]):
                blue.append(img[y][x][0])
                green.append(img[y][x][1])
                red.append(img[y][x][2])

    return (sum(red) / len(red), sum(green) / len(green), sum(blue) / len(blue))
