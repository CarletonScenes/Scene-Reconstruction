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
    # Add 3rd coord if necessary
    if planar:
        for i, point in enumerate(points):
            points[i] = [point[0], point[1], 1]

    # Write
    with open(filename, 'w') as f:
        def writeline(f, line):
            return f.write("{}\n".format(line))

        writeline(f, "ply")
        writeline(f, "format ascii 1.0")
        writeline(f, "element vertex {}".format(len(points)))
        writeline(f, "property float x")
        writeline(f, "property float y")
        writeline(f, "property float z")
        writeline(f, "end_header")

        for point in points:
            writeline(f, "%f %f %f" % (point[0], point[1], point[2]))
