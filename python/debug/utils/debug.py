import numpy as np
from CVFuncs import *
from KMatrix import *

def testFundamentalMat(F, points1, points2):
    print "F:"
    print F
    print "Rank:", np.linalg.matrix_rank(F)
    print "Testing constraint: (Should be zero):"
    for i in range(len(points1)):
        point1 = points1[i]
        point2 = points2[i]

        # Homogenize
        point1 = np.append(point1, [1])
        point2 = np.append(point2, [1])

        # Needs to be np array
        point1 = np.array([point1])
        point2 = np.array([point2]).transpose()

        result = np.dot(np.dot(point1, F), point2)
        print result[0][0]

def testEssentialMat(E, K, points1, points2):
    print "E:"
    print E
    print "Testing constraint: (Should be zero):"
    for i in range(len(points1)):
        point1 = points1[i]
        point2 = points2[i]

        # Homogenize
        point1 = np.append(point1, [1])
        point2 = np.append(point2, [1])

        # Needs to be np array
        point1 = np.array([point1])
        point2 = np.array([point2]).transpose()

        # Normalized coordinates
        point1 = np.dot(point1, np.linalg.inv(K.matrix).transpose())
        point2 = np.dot(np.linalg.inv(K.matrix), point2)

        result = np.dot(np.dot(point1, E), point2)
        print result[0][0]

def printRandTPossibilities(possibilities):
    for i, possibility in enumerate(possibilities):
        print "Possibility", i + 1
        print "R:"
        print possibility[0]
        print "T:"
        print possibility[1]
        print "---------------"

def drawRandTTransformation(r, t, points1, points2, filepath):
    writePoints = []
    P = composeRandT(r, t)

    for i in range(len(points1)):
        point1 = points1[i]
        point2 = points2[i]

        # Homogenize
        point1 = np.append(point1, [1])
        point2 = np.append(np.append(point2, [1]), [1])

        # Needs to be np array
        point1 = np.array([point1]).transpose()
        point2 = np.array([point2]).transpose()

        transformedPoint2 = np.dot(P, point2)
        writePoints.append(point1)
        writePoints.append(transformedPoint2)

    writePointsToFile(writePoints, filepath)

def writePointsToFile(points, filename, planar=False):
    points = points[:]
    # Add 3rd coord if necessary
    if planar:
        for i, point in enumerate(points):
            points[i] = [point[0], point[1], 1]

    # Write
    with open(filename, 'w') as f:
        def writeline(f,line):
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
