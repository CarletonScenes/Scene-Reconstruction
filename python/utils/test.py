import numpy as np
from CVFuncs import *
from KMatrix import *
from line import *


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
