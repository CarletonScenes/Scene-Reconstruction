import numpy as np
from CVFuncs import *
from KMatrix import *
from line import *


def transformPointsToViewingCoordinates(points):
    # Aka flip them around the x and y axes
    outpoints = []
    for point in points:
        outpoints.append((-point[0], -point[1], point[2]))
    return outpoints


def drawRandTTransformation(pts1, pts2, k, r, t, filename):
    cameraOnePoints = []
    cameraTwoPoints = []

    # IMAGE ONE
    for point in pts1:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(k)
        normalized = inv_k.dot(homogenous)
        cameraOnePoints.append(normalized)

    # IMAGE TWO
    for point in pts2:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(k)
        normalized = inv_k.dot(homogenous)
        cameraTwoPoints.append(normalized)

    cameraTwoPoints = applyRandTToPoints(r, t, cameraTwoPoints)
    writePoints = cameraOnePoints + cameraTwoPoints

    writePointsToFile(writePoints, filename)


def drawProjections(pts1, pts2, k, r, t, file):
    # Draws a helpful ply file with origins, image planes and projected images

    origin1 = (0, 0, 0)
    origin2 = (t[0][0], t[1][0], t[2][0])

    writePoints = [origin1, origin2]

    cameraOnePoints = []
    cameraTwoPoints = []

    # IMAGE ONE
    for point in pts1:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(k)
        normalized = inv_k.dot(homogenous)
        imgpoint = normalized
        planepoint = (imgpoint[0] * 20, imgpoint[1] * 20, imgpoint[2] * 20)
        cameraOnePoints.append(imgpoint)
        cameraOnePoints.append(planepoint)

    # IMAGE TWO
    for point in pts2:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(k)
        normalized = inv_k.dot(homogenous)
        imgpoint = normalized
        planepoint = (imgpoint[0] * 20, imgpoint[1] * 20, imgpoint[2] * 20)
        cameraTwoPoints.append(imgpoint)
        cameraTwoPoints.append(planepoint)

    cameraTwoPoints = applyRandTToPoints(r, t, cameraTwoPoints)
    writePoints += cameraOnePoints + cameraTwoPoints
    writePoints = transformPointsToViewingCoordinates(writePoints)
    file.emitPoints(writePoints)


def drawLines(lines, filename):
    writePoints = []
    for line in lines:
        for i in range(1000):
            t = (10.0 / 1000) * i
            point = line.atT(t)
            writePoints.append(point)
    writePointsToFile(writePoints, filename)
