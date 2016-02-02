import cv2
import numpy as np
from KMatrix import *
from PIL import Image as PILImage

def findMatches(image1, image2, filter=False):

    # Use openCV's brute force point matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(image1.descs, image2.descs)

    points1 = []
    points2 = []

    for match in matches:
        points1.append(image1.kps[match.queryIdx].pt)
        points2.append(image2.kps[match.trainIdx].pt)

    if filter:
        return filterMatches(points1, points2, matches)
    else:
        return points1, points2, matches

def filterMatches(points1, points2, matches):

    # Determine mean and stdev of point y values
    diff_ys = []
    for point1, point2 in zip(points1, points2):
        diff_ys.append(point2[1] - point1[1])

    stdev = np.std(diff_ys)
    mean = sum(diff_ys)/len(diff_ys)

    new_matches = []
    new_points1 = []
    new_points2 = []

    # Filter matches
    for i, match in enumerate(matches):
        if abs(mean - (points2[i][1] - points1[i][1])) <= stdev/2:
            new_matches.append(matches[i])
            new_points1.append(points1[i]) 
            new_points2.append(points2[i])
        
    return new_points1, new_points2, new_matches

def drawMatches(image1, image2, matches, filename):
    matchImage = cv2.drawMatches(image1.img, image1.kps, image2.img, image2.kps, matches, image1.img, flags=2)
    img = PILImage.fromarray(matchImage, 'RGB')
    img.save(filename)

def findFundamentalMat(points1, points2):
    return cv2.findFundamentalMat(np.array(points1), np.array(points2))

def findEssentialMat(points1, points2, K=KMatrix()):
    return cv2.findEssentialMat(np.array(points1), np.array(points2), focal=K.focalLength, pp=K.principalPoint)

def getEssentialMat():
    return False

def EFromF(F, K=KMatrix()):
    return np.dot(np.dot(K.matrix.transpose(), F), K.matrix)

def recoverPose(E, points1, points2, K=KMatrix()):
    return cv2.recoverPose(E, np.array(points1), np.array(points2), pp=K.principalPoint)

def crossProductForm(v):
    return np.array([
        [0, -v[2], v[1]], 
        [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]
    ])

def composeRandT(r, t):
    return np.append(r,t,1)

def decomposeEssentialMat(E):
    W, U, VT = cv2.SVDecomp(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = np.dot(np.dot(U, W), VT)
    R2 = np.dot(np.dot(U, W.transpose()), VT)

    T1 = [U[0][2], U[1][2], U[2][2]]
    T2 = [-1*T1[0], -1*T1[1], -1*T1[2]]

    return [(R1, T1), (R1, T2), (R2, T1), (R2, T2)]

# reimplement: https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/calib3d/src/triangulate.cpp#L54
def triangulatePoints(proj1mat, proj2mat, kps1, kps2):
    assert len(kps1) == len(kps2)

    matrA = np.zeros((4,4))
    # matrU = np.zeros((4,4))
    # matrW = np.zeros((4,1))
    # matrV = np.zeros((4,4))

    outputPoints = np.zeros((len(kps1),4))

    kps = [kps1,kps2]
    projMatrs = [proj1mat, proj2mat]

    for i in range(len(kps1)):
        
        # Row 1 (x1 * P1 3T - P1 1T)
        matrA[0][0] = kps1[i][0] * proj1mat[2][0] - proj1mat[0][0]
        matrA[0][1] = kps1[i][0] * proj1mat[2][1] - proj1mat[0][1]
        matrA[0][2] = kps1[i][0] * proj1mat[2][2] - proj1mat[0][2]
        matrA[0][3] = kps1[i][0] * proj1mat[2][3] - proj1mat[0][3]

        # Row 2 (y1 * P1 3T - P1 2T)
        matrA[1][0] = kps1[i][1] * proj1mat[2][0] - proj1mat[1][0]
        matrA[1][1] = kps1[i][1] * proj1mat[2][1] - proj1mat[1][1]
        matrA[1][2] = kps1[i][1] * proj1mat[2][2] - proj1mat[1][2]
        matrA[1][3] = kps1[i][1] * proj1mat[2][3] - proj1mat[1][3]

        # Row 3 (x2 * P2 3T - P1 1T)
        matrA[2][0] = kps2[i][0] * proj2mat[2][0] - proj2mat[0][0]
        matrA[2][1] = kps2[i][0] * proj2mat[2][1] - proj2mat[0][1]
        matrA[2][2] = kps2[i][0] * proj2mat[2][2] - proj2mat[0][2]
        matrA[2][3] = kps2[i][0] * proj2mat[2][3] - proj2mat[0][3]

        # Row 3 (y2 * P2 3T - P1 2T)
        matrA[3][0] = kps2[i][1] * proj2mat[2][0] - proj2mat[1][0]
        matrA[3][1] = kps2[i][1] * proj2mat[2][1] - proj2mat[1][1]
        matrA[3][2] = kps2[i][1] * proj2mat[2][2] - proj2mat[1][2]
        matrA[3][3] = kps2[i][1] * proj2mat[2][3] - proj2mat[1][3]

        U, s, matrV = np.linalg.svd(matrA, full_matrices=True)

        outputPoints[i][0] = matrV[3][0] # X
        outputPoints[i][1] = matrV[3][1] # Y
        outputPoints[i][2] = matrV[3][2] # Z
        outputPoints[i][3] = matrV[3][3] # W

    outputPoints = cv2.convertPointsFromHomogeneous(outputPoints)

    points = []
    for point in outputPoints:
        points.append((point[0][0], point[0][1], point[0][2]))
    return points
