import cv2
import math
import debug
import numpy as np
from KMatrix import *
from line import *
from PIL import Image as PILImage


def invertY(points):
    outpoints = []
    for point in points:
        outpoints.append((point[0], -point[1], point[2]))
    return outpoints


def normalizeCoordinates(points, K):
    normPoints = []
    for point in points:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(K.matrix)
        normalized = inv_k.dot(homogenous)

        normPoints.append(normalized)
    return normPoints


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


def sortMatchesByDistance(matches):
    '''
    Takes in the matches from a BF matcher (list of DMatch objects)
    Returns matches sorted by distance between the bitvectors.
    '''
    return sorted(matches, key=lambda x: x.distance)


def filterMatches(points1, points2, matches):

    # Determine mean and stdev of point y values
    diff_ys = []
    for point1, point2 in zip(points1, points2):
        diff_ys.append(point2[1] - point1[1])

    stdev = np.std(diff_ys)
    mean = sum(diff_ys) / len(diff_ys)

    new_matches = []
    new_points1 = []
    new_points2 = []

    # Filter matches
    for i, match in enumerate(matches):
        if abs(mean - (points2[i][1] - points1[i][1])) <= stdev / 2:
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


def getEssentialMat(points1, points2, K):

    # Build the Y matrix using the first 8 matches (assuming they're the best)
    yMat = np.empty([9, 9])
    kInv = np.linalg.inv(K)
    for i in range(9):
        print "points1", points1[i]
        pts1 = np.append(points1[i], 1)
        pts2 = np.append(points2[i], 1)
        p1 = kInv.dot(pts1)
        p2 = kInv.dot(pts2)
        # p1 = pts1
        # p2 = pts2
        tempColumn = np.array([p2[0] * p1[0], p2[0] * p1[1], p2[0], p2[1] * p1[0], p2[1] * p1[1], p2[1],
                               p1[0], p1[1], 1])
        yMat[i] = tempColumn

    # Solve the homogeneous linear system of equations
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    print "yMat", yMat
    print "b", b
    eVector = np.linalg.solve(yMat, b)
    # eVector = np.linalg.lstsq(yMat, b)
    print "eVector", eVector
    eEstMat = np.array([[eVector[0], eVector[1], eVector[2]],
                        [eVector[3], eVector[4], eVector[5]],
                        [eVector[6], eVector[7], eVector[8]]
                        ])

    # Extract E from E estimate
    W, U, VT = cv2.SVDecomp(eEstMat)
    S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    print "W", W
    print "S", S
    print "VT", VT
    E = W.dot(U.dot(VT))

    return E


def EFromF(F, K=KMatrix()):
    return np.dot(np.dot(K.matrix.transpose(), F), K.matrix)


def recoverPose(E, points1, points2, K=KMatrix()):
    decomposeEssentialMat(E)
    return cv2.recoverPose(E, np.array(points1), np.array(points2), pp=K.principalPoint)


def crossProductForm(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def composeRandT(r, t):
    return np.append(r, t, 1)


def composeRotations(r1, r2):
    return r1.dot(r2)


def composeTranslations(t1, t2):
    return t1 + t2


def decomposeEssentialMat(E):
    W, U, VT = cv2.SVDecomp(E)
    # print "W", W
    # print "U", U
    # print "VT", VT
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = np.dot(np.dot(U, W), VT)
    R2 = np.dot(np.dot(U, W.transpose()), VT)

    T1 = [U[0][2], U[1][2], U[2][2]]
    T2 = [-1 * T1[0], -1 * T1[1], -1 * T1[2]]

    return [(R1, T1), (R1, T2), (R2, T1), (R2, T2)]


def applyRandTToPoints(r, t, points):
    newPoints = []
    for point in points:
        transformed = (r.dot(point) + t.transpose())[0]
        # Convert from np array back to tuple
        newPoints.append((transformed[0], transformed[1], transformed[2]))
    return newPoints


def eucDist(pt1, pt2):
    return math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2) + ((pt1[2] - pt2[2]) ** 2))


def midpoint(pt1, pt2):
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2, (pt1[2] + pt2[2]) / 2)


def triangulateFromLines(line1, line2):
    # Iteratively finds the two points that are closest together,
    # Then returns their midpoint

    minDist = 100000000
    minPoints = [(0, 0, 0), (0, 0, 0)]

    searchRange = 20.0  # maximum t
    iterations = 31
    for i in range(iterations):
        for j in range(iterations):
            t1 = (searchRange / iterations) * i
            t2 = (searchRange / iterations) * j
            pt1 = line1.atT(t1)
            pt2 = line2.atT(t2)
            distance = eucDist(pt1, pt2)
            if distance < minDist:
                minDist = distance
                minPoints = [pt1, pt2]

    return midpoint(minPoints[0], minPoints[1])


def naiveTriangulate(pts1, pts2, k, r, t):
    # Transforms image planes by r and t, draws epipolar lines,
    # and uses those lines to triangulate points

    origin1 = (0, 0, 0)
    origin2 = (t[0][0], t[1][0], t[2][0])

    # Image plane points (normalized and transformed)
    imgpoints1 = []
    imgpoints2 = []

    # IMAGE ONE
    for point in pts1:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(k)
        normalized = inv_k.dot(homogenous)
        imgpoints1.append(normalized)

    # IMAGE TWO
    for point in pts2:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(k)
        normalized = inv_k.dot(homogenous)
        imgpoints2.append(normalized)

    imgpoints2 = applyRandTToPoints(r, t, imgpoints2)

    outpoints = []

    # Draw  lines and triangulate
    for pt1, pt2 in zip(imgpoints1, imgpoints2):
        line1 = Line(origin1, pt1)
        line2 = Line(origin2, pt2)
        outpoints.append(triangulateFromLines(line1, line2))

    return outpoints

# reimplement: https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/calib3d/src/triangulate.cpp#L54


def triangulatePoints(proj1mat, proj2mat, kps1, kps2):
    assert len(kps1) == len(kps2)

    matrA = np.zeros((4, 4))
    # matrU = np.zeros((4,4))
    # matrW = np.zeros((4,1))
    # matrV = np.zeros((4,4))

    outputPoints = np.zeros((len(kps1), 4))

    kps = [kps1, kps2]
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

        outputPoints[i][0] = matrV[3][0]  # X
        outputPoints[i][1] = matrV[3][1]  # Y
        outputPoints[i][2] = matrV[3][2]  # Z
        outputPoints[i][3] = matrV[3][3]  # W

    outputPoints = cv2.convertPointsFromHomogeneous(outputPoints)

    points = []
    for point in outputPoints:
        points.append((point[0][0], point[0][1], point[0][2]))
    return points
