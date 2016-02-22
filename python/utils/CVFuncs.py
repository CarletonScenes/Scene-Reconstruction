import cv2
import math
import numpy as np
from KMatrix import *
from line import *
from PIL import Image as PILImage


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
        F, mask = cv2.findFundamentalMat(np.array(points1), np.array(points2), method=cv2.FM_RANSAC)

        new_points1, new_points2, new_matches = [], [], []
        for i in range(len(matches)):
            if mask[i] == 1:
                new_points1.append(image1.kps[matches[i].queryIdx].pt)
                new_points2.append(image2.kps[matches[i].trainIdx].pt)
                new_matches.append(matches[i])

        return new_points1, new_points2, new_matches

    else:
        return points1, points2, matches


def findMatchesKnn(image1, image2, filter=True, ratio=True):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(image1.descs, image2.descs, k=2)
    points1 = []
    points2 = []
    new_matches = []
    for m, n in matches:
        if m.distance < .75 * n.distance and ratio:
            new_matches.append(m)
            points1.append(image1.kps[m.queryIdx].pt)
            points2.append(image2.kps[m.trainIdx].pt)
        elif not ratio:
            points1.append(image1.kps[m.queryIdx].pt)
            points2.append(image2.kps[m.trainIdx].pt)
            new_matches.append(m)

    if filter:
        F, mask = cv2.findFundamentalMat(np.array(points1), np.array(points2), method=cv2.FM_RANSAC)

        new_points1, new_points2, newer_matches = [], [], []
        for i in range(len(new_matches)):
            if mask[i] == 1:
                new_points1.append(image1.kps[new_matches[i].queryIdx].pt)
                new_points2.append(image2.kps[new_matches[i].trainIdx].pt)
                newer_matches.append(new_matches[i])

        return new_points1, new_points2, newer_matches
    else:
        return points1, points2, new_matches


def sortMatchesByDistance(matches):
    '''
    Takes in the matches from a BF matcher (list of DMatch objects)
    Returns matches sorted by distance between the bitvectors.
    '''
    return sorted(matches, key=lambda x: x.distance)


def filterMatchesYDist(points1, points2, matches):

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
    points, r, t, newMask = cv2.recoverPose(E, np.array(points1), np.array(points2), pp=K.principalPoint)
    r = np.linalg.inv(r)
    t = t * -1
    return points, r, t, newMask


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


def triangulateFromLinesIteratively(line1, line2):
    # Iteratively finds the two points that are closest together,
    # Then returns their midpoint

    minDist = 100000000
    minPoints = [(0, 0, 0), (0, 0, 0)]

    searchRange = 10.0  # maximum t
    iterations = 100
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


def normalize(arr):
    magnitude = (arr[0] ** 2 + arr[1] ** 2 + arr[2] ** 2) ** (1 / 2)
    arr[0] = arr[0] / magnitude
    arr[1] = arr[1] / magnitude
    arr[2] = arr[2] / magnitude
    return arr


def triangulateFromLinesDiscrete(lineObj1, lineObj2):
    # def triangulateFromLines(line1, line2):

    # Input Description:
    # P = [Px, Py, Pz], where P is a point on line 1,
    # R = [Rx, Ry, Rz], where R is a point on line 2,
    # D1 = [D1x, D1y, D1z], where D1 is a ray in dir. of line 1
    # D2 = [D2x, D2y, D2z], where D2 is a ray in dir. of line 2
    # line1 = [P, D1]
    # line2 = [R, D2]
    # In summary, line1 and line2 are the parametric equations of
    # the respective lines

    # Algorithm design:

    # You want to find a common perpendicular line to both lines in 3d
        # Write a parametric equation for each line: L1 = P1 + t (l1x, l1y, l1z); L2 = P2 + t (l2x, l2y, l2z)
        # Find the cross product between L1 and L2, the resulting vector is the common perpendicular
    # Now pick two random points on L1 and L2, calling them R1 and R2; find the difference between the two of them. Call this D
    # Find the vector of D along the common perpendicular, call this D_paralle
        # Find that as follows: Dot product D with the common perpendicular, and now multiply val of dot product by common perpendicular, call this the perp. dist.
    # Now place this vector, which is perpendicular to L1, at R1. R1 + perp. dist.. Now to get the equation of
    # the new line that needs to intersect L2: (R1 + per. dist.) + t (l1x, l1y, l1z)
    # Find the intersection between (R1 + per. dist.) + t (l1x, l1y, l1z) and P2 + t (l2x, l2y, l2z)

    # Find the cross product of the two lines

    line1 = [lineObj1.origin, lineObj1.direction]
    line2 = [lineObj2.origin, lineObj2.direction]
    # print "line1: ", line1
    # print "line2: ", line2

    DIMENSIONS = 3
    v1 = [0, 0, 0]
    v2 = [0, 0, 0]
    for i in range(DIMENSIONS):
        base1 = line1[0][i]
        offset1 = line1[1][i]
        base2 = line2[0][i]
        offset2 = line2[1][i]

        v1[i] = (base1 + offset1) - base1
        v2[i] = (base2 + offset2) - base2
    crossproduct = np.cross(v1, v2)
    # print "crossproduct: ", crossproduct.tolist()
    crossproduct = normalize(crossproduct)

    # Pick two random points, R1 and R2, one of line1 and line2, respectively. Find distance between them
    R1 = [0, 0, 0]
    R2 = [0, 0, 0]
    D = [0, 0, 0]
    for i in range(DIMENSIONS):
        R1[i] = line1[0][i] + 2 * line1[1][i]
        R2[i] = line2[0][i] + 2 * line2[1][i]
        D[i] = R1[i] - R2[i]

    # Dot the distance vector with the common perp.
    dotproduct = np.vdot(D, crossproduct)
    # print "dotproduct: ", dotproduct
    perpD = crossproduct * dotproduct
    perpD = perpD.tolist()
    # print "perpD", perpD

    # Construct the other line
    line3 = [[0, 0, 0], [0, 0, 0]]
    for i in range(DIMENSIONS):
        base3 = line2[0][i]
        # offset3 = dotproduct[i]
        offset3 = perpD[i]
        line3[0][i] = base3 + offset3
    line3[1] = line2[1]

    # Find the intersection between line 3 and line 2:
    # Outline:
    # Given the two lines:
    # line 2 = [X2, Y2, Z2] + s<X'_2, Y'_2, Z'_2>
    # line 3 = [X3, Y3, Z3] + t<X'_3, Y'_3, Z'_3>
    # line 2 = [X2 + s*X'_2, Y2 + s*Y'_2, Z2 + s*Z'_2]
    # line 3 = [X3 + t*X'_3, Y3 + t*Y'_3, Z3 + t*Z'_3]
    # looking for when each coordinate is the same, which means 3 lin. eq.s:
    # X2 + s*X'_2 = X3 + t*X'_3
    # Y2 + s*Y'_2 = Y3 + t*Y'_3
    # Z2 + s*Z'_2 = Z3 + t*Z'_3
    # Reorganizing the above equations to get a matrix
    # s * X'_2 - t * X'_3 = X3 - X2
    # s * Y'_2 - t * Y'_3 = Y3 - Y2
    # s * Z'_1 - t * Z'_3 = Z3 - Z2
    # Convert into the following matrix
    # [[X'_2, -X'_3],[Y'_2, - Y'_3], [Z'_2, Z'_3]][[s],[t]] = [[X3 - X2], [Y3 - Y2], [Z3 - Z2]]
    # x contains the solution [[s],[t]]

    a = np.array([[line1[1][0], -1 * line3[1][0]], [line1[1][1], -1 * line3[1][1]], [line1[1][2], -1 * line3[1][2]]])
    b = np.array([[line3[0][0] - line1[0][0]], [line3[0][1] - line1[0][1]], [line3[0][2] - line1[0][2]]])

    s, t = np.linalg.lstsq(a, b)[0]

    # find intersection point on line 1, using s
    inters2 = [0, 0, 0]
    s = s[0]

    for i in range(DIMENSIONS):
        inters2[i] = line1[0][i] + s * line1[1][i]

    # find the closest approach by taking inters2 and adding -0.5 * perp. distance line (0.5 because we want the half way point between
    # line 1 and line 2 FROM line 1).
    closest = [0, 0, 0]
    for i in range(DIMENSIONS):
        closest[i] = inters2[i] + -0.5 * perpD[i]

    return closest

def naiveTriangulate(pts1, pts2, k, r, t):
    # Transforms image planes by r and t, draws epipolar lines,
    # and uses those lines to triangulate points using iterative method

    lines1, lines2 = linesFromImagePoints(pts1, pts2, k, r, t)

    outpoints = []
    for line1, line2 in zip(lines1, lines2):
        outpoints.append(triangulateFromLinesIteratively(line1, line2))

    return outpoints

def discreteTriangulate(pts1, pts2, k, r, t):
    pass
    # Transforms image planes by r and t, draws epipolar lines,
    # and uses those lines to triangulate points using discrete method

def linesFromImagePoints(pts1, pts2, k, r, t):
    # Transforms image planes by r and t and returns epipolar lines of each feature

    origin1 = (0, 0, 0)
    origin2 = (t[0][0], t[1][0], t[2][0])

    # Image plane points (normalized and transformed)
    imgpoints1 = []
    imgpoints2 = []

    inverseK = np.linalg.inv(k)

    # IMAGE ONE
    for point in pts1:
        homogenous = np.append(np.array(point), [1]).transpose()
        normalized = inverseK.dot(homogenous)
        imgpoints1.append(normalized)

    # IMAGE TWO
    for point in pts2:
        homogenous = np.append(np.array(point), [1]).transpose()
        normalized = inverseK.dot(homogenous)
        imgpoints2.append(normalized)

    imgpoints2 = applyRandTToPoints(r, t, imgpoints2)

    lines1 = []
    lines2 = []

    # Draw  lines and triangulate
    for pt1, pt2 in zip(imgpoints1, imgpoints2):
        lines1.append(Line(origin1, pt1))
        lines2.append(Line(origin2, pt2))

    return lines1, lines2


