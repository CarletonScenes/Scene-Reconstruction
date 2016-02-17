import cv2
import math
import debug
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
    return cv2.recoverPose(E, np.array(points1), np.array(points2), pp=K.principalPoint)


def crossProductForm(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def composeRandT(r, t):
    return np.append(r, t, 1)


def decomposeEssentialMat(E):
    W, U, VT = cv2.SVDecomp(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = np.dot(np.dot(U, W), VT)
    R2 = np.dot(np.dot(U, W.transpose()), VT)

    T1 = [U[0][2], U[1][2], U[2][2]]
    T2 = [-1 * T1[0], -1 * T1[1], -1 * T1[2]]

    return [(R1, T1), (R1, T2), (R2, T1), (R2, T2)]


def eucDist(pt1, pt2):
    return math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2) + ((pt1[2] - pt2[2]) ** 2))


def midpoint(pt1, pt2):
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2, (pt1[2] + pt2[2]) / 2)

def normalize(arr):
    magnitude = (arr[0] ** 2 + arr[1] ** 2 + arr[2] ** 2) ** (1/2)
    arr[0] = arr[0]/magnitude
    arr[1] = arr[1]/magnitude
    arr[2] = arr[2]/magnitude
    return arr

def triangulateFromLines(lineObj1, lineObj2):
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

    line1 = [lineObj1.origin, lineObj1.other]
    line2 = [lineObj2.origin, lineObj2.other]

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
    dotproduct = np.inner(D, crossproduct)
    perpD = crossproduct * dotproduct

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
    # [[X'_2, -X'_3],[Y'_2, - Y'_3], [Z'_2, Z'_3]][[s],[t]] = [[X3, -X2], [Y3, -Y2], [Z3, -Z2]]
    # x contains the solution [[s],[t]]
    a = np.array([[line2[1][0], -1 * line3[1][0]], [line2[1][1], -1 * line3[1][1]], [line2[1][2], -1 * line3[1][2]]])
    b = np.array([[line3[0][0] - line2[0][0]], [line3[0][1] - line2[0][1]], [line3[0][2] - line2[0][2]]])
    x = np.linalg.lstsq(a, b)
    # x = np.linalg.solve(a, b)
    print "x", x

    # find intersection point on line 2, using s, which is x[0]
    inters2 = [0, 0, 0]
    s = x[0][0]
    print "s", s
    for i in range(DIMENSIONS):
        inters2[i] = line2[0][i] + s[0] * line2[1][i]
        print "inters", inters2[i]

    # find the closest approach by taking inters2 and adding 0.5 * perp. distance line (0.5 because we want the half way point between
    # line 1 and line 2)
    closest = [0, 0, 0]
    for i in range(DIMENSIONS):
        closest[i] = inters2[i] + 0.5 * perpD[i]
        print "clo", closest[i]
    return closest

    # Iteratively finds the two points that are closest together,
    # Then returns their midpoint

    # minDist = 100000000
    # minPoints = [(0, 0, 0), (0, 0, 0)]

    # searchRange = 10.0  # maximum t
    # iterations = 31
    # for i in range(iterations):
    #     for j in range(iterations):
    #         t1 = (searchRange / iterations) * i
    #         t2 = (searchRange / iterations) * j
    #         pt1 = line1.atT(t1)
    #         pt2 = line2.atT(t2)
    #         distance = eucDist(pt1, pt2)
    #         if distance < minDist:
    #             minDist = distance
    #             minPoints = [pt1, pt2]

    # return midpoint(minPoints[0], minPoints[1])


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
        transformed_point = (r.dot(normalized) + t.transpose())[0]
        imgpoints2.append((transformed_point[0], transformed_point[1], transformed_point[2]))

    outpoints = []

    # Draw  lines and triangulate
    count = 0
    print len(imgpoints1)
    for pt1, pt2 in zip(imgpoints1, imgpoints2):
        line1 = Line(origin1, pt1)
        line2 = Line(origin2, pt2)
        print count
        count += 1
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