import cv2
import math
import numpy as np
from KMatrix import *
from line import *
from PIL import Image as PILImage


def normalizeCoordinates(points, K):
    '''
    Takes list of nonhomogenous image coordinates, camera matrix K
    Returns normalized camera coordinates
    '''
    normPoints = []
    for point in points:
        homogenous = np.append(np.array(point), [1]).transpose()
        inv_k = np.linalg.inv(K.matrix)
        normalized = inv_k.dot(homogenous)

        normPoints.append(normalized)
    return normPoints


def findMatches(image1, image2, filter=False):
    '''
    Takes two images
    Returns list of matches between the keypoints in poth images
    '''
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
    '''
    Takes images
    Returns unsorted matches between keypoints in both images using the kNN match
    '''
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
    '''
    Takes in points from image1, image2, and matches
    Does a naive match filtering using the difference in y coordinates as a check
    '''
    diff_ys = []
    for point1, point2 in zip(points1, points2):
        diff_ys.append(point2[1] - point1[1])

    stdev = np.std(diff_ys)
    mean = sum(diff_ys) / len(diff_ys)

    new_matches = []
    new_points1 = []
    new_points2 = []

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
    '''
    Takes points1, points2, unnormalized, and camera matrix K
    Runs the 8 point algorithm
    Returns Essential Matrix, E, between the images of the two points
    '''
    yMat = np.empty([9, 9])
    kInv = np.linalg.inv(K)
    for i in range(9):
        pts1 = np.append(points1[i], 1)
        pts2 = np.append(points2[i], 1)
        p1 = kInv.dot(pts1)
        p2 = kInv.dot(pts2)
        tempColumn = np.array([p2[0] * p1[0], p2[0] * p1[1], p2[0], p2[1] * p1[0], p2[1] * p1[1], p2[1],
                               p1[0], p1[1], 1])
        yMat[i] = tempColumn

    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    eVector = np.linalg.solve(yMat, b)
    eEstMat = np.array([[eVector[0], eVector[1], eVector[2]],
                        [eVector[3], eVector[4], eVector[5]],
                        [eVector[6], eVector[7], eVector[8]]
                        ])

    W, U, VT = cv2.SVDecomp(eEstMat)
    S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
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
        newPoints.append((transformed[0], transformed[1], transformed[2]))
    return newPoints


def eucDist(pt1, pt2):
    return math.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2) + ((pt1[2] - pt2[2]) ** 2))


def midpoint(pt1, pt2):
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2, (pt1[2] + pt2[2]) / 2)


def triangulateFromLinesIteratively(line1, line2):
    '''
    Takes two lines, presumably skewed
    Iteratively finds the closest point to both lines
    Returns the closest point
    '''
    minDist = 100000000
    minPoints = [(0, 0, 0), (0, 0, 0)]

    searchRange = 10.0
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

    # Find the cross product of the two lines
    line1 = [lineObj1.origin, lineObj1.direction]
    line2 = [lineObj2.origin, lineObj2.direction]

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
    dotproduct = np.vdot(D, crossproduct)
    perpD = crossproduct * dotproduct
    perpD = perpD.tolist()

    # Construct the other line
    line3 = [[0, 0, 0], [0, 0, 0]]
    for i in range(DIMENSIONS):
        base3 = line2[0][i]
        offset3 = perpD[i]
        line3[0][i] = base3 + offset3
    line3[1] = line2[1]

    a = np.array([[line1[1][0], -1 * line3[1][0]], [line1[1][1], -1 * line3[1][1]], [line1[1][2], -1 * line3[1][2]]])
    b = np.array([[line3[0][0] - line1[0][0]], [line3[0][1] - line1[0][1]], [line3[0][2] - line1[0][2]]])
    s, t = np.linalg.lstsq(a, b)[0]

    inters2 = [0, 0, 0]
    s = s[0]

    for i in range(DIMENSIONS):
        inters2[i] = line1[0][i] + s * line1[1][i]

    closest = [0, 0, 0]
    for i in range(DIMENSIONS):
        closest[i] = inters2[i] + -0.5 * perpD[i]

    return closest


def naiveTriangulate(pts1, pts2, k, r, t):
    '''
    Transforms image planes by r and t, draws epipolar lines,
    and uses those lines to triangulate points
    '''

    lines1, lines2 = linesFromImagePoints(pts1, pts2, k, r, t)

    outpoints = []
    for line1, line2 in zip(lines1, lines2):
        outpoints.append(triangulateFromLinesIteratively(line1, line2))

    return outpoints


def discreteTriangulate(pts1, pts2, k, r, t):
    '''
    Transforms image planes by r and t, draws epipolar lines,
    and uses those lines to triangulate points using discrete method
    '''
    lines1, lines2 = linesFromImagePoints(pts1, pts2, k, r, t)

    outpoints = []
    for line1, line2 in zip(lines1, lines2):
        outpoints.append(triangulateFromLinesDiscrete(line1, line2))

    return outpoints

def discreteTriangulateWithTwoRT(pts1, pts2, k, r1, t1, r2, t2):
    '''
    Transforms image planes by r and t, draws epipolar lines,
    and uses those lines to triangulate points using discrete method
    First image plane transformed by r1 t1, second by r2, t2
    '''
    lines1, lines2 = linesFromImagePointsWithTwoRT(pts1, pts2, k, r1, t1, r2, t2)

    outpoints = []
    for line1, line2 in zip(lines1, lines2):
        outpoints.append(triangulateFromLinesDiscrete(line1, line2))

    return outpoints


def linesFromImagePoints(pts1, pts2, k, r, t):
    # Transforms image planes by r and t and returns epipolar lines of each feature

    origin1 = (0, 0, 0)
    origin2 = (t[0][0], t[1][0], t[2][0])

    imgpoints1 = []
    imgpoints2 = []

    inverseK = np.linalg.inv(k)

    # IMAGE ONE
    for point in pts1:
        homogenous = np.append(np.array(point), [1]).transpose()
        normalized = inverseK.dot(homogenous)
        imgpoints1.append(normalized)

    for point in pts2:
        homogenous = np.append(np.array(point), [1]).transpose()
        normalized = inverseK.dot(homogenous)
        imgpoints2.append(normalized)

    imgpoints2 = applyRandTToPoints(r, t, imgpoints2)

    lines1 = []
    lines2 = []

    for pt1, pt2 in zip(imgpoints1, imgpoints2):
        lines1.append(Line(origin1, pt1))
        lines2.append(Line(origin2, pt2))

    return lines1, lines2

def linesFromImagePointsWithTwoRT(pts1, pts2, k, r1, t1, r2, t2):
    # Transforms image planes by r and t and returns epipolar lines of each feature
    # first image plane transformed by r1 t1, the second by r2 t2

    origin1 = (t1[0][0], t1[1][0], t1[2][0])
    origin2 = (t2[0][0], t2[1][0], t2[2][0])

    imgpoints1 = []
    imgpoints2 = []

    inverseK = np.linalg.inv(k)

    # IMAGE ONE
    for point in pts1:
        homogenous = np.append(np.array(point), [1]).transpose()
        normalized = inverseK.dot(homogenous)
        imgpoints1.append(normalized)

    for point in pts2:
        homogenous = np.append(np.array(point), [1]).transpose()
        normalized = inverseK.dot(homogenous)
        imgpoints2.append(normalized)

    imgpoints1 = applyRandTToPoints(r1, t1, imgpoints1)
    imgpoints2 = applyRandTToPoints(r2, t2, imgpoints2)

    lines1 = []
    lines2 = []

    for pt1, pt2 in zip(imgpoints1, imgpoints2):
        lines1.append(Line(origin1, pt1))
        lines2.append(Line(origin2, pt2))

    return lines1, lines2


def cvTriangulate(pts1, pts2, k, r, t):
    proj1 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
    proj2 = np.append(r, t, 1)
    pts1 = np.array(pts1).transpose()
    pts2 = np.array(pts2).transpose()

    homogeneous_4d_coords = cv2.triangulatePoints(proj1, proj2, pts1, pts2)
    # return triangulatePoints(proj1, proj2, pts1, pts2)

    threeD_coords = cv2.convertPointsFromHomogeneous(homogeneous_4d_coords.transpose())

    output_points = []

    # print threeD_coords
    for point in threeD_coords:
        output_points.append((point[0][0], point[0][1], point[0][2]))
        # output_points.append(point[0])
    # for coord in homogeneous_4d_coords:

    return output_points
# reimplement: https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/calib3d/src/triangulate.cpp#L54

def findScalarGuess(low, high, oldPoints, points1, points2, K, lastR, lastT, r, t, zone=False):
    # find the best scalar multiple of t such that when you compose composedR and lastT with t
    # you have minimal error between the oldTriangulatedPoints and the triangulation of:
    #   points1(with lastR and lastT applied to put it in first cameras coordinate system)
    #   points2(with lastR and lastT applied, and also r and t*scalar applied)
    # return the list of well-triangulated points, along with the total T applied to points2
    # for a given range of scalars returns either a high and low guess for the best scalar
    # or returns a single scalar and its associated triangulated points
    scalars = []
    diff = high-low
    for i in range(6):
        scalars.append(low+diff*i/5.0)
    
    composedR = composeRotations(lastR, r)
    bestScalar = 0
    bestError = -1
    bestTriangulations = []
    print "---"
    for i in range(len(scalars)):
        # find the scalar that produces lowest error
        composedT = composeTranslations(lastT, (t[0]*scalars[i], t[1]*scalars[i], t[2]*scalars[i]))
        newTriangulatedPoints = discreteTriangulateWithTwoRT(points1, points2, K, lastR, lastT, composedR, composedT)
        totalError = findTriangulationError(oldPoints, newTriangulatedPoints)
        print "scalar: ", scalars[i], " error: ", totalError
        if bestError == -1 or (bestError >= 0 and bestError > totalError):
            bestScalar = i
            bestError = totalError
            bestTriangulations = newTriangulatedPoints
    
    if zone:
        lower = scalars[max(0,bestScalar-1)]
        higher = scalars[min(len(scalars)-1,bestScalar+1)]
        return lower, higher
    else:
        return bestTriangulations, composeTranslations(lastT, (t[0]*scalars[bestScalar], t[1]*scalars[bestScalar], t[2]*scalars[bestScalar]))

def minimizeError(oldTriangulatedPoints, points1, points2, K, lastR, r, lastT, t):
    #scalars to check = evenly distributed range between 1/6 and 6 
    
    low, high = findScalarGuess(1/6, 6, oldTriangulatedPoints, points1, points2, K, lastR, lastT, r, t, zone=True)
    low, high = findScalarGuess(low, high, oldTriangulatedPoints, points1, points2, K, lastR, lastT, r, t, zone=True)
    low, high = findScalarGuess(low, high, oldTriangulatedPoints, points1, points2, K, lastR, lastT, r, t, zone=True)
    low, high = findScalarGuess(low, high, oldTriangulatedPoints, points1, points2, K, lastR, lastT, r, t, zone=True)
    low, high = findScalarGuess(low, high, oldTriangulatedPoints, points1, points2, K, lastR, lastT, r, t, zone=True)
    return findScalarGuess(low, high, oldTriangulatedPoints, points1, points2, K, lastR, lastT, r, t, zone=False)
    

def findTriangulationError(points1, points2):
    #find the total sum of error between corresponding point lists

    totalError = 0
    for i in range(len(points1)):
        totalError += eucDist(points1[i], points2[i])

    return totalError

    
