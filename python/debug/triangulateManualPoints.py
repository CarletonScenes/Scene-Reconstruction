import os
import sys
import cv2
import math
import numpy as np
import utils.debug as Debug
import utils.CVFuncs as CVFuncs
from utils import KMatrix, Image

# -14 degrees between chapel1.jpg and chapel2.jpg


def getArtificalR(deg):
    rad = math.radians(deg)
    return np.array([[math.cos(rad), 0, math.sin(rad)],
                     [0, 1, 0],
                     [-1 * math.sin(rad), 0, math.cos(rad)]])


def getArtificialTranslation(x=0, y=0, z=0):
    # units are unclear
    return np.array([
        [x], 
        [y], 
        [z]])


def triangulateWithImagesAndPointFile(filename1, filename2, pointFile):
    ''' 
    Read images and detect features 
    '''
    current_dir = os.path.dirname(os.path.realpath(__file__))
    img1 = Image(os.path.join(current_dir, filename1))
    # img1.detect_features()
    img2 = Image(os.path.join(current_dir, filename2))
    # img2.detect_features()

    # Match keypoints
    # pts1, pts2, matches = CVFuncs.findMatches(img1, img2, filter=True)
    pts1, pts2, matches = readPointsFromFile(pointFile)

    ## CREATE KEYPOINTS
    kp1, kp2 = [], []
    for point1, point2 in zip(pts1, pts2):
        kp1.append(cv2.KeyPoint(point1[0], point1[1], 1))
        kp2.append(cv2.KeyPoint(point2[0], point2[1], 1))

    img1.kps = kp1
    img2.kps = kp2

    # CVFuncs.drawMatches(img1, img2, matches, "new.png")
    # print pts1
    # print pts2
    
    ''' 
    Find K 
    '''
    K = img1.K

    ''' 
    Get essential or fundamental matrix
    '''

    # F, mask = CVFuncs.findFundamentalMat(pts1, pts2)
    # Debug.testFundamentalMat(F, pts1, pts2)

    E, mask = CVFuncs.findEssentialMat(pts1, pts2, K)
    # E = CVFuncs.EFromF(F, K)
    # Debug.testEssentialMat(E, K, pts1, pts2)

    '''
    Get R and T (using artificial ones for now)
    '''
    points, r, t, newMask = CVFuncs.recoverPose(E, pts1, pts2, K)
    print t
    # print "R:", r
    # print "T:", t
    # r = np.linalg.inv(r)
    # t = t * -1
    # possibilities = CVFuncs.decomposeEssentialMat(E)
    # Debug.printRandTPossibilities(possibilities)

    # r = getArtificalR(-20)
    # t = getArtificialTranslation(5)

    ''' 
    Draw image projections using R and T
    '''
    Debug.drawProjections(pts1, pts2, K.matrix, r, t, "manualprojections.ply")
    # Debug.drawRandTTransformation(pts1, pts2, K.matrix, r, t, "projections.ply")

    '''
    Triangulate and draw points
    '''
    triangulated = CVFuncs.naiveTriangulate(pts1, pts2, K.matrix, r, t)
    return triangulated, r, t

    # OLD TRIANGULATION CODE

    # A1 = K.matrix.dot(np.append(np.identity(3), np.zeros((3,1)),1))
    # A2 = K.matrix.dot(CVFuncs.composeRandT(r, t))

    # pts1 = CVFuncs.normalizeCoordinates(pts1, K)
    # pts2 = CVFuncs.normalizeCoordinates(pts2, K)

    # Debug.drawRandTTransformation(r, t, K, pts1, pts2, "real_transformed.ply")

    # projectionMatrix1 = np.append(np.identity(3), np.zeros((3,1)),1)
    # projectionMatrix2 = CVFuncs.composeRandT(r, t)

    # triangulatedPoints = CVFuncs.triangulatePoints(A1, A2, pts1, pts2)

    # Debug.writePointsToFile(triangulatedPoints, "triangulated_pts.ply")
    # cmd = "open -a meshlab.app debug_out.ply".split(" ")
    # p = subprocess.Popen(cmd)

def readPointsFromFile(pointFile):
    points1, points2, matches = [], [], []
    with open(pointFile) as file:
        for i, line in enumerate(file):
            x1, y1, x2, y2 = [int(p) for p in line.split(",")]
            points1.append((x1, y1))
            points2.append((x2, y2))
            matches.append(cv2.DMatch(i, i, 1))
    return points1, points2, matches

def main():
    # Lil test
    points, r, t = triangulateWithImagesAndPointFile("images/c1.jpg", "images/c2.jpg", "pointsout.txt")
    Debug.writePointsToFile(points, "test.ply")

if __name__ == '__main__':
    main()

