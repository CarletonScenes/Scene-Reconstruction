import os, sys, subprocess
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


# pixel values seem to work
def getArtificialTranslation(x=0,y=0,z=0):
    return np.array([np.array([x,y,z]).transpose()]).transpose()

# pts2[i] * r + t, then "dehomogenize" should be pts1[i]
def sanityCheckRT(pts1,pts2, k, r,t):
    for ptPair in zip(pts1, pts2):
        print np.append(np.linalg.inv(k).dot(np.array(ptPair[0]),[1])).dot(r) + t, ptPair[1]

def main():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    img1 = Image(os.path.join(current_dir, "images/chapel1.jpg"))
    img1.detect_features()
    img2 = Image(os.path.join(current_dir, "images/chapel2.jpg"))
    img2.detect_features()

    pts1, pts2, matches = CVFuncs.findMatches(img1, img2, filter=True)

    # CVFuncs.drawMatches(img1, img2, matches, "test.png")
    # Debug.writePointsToFile(pts1, "test.ply", planar=True)

    principalPoint = (500, 500)
    K = KMatrix(principalPoint=principalPoint)

    F, mask = CVFuncs.findFundamentalMat(pts1, pts2)
    # Debug.testFundamentalMat(F, pts1, pts2)


    E, mask = CVFuncs.findEssentialMat(pts1, pts2, K)
    E = CVFuncs.EFromF(F, K)
    # Debug.testEssentialMat(E, K, pts1, pts2)

    points, r, t, newMask = CVFuncs.recoverPose(E, pts1, pts2, K)
    # print "R:", r
    # print "T:", t

    # possibilities = CVFuncs.decomposeEssentialMat(E)
    # Debug.printRandTPossibilities(possibilities)
    # sanityCheckRT(pts1,pts2,K.matrix,r,t)
    
    r = getArtificalR(-14)
    print t.shape
    t = getArtificialTranslation(250)
    print t.shape

    Debug.drawRandTTransformation(r, t, K, pts1, pts2, "real_transformed.ply")

    projectionMatrix1 = np.append(np.identity(3), np.zeros((3,1)),1)
    projectionMatrix2 = CVFuncs.composeRandT(r, t)

    triangulatedPoints = CVFuncs.triangulatePoints(projectionMatrix1, projectionMatrix2, pts1, pts2)

    Debug.writePointsToFile(triangulatedPoints, "triangulated_pts.ply")
    # cmd = "open -a meshlab.app debug_out.ply".split(" ")
    # p = subprocess.Popen(cmd)


if __name__ == '__main__':
    main()
