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

# units are unclear
def getArtificialTranslation(x=0,y=0,z=0):
    return np.array([np.array([x,y,z]).transpose()]).transpose()

def main():

    ''' 
    Read images and detect features 
    '''
    current_dir = os.path.dirname(os.path.realpath(__file__))
    img1 = Image(os.path.join(current_dir, "images/chapel1.jpg"))
    img1.detect_features()
    img2 = Image(os.path.join(current_dir, "images/chapel2.jpg"))
    img2.detect_features()

    # Match keypoints
    pts1, pts2, matches = CVFuncs.findMatches(img1, img2, filter=True)
    CVFuncs.drawMatches(img1, img2, matches, "test.png")
    exit(0)
    ''' 
    Find K 
    '''
    K = img1.K

    ''' 
    Get essential or fundamental matrix
    '''

    print CVFuncs.getEssentialMat(pts1, pts2, K.matrix)
    exit()

    # F, mask = CVFuncs.findFundamentalMat(pts1, pts2)
    # Debug.testFundamentalMat(F, pts1, pts2)

    # E, mask = CVFuncs.findEssentialMat(pts1, pts2, K)
    # E = CVFuncs.EFromF(F, K)
    # Debug.testEssentialMat(E, K, pts1, pts2)

    '''
    Get R and T (using artificial ones for now)
    '''
    # points, r, t, newMask = CVFuncs.recoverPose(E, pts1, pts2, K)
    # print "R:", r
    # print "T:", t
    # r = np.linalg.inv(r)
    # t = t * -1
    # possibilities = CVFuncs.decomposeEssentialMat(E)
    # Debug.printRandTPossibilities(possibilities)

    r = getArtificalR(-20)
    t = getArtificialTranslation(5)

    ''' 
    Draw image projections using R and T
    '''
    Debug.drawProjections(pts1, pts2, K.matrix, r, t, "projections.ply")

    '''
    Triangulate and draw points
    '''
    triangulated = CVFuncs.naiveTriangulate(pts1,pts2,K.matrix,r,t)
    Debug.writePointsToFile(triangulated, "triangulated.ply")


    ## OLD TRIANGULATION CODE

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


if __name__ == '__main__':
    main()
