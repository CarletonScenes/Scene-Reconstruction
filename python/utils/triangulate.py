import os
import sys
import cv2
import math
import numpy as np
import utils.output as output
import utils.draw as draw
import utils.test as test
from utils.ply_file import PlyFile
import utils.CVFuncs as CVFuncs
from utils import KMatrix, Image
import subprocess

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


def triangulateTwoImages(filename1, filename2, projections_file=None, naive=False):
    ''' 
    Read images from absolute filepaths and detect features, then triangulate.
    '''

    img1 = Image(filename1)
    img1.detect_features()
    img2 = Image(filename2)
    img2.detect_features()

    # Match keypoints
    pts1, pts2, matches = CVFuncs.findMatchesKnn(img1, img2, filter=True)
    # CVFuncs.drawMatches(img1, img2, matches, "test.png")

    ''' 
    Find K 
    '''
    K = img1.K

    ''' 
    Get essential or fundamental matrix
    '''

    # F, mask = CVFuncs.findFundamentalMat(pts1, pts2)
    # test.testFundamentalMat(F, pts1, pts2)

    E, mask = CVFuncs.findEssentialMat(pts1, pts2, K)
    # E = CVFuncs.EFromF(F, K)
    # test.testEssentialMat(E, K, pts1, pts2)

    '''
    Get R and T (using artificial ones for now)
    '''
    points, r, t, newMask = CVFuncs.recoverPose(E, pts1, pts2, K)
    # print "R:", r
    # print "T:", t
    # r = np.linalg.inv(r)
    # t = t * -1
    # possibilities = CVFuncs.decomposeEssentialMat(E)
    # output.printRandTPossibilities(possibilities)

    # r = getArtificalR(-20)
    # t = getArtificialTranslation(5)

    ''' 
    Draw image projections using R and T
    '''
    if projections_file:
        draw.drawProjections(pts1, pts2, K.matrix, r, t, projections_file)

    '''
    Triangulate and draw points
    '''
    if naive:
        triangulated = CVFuncs.naiveTriangulate(pts1, pts2, K.matrix, r, t)
    else:
        triangulated = CVFuncs.cvTriangulate(pts1, pts2, K.matrix, r, t)
    triangulated = draw.transformPointsToViewingCoordinates(triangulated)

    return triangulated, r, t

    # OLD TRIANGULATION CODE

    # A1 = K.matrix.dot(np.append(np.identity(3), np.zeros((3,1)),1))
    # A2 = K.matrix.dot(CVFuncs.composeRandT(r, t))

    # pts1 = CVFuncs.normalizeCoordinates(pts1, K)
    # pts2 = CVFuncs.normalizeCoordinates(pts2, K)

    # draw.drawRandTTransformation(r, t, K, pts1, pts2, "real_transformed.ply")

    # projectionMatrix1 = np.append(np.identity(3), np.zeros((3,1)),1)
    # projectionMatrix2 = CVFuncs.composeRandT(r, t)

    # triangulatedPoints = CVFuncs.triangulatePoints(A1, A2, pts1, pts2)

    # output.writePointsToFile(triangulatedPoints, "triangulated_pts.ply")
    # cmd = "open -a meshlab.app debug_out.ply".split(" ")
    # p = subprocess.Popen(cmd)


def triangulateFromImages(images, scene_file=sys.stdout, projections_file=None, silent=False, naive=False):
    if not silent:
        print "Triangulating from images:"
        for image in images:
            print image
        print "-------------"

    # Init R and T, which will be used to compose multiple point clouds
    r = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
    t = np.array([
        [0],
        [0],
        [0]])

    scene_ply_file = PlyFile()
    projections_ply_file = PlyFile()

    for i in range(len(images) - 1):
        image1 = images[i]
        image2 = images[i + 1]

        if not silent:
            print "Triangulating " + image1 + " and " + image2 + "..."

        points, new_r, new_t = triangulateTwoImages(image1, image2,
                                                    projections_file=projections_ply_file,
                                                    naive=naive)
        points = CVFuncs.applyRandTToPoints(r, t, points)
        r = CVFuncs.composeRotations(r, new_r)
        t = CVFuncs.composeTranslations(t, new_t)
        scene_ply_file.emitPoints(points)

    scene_ply_file.write_to_file(scene_file)
    if projections_file:
        projections_ply_file.write_to_file(projections_file)


def main():
    # Lil test
    points, r, t = triangulateTwoImages("../photos/chapel1.jpg", "../photos/chapel2.jpg")
    output.writePointsToFile(points, "../test.ply")

if __name__ == '__main__':
    main()
