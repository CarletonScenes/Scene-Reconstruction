import os
import sys
import cv2
import math
import numpy as np
import utils.output as output
import utils.draw as draw
import utils.test as test
import utils.CVFuncs as CVFuncs
from utils.ply_file import PlyFile
from utils import KMatrix, Image
from utils.line import Line


def triangulateWithImagesAndPointFile(filename1, filename2, pointFile, projections_file=None):
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

    # CREATE KEYPOINTS
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
    # test.testFundamentalMat(F, pts1, pts2)

    E, mask = CVFuncs.findEssentialMat(pts1, pts2, K)
    # E = CVFuncs.EFromF(F, K)
    # test.testEssentialMat(E, K, pts1, pts2)

    '''
    Get R and T (using artificial ones for now)
    '''
    points, r, t, newMask = CVFuncs.recoverPose(E, pts1, pts2, K)


    ''' 
    Draw image projections using R and T
    '''
    if projections_file:
        draw.drawProjections(pts1, pts2, K.matrix, r, t, projections_file)
    # draw.drawProjections(pts1, pts2, K.matrix, r, t, "manualprojections.ply")
    # draw.drawRandTTransformation(pts1, pts2, K.matrix, r, t, "projections.ply")

    '''
    Triangulate and draw points
    '''
    triangulated = CVFuncs.naiveTriangulate(pts1, pts2, K.matrix, r, t)
    triangulated = draw.transformPointsToViewingCoordinates(triangulated)
    print "hi"
    return triangulated, r, t


def triangulateManualAndOutput(filename1, filename2, pointsFile, output_file=None, projections_file=None):
    projections_ply_file = PlyFile()
    points, r, t = triangulateWithImagesAndPointFile(filename1, filename2, pointsFile, projections_file=projections_ply_file)

    scene_ply_file = PlyFile()
    scene_ply_file.emitPoints(points)

    if (output_file):
        scene_ply_file.save(output_file)
    if projections_file:
        projections_ply_file.save(projections_file)


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
    output.writePointsToFile(points, "test.ply")

if __name__ == '__main__':
    main()
