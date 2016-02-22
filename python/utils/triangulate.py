import os
import sys
import cv2
import math
import numpy as np
import utils.draw as draw
import utils.test as test
from utils.ply_file import PlyFile
import utils.CVFuncs as CVFuncs
from utils import KMatrix, Image
import subprocess


def triangulateTwoImages(filename1, filename2, projections_file=None, naive=False):
    ''' 
    Read images from absolute filepaths and detect features, then triangulate.
    '''

    img1 = Image(filename1)
    img1.detect_features()
    img2 = Image(filename2)
    img2.detect_features()

    pts1, pts2, matches = CVFuncs.findMatchesKnn(img1, img2, filter=True)

    ''' 
    Find K 
    '''
    K = img1.K

    ''' 
    Get essential or fundamental matrix
    '''
    E, mask = CVFuncs.findEssentialMat(pts1, pts2, K)

    '''
    Get R and T (using artificial ones for now)
    '''
    points, r, t, newMask = CVFuncs.recoverPose(E, pts1, pts2, K)

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


def triangulateFromImages(images, scene_file=None, projections_file=None, silent=False, naive=False):
    if not silent:
        print "Triangulating from images:"
        for image in images:
            print image
        print "-------------"

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
    if scene_file:
        scene_ply_file.save(scene_file)
    if projections_file:
        projections_ply_file.write_to_file(projections_file)

