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


def triangulateTwoImages(filename1, filename2, projections_file=None, cv=False):
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
    if cv:
        triangulated = CVFuncs.cvTriangulate(pts1, pts2, K.matrix, r, t)
    else:
        triangulated = CVFuncs.naiveTriangulate(pts1, pts2, K.matrix, r, t)
    triangulated = draw.transformPointsToViewingCoordinates(triangulated)

    return triangulated, r, t


        
        
