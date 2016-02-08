import os
import sys
import subprocess
import cv2
import math
import numpy as np
import utils.debug as Debug
import utils.CVFuncs as CVFuncs
from utils import KMatrix, Image
from triangulate import *

def triangulateFromImages(*images):
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

	for i in range(len(images)-1):
		image1 = images[i]
		image2 = images[i+1]

		print "Triangulating " + image1 + " and " + image2 + "..."

		points, new_r, new_t = triangulateTwoImages(image1, image2)
		points = CVFuncs.applyRandTToPoints(r, t, points)
		r = CVFuncs.composeRotations(r, new_r)
		t = CVFuncs.composeTranslations(t, new_t)
		Debug.writePointsToFile(points, "points{}.ply".format(i))

def main():
	triangulateFromImages("images/chapel1.jpg", "images/chapel2.jpg", "images/chapel3.jpg")
    # points = triangulateTwoImages("images/chapel2.jpg", "images/chapel3.jpg")
    # r = getArtificalR(-20)
    # t = getArtificialTranslation(5)
    # points2 = []
    # for point in points:
    #     points2.append((r.dot(point) + t.transpose())[0])

    # Debug.writePointsToFile(points, "points1.ply")
    # Debug.writePointsToFile(points2, "points2.ply")


if __name__ == '__main__':
    main()
