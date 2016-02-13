import os
from sys import argv
import subprocess
import cv2
import math
import numpy as np
import utils.debug as Debug
import utils.CVFuncs as CVFuncs
from utils import KMatrix, Image
from triangulate import *
from PIL import Image as PILImage

f1, f2, f3 = argv[1], argv[2], argv[3]
#
image1, image2 = Image(f1), Image(f2)
#
points1, points2, matches, image1.kps, image2.kps = [], [], [], [] ,[]

i = 0
for line in file(f3):
    pointList = line.split(",")
    points1.append((int(pointList[0]),int(pointList[1])))
    points2.append((int(pointList[2]),int(pointList[3])))
    matches.append(cv2.DMatch(i, i, 1))
    image1.kps.append(cv2.KeyPoint(int(pointList[0]),int(pointList[1]), 1, 1,1))
    image2.kps.append(cv2.KeyPoint(int(pointList[2]),int(pointList[3]), 1, 1,1))   
    i += 1
    
print len(image1.kps)
print len(image2.kps)
print len(matches)

matchImage = cv2.drawMatches(image1.img, image1.kps, image2.img, image2.kps, matches, image1.img, flags=2)
img = PILImage.fromarray(matchImage, 'RGB')
img.save("out.jpg")