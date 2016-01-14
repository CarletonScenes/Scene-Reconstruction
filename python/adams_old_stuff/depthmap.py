import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('1.jpg',0)
imgR = cv2.imread('2.jpg',0)

stereo = cv2.createStereoBM(numDisparities=320, blockSize=5)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()