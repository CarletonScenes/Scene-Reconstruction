import cv2
from matplotlib import pyplot as plt

img = cv2.imread('greenej.jpg',0)

surf = cv2.SURF(400)
kp, des = surf.detectAndCompute(img,None)
print kp[0], dir(kp[0])
print kp[0].octave, kp[0].pt, kp[0].angle, kp[0].size, kp[0].response
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2)
plt.show()
