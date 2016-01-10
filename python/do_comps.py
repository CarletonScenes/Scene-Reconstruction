import os
import cv2
import numpy
from matplotlib import pyplot as plt

from image import Image

images = []
count = 0

current_dir = os.path.dirname(os.path.realpath(__file__))
for f in os.listdir(os.path.join(current_dir,"photos")):
    img = Image(os.path.join(current_dir, "photos/{}".format(f)))
    img.detect_features()
    images.append(img)
    count += 1
    if count > 1:
        break

import pprint



img1 = images[0]
img2 = images[1]


# img1 = cv2.imread("testimg.jpg")
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# (kps, descs) = sift.detectAndCompute(gray, None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# print img1.descs, img2.descs

matches = bf.match(img1.descs, img2.descs)
print "Matches:"
# pprint.pprint(matches)
print len(matches)
print type(matches)
for i in matches:
    print i

pts1 = []
pts2 = []

for match in matches:
    pts2.append(img2.kps[match.trainIdx].pt)
    pts1.append(img1.kps[match.queryIdx].pt)

pts1 = numpy.int32(pts1)
pts2 = numpy.int32(pts2)

# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
E, mask = cv2.findEssentialMat(pts1, pts2)

_, r, t, newMask = cv2.recoverPose(E, pts1, pts2)

proj1mat = numpy.append(numpy.identity(3), numpy.zeros((3,1)),1)
proj2mat = numpy.append(r,t,1)

# reimplement: https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/calib3d/src/triangulate.cpp#L54
def ourTriangulatePoints(proj1mat, proj2mat, kps1, kps2):
    assert len(kps1) == len(kps2)

    matrA = numpy.zeros((4,4))
    matrU = numpy.zeros((4,4))
    matrW = numpy.zeros((4,1))
    matrV = numpy.zeros((4,4))

    outputPoints = numpy.zeros((4,len(kps1)))

    kps = [kps1.transpose(),kps2.transpose()]
    projMatrs = [proj1mat, proj2mat]

    for i in range(len(kps1)):
        for j in range(2):
            x = kps[j][0][i]
            y = kps[j][1][i]
            for k in range(4):
                matrA[j*2 + 0][k] = x * projMatrs[j][2][k] - projMatrs[j][0][k]
                matrA[j*2 + 1][k] = x * projMatrs[j][2][k] - projMatrs[j][2][k]


        cv2.SVDecomp(matrA, matrW, matrU, matrV)

        outputPoints[0][i] = matrV[3][0]
        outputPoints[1][i] = matrV[3][1]
        outputPoints[2][i] = matrV[3][2]
        outputPoints[3][i] = matrV[3][3]

    return outputPoints

def homogeneousCoordinatesToRegular(arr):
    outputArr = numpy.zeros((3,arr.shape[1]))

    for i in range(arr.shape[1]):
        # TODO: Throw out point if div by zero?
        outputArr[0][i] = arr[0][i] / arr[3][i]
        outputArr[1][i] = arr[1][i] / arr[3][i]
        outputArr[2][i] = arr[2][i] / arr[3][i]

    return outputArr


def ptsToFile(pts, filename):
    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(some_array, 'pts')
    PlyData([el]).write('pts_binary.ply')

m = ourTriangulatePoints(proj1mat, proj2mat, pts1, pts2)
n = homogeneousCoordinatesToRegular(m)


# print cv2.triangulatePoints(proj1mat,proj2mat,pts1.transpose(),pts2.transpose())


#### DRAWING EPIPOLAR LINES STUFF ####
# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     print img1.shape
#     r,c, _ = img1.shape
#     # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(numpy.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2


# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1.img,img2.img,lines1,pts1,pts2)

# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2.img,img1.img,lines2,pts2,pts1)

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()

# print pts2
# print pts2
    
# exit()

#### /DRAWING EPIPOLAR LINES STUFF ####








# store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# if len(good)>MIN_MATCH_COUNT:
#     src_pts = numpy.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = numpy.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()

#     h,w = img1.shape
#     pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)

#     img2img = cv2.polylines(img2.img,[numpy.int32(dst)],True,255,3, cv2.LINE_AA)

# else:
#     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#     matchesMask = None

# print "Matches Mask"
# pprint.pprint(matchesMask) 
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)

# img3 = cv2.drawMatches(img1.img,img1.kps,img2img,img2.kps,good,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()
# # print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
