import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint

from image import Image
from track_creator import *


def getKeypointColor(img, kps, kpindex):
    red = []
    green = []
    blue = []

    xcoord = int(kps[kpindex].pt[0])
    ycoord = int(kps[kpindex].pt[1])

    for y in range(ycoord - 5, ycoord + 5):
        for x in range(xcoord - 5, xcoord + 5):
            if y >= 0 and y < len(img) and x >= 0 and x < len(img[0]):
                blue.append(img[y][x][0])
                green.append(img[y][x][1])
                red.append(img[y][x][2])

    return (sum(red) / len(red), sum(green) / len(green), sum(blue) / len(blue))

images = []
count = 0

current_dir = os.path.dirname(os.path.realpath(__file__))
for f in os.listdir(os.path.join(current_dir, "photos")):
    print f
    img = Image(os.path.join(current_dir, "photos/{}".format(f)))
    img.detect_features()
    print len(img.kps)
    images.append(img)
    count += 1
    if count > 3:
        break

point_list = []
TrackCreator.computeTracks(images, point_list)

points1 = []
points2 = []
kps1 = []
kps2 = []
for i in range(len(point_list)):
    if point_list[i].image_index == 0 and point_list[i].track_number >= 0:
        for j in range(len(point_list[i].match_index_list)):
            secondPoint = point_list[point_list[i].match_index_list[j]]
            if secondPoint.image_index == 1:
                points1.append(images[0].kps[point_list[i].point_index].pt)
                points2.append(images[1].kps[secondPoint.point_index].pt)
                kps1.append(images[0].kps[point_list[i].point_index])
                kps2.append(images[1].kps[secondPoint.point_index])

matches = []
for i in range(len(points1)):
    matches.append(cv2.DMatch(i, i, 0, 0))

img3 = cv2.drawMatches(images[0].img, kps1, images[1].img, kps2, matches, images[1].img, flags=2)


plt.imshow(img3), plt.show()
plt.savefig("newimg.png")

new_pts1 = numpy.int32(points1)
new_pts2 = numpy.int32(points2)

focalLength = images[0].k[0][0]
img1 = images[0]
img2 = images[1]

E, mask = cv2.findEssentialMat(new_pts1, new_pts2, focal=focalLength)

print "E"
print E
#
points, r, t, newMask = cv2.recoverPose(E, new_pts1, new_pts2, mask=mask)
# print points
print "E-R"
print r
print "E-T"
print t

F1, mask = cv2.findFundamentalMat(new_pts1, new_pts2, method=cv2.FM_8POINT)
F2, mask = cv2.findFundamentalMat(new_pts1, new_pts2)
F = F2
print "F1"
print F1

points, r, t, newMask = cv2.recoverPose(img1.k.transpose().dot(F1).dot(img2.k), new_pts1, new_pts2, mask=mask)
# print points
print "F1-R"
print r
print "F1-T"
print t
print "F2"
print F2
points, r, t, newMask = cv2.recoverPose(img1.k.transpose().dot(F2).dot(img2.k), new_pts1, new_pts2, mask=mask)

print "F2-R"
print r
print "F2-T"
print t

# reimplement: https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/calib3d/src/triangulate.cpp#L54


def ourTriangulatePoints(proj1mat, proj2mat, kps1, kps2):
    assert len(kps1) == len(kps2)

    matrA = np.zeros((4, 4))
    matrU = np.zeros((4, 4))
    matrW = np.zeros((4, 1))
    matrV = np.zeros((4, 4))

    outputPoints = np.zeros((len(kps1), 4))

    kps = [kps1, kps2]
    projMatrs = [proj1mat, proj2mat]

    for i in range(len(kps1)):

        # Row 1 (x1 * P1 3T - P1 1T)
        matrA[0][0] = kps1[i][0] * proj1mat[2][0] - proj1mat[0][0]
        matrA[0][1] = kps1[i][0] * proj1mat[2][1] - proj1mat[0][1]
        matrA[0][2] = kps1[i][0] * proj1mat[2][2] - proj1mat[0][2]
        matrA[0][3] = kps1[i][0] * proj1mat[2][3] - proj1mat[0][3]

        # Row 2 (y1 * P1 3T - P1 2T)
        matrA[1][0] = kps1[i][1] * proj1mat[2][0] - proj1mat[1][0]
        matrA[1][1] = kps1[i][1] * proj1mat[2][1] - proj1mat[1][1]
        matrA[1][2] = kps1[i][1] * proj1mat[2][2] - proj1mat[1][2]
        matrA[1][3] = kps1[i][1] * proj1mat[2][3] - proj1mat[1][3]

        # Row 3 (x2 * P2 3T - P1 1T)
        matrA[2][0] = kps2[i][0] * proj2mat[2][0] - proj2mat[0][0]
        matrA[2][1] = kps2[i][0] * proj2mat[2][1] - proj2mat[0][1]
        matrA[2][2] = kps2[i][0] * proj2mat[2][2] - proj2mat[0][2]
        matrA[2][3] = kps2[i][0] * proj2mat[2][3] - proj2mat[0][3]

        # Row 3 (y2 * P2 3T - P1 2T)
        matrA[3][0] = kps2[i][1] * proj2mat[2][0] - proj2mat[1][0]
        matrA[3][1] = kps2[i][1] * proj2mat[2][1] - proj2mat[1][1]
        matrA[3][2] = kps2[i][1] * proj2mat[2][2] - proj2mat[1][2]
        matrA[3][3] = kps2[i][1] * proj2mat[2][3] - proj2mat[1][3]

        # for j in range(2):
        #     x = kps[j][i][0]
        #     y = kps[j][i][1]
        #     for k in range(4):
        #         matrA[j*2 + 0][k] = x * projMatrs[j][2][k] - projMatrs[j][0][k]
        #         matrA[j*2 + 1][k] = y * projMatrs[j][2][k] - projMatrs[j][1][k]

        # cv2.SVDecomp(matrA, matrW, matrU, matrV)
        # newMatr = matrA * matrA.transpose()
        # w, v = np.linalg.eig(newMatr)
        # minEigVec = v[np.where(w == w.min())]
        U, s, matrV = np.linalg.svd(matrA, full_matrices=True)
        # print s
        # ls = list(s)
        # print ls == sorted(ls)
        # print U
        # exit()

        # print matrV
        # print matrV[3]

        # outputPoints[i][0] = minEigVec[0][0] # X
        # outputPoints[i][1] = minEigVec[0][1] # Y
        # outputPoints[i][2] = minEigVec[0][2] # Z
        # outputPoints[i][3] = minEigVec[0][3] # W

        outputPoints[i][0] = matrV[3][0]  # X
        outputPoints[i][1] = matrV[3][1]  # Y
        outputPoints[i][2] = matrV[3][2]  # Z
        outputPoints[i][3] = matrV[3][3]  # W

    return outputPoints


def homogeneousCoordinatesToRegular(arr):
    num_keypoints = arr.shape[0]
    outputArr = np.zeros((num_keypoints, 3))

    for i in range(num_keypoints):
        # TODO: Throw out point if div by zero?
        outputArr[i][0] = arr[i][0] / arr[i][3]
        outputArr[i][1] = arr[i][1] / arr[i][3]
        outputArr[i][2] = arr[i][2] / arr[i][3]

        # print outputArr[i]

    return outputArr


def ptsToFile(pts, filename):
    with open(filename, 'w') as f:
        def writeline(f, line):
            return f.write("{}\n".format(line))

        writeline(f, "ply")
        writeline(f, "format ascii 1.0")
        writeline(f, "element vertex {}".format(pts.shape[0]))
        writeline(f, "property float x")
        writeline(f, "property float y")
        writeline(f, "property float z")
        writeline(f, "end_header")

        for row_num in range(pts.shape[0]):
            row = pts[row_num][0]
            writeline(f, "%f %f %f" % (row[0], row[1], row[2]))


def ptsToFileColor(pts, filename, image1, kps1, image2, kps2):
    with open(filename, 'w') as f:
        def writeline(f, line):
            return f.write("{}\n".format(line))

        writeline(f, "ply")
        writeline(f, "format ascii 1.0")
        writeline(f, "element vertex {}".format(pts.shape[0]))
        writeline(f, "property float x")
        writeline(f, "property float y")
        writeline(f, "property float z")
        writeline(f, "property uchar red")
        writeline(f, "property uchar green")
        writeline(f, "property uchar blue")
        writeline(f, "end_header")

        for row_num in range(pts.shape[0]):
            row = pts[row_num][0]
            line = "%f %f %f" % (row[0], row[1], row[2])
            color1 = getKeypointColor(image1, kps1, row_num)
            color2 = getKeypointColor(image2, kps2, row_num)
            red = (color1[0] + color2[0]) / 2
            green = (color1[1] + color2[1]) / 2
            blue = (color1[2] + color2[1]) / 2
            line2 = "%i %i %i" % (red, green, blue)
            writeline(f, line + " " + line2)


points, r, t, newMask = cv2.recoverPose(E, new_pts1, new_pts2, mask=mask)
proj1mat = np.append(np.identity(3), np.zeros((3, 1)), 1)
proj2mat = np.append(r, t, 1)

m = ourTriangulatePoints(proj1mat, proj2mat, new_pts1, new_pts2)
# n = homogeneousCoordinatesToRegular(m)
n = cv2.convertPointsFromHomogeneous(m)
print n.shape
ptsToFile(n, 'pts_fixed.ply')
print "hello"

#cmd = "open -a meshlab.app pts_fixed.ply".split(" ")
#
#import subprocess
#p = subprocess.Popen(cmd)
# p.kill()


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
#         color = tuple(np.random.randint(0,255,3).tolist())
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
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()

#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)

#     img2img = cv2.polylines(img2.img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

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


E, mask = cv2.findEssentialMat(new_pts1, new_pts2)

points, r, t, newMask = cv2.recoverPose(E, new_pts1, new_pts2, mask=mask)
print "R:"
print r
print "T:"
print t


m = ourTriangulatePoints(proj1mat, proj2mat, new_pts1, new_pts2)
n = cv2.convertPointsFromHomogeneous(m)
ptsToFileColor(n, 'debug_out.ply', images[0].img, kps1, images[1].img, kps2)
