import sys
import argparse
import utils.triangulate as triangulate


def print_help():
    print """Welcome to do_comps.py!
        To run this program, you'll need to select one of the
        modes below and perhaps provide more input.

        modes:
            python do_comps.py detect -i img1.jpg [-i img2.jpg ...] [-f input_folder] [-o output.jpg]
            python do_comps.py match -i img1.jpg -i img2.jpg [-i img3.jpg ...] [-f input_folder] [-o output.jpg]
            python do_comps.py triangulate -i img1.jpg -i img2.jpg [-i img3.jpg ...] [-f input_folder]
                                --scene_output scene.ply [--projection_output projection.ply]
    """


def main():
    if len(sys.argv) < 2:
        print_help()
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default=None, type=str)
    parser.add_argument('-i', default=[], action='append', nargs='?', type=str)
    parser.add_argument('-f', default=None, type=str)
    parser.add_argument('-o', default='output.jpg', type=str)
    parser.add_argument('--scene_output', default=sys.stdout, type=str)
    parser.add_argument('--projection_output', default=None, type=str)
    parser.add_argument('--silent', default=False, type=bool)

    args = parser.parse_args()

    mode = args.mode

    if mode == 'detect':
        if not silent:
            print 'Detecting images: {}'.format(", ".join(args.i))
            print 'Outputting to: {}'.format(args.o)
        # detect()

    elif mode == 'match':
        if not silent:
            print 'Matching images: {}'.format(", ".join(args.i))
            print 'Outputting to: {}'.format(args.o)
        # match()

    elif mode == 'triangulate':
        if not silent:
            print 'Triangulating images: {}'.format(", ".join(args.i))
            print 'Outputting scene to: {}'.format(args.scene_output)
            if args.projection_output:
                print 'Outputting projections to: {}'.format(args.projection_output)
        triangulate.triangulateFromImages(args.i, scene_output=args.scene_output, projections_output=args.projection_output)


if __name__ == '__main__':
    main()


# import os
# import cv2
# import numpy as np
# # from matplotlib import pyplot as plt

# from image import Image

# images = []
# count = 0

# current_dir = os.path.dirname(os.path.realpath(__file__))
# for f in os.listdir(os.path.join(current_dir, "photos")):
#     print f
#     img = Image(os.path.join(current_dir, "photos/{}".format(f)))
#     img.detect_features()
#     images.append(img)
#     count += 1
#     if count > 5:
#         break

# import pprint

# img1 = images[2]
# img2 = images[3]


# # img1 = cv2.imread("testimg.jpg")
# # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# # sift = cv2.xfeatures2d.SIFT_create()
# # (kps, descs) = sift.detectAndCompute(gray, None)


# # FLANN_INDEX_KDTREE = 0
# # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# # search_params = dict(checks = 50)

# # bf = cv2.FlannBasedMatcher(cv2.NORM_L1, crossCheck=True)


# # print img1.descs, img2.descs

# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# matches = bf.match(img1.descs, img2.descs)
# exit()
# # matches = cv2.FlannBasedMatcher(img1.descs, img2.descs)

# print "Matches:"
# # pprint.pprint(matches)
# print len(matches)
# print type(matches)
# # for i in matches:
# #     print i

# pts1 = []
# pts2 = []

# for match in matches:
#     pts2.append(img2.kps[match.trainIdx].pt)
#     pts1.append(img1.kps[match.queryIdx].pt)

# # print(pts1)
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)

# diff_ys = []
# diff_xs = []
# for pt1, pt2 in zip(pts1, pts2):
#     diff_ys.append(pt2[1] - pt1[1])
#     diff_xs.append(pt2[0] - pt1[0])

# threshold = np.std(diff_ys)

# new_pts1 = []
# new_pts2 = []
# new_kps1 = []
# new_kps2 = []
# new_matches = []
# for i, match in enumerate(matches):
#     if abs(pts1[i][1] - pts2[i][1]) < threshold * 0.5:
#         new_matches.append(matches[i])
#         new_pts1.append(pts1[i])
#         new_pts2.append(pts2[i])
#         new_kps1.append(img1.kps[i])
#         new_kps2.append(img2.kps[i])

# # exit()

# # _, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
# # new_pts1 = []
# # new_pts2 = []
# # new_matches = []
# # for i, (mask_incl, match) in enumerate(zip(mask, matches)):
# #     if mask_incl[0] == 1:
# #         new_matches.append(matches[i])
# #         new_pts1.append(pts1[i])
# #         new_pts2.append(pts2[i])

# img3 = cv2.drawMatches(img1.img, img1.kps, img2.img, img2.kps, new_matches, images[1].img, flags=2)

# # plt.imshow(img3),plt.show()

# # E, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# focalLength = img2.k[0][0]
# print "Focal", focalLength

# new_pts1 = np.int32(new_pts1)
# new_pts2 = np.int32(new_pts2)

# E, mask = cv2.findEssentialMat(new_pts1, new_pts2, focal=focalLength)
# print "E"
# print E

# points, r, t, newMask = cv2.recoverPose(E, new_pts1, new_pts2, mask=mask)
# # print points
# print "E-R"
# print r
# print "E-T"
# print t

# F1, mask = cv2.findFundamentalMat(new_pts1, new_pts2, method=cv2.FM_8POINT)
# F2, mask = cv2.findFundamentalMat(new_pts1, new_pts2)
# print "F1"
# print F1

# points, r, t, newMask = cv2.recoverPose(img1.k.transpose().dot(F1).dot(img2.k), new_pts1, new_pts2, mask=mask)
# # print points
# print "F1-R"
# print r
# print "F1-T"
# print t
# print "F2"
# print F2
# points, r, t, newMask = cv2.recoverPose(img1.k.transpose().dot(F2).dot(img2.k), new_pts1, new_pts2, mask=mask)

# print "F2-R"
# print r
# print "F2-T"
# print t


# exit()
# print mask.shape

# points, r, t, newMask = cv2.recoverPose(F, pts1, pts2, mask=mask)
# print points

# proj1mat = np.append(np.identity(3), np.zeros((3, 1)), 1)
# proj2mat = np.append(r, t, 1)

# # reimplement: https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/calib3d/src/triangulate.cpp#L54


# def ourTriangulatePoints(proj1mat, proj2mat, kps1, kps2):
#     assert len(kps1) == len(kps2)

#     matrA = np.zeros((4, 4))
#     matrU = np.zeros((4, 4))
#     matrW = np.zeros((4, 1))
#     matrV = np.zeros((4, 4))

#     outputPoints = np.zeros((len(kps1), 4))

#     kps = [kps1, kps2]
#     projMatrs = [proj1mat, proj2mat]

#     for i in range(len(kps1)):

#         # Row 1 (x1 * P1 3T - P1 1T)
#         matrA[0][0] = kps1[i][0] * proj1mat[2][0] - proj1mat[0][0]
#         matrA[0][1] = kps1[i][0] * proj1mat[2][1] - proj1mat[0][1]
#         matrA[0][2] = kps1[i][0] * proj1mat[2][2] - proj1mat[0][2]
#         matrA[0][3] = kps1[i][0] * proj1mat[2][3] - proj1mat[0][3]

#         # Row 2 (y1 * P1 3T - P1 2T)
#         matrA[1][0] = kps1[i][1] * proj1mat[2][0] - proj1mat[1][0]
#         matrA[1][1] = kps1[i][1] * proj1mat[2][1] - proj1mat[1][1]
#         matrA[1][2] = kps1[i][1] * proj1mat[2][2] - proj1mat[1][2]
#         matrA[1][3] = kps1[i][1] * proj1mat[2][3] - proj1mat[1][3]

#         # Row 3 (x2 * P2 3T - P1 1T)
#         matrA[2][0] = kps2[i][0] * proj2mat[2][0] - proj2mat[0][0]
#         matrA[2][1] = kps2[i][0] * proj2mat[2][1] - proj2mat[0][1]
#         matrA[2][2] = kps2[i][0] * proj2mat[2][2] - proj2mat[0][2]
#         matrA[2][3] = kps2[i][0] * proj2mat[2][3] - proj2mat[0][3]

#         # Row 3 (y2 * P2 3T - P1 2T)
#         matrA[3][0] = kps2[i][1] * proj2mat[2][0] - proj2mat[1][0]
#         matrA[3][1] = kps2[i][1] * proj2mat[2][1] - proj2mat[1][1]
#         matrA[3][2] = kps2[i][1] * proj2mat[2][2] - proj2mat[1][2]
#         matrA[3][3] = kps2[i][1] * proj2mat[2][3] - proj2mat[1][3]

#         # for j in range(2):
#         #     x = kps[j][i][0]
#         #     y = kps[j][i][1]
#         #     for k in range(4):
#         #         matrA[j*2 + 0][k] = x * projMatrs[j][2][k] - projMatrs[j][0][k]
#         #         matrA[j*2 + 1][k] = y * projMatrs[j][2][k] - projMatrs[j][1][k]

#         # cv2.SVDecomp(matrA, matrW, matrU, matrV)
#         # newMatr = matrA * matrA.transpose()
#         # w, v = np.linalg.eig(newMatr)
#         # minEigVec = v[np.where(w == w.min())]
#         U, s, matrV = np.linalg.svd(matrA, full_matrices=True)
#         # print s
#         # ls = list(s)
#         # print ls == sorted(ls)
#         # print U
#         # exit()

#         # print matrV
#         # print matrV[3]

#         # outputPoints[i][0] = minEigVec[0][0] # X
#         # outputPoints[i][1] = minEigVec[0][1] # Y
#         # outputPoints[i][2] = minEigVec[0][2] # Z
#         # outputPoints[i][3] = minEigVec[0][3] # W

#         outputPoints[i][0] = matrV[3][0]  # X
#         outputPoints[i][1] = matrV[3][1]  # Y
#         outputPoints[i][2] = matrV[3][2]  # Z
#         outputPoints[i][3] = matrV[3][3]  # W

#     return outputPoints


# def homogeneousCoordinatesToRegular(arr):
#     num_keypoints = arr.shape[0]
#     outputArr = np.zeros((num_keypoints, 3))

#     for i in range(num_keypoints):
#         # TODO: Throw out point if div by zero?
#         outputArr[i][0] = arr[i][0] / arr[i][3]
#         outputArr[i][1] = arr[i][1] / arr[i][3]
#         outputArr[i][2] = arr[i][2] / arr[i][3]

#         # print outputArr[i]

#     return outputArr


# def ptsToFile(pts, filename):
#     with open(filename, 'w') as f:
#         def writeline(f, line):
#             return f.write("{}\n".format(line))

#         writeline(f, "ply")
#         writeline(f, "format ascii 1.0")
#         writeline(f, "element vertex {}".format(pts.shape[0]))
#         writeline(f, "property float x")
#         writeline(f, "property float y")
#         writeline(f, "property float z")
#         writeline(f, "end_header")

#         for row_num in range(pts.shape[0]):
#             row = pts[row_num][0]
#             writeline(f, "%f %f %f" % (row[0], row[1], row[2]))

# m = ourTriangulatePoints(proj1mat, proj2mat, pts1, pts2)
# # n = homogeneousCoordinatesToRegular(m)
# n = cv2.convertPointsFromHomogeneous(m)
# print n.shape
# ptsToFile(n, 'pts_fixed.ply')

# cmd = "open -a meshlab.app pts_fixed.ply".split(" ")

# import subprocess
# p = subprocess.Popen(cmd)
# # p.kill()


# # print cv2.triangulatePoints(proj1mat,proj2mat,pts1.transpose(),pts2.transpose())


# #### DRAWING EPIPOLAR LINES STUFF ####
# # def drawlines(img1,img2,lines,pts1,pts2):
# #     ''' img1 - image on which we draw the epilines for the points in img2
# #         lines - corresponding epilines '''
# #     print img1.shape
# #     r,c, _ = img1.shape
# #     # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
# #     # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
# #     for r,pt1,pt2 in zip(lines,pts1,pts2):
# #         color = tuple(np.random.randint(0,255,3).tolist())
# #         x0,y0 = map(int, [0, -r[2]/r[1] ])
# #         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
# #         img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
# #         img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
# #         img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
# #     return img1,img2


# # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# # lines1 = lines1.reshape(-1,3)
# # img5,img6 = drawlines(img1.img,img2.img,lines1,pts1,pts2)

# # # Find epilines corresponding to points in left image (first image) and
# # # drawing its lines on right image
# # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# # lines2 = lines2.reshape(-1,3)
# # img3,img4 = drawlines(img2.img,img1.img,lines2,pts2,pts1)

# # plt.subplot(121),plt.imshow(img5)
# # plt.subplot(122),plt.imshow(img3)
# # plt.show()

# # print pts2
# # print pts2

# # exit()

# #### /DRAWING EPIPOLAR LINES STUFF ####


# # store all the good matches as per Lowe's ratio test.
# # good = []
# # for m,n in matches:
# #     if m.distance < 0.7*n.distance:
# #         good.append(m)

# # if len(good)>MIN_MATCH_COUNT:
# #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

# #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
# #     matchesMask = mask.ravel().tolist()

# #     h,w = img1.shape
# #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# #     dst = cv2.perspectiveTransform(pts,M)

# #     img2img = cv2.polylines(img2.img,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# # else:
# #     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
# #     matchesMask = None

# # print "Matches Mask"
# # pprint.pprint(matchesMask)
# # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
# #                    singlePointColor = None,
# #                    matchesMask = matchesMask, # draw only inliers
# #                    flags = 2)

# # img3 = cv2.drawMatches(img1.img,img1.kps,img2img,img2.kps,good,None,**draw_params)

# # plt.imshow(img3, 'gray'),plt.show()
# # # print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))


# #### /ADDING MORE PICTURES STUFF ####

# # Ideally, we start the process with two images with a "large" number of matches
# #   subject to the condition that the matches cannot be modelled by a single homography
# #   We could also just try doing this manually

# # To add a next camera, pick the camera that shares the most keypoints already recovered.
# #   Starting with another camera, we could compute either the essential or fundamental matrices
# #   If we compute essential, the pose we recover must be composed with the R and t of the camera
# #       that we just assumed to be I 0
# #   Add tracks observed by current camera if 1) they have already been observed by a recovered
# #       camera, and 2) if triangulating the track gives a "well-conditioned estimate of its position"

# # Useful methods:
# #   RecoverPose
# #   FindFundamentalMat/FindEssentialMat

# # We need to make sure we can easily identify cameras by the number of tracks already discovered
# #   So that we know which camera to choose next
