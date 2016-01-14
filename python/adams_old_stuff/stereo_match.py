#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


if __name__ == '__main__':
    print 'loading images...'
    imgL = cv2.pyrDown( cv2.imread('5.jpg') )  # downscale images for faster processing
    imgR = cv2.pyrDown( cv2.imread('6.jpg') )

    # disparity range is tuned for 'aloe' image pair
    window_size = 2
    min_disp = 8
    num_disp = 320 #320-min_disp
    stereo = cv2.createStereoSGBM(
        minDisparity = min_disp,
        numDisparities = num_disp,
        # # SADWindowSize = window_size,
        uniquenessRatio = 1, # Good
        speckleWindowSize = 20,
        speckleRange = 16,
        disp12MaxDiff = 20,
        P1 = 30000,
        P2 = 30000,
        # # fullDP = False,
        blockSize=4, # Good
        mode=2
    )

    print 'computing disparity...'
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print 'generating 3d point cloud...',
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = '1.ply'
    write_ply(out_fn, out_points, out_colors)
    print '%s saved' % out_fn

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey(33)
    cv2.destroyAllWindows()
