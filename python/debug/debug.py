import cv2
import numpy as np
import subprocess

'''Tests our triangulation method using hard-coded feature points'''

# reimplement: https://github.com/Itseez/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/modules/calib3d/src/triangulate.cpp#L54
def ourTriangulatePoints(proj1mat, proj2mat, kps1, kps2):
	assert len(kps1) == len(kps2)

	matrA = np.zeros((4,4))
	matrU = np.zeros((4,4))
	matrW = np.zeros((4,1))
	matrV = np.zeros((4,4))

	outputPoints = np.zeros((len(kps1),4))

	kps = [kps1,kps2]
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

		U, s, matrV = np.linalg.svd(matrA, full_matrices=True)

		outputPoints[i][0] = matrV[3][0] # X
		outputPoints[i][1] = matrV[3][1] # Y
		outputPoints[i][2] = matrV[3][2] # Z
		outputPoints[i][3] = matrV[3][3] # W

	return outputPoints

def homogeneousCoordinatesToRegular(arr):
	num_keypoints = arr.shape[0]
	outputArr = np.zeros((num_keypoints,3))

	for i in range(num_keypoints):
		# TODO: Throw out point if div by zero?
		outputArr[i][0] = arr[i][0] / arr[i][3]
		outputArr[i][1] = arr[i][1] / arr[i][3]
		outputArr[i][2] = arr[i][2] / arr[i][3]

	return outputArr

def ptsToFile(pts, filename):
	with open(filename, 'w') as f:
		def writeline(f,line):
			return f.write("{}\n".format(line))

		writeline(f,"ply")
		writeline(f,"format ascii 1.0")
		writeline(f, "element vertex {}".format(pts.shape[0]))
		writeline(f, "property float x")
		writeline(f, "property float y")
		writeline(f, "property float z")
		writeline(f,"end_header")

		for row_num in range(pts.shape[0]):
			row = pts[row_num][0]
			writeline(f, "%f %f %f" % (row[0], row[1], row[2]))

def main():
	# Hard-coded from annotated images
	imagePoints1 = [
		(92, 83), 
		(333, 43),
		(520, 120),
		(292, 102),
		(196, 209),
		(136, 304),
		(161, 263),
		(226, 483),
		(372, 304),
		(468, 368)
	]
	imagePoints2 = [
		(65, 104),
		(290, 37),
		(513, 100),
		(286, 104),
		(282, 232),
		(120, 329),
		(194, 285),
		(286, 486),
		(402, 289),
		(462, 330)
	]

	imagePoints1 = np.int32(imagePoints1)
	imagePoints2 = np.int32(imagePoints2)

	principlePoint = (300, 250)

	E, mask = cv2.findEssentialMat(imagePoints1, imagePoints2, pp=principlePoint)

	print "E:"
	print E

	K = np.array([[1, 0, principlePoint[0]], [0, 1, principlePoint[1]], [0, 0, 1]])
	# K inverse and K inverse-transpose
	KI = np.linalg.inv(K)
	KIT = KI.transpose()

	print "Essential mat test: (Should be zero)"
	for i in range(len(imagePoints1)):
		ip1 = imagePoints1[i]
		ip2 = imagePoints2[i]

		# Homogenize
		ip1 = np.append(ip1, [1])
		ip2 = np.append(ip2, [1])

		# Needs to be np array
		arr1 = np.array([ip1])
		arr2 = np.array([ip2]).transpose()

		# Normalized coordinates
		norm1 = np.dot(arr1, KIT)
		norm2 = np.dot(KI, arr2)

		result = np.dot(np.dot(norm1, E), norm2)
		print result

	points, r, t, newMask = cv2.recoverPose(E, imagePoints1, imagePoints2, mask=mask)
	print "R:"
	print r
	print "T:"
	print t

	proj1mat = np.append(np.identity(3), np.zeros((3,1)),1)
	proj2mat = np.append(r,t,1)

	m = ourTriangulatePoints(proj1mat, proj2mat, imagePoints1, imagePoints2)
	n = cv2.convertPointsFromHomogeneous(m)
	ptsToFile(n, 'debug_out.ply')

	cmd = "open -a meshlab.app debug_out.ply".split(" ")
	p = subprocess.Popen(cmd)


if __name__ == '__main__':
	main()