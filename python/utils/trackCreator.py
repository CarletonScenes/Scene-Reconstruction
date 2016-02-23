import CVFuncs
import cv2
from image import *
        
def getOverlapIndices(list1, list2):
    outList = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i] == list2[j]:
                outList.append((i,j))

class TrackCreator:
    
    def __init__(self, imageList):     
        self.imageList = imageList
        self.kpMatrix = []
        self.triangulatedPoints = []
        for image in self.imageList:
            image.detect_features()
            self.kpMatrix.append([]) 
            self.triangulatedPoints.append([])
            
    def matchPairs(self, KNN=True, filter=True):
        #this function will create a matrix self.kpMatrix like the following:
        #[a, b, none, c, none]
        #[d, e, f, g, none]
        #[h, none, j, k, l]
        #[none, none, m, o, p]
        #if these matches exist: 
        #   image1:a - image2:d
        #   image2:d - image3:h
        #   image1:b - image2:e
        #   ...
        #where a,b,c... are pixel coordinates
        
        
        for i in range(len(self.imageList)-1):
            if KNN:
                if filter:
                    points1, points2, matches = CVFuncs.findMatchesKnn(self.imageList[i], self.imageList[i+1],filter=True, ratio=True)
                else:
                    points1, points2, matches = CVFuncs.findMatchesKnn(self.imageList[i], self.imageList[i+1],filter=False, ratio=True)
            else:
                if filter:
                    points1, points2, matches = CVFuncs.findMatches(self.imageList[i], self.imageList[i+1],filter=True)
                else:
                    points1, points2, matches = CVFuncs.findMatches(self.imageList[i], self.imageList[i+1],filter=False)
                    
            #TESTING        
            matches = [cv2.DMatch(0,1,0),cv2.DMatch(1,2,0),cv2.DMatch(2,3,0)]        
                    
            # for each match...
            for match in matches:
                point1 = match.queryIdx #self.imageList[i].kps[match.queryIdx]

                
                # find out if the self.imageList[i] point was matched to already
                # if the point has been matched already, add its corresponding point in self.imageList[i+1] to the matrix at the same index
                if self.kpMatrix[i].count(point1):
                    idx = self.kpMatrix[i].index(point1)
                    self.kpMatrix[i+1][idx] = match.trainIdx#self.imageList[i+1].kps[match.trainIdx]
                
                else:
                    # otherwise, at a new entry to every image's row in the matrix
                    for j in range(len(self.imageList)):

                        #if it's not image i or i+1, then make it empty
                        if j not in [i,i+1]:
                            self.kpMatrix[j].append(None)
                        #otherwise, add the appropriate points
                        elif j == i:
                            self.kpMatrix[i].append(match.queryIdx) #self.imageList[i].kps[match.queryIdx])
                        else: #j == i+1
                            self.kpMatrix[i+1].append(match.trainIdx) #self.imageList[i+1].kps[match.trainIdx])
            
    def getIndexCorrespondences(self, imageIdx1, imageIdx2):
        # this function returns two lists  points1 and points2
        # indices1 and indices2 are correspondeces between image1 and image2 where entries are indicies in kp lists
        
        indices1, indices2 = [], []
        
        # copy over the relevant rows of kpMatrix
        # copy the points, not the keypoint objects
        # only copy the rows where both images have a point on the track
        for i in range(len(self.kpMatrix[0])):
            if (not self.kpMatrix[imageIdx1][i] == None) and (not self.kpMatrix[imageIdx2][i] == None):
                indices1.append(self.kpMatrix[imageIdx1][i])
                indices2.append(self.kpMatrix[imageIdx2][i])
                            
        return indices1, indices2
    
    def getPointCorrespondences(self, imageIdx1, imageIdx2):
        # this function returns two lists  points1 and points2
        # points1 and points2 are correspondeces between image1 and image2 where entries are pixel coordinates
        
        points1, points2 = [], []
        
        # copy over the relevant rows of kpMatrix
        # copy the points, not the keypoint objects
        # only copy the rows where both images have a point on the track
        for i in range(len(self.kpMatrix[0])):
            if (not self.kpMatrix[imageIdx1][i] == None) and (not self.kpMatrix[imageIdx2][i] == None):
                #append the points found at the given index of the image's keypoints
                points1.append(self.imageList[imageIdx1].kps[self.kpMatrix[imageIdx1][i]].pt)
                points2.append(self.imageList[imageIdx2].kps[self.kpMatrix[imageIdx2][i]].pt)
                            
        return points1, points2

    
    def triangulateImages(self, KNN=True, filter=True):
        # this function will fill the self.triangulatedPoints matrix. Some row i of this matrix would be:
        # [(1,1,1), (2,2,2), (3,3,3)] 
        # if images i and i+1 have 3 keypoint matches that triangulate to (1,1,1), (2,2,2), (3,3,3)
        
        self.matchImages(KNN=KNN, filter=filter)
        
        lastIdxCorrespondences = []
        
        # find corresponding lists of matched image coordinates in images i and i+1
        # also find what indices those coordinates have in their image's keypoint list
        # recover the pose between those two cameras
        for i in range(len(self.imageList-1)):
            points1, points2 = self.getPointCorrespondences(i, i+1)
            indices1, indices2 = self.getIndexCorrespondences(i, i+1)
            K = self.ImageList[i].K
            E, mask = CVFuncs.findEssentialMat(points1, points2, K)
            pts, firstR, firstT, newMask = CVFuncs.recoverPose(E, points1, points2, K)
            
            # if it's the first image pair, then there's no reprojection error to minimize. Keep these triangulated points
            if i == 0:
                r, t = firstR, firstT
                self.triangulatedPoints[i].extend(CVFuncs.discreetTriangulate(points1, points2, K, r, t))
                
            # find triangulated points from cameras i and i+1 using the r and t from recoverPose
            # find a list of (a,b) tuples, overlap, where:
            #   a is the index in self.triangulatedPoints[i-1] of a triangulated point between image i-1 and i
            #   b is the index in firstTriangulation of a triangulated point between image i and i+1
            #   and both were triangulated using the same keypoint in image i
            # make corresponding lists of triangulated points where oldPoints[j] and newPoints[j] both come from the same feature in image i
            # find the r and t that minimize reprojection error, and save all the new triangulations in self.triangulatedPoints[i]
            else:
                firstTriangulation = CVFuncs.discreetTriangulate(points1, points2, K, firstR, firstT)
                overlap = getOverlapIndices(lastIdxCorrespondences, indices1)
                
                oldPoints = []
                newPoints = []
                for pair in overlap:
                    oldPoints.append(self.triangulatedPoints[i-1][pair[0]])
                    newPoints.append(firstTriangulation[pair[1]])
                    
                r, t = minimizeError(oldPoints, newPoints, firstR, firstT)
                
                self.triangulatedPoints[i].extend(CVFuncs.discreetTriangulate(points1, points2, K, r, t))
                
            lastIdxCorrespondences = indices2
            
        
            
def main():
    track = TrackCreator([Image("../photos/pdp1.jpeg"),Image("../photos/pdp2.jpeg"),Image("../photos/pdp1.jpeg"),Image("../photos/pdp2.jpeg")])
    track.matchPairs()
    print track.kpMatrix
    points1, points2 = track.getPointCorrespondences(1,2)
    print zip(points1, points2)
main()