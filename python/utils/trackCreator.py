import CVFuncs
import cv2
from image import *
from ply_file import PlyFile
import os
import draw
        
def getOverlapIndices(list1, list2):
    '''
    takes two lists, returns tuples (i,j) that indicate the same item in list1[i] and list2[j]
    '''
    outList = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i] == list2[j]:
                outList.append((i,j))
                
    return outList

class TrackCreator:
    
    def __init__(self, imageFiles):     
        self.imageList = []
        self.kpMatrix = []
        self.triangulatedPoints = []
        self.projectedPoints = []
        self.RTList = []
        for image in imageFiles:
            self.imageList.append(Image(image))
            self.imageList[len(self.imageList)-1].detect_features()
            self.kpMatrix.append([]) 
            self.triangulatedPoints.append([])
            self.projectedPoints.append([])
            
    
    def matchPairs(self, KNN=True, filter=True):
        '''
        this function will create a matrix self.kpMatrix like the following:
        [a, b, none, c, none]
        [d, e, f, g, none]
        [h, none, j, k, l]
        [none, none, m, o, p]
        if these matches exist: 
           image1:a - image2:d
           image2:d - image3:h
           image1:b - image2:e
           ...
        where a,b,c... are pixel coordinates
        '''
        
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
        '''
        returns two lists  points1 and points2
        indices1 and indices2 are correspondeces between image1 and image2 where entries are indicies in kp lists
        '''
        
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
        '''
        returns two lists  points1 and points2
        points1 and points2 are correspondeces between image1 and image2 where entries are pixel coordinates
        '''
        
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

    
    def triangulateImages(self, scene_file=None, projections_file=None, silent=False):
        '''
        Triangulates all image matches and outputs triangulates and projections
        '''
        
        self.matchPairs(KNN=True, filter=True)
        
        lastIdxCorrespondences = []
        
        # find corresponding lists of matched image coordinates in images i and i+1
        # also find what indices those coordinates have in their image's keypoint list
        # recover the pose between those two cameras
        for i in range(len(self.imageList)-1):
            if not silent:
                print "Triangulating " + self.imageList[i].fname + " and " + self.imageList[i+1].fname + "..."
            points1, points2 = self.getPointCorrespondences(i, i+1)
            indices1, indices2 = self.getIndexCorrespondences(i, i+1)
            K = self.imageList[i].K
            E, mask = CVFuncs.findEssentialMat(points1, points2, K)
            pts, r, t, newMask = CVFuncs.recoverPose(E, points1, points2, K)
            
            if i == 0:
                # if it's the first image pair, then there's no reprojection error to minimize. Keep these triangulated points
                self.triangulatedPoints[i].extend(CVFuncs.discreteTriangulate(points1, points2, K.matrix, r, t))
                self.RTList.append([r,t])
                if projections_file:
                    self.projectedPoints[i].extend(draw.getProjections(points1, points2, K.matrix, r, t))
            
            else:
                # find a list of (a,b) tuples, overlap, where:
                #   a is the index in self.triangulatedPoints[i-1] of a triangulated point between image i-1 and i
                # make corresponding lists of triangulated points where oldPoints[j] and newPoints[j] both come from the same feature in image i
                # find the r and t that minimize reprojection error, and save all the new triangulations in self.triangulatedPoints[i]
                lastR = self.RTList[i-1][0]
                lastT = self.RTList[i-1][1]
                overlap = getOverlapIndices(lastIdxCorrespondences, indices1)
                
                oldTriangulatedPoints = []
                newImagePoints1, newImagePoints2 = [], []
                for pair in overlap:
                    oldTriangulatedPoints.append(self.triangulatedPoints[i-1][pair[0]])
                    newImagePoints1.append(points1[pair[1]])
                    newImagePoints2.append(points2[pair[1]])
                    
                minimumErrorT = CVFuncs.minimizeError(oldTriangulatedPoints, newImagePoints1, newImagePoints2, K.matrix, lastR, lastT, r, t)
                newTriangulations = CVFuncs.discreteTriangulateWithTwoRT(points1, points2, K.matrix, lastR, lastT, r, minimumErrorT)
                
                self.RTList.append([r, minimumErrorT])
                self.triangulatedPoints[i].extend(newTriangulations)
                if projections_file:
                    self.projectedPoints[i].extend(draw.getProjectionsWithTwoRT(points1, points2, K.matrix, lastR, lastT, r, minimumErrorT))
                            
            lastIdxCorrespondences = indices2
            
        if scene_file:
            for i in range(len(self.imageList)-1):
                scenePly = PlyFile()
                scenePly.emitPoints(self.triangulatedPoints[i])
                scenePly.save(file(CVFuncs.addPostToPath(scene_file, str(i)+"-"+str(i+1)), "w"))
        if projections_file:
            for i in range(len(self.imageList)-1):
                projPly = PlyFile()
                projPly.emitPoints(self.projectedPoints[i])
                projPly.save(file(CVFuncs.addPostToPath(projections_file, str(i)+"-"+str(i+1)), "w"))
                     