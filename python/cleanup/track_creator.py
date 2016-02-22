
class KeypointMetadata:

    def __init__(self, kp, image):
        self.point_index = kp
        self.image_index = image
        self.visited = 0
        self.track_number = -1
        self.match_index_list = []


class TrackCreator:

    @classmethod
    def computeTracks(cls, images):
        fundamental_mat_array = []
        point_list = []
        key1 = []
        key2 = []

        image1 = images[0]
        image2 = images[1]

        matches = []

        point1 = None
        point2 = None

        total_matches = 0
        repeats = 0
        different = 0

        for i in range(len(images)):
            fundamental_mat_array.append([])
            # for j in range(0,i+1):
            #     fundamental_mat_array[i].
            for j in range(i + 1, len(images)):
                key1 = []
                key2 = []
                image1 = images[i]
                image2 = images[j]
                matches = []

                key1, key2, matches, fundamental_mat = self.computerAndFilterPairMatches(image1, image2)

                if i == 0:
                    if j == 1:
                        for a in range(len(key1)):
                            point_list.append(KeypointMetadata(a, i))
                    for b in range(len(key2)):
                        point_list.append(KeypointMetadata(b, j))

                for k in range(0, len(matches)):
                    total_matches += 1
                    for p in range(0, len(point_list)):
                        if point_list[p].image_index == i and point_list[p].point_index = matches[k].queryIdx:
                            for q in range(0, len(point_list)):
                                if point_list[q].image_index == j and point_list[q].point_index == matches[k].trainIdx:
                                    point_list[q].match_index_list.append(p)
                                    point_list[p].match_index_list.append(q)

            track_list = cls.find_and_filter_subgraphs(point_list)

        # print results
        matched = 0
        unmatched = 1
        for p in range(0, len(point_list)):
            print point_list[p].image_index, ": ", point_list[p].track_number

        print "Does matrix[y][x] exist?"
        for m in range(0, len(fundamental_mat_array)):
            for n in range(0, len(fundamental_mat_array[m])):
                if fundamental_mat_array[m][n].rows == 0:
                    print 0
                else:
                    print 1
            print

    # Finds valid tracks which are connected subgraphs of at least two points that have each image no more than once
    # Updates a referenced pointList of the graph
    @classmethod
    def find_and_filter_subgraphs(cls, point_list):
        current_track = 0
        currrent_track_indicies = []
        images_used = []
        repeated_image = 0

        for i in range(0, len(point_list)):
            currrent_track_indicies = []
            images_used = []
            if point_list[i].visited == 0:
                cls.traverse_from_point(i, point_list, currrent_track_indicies)

                if len(currrent_track_indicies) > 1:
                    repeated_image = 0
                    # if this point's image hasn't been seen in this track
                    for j in range(0, len(currrent_track_indicies)):
                        if currrent_track_indicies[j].imageIndex in images_used:
                            images_used.append(point_list[currrent_track_indicies[j]].image_index)
                        else:
                            repeated_image = 1

                    if repeated_image = 0:
                        track_list.append([])
                        for k in range(0, len(currrent_track_indicies)):
                            point_list[currrent_track_indicies[k]].track_number = current_track
                            track_list[current_track].append(point_list[currrent_track_indicies[k]])

                        current_track += 1

                    else:
                        for k in range(0, len(currrent_track_indicies)):
                            point_list[currrent_track_indicies[k]].track_number = -2

    @classmethod
    def traverse_from_point(cls, index, point_list, currrent_track_indicies):
        currrent_track_indicies.append(index)
        point_list[index].visited = 1
        for i in range(0, len(point_list[index].match_index_list)):
            if point_list[point_list[index].match_index_list[i]].visited == 0:
                cls.traverse_from_point(point_list[index].match_index_list[i], point_list, currrent_track_indicies)

    # returns key1, key2, matches, fundamental_mat
    @classmethod
    def compute_and_filter_pair_matches(cls, image1, image2):
        sift = cv2.xfeatures2d.SIFT_create()
        image1kps, image1descs = sift.detectAndCompute(image1, None)
        image2kps, image2descs = sift.detectAndCompute(image2, None)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(image1descs, image2descs)

        # query_indexes = []
        # train_indexes = []

        # for i in range(0, len(matches)):
        #     query_indexes.append(matches[i].queryIdx)
        #     train_indexes.append(matches[i].trainIdx)

        # fundamental_mat =

        pts1 = []
        pts2 = []

        for match in matches:
            pts2.append(img2.kps[match.trainIdx].pt)
            pts1.append(img1.kps[match.queryIdx].pt)

        pts1 = numpy.int32(pts1)
        pts2 = numpy.int32(pts2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        # E, mask = cv2.findEssentialMat(pts1, pts2)

        # TODO: do something to filter matches?

        return pts1, pts2, matches, F
