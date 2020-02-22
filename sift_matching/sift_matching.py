import cv2
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

def main():
    # Load the images
    mountain1 = cv2.imread('data/mountain1.png')
    mountain2 = cv2.imread('data/mountain2.png')

    # extract sift keypoints and descriptors
    sift_model = cv2.xfeatures2d.SIFT_create()
    keypoints_mountain_1, descriptors_mountain_1 = sift_model.detectAndCompute(mountain1,None)
    keypoints_mountain_2, descriptors_mountain_2 = sift_model.detectAndCompute(mountain2,None)
    mountain1_with_kp = cv2.drawKeypoints(mountain1,keypoints_mountain_1,None)
    mountain2_with_kp = cv2.drawKeypoints(mountain2,keypoints_mountain_2,None)

    # your own implementation of matching
    distances = pairwise_distances(descriptors_mountain_1, descriptors_mountain_2)
    sorted_distances = np.sort(distances, axis=1)
    ratio = sorted_distances[:, 0] / sorted_distances[:, 1]
    good_matches = np.where(ratio < 0.4)[0]
    distances = distances[good_matches, :]
    nearest_matches = np.argsort(distances)[:, 0]
    good_keypoint_1 = np.array(keypoints_mountain_1)[good_matches]
    good_keypoint_2 = np.array(keypoints_mountain_2)[nearest_matches]
    match_list = []
    for i,kp in enumerate(good_keypoint_1):
        match_list.append(cv2.DMatch(i,i,distances[i][0]))

    # display the matches
    matched_mapped = cv2.drawMatchesKnn(mountain1,good_keypoint_1,mountain2, good_keypoint_2,[match_list],outImg=None)
    cv2.imwrite("mapped_maches.png",matched_mapped)




if __name__ == '__main__':
    main()
