

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def display_image(image, title="random"):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plot_image(image, title="random"):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()

def find_edges(image, t1, t2):
    edges = cv.Canny(img, t1, t2)
    return edges

def distance_transform(edges):
    edges[np.where(edges == 255)] = 1.0
    edges[np.where(edges == 0)] = 255.0
    edges[np.where(edges == 1)] = 0.0
    dist = cv.distanceTransform(edges, cv.DIST_L2, 3).astype(np.uint8)
    return dist

def gradient_D(D):
    Gy, Gx = np.gradient(D)
    return Gy, Gx


def read_text_file(filepath):
    with open(filepath) as f:
        data = [tuple(map(int, i[1:-2].split(','))) for i in f]
    return np.array(data)



def iterartive_closest_points(img):
    '''
    step 1 : Find edges of the hand image using Canny
    step 2 : Pre-compute the distance transform of the image
    step 3 : For each point w find the closest point in the edges.(Correspondence)
             a. w = Point on the shape model. (trasnformed)
             b. E = Point in the edge list.
             c. D : Distance transform at point (w)
             d. G = Find the gradient of the distance transform.
             e. x = (w - (D/Magnitude(G))*(Gx*Gy))
    step 4 : Find an affine transformation using closed form solution.
    '''
    E = find_edges(img, 40, 80)
    D = distance_transform(E).astype(np.float32)
    Gy, Gx = gradient_D(D)
    G = np.array([Gy.T, Gx.T]).T
    G_magnitude = np.hypot(Gx, Gy)
    '''
    Read the hand landmark points. 
    '''
    hand_landmarks = read_text_file('data/hand_landmarks.txt')
    hand_landmarks = np.array([hand_landmarks.T[1], hand_landmarks.T[0]]).T
    hand_landmarks_org = np.copy(hand_landmarks)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    '''
    Find correspondence by putting value of w' in above values.
    '''
    params = np.zeros((6,))
    counter_iter = 0
    transformed_points = np.zeros_like(hand_landmarks)
    while (True):
        D_sub = D[hand_landmarks.T[0], hand_landmarks.T[1]]
        G_sub = G[hand_landmarks.T[0], hand_landmarks.T[1]]
        G_magnitude_sub = G_magnitude[hand_landmarks.T[0], hand_landmarks.T[1]]
        second_numerator = D_sub.reshape(-1, 1) * G_sub
        second_denomenator = G_magnitude_sub.reshape(-1, 1)
        second_term = np.divide(second_numerator, second_denomenator, where=second_denomenator != 0)
        second_term = second_term
        x = (hand_landmarks - second_term).astype(int)
        A = []
        for point in hand_landmarks:
            A = A + [[point[1], point[0], 0, 0, 1, 0]]
            A = A + [[0, 0, point[1], point[0], 0, 1]]
        A = np.array(A)
        b = np.array([x.T[1],x.T[0]]).T.flatten()
        params_new = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)
        if np.array_equal(params_new,params):
            break
        params = params_new
        transformed_points = np.dot(A, params).reshape(65, 2)
        transformed_points = np.array([transformed_points.T[1], transformed_points.T[0]]).T.astype(int)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.scatter(hand_landmarks_org[:, 1], hand_landmarks_org[:, 0], c='r', s=5)
        plt.scatter(transformed_points[:, 1], transformed_points[:, 0], c='#67eb34',s=5)
        plt.pause(0.3)
        hand_landmarks = np.round(transformed_points).astype(int)
        counter_iter += 1
    plt.cla()
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.scatter(transformed_points[:, 1], transformed_points[:, 0], c='#67eb34', s=5)
    plt.pause(5)

img = cv.imread('data/hand.jpg', 0)
iterartive_closest_points(img)





