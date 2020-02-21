import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
QUESTION 2
'''
'''
Read the data.
'''
data = np.loadtxt('data/hands_aligned_train.txt',skiprows=1)
'''
Caclulate mu
'''
mu=np.mean(data,axis=1)
'''
Calculate W = w-mu
'''
W = data - mu.reshape(-1,1)
'''
Caclulate covariance matrix W.Wt
'''
cov = np.cov(W)
'''
Find the eigen vectors and eigen value of covariance matrix
'''
eigen_value,eigen_vectors=np.linalg.eig(cov)
'''
Take the first 5 eigen vectors and find phi and sigma square.
'''
K=5
sigma_square = 1/(112-K)*(np.sum(eigen_value[K+1:]))
L2 = np.diag(eigen_value)
second = L2[:K,:K] - sigma_square*np.eye(K,K)
phi = np.dot(eigen_vectors[:,:K],np.sqrt(second))
'''
Calculate wi with the weights given in the sheet.
'''
wi=(mu + np.dot(phi,np.array([-0.4,-0.2,0,0.2,0.4])).real)
'''
Plot the wi.
'''
plt.suptitle("Statistical Shape Model , K = 5 and W = [-0.4,-0.2,0,0.2,0.4]")
plt.plot(wi[:56],wi[56:])
plt.show()
plt.pause(2)
plt.close()



'''
QUESTION 3
'''

'''
Read the test shape
'''
test_shape = np.loadtxt('data/hands_aligned_test.txt',skiprows=1)
h = np.array([0,0,0,0,0]) #Assume random h
h_org = np.copy(h)
count_iter=0
while(True):
    w = mu + np.dot(phi,h) # w
    w=w.real
    '''
    Shape changes to calculate psuedo inverse equation components
    '''
    w_xy = np.array([np.split(w,2)[0],np.split(w,2)[1]]).T
    A= []
    for point in w_xy:
        A = A + [[point[0], point[1], 0, 0, 1, 0]]
        A = A + [[0, 0, point[0], point[1], 0, 1]]
    A = np.array(A)
    test_split = np.split(test_shape, 2)
    y = np.array([test_split[0], test_split[1]]).T.flatten()
    '''
    Calculate transformation vector psi 
    '''
    psi = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
    psi_1 = psi[:4].reshape(2,2)
    psi_2 = psi[4:]
    '''
    Transform points according to psi
    '''
    w = np.dot(A,psi).real
    '''
    Caclulate parts of the h update equation
    '''
    second_term = y - (np.dot(A,psi))
    second_term=np.array(np.split(second_term, 56)).T.flatten()
    second_sum = np.zeros((5,1))
    for n in range(56):
        phi_n=np.array([phi[n, :], phi[n+ 56, :]])
        first_n = np.dot(psi_1,phi_n).T.real
        second_n = np.array([second_term[n],second_term[n+56]]).reshape(-1,1)
        second_sum = second_sum + np.dot(first_n,second_n)

    sum_first = np.zeros((5,5))
    for n in range(56):
        phi_n = np.array([phi[n, :], phi[n + 56, :]])
        sum_first += phi_n.T.dot(psi_1.T).dot(psi_1).dot(phi_n).real
    sum_first = sum_first+ sigma_square*np.eye(5,5)
    sum_first_inv = np.linalg.inv(sum_first)
    '''
    Evalulate h
    '''
    h = np.dot(sum_first_inv,second_sum).real.flatten()

    print(h)
    '''
    Check if h has converged
    '''
    if count_iter>15:
        plt.plot(np.array(np.split(w, 56))[:, 0], np.array(np.split(w, 56))[:, 1], c='r')
        plt.show()
        break
    else:
        h_org=h
    count_iter+=1
    '''
    Calculate RMS error
    '''
    RMS = np.sqrt(np.sum(np.square(w-y)))
    plt.suptitle("Inference Step - RMS" + str(RMS))
    plt.plot(np.array(np.split(w,56))[:,0],np.array(np.split(w,56))[:,1],c='r')
    plt.plot(test_shape[:56],test_shape[56:])
    plt.pause(0.2)
    plt.cla()




