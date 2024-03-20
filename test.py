import numpy as np
import sys
import pdb
np.set_printoptions(threshold=sys.maxsize)

''' k, W = np.zeros(5), np.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    print(k)
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W'''

def kernel2matrix(K,Xlen):
    OstoPad = (Xlen - 2)
    k = np.zeros(4 + OstoPad)
    k[:2] = K[0, :]
    k[2 + OstoPad:4 + OstoPad] = K[1, :]
    W = np.zeros(((Xlen - 1) * (Xlen - 1), Xlen * Xlen))

    for i in range(Xlen - 1):
        for j in range(Xlen - 1):
            W[(Xlen - 1) * i + j][(Xlen) * i + j:(Xlen) * i + j + len(k)] = k
    return W



#Xlen is the length of the input matrix X row
def kernel3matrix(K,Xlen):
    OstoPad = (Xlen - 3)
    k = np.zeros(9 + 2* OstoPad)
    k [:3] = K[0,:]
    k [3+OstoPad:6+OstoPad] = K[1,:]
    k [6+2*OstoPad: 9+2*OstoPad] = K[2,:]
    W = np.zeros(((Xlen-2)* (Xlen-2),Xlen*Xlen))

    for i in range(Xlen-2):
        for j in range(Xlen-2):
            print(i,j)
            print(W[(Xlen-2)*i+j][(Xlen-2)*i+j:(Xlen-2)*i+j+len(k)].shape)
            W[(Xlen-2)*i+j][(Xlen)*i+j:(Xlen)*i+j+len(k)] = k
    return W

X = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
K = np.asarray([[1,2],[3,4]])
W = kernel2matrix(K,4)
print(W)

#X = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
#k = np.asarray([[1,2,3],[4,5,6],[7,8,9]])

#W = kernel3matrix(k,4)

print(f"Conv: {np.matmul(W,X.flatten()).reshape(3,3)}")