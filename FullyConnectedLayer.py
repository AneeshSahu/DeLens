import math

from Layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    # Input:  sizeIn, the number of features of data coming in
    # Input:  sizeOut, the number of features for the data coming out.
    # Output:  None
    def __init__(self, sizeIn, sizeOut,Xaview =False, Adam = False):
        super().__init__()
        '''
        | ------ SizeOut
        |
        |
        |
        sizeIn
        '''
        # np.random.rand gives values in range [0,1). Need to change it +- 10^-4
        if Xaview:
            self.Xaview(sizeIn,sizeOut)
        else:
            self.weights = (np.random.randn(sizeIn, sizeOut) - 0.5) / 5000
        # print(self.weights)
            self.bias = (np.random.randn(sizeOut) - 0.5) / 5000
        # print(self.bias)
        if Adam:
            self.s = 0
            self.r = 0
    def Xaview(self,sizeIn, sizeOut):

        self.weights = np.random.uniform(-math.sqrt(6/(sizeIn+sizeOut)),math.sqrt(6/(sizeIn+sizeOut)),(sizeIn,sizeOut))
        self.bias = np.zeros(sizeOut)
    # Input:  None
    # Output: The sizeIn x sizeOut weight matrix.
    def getWeights(self):
        return self.weights

    # Input: The sizeIn x sizeOut weight matrix.
    # Output: None
    def setWeights(self, weights):
        self.weights = weights

    # Input:  The 1 x sizeOut bias vector
    # Output: None
    def getBiases(self):
        return self.bias

    # Input:  None
    # Output: The 1 x sizeOut biase vector
    def setBiases(self, biases):
        self.bias = biases

    # Input:  dataIn, an NxD data matrix
    # Output:  An NxK data matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        output = np.dot(dataIn, self.weights) + self.bias
        self.setPrevOut(output)
        return output

    def gradient(self):
        return self.weights.T

    def backward(self, gradIn):
        #print(gradIn.shape, self.gradient().shape)
        return gradIn @ self.gradient()

    def updateWeights(self, gradIn, eta, Adam = False, epoch = 0):
        #print(gradIn.shape)
        #print(gradIn)

        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
        #print(self.getPrevIn().T)
        #print(max(dJdb))
        #print(dJdW)
        #print(dJdb)

        if Adam:
            t = epoch
            p1 = 0.9
            p2 = 0.999
            d = math.pow(10,-8)
            self.s = p1*self.s + (1-p1)*dJdW
            self.r = p2*self.r + (1-p2)*(dJdW* dJdW)
            blend = (self.s/(1-math.pow(p1,t)))
            blend = blend/(np.sqrt(self.r / (1-math.pow(p2,t))) + d)
            self.weights -= eta*blend

            self.bias -= eta * dJdb

        else:


            #print(dJdW,'\n', self.weights)
            #print(dJdb.shape, self.bias.shape)
            self.weights -= eta*dJdW
            self.bias -= eta*dJdb

if __name__ == '__main__':
    '''f = FullyConnectedLayer(4,3)

    X = np.asarray([[1, 1 , 1 , 1],[2,2,2,2]])
    print(f.forward(X))'''

    sm = FullyConnectedLayer(3,2)
    H = np.asarray([[1, 2, 3], [4, 5, 6]])
    sm.setWeights(np.asarray([[1,2],[3,4],[5,6]]))
    sm.setBiases(np.asarray([-1,2]))
    sm.forward(H)
    print(sm.gradient())