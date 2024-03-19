import math

from Layer import Layer
import numpy as np


class UpsamplingLayer(Layer):
    # Input:  sizeIn, the number of features of data coming in
    # Input:  sizeOut, the number of features for the data coming out.
    # Output:  None
    def __init__(self, sizeOut):
        self.sizeOut = sizeOut
        super().__init__()

    # Input:  dataIn, an NxD data matrix
    # Output:  An NxK data matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        output = np.array([[[0.0 for i in range(self.sizeOut)] for j in range(self.sizeOut)] for k in range(dataIn.shape[0])])
        for layer in range(dataIn.shape[0]):
            factor = self.sizeOut //  dataIn[layer].shape[0]
            for i in range(dataIn[layer].shape[0]):
                for j in range(dataIn[layer].shape[1]):
                    for k in range(i*factor, factor+i*factor):
                        for l in range(j*factor, factor+j*factor):
                            output[layer][k][l] = dataIn[layer][i][j]
        return output

    def gradient(self):
        return None

    def backward(self, gradIn):
        output = np.array([[[0.0 for i in range(self.getPrevIn().shape[1])] for j in range(self.getPrevIn().shape[2])] for k in range(self.getPrevIn().shape[0])])
        factor = gradIn.shape[1] // self.getPrevIn().shape[1]
        for layer in range(self.getPrevIn().shape[0]):
            for i in range(0, len(gradIn[layer]), factor):
                for j in range(0, len(gradIn[layer]), factor):
                        output[layer][i // factor][j // factor] = gradIn[layer][i][j]
        return output



if __name__ == "__main__":
    L1 = UpsamplingLayer(8)
    dataIn = np.array([[[1,2],[3,4]] for i in range(3)])
    forward = L1.forward(dataIn)
    backward = L1.backward(forward)
    print(forward, backward)