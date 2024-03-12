from Layer import Layer
import numpy as np


class InputLayer(Layer):
    # Input:  dataIn, an NxD matrix
    # Output:  None
    def __init__(self, dataIn,Zscore = True):
        super().__init__()
        if Zscore == True:
            self.meanX = dataIn.mean(axis=0)
            self.stdX = dataIn.std(axis=0, ddof=1)
        #self.stdX = list(self.stdX); self.stdX[2]= 0 ; self.stdX = np.asarray(self.stdX)
            self.stdX[self.stdX == 0] = 1

        self.Zscore = Zscore


    # Input:  dataIn, an NxD matrix
    # Output: An NxD matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        if self.Zscore :
            out = (dataIn - self.meanX) / self.stdX
        else:
            out = dataIn
        self.setPrevOut(out)
        return out

    # We'll worry about these later...
    def gradient(self):
        pass

    def backward(self, gradIn):
        pass


if __name__ == '__main__':
    from loadKC import load_data

    X, Y = load_data()
    i = InputLayer(X)
    print(X.shape)
    print(i.meanX.shape)
    print(i.stdX)
    print(X)
    print(i.forward(X))
