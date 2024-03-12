from Layer import Layer
import numpy as np


class LinearLayer(Layer):
    # Input: None
    # Output:  None
    def __init__(self):
        super().__init__()
    # Input:  dataIn, an NxD matrix
    # Output: An NxD matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn


    # We'll worry about these later...
    def gradient(self):
        data = self.getPrevOut()
        if (len(data.shape) == 1): # 2d
            K = len(data)
            N = 1
            #grad = np.identity(K)
        else:
            K = len(data[0])
            N = len(data)
            #grad = np.asarray([np.identity(K) for i in range(N)])
        grad = np.ones((N,K))
        return grad

    def backward(self, gradIn):
        return gradIn * self.gradient()


if __name__ == '__main__':
    from loadKC import load_data
    l = LinearLayer()
    test = np.asarray([1,2,3])
    #print(len(test.shape))
    l.forward(test)
    print(l.gradient())

    test = np.asarray([[1,2,3],[3,4,5],[6,7,8]])
    l.forward(test)
    print(l.gradient())
    test = np.asarray([[1,2,3,4],[3,4,5,5],[6,7,8,5]])
    l.forward(test)
    print(l.gradient())