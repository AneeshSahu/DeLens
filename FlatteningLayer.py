from Layer import Layer
import numpy as np

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        length = dataIn.shape[1]*dataIn.shape[2]
        res = dataIn.reshape((dataIn.shape[0],length))
        self.setPrevOut(res)
        return res

    def backward(self,gradIn):
        return np.reshape(gradIn, self.getPrevIn().shape)

    def gradient(self):
        pass


if __name__ == '__main__':
    layer1 = FlatteningLayer()
    X = np.asarray([[1, 2, 3, 4], [2, 2, 3, 2], [1, 3, 3, 3], [4, 4, 4, 4]])
    print(layer1.forward(X))
    print(layer1.backward(np.arange(16)))

    X = np.asarray(
        [[1, 1, 0, 1, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1]])

    print(layer1.forward(X))
    print(layer1.backward(np.arange(64)))