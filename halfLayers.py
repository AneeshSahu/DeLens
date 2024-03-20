from Layer import Layer
import numpy as np
import math

class halfLayers(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        out = []
        for i in range(len(dataIn)//2):
            print(i*2)
            print(i * 2 + 1)
            out.append((dataIn[i*2] + dataIn[i * 2 + 1])/2)
        if i != math.ceil(len(dataIn)/2):
            out.append(dataIn[-1])
        self.setPrevOut(np.asarray(out))
        return np.asarray(out)
    def backward(self,gradIn):
        layers = self.getPrevIn().shape[0]
        out = []
        for i in range(layers//2):
            out.append(gradIn[i])
            out.append(gradIn[i])
        if i != math.ceil(layers/2):
            out.append(gradIn[-1])
        return np.asarray(out)
    def gradient(self):
        pass





if __name__ == "__main__":
    inp = np.random.rand(31, 10, 10)
    h = halfLayers()
    print(h.forward(inp).shape)
    print(h.backward(np.random.rand(16,10,10)).shape)