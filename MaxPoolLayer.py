from Layer import Layer
import numpy as np
import math

class MaxPoolLayer(Layer):
    def __init__(self, width, stride):
        super().__init__()
        self.width = width
        self.stride = stride
        self.semiGradient = None

    def forward(self,dataIn):
        self.semiGradient = np.zeros_like(dataIn)
        D = dataIn.shape[1]
        E = dataIn.shape[2]
        dataOut = np.zeros((dataIn.shape[0],math.floor(((D - self.width) / self.stride) + 1),
                            math.floor(((E - self.width) / self.stride) + 1)))

        for n in range(len(dataIn)):
            currentPos = [0, 0]
            for j in range(math.ceil(self.width / 2) - 1, D - math.floor(self.width / 2), self.stride):
                for i in range(math.ceil(self.width/2)-1,E - math.floor(self.width/2) , self.stride):
                    ystart = j - math.ceil(self.width / 2) + 1
                    ystop = j + math.floor(self.width / 2) + 1
                    xstart = i - math.ceil(self.width / 2) + 1
                    xstop = i + math.floor(self.width / 2) + 1
                    maxval = np.argmax(dataIn[n][ystart:ystop, xstart:xstop])
                    index = np.unravel_index( maxval,
                                              dataIn[n][ystart:ystop,xstart:xstop].shape)
                    self.semiGradient[n][ystart+ index[0]][xstart+index[1]] = 1
                    dataOut[n][currentPos[1],currentPos[0]] = dataIn[n][ystart+ index[0]][xstart+index[1]]
                    currentPos[0]+=1
                currentPos[0] = 0
                currentPos[1] +=1

        self.setPrevOut(dataOut)
        return dataOut

    def backward(self, gradIn):
        gradIn = np.reshape(gradIn,-1)
        out = self.semiGradient.copy()
        gradItter = 0
        D = out.shape[1]
        E = D

        for n in range(len(out)):
            currentPos = [0, 0]
            for j in range(math.ceil(self.width / 2) - 1, D - math.floor(self.width / 2), self.stride):
                for i in range(math.ceil(self.width / 2) - 1, E - math.floor(self.width / 2), self.stride):
                    ystart = j - math.ceil(self.width / 2) + 1
                    ystop = j + math.floor(self.width / 2) + 1
                    xstart = i - math.ceil(self.width / 2) + 1
                    xstop = i + math.floor(self.width / 2) + 1
                    maxval = np.argmax(out[n][ystart:ystop, xstart:xstop])
                    index = np.unravel_index(maxval,
                                             out[n][ystart:ystop, xstart:xstop].shape)
                    out[n][ystart+ index[0]][xstart+index[1]] = gradIn[gradItter]
                    gradItter += 1
                    currentPos[0] += 1
                currentPos[0] = 0
                currentPos[1] += 1
        return out
    def gradient(self):
        pass




if __name__ == "__main__":
    inp = np.random.rand(4, 6, 6)
    mpl = MaxPoolLayer(2, 2)
    print(mpl.forward(inp))
    print(mpl.semiGradient)
    grad = np.random.rand(4,3,3)
    print(grad)
    print(mpl.backward(grad))