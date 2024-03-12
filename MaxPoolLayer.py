from Layer import Layer
import numpy as np
import math

class MaxPoolLayer(Layer):
    def __init__(self, width, height, stride):
        super().__init__()
        self.width = width
        self.height = height
        self.stride = stride
        self.semiGradient = None

    def forward(self,dataIn):
        self.semiGradient = np.zeros_like(dataIn)
        D = dataIn.shape[1]
        E = dataIn.shape[2]
        dataOut = np.zeros((dataIn.shape[0],math.floor(((D - self.height) / self.stride) + 1),
                            math.floor(((E - self.width) / self.stride) + 1)))

        for n in range(len(dataIn)):
            currentPos = [0, 0]
            for j in range(math.ceil(self.height / 2) - 1, D - math.floor(self.height / 2), self.stride):
                for i in range(math.ceil(self.width/2)-1,E - math.floor(self.width/2) , self.stride):
                    ystart = j - math.ceil(self.height / 2) + 1
                    ystop = j + math.floor(self.height / 2) + 1
                    xstart = i - math.ceil(self.width / 2) + 1
                    xstop = i + math.floor(self.width / 2) + 1
                    maxval = np.argmax(dataIn[n][ystart:ystop, xstart:xstop])
                    index = np.unravel_index( maxval,
                                              dataIn[n][ystart:ystop,xstart:xstop].shape)
                    self.semiGradient[n][ystart+ index[0]][xstart+index[1]] = 1
                    dataOut[n][currentPos[1],currentPos[0]] = maxval
                    currentPos[0]+=1
                currentPos[0] = 0
                currentPos[1] +=1

        self.setPrevOut(dataOut)
        return dataOut

    def backward(self, gradIn):
        gradIn = np.reshape(gradIn,-1)
        out = self.semiGradient.copy()
        gradItter = 0
        D = out.shape[0]
        E = out.shape[1]

        for j in range(math.ceil(self.height / 2) - 1, D - math.floor(self.height / 2), self.stride):
            for i in range(math.ceil(self.width / 2) - 1, E - math.floor(self.width / 2), self.stride):

                ystart = j - math.ceil(self.height / 2) + 1
                ystop = j + math.floor(self.height / 2) + 1
                xstart = i - math.ceil(self.width / 2) + 1
                xstop = i + math.floor(self.width / 2) + 1
                index = np.unravel_index(np.argmax(out[ystart:ystop,xstart:xstop]),
                                         out[ystart:ystop,xstart:xstop].shape
                                         )
                out[ystart+ index[0]][xstart+index[1]] = gradIn[gradItter]
                gradItter+=1
        return  out

    def gradient(self):
        pass




if __name__ == "__main__":
    X = np.asarray([[4, 7, 1, 7, 2, 3,],[6, 3, 5, 6, 4, 2,], [6, 5, 6, 4, 3, 7,],
                    [4, 2, 5, 2, 5, 0,],[5, 6, 6, 2, 5, 3,], [2, 1, 2, 3, 2, 3,]])

    L = MaxPoolLayer(3,3,3)
    print(L.forward(X))
    print(L.backward(np.asarray([[-2,0],[6,-2]])))

    X = np.asarray([[1,1,2,4],[5,6,7,8],[3,2,1,0],[1,2,3,4]])
    L = MaxPoolLayer(2,2,2)
    print(X.shape)
    print(L.forward(X))
    print(L.backward(np.asarray([[1, 2], [3, 4]])))
