import math

from Layer import Layer
import numpy as np

class Concat(Layer):
    def __init__(self,OtherClass: Layer):
        super().__init__()
        self.OtherClass = OtherClass

    def forward(self,dataIn):
        datatoconcat = self.OtherClass.getPrevOut()
        self.setPrevIn((dataIn, datatoconcat))
        if (datatoconcat.shape[1] != dataIn.shape[1]) or (datatoconcat.shape[2] != dataIn.shape[2]):
            xwant = dataIn.shape[1]
            ywant = dataIn.shape[2]

            x = datatoconcat.shape[1]
            y = datatoconcat.shape[2]
            xmid = x//2
            ymid = y//2
            xstart = xmid - xwant//2
            ymid = ymid - ywant//2

            datatoconcat = datatoconcat[:,xstart:xstart+xwant,ymid:ymid+ywant]

        dataOut = np.concatenate((dataIn,datatoconcat),axis=0)
        self.setPrevOut(dataOut)
        return dataOut
    def backward(self,gradIn):
        dataIn, datatoconcat = self.getPrevIn()
        gradIn1 = gradIn[:len(dataIn)]
        gradIn2 = gradIn[len(dataIn):]
        if (datatoconcat.shape[1] != dataIn.shape[1]) or (datatoconcat.shape[2] != dataIn.shape[2]):
            gradIn2 = np.pad(gradIn2,[(0,0),
                                                ( (datatoconcat.shape[1] - dataIn.shape[1])//2, math.ceil(( (datatoconcat.shape[1] - dataIn.shape[1])/2))) ,
                                                ( (datatoconcat.shape[2] - dataIn.shape[2])//2, math.ceil(( (datatoconcat.shape[2] - dataIn.shape[2])/2)))])

        self.OtherClass.setResidual(gradIn2)

        return gradIn1
    def gradient(self):
        pass
if __name__ == "__main__":
    from ConvolutionalLayer import ConvolutionalLayer
    inp = np.random.rand(1, 10, 10)
    c1 = ConvolutionalLayer(3, 2)
    c2 = ConvolutionalLayer(3, 2)

    con = Concat(c1)
    print(inp)
    h = c1.forward(inp)
    print(h)
    h = c2.forward(h)
    print(h)
    h = con.forward(h)
    print(h)

    grad = np.random.rand(6, 6, 6)

    print(con.backward(h).shape)
    print(c1.getResidual())

