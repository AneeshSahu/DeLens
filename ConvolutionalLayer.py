from Layer import Layer
import numpy as np
import math

class ConvolutionalLayer(Layer):
    def __init__(self,kwidth,kheight,Xavier= False,Adam = False):
        super().__init__()
        self.kwidth = kwidth
        self.kheight = kheight
        if Xavier:
            self.kernel = np.random.uniform(
                -math.sqrt(6 / (kwidth + kheight)),
                math.sqrt(6 / (kwidth + kheight)), (kwidth, kheight))
        else:
            self.kernel = (np.random.rand(kwidth, kheight) - 0.5) / 5000
        self.kernel.reshape(kheight, kwidth)

        if Adam:
            self.r=0
            self.s=0

    @staticmethod
    def crossCorrelate2D(dataIn, kernel):
        height = dataIn.shape[0]
        width = dataIn.shape[1]
        kheight = kernel.shape[0]
        kwidth = kernel.shape[1]
        dataOut=np.zeros((height-kheight+1, width-kwidth+1))
        currentPos = [0,0]
        for j in range(math.ceil(kheight / 2) - 1, height - math.floor(kheight / 2)):
            for i in range(math.ceil(kwidth/2)-1,width - math.floor(kwidth/2)):
                dataOut[currentPos[1]][currentPos[0]] = np.sum(kernel * dataIn[
                      j - math.ceil(kheight / 2) + 1:j + math.floor(kheight / 2) + 1,
                      i - math.ceil(kwidth / 2) + 1:i + math.floor(kwidth / 2) + 1])

                currentPos[0]+=1
            currentPos[0] = 0
            currentPos[1] +=1
        return dataOut

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        out = []
        for i in range(len(dataIn)):
            out.append(ConvolutionalLayer.crossCorrelate2D(dataIn[i],self.kernel))
        out = np.asarray(out)
        self.setPrevOut(out)
        return out

    def gradient(self):
        pass

    def backward(self,gradIn):
        pass

    def updateWeights(self, gradIn, eta, Adam = False, epoch = 0):
        X = self.getPrevIn()
        n = X.shape[0]
        #print(f"{n} is n")
        #print(X.shape)
        #print(gradIn.shape)
        dJdK = np.zeros((X.shape[1]-gradIn.shape[1]+1,X.shape[2]-gradIn.shape[2]+1))
        for i in range(n):
            dJdK += ConvolutionalLayer.crossCorrelate2D(X[i],gradIn[i])
        dJdK = dJdK/n
        #print(dJdK)

        if Adam:
            t = epoch
            p1 = 0.9
            p2 = 0.999
            d = math.pow(10, -8)
            self.s = p1 * self.s + (1 - p1) * dJdK
            self.r = p2 * self.r + (1 - p2) * (dJdK * dJdK)
            blend = (self.s / (1 - math.pow(p1, t)))
            blend = blend / (np.sqrt(self.r / (1 - math.pow(p2, t))) + d)
            self.kernel -= eta * blend

        else:
            self.kernel -= eta * dJdK

if __name__ == '__main__':
    I = np.random.randn(8, 8)
    L = ConvolutionalLayer(3,3)
    #res = L.crossCorrelate2D(I, 8 ,8)

    I = np.random.randn(7, 7)
    L = ConvolutionalLayer(3, 2)
    #res = L.crossCorrelate2D(I, 7, 7)

    X = np.asarray([[1,2,3,4],[2,2,3,2],[1,3,3,3],[4,4,4,4]])
    k = np.asarray([[1,2,3],[2,2,3],[1,3,3]])
    L = ConvolutionalLayer(3,3)
    L.kernel = k
    res = L.forward(X)
    print(res)

    X = np.asarray([[1,1,0,1,0,0,1,1],[1,1,1,1,0,0,1,0],[0,0,1,1,0,1,0,1],[1,1,1,0,1,1,1,0],[1,1,1,1,1,0,1,1],[0,0,0,0,0,0,0,0],[0,1,1,1,1,0,0,1],[1,0,1,0,0,1,0,1]])
    k = np.asarray([[2,-1,2],[2,-1,0],[1,0,2]])
    L = ConvolutionalLayer(3, 3)
    L.kernel = k
    print(X)
    print(k)
    res = L.forward(X)
    print(res)