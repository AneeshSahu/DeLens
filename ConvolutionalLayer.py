from Layer import Layer
import numpy as np
import math

class ConvolutionalLayer(Layer):
    def __init__(self,kwidth, kcount ,Xavier= False,Adam = False):
        super().__init__()
        self.kwidth = kwidth
        if Xavier:
            self.kernel = np.random.uniform(
                -math.sqrt(6 / (2*kwidth)),
                math.sqrt(6 / (2*kwidth)), (kcount * 2 * kwidth))
        else:
            self.kernel = (np.random.rand(kcount, kwidth, kwidth) - 0.5) / 5000
        self.kernel.reshape(kcount, kwidth, kwidth)

        if Adam:
            self.r=0
            self.s=0

    @staticmethod
    def kernelNmatrix(K, Xlen):
        Kwidth = K.shape[1]
        OstoPad = (Xlen - Kwidth)
        k = np.zeros(Kwidth ** 2 + (Kwidth - 1) * OstoPad)
        for i in range(Kwidth):
            k[i * (Kwidth + OstoPad):(i + 1) * Kwidth + i * OstoPad] = K[i, :]
        W = np.zeros(((Xlen - Kwidth + 1) ** 2, Xlen ** 2))

        for i in range(Xlen - Kwidth + 1):
            for j in range(Xlen - Kwidth + 1):
                W[(Xlen - Kwidth + 1) * i + j][(Xlen) * i + j:(Xlen) * i + j + len(k)] = k
        return W
    @staticmethod
    def crossCorrelate2D(dataIn, kernel):
        '''width = dataIn.shape[1]
        kwidth = kernel.shape[1]
        dataOut=np.zeros((width-kwidth+1, width-kwidth+1))
        currentPos = [0,0]
        for j in range(math.ceil(kwidth / 2) - 1, width - math.floor(kwidth / 2)):
            for i in range(math.ceil(kwidth/2)-1,width - math.floor(kwidth/2)):
                dataOut[currentPos[1]][currentPos[0]] = np.sum(kernel * dataIn[
                      j - math.ceil(kwidth / 2) + 1:j + math.floor(kwidth / 2) + 1,
                      i - math.ceil(kwidth / 2) + 1:i + math.floor(kwidth / 2) + 1])
                currentPos[0]+=1
            currentPos[0] = 0
            currentPos[1] +=1
        return dataOut'''
        width = dataIn.shape[1]
        kwidth = kernel.shape[1]
        W = ConvolutionalLayer.kernelNmatrix(kernel, width)
        return np.matmul(W, dataIn.flatten()).reshape(width-kwidth+1, width-kwidth+1)


    # input is only one image
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        out = []
        # loop over all layers of input
        for i in range(len(dataIn)):
        # loop over all kernels
            for k in range(len(self.kernel)):
                out.append(ConvolutionalLayer.crossCorrelate2D(dataIn[i],self.kernel[k]))
        out = np.asarray(out)
        self.setPrevOut(out)
        return out

    def gradient(self):
        pass

    def backward(self,gradIn : np.ndarray):
        out = np.zeros_like(self.getPrevIn())

        # for each layer of input
        for i in range(len(out)):
            # for each kernel
            for k in range(len(self.kernel)):
                # for each layer of gradIn
                for j in range(len(gradIn)):
                    out[i] += ConvolutionalLayer.crossCorrelate2D(
                        np.pad(
                            gradIn[j],
                            [(self.kwidth-1,self.kwidth-1),(self.kwidth-1,self.kwidth-1)]),
                        self.kernel[k].T)

        return out



    def updateWeights(self, gradIn, eta, Adam = False, epoch = 0):
        if self.residual is not None:
            gradIn += self.residual
        X = self.getPrevIn()
        n = X.shape[0]
        #print(f"{n} is n")
        #print(X.shape)
        #print(gradIn.shape)
        dJdK = np.zeros_like(self.kernel)
        for i in range(n):
            dJdK += ConvolutionalLayer.crossCorrelate2D(X[i],gradIn[i])
        dJdK = dJdK/n
        #print(f"Shape of dJdK is {dJdK.shape}")
        #print(dJdK)
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


if __name__ == "__main__" :
    inp = np.random.rand(16,6,6)
    print(f"Input is {inp}\n shape is {inp.shape}")
    c1 = ConvolutionalLayer(2,2)
    c2 = ConvolutionalLayer(2,2)

    print("Forwards")
    h = c1.forward(inp)
    print(f"Output of c1 is {h}\n shape is {h.shape}")
    h = c2.forward(h)
    print(f"Output of c2 is {h}\n shape is {h.shape}")

    grad = np.random.rand(64,4,4)
    print(f"grad is {grad}\n shape is {grad.shape}")

    print("Backwards")
    h = c2.backward(grad)
    c2.updateWeights(grad,0.1)
    print(f"Output of c2 is {h}\n shape is {h.shape}")
    h2 = c1.backward(h)
    c1.updateWeights(h,0.1)
    print(f"Output of c1 is {h2}\n shape is {h2.shape}")

    '''other = np.asarray([[0,1,2],[3,4,5],[6,7,8]])
    X = np.asarray([[0,1],[2,3]])
    print(ConvolutionalLayer.crossCorrelate2D(other,X))
    K = np.asarray([[3.6,0.4],[-4,-6]])
    grad = np.asarray([[-4,-4,-2],[-4,-8,8],[4,20,14]])
    print(X.shape)
    print(grad.shape)

    #print(ConvolutionalLayer.crossCorrelate2D(grad,X))
    print(ConvolutionalLayer.crossCorrelate2D(grad,K))'''
    '''X = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    K = np.asarray([[1, 2], [3, 4]])
    print(ConvolutionalLayer.crossCorrelate2D(X,K))'''