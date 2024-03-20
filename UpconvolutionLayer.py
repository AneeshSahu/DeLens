from Layer import Layer
import numpy as np
import math

class UpconvolutionalLayer(Layer):
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
    def crossCorrelate2D(dataIn, kernel):
        width = dataIn.shape[1]
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
        return dataOut

    # input is only one image
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        Y = np.zeros((dataIn.shape[0], dataIn.shape[1] + self.kwidth - 1, self.kwidth + dataIn.shape[1] - 1))
        for layer in range(dataIn.shape[0]):
            for i in range(dataIn.shape[1]):
                for j in range(dataIn.shape[2]):
                    Y[layer][i:i+self.kwidth, j:j+self.kwidth] += (dataIn[layer, i, j]* self.kernel)[0]
        return Y
        # self.setPrevIn(dataIn)
        # out = []
        # # loop over all layers of input
        # for i in range(len(dataIn)):
        # # loop over all kernels
        #     for k in range(len(self.kernel)):
        #         print(f"zpadded shape: ")
        #         out.append(UpconvolutionalLayer.crossCorrelate2D(np.pad(
        #                     dataIn[i],
        #                     [(dataIn.shape[1] % self.kwidth,dataIn.shape[1] % self.kwidth),(dataIn.shape[1] % self.kwidth,dataIn.shape[1] % self.kwidth)]),self.kernel[k].T))
        # out = np.asarray(out)
        # self.setPrevOut(out)
        # return out

    def gradient(self):
        pass

    def backward(self,gradIn : np.ndarray):
        out = np.zeros_like(self.getPrevIn())

        # for each layer of input
        for i in range(len(out)):
            # for each kernel
            for k in range(len(out)):
                # for each layer of gradIn
                for j in range(len(gradIn)):
                    out[i] += UpconvolutionalLayer.crossCorrelate2D(
                        np.pad(gradIn[j], [(self.getPrevIn().shape[1] % self.kwidth,self.getPrevIn().shape[1] % self.kwidth),(self.getPrevIn().shape[1] % self.kwidth,self.getPrevIn().shape[1] % self.kwidth)]),
                        self.kernel[k].T)

        return out


    def updateWeights(self, gradIn, eta, Adam = False, epoch = 0):
        X = self.getPrevIn()
        n = X.shape[0]
        #print(f"{n} is n")
        #print(X.shape)
        #print(gradIn.shape)
        dJdK = np.zeros_like(self.kernel)
        for i in range(n):
            dJdK += UpconvolutionalLayer.crossCorrelate2D(X[i],gradIn[i])
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
    inp = np.random.rand(1,10,10)
    print(f"Input is shape is {inp.shape}")
    c1 = UpconvolutionalLayer(3,1)
    c2 = UpconvolutionalLayer(3,1)

    print("Forwards")
    h = c1.forward(inp)
    print(f"Output of c1 is shape is {h.shape}")
    h = c2.forward(h)
    print(f"Output of c2 is  shape is {h.shape}")

    grad = np.random.rand(4,10,10)
    print(f"grad is  shape is {grad.shape}")

    print("Backwards")
    h = c2.backward(grad)
    c2.updateWeights(grad,0.1)
    print(f"Output of c2 is {h}\n shape is {h.shape}")
    h2 = c1.backward(h)
    c1.updateWeights(h,0.1)
    print(f"Output of c1 is {h2}\n shape is {h2.shape}")