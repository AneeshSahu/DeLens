# LogisticSigmoidLayer.py
import numpy as np
from Layer import Layer


class LogisticSigmoidLayer(Layer):
    logit = lambda x: 1 / (1 + np.exp(-x))
    def __init__(self):
        super().__init__()

    def forward(self, dataIn: np.ndarray):
        self.setPrevIn(dataIn)
        self.setPrevOut(np.atleast_2d(LogisticSigmoidLayer.logit(dataIn)))
        return self.getPrevOut()

    def gradient(self):
        def LogitHelper(data,identity):
            for i in range(len(data)):
                identity[i][i] = data[i] * (1-data[i])
                #print(data[i])
            return identity
        
        data = self.getPrevOut()
        if (len(data.shape) == 1): # 2d
            K = len(data)
            N = 1
            grad = LogitHelper(data,np.identity(K))
        else:
            K = len(data[0])
            N = len(data)
            grad = np.asarray([LogitHelper(data[i],np.identity(K)) for i in range(N)])
        #print(grad.shape)
        return grad.diagonal(0,1,2)

    def backward(self, gradIn):
        return gradIn * self.gradient()

if __name__ == '__main__':
    from loadKC import load_data
    '''layer1 = LogisticSigmoidLayer()
    X,Y = load_data()
    print(X)
    Xcap = layer1.forward(X)
    print(Xcap)
    print(X.shape,Xcap.shape)'''

    '''l = LogisticSigmoidLayer()
    test = np.asarray([-20,0,3])
    #print(len(test.shape))
    l.forward(test)
    print(test,'\n', l.gradient())

    test = np.asarray([[1,-2,3],[3,4,-5],[-6,7,8]])
    l.forward(test)
    print(test,'\n',l.gradient())
    test = np.asarray([[1,2,3,-4],[3,-4,5,5],[6,7,8,-5]])
    l.forward(test)
    print(test,'\n',l.gradient())

    print(f"{LogisticSigmoidLayer.logit(-20)}")'''

    sm = LogisticSigmoidLayer()
    H = np.asarray([[1, 2, 3], [4, 5, 6]])
    sm.forward(H)
    print(sm.gradient())
