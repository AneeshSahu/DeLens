# LogisticSigmoidLayer.py
import numpy as np
from Layer import Layer


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn: np.ndarray):
        self.setPrevIn(dataIn)
        dataIn = list(np.atleast_2d(dataIn))
        for i in range(len(dataIn)):
            sigma = sum(list(map(np.exp, list(map(float,dataIn[i])))))
            dataIn[i] = np.asarray(np.exp(dataIn[i])/sigma)
        dataIn = np.asarray(dataIn)
        self.setPrevOut(dataIn)
        return dataIn



    def gradient(self):
        def SoftmaxgradHelper(data,matrix):
            for i in range(len(data)):
                for j in range(len(data)):
                    matrix[i][j] = data[i]*((1 if i == j else 0 )- data[j])
                #print(data[i])
            return matrix
        
        data = self.getPrevOut()
        if (len(data.shape) == 1): # 2d
            K = len(data)
            #N = 1
            grad = SoftmaxgradHelper(data,np.identity(K))
        else:
            K = len(data[0])
            N = len(data)
            grad = np.asarray([SoftmaxgradHelper(data[i],np.identity(K)) for i in range(N)])
        return grad

    def backward(self, gradIn):
        return np.einsum('...i,...ij',gradIn,self.gradient())

if __name__ == '__main__':
    from loadKC import load_data
    '''layer1 = Softmax()
    X,Y = load_data()
    print(X)
    Xcap = layer1.forward(X)
    print(Xcap)
    print(X.shape,Xcap.shape)'''

    '''l = Softmax()
    test = np.asarray([-20,0,3])
    #print(len(test.shape))
    l.forward(test)
    print(test,'\n', l.gradient())

    test = np.asarray([[1,-2,3],[3,4,-5],[-6,7,8]])
    l.forward(test)
    print(test,'\n',l.gradient())
    test = np.asarray([[1,2,3,-4],[3,-4,5,5],[6,7,8,-5]])
    l.forward(test)
    print(test,'\n',l.gradient())'''

    sm = Softmax()
    H = np.asarray([[1, 2, 3], [4, 5, 6]])
    sm.forward(H)
    print(sm.gradient())
