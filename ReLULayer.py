import numpy as np

from Layer import Layer


class ReLULayer(Layer):
    # Input: None
    # Output:  None
    relu = lambda x: max(0, x)
    def __init__(self):
        super().__init__()

    # Input:  dataIn, an NxD matrix
    # Output: An NxD matrix
    def forward(self, dataIn: np.ndarray):
        self.setPrevIn(dataIn)
        res = Layer.applyOver(np.atleast_2d(dataIn),ReLULayer.relu)
        self.setPrevOut(res)
        return res


    # We'll worry about these later...
    def gradient(self):
        def reluGradHelp(data,identity):
            for i in range(len(data)):
                if data[i] == 0 : 
                    identity[i][i] = 0
            return identity
        
        data = self.getPrevOut()
        if (len(data.shape) == 1): # 2d
            K = len(data)
            N = 1
            grad = reluGradHelp(data,np.identity(K))
        else:
            K = len(data[0])
            N = len(data)
            grad = np.asarray([reluGradHelp(data[i],np.identity(K)) for i in range(N)])
        #print(f"diagonal : {grad.diagonal(0,1,2)}, grad shape : {grad.shape}")
        return grad.diagonal(0,1,2)

        

    def backward(self, gradIn):
        return gradIn * self.gradient()


if __name__ == '__main__':
    from loadKC import load_data

    '''relu = ReLULayer()
    X,Y = load_data()
    print(X.shape)
    X[0][0] = -12
    forward = relu.forward(X)
    print(X)
    print(forward)
    print(forward.shape)'''
    '''l = ReLULayer()
    test = np.asarray([[-20,2,3]])
    #print(len(test.shape))
    l.forward(test)
    print(test,l.gradient())

    test = np.asarray([[1,-2,3],[3,4,-5],[-6,7,8]])
    l.forward(test)
    print(test,l.gradient())
    test = np.asarray([[1,2,3,-4],[3,-4,5,5],[6,7,8,-5]])
    l.forward(test)
    print(test,l.gradient())'''

    relu = ReLULayer()
    H = np.asarray([[1,2,3],[4,5,6]])
    relu.forward(H)
    print(relu.gradient())