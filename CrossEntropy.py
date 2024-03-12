import numpy as np

class CrossEntropy():

    @staticmethod
    def oneHotEncoding(matrix,columns):
        matrix = np.atleast_2d(matrix)
        newmatrix = np.zeros((len(matrix),columns))
        for i in range(len(matrix)):
            newmatrix[i][matrix[i]] = 1
        #print(newmatrix)
        #print(newmatrix.shape)
        return newmatrix

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Where N can be any integer>=1
    # Output:  A single floating point value.
    def eval(self, Y, Yhat):
        e = 0.0000001

        #maxval = max(np.max(Y), np.max(Yhat))
        #Y = CrossEntropy.oneHotEncoding(Y,maxval+1)
        #Yhat = CrossEntropy.oneHotEncoding(Yhat,maxval+1)
        res = []

        for i in range(len(Y)):
            sum = 0
            for j in range(len(Y[i])):
                #print(Yhat[i][j] + e)
                sum = sum - Y[i][j] * np.log(Yhat[i][j] + e)
            res.append([sum])
        return np.mean(np.asarray(res))

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        e = 0.0000001

        #maxval = max(np.max(Y), np.max(Yhat))
        #Y = CrossEntropy.oneHotEncoding(Y, maxval + 1)
        #Yhat = CrossEntropy.oneHotEncoding(Yhat, maxval + 1)
        #return np.atleast_2d(np.max(((Y)/(Yhat + e)),axis = 1)).T

        #confused about the output. should it be a column vector or matrix with mostly 0??
        # for matrix:
        return -((Y) / (Yhat + e))


if __name__ == '__main__':
    '''cross = CrossEntropy()
    X = 1
    print(cross.eval(X,X))
    X = np.asarray([[1],[2],[3],[1],[0]])
    Xhat = np.asarray([[1], [3], [2], [1], [0]])
    print(cross.eval(X,Xhat))
    print(X.shape)
    print(cross.gradient(X,Xhat),cross.gradient(X,Xhat).shape)'''

    Y = np.asarray([[1,0,0],[0,1,0]])
    Yhat = np.asarray([[0.2, 0.2, 0.6], [0.2, 0.7, 0.1]])
    err = CrossEntropy()

    print(err.eval(Y,Yhat))
    print(err.gradient(Y,Yhat))