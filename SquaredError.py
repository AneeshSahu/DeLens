import numpy as np


class SquaredError():
    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Where N can be any integer>=1
    # Output:  A single floating point value.
    def eval(self, Y, Yhat):
        #print((Y - Yhat) * (Y - Yhat))
        return np.mean((Y - Yhat) * (Y - Yhat))

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        #print(-2*(Y - Yhat))
        return -2*(Y - Yhat)


if __name__ == '__main__':
    '''Y = np.asarray([[1],[2],[3],[4],[5]])
    Yhat = np.asarray([[1], [1], [3], [2], [5]])
    sq = SquarredError()
    print (sq.eval(Y,Yhat))
    print (sq.gradient(Y,Yhat))'''
    Y = np.asarray([[0],[1]])
    Yhat = np.asarray([[0.2],[0.3]])
    err = SquaredError()
    print(err.eval(Y,Yhat))
    print(err.gradient(Y,Yhat))