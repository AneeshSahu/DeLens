import numpy as np


class LogLoss():
    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Where N can be any integer>=1
    # Output:  A single floating point value.
    def eval(self, Y, Yhat):
        e = 0.0000001
        #print(-(Y * np.log(Yhat + e) + (1 - Y) * np.log(1 - Yhat + e)))
        return np.mean(-(Y * np.log(Yhat + e) + (1 - Y) * np.log(1 - Yhat + e)))

    # Input: Y is an N by K matrix of target values.
    # Input: Yhat is an N by K matrix of estimated values.
    # Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        e = 0.0000001
        return - (Y - Yhat) / (Yhat * (1 - Yhat) + e)

if __name__ == '__main__':
    '''Y = np.asarray([[1],[0],[1],[0],[1]])
    Yhat = np.asarray([[1], [1], [1], [0], [0]])
    ll = LogLoss()
    print (ll.eval(Y,Yhat))
    print (ll.gradient(Y,Yhat))'''

    Y = np.asarray([[0], [1]])
    Yhat = np.asarray([[0.2], [0.3]])
    err = LogLoss()
    print(err.eval(Y, Yhat))
    print(err.gradient(Y,Yhat))