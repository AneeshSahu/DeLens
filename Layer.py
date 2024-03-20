#Layer.py
from abc import ABC, abstractmethod

import numpy as np


##########BASE CLASS###########
class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []
        self.residual = None

    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn;

    def getPrevOut(self):
        return self.__prevOut
    def setResidual(self, residual):
        self.residual = residual
    def getResidual(self):
        return self.residual

    @staticmethod
    def applyOver(data: np.asarray, f):
        data = list(data)
        for i in range(len(data)):
            data[i] = list(data[i])
            for j in range(len(data[i])):
                # print(dataIn[i][j])
                data[i][j] = f(data[i][j])
            data[i] = np.asarray(data[i])
        data = np.asarray(data)
        return data

    @abstractmethod
    def forward(self,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward(self,gradIn):
        pass