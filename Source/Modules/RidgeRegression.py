import scipy.io as sio
import numpy as np

from inference import Regression

class RidgeRegression(Regression):

    def __init__(self, trainStep):
        self.trainStep = trainStep

    def importData(self):
        dataDictonary =  sio.loadmat("AssignmentIV_data_set.mat")
        tempData = dataDictonary.get("TempField")
        self.data = tempData[:,:, 0] #x * y


    def generateTrainingSubset(self):
        self.trainSubset = []

        count = 0
        for x in self.data:
            ys = []
            for y in x:
                count = count + 1
                if count == self.trainStep:
                    ys.append(y)
                    count = 0

            self.trainSubset.append(ys)

    #def computeLinearRidgeRegression(self, lambdaValue):
