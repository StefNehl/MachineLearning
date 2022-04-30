import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from inference import Regression

class RidgeRegression(Regression):

    def __init__(self, trainStep):
        self.trainStep = trainStep

    def importData(self):
        dataDictonary =  sio.loadmat("AssignmentIV_data_set.mat")
        tempData = dataDictonary.get("TempField")
        self.dataSet = tempData[:,:, 0] #x * y
        self.numberOfSamples = len(self.dataSet)


    def generateTrainingSubset(self):
        self.trainSubset = np.array(self.dataSet)

        count = 0
        for x in range(len(self.dataSet)):
            for y in range(len(self.dataSet[x])):
                count = count + 1
                if count == self.trainStep:
                    count = 0
                    continue

                self.trainSubset[x, y] = 0

    def computeLinearRidgeRegression(self, lambdaValue):
        return 0

    def plotRawData(self):
        plotRange = range(len(self.dataSet))
        #x = np.linspace(min(self.dataSet), max(self.dataSet), self.numberOfSamples)
        #y = self.calculateGaussen1D(x)

        plt.figure(figsize=(8, 8))

        plt.subplot(2, 2, 1)
        #plt.hist(self.dataSet, bins=60, density=True, label="Histogram")
        #plt.plot(x, y, "r-", linewidth=1, label="Distribution")

        plt.title("Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        # plt.legend(loc="upper right")

        plt.subplot(2, 2, 3)
        hm = plt.imshow(self.dataSet,cmap='Reds', interpolation='none',
                        extent=[0,400,0,400])

        plt.colorbar(hm)
        plt.title(f"Raw data")
        plt.xlabel("Latitude coordinate")
        plt.ylabel("Longitude coordinate")

        plt.subplot(2, 2, 4)
        hm = plt.imshow(self.trainSubset,cmap='Reds', interpolation='none',
                        extent=[0,400,0,400])

        plt.colorbar(hm)
        plt.title(f"Train Subset")
        plt.xlabel("Latitude coordinate")
        plt.ylabel("Longitude coordinate")

        plt.tight_layout()
        plt.show()
