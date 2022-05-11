import matplotlib.pyplot
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sbr
import pandas as pd

from inference import Regression
from GaussDistribution import GaussDistribution

class GaussianProcess(Regression):

    def __init__(self, trainStep):
        self.trainStep = trainStep
        self.hasGenerated1DTestData = False

    def importData(self):
        dataDictonary =  sio.loadmat("AssignmentIV_data_set.mat")

        tempTimeData = dataDictonary.get("TempField")
        self.tempData = np.array(tempTimeData[:,:, 0])

        longData = dataDictonary.get("LongitudeScale") #x-value: Long
        latData = dataDictonary.get("LatitudeScale") #y-value: Lat

        self.x_test = np.array(dataDictonary.get("x_test")) # Testdata from DataSet
        self.y_test = np.array(dataDictonary.get("y_test")) # Testdata from DataSet

        longLatData = []
        arrayValues = []

        for y in range(len(self.tempData)):
            for x in range(len(self.tempData[y])):
                value = self.tempData[y,x]
                if np.isnan(value):
                    continue

                arrayValues.append(value)
                longLatData.append((latData[y][0], longData[x][0])) #[0] needed because of strange import of long/latData

        self.inputValues = np.array(longLatData)
        self.outputValues = np.array(arrayValues)
        self.numberOfSamples = len(self.inputValues)

    def generateTrainingSubset(self):
        self.trainSubsetInput = np.array(self.inputValues[0:len(self.inputValues):self.trainStep])
        self.trainSubsetOutput = np.array(self.outputValues[0:len(self.outputValues):self.trainStep])

    def computeLinearRidgeRegression(self, lambdaValue):
        self.lambdaValue = lambdaValue
        self.weightVector = np.array((1,1,1))
        return self.weightVector

    def testModel(self, weight):
        self.yResult = np.hstack([x @ self.weightVector for x in self.x_test])

    def getYTestData(self):
        return self.y_test

    def computeError(self, yStar):
        self.yError = np.transpose(abs(self.yResult - yStar))
        reversedArray = np.flip(np.sort(self.yError, 0))
        self.errorDataFrame = pd.DataFrame({
            'values': range(len(reversedArray)),
            'error': [error[0] for error in reversedArray]
        })

    def getDescSortedError(self):
        return self.errorDataFrame

    def computMeanOfError(self):
        self.meanError = np.mean(self.yError)

    def getMeanError(self):
        return self.meanError

    def plotError(self):
        plt.figure(figsize=(8, 6))
        errorPlot = sbr.barplot(data=self.errorDataFrame, x="values", y="error", palette="coolwarm_r")

        plt.xlabel("Descending Sorted Y Errors")
        plt.ylabel("Error |yResult - yStar| [C]")
        plt.title("Error |yResult - yStar| [C] with Lambda: " + str(self.lambdaValue))
        plt.tight_layout()
        matplotlib.pyplot.show()

    def plotHeatMap(self):
        tempErrorData = \
            {
                'Lat': [round(long,2) for long in self.x_test[:, 1]],
                'Long': [round(lat,2) for lat in self.x_test[:, 2]],
                'error':[error[0] for error in self.yError]
            }

        tempDataFrame = pd.DataFrame(tempErrorData)
        tempDataFrame = tempDataFrame.pivot("Long", "Lat", "error")
        reversedTempErrorData = tempDataFrame.sort_values(("Long"), ascending=False)

        def fmt(x, y):
            return '{:,.2f}'.format(x)

        plt.figure(figsize=(8,6))
        errorHeatMap = sbr.heatmap(reversedTempErrorData, vmin=0.0, cmap="coolwarm", cbar_kws={"label":"Error |yResult - yStar| [C]"})
        ax = errorHeatMap.axes

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Error |yResult - yStar| [C] with Lambda: " + str(self.lambdaValue))
        plt.tight_layout()
        matplotlib.pyplot.show()