import matplotlib.pyplot
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sbr
import pandas as pd

from inference import Regression
from GaussDistribution import GaussDistribution
from BasicStatistics import BasicStatistics

class RidgeRegression(Regression):

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

        normalizedY_TestValues = np.matrix(BasicStatistics(dataDictonary.get("y_test")[0]).getNormalizeDataSet())
        self.y_test = np.transpose(np.array(normalizedY_TestValues)) # Testdata from DataSet

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

        normalizedYValues =  BasicStatistics(arrayValues)
        self.outputValues = np.array(normalizedYValues.getNormalizeDataSet())
        self.numberOfSamples = len(self.inputValues)

    def generateTrainingSubset(self):
        self.trainSubsetInput = np.array(self.inputValues[0:len(self.inputValues):self.trainStep])
        self.trainSubsetOutput = np.array(self.outputValues[0:len(self.outputValues):self.trainStep])

    def createFeatureVector(self, x):
        featureVector = []
        featureVector.append(1)

        for i in range(len(x)):
            newXVector = x[i]
            featureVector.append(newXVector)

        return featureVector

    def computeLinearRidgeRegression(self, lambdaValue):
        self.lambdaValue = lambdaValue
        X = np.vstack(([self.createFeatureVector(x) for x in self.trainSubsetInput]))
        Y = np.vstack(([y for y in self.trainSubsetOutput]))

        XT = np.transpose(X)
        XTX = np.matmul(XT, X) + self.lambdaValue * np.identity(X.shape[1])
        self.weightVector = np.matmul(np.matmul(np.linalg.inv(XTX), XT), Y)
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
        self.settingsString = (f"Train Step: {self.trainStep}, "
                               f"Lambda: {self.lambdaValue}, "
                               f"Kernel: Ridge Regression, "
                               f"Mean Error: {round(self.meanError, 5)}")

    def getMeanError(self):
        return self.meanError

    def getSettingsString(self):
        return self.settingsString

    def plotError(self):
        plt.figure(figsize=(8, 6))
        errorPlot = sbr.barplot(data=self.errorDataFrame, x="values", y="error", palette="coolwarm_r")

        plt.xlabel("Descending Sorted Y Errors")
        plt.ylabel("Error |yResult - yStar| [C]")
        plt.title(self.settingsString)
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
        plt.title(self.settingsString)
        plt.tight_layout()
        matplotlib.pyplot.show()


