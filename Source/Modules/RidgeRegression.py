import matplotlib.pyplot
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sbr
import pandas as pd

from inference import Regression
from GaussDistribution import GaussDistribution

class RidgeRegression(Regression):

    def __init__(self, trainStep, order):
        self.trainStep = trainStep
        self.hasGenerated1DTestData = False
        self.order = order

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

    def plotData(self):
        self.plot2DData()

    def plot2DData(self):

        ys = np.hstack([x @ self.weightVector for x in self.x_test])

        self.y_error = np.transpose(abs(ys - self.y_test))
        x_error = range(self.y_error.shape[0])

        tempErrorData = \
            {
                'Lat': [round(long,2) for long in self.x_test[:, 1]],
                'Long': [round(lat,2) for lat in self.x_test[:, 2]],
                'error':[error[0] for error in self.y_error]
            }

        tempDataFrame = pd.DataFrame(tempErrorData)
        tempDataFrame = tempDataFrame.pivot("Long", "Lat", "error")
        reversedTempErrorData = tempDataFrame.sort_values(("Long"), ascending=False)

        reversedArray = np.flip(np.sort(self.y_error,0))
        errorDataFrame = pd.DataFrame({
            'values':range(len(reversedArray)),
            'error [°C]':[error[0] for error in reversedArray]
        })

        plt.figure(figsize=(8, 6))
        errorPlot = sbr.barplot(data=errorDataFrame, x="values", y="error [°C]", palette="coolwarm_r")

        plt.title("Error with lambda: " + str(self.lambdaValue))
        plt.tight_layout()
        matplotlib.pyplot.show()


        def fmt(x, y):
            return '{:,.2f}'.format(x)

        plt.figure(figsize=(8,6))
        errorHeatMap = sbr.heatmap(reversedTempErrorData, vmin=0.0, cmap="coolwarm", cbar_kws={"label":"[°C]"})
        ax = errorHeatMap.axes

        plt.title("Error with lambda: " + str(self.lambdaValue))
        plt.tight_layout()
        matplotlib.pyplot.show()
