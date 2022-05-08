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

    def __init__(self, trainStep, trainingsIterations, order):
        self.trainStep = trainStep
        self.trainingsIteration = trainingsIterations
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
        self.mu = [np.mean(self.inputValues[:, 0]), np.mean(self.inputValues[:, 1])]

    def generateTestDate1D(self):
        self.hasGenerated1DTestData = True
        c2 = 0.01
        c1 = 1.3
        c0 = 3.456
        self.inputValues = np.linspace(-10.0, 20.2, 200)
        self.outputValues = c1 * self.inputValues ** 2 + c1 * self.inputValues + c0 + 500.0 * np.random.rand(len(self.inputValues))
        self.mu = np.mean(self.inputValues)
        self.calculateVariance1D()

    def calculateVariance1D(self):
        length = len(self.inputValues)
        mean = np.mean(self.inputValues)

        squareDeviations = [(x - mean) ** 2 for x in self.inputValues]

        # Bessel's correction (n-1) instead of n for better results
        self.variance = sum(squareDeviations) / (length - 1)

    def generateTrainingSubset(self):
        self.trainSubsetInput = np.array(self.inputValues[0:len(self.inputValues):self.trainStep])
        self.trainSubsetOutput = np.array(self.outputValues[0:len(self.outputValues):self.trainStep])


    def createPolynomialFeatureVector(self, x):
        featureVector = []

        featureVector.append(1)

        for i in range(1, self.order):
            newXVector = self.calculateGaussBasisFunction(x) ** i
            featureVector.append(newXVector)

        featureVector = np.array(featureVector)
        return np.transpose(featureVector)

    def calculateGaussBasisFunction(self, xVector):

        if(len(xVector.shape) == 0):
            return self.calculateGaussBasisFunction1D(xVector)
        else:
            return self.calculateGaussBasisFunction2D(xVector)

    def calculateGaussBasisFunction1D(self, x):
        return x
        exponent = -(1 / (2 * self.variance ** 2)) * (x - self.mu) ** 2
        result = np.exp(exponent)
        return result

    def calculateGaussBasisFunction2D(self, xVector):
        xs = [np.mean(xVector[0] - self.mu[0]), 0]
        ys = [0, np.mean(xVector[1] - self.mu[1])]
        covariance = [xs, ys]

        xMU = (xVector - self.mu)
        xMUT = np.transpose(xMU)
        covarianceInv = np.linalg.inv(covariance)

        exponent = (-0.5) * (xMUT @ covarianceInv @ xMU)
        result = np.e ** exponent
        return result

    def computeLinearRidgeRegression(self, lambdaValue):
        self.lambdaValue = lambdaValue
        X = np.vstack(([self.createPolynomialFeatureVector(x) for x in self.trainSubsetInput]))
        Y = np.vstack(([y for y in self.trainSubsetOutput]))

        XT = np.transpose(X)
        XTX = np.matmul(XT, X) + self.lambdaValue * np.identity(X.shape[1])
        self.weightVector = np.matmul(np.matmul(np.linalg.inv(XTX), XT), Y)
        return self.weightVector

    def plotData(self):
        if self.hasGenerated1DTestData:
            self.plot1DTestData()
        else:
            self.plot2DData()

    def plot1DTestData(self):
        ys = self.weightVector[0]
        for w in range(1,self.weightVector.shape[0]):
            ys = ys + (self.trainSubsetInput * self.weightVector[w])**(w)

        ys = np.array(ys)

        plt.figure(figsize=(8, 8))
        plt.scatter(self.inputValues, self.outputValues, label="RawData")
        plt.plot(self.trainSubsetInput, ys, '-r', label="Learned")

        plt.title("Distribution")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, xVector):
        featureVector = self.createPolynomialFeatureVector(xVector)
        y = np.transpose(featureVector) @ self.weightVector

        return y

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

        reversedArray = np.sort(self.y_error)[::-1]
        errorDataFrame = pd.DataFrame({
            'values':range(len(reversedArray)),
            'error [°C]':[error[0] for error in reversedArray]
        })

        plt.figure(figsize=(8, 6))
        errorPlot = sbr.relplot(data=errorDataFrame, kind="line", x="values", y="error [°C]")

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
