import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

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

        #init Gauss Distribution
        #self.gaussDistr = GaussDistribution(2, dataArray=self.inputValues)

    def generateTestDate1D(self):
        self.hasGenerated1DTestData = True
        c2 = 0.01
        c1 = 1.3
        c0 = 3.456
        self.inputValues = np.linspace(-10.0, 20.2, 200)
        self.outputValues = c1 * self.inputValues ** 2 + c1 * self.inputValues + c0 + 500.0 * np.random.rand(len(self.inputValues))
        self.gaussDistr = GaussDistribution(1, dataArray=self.inputValues)

    def generateTrainingSubset(self):
        self.trainSubsetInput = np.array(self.inputValues[0:len(self.inputValues):self.trainStep])
        self.trainSubsetOutput = np.array(self.outputValues[0:len(self.outputValues):self.trainStep])


    def createPolynomialFeatureVector(self, xVector):
        featureVector = []

        featureVector.append(1)

        for i in range(1, self.order):
            newXVector = self.calculateGaussBasisFunction(xVector) ** i
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
        variance = np.var(x)

        exponent = -(1 / (2 * variance ** 2)) * (x - self.mu) ** 2
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

        self.mu = [np.mean(self.inputValues[:,0]),np.mean(self.inputValues[:,1])]

        X = np.vstack(([self.createPolynomialFeatureVector(x) for x in self.inputValues]))
        Y = np.vstack(([y for y in self.outputValues]))

        XT = np.transpose(X)
        XTX = np.matmul(XT, X) + lambdaValue * np.identity(X.shape[1])
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
        y = self.weightVector[0]
        for w in range(1,self.weightVector.shape[0]):
            y = y + (xVector * self.weightVector[w])**(w)

        return y

    def plot2DData(self):
        ys = self.weightVector[0]
        for w in range(1,self.weightVector.shape[0]):
            ys = ys + (self.trainSubsetInput * self.weightVector[w])**(w)

        ys = np.array(ys)

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)

        plt.title("Distribution")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()

        plt.subplot(2, 2, 3)
        hm = plt.imshow(self.tempData,cmap='Reds', interpolation='none',
                        extent=[0,400,0,400])

        plt.colorbar(hm)
        plt.title(f"Raw data")
        plt.xlabel("Latitude coordinate")
        plt.ylabel("Longitude coordinate")

        plt.subplot(2, 2, 4)
        hm = plt.imshow(self.tempData,cmap='Reds', interpolation='none',
                        extent=[0,400,0,400])

        plt.colorbar(hm)
        plt.title(f"Train Subset")
        plt.xlabel("Latitude coordinate")
        plt.ylabel("Longitude coordinate")

        plt.tight_layout()
        plt.show()
