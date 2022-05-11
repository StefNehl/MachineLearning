import matplotlib.pyplot
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sbr
import pandas as pd
import GPy

from inference import Regression
from GaussDistribution import GaussDistribution
from enum import Enum

class KernelSetting(Enum):
    LinearKernel = "Linear Kernel"
    RBFWithGpu = "RBF with GPU"
    RBF = "RBF"
    Matern52 = "Matern 52"

class GaussianProcess(Regression):

    def __init__(self, trainStep):
        self.trainStep = trainStep

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

        for i in range(len(x)):
            newXVector = x[i]
            featureVector.append(newXVector)

        return featureVector

    def computeGaussianProcessRegression(self, kernelSetting:KernelSetting):
        X = np.vstack(([self.createFeatureVector(x) for x in self.trainSubsetInput]))
        Y = np.vstack(([y for y in self.trainSubsetOutput]))

        kernel = object
        ard = True
        self.settingsString = (f"Train Step: {self.trainStep}, Kernel: {kernelSetting.value}, ARD: {ard}")
        if kernelSetting == KernelSetting.LinearKernel:
            kernel = GPy.kern.Linear(2, ARD=ard)

        if kernelSetting == KernelSetting.RBF:
            kernel = GPy.kern.RBF(2, ARD=ard, useGPU=False)

        if kernelSetting == KernelSetting.RBFWithGpu:
            kernel = GPy.kern.RBF(2, ARD=ard, useGPU=True)

        if kernelSetting == KernelSetting.Matern52:
            kernel = GPy.kern.Matern52(2, ARD=ard)

        self.model = GPy.models.GPRegression(X=X, Y=Y, kernel=kernel)
        self.model.optimize(messages=True, max_iters=1000)

        return self.model

    def testModel(self):
        self.yResult = self.model.Y

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
        plt.title("Error |yResult - yStar| [C] with " + self.settingsString)
        plt.tight_layout()
        matplotlib.pyplot.show()

    def plotHeatMap(self):

        GPy.plotting.change_plotting_library("matplotlib")
        fig = self.model.plot()

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(self.settingsString)
        matplotlib.pyplot.show()
        return