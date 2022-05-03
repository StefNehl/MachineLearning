"""
@author: Nehl Stefan
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from ContiniousDustribution import ContiniousDustribution

class GaussDistribution(ContiniousDustribution):

    def __init__(self, dimension, fileName = None, numberOfSamplesToGenerate = None, dataArray = None, mean = None, variance = None):
        if((fileName is not None) & (numberOfSamplesToGenerate is not None) & (dataArray is not None)):
            raise Exception("Can't load data and generate samples")

        ContiniousDustribution.__init__(self)
        self.dimension = dimension

        if(fileName is not None):
            self.importCsv(fileName)
            self.numberOfSamples = len(self.dataSet)
            self.calculateMean()
            self.calculateVariance()

        if(dataArray is not None):
            self.dataSet = dataArray
            self.numberOfSamples = len(self.dataSet)
            self.calculateMean()
            self.calculateVariance()

        if(numberOfSamplesToGenerate is not None) & \
                (mean is not None) & \
                (variance is not None):
            self.numberOfSamples = numberOfSamplesToGenerate
            self.mean = np.array(mean)
            self.variance = np.array(variance)
            self.generateSampels()

        if(len(self.dataSet) == 0):
            raise Exception("Could not generate data, verify parameters")

        self.calculateStandardDeviation()
        self.gaussenDistribution = []
        self.generateGaussen()

    def generateSampels(self):
        if len(self.dataSet) != 0:
            raise Exception("Data already added")

        self.dataSet =  np.random.default_rng().normal(self.mean, self.variance, size=(self.numberOfSamples, self.dimension))

    def generateGaussen(self):
        if len(self.dataSet) == 0:
            raise Exception("No Data added")

        if self.dimension == 1:
            return self.generateGaussen1D()

        return self.generateGaussen2D()

    def calculateMean(self):
        if self.dimension == 1:
            self.mean = np.mean(self.dataSet)
        elif self.dimension == 2:
            self.mean = np.array(np.mean(self.dataSet, axis=0))
        else:
            raise Exception("Not implemented yet")


    def calculateStandardDeviation(self):
        if self.dimension == 1:
            self.standardDeviation = np.std(self.dataSet)
        else:
            self.standardDeviation = np.array((math.sqrt(self.mean[0]), math.sqrt(self.mean[1])))

    def calculateGaussen1D(self, x):
        exponentialTerm = (-(1 / (2 * self.variance ** 2)) * (x - self.mean) ** 2)
        denominator = (2 * math.pi * self.variance ** 2) ** (0.5)
        return (1 / denominator) * math.e ** (exponentialTerm)

    def generateGaussen1D(self):
        vectorArray = np.array(self.dataSet)
        self.gaussen = []

        for x in vectorArray:
            self.gaussen.append(self.calculateGaussen1D(x))

    def calculateGaussen2D(self, vector):
        xs = [self.mean[0], 0]
        ys = [0, self.mean[1]]
        covariance = [xs, ys]
        inverseCovariance = np.linalg.inv(covariance)
        determinantCovariance = np.linalg.det(covariance)

        # exponentialTerm = (-0.5 * np.transpose(vector - self.mean)) * inverseCovariance * (vector - self.mean)
        exponentialTerm = -(np.linalg.solve(covariance, (vector - self.mean)).T.dot((vector - self.mean))) / 2
        denominator = ((2 * math.pi) ** (self.dimension / 2)) * determinantCovariance ** (0.5)
        result =  (1 / denominator) * math.exp(exponentialTerm)
        return result

    def generateGaussen2D(self):
        vectorArray = np.array(self.dataSet)

        self.gaussenDistribution = []

        for vector in vectorArray:
            self.gaussenDistribution.append(self.calculateGaussen2D(vector))

    def plotData(self):
        if len(self.dataSet) == 0:
            raise Exception("No Data added")

        if self.dimension == 1:
            self.plotData1D()
        else:
            self.plotData2D()

    def plotData1D(self):
        plotRange = range(len(self.dataSet))
        x = np.linspace(min(self.dataSet), max(self.dataSet), self.numberOfSamples)
        y = self.calculateGaussen1D(x)

        plt.figure(figsize=(6, 6))

        plt.subplot(2, 1, 1)
        plt.hist(self.dataSet, bins=60, density=True, label="Histogram")
        plt.plot(x, y, "r-", linewidth=1, label="Distribution")

        plt.title("Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.scatter(plotRange, self.dataSet, label="Data", s=2)

        plt.title(f"Raw data with n = {self.numberOfSamples} sample points")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend(loc="best")

        plt.suptitle(f"Gaus Distribution with $\mu$ = {self.mean} and $\sigma$ = {self.variance}")

        plt.tight_layout()
        plt.show()

    def plotData2D(self):
        plt.figure(figsize=(8, 12))

        hist, xedges, yedges = np.histogram2d(self.dataSet[:,0], self.dataSet[:,1], bins=60)

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the 16 bars.
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()

        ax = plt.subplot(2, 1, 1, projection="3d")
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
        ax.set_zlabel("Frequency")

        xy = np.linspace([min(self.dataSet[0]),min(self.dataSet[1])], [max(self.dataSet[0]), max(self.dataSet[1])], self.numberOfSamples)
        # y = np.linspace(min(self.dataSet[1]), max(self.dataSet[1]), self.numberOfSamples)
        z = np.array([self.calculateGaussen2D(v) for v in xy])
        # does not work
        # ax.plot_surface(xy[0], xy[1], z, )

        plt.title("Distribution")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="best")

        ax = plt.subplot(2, 1, 2, projection="3d")
        plotRange = range(self.numberOfSamples)
        ax.scatter3D(plotRange, self.dataSet[:, 0], self.dataSet[:, 1], label="Data", s=2)

        plt.title(f"Raw data with n = {self.numberOfSamples} sample points")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend(loc="best")

        plt.suptitle(f"Gaus Distribution with $\mu$ = {self.mean} and $\sigma$ = {self.variance}")

        plt.tight_layout()
        plt.show()