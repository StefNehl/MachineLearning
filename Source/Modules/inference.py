import csv
import numpy as np
import matplotlib.pyplot as plt
import math

from abc import ABC, abstractmethod

class ContiniousDustribution():

    def __init__(self):
        self.dataSet = []
        self.mean = None
        self.median = None
        self.variance = None
        self.standardDeviation = None
        self.probabilityDensity = None

    def importCsv(self, filename):
        if len(self.dataSet) != 0:
            raise Exception("Data already added")

        with open(filename, mode="r") as file:
            csvFile = csv.reader(file)

            for row in csvFile:
                self.dataSet.append(row)

    def exportCsv(self, filename):
        if len(self.dataSet) == 0:
            raise Exception("No Data added")

        with open(filename, mode="w") as file:
            csvWriter = csv.writer(file, delimiter =  ";")
            csvWriter.writerows(self.dataSet)

    def calculateMean(self):
        self.mean = np.mean(self.dataSet)

    def getMean(self):
        return self.mean

    def calculateVariance(self):
        length = len(self.dataSet)
        mean = self.getMean()

        squareDeviations = [(x - mean) ** 2 for x in self.dataSet]

        # Bessel's correction (n-1) instead of n for better results
        self.variance = sum(squareDeviations) / (length - 1)
        return self.variance

    def getVariance(self):
        return self.variance

    def calculateStandardDeviation(self):
        self.standardDeviation = math.sqrt(self.variance)
        return self.standardDeviation

    def getStandardDeviation(self):
        return self.standardDeviation

    def calculateProbabilityDensity(self, values, bins):
        probabilityDensity, binEdges = np.histogram(values,bins=bins, density=True)
        return probabilityDensity, binEdges

    @abstractmethod
    def generateSampels(self, numberOfSamples):
        pass

    @abstractmethod
    def plotData(self):
        pass

class GaussDistribution(ContiniousDustribution):

    def __init__(self, dimension, fileName = None, numberOfSamplesToGenerate = None, mean = None, variance = None):
        if((fileName is not None) & (numberOfSamplesToGenerate is not None)):
            raise Exception("Can't load data and generate samples")

        ContiniousDustribution.__init__(self)
        self.dimension = dimension

        if(fileName is not None):
            self.importCsv(fileName)
            self.calculateMean()
            self.calculateVariance()

        if(numberOfSamplesToGenerate is not None) & \
                (mean is not None) & \
                (variance is not None):
            self.numberOfPoints = numberOfSamplesToGenerate
            self.mean = mean
            self.variance = variance
            self.generateSampels()

        if(len(self.dataSet) == 0):
            raise Exception("Could not generate data, verify parameters")

        self.calculateStandardDeviation()
        #self.calculateProbabilityDensity()
        self.gaussen = []
        self.generateGaussen()

    def generateSampels(self):
        if len(self.dataSet) != 0:
            raise Exception("Data already added")

        self.dataSet =  np.random.default_rng().normal(self.mean, self.variance, size=(self.numberOfPoints, self.dimension))

    def generateGaussen(self):
        if len(self.dataSet) == 0:
            raise Exception("No Data added")

        if self.dimension == 1:
            return self.generateGaussen1D()

        return self.generateGaussen2D()

    def generateGaussen1D(self):
        vectorArray = np.array(self.dataSet)
        mean = self.getMean()
        variance = self.getVariance()

        self.gaussen = []

        for x in vectorArray:
            exponentialTerm = (-(1/(2 * variance**2)) * (x-mean)**2)
            denominator = (2 * math.pi * variance**2)**(0.5)
            result = (1/denominator) * math.e**(exponentialTerm)
            self.gaussen.append(result)

    def generateGaussen2D(self):
        vectorArray = np.array(self.dataSet)
        mean = self.getMean()

        resultArray = []

        for vector in vectorArray:
            xs = [vector[0], mean[0]]
            ys = [vector[1], mean[1]]
            # covariance = np.mean(vector[0], vector[1], axis=1) * np.mean(vector[0], axis=1) * np.mean(vector[1], axis=1)
            covariance = np.cov(xs, ys, bias=True)
            # covariance = np.mean(vector[0] * vector[1], axis=1) - np.mean(vector[0], axis=1) * np.mean(vector[1], axis=1)

            inverseCovariance = np.linalg.inv(covariance)
            determinantCovariance = np.linalg.det(covariance)

            exponentialTerm = (-0.5 * np.transpose(vector - mean)) * inverseCovariance * (vector - mean)
            denominator = (2 * math.pi) ** (self.dimension/2) * determinantCovariance ** (0.5)
            result = (1/denominator) * math.e**(exponentialTerm)

        return resultArray



    def plotData(self):
        if len(self.dataSet) == 0:
            raise Exception("No Data added")

        mean = self.getMean()
        variance = self.getVariance()

        title = f"Gaus Distribution with [$\mu$] = {mean} and {variance}"
        if self.dimension == 1:
            self.plotData1D(title, self.dataSet, self.gaussen)
        else:
            self.plotData2D(title, self.dataSet)

    def plotData1D(self, title, dataSet, result):
        plotRange = range(len(dataSet))

        plt.figure(figsize=(8, 6))

        plt.subplot(2, 1, 1)
        plt.hist(dataSet, bins=60, density=True, label="Histogram")

        plt.title("Distribution")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.scatter(plotRange, dataSet, label="Data", s=2)
        plt.plot(plotRange, result, color="y")


        plt.title(f"Raw data with n = {len(plotRange)} sample points")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend(loc="best")

        plt.suptitle(title)

        plt.tight_layout()
        plt.show()

    def plotData2D(self, title, dataSet, result):
        plt.figure(figsize=(8, 12))


        hist, xedges, yedges = np.histogram2d(dataSet[:,0], dataSet[:,1], bins=60)

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
        plt.title("Distribution")
        plt.xlabel("X")
        plt.ylabel("Y")
        #plt.legend(loc="upper right")

        ax = plt.subplot(2, 1, 2, projection="3d")
        plotRange = range(len(dataSet))
        ax.scatter3D(plotRange, dataSet[:, 0], dataSet[:, 1], label="Data", s=2)

        plt.title(f"Raw data with n = {len(plotRange)} sample points")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend(loc="best")

        plt.suptitle(title)

        plt.tight_layout()
        plt.show()

class BetaDistribution(ContiniousDustribution):

    def __init__(self, a, b, fileName = None, numberOfSamplesToGenerate = None):
        if((fileName is not None) & (numberOfSamplesToGenerate is not None)):
            raise Exception("Can't load data and generate samples")

        if((fileName is None) & (numberOfSamplesToGenerate is None)):
            raise Exception("No parameters for data")

        ContiniousDustribution.__init__(self)
        self.a = a
        self.b = b
        self.generatedBetaDistribution = []

        if(numberOfSamplesToGenerate is not None):
            self.generateSampels(numberOfSamplesToGenerate)

        if (fileName is not None):
            self.importCsv(fileName)

        self.generateBetaDistribution()

    def generateSampels(self, numberOfPoints):
        if len(self.dataSet) != 0:
            raise Exception("Data already added")

        self.dataSet = np.random.default_rng().beta(self.a, self.b, size=numberOfPoints)

    def generateBetaDistribution(self):
        betaFunction = (math.gamma(self.a + self.b) / (math.gamma(self.a) + math.gamma(self.b)))

        self.generatedBetaDistribution = []

        for x in self.dataSet:
            result = betaFunction * pow(x,(self.a-1)) * pow((1 - x), (self.b - 1))
            self.generatedBetaDistribution.append(result)

    def plotData(self):
        plotRange = range(len(self.dataSet))
        pdf, binEdges = self.calculateProbabilityDensity(self.dataSet, 60)

        plt.figure(figsize=(8, 6))

        plt.subplot(3, 1, 1)
        plt.hist(self.dataSet, bins=60, density=True, label="Histogram")
        plt.plot(binEdges[1:], pdf)

        plt.title("Distribution")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 2)
        plt.scatter(plotRange, self.dataSet, label="Data", s=2)
        plt.plot(plotRange, self.generatedBetaDistribution, "y-")

        plt.title(f"Raw data with n = {len(plotRange)} sample points")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend(loc="best")

        plt.subplot(3, 1, 3)
        pdf, binEdges = self.calculateProbabilityDensity(self.generatedBetaDistribution, 60)
        plt.plot(binEdges[1:], pdf, "y-")


        plt.title(f"Raw data with n = {len(plotRange)} sample points")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend(loc="best")

        plt.suptitle("Test")

        plt.tight_layout()
        plt.show()