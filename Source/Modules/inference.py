import csv
import numpy as np
import matplotlib.pyplot as plt
import math

from abc import ABC, abstractmethod

class ContiniousDustribution():

    @abstractmethod
    def importCsv(self):
        pass  # supresses compile errors

    @abstractmethod
    def exportCsv(self):
        pass

    @abstractmethod
    def getMean(self):
        pass

    @abstractmethod
    def getStandardDeviation(self):
        pass

    @abstractmethod
    def generateSampels(self):
        pass

    @abstractmethod
    def plotData(self, result):
        pass

    def plotData1D(self, title, dataSet, result):
        plotRange = range(len(dataSet))

        plt.figure(figsize=(8, 6))

        plt.subplot(2, 1, 1)
        plt.hist(dataSet, bins=60, density=True, label='Histogram')

        plt.title("Distribution")
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.scatter(plotRange, dataSet, label='Data', s=2)
        plt.plot(plotRange, result, color='y')


        plt.title(f"Raw data with n = {len(plotRange)} sample points")
        plt.xlabel('Sample')
        plt.ylabel('Value')
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

        ax = plt.subplot(2, 1, 1, projection='3d')
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
        ax.set_zlabel('Frequency')
        plt.title("Distribution")
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.legend(loc="upper right")

        ax = plt.subplot(2, 1, 2, projection='3d')
        plotRange = range(len(dataSet))
        ax.scatter3D(plotRange, dataSet[:, 0], dataSet[:, 1], label='Data', s=2)

        plt.title(f"Raw data with n = {len(plotRange)} sample points")
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend(loc="best")

        plt.suptitle(title)

        plt.tight_layout()
        plt.show()


class GaussDistribution(ContiniousDustribution):

    def __init__(self, dimension):
        self.dimension = dimension
        self.dataSet = []
        self.mean = None
        self.median = None

    def importCsv(self, filename):
        if len(self.dataSet) != 0:
            raise Exception('Data already added')

        with open(filename, mode='r') as file:
            csvFile = csv.reader(file)

            for row in csvFile:
                self.dataSet.append(row)

    def exportCsv(self, filename):
        if len(self.dataSet) == 0:
            raise Exception('No Data added')

        with open(filename, mode='w') as file:
            csvWriter = csv.writer(file, delimiter =  ';')
            csvWriter.writerows(self.dataSet)

    def getMean(self):
        return np.mean(self.dataSet, 0)

    def getVariance(self):
        length = len(self.dataSet)
        mean = self.getMean()

        squareDeviations = [(x - mean) ** 2 for x in self.dataSet]

        #Bessel's correction (n-1) instead of n for better results
        variance = sum(squareDeviations) / (length-1)
        return variance

    def getStandardDeviation(self):
        variance = self.getVariance()
        standardDeviation = math.sqrt(variance)
        return standardDeviation

    def generateSampels(self, mean, variance, numberOfPoints):
        if len(self.dataSet) != 0:
            raise Exception('Data already added')

        self.dataSet =  np.random.default_rng().normal(mean, variance, size=(numberOfPoints, self.dimension))

    def generateGaussen(self):
        if len(self.dataSet) == 0:
            raise Exception('No Data added')


        if self.dimension == 1:
            return self.generateGaussen1D()

        return self.generateGaussen2D()

    def generateGaussen1D(self):
        vectorArray = np.array(self.dataSet)
        mean = self.getMean()
        variance = self.getVariance()

        resultArray = []

        for x in vectorArray:
            exponentialTerm = (-(1/(2 * variance**2)) * (x-mean)**2)
            denominator = (2 * math.pi * variance**2)**(0.5)
            result = (1/denominator) * math.e**(exponentialTerm)
            resultArray.append(result)

        return resultArray

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

    def plotData(self, result):
        if len(self.dataSet) == 0:
            raise Exception('No Data added')

        mean = self.getMean()
        variance = self.getVariance()

        title = f'Gaus Distribution with [$\mu$] = {mean} and {variance}'
        if self.dimension == 1:
            self.plotData1D(title, self.dataSet, result)
        else:
            self.plotData2D(title, self.dataSet)

