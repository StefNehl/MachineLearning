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
    def plotData(self):
        pass


class GaussDistribution(ContiniousDustribution):


    def __init__(self, dimension):
        self.dimension = dimension
        self.dataSet = []
        self.mean = None
        self.median = None

    def importCsv(self, filename):
        with open(filename, mode='r') as file:
            csvFile = csv.reader(file)

            for row in csvFile:
                self.dataSet.append(row)

    def exportCsv(self, filename):
        with open(filename, mode='w') as file:
            csvWriter = csv.writer(file, delimiter =  ';')
            csvWriter.writerows(self.dataSet)

    def getMean(self):
        length = len(self.dataSet)
        sumValue = sum(self.dataSet)
        mean = sumValue / length
        return mean

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

        self.dataSet = np.random.default_rng().normal(mean, variance, numberOfPoints)

    def plotData(self):

        mean = self.getMean()
        variance = self.getVariance()

        plt.figure(figsize=(8, 6))

        plt.subplot(2, 1, 1)
        plt.hist(self.dataSet, bins=60, density=True, label='Histogram')
        # plt.axvline(self.getMean(), label='Mean', color='r', ls='--')
        # plt.axvline(standardizedMedian, label='Median', color='y', ls='--')
        # plt.axvline(self.getMean() + self.getStandardDeviation(), label='Standard Deviation', color='g', ls='--')
        # plt.axvline(self.getMean() - self.getStandardDeviation(), color='g', ls='--')
        plt.title("Distribution")
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plotRange = range(len(self.dataSet))
        plt.scatter(plotRange, self.dataSet, label='Data', s=2)
        # plt.axhline(self.getMean(), label='Mean', color='r', ls='--')
        # plt.axhline(median, label='Median', color='y', ls='--')
        # plt.axhline(self.getMean() + self.getStandardDeviation(), label='Standard Deviation', color='g', ls='--')
        # plt.axhline(self.getMean() - self.getStandardDeviation(), color='g', ls='--')

        plt.title(f"Raw data with n = {len(plotRange)} sample points")
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend(loc="best")

        plt.suptitle(f'Gaus Distribution with [$\mu$] = {mean} and {variance}')

        plt.tight_layout()
        plt.show()
