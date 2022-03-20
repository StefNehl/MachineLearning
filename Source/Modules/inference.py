import csv
import numpy as np
import matplotlib.pyplot as plt
import math

from abc import ABC, abstractmethod

class ContiniousDustribution():

    def __init__(self):
        self.dataSet = []
        self.normalizeDataSet = []
        self.mean = None
        self.median = None
        self.variance = None
        self.standardDeviation = None
        self.probabilityDensity = None
        self.binEdges = None

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

    def calculateVariance(self):
        length = len(self.dataSet)
        mean = self.mean

        squareDeviations = [(x - mean) ** 2 for x in self.dataSet]

        # Bessel's correction (n-1) instead of n for better results
        self.variance = sum(squareDeviations) / (length - 1)
        return self.variance

    def calculateStandardDeviation(self):
        self.standardDeviation = math.sqrt(self.variance)
        return self.standardDeviation

    def normalizeDataSet(self):
        self.dataSet = [((x - self.mean)/self.standardDeviation) for x in self.dataSet]

    @abstractmethod
    def generateSampels(self):
        pass

    @abstractmethod
    def plotData(self):
        pass





