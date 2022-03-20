"""
@author: Nehl Stefan
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from inference import ContiniousDustribution

class BetaDistribution(ContiniousDustribution):

    def __init__(self, a, b, fileName = None, numberOfSamplesToGenerate = None):
        if((fileName is not None) & (numberOfSamplesToGenerate is not None)):
            raise Exception("Can't load data and generate samples")

        if((fileName is None) & (numberOfSamplesToGenerate is None)):
            raise Exception("No parameters for data")

        ContiniousDustribution.__init__(self)
        self.a = a
        self.b = b
        self.calculateAndSetBFromAAndB()
        self.generatedBetaDistribution = []

        if(numberOfSamplesToGenerate is not None):
            self.numberOfSamples = numberOfSamplesToGenerate
            self.generateSampels()

        if (fileName is not None):
            self.importCsv(fileName)
            self.numberOfSamples = len(self.dataSet)

        self.calculateMean()
        self.calculateVariance()
        self.calculateStandardDeviation()
        self.generateBetaDistribution()

    def generateSampels(self):
        if len(self.dataSet) != 0:
            raise Exception("Data already added")

        self.dataSet = np.random.default_rng().beta(self.a, self.b, size=self.numberOfSamples)

    def calculateAndSetBFromAAndB(self):
        self.bFromAAndB = (math.gamma(self.a + self.b) / (math.gamma(self.a) + math.gamma(self.b)))

    def calculateBeta(self, x):
        return self.bFromAAndB * pow(x, (self.a - 1)) * pow((1 - x), (self.b - 1))

    def generateBetaDistribution(self):
        self.generatedBetaDistribution = []

        for x in self.dataSet:
            result = self.calculateBeta(x)
            self.generatedBetaDistribution.append(result)

    def plotData(self):
        plotRange = range(len(self.dataSet))
        x = np.linspace(0.01, 0.99, self.numberOfSamples)
        y = self.calculateBeta(x)

        plt.figure(figsize=(6, 6))

        plt.subplot(2, 1, 1)
        plt.hist(self.dataSet, bins=60, density=True, label="Histogram")
        plt.plot(x, y, "r-", linewidth=1, label=f"alpha = {self.a}, beta = {self.b}")

        plt.title("Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.scatter(plotRange, self.dataSet, label="Data", s=2)

        plt.title("Raw Data")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend(loc="upper right")

        plt.suptitle(f"Beta Distribution with alpha = {self.a} and beta = {self.b}")

        plt.tight_layout()
        plt.show()

    def plotDataWithDifferentAlphasAndBetas(self, alphaAndBetas):
        if len(alphaAndBetas) == 0:
            return

        plt.figure(figsize=(4, 4))
        plt.title("Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        for alphaAndBeta in alphaAndBetas:
            self.a = alphaAndBeta[0]
            self.b = alphaAndBeta[1]
            self.calculateAndSetBFromAAndB()

            x = np.linspace(0.01, 0.99, self.numberOfSamples)
            y = self.calculateBeta(x)
            plt.plot(x, y, linewidth=1, label=f"alpha = {self.a}, beta = {self.b}")

        plt.legend(loc = "best")

        plt.tight_layout()
        plt.show()


