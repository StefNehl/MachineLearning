import csv
import numpy as np
import matplotlib.pyplot as plt
import math

from abc import ABC, abstractmethod

class ContiniousDustribution():

    @abstractmethod
    def importCsv(self, filename):
        pass

    @abstractmethod
    def exportCsv(self, filename):
        pass

    @abstractmethod
    def calculateMean(self):
        pass

    @abstractmethod
    def calculateVariance(self):
        pass

    @abstractmethod
    def calculateStandardDeviation(self):
        pass

    @abstractmethod
    def normalizeDataSet(self):
        pass

    @abstractmethod
    def generateSampels(self):
        pass

    @abstractmethod
    def plotData(self):
        pass

class Regression():

    @abstractmethod
    def importData(self):
        pass

    @abstractmethod
    def generateTrainingSubset(self):
        pass

    @abstractmethod
    def computeLinearRidgeRegression(self, lambdaValue):
        pass

    @abstractmethod
    def testModel(self, weight):
        pass

    @abstractmethod
    def computeError(self, yStar):
        pass

    @abstractmethod
    def plotError(self):
        pass

    @abstractmethod
    def plotHeatMap(self):
        pass

    @abstractmethod
    def computMeanOfError(self):
        pass



