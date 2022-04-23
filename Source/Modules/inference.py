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





