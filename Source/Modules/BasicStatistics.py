"""
@author: Nehl Stefan
"""

import math

class BasicStatistics:

    def __init__(self, dataSet):
        if self.checkDataSet(dataSet) == False:
            return None

        #clone dataSet
        self.dataSet = dataSet[:]
        self.sortedDataSet = dataSet[:]
        self.sortedDataSet.sort()

    def checkDataSet(self, dataSet):
        if dataSet is None:
            return False

        if len(dataSet) == 0:
            return False

        return True

    def getMean(self):
        length = len(self.dataSet)
        sumValue = sum(self.dataSet)
        mean = sumValue / length
        return mean

    def getMedian(self):
        length = len(self.dataSet)
        mid = length // 2

        if length % 2 == 0:
            median1 = self.sortedDataSet[mid]
            median2 = self.sortedDataSet[mid - 1]
            median = (median1 + median2) / 2
        else:
            median = self.sortedDataSet[mid]

        return median

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

    def getMinValue(self):
        return self.sortedDataSet[0]

    def getMaxValue(self):
        return self.sortedDataSet[len(self.sortedDataSet) - 1]

    def getNormalizeDataSetOld(self):
        min = self.getMinValue()
        max = self.getMaxValue()

        dataNorm = [(i - min) / (max - min) for i in self.dataSet]
        return dataNorm

    def getNormalizeDataSet(self):
        mean = self.getMean()
        standardDeviation = self.getStandardDeviation()
        standardizeDataSet = [((x - mean)/standardDeviation) for x in self.dataSet]

        return standardizeDataSet