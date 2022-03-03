import math

class BasicStatistics:

    def checkDataSet(self, dataSet):
        if dataSet == None:
            return False

        if len(dataSet) == 0:
            return False

        return True

    def getMean(self, dataSet):
        if self.checkDataSet(dataSet) == False:
            return None

        length = len(dataSet)
        sumValue = sum(dataSet)
        mean = sumValue / length
        return mean


    def getMedian(self, dataSet):
        if self.checkDataSet(dataSet) == False:
            return None

        length = len(dataSet)
        dataSet.sort()
        mid = length // 2

        if length % 2 == 0:
            median1 = dataSet[mid]
            median2 = dataSet[mid - 1]
            median = (median1 + median2) / 2
        else:
            median = dataSet[mid]

        return median


    def getVariance(self, dataSet):
        if self.checkDataSet(dataSet) == False:
            return None

        length = len(dataSet)
        mean = self.getMean(dataSet)

        squareDeviations = [(x - mean) ** 2 for x in dataSet]
        variance = sum(squareDeviations) / (length - 1)
        return variance


    def getStandardDeviation(self, dataSet):
        if self.checkDataSet(dataSet) == False:
            return None

        variance = self.getVariance(dataSet)
        standardDeviation = math.sqrt(variance)
        return standardDeviation

    def findMinValue(self, dataSet):
        if self.checkDataSet(dataSet) == False:
            return None

        dataSetSorted = dataSet[:]
        dataSetSorted.sort()
        return dataSetSorted[0]


    def findMaxValue(self, dataSet):
        if self.checkDataSet(dataSet) == False:
            return None

        dataSetSorted = dataSet[:]
        dataSetSorted.sort()
        return dataSetSorted[len(dataSetSorted) - 1]

    def normalizeDataSet(self, dataSet):
        min = self.findMinValue(dataSet)
        max = self.findMaxValue(dataSet)

        dataNorm = [(i - min) / (max - min) for i in dataSet]
        return dataNorm

    def standardizeDataSet(self, dataSet):
        mean = self.getMean(dataSet)
        standardDeviation = self.getStandardDeviation(dataSet)
        standardizeDataSet = [((x - mean)/standardDeviation) for x in dataSet]

        return standardizeDataSet