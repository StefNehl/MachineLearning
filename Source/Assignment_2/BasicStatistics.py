

class BasicStatistics:

    def getMean(self, dataSet):
        if dataSet == None:
            return None;

        length = len(dataSet)
        if length == 0:
            return None

        sum = sum(dataSet)
        mean = sum / length
        return mean


    def getMedian(self, dataSet):
        if dataSet == None:
            return None;

        length = len(dataSet)
        if length == 0:
            return None

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
        return 0

    def getStandardDeviation(self, dataSet):
        return 0

    def standardizeDataSet(self, dataSet):
        return dataSet