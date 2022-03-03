from BasicStatistics import BasicStatistics

dataSet1 = [1, 2, 3, 4, 5]
dataSet2 = [1, 2, 3, 4, 5, 6]

basicStatistics = BasicStatistics()
result = -1

#Mean
print('Mean')
expectedResult = 3
result = basicStatistics.getMedian(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 3.5
result = basicStatistics.getMedian(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#Median
print('Median')
expectedResult = 3
result = basicStatistics.getMedian(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 3.5
result = basicStatistics.getMedian(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#Variance
print('Variance')
expectedResult = 2.5
result = basicStatistics.getVariance(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 3.5
result = basicStatistics.getVariance(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#StandardDefiation
print('Standard Defiation')
expectedResult = 1.5811388
result = basicStatistics.getStandardDeviation(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 1.8708287
result = basicStatistics.getStandardDeviation(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#FindMinValue
print('Find MinValue')
expectedResult = 1
result = basicStatistics.findMinValue(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

print('Find MinValue')
expectedResult = 1
result = basicStatistics.findMinValue(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

#FindMaxValue
print('Find MaxValue')
expectedResult = 5
result = basicStatistics.findMaxValue(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

print('Find MaxValue')
expectedResult = 6
result = basicStatistics.findMaxValue(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

#Normalize Data
print('Standardized Data')
expectedResult = "??"
result = basicStatistics.normalizeDataSet(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = "??"
result = basicStatistics.normalizeDataSet(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#Standardized Data
print('Standardized Data')
expectedResult = "??"
result = basicStatistics.standardizeDataSet(dataSet1)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = "??"
result = basicStatistics.standardizeDataSet(dataSet2)
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()









