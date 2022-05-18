"""
@author: Nehl Stefan
"""

from Modules.BasicStatistics import BasicStatistics

dataSet1 = [1, 2, 3, 4, 5]
dataSet2 = [1, 2, 3, 4, 5, 6]

basicStatistics1 = BasicStatistics(dataSet1)
basicStatistics2 = BasicStatistics(dataSet2)
result = -1

#Mean
print('Mean')
expectedResult = 3
result = basicStatistics1.getMedian()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 3.5
result = basicStatistics2.getMedian()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#Median
print('Median')
expectedResult = 3
result = basicStatistics1.getMedian()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 3.5
result = basicStatistics2.getMedian()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#Variance
print('Variance')
expectedResult = 2.5
result = basicStatistics1.getVariance()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 3.5
result = basicStatistics2.getVariance()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#StandardDefiation
print('Standard Defiation')
expectedResult = 1.5811388
result = basicStatistics1.getStandardDeviation()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = 1.8708287
result = basicStatistics2.getStandardDeviation()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#FindMinValue
print('Find MinValue')
expectedResult = 1
result = basicStatistics1.getMinValue()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

print('Find MinValue')
expectedResult = 1
result = basicStatistics2.getMinValue()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

#FindMaxValue
print('Find MaxValue')
expectedResult = 5
result = basicStatistics1.getMaxValue()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

print('Find MaxValue')
expectedResult = 6
result = basicStatistics2.getMaxValue()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

#Normalize Data
print('Standardized Data')
expectedResult = "??"
result = basicStatistics1.getNormalizeDataSetOld()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = "??"
result = basicStatistics2.getNormalizeDataSetOld()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()

#Standardized Data
print('Standardized Data')
expectedResult = "??"
result = basicStatistics1.getNormalizeDataSet()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))

expectedResult = "??"
result = basicStatistics2.getNormalizeDataSet()
print('Expected: ' + str(expectedResult) + ' Result: ' + str(result))
print()









