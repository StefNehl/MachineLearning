"""
@author: Nehl Stefan
"""

import csv

import matplotlib.pyplot as plt
from Modules.BasicStatistics import BasicStatistics

dataSet = []


with open('gauss.csv', mode='r') as file:
    csvFile = csv.reader(file)

    for line in csvFile:
        dataSet.append(float(line[0]))

basicStatistics = BasicStatistics(dataSet)

#dataSet = basicStatistics.normalizeDataSet(dataSet)
mean = basicStatistics.getMean()
median = basicStatistics.getMedian()
variance = basicStatistics.getVariance()
standardDeviation = basicStatistics.getStandardDeviation()
standardizedDataSet = basicStatistics.getNormalizeDataSet()

standardizedStatistics = BasicStatistics(standardizedDataSet)
standardizedMean = standardizedStatistics.getMean()
standardizedMedian = standardizedStatistics.getMedian()
standardizedStandardDeviation = standardizedStatistics.getStandardDeviation()


length = len(standardizedDataSet)
plotRange = range(length)

plt.suptitle('Data Distribution')

plt.figure(figsize=(8,6))
# plot histogram (left, right, top)
plt.subplot(2, 1, 1)

plt.hist(standardizedDataSet, bins=20, density=True, label='Histogram')
plt.axvline(standardizedMean, label='Mean', color='r', ls='--')
plt.axvline(standardizedMedian, label='Median', color='y', ls='--')
plt.axvline(standardizedMean - standardizedStandardDeviation, label='Standard Deviation', color='g', ls='--')
plt.axvline(standardizedMean + standardizedStandardDeviation, color='g', ls='--')

plt.title("Data Distribution")
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend(loc="upper right")

# plot raw data (left, bottom)
plt.subplot(2, 2, 3)

plt.scatter(plotRange, dataSet, label='Data', s=2)
plt.axhline(mean, label='Mean', color='r', ls='--')
#plt.axhline(median, label='Median', color='y', ls='--')
plt.axhline(mean + standardDeviation, label='Standard Deviation', color='g', ls='--')
plt.axhline(mean - standardDeviation, color='g', ls='--')

plt.title('Raw Data')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend(loc="best")

# plot standardizedData(left, bottom)
plt.subplot(2, 2, 4)
plt.title('Normalized Data')
plt.xlabel('Sample')
plt.ylabel('Standardized Value')

plt.scatter(plotRange, standardizedDataSet, label='Data', s=2)
plt.axhline(standardizedMean, label='Mean', color='r', ls='--')
#plt.axhline(standardizedMedian, label='Median', color='y', ls='--')
plt.axhline(standardizedMean + standardizedStandardDeviation, label='Standard Deviation', color='g', ls='--')
plt.axhline(standardizedMean - standardizedStandardDeviation, color='g', ls='--')

plt.legend(loc="best")

plt.tight_layout()
plt.show()


