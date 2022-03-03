import csv

import matplotlib.pyplot as plt
import numpy
from BasicStatistics import BasicStatistics

dataSet = []


with open('gauss.csv', mode='r') as file:
    csvFile = csv.reader(file)

    for line in csvFile:
        dataSet.append(float(line[0]))

basicStatistics = BasicStatistics()

#dataSet = basicStatistics.normalizeDataSet(dataSet)
mean = basicStatistics.getMean(dataSet)
median = basicStatistics.getMedian(dataSet)
variance = basicStatistics.getVariance(dataSet)
standardDeviation = basicStatistics.getStandardDeviation(dataSet)

normalizedDataSet = basicStatistics.normalizeDataSet(dataSet)
standardizedDataSet = basicStatistics.standardizeDataSet(dataSet)
standardizedMean = basicStatistics.getMean(standardizedDataSet)
standardizedMedian = basicStatistics.getMedian(standardizedDataSet)
standardizedStandardDeviation = basicStatistics.getStandardDeviation(standardizedDataSet)


length = len(standardizedDataSet)
plotRange = range(length)

plt.suptitle('Data Distribution')

plt.figure(figsize=(8,6))
# plot histogram (left, right, top)
plt.subplot(2, 1, 1)
plt.tight_layout()


plt.hist(dataSet, bins=20, density=True, label='Histogram')
plt.axvline(mean, label='Mean', color='r', ls='--')
plt.axvline(median, label='Median', color='y', ls='--')
plt.axvline(mean - standardDeviation, label='Standard Deviation', color='g', ls='--')
plt.axvline(mean + standardDeviation, color='g', ls='--')

plt.title("Raw Data")
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend(loc="upper right")

# plot raw data (left, bottom)
plt.subplot(2, 2, 3)
plt.tight_layout()

plt.scatter(plotRange, dataSet, label='Data')
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
plt.tight_layout()
plt.title('Normalized Data')
plt.xlabel('Sample')
plt.ylabel('Standardized Value')

plt.scatter(plotRange, standardizedDataSet, label='Data')
plt.axhline(standardizedMean, label='Mean', color='r', ls='--')
#plt.axhline(standardizedMedian, label='Median', color='y', ls='--')
plt.axhline(standardizedMean + standardizedStandardDeviation, label='Standard Deviation', color='g', ls='--')
plt.axhline(standardizedMean - standardizedStandardDeviation, color='g', ls='--')

plt.legend(loc="best")

plt.show()


