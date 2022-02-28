import csv

import matplotlib.pyplot as plt
from BasicStatistics import BasicStatistics

dataSet = []


with open('gauss.csv', mode='r') as file:
    csvFile = csv.reader(file)

    for line in csvFile:
        dataSet.append(float(line[0]))

basicStatistics = BasicStatistics()

mean = basicStatistics.getMean(dataSet)
median = basicStatistics.getMedian(dataSet)
variance = basicStatistics.getVariance(dataSet)
standardDeviation = basicStatistics.getStandardDeviation(dataSet)
standardizedDataSet = basicStatistics.standardizeDataSet(dataSet)

length = len(standardizedDataSet)
range = range(length)

fig = plt.figure()
fig.suptitle('Raw Data')
plt.xlabel('Samples')
plt.ylabel('Values')

x = range
y = dataSet

plt.plot(x, y)
plt.show()


