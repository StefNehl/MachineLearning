import csv
from BasicStatistics import BasicStatistics

dataSet = []


with open('gauss.csv', mode='r') as file:
    csvFile = csv.reader(file)

    for line in csvFile:
        dataSet.append(line)

basicStatistics = BasicStatistics()

print(basicStatistics.getMeadian(dataSet))


