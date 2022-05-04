import sys
sys.path.insert(0, '../modules')
from RidgeRegression import RidgeRegression

ridgeRegression = RidgeRegression(100, 100, 3)
print("import data")
ridgeRegression.importData()
#ridgeRegression.generateTestDate1D()
print("generate Trainingset")
ridgeRegression.generateTrainingSubset()
print("Train")
weightVector = ridgeRegression.computeLinearRidgeRegression(0.1)
print("plot")
ridgeRegression.plotData()
