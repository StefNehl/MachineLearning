import sys
sys.path.insert(0, '../modules')
from RidgeRegression import RidgeRegression

ridgeRegression = RidgeRegression(4, 100, 3)
ridgeRegression.importData()
#ridgeRegression.generateTestDate1D()
ridgeRegression.generateTrainingSubset()
weightVector = ridgeRegression.computeLinearRidgeRegression(0.1)
ridgeRegression.plotData()
