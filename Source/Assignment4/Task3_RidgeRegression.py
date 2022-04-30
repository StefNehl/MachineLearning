import sys
sys.path.insert(0, '../modules')
from RidgeRegression import RidgeRegression

ridgeRegression = RidgeRegression(4)
ridgeRegression.importData()
ridgeRegression.generateTrainingSubset()
ridgeRegression.plotRawData()
