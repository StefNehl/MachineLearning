import sys
sys.path.insert(0, '../modules')
from RidgeRegression import RidgeRegression

def doTest(lambdaValue, trainStep):
    ridgeRegression = RidgeRegression(100)
    print("import data")
    ridgeRegression.importData()
    print("generate Trainingset")
    ridgeRegression.generateTrainingSubset()
    print("Train")
    weightVector = ridgeRegression.computeLinearRidgeRegression(lambdaValue)
    ridgeRegression.testModel(weightVector)
    print("Calculate Error")
    yTest = ridgeRegression.getYTestData()
    ridgeRegression.computeError(yTest)
    ridgeRegression.computMeanOfError()
    print("Mean Error with Lambda " + str(lambdaValue) + ": " + str(ridgeRegression.getMeanError()))
    print("plot error")
    ridgeRegression.plotError()
    print("plot heatmap")
    ridgeRegression.plotHeatMap()


doTest(0.1, 4)



