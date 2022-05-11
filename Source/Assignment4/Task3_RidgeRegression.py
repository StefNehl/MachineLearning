import sys
import seaborn as sbr
import matplotlib.pyplot as plt

sys.path.insert(0, '../modules')
from RidgeRegression import RidgeRegression

def doTest(lambdaValue, trainStep, plot=False):
    ridgeRegression = RidgeRegression(trainStep)
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

    if plot:
        print("plot error")
        ridgeRegression.plotError()
        print("plot heatmap")
        ridgeRegression.plotHeatMap()

    return ridgeRegression.getDescSortedError()

firstTrainSetErrors = []

trainStep = 4
firstTrainSetErrors.append((0.1, trainStep,(doTest(0.1, trainStep, plot=True))))
firstTrainSetErrors.append((0.5, trainStep, (doTest(0.5, trainStep))))
firstTrainSetErrors.append((1, trainStep, (doTest(1, trainStep))))
firstTrainSetErrors.append((10, trainStep, (doTest(10, trainStep))))
firstTrainSetErrors.append((50, trainStep, (doTest(50, trainStep, plot=True))))

trainStep = 1
firstTrainSetErrors.append((0.1, trainStep,(doTest(0.1, trainStep))))
firstTrainSetErrors.append((0.5, trainStep, (doTest(0.5, trainStep))))
firstTrainSetErrors.append((1, trainStep, (doTest(1, trainStep))))
firstTrainSetErrors.append((10, trainStep, (doTest(10, trainStep))))
firstTrainSetErrors.append((50, trainStep, (doTest(50, trainStep))))

for testSet in firstTrainSetErrors:
    lambdaValue = testSet[0]
    trainStep = testSet[1]
    errors = testSet[2]
    x = errors["values"]
    y = errors["error"]
    plt.plot(x, y, linewidth=1, label=f"TrainStep: {trainStep}, Lambda = {lambdaValue}")

plt.title("Errors with different Lambdas and TrainStep")
plt.xlabel("Descending Sorted Y Errors")
plt.ylabel("Error |yResult - yStar| [C]")
plt.tight_layout()
plt.legend()
plt.show()









