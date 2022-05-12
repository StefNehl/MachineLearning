import sys
import seaborn as sbr
import matplotlib.pyplot as plt

sys.path.insert(0, '../modules')
from RidgeRegression import RidgeRegression
from GaussianProcess import GaussianProcess
from GaussianProcess import KernelSetting

def doTestGauss(trainStep, kernelSetting, plot=False):
    gausRegression = GaussianProcess(trainStep)
    print("import data")
    gausRegression.importData()
    print("generate Trainingset")
    gausRegression.generateTrainingSubset()
    print("Train")
    gausRegression.computeGaussianProcessRegression(kernelSetting)
    gausRegression.testModel()
    print("Calculate Error")
    yTest = gausRegression.getYTestData()
    gausRegression.computeError(yTest)
    gausRegression.computMeanOfError()
    print("Mean Error with Lambda " )

    if plot:
        print("plot error")
        gausRegression.plotError()
        print("plot heatmap")
        gausRegression.plotHeatMap()

    return gausRegression.getDescSortedError(), gausRegression.getSettingsString()

def doTestRidge(trainStep, lambdaValue, plot=False):
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

    return ridgeRegression.getDescSortedError(), ridgeRegression.getSettingsString()

firstTrainSetErrors = []

trainStep = 50
firstTrainSetErrors.append(doTestGauss(trainStep, kernelSetting=KernelSetting.Matern52))
firstTrainSetErrors.append(doTestGauss(trainStep, kernelSetting=KernelSetting.RBF))
#firstTrainSetErrors.append(doTestGauss(trainStep, kernelSetting=KernelSetting.RBFWithGpu))
firstTrainSetErrors.append(doTestGauss(trainStep, kernelSetting=KernelSetting.LinearKernel))
firstTrainSetErrors.append(doTestRidge(trainStep, 0.1))
firstTrainSetErrors.append(doTestRidge(trainStep, 0.5))
firstTrainSetErrors.append(doTestRidge(trainStep, 1))
#%firstTrainSetErrors.append((0.5, trainStep, (doTest(0.5, trainStep))))
#firstTrainSetErrors.append((1, trainStep, (doTest(1, trainStep))))
#firstTrainSetErrors.append((10, trainStep, (doTest(10, trainStep))))
#firstTrainSetErrors.append((50, trainStep, (doTest(50, trainStep, plot=True))))

#trainStep = 1
#firstTrainSetErrors.append((0.1, trainStep,(doTest(0.1, trainStep))))
#firstTrainSetErrors.append((0.5, trainStep, (doTest(0.5, trainStep))))
#firstTrainSetErrors.append((1, trainStep, (doTest(1, trainStep))))
#firstTrainSetErrors.append((10, trainStep, (doTest(10, trainStep))))
#firstTrainSetErrors.append((50, trainStep, (doTest(50, trainStep))))

plt.figure(figsize=(8, 8))
for testSet in firstTrainSetErrors:
    errors = testSet[0]
    settings = testSet[1]
    x = errors["values"]
    y = errors["error"]
    plt.plot(x, y, linewidth=1, label=settings)

plt.title("Errors with different Lambdas and TrainStep")
plt.xlabel("Descending Sorted Y Errors")
plt.ylabel("Error |yResult - yStar| [C]")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=1)
plt.tight_layout()
plt.show()