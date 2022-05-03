import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from inference import Regression

class RidgeRegression(Regression):

    def __init__(self, trainStep, trainingsIterations, order):
        self.trainStep = trainStep
        self.trainingsIteration = trainingsIterations
        self.hasGeneratedTestData = False
        self.order = order

    def importData(self):
        dataDictonary =  sio.loadmat("AssignmentIV_data_set.mat")
        tempData = dataDictonary.get("TempField")
        self.dataSet = np.array(tempData[:,:, 0]) #x * y
        self.numberOfSamples = len(self.dataSet)

    def generateTestDate(self):
        self.hasGeneratedTestData = True
        c2 = 0.01
        c1 = 1.3
        c0 = 3.456
        self.inputValues = np.linspace(-10.0, 20.2, 200)
        self.outputValues = c1 * self.inputValues ** 2 + c1 * self.inputValues + c0 + 500.0 * np.random.rand(len(self.inputValues))

    def generateTrainingSubset(self):
        #if self.hasGeneratedTestData:
        #    self.trainSubsetInput = self.inputValues
        #    self.trainSubsetOutput = self.outputValues
        #    return
        self.trainSubsetInput = np.array(self.inputValues[0:len(self.inputValues):self.trainStep])
        self.trainSubsetOutput = np.array(self.outputValues[0:len(self.outputValues):self.trainStep])


    def createPolynomialFeatureVector(self, xVector):
        featureVector = np.ones(xVector.shape)

        for i in range(1, self.order):
            newXVector = xVector ** i
            featureVector = np.hstack((featureVector, newXVector))

        # featureVector = np.hstack(([1.0], xVector, np.square(xVector)))
        return featureVector


    def computeLinearRidgeRegression(self, lambdaValue):
        X = np.vstack(([self.createPolynomialFeatureVector(x) for x in self.inputValues]))
        Y = np.vstack(([y for y in self.outputValues]))

        #AT = np.transpose(A)
        #I = np.identity(A.shape[1])
        #ATAwithLambda = np.matmul(AT,A) + lambdaValue * I
        #result = ATAwithLambda.dot(AT)
        XT = np.transpose(X)
        XTX = np.matmul(XT, X) + lambdaValue * np.identity(X.shape[1])
        self.weightVector = np.matmul(np.matmul(np.linalg.inv(XTX), XT), Y)
        return self.weightVector


    def updateWeight(self, xVector, weightVector):
        yVectorPred = self.predict(weightVector)

        X = np.vstack(([self.createPolynomialFeatureVector(x) for x in self.trainSubsetInput]))
        Y = np.vstack(([y for y in self.trainSubsetOutput]))

        XT = np.transpose(X)
        XTXwithLamda = np.matmul(XT, X) + lambdaValue * np.identity(X.shape[1])
        weightVector = np.matmul(np.matmul(np.linalg.inv(XTXwithLamda), XT), Y)

        n, m = dataSet.shape()
        I = np.identity(m)

        return np.dot(
            np.dot(
                np.linalg.inv(
                    np.dot(dataSet.T, A) + lambdaValue * I), dataSet.T), yVectorPred)

    def predict(self,xVector, weightVector):
        return xVector.dot(weightVector)

    def plotRawData(self):
        ys = self.weightVector[0]
        for w in range(1,self.weightVector.shape[0]):
            ys = ys + (self.trainSubsetInput * self.weightVector[w])**(w)

        ys = np.array(ys)

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.scatter(self.inputValues, self.outputValues, label="RawData")
        plt.plot(self.trainSubsetInput, ys, '-r', label="Learned")

        plt.title("Distribution")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()

        plt.subplot(2, 2, 3)
        #hm = plt.imshow(self.inputValues,cmap='Reds', interpolation='none',
        #                extent=[0,400,0,400])

        #plt.colorbar(hm)
        plt.title(f"Raw data")
        plt.xlabel("Latitude coordinate")
        plt.ylabel("Longitude coordinate")

        plt.subplot(2, 2, 4)
        #hm = plt.imshow(self.outputValues,cmap='Reds', interpolation='none',
        #                extent=[0,400,0,400])

        #plt.colorbar(hm)
        plt.title(f"Train Subset")
        plt.xlabel("Latitude coordinate")
        plt.ylabel("Longitude coordinate")

        plt.tight_layout()
        plt.show()
