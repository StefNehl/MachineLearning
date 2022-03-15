

import numpy

class ContiniousDustribution():

    @absractmethod
    def ImportCsv(self):
        pass  # supresses compile errors

    @absractmethod
    def ExportCsv(self):
        pass

    @abstractmethod
    def ComputeMean(self):
        pass

    @absractmethod
    def ComputeStandardDeviation(self):
        pass

    @abstractmethod
    def plotDistribution(self):
        pass

    @abstractmethod
    def plotRawData(self):
        pass

    @abstractmethod
    def plotGeneratedSamples(self):
        pass


class GaussDistribution(ContiniousDustribution):
    