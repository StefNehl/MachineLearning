"""
@author: Nehl Stefan
"""

import sys
sys.path.insert(0, '../modules')
import inference

gaussDistr = inference.GaussDistribution(1, numberOfSamplesToGenerate = 1000, mean = 2, variance = 3)
gaussDistr.plotData()
# gaussDistr.generateSampels(2,3,1000)
# result = gaussDistr.generateGaussen()



# betaDistr = inference.BetaDistribution(0.5,0.5, numberOfSamplesToGenerate = 1000)
# betaDistr.plotData()



