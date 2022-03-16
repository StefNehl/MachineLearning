"""
@author: Nehl Stefan
"""

import sys
sys.path.insert(0, '../modules')
import inference

gaussDistr = inference.GaussDistribution(1)
gaussDistr.generateSampels(2,3,1000)
mean = gaussDistr.getMean()
result = gaussDistr.generateGaussen()

gaussDistr.plotData(result)



