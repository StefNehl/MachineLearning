"""
@author: Nehl Stefan
"""

import sys
sys.path.insert(0, '../modules')
import inference

gaussDistr = inference.GaussDistribution(2)
gaussDistr.generateSampels(2,3,1000)
gaussDistr.plotData()



