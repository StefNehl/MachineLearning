"""
@author: Nehl Stefan
"""

import sys
sys.path.insert(0, '../modules')
import inference
from BetaDistribution import BetaDistribution
from GaussDistribution import GaussDistribution

gaussDistr = GaussDistribution(1, numberOfSamplesToGenerate = 1000, mean = 2, variance = 1)
gaussDistr.plotData()

# gaussDistr = GaussDistribution(2, numberOfSamplesToGenerate = 1000, mean = (2, 2), variance = (1, 1))
# gaussDistr.plotData()

# betaDistr = BetaDistribution(0.5,0.5, numberOfSamplesToGenerate = 1000)
# betaDistr.plotData()
#
# betaDistr = BetaDistribution(2,5, numberOfSamplesToGenerate = 1000)
# betaDistr.plotData()
#
# betaDistr = BetaDistribution(5,2, numberOfSamplesToGenerate = 1000)
# betaDistr.plotData()
#
# settings = [(0.5, 0.5), (2, 5), (5, 2)]
# betaDistr.plotDataWithDifferentAlphasAndBetas(settings)



