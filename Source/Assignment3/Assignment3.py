"""
@author: Nehl Stefan
"""

import sys
sys.path.insert(0, '../modules')
import inference

# gaussDistr = inference.GaussDistribution(1)
# gaussDistr.generateSampels(2,3,1000)
# result = gaussDistr.generateGaussen()

# gaussDistr.plotData(result)

betaDistr = inference.BetaDistribution(0.5,0.5)
betaDistr.generateSampels(1000)
result = betaDistr.generateBetaDistribution()
betaDistr.plotData(result)



