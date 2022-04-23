import numpy as np

populationAustria = 9095538
activeCases = 441098

covTestSensitivity = 0.971
covTestSpecific = 0.995

pAInfected = activeCases / populationAustria
pANotInfected = 1 - pAInfected

pB = covTestSpecific
pBA = covTestSensitivity

#BayersTheorem
pAB = pBA * pA / pB
print(pAB)
