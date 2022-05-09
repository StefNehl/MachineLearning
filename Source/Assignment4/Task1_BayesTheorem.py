import numpy as np

populationAustria = 9095538
activeCases = 441098
covTestSensitivity = 0.971
covTestSpecific = 0.995

#A...[infected, not-infected]
#B...[positive, negative]

pInfected = activeCases / populationAustria
pNotInfected = 1 - pInfected

pPositiveInfected = covTestSensitivity
pNegativeNotInfected = covTestSensitivity

pInfectedAndPositive = pInfected * pPositiveInfected
pInfectedAndNegative = pInfected - pInfectedAndPositive

pNotInfectedAndNegative = pNotInfected * pNegativeNotInfected
pNegative = pInfectedAndNegative + pNotInfectedAndNegative

pNotInfectedAndPositive = pNotInfected - pNotInfectedAndNegative
pPositive = pInfectedAndPositive + pNotInfectedAndPositive

pNegativeInfected = pInfectedAndNegative / pNegative
pPositiveNotInfected = pNotInfectedAndPositive / pNotInfected

#BayersTheorem
pInfectedPositive = pPositiveInfected * pInfected / pPositive

print("p(-|infected): " + str(pNegativeInfected))
print("p(+|non-infected): " + str(pPositiveNotInfected))
print("p(infected): " + str(pInfected))
print("p(non-infected): " + str(pNotInfected))
print("p(+): " + str(pPositive))
print("p(infected|+): " + str(pInfectedPositive))
