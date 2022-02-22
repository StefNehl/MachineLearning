# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:33:44 2020

@author: hongh
"""

import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()  
x = np.linspace(0, 500, 500)
y_1 = np.log(x)
std_y_1 = 0.02 *x
plt.plot(x, y_1, color = 'b', alpha = 0.6, label='method 1')
plt.fill_between(x, y_1-std_y_1, y_1+std_y_1 ,color = 'b', alpha = 0.1)  ##539caf        

y_2 = x*0.1
std_y_2 = 0.01 *x
plt.plot(x, y_2, color = 'r', alpha = 0.6, label='method 2')         
plt.fill_between(x, y_2-std_y_2, y_2+std_y_2 ,color = 'r', alpha = 0.1)  ##539caf        


#pl.title('Episodic spillage average')
plt.xlabel('training episodes')
plt.ylabel('epsodic reward')  
plt.legend(loc='upper left')
plt.show()

fig.savefig("example plot.svg", format="svg")