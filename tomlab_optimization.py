"""
Tomlab optimization

Yield the optimal time path of fuel reduction m*_i(t)
"""

# get to directory with sample data and import data
import os
import math
import numpy as np
from scipy.integrate import *
from scipy.optimize import *
os.chdir("C:/Users/elizabethcox/Documents/Python Files")
import Tonlab_ES_Fire3_110618 as tonlab

# Define F
def f(t):
    intgrl, abserr = quad(lambda t : - math.exp(-tonlab.delta_ * t) * ((1 - tonlab.F1) * tonlab.NB1 + (1 - tonlab.F2) * tonlab.NB2 + (1 - tonlab.F3) * tonlab.NB3), 0, np.inf)
    return intgrl

# Objective: Maximize the integral of the above function (from 0 to infinity) with the initial condition
# dv_i/dt = [1 - F(phi_i)][g(v_i)-m_i*v_i]
# and m_i >= 0
x0 = [tonlab.S0[0] * tonlab.K[0], tonlab.S0[1] * tonlab.K[1], tonlab.S0[2] * tonlab.K[2]] #initial condition

# Integrate the above
l = np.linspace(1,10,num = 10)
res = minimize(f, x0, method = 'Nelder-Mead')
