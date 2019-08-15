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
from Tomlab_ES_Fire3_110618 import *
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Define F - Imposing a negative integral to maximize instead of minimize
def f(t):
    intgrl, abserr = quad(lambda t : - np.exp(delta_ * t) * ((1 - F1) * NB1 + (1 - F2) * NB2 + (1 - F3) * NB3), 0, np.inf)
    return intgrl
# Objective: Maximize the integral of the above function (from 0 to infinity) with the initial condition
# dv_i/dt = [1 - F(phi_i)][g(v_i)-m_i*v_i]
# and m_i >= 0
x0 = np.array([S0[0] * K[0], S0[1] * K[1], S0[2] * K[2]], dtype = np.float128) #initial condition
# Integrate the above

res = minimize(f, x0, method = 'Nelder-Mead')

r = 0.05
m = 0.5
# Interpolate solutions off of the collocation nodes



# find v(t_i)
def deriv_z1(z, t):
    v, dv = z
    return [dv, r * v * (1 - v/K[0]) - m / K[0]]

def deriv_z2(z, t):
    v, dv = z
    return [dv, r * v * (1 - v/K[1]) - m/ K[1]]

def deriv_z3(z, t):
    v, dv = z
    return [dv, r * v * (1 - v/K[2]) - m / K[2]]

vinit = [0.5, 0]
s1 = integrate.odeint(deriv_z1, vinit, t)
s2 = integrate.odeint(deriv_z2, vinit, t)
s3 = integrate.odeint(deriv_z3, vinit, t)

# Find H
H1 = r * s1 * (1 - s1 / K[0]) * K[0]
H2 = r * s2 * (1 - s2 / K[1]) * K[1]
H3 = r * s3 * (1 - s3 / K[2]) * K[2]

"""
Calculating ecosystem services
"""
# Specifying percent of watershed covered in forest
Pi1plot = z1 * s1 / (1 + z1 * s1)
Pi2plot = z2 * z2 / (1 + z2 * s2)
Pi3plot = z3 * s3 / (1 + z3 * s3)
# Specifying water quality services (avoided costs $)
PQ1plot = 32 + ((z1 * S0[0] * K[0] / (1 + z1 * S0[0] * K[0])) - Pi1plot) * 3 * 0.19 * 11.3
PQ2plot = 32 + ((z2 * S0[1] * K[1] / (1 + z2 * S0[1] * K[1])) - Pi2plot) * 3 * 0.19 * 11.3
PQ3plot = 32 + ((z3 * S0[2] * K[2] / (1 + z3 * S0[2] * K[2])) - Pi3plot) * 3 * 0.19 * 11.3
# Specifying outdoor recreation (recreation days)
OR1plot = a1 * (s1 ** b)
OR2plot = a2 * (s2 ** b)
OR3plot = a3 * (s3 ** b)
# Specifying hunting (hunting days)
HT1plot = g1 * Pi1plot - g1 * Pi1plot ** 2
HT2plot = g2 * Pi2plot - g2 * Pi2plot ** 2
HT3plot = g3 * Pi3plot - g3 * Pi3plot ** 2
# Specifying water yield (inches x area of watershed = cubic meters of water -> acre feet)
WY1plot = (49 - 48.82209 - 0.14884 * Pi2plot) * 0.0254 * (A[0] * 2.59e6 * 0.000810714)
WY2plot = (49 - 48.82209 - 0.14884 * Pi2plot) * 0.0254 * (A[1] * 2.59e6 * 0.000810714)
WY3plot = (49 - 48.82209 - 0.14884 * Pi3plot) * 0.0254 * (A[2] * 2.59e6 * 0.000810714)
# Specifying grazing (AUM's)
GZ1plot = AUM[0] * (1 - Pi1plot) * A[0]/allot[0]
GZ2plot = AUM[1] * (1 - Pi2plot) * A[1]/allot[1]
GZ3plot = AUM[2] * (1 - Pi3plot) * A[2]/allot[2]
# Specifying timber harvest Value
TV1plot = m0[0] * np.exp(m1[0] * s1)
TV2plot = m0[1] * np.exp(m1[1] * s2)
TV3plot = m0[2] * np.exp(m1[2] * s3)
"""
Plotting the Results
"""

# Need to get a loop going to get these plots
# optimal biomass figure
plt.plot(t,s1,'r',t,s2,'g',t,s3,'b')
plt.ylabel('Forest biomass (million short tons)')
plt.xlabel('Time')
plt.legend(['Watershed 1','Watershed 2', 'Watershed 3'])
plt.show()
# optimal fuel management
plt.plot(t, H1 * s1, 'r', t, H2 * s2, 'g', t, H3 * s3, 'b')
plt.ylabel('Fuel reduction (million short tons)')
plt.xlabel('Time')
plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
plt.show()

plot_path = 1 # 1 = yes
if plot_path == 1:
    #Optimal water quality ecosystem service figure
    plt.plot(t, PQ1plot, 'r', t, PQ2plot, 'k', t, PQ3plot, 'b')
    plt.ylabel('Water quality costs')
    plt.xlabel('Time')
    plt.legend(['PQ1', 'PQ2', 'PQ3'])
    plt.show()
    # optimal outdoor rec figure
    plt.plot(t, POR[0] * OR1plot, 'r', t, POR[1] * OR2plot, 'k', t, POR[2] * OR3plot, 'b')
    plt.ylabel('Value of recreation')
    plt.xlabel('Time')
    plt.legend(['OR1', 'OR2', 'OR3'])
    plt.show()
    # optimal hunting figure
    plt.plot(t, PHT[0] * HT1plot, 'r', t, PHT[1] * HT2plot, 'k', t, PHT[2] * HT3plot, 'b')
    plt.ylabel('Value of hunting')
    plt.xlabel('Time')
    plt.legend(['HT1', 'HT2', 'HT3'])
    plt.show()
    # optimal water yield figure
    plt.plot(t, PWY[0] * WY1plot, 'r', t, PWY[1] * WY2plot, 'k', t, PWY[2] * WY3plot, 'b')
    plt.ylabel('Value of water yield')
    plt.xlabel('Time')
    plt.legend(['WY1', 'WY2', 'WY3'])
    plt.show()
    # optimal grazing figure
    plt.plot(t, PGZ[0] * GZ1plot, 'r', t, PWY[1] * WY2plot, 'k', t, PWY[2] * WY3plot, 'b')
    plt.ylabel('Value of Grazing')
    plt.xlabel('Time')
    plt.legend(['GZ1', 'GZ2', 'GZ3'])
    plt.show()
    # optimal timber harvesting figure
    plt.plot(t, TV1plot, 'r', t, TV2plot, 'k', t, TV3plot, 'b')
    plt.ylabel('Value of timber')
    plt.xlabel('Time')
    plt.legend(['TV1', 'TV2', 'TV3'])
    plt.show()
    # optimal forested area figure
    plt.plot(t, Pi1plot, 'r', t, Pi2plot, 'k', t, Pi3plot, 'b')
    plt.ylabel('Percent of watershed in forest')
    plt.xlabel('Time')
    plt.legend(['Pi1', 'Pi2', 'Pi3'])
    plt.show()
# end
