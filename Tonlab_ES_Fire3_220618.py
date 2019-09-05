######## tomlab Forest ecosystem services with fire code -- Model 1 with fire entering as rate #########
## C Sims 1-26-2017
# Translated to Python by Ellie Cox 8-1-2019

# Setting this up as a function to nest any additional user functions in this file rather than as separate files

# Import Packages
import numpy
import matplotlib.pyplot as plt
import math
import scipy
import pulp
import scipy.integrate as integrate
from scipy.optimize import *

### Section One - Parameters
## Ecological Parameters

A = numpy.array([1456, 1456, 1456], dtype = numpy.float128) # average area of watersheds in Montana (square miles)
B = numpy.array([0.019, 0.019, 0.019], dtype = numpy.float128) # biomass per area (million short tons per square mile) Source: ECALIDATOR
S0 = numpy.array([0.1, 0.1, 0.1], dtype = numpy.float128) # initial percent of carrying capacity
R = numpy.array([0.05, 0.05, 0.05], dtype = numpy.float128) # forest biomass intrinsic growth rate
K = numpy.array(A * B, dtype = numpy.float128) # forest biomass carrying capacity (million short tons per watershed)
beta = 2 # hazard function exponent - when beta = 1 constant hazard rate
lamda = 0.05 # lamda is intentionally spelled wrong to avoid conflict with built in lambda function
d1 = numpy.array([1, 0.5, 0], dtype = numpy.float128) # determines relationship between fuel load in watershed i and fire frequency in watershed 1
d2 = numpy.array([0.25, 1, 0], dtype = numpy.float128) # determines relationship between fuel load in watershed i and fire frequency in watershed 2
d3 = numpy.array([0.25, 0.5, 1], dtype = numpy.float128) # determines relationship between fuel load in watershed i and fire frequency in watershed 3
Pibar = numpy.array([0.75, 0.75, 0.75], dtype = numpy.float128) # max percent of watershed covered in forest at carrying capacity
z1 = Pibar[0]/K[0]*(1-Pibar[0]) # calibrated so Pi = Pibar at S=K
z2 = Pibar[1]/K[1]*(1-Pibar[1]) # calibrated so Pi = Pibar at S=K
z3 = Pibar[2]/K[2]*(1-Pibar[2]) # calibrated so Pi = Pibar at S=K
Pi01 = z1*(S0[0]*K[0])/(1+z1*(S0[0]*K[0])) # initial percent of watershed covered in Forest
Pi02 = z2*(S0[1]*K[1])/(1+z2*(S0[1]*K[1])) # initial percent of watershed covered in Forest
Pi03 = z3*(S0[2]*K[2])/(1+z3*(S0[2]*K[2])) # initial percent of watershed coverage in forest

## Ecosystem service Parameters
# Water quality
consumers = numpy.array([81327, 67773, 0], dtype = numpy.float128) # water utility customers (1: Cascade Co (Great Falls), 2: Lewis and Clark Co (Helena), 3: boonies)
wpp = 100 # gallons used per day
gpd = consumers * wpp # size of water treatment plant watershed serves (gallons per day treated)
WQ = numpy.array(gpd * 365 / 1000000, dtype = numpy.float128)

# Outdoor recreation
visitors1 = 2000000/1856.4564 # visitors per sq mile in Custer NF 2003
visitors2 = 528855/2912 # visitors per sq mile in HLC NF in 2003
visitors3 = 0
Rec = numpy.array([visitors1 * A[0], visitors2 * A[1], visitors3 * A[2]], dtype = numpy.float128)
b = 0.3
a1 = Rec[0]/K[0] ** b # a parameterized to = visitors when v = k
a2 = Rec[1]/K[1] ** b
a3 = Rec[2]/K[2] ** b

# Hunting
# HT = g*pi-g*pi^2
hunters = 0.36 * visitors1 # hunters per sq mile
hunt = [0, 0, hunters * A[2]]
g1 = hunt[0] / (0.5 - (0.5 ** 2)) # calibrated so peak of hunter funtion = hunt
g2 = hunt[1] / (0.5 - (0.5 ** 2)) # calibrated so peak of hunter funtion = hunt
g3 = hunt[2] / (0.5 - (0.5 ** 2)) # calibrated so peak of hunter function = hunt

# Grazing (total animal per unit months)
# GZ = AUM*(1-Pi)*A/ALLOT
allot = numpy.array([12.266, 12.266, 12.266], dtype = numpy.float128) # size (sq miles) of average grazing allotment in Helena-Lewis and Clark NF
AUM = numpy.array([50*6*915, 50*6*915, 50*6*915], dtype = numpy.float128) # animal unit month per allotment per year

# Timber Harvesting (value of timber harvested, $)
# TV = m0*exp(m1*v)
m0 = numpy.array([10207.968130941700, 0, 10207.968130941700], dtype = numpy.float128)
m1 = numpy.array([0.073560707399, 0, 0.073560707399], dtype = numpy.float128)

# Economic Parameters
delta_ = 0.4 # discount rate
PWY = numpy.array([1500, 1500, 300], dtype = numpy.float128) # value of water yield ($/acre feet)
POR = numpy.array([53.45, 95.96, 0], dtype = numpy.float128) # value of outdoor recreation ($/recreation day)
PHT = numpy.array([0, 0, 75.99], dtype = numpy.float128) # value of hunting ($/hunting day)
PGZ = numpy.array([1.69, 1.69, 1.69], dtype = numpy.float128) # grazing fees ($/AUM)
# p_st = 350*0.974*1000000 # convert $/MBF to $/million short tons
# PTM = [p_st, 0, p_st] # mill price for timber ($/million short tons)
c_st = 216 * 0.974 * 1000000 # convert harvest cost to $/MBF to $/million short tons
CTM = numpy.array([c_st, 0, c_st], dtype = numpy.float128) # timber harvesting costs ($/million short tons)

# Fuel management cost terms
# The cii terms are calibrated so that with linear costs the breakeven
# biomass is alph_% of the max type 1 ES and the quadratic costs are 1/4 of the linear costs
alph_, fract = 50000000, 0.25
c = numpy.matrix([[alph_, fract*alph_],
    [alph_, fract*alph_],
    [alph_, fract*alph_]], dtype = numpy.float128)

### Section 2 - Functional Forms
plot_fxn = 2 # 0 = no figures, 1 = yes to production functions, 2 = yes to PPF
s = numpy.linspace(0, K[0], num = 100) # biomass
h = numpy.linspace(0, 1, num = 100) # harvest

# forest growth
GR1fxn = R[0] * s * (1 - s/K[0])
GR2fxn = R[1] * s * (1 - s/K[1])
GR3fxn = R[2] * s * (1 - s/K[2])
plt.plot(s, GR1fxn, 'r', s, GR2fxn, 'b', s, GR3fxn, 'g')
plt.axis([0, K[0], 0, 0.35])
plt.ylabel('Forest growth g_i(v_i)')
plt.xlabel('Biomass (million dry short tons)')
plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
plt.show()

# fire hazard probability
F1fxn = 1 - numpy.exp(-lamda * (d1[0] * s + d1[1] * s + d1[2] * s) ** beta / beta)
F2fxn = 1 - numpy.exp(-lamda * (d2[0] * s + d2[1] * s + d2[2] * s) ** beta / beta)
F3fxn = 1 - numpy.exp(-lamda * (d3[0] * s + d3[1] * s + d3[2] * s) ** beta / beta)


plt.plot(s, F1fxn, 'r', s, F2fxn, 'b', s, F3fxn, 'g')
plt.ylabel('Probability fire has occured F(phi_i)')
plt.xlabel('Biomass (million dry short tons)')
plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
plt.show()

# Percent of watershed covered in Forest
Pi1fxn = z1 * s / (1 + z1 * s)
Pi2fxn = z2 * s / (1 + z2 * s)
Pi3fxn = z3 * s / (1 + z3 * s)
plt.plot(s, Pi1fxn, 'r', s, Pi2fxn, 'b', s, Pi3fxn, 'g')
plt.axis([0, K[0], 0, 0.8])
plt.xlabel('Biomass (million dry short tons)')
plt.ylabel('Percent of watershed in forest (pi_i)')
plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
plt.show() # is this right?

# Fuel management cost function
C1fxn = c[0, 0] * h + c[0, 1] * (h ** 2)
C2fxn = c[1, 0] * h + c[1, 1] * (h ** 2)
C3fxn = c[2, 0] * h + c[2, 1] * (h ** 2)
plt.plot(h, C1fxn, 'r', h, C2fxn, 'b', h, C3fxn, 'g')
plt.axis([0, 1, 0, 70000000])
plt.ylabel('Fuel management costs (C(m_i))')
plt.xlabel('Harvest Rage')
plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
plt.show()

# Water quality services (cost per million gallons treated)
# Current savings: $27 / ML for conventional treatment and $35 / ML for direct treatment
# Current savings: 35-27 = $8 / ML * (1ML / 0.25 mil gal) = $32 / mil gal
# Current Savings: ($32 / mil gal) * WQ
# Mean chemical costs $27.82 / ML = $27/ ML * (1ML . 0.25 mil gal) = $111.3 / mil gallon
# 1 percent change in forest cover causes a 3 percent change in turbidity
# a 1 percent increase in turbidity causes a 0.19 percent change in costs
# Percent change in turbidity = (Pi0 - Pi) * 3
# ---- make sure the sign of the above is correct ----
# Avoided costs = (Percent change in turbidity * 0.19) * ($111.3 / mil gal) * WQ
# total benefits = current savings + avoided costs
PQ1fxn = 32 + ((z1 * S0[0] * K[0] / (1 + z1 * S0[0] * K[0])) - Pi1fxn) * 3 * 0.19 * 11.3
PQ2fxn = 32 + ((z2 * S0[1] * K[1] / (1 + z2 * S0[1] * K[1])) - Pi2fxn) * 3 * 0.19 * 11.3
PQ3fxn = 32 + ((z3 * S0[2] * K[2] / (1 + z3 * S0[2] * K[2])) - Pi3fxn) * 3 * 0.19 * 11.3

# Outdoor recreation (recreation days)
OR1fxn = a1 * (s ** b)
OR2fxn = a2 * (s ** b)
OR3fxn = a3 * (s ** b)

# Hunting days
HT1fxn = g1 * Pi1fxn - g1 * (Pi1fxn ** 2)
HT2fxn = g2 * Pi2fxn - g2 * (Pi2fxn ** 2)
HT3fxn = g3 * Pi3fxn - g3 * (Pi3fxn ** 2)

# Wateryield (inches x area of watershed = cubic meters of water -> acre feet)
WY1fxn = (49 - 48.82209 - 0.14884 * Pi1fxn) * 0.0254 * (A[0] * 2.59e6) * 0.000810714
WY2fxn = (49 - 48.82209 - 0.14884 * Pi2fxn) * 0.0254 * (A[1] * 2.59e6) * 0.000810714
WY3fxn = (49 - 48.82209 - 0.14884 * Pi3fxn) * 0.0254 * (A[2] * 2.59e6) * 0.000810714

# Grazing (total animal unit months per watershed)
GZ1fxn = AUM[0] * (1-Pi1fxn) * A[0]/allot[0]
GZ2fxn = AUM[1] * (1 - Pi2fxn) * A[1] / allot[1]
GZ3fxn = AUM[2] * (1 - Pi3fxn) * A[2] / allot[2]

# Timber value ($)
TV1fxn = m0[0] * numpy.exp(m1[0] * s)
TV2fxn = m0[1] * numpy.exp(m1[1] * s)
TV3fxn = m0[2] * numpy.exp(m1[2] * s)

if plot_fxn ==1:
    # water quality services (cost per million gallons treated)
    plt.plot(s, PQ1fxn * WQ[0], 'r', s, PQ2fxn * WQ[1], 'b', PQ3fxn * WQ[2], 'g')
    plt.ylabel('Avoided water treatment costs ($)')
    plt.xlabel('Biomass (million dry short tons)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    # Outdoor recreation (recreation days)
    plt.plot(s, POR[0] * OR1fxn, 'r', s, POR[1] * OR2fxn, 'b', s, POR[2] * OR3fxn, 'g')
    plt.ylabel('Value of Recreation ($)')
    plt.xlabel('Biomass (million dry short tons)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    # Hunting days
    plt.plot(s, PHT[0] * HT1fxn, 'r', s, PHT[1] * HT2fxn, 'b', s, PHT[2] * HT3fxn, 'g')
    plt.ylabel('Value of Hunting ($)')
    plt.xlabel('Biomass (million dry short tons)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    # Water yield (inches x area of watershed = cubic meters of water -> acre feet)
    plt.plot(s, PWY[0] * WY1fxn, 'r', s, PWY[1] * WY2fxn, 'b', s, PWY[2] * WY3fxn, 'g')
    plt.ylabel('Value of water yield ($)')
    plt.xlabel('Biomass (million dry tons)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    # Grazing (total animal unit months per watershed)
    plt.plot(s, PGZ[0] * GZ1fxn, 'r', s, PGZ[1] * GZ2fxn, 'b', s, PGZ[2] * GZ3fxn, 'g')
    plt.ylabel('Value of Grazing')
    plt.xlabel('Biomass (million dry short tons)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    # Timber value ($)
    plt.plot(s, TV1fxn, 'r', s, TV2fxn, 'b', s, TV3fxn, 'g')
    plt.ylabel('Value of Timber ($)')
    plt.xlabel('Biomass (million dry short tons)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

elif plot_fxn == 2:
    plt.plot(PWY[0] * WY1fxn, PQ1fxn * WQ[0], 'r', PWY[1] * WY2fxn, PQ2fxn * WQ[1], 'b', PWY[2] * WY3fxn, PQ3fxn * WQ[2], 'g')
    plt.xlabel('Value of water yield ($)')
    plt.ylabel('Avoided water treatment costs ($)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    plt.plot(PWY[0] * WY1fxn, POR[0] * OR1fxn, 'r', PWY[1] * WY2fxn, POR[1] * OR2fxn, 'b', PWY[2] * WY3fxn, POR[2] * OR3fxn, 'g')
    plt.xlabel('Value of Water Yield ($)')
    plt.ylabel('Value of recreation ($)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    plt.plot(PWY[0] * WY1fxn, PHT[0] * HT1fxn, 'r', PWY[1] * WY2fxn, PHT[1] * HT2fxn, 'b', PWY[2] * WY3fxn, PHT[2] * HT3fxn, 'g')
    plt.xlabel('Value of water yield ($)')
    plt.ylabel('Value of hunting ($)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    plt.plot(PWY[0] * WY1fxn, PGZ[0] * GZ1fxn, 'r', PWY[1] * WY2fxn, PGZ[1] * GZ2fxn, 'b', PWY[2] * WY3fxn, PGZ[2] * GZ3fxn, 'g')
    plt.xlabel('Value of water yield ($)')
    plt.ylabel('Value of grazing ($)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()

    plt.plot(PWY[0] * WY1fxn, TV1fxn, 'r', PWY[1] * WY2fxn, TV2fxn, 'b', PWY[2] * WY3fxn, TV3fxn, 'g') #compile error here for some reason
    plt.xlabel('Value of water yield ($)')
    plt.ylabel('Value of timber ($)')
    plt.legend(['Watershed 1', 'Watershed 2', 'Watershed 3'])
    plt.show()
#end

### Section 3 - Initial tomlab
# toms t
t = numpy.linspace(1, 80, num = 80)
T = 100 # set final T
Nset = [10, 20, 30]
Domain = [0, K[0]]


#for i in range(len(Nset)): # collocation nodes (finer and finder grids)

    # Spline finite element
    # finite element Use sfepy package
    # Gaussian collocation points numpy.polynomial.legendre.leggauss()
    # Cheby collocation points Chebyshev()


# Initial Guess
#if i == 1:

x0 = numpy.polynomial.chebyshev.chebval([0.5 * K[0], 0.5 * K[1], 0.5 * K[2], 0.25 * R[0] *K[0], 0.25 * R[1] * K[1], 0.25 * R[2] * K[2]], t) # Using chebyshev points
x0 = numpy.array(x0, dtype = numpy.float128)

"""
# using gaussian points
a, b = 0, 10
f = lambda x: numpy.cos(x)
# Gaussian default interval is [-1,1]
deg = 6
x, w = numpy.polynomial.legendre.leggauss(deg)
# Translate from default interval
x0 = 0.5 * (x + 1) * (b - a) + a
# Using finite elements
"""

# else
# Copy solution from previous Nset for initializing next Nset
# (finer grid)
#x0 = numpy.polynomial.chebyshev.chebval([s1_init, s2_init, s3_init, h1_init, h2_init, h3_init])
# end

cbox = numpy.array(numpy.polynomial.chebyshev.chebval([K[0] * 1.5, K[1] * 1.5, K[2] * 1.5, x0[3], x0[4], x0[5]], t), dtype = numpy.float128)

cterm = [] # If we want to impose a T condition, we can add it here

#Specifying percent of watershed covered in Forest
Pi1 = z1 * x0[0]/(1+z1*x0[0])
Pi2 = z2 * x0[1]/(1+z2*x0[1])
Pi3 = z3 * x0[3]/(1+z3*x0[2])

# Specifying water quality services (avoided costs $, 1000s)
PQ1 = 32 + ((z1 * S0[0] * K[0]/(1 + z1 * S0[0] * K[0])) - Pi1) * 3 * 0.19 * 11.3
PQ2 = 32 + ((z2 * S0[1] * K[1]/(1 + z2 * S0[1] * K[1])) - Pi2) * 3 * 0.19 * 11.3
PQ3 = 32 + ((z3 * S0[2] * K[2]/(1 + z3 * S0[2] * K[2])) - Pi3) * 3 * 0.19 * 11.3

# Specifying outdoor recreation (recreation days per year)
OR1 = a1 * (x0[0] ** b)
OR2 = a2 * (x0[1] ** b)
OR3 = a3 * (x0[2] ** b)

# Specifying hunting (hunting days per year)
HT1 = g1 * Pi1 - g1 * (Pi1 ** 2)
HT2 = g2 * Pi2 - g2 * (Pi2 ** 2)
HT3 = g3 * Pi3 - g3 * (Pi3 ** 2)

# Specifying water yeild (inches x area of watershed = cubic meters of water -> acre feet per year)
# 1 cubic meter = 0.000810714 AF
# 1 inch = 0.254 meters
# 1 sq mile = 2.29e6 sq miles
WY1 = (49 - 48.82209 - 0.14884 * Pi1) * 0.0254 * (A[0] * 2.59e6 * 0.000810714)
WY2 = (49 - 48.82209 - 0.14884 * Pi2) * 0.0254 * (A[1] * 2.59e6 * 0.000810714)
WY3 = (49 - 48.82209 - 0.14884 * Pi3) * 0.0254 * (A[2] * 2.59e6 * 0.000810714)

# Specifying grazing (animal unit months per year)
GZ1 = AUM[0] * (1 - Pi1) * A[0] / allot[0]
GZ2 = AUM[1] * (1 - Pi2) * A[1] / allot[1]
GZ3 = AUM[2] * (1 - Pi3) * A[2] / allot[2]

# Specifying timber Harvesting
TV1 = m0[0] * numpy.exp(m1[0] * x0[0])
TV2 = m0[1] * numpy.exp(m1[1] * x0[1])
TV3 = m0[2] * numpy.exp(m1[2] * x0[2])

# Specifying fire hazard Probability
F1 = 1 - numpy.exp(-lamda * (d1[0] * x0[0] + d1[1] * x0[1] + d1[2] * x0[2]) ** beta / beta)
F2 = 1 - numpy.exp(-lamda * (d2[0] * x0[0] + d2[1] * x0[1] + d2[2] * x0[2]) ** beta / beta)
F3 = 1 - numpy.exp(-lamda * (d3[0] * x0[0] + d3[1] * x0[1] + d3[2] * x0[2]) ** beta / beta)

# Specifying the cost function
C1 = c[0, 0] * x0[3] + c[0, 1] * x0[3] ** 2
C2 = c[1, 0] * x0[4] + c[1, 1] * x0[4] ** 2
C3 = c[2, 0] * x0[5] + c[2, 1] * x0[5] ** 2

# Specifying the net benefits per year
NB1 = PWY[0] * WY1 + POR[0] * OR1 + PHT[0] * HT1 + PGZ[0] * GZ1 + TV1 - PQ1 * WQ[0] - C1
NB2 = PWY[1] * WY2 + POR[1] * OR2 + PHT[1] * HT2 + PGZ[1] * GZ2 + TV2 - PQ2 * WQ[1] - C2
NB3 = PWY[2] * WY3 + POR[2] * OR3 + PHT[2] * HT3 + PGZ[2] * GZ3 + TV3 - PQ3 * WQ[2] - C3

# Define F - Imposing a negative integral to maximize instead of minimize
def f(t):
    intgrl, abserr = integrate.quad(lambda t : - numpy.exp(delta_ * t) * ((1 - F1) * NB1 + (1 - F2) * NB2 + (1 - F3) * NB3), 0, numpy.inf)
    return intgrl
# Objective: Maximize the integral of the above function (from 0 to infinity) with the initial condition
# dv_i/dt = [1 - F(phi_i)][g(v_i)-m_i*v_i]
# and m_i >= 0
x0 = numpy.array([S0[0] * K[0], S0[1] * K[1], S0[2] * K[2]], dtype = numpy.float128) #initial condition
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
TV1plot = m0[0] * numpy.exp(m1[0] * s1)
TV2plot = m0[1] * numpy.exp(m1[1] * s2)
TV3plot = m0[2] * numpy.exp(m1[2] * s3)
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
    plt.plot(t, PQ1plot, 'r', t, PQ2plot, 'g', t, PQ3plot, 'b')
    plt.ylabel('Water quality costs')
    plt.xlabel('Time')
    plt.legend(['PQ1', 'PQ2', 'PQ3'])
    plt.show()
    # optimal outdoor rec figure
    plt.plot(t, POR[0] * OR1plot, 'r', t, POR[1] * OR2plot, 'g', t, POR[2] * OR3plot, 'b')
    plt.ylabel('Value of recreation')
    plt.xlabel('Time')
    plt.legend(['OR1', 'OR2', 'OR3'])
    plt.show()
    # optimal hunting figure
    plt.plot(t, PHT[0] * HT1plot, 'r', t, PHT[1] * HT2plot, 'g', t, PHT[2] * HT3plot, 'b')
    plt.ylabel('Value of hunting')
    plt.xlabel('Time')
    plt.legend(['HT1', 'HT2', 'HT3'])
    plt.show()
    # optimal water yield figure
    plt.plot(t, PWY[0] * WY1plot, 'r', t, PWY[1] * WY2plot, 'g', t, PWY[2] * WY3plot, 'b')
    plt.ylabel('Value of water yield')
    plt.xlabel('Time')
    plt.legend(['WY1', 'WY2', 'WY3'])
    plt.show()
    # optimal grazing figure
    plt.plot(t, PGZ[0] * GZ1plot, 'r', t, PWY[1] * WY2plot, 'g', t, PWY[2] * WY3plot, 'b')
    plt.ylabel('Value of Grazing')
    plt.xlabel('Time')
    plt.legend(['GZ1', 'GZ2', 'GZ3'])
    plt.show()
    # optimal timber harvesting figure
    plt.plot(t, TV1plot, 'r', t, TV2plot, 'g', t, TV3plot, 'b')
    plt.ylabel('Value of timber')
    plt.xlabel('Time')
    plt.legend(['TV1', 'TV2', 'TV3'])
    plt.show()
    # optimal forested area figure
    plt.plot(t, Pi1plot, 'r', t, Pi2plot, 'g', t, Pi3plot, 'b')
    plt.ylabel('Percent of watershed in forest')
    plt.xlabel('Time')
    plt.legend(['Pi1', 'Pi2', 'Pi3'])
    plt.show()
# end
