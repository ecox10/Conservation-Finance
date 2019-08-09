######## tomlab Forest ecosystem services with fire code -- Model 1 with fire entering as rate #########
## C Sims 1-26-2017
# Translated to Python by Ellie Cox 8-1-2019

# Setting this up as a function to nest any additional user functions in this file rather than as separate files

# Import Packages
import numpy
import matplotlib.pyplot as plt
import math
import Chebyshev #see separate file where this funtion is defined
import scipy
import scipy.integrate
import pulp

### Section One - Parameters
## Ecological Parameters

A = [1456, 1456, 1456] # average area of watersheds in Montana (square miles)
B = [0.019, 0.019, 0.019] # biomass per area (million short tons per square mile) Source: ECALIDATOR
S0 = [0.1, 0.1, 0.1] # initial percent of carrying capacity
R = [0.05, 0.05, 0.05] # forest biomass intrinsic growth rate
K = [0, 0, 0]
for i in range(len(A)):
    K[i] = A[i] * B[i] # forest biomass carrying capacity (million short tons per watershed)
# end
beta = 2 # hazard function exponent - when beta = 1 constant hazard rate
lamda = 0.05 # lamda is intentionally spelled wrong to avoid conflict with built in lambda function
d1 = [1, 0.5, 0] # determines relationship between fuel load in watershed i and fire frequency in watershed 1
d2 = [0.25, 1, 0] # determines relationship between fuel load in watershed i and fire frequency in watershed 2
d3 = [0.25, 0.5, 1] # determines relationship between fuel load in watershed i and fire frequency in watershed 3
Pibar = [0.75, 0.75, 0.75] # max percent of watershed covered in forest at carrying capacity
z1 = Pibar[0]/K[0]*(1-Pibar[0]) # calibrated so Pi = Pibar at S=K
z2 = Pibar[1]/K[1]*(1-Pibar[1]) # calibrated so Pi = Pibar at S=K
z3 = Pibar[2]/K[2]*(1-Pibar[2]) # calibrated so Pi = Pibar at S=K
Pi01 = z1*(S0[0]*K[0])/(1+z1*(S0[0]*K[0])) # initial percent of watershed covered in Forest
Pi02 = z2*(S0[1]*K[1])/(1+z2*(S0[1]*K[1])) # initial percent of watershed covered in Forest
Pi03 = z3*(S0[2]*K[2])/(1+z3*(S0[2]*K[2])) # initial percent of watershed coverage in forest

## Ecosystem service Parameters
# Water quality
consumers = [81327, 67773, 0] # water utility customers (1: Cascade Co (Great Falls), 2: Lewis and Clark Co (Helena), 3: boonies)
wpp = 100 # gallons used per day
gpd = consumers * wpp # size of water treatment plant watershed serves (gallons per day treated)
WQ = [0, 0, 0]
for i in range(len(consumers)):
    WQ[i] = gpd[i]*365/1000000
# end

# Outdoor recreation
visitors1 = 2000000/1856.4564 # visitors per sq mile in Custer NF 2003
visitors2 = 528855/2912 # visitors per sq mile in HLC NF in 2003
visitors3 = 0
Rec = [visitors1 * A[0], visitors2 * A[1], visitors3 * A[2]]
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
allot = [12.266, 12.266, 12.266] # size (sq miles) of average grazing allotment in Helena-Lewis and Clark NF
AUM = [50*6*915, 50*6*915, 50*6*915] # animal unit month per allotment per year

# Timber Harvesting (value of timber harvested, $)
# TV = m0*exp(m1*v)
m0 = [10207.968130941700, 0, 10207.968130941700]
m1 = [0.073560707399, 0, 0.073560707399]

# Economic Parameters
delta_ = 0.4 # discount rate
PWY = [1500, 1500, 300] # value of water yield ($/acre feet)
POR = [53.45, 95.96, 0] # value of outdoor recreation ($/recreation day)
PHT = [0, 0, 75.99] # value of hunting ($/hunting day)
PGZ = [1.69, 1.69, 1.69] # grazing fees ($/AUM)
# p_st = 350*0.974*1000000 # convert $/MBF to $/million short tons
# PTM = [p_st, 0, p_st] # mill price for timber ($/million short tons)
c_st = 216 * 0.974 * 1000000 # convert harvest cost to $/MBF to $/million short tons
CTM = [c_st, 0, c_st] # timber harvesting costs ($/million short tons)

# Fuel management cost terms
# The cii terms are calibrated so that with linear costs the breakeven
# biomass is alph_% of the max type 1 ES and the quadratic costs are 1/4 of the linear costs
alph_, fract = 50000000, 0.25
c = [[alph_, fract*alph_],
    [alph_, fract*alph_],
    [alph_, fract*alph_]]

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
F1fxn = numpy.linspace(0, 0, num = 100)
F2fxn = numpy.linspace(0, 0, num = 100)
F3fxn = numpy.linspace(0, 0, num = 100)
for i in range(len(s)):
    F1fxn[i] = 1 - math.exp(-lamda * (d1[0] * s[i] + d1[1] * s[i] + d1[2] * s[i]) ** beta / beta)
    F2fxn[i] = 1 - math.exp(-lamda * (d2[0] * s[i] + d2[1] * s[i] + d2[2] * s[i]) ** beta / beta)
    F3fxn[i] = 1 - math.exp(-lamda * (d3[0] * s[i] + d3[1] * s[i] + d3[2] * s[i]) ** beta / beta)
# end

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
C1fxn = c[0][0] * h + c[0][1] * (h ** 2)
C2fxn = c[1][0] * h + c[1][1] * (h ** 2)
C3fxn = c[2][0] * h + c[2][1] * (h ** 2)
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
TV1fxn = numpy.linspace(0, 0, num = len(s))
TV2fxn = numpy.linspace(0, 0, num = len(s))
TV3fxn = numpy.linspace(0, 0, num = len(s))
for i in range(len(s)):
    TV1fxn[i] = m0[0] * math.exp(m1[0] * s[i])
    TV2fxn[i] = m0[1] * math.exp(m1[1] * s[i])
    TV3fxn[i] = m0[2] * math.exp(m1[2] * s[i])
# end

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
t = numpy.linspace(1, 10, num = 10)
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
x0 = numpy.polynomial.chebyshev.chebval([0.5 * K[0], 0.5 * K[1], 0.5 * K[2], 0.25 * R[0] *K[0], 0.25 * R[1] * K[1], 0.25 * R[2] * K[2]], Domain)
#else

# Copy solution from previous Nset for initializing next Nset
# (finer grid)
#x0 = numpy.polynomial.chebyshev.chebval([s1_init, s2_init, s3_init, h1_init, h2_init, h3_init])
# end

cbox = numpy.polynomial.chebyshev.chebval([K[0] * 1.5, K[1] * 1.5, K[2] * 1.5, x0[3], x0[4], x0[5]], Domain)

cterm = [] # If we want to impose a T condition, we can add it here
cbnd = [S0[0] * K[0], S0[1] * K[1], S0[2]* K[2]] # initial conditions

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
TV1 = m0[0] * math.exp(m1[0] * x0[0])
TV2 = m0[1] * math.exp(m1[1] * x0[1])
TV3 = m0[2] * math.exp(m1[2] * x0[2])

# Specifying fire hazard Probability
F1 = 1 - math.exp(-lamda * (d1[0] * x0[0] + d1[1] * x0[1] + d1[2] * x0[2]) ** beta / beta)
F2 = 1 - math.exp(-lamda * (d2[0] * x0[0] + d2[1] * x0[1] + d2[2] * x0[2]) ** beta / beta)
F3 = 1 - math.exp(-lamda * (d3[0] * x0[0] + d3[1] * x0[1] + d3[2] * x0[2]) ** beta / beta)

# Specifying the cost function
C1 = c[0][0] * x0[3] + c[0][1] * x0[3] ** 2
C2 = c[1][0] * x0[4] + c[1][1] * x0[4] ** 2
C3 = c[2][0] * x0[5] + c[2][1] * x0[5] ** 2

# Specifying the net benefits per year
NB1 = PWY[0] * WY1 + POR[0] * OR1 + PHT[0] * HT1 + PGZ[0] * GZ1 + TV1 - PQ1 * WQ[0] - C1
NB2 = PWY[1] * WY2 + POR[1] * OR2 + PHT[1] * HT2 + PGZ[1] * GZ2 + TV2 - PQ2 * WQ[1] - C2
NB3 = PWY[2] * WY3 + POR[2] * OR3 + PHT[2] * HT3 + PGZ[2] * GZ3 + TV3 - PQ3 * WQ[2] - C3