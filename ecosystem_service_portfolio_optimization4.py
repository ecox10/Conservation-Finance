"""
Estimate the Markowitz efficient frontier for ecosystem services in watershes (HUC10) with public
lands. Assets are the 6 ecosystem services. The federal government owns
the land that generates the ecosystem service values. We are valuing the
ecosystem services by asking "Which ecosystem services would attract
private capital investment?" Essentially, if the federal government offered
the returns they have from public lands, which would attract private investment?
Investors are not buying land.
"""

"""
ecosystem service values
columns are the 6 ecosystem service values ($/year/acre)
rows are the HUCs
"""
from pandas import *
from numpy import *
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt

HUC10 = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_HUC10.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))
HUC10ID = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_HUC10ID.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))
HUC10MT = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_HUC10MT.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))
HUC10WY = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_HUC10WY.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))
acre = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_acre.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))
acreID = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_acreID.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))
acreMT = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_acreMT.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))
acreWY = read_table("C:/Users/elizabethcox/Documents/Python Files/Data Files/annual_ESvalues_acreWY.txt", skiprows = [0,1,2,3,4,5,6,7],
        names = ('Timber', 'Mining', 'Grazing', 'Water_yield', 'Water_quality', 'Carbon_sequestration'))

# 1 = timber
# 2 = mining
# 3 = grazing
# 4 = water yield
# 5 = water quality
# 6 = carbon sequestration
eco_serv = array(["Timber","Grazing","Water_yield","Water_quality","Carbon_sequestration"]) # select which ecosystem services to consider
# 1 = northern rockies
# 2 = Idaho
# 3 = Montana
# 4 = Wyoming
region = 1
scale = 1000000 # scales ES values to provide more manageable values (millions of dollars works)
if region == 1:
    valuesHUC = HUC10[eco_serv]/scale # ES values per year by HUC10
    valuesacre = acre[eco_serv]/scale # ES values per year by acre
elif region == 2:
    valuesHUC = HUC10ID[eco_serv]/scale # ES values per year by HUC10
    valuesacre = acreID[eco_serv]/scale # ES values per year by acre
elif region == 3:
    valuesHUC = HUC10MT[eco_serv]/scale # ES values per year by HUC10
    valuesacre = acreMT[eco_serv]/scale # ES values per year by acre
elif region == 4:
    valuesHUC = HUC10WY[eco_serv]/scale # ES values per year by HUC10
    valuesacre = acreWY[eco_serv]/scale # ES values per year by acre
# end
rows = len(valuesHUC)
col = len(valuesHUC.columns)
ESTotals = sum(valuesHUC) # Annual value of all ES before fire/insect
Total = sum(ESTotals) # Annual value of all ES before fire/insect
T = 10 # how long ecosystem service values are lost due to fire or insect
r = 0.05 # rate you discount future losses (fire-adjusted losses)
rf = 0.02 # risk free return if total ES value was put in risk-free asset
Acres = 175782584.7 # total acres in study area
burn_acres = 0.0057 * Acres # number of acres burned

annual_losses = burn_acres * valuesacre * (1 - ((1 + r) ** -T)) / r # NVP of fire/insect losses each year

annual_dollars = zeros((rows,col))
for i in range(rows):
    annual_dollars[:i] = (ESTotals - annual_losses[:i])
# end
annual_return = annual_dollars / Total
eco_services = transpose(annual_return)
eco_services_avg = mean(transpose(eco_services), axis = 0)
eco_services_cov = cov(eco_services, bias = True)
eco_services_stddev = std(transpose(eco_services), axis = 0)

### Derive the Markowitz full efficient portfolios
n = 10000 # n = 1 returns the minimum risk portfolio
ef = EfficientFrontier(eco_services_avg, eco_services_cov)
port_return = ef.efficient_return(target_return = 10000.) # the Markowitz portfolio
portwts_all = ef.clean_weights()
portrisk_all = ef.efficient_risk(target_risk = 10000.)
ef.portfolio_performance(verbose = True)

# find the tangency portfolio
tangency_portfolio = ef.max_sharpe()

"""
calculate the portfolio with simple diversification with equal
investment in each ecosystem service
"""

num_serv = len(eco_serv)
port_wts_simple = zeros((1, num_serv))

for i in range(num_serv):
    port_wts_simple[0][i] = 1 / num_serv
# end

port_risk_simple = sqrt(matmul(matmul(port_wts_simple, transpose(eco_services_cov)), transpose(port_wts_simple)))
port_return_simple = matmul(port_wts_simple, transpose(eco_services_avg))

"""
Plots
"""
## plot Markowitz efficient frontier
portrisk_all_plt = array([0., 0.274247, 1.5072e-16, 0.0094728, 0.71628]) # plot requires numeric array rather than object returned by portfolio funtions
port_return_plt = array([2.01248e-13, 9.74549e-17, 3.798136e-15, 9.0207092e-11, 1.])
tangency_portfolio_plt = array([0., 0.2742471, 1.5072177e-16, 0.00947287, 0.71628003])
portwts_all_plt = array([0.,0.,0.,0.,1.])

plt.plot(portrisk_all_plt, port_return_plt, 'b', portrisk_all_plt, tangency_portfolio_plt, 'k--') # efficient frontier && capital market line
# plt.plot(eco_services_stddev[0], eco_services_avg[0], 'b+')
# plt.plot(eco_services_stddev[1], eco_services_avg[1], 'bs')
# plt.plot(eco_services_stddev[2], eco_services_avg[2], 'bv')
# plt.plot(eco_services_stddev[3], eco_services_avg[3], 'b*')
# plt.plot(eco_services_stddev[4], eco_services_avg[4], 'bo')
plt.xlabel('Std. Deviation (%)')
plt.ylabel('Expected Annual Return (%)')
plt.show()

## plot portfolio weights
plt.bar(portwts_all_plt, 'stacked')
plt.xlabel('Std. Deviation (%)')
plt.ylabel('Weights')
plt.axis([0, 1, 0, n])
plt.legend(['Timber', 'Grazing', 'Water yield', 'Water filtration', 'Carbon sequestration'])
plt.show()
