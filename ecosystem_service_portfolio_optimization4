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
port_return = ef.efficient_return()
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
