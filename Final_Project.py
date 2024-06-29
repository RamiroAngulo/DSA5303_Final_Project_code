#Ramiro Angulo
#Final Project
#DSA 5303

#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

#The purpose of this application is to develop two programs for the following financial models and analyze the results: CAPM and Monte Carlo Simulation.


#CAPITAL ASSET PRICING MODEL

# importing the historical stock data BA The Boeing Company
stock_data = yf.download('BA', start='2022-01-01', end='2023-01-01')
# import stock for a year later on the day
stock_year = yf.download('BA', start='2024-01-01', end='2024-01-05')
# importing the historial market data ^GSPC S&P 500 Index
market_data = yf.download('^GSPC', start='2022-01-01', end='2023-01-01')
#print(market_data)
# Calculate the log daily returns for stock data
#assign to new Column named Returns
stock_data['Returns'] = np.log(stock_data['Adj Close']) - np.log(stock_data['Adj Close'].shift(1))
# Calculate the log daily returns for market data
#assign to new Column named Market Returns
market_data['Market Returns'] = np.log(market_data['Adj Close']) - np.log(market_data['Adj Close'].shift(1))

#plot log market returns
plt.figure(figsize=(10,5))
plt.plot(market_data)
plt.title('S&P 500 Index Return')
plt.xlabel('Time (days)')
plt.ylabel('Return')
plt.show()
#plot log stock returns
plt.figure(figsize=(10,5))
plt.plot(stock_data)
plt.title('The Boeing Company Return')
plt.xlabel('Time (days)')
plt.ylabel('Return')
plt.show()

#caclulate the expected rate of return for stock log returns
expected_stock_ror = stock_data['Returns'].mean()
#print(expected_stock_ror)
#calculate the expected rate of return for market log returns
expected_market_ror = market_data['Market Returns'].mean()
#print(expected_market_ror)
risk_free_rate = .0509  #1 Year Treasury Rate 5.09% risk-free rate as of 28 June 2024 
# create dataframe which contains both Stock and Market logorithmic returns
# drop na values to allow for covariance and variacne calculations
returns = pd.DataFrame(pd.concat([stock_data['Returns'], market_data['Market Returns']], axis=1).dropna())
#rename columns for unique identifiers to alleviate confusion in the code
returns = returns.rename(columns={'Returns': 'Stock', 'Market Returns': 'Market'})
#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#    print(returns)
# create a covariance matrix from the returns dataframe
cov_matrix = returns.cov()
#print(cov_matrix)
# retrieve the covariance between Stock and Market from the covariance matrix
stock_market_cov = cov_matrix.loc['Stock', 'Market']
#print(stock_market_cov)
#caclulate the Market variance
market_var = returns['Market'].var()
#print(market_var)
# using the covariance between Stock and Market, cacluate the beta value.
beta = (stock_market_cov) / (market_var)
print("The Beta of the stock is: ", beta)
#Using the CAPM formula, calculate the expected rate of return of the stock
expected_return_capm = risk_free_rate + beta * (expected_market_ror - risk_free_rate)
#view result
print("Expected Return based on CAPM:", expected_return_capm)
#using the estimated rate of return, calculate estimated final price
print("Expected Price based on CAPM:", stock_data['Adj Close'][-1]*(expected_return_capm+1))

#MONTE CARLO SIMULATION

# PARAMETERS
#intilize stock price using last price of historial stock data
S0 = stock_data['Adj Close'][-1]
T = 1  # Total time interval
N = 252  # step increment
dt = T/N # will allow a 252 trading day year worth of price predictions
mu = expected_stock_ror #pull expected stock rate of return
sigma = stock_data['Returns'].std() #standard deviation of target stock returns

#Random seed forrandom number generator used for Z
np.random.seed(42)
simulations = 1000 #number of simulations
price_i = np.zeros((N, simulations)) #initialize array
#insert initial value
price_i[0] = S0
price_i_mean = np.zeros(simulations) #initialize array
#execute the price simulations
for t in range(1, N):
    z = np.random.standard_normal(simulations)
    price_i[t] = price_i[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)    
#convert to dataframe
price_i = pd.DataFrame(price_i) 

#caclulate each simulation's mean
for i in range(price_i.shape[1]):
    price_i_mean[i] = price_i.iloc[:, i].mean()
#convert to dataframe
price_i_mean = pd.DataFrame(price_i_mean)
#create price rolling mean data frame
prm = pd.DataFrame(np.zeros(simulations))
#caclulate the rolling mean of the simulation means
for i in range(price_i_mean.shape[0]):
    prm[i] = price_i_mean.rolling(i).mean()
#print(prm.iloc[simulations-1, 0:])

#plot the simulations
plt.figure(figsize=(10,5))
plt.plot(price_i)
plt.title('Monte Carlo Simulation of Stock Prices')
plt.xlabel('Time (days)')
plt.ylabel('Stock Price')
plt.show()
#plot the rolling mean of the simulations
plt.figure(figsize=(10,5))
plt.plot(prm.iloc[simulations-1, 0:])
plt.title('Monte Carlo Simulation Rolling Mean')
plt.xlabel('Simulation Count')
plt.ylabel('Stock Price')
plt.show()

#retrieve final prices
#final_prices = prm.iloc[:,-1]
final_prices = price_i.iloc[-1,:]
#caclulte the mean 
mean_final_price = final_prices.mean()
#cacluate forcasted expected rate of return using the simulated prices
# Calculate the returns for each path
#expected_return_monte = ((final_prices - S0) / S0).mean()
expected_return_monte = ((mean_final_price - S0) / S0)
print("Expected Rate of Return for Monte Carlo Simulation: ", expected_return_monte)
#caculate the 97% confidence interval
confidence_interval = np.percentile(final_prices, [3, 97])
#print results
print("Mean final stock price: ", mean_final_price )
print("97% confidence interval: ", confidence_interval)

#retrieve the actual stock price a year later
print("Actual Price a year later: ", stock_year["Adj Close"][0])
#calculate the actual rate of return 
print("Actual rate of return a year later:", (stock_year["Adj Close"][0] - S0)/S0)


#caculate different confidence intervals
#confidence_interval_test1 = np.percentile(final_prices, [5, 95])
#print("95% confidence interval: ", confidence_interval_test1)
#confidence_interval_test2 = np.percentile(final_prices, [10, 90])
#print("90% confidence interval: ", confidence_interval_test2)
#confidence_interval_test3 = np.percentile(final_prices, [15, 85])
#print("85% confidence interval: ", confidence_interval_test3)
