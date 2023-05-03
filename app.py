import streamlit as st
from pickle import dump
from pickle import load
import statsmodels.api as sm
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
from yahooquery import Ticker
from scipy.optimize import minimize
import plotly.express as px



filename = 'finalized_robo_advice_model.sav'
# load the model from disk
loaded_model = load(open(filename, 'rb'))
investors = pd.read_csv('InputData.csv', index_col = 0 )
assets = pd.read_csv('SP500Data.csv',index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)

missing_fractions.head(10)

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

assets.drop(labels=drop_list, axis=1, inplace=True)
assets.shape
# Fill the missing values with the last value available in the dataset. 
assets=assets.fillna(method='ffill')
options=np.array(assets.columns)
# str(options)
options = []

for tic in assets.columns:
    #{'label': 'user sees', 'value': 'script sees'}
    mydict = {'label': tic, 'value': tic}
    options.append(mydict["value"])
    
def predict_riskTolerance(X_inpt):
    filename = 'finalized_robo_advice_model.sav'
    loaded_model = load(open(filename, 'rb'))
    return loaded_model.predict(X_inpt)


def update_risk_tolerance(Age, NetWorth, Income, EduLevel, MaritalStat, Kids, Occu, RiskTol):
    RiskTolerance = 0
    X_inpt = [[Age, NetWorth, Income, EduLevel, MaritalStat, Kids, Occu, RiskTol]]
    cols = ['AGE07', 'EDCL07', 'MARRIED07', 'KIDS07', 'OCCAT107', 'INCOME07', 'RISK07', 'NETWORTH07']
    X_inpt = pd.DataFrame(data=[[Age, NetWorth, Income, EduLevel, MaritalStat, Kids, Occu, RiskTol]], columns=cols)
    # print(X_inpt)
    RiskTolerance = predict_riskTolerance(X_inpt)
    return list[round(float(RiskTolerance * 100), 2)]

def get_asset_allocation(riskTolerance, stock_ticker):
    assets_selected = assets.loc[:, stock_ticker]
    return_vector = np.array(assets_selected.pct_change().dropna(axis=0)).T
    n = len(return_vector)
    returns = np.asmatrix(return_vector)
    mus = 1-(riskTolerance / 100)
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(return_vector))
    pbar = opt.matrix(np.mean(return_vector, axis=1))
    # Constraint Matrices
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Efficient Frontier Calculations
    portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)
    w = portfolios['x'].T
    # print(w)
    Alloc = pd.DataFrame(data = np.array(portfolios['x']),index = assets_selected.columns)
    # Calculating efficient frontier weights
    portfolios = solvers.qp(mus*S, -pbar, G, h, A, b)
    returns_final = (np.array(assets_selected) * np.array(w))
    returns_sum = np.sum(returns_final, axis=1)
    returns_sum_pd = pd.DataFrame(returns_sum, index=assets.index)
    returns_sum_pd = returns_sum_pd - returns_sum_pd.iloc[0,:] + 100
    return Alloc,returns_sum_pd

def update_asset_allocationChart(risk_tolerance, stock_ticker):
    Allocated, InvestmentReturn = get_asset_allocation(risk_tolerance, stock_ticker)
    return Allocated, InvestmentReturn
    
riskTolNumber = 0
with st.sidebar:
    st.subheader('Select Investor Profile')
    Age = st.slider("Age:",10,70, 35, key="investorAge")
    NetWorth = st.slider("Net-worth:", -100000, 2000000, 0, key="investorNetWorth")
    Income = st.slider("Income:", -100000, 2000000, 0, key="investorIncome")
    EduLevel = st.slider("Education Level:",1, 4, 2, key="investorEducation")
    MaritalStat = st.slider("Married:",1, 2, 1,key="investorMarried")
    Kids = st.slider("Kids:", 1, 12, 3, key="investorKids")
    Occu = st.slider("Occupation:",1, 4, 3, key="investorOccupation")
    RiskTol = st.slider("Willingness to  take Risk:",1, 4, 3, key="investorRisk")
    if st.button("Calculate Risk Tolerance"):
        risk_tolerance = update_risk_tolerance(Age, EduLevel, MaritalStat, Kids, Occu, Income, RiskTol, NetWorth)
        # print(risk_tolerance)
        numeric_types = [arg for arg in risk_tolerance.__args__ if isinstance(arg, (int, float))]
        # If there's at least one numeric type, get the first one
        number = numeric_types[0] if numeric_types else None
        # print(number)
        riskTolNumber += number
st.header("Robo-advisor Dashboard 🤖🦾")
st.write("This application uses a Random Forest Machine Learning  model to predict a user's risk tolerance according to a profile created by the user. The application then uses the predicted risk tolerance coefficient to run simualtions on 120,000 potential portfolios to find the best one for the given profile's risk appetite. The user starts by entering various securities to compose the portfolio and then predict the user's risk profile to find the optimal weighting of the given securities for the portfolio.")
st.subheader(f"Risk Tolerance is {riskTolNumber} on a scale of 100")
st.write("Please predict a risk profile by adjusting the sliders in the panel to the left")
portfolio_symbols = st.text_input("Enter the ticker symbols separated by commas👇🏾", placeholder="Ticker symbol", key="portfolioInput")

if len(portfolio_symbols) < 1:
    st.write("Please enter portfolio Ticker symbols separated by commas to start")
else:
    stock_symbols = portfolio_symbols.split(",")
    data = Ticker(stock_symbols).history(start='2013-01-01', end='2023-01-01')
    adj_close = pd.DataFrame(data['adjclose'])
    df = adj_close.pivot_table(index='date', columns='symbol', values='adjclose')
    returns = np.log(df/df.shift(1))
    mean_returns = returns.mean() * 252
    covariance = returns.cov() * 252
    std_returns = returns.std() * 252
    
    num_portfolios = 120000
    portfolio_weights = np.zeros((num_portfolios, len(stock_symbols)))
    for i in range(num_portfolios):
        weights = np.random.random(len(stock_symbols))
        weights /= np.sum(weights)
        portfolio_weights[i,:] = weights  
    portfolio_returns = np.dot(portfolio_weights, mean_returns)
    portfolio_volatility = np.zeros(num_portfolios)
    for i in range(num_portfolios):
        portfolio_volatility[i] = np.sqrt(np.dot(portfolio_weights[i,:].T, np.dot(covariance, portfolio_weights[i,:])))
        
    rand_weights = np.zeros((num_portfolios, len(stock_symbols)))
    return_arr = np.zeros(num_portfolios)
    volatility_arr = np.zeros(num_portfolios)
    sharpe_ratio_arr = np.zeros(num_portfolios)
    
    for iteration in range(num_portfolios):
        new_weights = np.array(np.random.random(len(stock_symbols)))
        new_weights = np.array(new_weights / np.sum(new_weights))
        rand_weights[iteration,:] = new_weights
        return_arr[iteration] = np.sum(new_weights * returns.mean()) * 252
        volatility_arr[iteration] = np.sqrt(np.dot(new_weights.T, np.dot(returns.cov() * 252, new_weights)))
        sharpe_ratio_arr[iteration] = return_arr[iteration]/volatility_arr[iteration]
        
    max_sharpe = sharpe_ratio_arr.max()
    max_cord_val = sharpe_ratio_arr.argmax() 
    max_sr_ret = return_arr[max_cord_val]
    max_sr_vol = volatility_arr[max_cord_val]
    st.write(max_sr_vol)
    ideal_weights_arr = rand_weights[max_cord_val,:]
    
    def objective_function(weights):
        return -((1 - risk_tolerance) * np.dot(weights, mean_returns) + risk_tolerance * np.sqrt(np.dot(weights.T, np.dot(covariance, weights))))
    initial_weights = np.ones(len(stock_symbols)) / len(stock_symbols)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for i in range(len(stock_symbols)))
    risk_tolerance = riskTolNumber
    result = minimize(objective_function, ideal_weights_arr, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    
    monte_ports = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatility})
    monte_ports.plot(x='Volatility', y='Return', kind='scatter',figsize=(20, 10))
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel('Expected Portfolio Volatility', fontsize=16)
    plt.ylabel('Expected Portfolio Return', fontsize=16)
    st.pyplot(plt)
    
    for idx in range(len(stock_symbols)):
        st.write(f"Ideal portfolio security weight => {stock_symbols[idx]}: {100 * ideal_weights_arr[idx]:.2f}%")
