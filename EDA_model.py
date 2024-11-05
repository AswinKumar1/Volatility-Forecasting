!pip install yfinance pandas numpy arch scikit-learn matplotlib seaborn

!pip install ta

# EDA 

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta

sns.set(style='whitegrid')

# Historical price data for S&P 500 from Yahoo Finance
ticker = '^GSPC'
data = yf.download(ticker, start='2010-01-01', end='2024-01-01', interval='1d')

vix = yf.download('^VIX', start='2010-01-01', end='2024-01-01', interval='1d')
vix = vix[['Close']].rename(columns={'Close': 'VIX'})

# Merge SP500 and VIX data on the Date index
data = data.join(vix, how='inner')

print("First few rows of the dataset:")
print(data.head())

print("\nMissing values in the dataset:")
print(data.isnull().sum())

data.dropna(inplace=True)

# log returns
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

# Creating additional features like 20-day simple moving average, 50-day moving average, RSI and Rolling Std
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['RSI'] = ta.momentum.RSIIndicator(data['Close'].squeeze()).rsi()
data['Volatility'] = data['Close'].rolling(window=20).std()

print("\nCleaned data with additional features:")
print(data.head())

# Summary Statistics
print("\nSummary statistics:")
print(data.describe())

# Closing Price Over Time
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.title('S&P 500 Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Log Returns Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data['Log_Return'].dropna(), bins=50, kde=True)
plt.title('Distribution of Log Returns')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()

# Time Series of Log Returns
plt.figure(figsize=(12, 6))
plt.plot(data['Log_Return'], label='Log Returns', color='orange')
plt.title('Log Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.axhline(0, color='black', lw=2, linestyle='--')
plt.legend()
plt.show()

# Rolling Mean and Standard Deviation
rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()

plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(rolling_mean, label='Rolling Mean (20 days)', color='red')
plt.plot(rolling_std, label='Rolling Std Dev (20 days)', color='green')
plt.title('S&P 500 Close Price with Rolling Mean and Std Dev')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA_20'], label='20-Day SMA', color='red')
plt.plot(data['SMA_50'], label='50-Day SMA', color='green')
plt.title('S&P 500 Close Price with 20-Day and 50-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# RSI
plt.figure(figsize=(12, 6))
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('Relative Strength Index (RSI) Over Time')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()

# Volatility vs VIX
plt.figure(figsize=(12, 6))
plt.plot(data['Volatility'], label='20-Day Rolling Volatility', color='orange')
plt.plot(data['VIX'], label='VIX Index', color='purple', linestyle='--', alpha=0.6)
plt.title('Rolling Volatility for Goldman Sachs and VIX Index')
plt.xlabel('Date')
plt.ylabel('Volatility / VIX')
plt.legend()
plt.show()

# VIX
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='S&P 500 Close', color='blue')
plt.plot(data['SMA_20'], label='20-Day SMA', color='red')
plt.plot(data['SMA_50'], label='50-Day SMA', color='green')
plt.plot(data['VIX'], label='VIX (Volatility Index)', color='purple', linestyle='--')
plt.title('S&P 500 Close, SMAs, and VIX')
plt.xlabel('Date')
plt.ylabel('Price / VIX Level')
plt.legend()
plt.show()


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Comparison of Vanilla Linear Regression and SVR model

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

data = yf.download('^GSPC', start='2010-01-01', end='2024-01-01')
data = data[['Close']]

data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Volatility'] = data['Log_Return'].rolling(window=21).std() * np.sqrt(252)

delta = data['Log_Return'].dropna()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

vix = yf.download('^VIX', start='2010-01-01', end='2024-01-01')
data['VIX'] = vix['Close']

data.dropna(inplace=True)

X = data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'Log_Return', 'VIX']]
y = data['Volatility']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
y_pred_lin_reg = lin_reg_model.predict(X_test)

# Evaluation metrics for Linear Regression
lin_reg_mse = mean_squared_error(y_test, y_pred_lin_reg)
lin_reg_r2 = r2_score(y_test, y_pred_lin_reg)

print("Linear Regression Mean Squared Error:", lin_reg_mse)
print("Linear Regression R-squared Score:", lin_reg_r2)

plt.figure(figsize=(18, 6))
plt.plot(y_test.values, label='True Volatility', color='blue')
plt.plot(y_pred_lin_reg, label='Predicted Volatility (Linear Regression)', color='red', alpha=0.7)
plt.title("Linear Regression: True vs Predicted Volatility")
plt.xlabel("Test Samples")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# Support Vector Regression (SVR)
svr_model = SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.01)
svr_model.fit(X_train, y_train)

y_pred_svr = svr_model.predict(X_test)

# Evaluation metrics for SVR
svr_mse = mean_squared_error(y_test, y_pred_svr)
svr_r2 = r2_score(y_test, y_pred_svr)

print("Support Vector Regression Mean Squared Error:", svr_mse)
print("Support Vector Regression R-squared:", svr_r2)

plt.figure(figsize=(18, 6))
plt.plot(y_test.values, label='True Volatility', color='blue')
plt.plot(y_pred_svr, label='Predicted Volatility (SVR)', color='green', alpha=0.7)
plt.title("Support Vector Regression: True vs Predicted Volatility")
plt.xlabel("Test Samples")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# Summary of models' performance
print("\nSummary of Model Performance:")
print(f"Linear Regression MSE: {lin_reg_mse:.6f}, R-squared: {lin_reg_r2:.6f}")
print(f"Support Vector Regression MSE: {svr_mse:.6f}, R-squared: {svr_r2:.6f}")

# Optimized SVR with GridSearch 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

X = data[['Close', 'SMA_20', 'SMA_50', 'RSI', 'Log_Return', 'VIX']]
y = data['Volatility']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100, 500, 1000],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001],
    'epsilon': [0.001, 0.01, 0.1, 0.25, 0.5, 1]
}

svr = SVR(kernel='rbf')

# Grid search with hyperparameter tuning
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Choosing the best parameters
best_params = grid_search.best_params_
print("Best parameters for SVR:", best_params)

best_svr = grid_search.best_estimator_

y_pred_svr = best_svr.predict(X_test)

mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Results
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_svr, color='red', alpha=0.7, label='Predicted Volatility')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Volatility')
plt.ylabel('Predicted Volatility')
plt.title('SVR Predictions vs Actual Volatility')
plt.grid()
# plt.legend()
plt.show()

print("\nSummary of Model Performance:")
print(f"Optimized SVR MSE: {mse_svr:.6f}, R-squared: {r2_svr:.6f}")

