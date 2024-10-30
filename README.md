# Volatility-Forecasting

Volatility Forecasting in Financial Markets: A Comparative Analysis of GARCH and SVR Models

Volatility forecasting is an important aspect in finance as it helps in making decisions on risk management, derivative pricing, market making, portfolio selection and many other financial activities. It’s essential as a risk manager would like to know today the likelihood of decline in his  portfolio in future. An option trader would like to know the volatility that can be expected over the future life of the contract. A portfolio manager would like to know if he/she has to sell a stock or a portfolio before it becomes too volatile. As the current trend in trading is increasing amongst students and youngsters, it's important for them to understand the volatility of a stock before making an investment decision. 

Through this work, we can compare the performance of a traditional model such as GARCH (Generalized Autoregressive Conditional Heteroskedasticity), and a machine learning model, Support Vector Regression (SVR), in forecasting the volatility of financial assets. 
That's to be done by developing and implementing GARCH and SVR models for forecasting financial market volatility by using historical price data and predict volatility to compare predicted vs. realized volatility and assess each model’s performance in stable and volatile market conditions using metrics such as Mean Squared Error (MSE) and R-squared. 

### Data Sources: 
Collect historical price data from Yahoo Finance focusing on daily closing prices of a chosen financial asset (e.g., S&P 500) and use VIX (Volatility Index) as a benchmark for evaluating model predictions. 

### Methodology: 
Data Preprocessing: Clean and preprocess historical price data by calculating log returns.
Model Implementation:
- GARCH Model: Use the ARCH package in Python to forecast time-varying volatility.
- SVR Model: Implement using scikit-learn, optimizing parameters via cross-validation.

### Evaluation: Compare the performance of both models against realized volatility using MSE and R-squared as this is regression tasks. 
#### Tools and Technologies: 
- Python, pandas, NumPy, ARCH, and scikit-learn for modeling and data processing.
- Matplotlib, seaborn and Tableau (optional) for data visualization.
