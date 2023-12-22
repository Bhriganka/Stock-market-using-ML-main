# var train = pd.read_csv("train.csv")
import pandas as pd
import yfinance as yf
from index4 import ticker_symbol
# Calculate moving averages for different time horizons
stock = yf.Ticker(ticker_symbol)
from sklearn.ensemble import RandomForestClassifier
def backtest(stock, model, predictors):
    predictions = []
    for i in range(100, len(stock), 100):
        train = stock.iloc[:i]
        test = stock.iloc[i:i+100]
        preds = predict(train, test, predictors, model)
        predictions.append(preds)
    predictions = pd.concat(predictions)
    return predictions
from sklearn.metrics import precision_score






horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = stock.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    stock[ratio_column] = stock["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    stock[trend_column] = stock.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]

stock = stock.dropna(subset=stock.columns[stock.columns != "Tomorrow"])
stock.index.name = ""
pd.set_option('display.max_columns', None)


#BACKTESTING WITH NEW PREDICTORS
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

predictions = backtest(stock, model, new_predictors)

precision = precision_score(predictions["Target"], predictions["Predictions"])
