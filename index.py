# import numpy as np
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt

ticker_symbol = "^GSPC"  # ticker symbol
filename = "sp500.csv"  # CSV file to save the data
file = "sp500"

if os.path.exists(filename):
    sp500 = pd.read_csv(filename, index_col=0)
else:
    sp500 = yf.Ticker(ticker_symbol)
    sp500 = sp500.history(period="max")
    sp500.to_csv(filename)

sp500.index = pd.to_datetime(sp500.index, utc=True)
del sp500["Dividends"]
del sp500["Stock Splits"]

print("1")
# Plot the data

sp500["Tomorrow"] = sp500["Close"].shift(-1)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500.index = pd.to_datetime(sp500.index, utc=True)
start_date = pd.to_datetime("1990-01-03")
sp500 = sp500.loc[start_date:].copy()
# print(sp500.head())


#TRAINING THE MODEL
from sklearn.ensemble import RandomForestClassifier
print("2")
model = RandomForestClassifier(n_estimators=100, min_samples_split=100,random_state=1)

                    #timeseries data so can't use cross validation

train = sp500.iloc[:-100]       # 100 days for testing
test = sp500.iloc[-100:]        # 100 days for testing

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])     #prediction score
preds = pd.Series(preds, index=test.index)  

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()

plt.title('Predicted vs Actual Targets')
plt.legend(['Actual', 'Predicted'])
# plt.show()


#BACK TESTING
print("3")
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
# predictions["Predictions"].value_counts()
print("Precision count ->")
# print(predictions["Predictions"].value_counts())



#ADDING ADDITIONAL PREDICTORS TO IMPROVE MODEL (MOVING AVERAGES)
print("4")
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]

sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
pd.set_option('display.max_columns', None)


print(sp500)
# print(precision_score(predictions["Target"], predictions["Predictions"]))