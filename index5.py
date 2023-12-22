import numpy as np
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
# ticker = sys.argv[1]
ticker = "MSFT"
ticker_symbol = f"{ticker}"  # ticker symbol
filename = "data.csv"  # CSV file to save the data
file = "sp500"

if os.path.exists(filename):
    os.remove(filename)
    # stock = pd.read_csv(filename, index_col=0)


stock = yf.Ticker(ticker_symbol)
stock = stock.history(period="max")
stock.to_csv(filename)

stock.index = pd.to_datetime(stock.index)
del stock["Dividends"]
del stock["Stock Splits"]

# print(stock)
# Plot the data
# print("1)")
stock["Tomorrow"] = stock["Close"].shift(-1)

stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)
stock = stock.loc["1990-01-03":].copy()
# print(stock.head())

#TRAINING THE MODEL
print("1)")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

model = RandomForestClassifier(n_estimators=100, min_samples_split=100,random_state=1)

                    #timeseries data so can't use cross validation

train = stock.iloc[:-50]       # 50 days for testing
test = stock.iloc[-50:]        # 50 days for testing

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])



preds = model.predict(test[predictors])     #prediction score
preds = pd.Series(preds, index=test.index)  

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()

precision = precision_score(test["Target"], preds)
precision = round(precision, 4)
print("Precision Score:", precision)
#print(precision)

plt.title('Predicted vs Actual Targets')
plt.legend(['Actual', 'Predicted'])
# plt.show()


#BACK TESTING
print("2)")
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

# ...

predictions = backtest(stock, model, predictors)
predictions_counts = predictions["Predictions"].value_counts()

precision = precision_score(predictions["Target"], predictions["Predictions"])
print("Precision score after back testing: ", precision)
print("Predictions count:", predictions_counts)


#ADDING ADDITIONAL PREDICTORS TO IMPROVE MODEL (MOVING AVERAGES)
print("3)")
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
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
predictions = backtest(stock, model, new_predictors)
predictions_counts = predictions["Predictions"].value_counts()
precision = precision_score(predictions["Target"], predictions["Predictions"])

print("Precision score after considering the moving averages: ", precision)
print(precision)
print(predictions_counts)

# Define and train the LSTM model
print("4)")
train_x = train[predictors].values
model = Sequential()
model.add(
    LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], 1))
)
model.add(LSTM(units=50))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(train[predictors].values, train["Target"].values, epochs=20, batch_size=32)

# Evaluate LSTM model on test set
test_y_pred_probs = model.predict(test[predictors].values)

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
threshold = 0.5
test_y_pred = (test_y_pred_probs > threshold).astype(int)

# Precision score
precision = precision_score(test["Target"].values, test_y_pred)
print("Precision Score (LSTM):", precision)

# Prediction counts
predictions_counts = pd.Series(test_y_pred.flatten()).value_counts()

# Print prediction counts in the desired format
print("Predictions")
for label, count in predictions_counts.items():
    print(f"{float(label)}: {count}")
print("5)")

# Define training and testing data
train = stock.iloc[:-50]
test = stock.iloc[-50:]
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Create XGBoost model
model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=200,
    min_child_weight=1,
    max_depth=5,
    learning_rate=0.1,
    random_state=1,
)

# Train the model
model.fit(train[predictors], train["Target"])

# Make predictions
preds_probs = model.predict_proba(test[predictors])[:, 1]
preds = (preds_probs > 0.5).astype(int)
preds = pd.Series(preds, index=test.index, name="Predictions")

# Calculate precision score and prediction counts
combined = pd.concat([test["Target"], preds], axis=1)
precision = precision_score(test["Target"], preds)
predictions_counts = preds.value_counts()

# Print prediction counts in the desired format
print("Precision Score (XGBoost):", precision)
print("Predictions")
for label, count in predictions_counts.items():
    print(f"{int(label)}: {count}")

