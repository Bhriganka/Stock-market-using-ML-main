import numpy as np
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ticker = sys.argv[1]
ticker = "MSFT"
ticker_symbol = f"{ticker}"  # ticker symbol
filename = "data.csv"  # CSV file to save the data
file = "sp500"

if os.path.exists(filename):
    os.remove(filename)

# Download stock data
stock = yf.Ticker(ticker_symbol)
stock = stock.history(period="max")
stock.to_csv(filename)

# Format and prepare data
stock.index = pd.to_datetime(stock.index)
del stock["Dividends"]
del stock["Stock Splits"]

# Add columns for prediction and target
stock["Tomorrow"] = stock["Close"].shift(-1)
stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)
stock = stock.loc["1990-01-03":].copy()

# Split data into training and testing sets
train = stock.iloc[:-100]
test = stock.iloc[-100:]

# Extract predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Define and train the LSTM model
train_x = train[predictors].values
model = Sequential()
model.add(
    LSTM(units=100, return_sequences=True, input_shape=(train_x.shape[1], 1))
)
model.add(LSTM(units=100))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(train[predictors].values, train["Target"].values, epochs=10, batch_size=32)

# Evaluate LSTM model on test set
test_y_pred = model.predict(test[predictors].values)
precision = precision_score(test["Target"].values, test_y_pred > 0.5)
print("Precision Score (LSTM):", precision)

# Perform backtesting using Random Forest classifier


def predict(train, test, predictors, model):
    # Train the model on the training data
    model.fit(train[predictors], train["Target"])

    # Make predictions on the test data
    preds = model.predict(test[predictors])

    # Convert predictions to a pandas Series with the index of the test data
    preds = pd.Series(preds, index=test.index, name="Predictions")

    # Combine the predictions with the actual targets for evaluation
    combined = pd.concat([test["Target"], preds], axis=1)

    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    # Iterate over a range of data points to perform backtesting
    for i in range(start, data.shape[0], step):
        # Split data into training and test sets for the current iteration
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()

        # Generate predictions for the test set using the trained model
        predictions = predict(train, test, predictors, model)

        # Append the predictions to the overall list
        all_predictions.append(predictions)

    # Combine the predictions from all iterations
    combined_predictions = pd.concat(all_predictions)

    return combined_predictions

# Backtest with Random Forest classifier
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictions = backtest(stock, model, predictors)
predictions_counts = predictions["Predictions"].value_counts()
precision = precision_score(predictions["Target"], predictions["Predictions"])

print("Precision Score (Backtesting):", precision)
print(precision)
print(predictions_counts)