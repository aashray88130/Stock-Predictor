#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install yfinance pandas numpy matplotlib scikit-learn


# In[ ]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Fetch Historical Stock Data
def fetch_data(stock_symbol, period='1y'):
    data = yf.download(stock_symbol, period=period)
    return data

# 2. Calculate EMAs (Exponential Moving Averages)
def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

# 3. Calculate Relative Volume (RVOL) and Median Volume
def calculate_rvol(data, window=20):
    data['Avg_Volume'] = data['Volume'].rolling(window=window).mean()
    data['RVOL'] = data['Volume'] / data['Avg_Volume']
    # Calculate median volume of the given window
    data['Median_Volume'] = data['Volume'].rolling(window=window).median()
    return data

# 4. Create Buy and Sell Signals based on EMA crossover
def generate_signals(data):
    data['EMA3'] = calculate_ema(data, 3)
    data['EMA8'] = calculate_ema(data, 8)
    data['Buy_Signal'] = np.where((data['EMA3'] > data['EMA8']) & (data['EMA3'].shift(1) <= data['EMA8'].shift(1)), 1, 0)
    data['Sell_Signal'] = np.where((data['EMA3'] < data['EMA8']) & (data['EMA3'].shift(1) >= data['EMA8'].shift(1)), 1, 0)
    return data

# 5. Feature Engineering for ML (using EMA values as features)
def prepare_features(data):
    features = ['EMA3', 'EMA8']
    X = data[features].dropna()  # Drop rows with NaN values
    y = data['Buy_Signal'].shift(-1).dropna()  # Predict the next Buy_Signal
    X = X.iloc[:-1]  # Align X with y
    return X, y

# 6. Train Machine Learning Model (Random Forest as an example)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return model

# 7. Plot Data with Buy/Sell Signals and Relative Volume
def plot_data(data):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

    # Plot stock price and EMAs
    ax1.plot(data['Close'], label='Close Price', color='blue')
    ax1.plot(data['EMA3'], label='EMA 3', color='red')
    ax1.plot(data['EMA8'], label='EMA 8', color='green')

    # Plot Buy and Sell signals
    buy_signals = data[data['Buy_Signal'] == 1]
    sell_signals = data[data['Sell_Signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Close'] + 2, marker='^', color='green', label='Buy Signal', s=150, edgecolor='black')
    ax1.scatter(sell_signals.index, sell_signals['Close'] - 2, marker='v', color='red', label='Sell Signal', s=150, edgecolor='black')

    ax1.set_title('Stock Price and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot Relative Volume with conditional coloring
    median_rvol = data['RVOL'].dropna().median()  # Median of RVOL
    colors = np.where(data['RVOL'] >= median_rvol, 'green', 'red')
    ax2.bar(data.index, data['RVOL'], color=colors, label='Relative Volume', alpha=0.5)
    ax2.axhline(median_rvol, color='orange', linestyle='--', label='Median RVOL')
    ax2.set_title('Relative Volume')
    ax2.set_ylabel('RVOL')
    ax2.legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

# 8. Main function to run the Stock Predictor
def main():
    stock_symbol = input("Enter the stock symbol: ").upper()  # Input stock symbol
    data = fetch_data(stock_symbol)
    data = generate_signals(data)
    data = calculate_rvol(data)  # Calculate Relative Volume and Median Volume

    X, y = prepare_features(data)
    model = train_model(X, y)

    plot_data(data)

# Run the stock predictor
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




