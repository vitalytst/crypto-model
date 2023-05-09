import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_historical_klines(symbol, interval, startTime, endTime):
    url = 'https://testnet.binancefuture.com/fapi/v1/klines'
    params = {'symbol': symbol, 'interval': interval, 'startTime': startTime, 'endTime': endTime, 'limit': 400}
    response = requests.get(url, params=params)
    klines = response.json()
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

def prepare_data(data):
    X = data[['open', 'high', 'low', 'close', 'volume']]
    y = data['close'].shift(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_next_day_price(model, X_test):
    last_day_data = X_test.iloc[-1]
    next_day_data = np.array(last_day_data).reshape(1, -1)
    next_day_price = model.predict(next_day_data)
    return next_day_price[0]
