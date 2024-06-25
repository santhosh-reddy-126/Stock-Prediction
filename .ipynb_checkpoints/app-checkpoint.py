from flask import Flask, request, jsonify, render_template
import joblib
import yfinance as yf
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
app = Flask(__name__)
model = tf.keras.models.load_model("stock_predictor.keras")
scaler = joblib.load("sc.pkl")
@app.route('/')
def Home():
    
    ticker_symbol = 'GOOG'
    stock_data = yf.Ticker(ticker_symbol)
    stock_data = stock_data.history(period='6mo')
    stock_data=stock_data.tail(100)
    stock_data.reset_index(inplace=True)
    stock_data=pd.DataFrame(stock_data["Close"])
    stock_data_scale = scaler.fit_transform(stock_data)
    stock_data_scale=stock_data_scale.reshape((1,100,1))
    for i in range(10):
        prediction = model.predict(stock_data_scale)
        stock_data_scale[0]=np.append(stock_data_scale[0],prediction,axis=0)[1:]
    predictions = scaler.inverse_transform(stock_data_scale[0])[-10:]
    return render_template("app.html",pred=predictions)


if __name__=="__main__":
    app.run()