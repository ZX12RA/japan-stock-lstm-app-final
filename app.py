import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def add_technical_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    return df.dropna()

def prepare_data(df, seq_len=60):
    data = df[['Close', 'MA20', 'RSI14', 'MACD']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler, scaled

def train_model(X, y):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(100))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    early = EarlyStopping(monitor='loss', patience=3)
    model.fit(X, y, epochs=10, batch_size=32, callbacks=[early], verbose=0)
    return model

def forecast(model, last_seq, scaler, steps=30):
    preds = []
    for _ in range(steps):
        inp = last_seq.reshape(1, last_seq.shape[0], last_seq.shape[1])
        pred = model.predict(inp, verbose=0)[0][0]
        preds.append(pred)
        new_row = last_seq[-1].copy()
        new_row[0] = pred
        last_seq = np.append(last_seq[1:], [new_row], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

def save_model(model, symbol):
    model.save(os.path.join(MODEL_DIR, f"{symbol}.h5"))

def load_saved_model(symbol):
    path = os.path.join(MODEL_DIR, f"{symbol}.h5")
    return load_model(path) if os.path.exists(path) else None

def get_trade_signal(df):
    last_rsi = df['RSI14'].iloc[-1]
    last_macd = df['MACD'].iloc[-1]
    if last_rsi < 30 and last_macd > 0:
        return "✅ 買いシグナル"
    elif last_rsi > 70 and last_macd < 0:
        return "❌ 売りシグナル"
    else:
        return "⏸ 様子見"

st.title("📈 日本株 LSTM 株価予測アプリ")
symbols = st.multiselect("銘柄コードを選んでください（複数可）",
                         ["7203.T", "6758.T", "9984.T", "9434.T", "9983.T"],
                         default=["7203.T", "6758.T"])
days = st.slider("予測日数（営業日）", 10, 60, 30)

if st.button("予測開始"):
    for symbol in symbols:
        st.header(f"📊 {symbol} の予測と分析")
        df = yf.download(symbol, start="2015-01-01", end="2025-07-01")
        df = add_technical_indicators(df)
        X, y, scaler, scaled = prepare_data(df)

        model = load_saved_model(symbol)
        if model is None:
            model = train_model(X, y)
            save_model(model, symbol)
            st.success(f"{symbol} のモデルを保存しました")
        else:
            st.success(f"{symbol} の保存済みモデルを読み込みました")

        preds = forecast(model, scaled[-60:], scaler, steps=days)
        future_dates = pd.date_range(df.index[-1], periods=days + 1, freq='B')[1:]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Close'], label="実株価")
        ax.plot(future_dates, preds, label="予測", color='red')
        ax.set_title(f"{symbol} 株価予測（LSTM + RSI + MACD）")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        st.markdown(f"### 🚦 売買シグナル: {get_trade_signal(df)}")
        st.dataframe(pd.DataFrame({'日付': future_dates, '予測株価': preds.flatten()}).set_index('日付'))
