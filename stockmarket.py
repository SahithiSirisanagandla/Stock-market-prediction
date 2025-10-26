from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from datetime import datetime 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



nifty_50 = {
    "ADANIENT": "Adani Enterprises Ltd.",
    "ADANIPORTS": "Adani Ports and Special Economic Zone Ltd.",
    "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd.",
    "ASIANPAINT": "Asian Paints Ltd.",
    "AXISBANK": "Axis Bank Ltd.",
    "BAJAJ-AUTO": "Bajaj Auto Ltd.",
    "BAJFINANCE": "Bajaj Finance Ltd.",
    "BAJAJFINSV": "Bajaj Finserv Ltd.",
    "BPCL": "Bharat Petroleum Corporation Ltd.",
    "BHARTIARTL": "Bharti Airtel Ltd.",
    "BRITANNIA": "Britannia Industries Ltd.",
    "CIPLA": "Cipla Ltd.",
    "COALINDIA": "Coal India Ltd.",
    "DIVISLAB": "Divi's Laboratories Ltd.",
    "DRREDDY": "Dr. Reddy's Laboratories Ltd.",
    "EICHERMOT": "Eicher Motors Ltd.",
    "GRASIM": "Grasim Industries Ltd.",
    "HCLTECH": "HCL Technologies Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "HDFCLIFE": "HDFC Life Insurance Company Ltd.",
    "HEROMOTOCO": "Hero MotoCorp Ltd.",
    "HINDALCO": "Hindalco Industries Ltd.",
    "HINDUNILVR": "Hindustan Unilever Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "IOC": "Indian Oil Corporation Ltd.",
    "INDUSINDBK": "IndusInd Bank Ltd.",
    "INFY": "Infosys Ltd.",
    "ITC": "ITC Ltd.",
    "JSWSTEEL": "JSW Steel Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
    "LT": "Larsen & Toubro Ltd.",
    "M&M": "Mahindra & Mahindra Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.",
    "NTPC": "NTPC Ltd.",
    "NESTLEIND": "Nestle India Ltd.",
    "ONGC": "Oil & Natural Gas Corporation Ltd.",
    "POWERGRID": "Power Grid Corporation of India Ltd.",
    "RELIANCE": "Reliance Industries Ltd.",
    "SBILIFE": "SBI Life Insurance Company Ltd.",
    "SBIN": "State Bank of India",
    "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
    "TATACONSUM": "Tata Consumer Products Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "TATAMOTORS": "Tata Motors Ltd.",
    "TATASTEEL": "Tata Steel Ltd.",
    "TECHM": "Tech Mahindra Ltd.",
    "TITAN": "Titan Company Ltd.",
    "ULTRACEMCO": "UltraTech Cement Ltd.",
    "UPL": "UPL Ltd.",
    "WIPRO": "Wipro Ltd."
}

sid_obj = SentimentIntensityAnalyzer()

def sentiment_scores(sentence):
    sentiment_dict = sid_obj.polarity_scores(str(sentence))
    return sentiment_dict['compound']

import re

def preprocess_for_vader(text):
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    # 1) Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # 2) Replace currency symbols exactly as you had
    text = text.replace("₹", "rupees").replace("$", "dollars").replace("€", "euros")
    text = text.replace("yoy", "year-over-year")
    # 3) Expand some common contractions (optional but helps sentiment)
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'d": " would", 
        "'ll": " will", "'ve": " have", "'m": " am"
    }
    for contr, full in contractions.items():
        text = text.replace(contr, full)
    # 4) Collapse multiple exclamation/question marks
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    # 5) Remove other punctuation (except ! ? .) – because VADER still uses punctuation weighting
    text = re.sub(r'[^a-z0-9\s\!\?\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

score_df = pd.read_csv("/kaggle/input/india-headlines-news-dataset/india-news-headlines.csv")
score_df.head()

score_df.drop('headline_category', axis = 1, inplace=True)
score_df.head()

score_df['publish_date'] = pd.to_datetime(score_df['publish_date'], format='%Y%m%d')
score_df.rename(columns={'publish_date':'Date'},inplace=True)
score_df

print("Total rows:", len(score_df))
print("Rows with null headline_text:", score_df['headline_text'].isna().sum())
score_df = score_df.dropna(subset=['headline_text'])

score_df = score_df[score_df['Date'].dt.year >= 2015]

print(score_df)
score_df.isnull().sum()
print("Select a stock from Nifty 50:")
stock_name = input()
score_df = score_df[score_df['headline_text'].str.contains(stock_name, case=False, na=False)]

print(score_df)

stock_name = stock_name + " Ltd."  
# stock_name.strip().upper()
print(stock_name)
if stock_name in nifty_50.values():
    file = f"/kaggle/input/stock-data-info/stock info/{stock_name}.csv"
    stock_df = pd.read_csv(file)
print(stock_df)
     

score_df['cleaned_text'] = score_df['headline_text'].fillna('').apply(preprocess_for_vader)
score_df['sentiment_score'] = score_df['cleaned_text'].apply(sentiment_scores)

score_df

stock_df.drop([0,1],inplace=True)
stock_df = stock_df.filter(items=["Date", "Close"])
stock_df = stock_df.dropna(axis=0,how='any') 
stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%d-%m-%Y')

stock_df

score_df = score_df.groupby('Date')['sentiment_score'].mean().reset_index()
# score_df['Date'] = score_df['Date'] + pd.Timedelta(days=1)  
# print(score_df)

x = []
y = []
for i in range(100, len(merged_df)):
    x.append(merged_df['Close'][i-100: i])
    y.append(merged_df['Close'][i]) 
merged_df = pd.merge(stock_df ,score_df , on='Date', how='left')
merged_df

merged_df = merged_df[merged_df['Date'] <= '2023-06-30']
merged_df.reset_index(drop=True, inplace=True)
merged_df

merged_df = merged_df.infer_objects(copy=False)
merged_df.interpolate(method='linear', inplace=True)
merged_df = merged_df.fillna(0)
merged_df

merged_df.isna().sum()
# print(merged_df.describe())
# print(merged_df.info())

print(merged_df['sentiment_score'].shape)
print(merged_df['Close'].shape)
scaler = MinMaxScaler(feature_range=(0, 1))
merged_df['Close'] = merged_df['Close'].astype(float)
merged_df['Close'] = scaler.fit_transform(merged_df[['Close']])

merged_df



n_steps = 100
X, y = [], []

for i in range(n_steps, len(merged_df)):
    # Grab the previous n_steps of both features
    past_close = merged_df['Close'].iloc[i-n_steps:i].values
    past_sent = merged_df['sentiment_score'].iloc[i-n_steps:i].values
    
    # Stack them so each sample is shape (60, 2)
    sequence = np.column_stack((past_close, past_sent))
    X.append(sequence)
    
    # Target is the close price at time i (scaled)
    y.append(merged_df['Close'].iloc[i])

X = np.array(X)  # shape: (2063 - 60, 60, 2) = (2003, 60, 2)
y = np.array(y)  # shape: (2003,)
X = X.astype(float)
y = y.astype(float)
print("X shape:", X.shape)
print("y shape:", y.shape)

X

split_index = int(0.8 * len(X))  # 0.8 * 2003 ≈ 1602
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Training samples:", X_train.shape[0])  # ~1602
print("Test samples:", X_test.shape[0]) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential()

# First LSTM layer (return_sequences=True since we're stacking LSTMs)
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, 2)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Final Dense layer → single output (the predicted close price)
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=20,            # start with 20; adjust later
    batch_size=32,
    validation_split=0.1, # 10% of training used as “validation”
    verbose=1
)

# 1. preds_scaled is already (393, 1), so no change needed there:
print("preds_scaled shape:", preds_scaled.shape)  # (393, 1)

# 2. y_test is (393,), so reshape it to 2D:
print("y_test shape before reshape:", y_test.shape)  # (393,)
y_test_reshaped = y_test.reshape(-1, 1)
print("y_test shape after reshape:", y_test_reshaped.shape)  # (393, 1)

# 3. Now inverse-transform both arrays with price_scaler:
preds_scaled = model.predict(X_test)
preds = scaler.inverse_transform(preds_scaled)            # from (393,1) → (393,1)
y_test_actual = scaler.inverse_transform(y_test_reshaped)  # from (393,1) → (393,1)

# 4. (Optional) Flatten back to 1D if you prefer:
preds = preds.flatten()           # now shape (393,)
y_test_actual = y_test_actual.flatten()  # now shape (393,)

print("Final preds shape:", preds.shape)            # (393,)
print("Final y_test_actual shape:", y_test_actual.shape)  # (393,)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mse = mean_squared_error(y_test_actual, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, preds)

print(f"Test RMSE: {rmse:.2f} rupees")
print(f"Test MAE: {mae:.2f} rupees")

# predict =  np.argmax(model.predict(X_test), axis=-1)

# #Accuracy with the test data
# print('Test Data accuracy: ',accuracy_score(lab, predict)*100)

plt.figure(figsize=(12, 4))
plt.plot(y_test_actual, label='Actual Close Price')
plt.plot(preds, linestyle='--', label='Predicted Close Price')
plt.title('Actual vs. Predicted Close Price (Test Set)')
plt.xlabel('Test Sample Index')
plt.ylabel('Price (₹)')
plt.legend()
plt.show()