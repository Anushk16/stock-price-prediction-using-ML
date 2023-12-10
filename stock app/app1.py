import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()
st.title('Stock Trend prediction')

start = st.date_input('Enter the starting date', datetime(2010, 11, 1))
end = st.date_input('Enter the ending date', datetime(2022, 12, 12))

start = start.strftime('%Y-%m-%d')
end = end.strftime('%Y-%m-%d')

stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB']
user_input = st.selectbox('Select Stock Ticker', stock_tickers, index=0)
st.write('You selected:', user_input)

df=pdr.get_data_yahoo(user_input,start,end)




#Describing Data
st.subheader(f'Data from Start Date to End Date of: {user_input}')
st.write(df.describe()) 

#Visualizations

st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize =  (12,6))
plt.plot(ma100,'r')
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


 #Splitting Data into Training and Testing


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])



scaler = MinMaxScaler(feature_range=(0,1)) 
data_training_array = scaler.fit_transform(data_training)


#Splitting data into x_train and y_train
x_train=[]
y_train=[]


for i in range(100, data_training_array.shape[0]):
   x_train.append(data_training_array[i-100: i])
   y_train.append(data_training_array[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

#Load my model
model = load_model('E:/Projects/stock-price-prediction-using-ML-main/python 1/my_model.keras')


#Testing Part
past_100_days =  pd.DataFrame(data_training.tail(100))
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted =   model.predict(x_test)
scaler = scaler.scale_

scale_factor=1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label ='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

