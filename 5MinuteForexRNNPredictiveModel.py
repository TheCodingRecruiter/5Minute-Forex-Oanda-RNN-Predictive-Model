# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:17:40 2018

@author: reby
"""


# Recurrent Neural Network



# Part 1 - Data Preprocessing

import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import numpy as np
import matplotlib.pyplot as plt
import datetime
import configparser

started = datetime.datetime.time(datetime.datetime.now())

config = configparser.ConfigParser()

config.read('../config/config_v20.ini')
accountID = config['oanda']['account_id']
access_token = config['oanda']['api_key']

accountID = account_id
client = oandapyV20.API(access_token=access_token)

# Getting the data


######################### 5 MINUTE DATAFRAME ################################


def historydatahead():
    historicaldata = {
          "count": 5000,
          "granularity": "M5"}
    r = instruments.InstrumentsCandles(instrument="EUR_USD",
                                   params=historicaldata)
    client.request(r)
    r.response['candles'][0]['mid']
    r.response['candles'][0]['time']
    r.response['candles'][0]['volume']
    dat = []
    for oo in r.response['candles']:
        dat.append([oo['time'], oo['volume'], oo['mid']['o'], oo['mid']['h'], oo['mid']['l'], oo['mid']['c']])
    global pulledhistorydata
    df = pd.DataFrame(dat)
    df.columns = ['Time', 'Volume', 'Open', 'High', 'Low', 'Close']
    df = df.set_index('Time')
    pulledhistorydata = df.head()
    

def historydata():
    historicaldata = {
          "count": 5000,
          "granularity": "M5"}
    r = instruments.InstrumentsCandles(instrument="EUR_USD",
                                   params=historicaldata)
    client.request(r)
    r.response['candles'][0]['mid']
    r.response['candles'][0]['time']
    r.response['candles'][0]['volume']
    dat = []
    for oo in r.response['candles']:
        dat.append([oo['time'], oo['volume'], oo['mid']['o'], oo['mid']['h'], oo['mid']['l'], oo['mid']['c']])
    global pulledhistorydata
    df = pd.DataFrame(dat)
    df.columns = ['Time', 'Volume', 'Open', 'High', 'Low', 'Close']
    df = df.set_index('Time')
    pulledhistorydata = df
    


############################# 4 MINUTE DATAFRAME ################################


def getdataframe():
    historydatahead()
    historydata()

getdataframe()

forexmodeltraining = pulledhistorydata.iloc[:5000, :]
#forexmodeltesting = pulledhistorydata.iloc[2970:, :]

# Importing the training set
dataset_train = forexmodeltraining
training_set = dataset_train.iloc[:, 2:3].values
#dataset_testing = forexmodeltesting
#testing_set = dataset_testing.iloc[:, 2:3].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 120 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 5000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))


#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 25, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price
#dataset_test = forexmodeltesting
#real_stock_price = dataset_test.iloc[:, 2:3].values



# Getting the predicted stock price of 2017
dataset_total = dataset_train.iloc[:4940, :]['Open']
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#testing_set = testing_set.reshape(-1,1)
#testing_set = sc.transform(testing_set)
#testing_set = sc.inverse_transform(testing_set)

# Visualising the results
#plt.plot(testing_set, color = 'red', label = 'Real EURUSD Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted EURUSD Stock Price')
plt.title('EURUSD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('EURUSD Stock Price')
plt.legend()
plt.show()

predicteddata = pd.DataFrame(predicted_stock_price)
predicteddata = predicteddata.to_excel(r'C:\Users\reby\Desktop\Python Projects\predicteddataexcel\9_5_min.xlsx', index=None, header=True)


print(started)
finished = datetime.datetime.time(datetime.datetime.now())
print(finished)