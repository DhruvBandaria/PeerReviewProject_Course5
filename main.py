import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,SimpleRNN,GRU
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

#Helper function
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real CISCO Stock Price')
    plt.plot(predicted, color='blue',label='Predicted CISCO Stock Price')
    plt.title('CISCO Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()


data = pd.read_csv('Data/CSCO_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

print(data.head(10))

print('\n\n')
print(data.dtypes)
print('\n\n')
print(data.describe())

x = data[:'2016'].iloc[:,1:2].values
y = data['2017':].iloc[:,1:2].values

data["High"][:'2016'].plot(figsize=(16,4),legend=True)
data["High"]['2017':].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])
plt.title('CISCO stock price')
plt.show()

sc = MinMaxScaler(feature_range=(0,1))
x_scaled = sc.fit_transform(x)

x_train, y_train = [],[]
for i in range(60,2766):
    x_train.append(x_scaled[i-60:i,0])
    y_train.append(x_scaled[i,0])

x_train, y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Model 1- RNN
regressor_RNN = Sequential()
regressor_RNN.add(SimpleRNN(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor_RNN.add(Dropout(0.2))
regressor_RNN.add(SimpleRNN(units=50,return_sequences=True))
regressor_RNN.add(Dropout(0.2))
regressor_RNN.add(SimpleRNN(units=50,return_sequences=True))
regressor_RNN.add(Dropout(0.2))
regressor_RNN.add(SimpleRNN(units=50))
regressor_RNN.add(Dropout(0.2))
regressor_RNN.add(Dense(units=1))

regressor_RNN.compile(optimizer='rmsprop', loss='mean_squared_error')

print('\n\n')
print(regressor_RNN.summary())

#Model 2 - LSTM

regressor_LSTM = Sequential()
regressor_LSTM.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor_LSTM.add(Dropout(0.2))
regressor_LSTM.add(LSTM(units=50,return_sequences=True))
regressor_LSTM.add(Dropout(0.2))
regressor_LSTM.add(LSTM(units=50,return_sequences=True))
regressor_LSTM.add(Dropout(0.2))
regressor_LSTM.add(LSTM(units=50))
regressor_LSTM.add(Dropout(0.2))
regressor_LSTM.add(Dense(units=1))

regressor_LSTM.compile(optimizer='rmsprop', loss='mean_squared_error')

print('\n\n')
print(regressor_LSTM.summary())

#Model 3 - GRU

regressor_GRU = Sequential()
regressor_GRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressor_GRU.add(Dropout(0.2))
regressor_GRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressor_GRU.add(Dropout(0.2))
regressor_GRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressor_GRU.add(Dropout(0.2))
regressor_GRU.add(GRU(units=50, activation='tanh'))
regressor_GRU.add(Dropout(0.2))
regressor_GRU.add(Dense(units=1))

regressor_GRU.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=False),loss='mean_squared_error')
print('\n\n')
print(regressor_GRU.summary())

dataset_total = pd.concat((data["High"][:'2016'],data["High"]['2017':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(y) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)
x_test = []
for i in range(60,311):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

regressor_RNN.fit(x_train,y_train,epochs=50,batch_size=32)

RNN_Predict = regressor_RNN.predict(x_test)
RNN_Predict = sc.inverse_transform(RNN_Predict)

plot_predictions(y,RNN_Predict)

regressor_LSTM.fit(x_train,y_train,epochs=50,batch_size=32)
LSTM_predict = regressor_LSTM.predict(x_test)
LSTM_predict = sc.inverse_transform(LSTM_predict)

plot_predictions(y,LSTM_predict)

regressor_GRU.fit(x_train,y_train,epochs=50,batch_size=32)
GRU_Predict = regressor_GRU.predict(x_test)
GRU_Predict = sc.inverse_transform(GRU_Predict)

plot_predictions(y,GRU_Predict)




