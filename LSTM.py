from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, date_range
from pandas import DataFrame
from pandas import concat
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN, Average
import time
import numpy as np
import pandas as pd

# convert series to supervised learning
from PreProsses import PreProcess


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


dataset = PreProcess('Tehran').process_input_data()
# print(dataset)

# specify the number of lag hours
n_hours = 4
n_features = 11

# frame as supervised learning
reframed = series_to_supervised(dataset.values, n_hours, 1)
# print(reframed)

# split into train and test sets
values = reframed.values
split = int(len(dataset.index) * 0.8)
train = values[:split, :]
test = values[split:, :]
# print(len(train), len(test))
# split into input and outputs
# print(pd.DataFrame(train))

n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(train_y)
print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# exit()

# design network
model = Sequential()
# model.add(Dense(11, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(9))
# model.add(Dense(7))
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(LSTM(50, recurrent_dropout=0.3))
model.add(Dense(9))
# model.add(Dense(11))
model.add(Dense(1))
model.compile(loss='mae', optimizer='rmsprop', metrics=['mse', 'mae'])
# fit network
start = time.time()
# history = model.fit(train_X, train_y, batch_size=72, epochs=30000, validation_split=0.2, verbose=2,
#                     shuffle=False)

history = model.fit(train_X, train_y, batch_size=16, epochs=100, validation_data=(test_X, test_y), verbose=2)
print("Elapsed Time for fitiing: ", time.time()-start)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()


# make a prediction
y_predict = model.predict(test_X)
y_predict_train = model.predict(train_X)
print(y_predict_train)
print()
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))

print(y_predict)
print(test_y)

# plot history
pyplot.plot(y_predict, 'bo', label='predict')
pyplot.plot(test_y, 'yo', label='real')
pyplot.legend()
pyplot.show()

# plot history
pyplot.plot(y_predict, label='predict')
pyplot.plot(test_y, label='real')
pyplot.legend()
pyplot.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(test_y, y_predict))
print('Test RMSE: %.3f' % rmse)