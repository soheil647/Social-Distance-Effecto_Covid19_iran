
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PreProsses import PreProcess
from keras.models import Sequential
from keras.layers import Flatten, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Dense, Conv2D, MaxPooling2D, \
    Reshape
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd


class my_NN:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def createCNN(self):
        model = Sequential()
        # model.add(Reshape((1, input_shape, 1)))
        model.add(Conv1D(filters=8, kernel_size=5, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Flatten())

        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(1))
        return model


train, target = PreProcess('Tehran').process_input_data()

print(np.array(train).shape)
split = int(len(train.index) * 0.8)
train_x = np.array(train)[:split]
test_x = np.array(train)[split:]

train_y = np.array(target)[:split]
test_y = np.array(target)[split:]

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# dataset = tf.data.Dataset.from_tensor_slices((train.values, target.values))
# for feat, targ in dataset.take(5):
#     print('Features: {}, Target: {}'.format(feat, targ))
# print(tf.constant(train))
# train_dataset = dataset.shuffle(len(train)).batch(1)
# print(train_dataset)

input_shape = (10, 92, 1)
learning_rate = 0.001
epochs = 200
batch_size = 64
# model = my_NN(input_shape).createCNN()

model = Sequential()
model.add(Dense(300, activation="relu"))
model.add(Dense(11, activation="relu"))
model.add(Dense(9, activation="relu"))
model.add(Dense(7, activation="relu"))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
# model.add(Dense(7, activation="relu"))
# model.add(Dense(9, activation="relu"))
# model.add(Dense(11, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))


optimizer = Adam(lr=learning_rate, decay=learning_rate / (epochs))

model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
model_fit = model.fit(
    x=np.array(train_x),
    y=np.array(train_y),
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2)

history_model = model_fit.history

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history_model['loss'])
plt.plot(history_model['val_loss'])
plt.legend(['train', 'validation'])
plt.show()

test_loss = model.evaluate(test_x, test_y)
print("Test Loss: ", test_loss)
# make a prediction
y_predict = model.predict(test_x)
test_y = np.array(test_y).reshape(len(test_y))
y_predict = np.array(y_predict).reshape(len(y_predict))
print(np.array(test_y).reshape(len(test_y)).shape, len(y_predict))
df = pd.DataFrame({'Actual': test_y, 'Predicted': y_predict})
print(df)
print('Mean Squared Error:', mean_absolute_error(test_y, y_predict))