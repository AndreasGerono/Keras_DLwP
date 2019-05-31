from keras.datasets import boston_housing

(train_data, train_labels), (test_data,test_labels) = boston_housing.load_data()

mean = train_data.mean(axis = 0)	#średnia
train_data -= mean
std = train_data.std(axis=0)	#std - odchylenie standardowe
train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation = 'relu', input_shape = (13, ) ))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(1))	#aktywacja liniowa

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])


x_val = train_data[:100]
partial_x_train = train_data[100:]

y_val = train_labels[:100]
partial_y_train = train_labels[100:]

history = model.fit(partial_x_train, partial_y_train, epochs = 200, batch_size=10, validation_data=(x_val, y_val))

import matplotlib.pylab as plt

acc =  history.history['mean_absolute_error']
val_acc = history.history['val_mean_absolute_error']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)


plt.figure(1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Val loss')
plt.legend()

plt.figure(2)

plt.plot(epochs, acc, 'bo', label = 'Training error')
plt.plot(epochs, val_acc, 'b', label = 'Val error')
plt.legend()

plt.show()
