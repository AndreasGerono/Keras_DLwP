import numpy as np
from keras.datasets import reuters
from keras.utils import to_categorical

def vectorize_sequences(sequences, dimension=10000):
	result = np.zeros((len(sequences), dimension)) 
	for i, sequence in enumerate(sequences):
		result[i,sequence] = 1
	return result
	
def to_one_hot(labels, dimensions=46):	#to samo co to_caterogical!
	result = np.zeros((len(labels), dimensions))
	for i, label in enumerate(labels):
		result[i,label] = 1
	return result



(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000) #chcemy tylko 10000 najpopularniejszych


x_test = vectorize_sequences(test_data)
y_test = to_categorical(test_labels)

x_train = vectorize_sequences(train_data)
y_train = to_categorical(train_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#x_val = x_train[:1000]
#partial_x_train = x_train[1000:]
#
#y_val = y_train[:1000]
#partial_y_train = y_train[1000:]
#
#history = model.fit(partial_x_train, partial_y_train, epochs = 20, batch_size=512, validation_data=(x_val, y_val))
#
import matplotlib.pylab as plt
#
#acc =  history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(1, len(acc)+1)
#
#
#plt.figure(1)
#
#plt.plot(epochs, loss, 'bo', label = 'Training loss')
#plt.plot(epochs, val_loss, 'b', label = 'Val loss')
#plt.legend()
#
#plt.figure(2)
#
#plt.plot(epochs, acc, 'bo', label = 'Training accuracy')
#plt.plot(epochs, val_acc, 'b', label = 'Val accuracy')
#plt.legend()
#


model.fit(x_train, y_train, epochs=9, batch_size = 512)

print(model.evaluate(x_test, y_test))

prediction = model.predict(x_test[:10])

print(prediction.shape, prediction[0])

for i in range(len(prediction)):
	plt.plot(range(1,47), prediction[i], 'bo', markersize = 2)
plt.show()