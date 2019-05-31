from keras.datasets import mnist
from keras import models
from keras import layers

(train_images, train_labels), (test_images, test_labels)  = mnist.load_data()


train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

network = models.Sequential()	#nowa sieć

network.add(layers.Dense(200, activation='relu', input_shape = (28*28,)))	#warstwy sieci
#rozmiar wyjścia i funkcja aktywacyjna

network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=4, batch_size=128)


test_loss, test_acc = network.evaluate(test_images, test_labels)

print(test_acc, test_loss)
