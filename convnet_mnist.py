from keras.datasets import mnist

((train_data, train_labels) ,(test_data,test_labels)) = mnist.load_data()

from keras import layers,models

print(train_data.shape)

train_data = train_data.reshape((60000,28,28,1))
train_data = train_data.astype('float32')/255

print(train_data.shape)

test_data = test_data.reshape((10000,28,28,1))
test_data = test_data.astype('float32')/255

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1) )) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))	# (3,3) - rozmiar okna do uczenia
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.Flatten())									#sieć klasyfikujaca dense przyjmuje 1d - musimy spłaszczyć
model.add(layers.Dense(64, activation='relu' ))				#jak wcześniej w dense network 
model.add(layers.Dense(10, activation='softmax' ))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_data, train_labels, batch_size = 64, epochs = 4)


print(model.evaluate(test_data,test_labels))


#Wyjściem każdego 