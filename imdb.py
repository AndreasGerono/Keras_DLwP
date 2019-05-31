from keras.datasets import imdb
import numpy as np


def vectorize(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))	#domyślny typ float64
	for i,sequence in enumerate(sequences):		#i - counter/ sequence-value -> do iteracji po listach by mieć counter i wartość
		results[i,sequence] = 1	#pod indeksem sequence ustawia 1
	return results
	


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #data set 10000 najczęstszych słów

x_train = vectorize(train_data)
x_test = vectorize(test_data)

y_train = train_labels.astype('float32')	#zmiana typu na float bo na takich operuje api
y_test = test_labels.astype('float32') 


from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()

model.add(layers.Dense(8,activation='relu', input_shape=(10000,))) #pierwszej warstwie trzeba nadać wymiar jakiego oczekuje
#zwraca tensor o rozmierze (*,16) metoda aktywacji->operacja na tensorze
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid')) #klasyfikacja binarna-> 

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
	
#wydzielamy porcję danych do validacji podczas uczenia
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=1, batch_size=512, validation_data=(x_val, y_val)) #zwraca historię, która jest słownikiem,  można użyć do plota, można sprawdzić przy której rundzie jest peak - za dużo epoch-overfitting

import matplotlib.pylab as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.figure(1)
plt.plot(epochs,loss,'bo', label='Training loss')
plt.plot(epochs,val_loss,'b', label = 'Validation loss')
plt.title('Training and valid loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.figure(2)
plt.plot(epochs,acc,'bo', label='Training accuracy')
plt.plot(epochs,val_acc,'b', label = 'Validation accuracy')
plt.title('Training and valid accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()


#Trening na całych danych ze zoptymalizowanym epoch
model.fit(x_train, y_train, epochs=12, batch_size=512) 

results = model.evaluate(x_test, y_test)



print(results)
#print(model.predict(x_test))
#plt.show()
