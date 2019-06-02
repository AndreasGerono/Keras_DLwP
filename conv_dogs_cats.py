import os,shutil

def mkDir(path):
	try:
		os.mkdir(path)
		print('Directory created!')
	except:
		print('Directory already exists!')

main_path = os.path.dirname(os.path.abspath(__file__)) #path to current working directory

data_path = main_path+'/Dane.nosync/catDogs'
base_path = main_path+'/Dane.nosync/catDogs-small'
mkDir(base_path)
#Dir for partial data
train_path = os.path.join(base_path, 'train')
mkDir(train_path)
train_cats_path = os.path.join(train_path, 'cats')
mkDir(train_cats_path)
train_dogs_path = os.path.join(train_path, 'dogs')
mkDir(train_dogs_path)

validation_path = os.path.join(base_path, 'validation')
mkDir(validation_path)
validation_cats_path = os.path.join(validation_path, 'cats')
mkDir(validation_cats_path)
validation_dogs_path = os.path.join(validation_path, 'dogs')
mkDir(validation_dogs_path)

test_path = os.path.join(base_path, 'test')
mkDir(test_path)
test_cats_path = os.path.join(test_path, 'cats')
mkDir(test_cats_path)
test_dogs_path = os.path.join(test_path, 'dogs')
mkDir(test_dogs_path)

#copy first 1000 cats images to train_cats

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] 	#format wstawia w {} 'i' do stringa
for fname in fnames:
	src = os.path.join(data_path, 'train',fname)
	dst = os.path.join(train_cats_path, fname)
	shutil.copyfile(src, dst)

#copy next 500 cats to validation_path

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)] 	
for fname in fnames:
	src = os.path.join(data_path, 'train',fname)
	dst = os.path.join(validation_cats_path, fname)
	shutil.copyfile(src, dst)
	
#copy next 500 cats to test_cats_path

fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)] 	
for fname in fnames:
	src = os.path.join(data_path, 'train',fname)
	dst = os.path.join(test_cats_path, fname)
	shutil.copyfile(src, dst)

#copy 1000 dogs to train_path

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)] 	#format wstawia w {} 'i' do stringa
for fname in fnames:
	src = os.path.join(data_path, 'train',fname)
	dst = os.path.join(train_dogs_path, fname)
	shutil.copyfile(src, dst)

#copy next 500 dogs to validation_path

fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)] 	
for fname in fnames:
	src = os.path.join(data_path, 'train',fname)
	dst = os.path.join(validation_dogs_path, fname)
	shutil.copyfile(src, dst)
	
#copy next 500 dogs to test_cats_path

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)] 	
for fname in fnames:
	src = os.path.join(data_path, 'train',fname)
	dst = os.path.join(test_dogs_path, fname)
	shutil.copyfile(src, dst)
	

#Creating model

from keras import models,layers

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',  input_shape = (150,150,3) ))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop' ,loss = 'binary_crossentropy', metrics = ['accuracy'])


print(model.summary())

#Data preprocessing (jpg->np.array) - za pomocą generatora

from keras.preprocessing.image import ImageDataGenerator

#train_dataGen = ImageDataGenerator(rescale = 1./255)
#test_dataGen = ImageDataGenerator(rescale = 1./255)

dataGen = ImageDataGenerator(rescale = 1./255)

train_generator = dataGen.flow_from_directory(
	train_path, 
	target_size =(150,150), 
	batch_size = 20,
	class_mode = 'binary' #binarne labele
)

validation_generator = dataGen.flow_from_directory(
	validation_path, 
	target_size =(150,150), 
	batch_size = 20,
	class_mode = 'binary' #binarne labele
)

#Uwaga, generator niewie kiedy jest koniec danych i będzie je generował bez końca
# dla model.fit generator steps_per_epochs (lub validation_steps) = ileDanych/generator.batch_size

history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 30, validation_data = validation_generator, validation_steps = 50)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
steps = range(1, len(acc)+1)


import matplotlib.pylab as plt

plt.plot(acc, steps, 'bo', label = 'train acc')
plt.plot(val_acc, steps, 'b', label = 'val acc')
plt.legend()

plt.figure()
plt.plot(loss,steps, 'bo', label = 'train loss')
plt.plot(val_loss,'b', label = 'val loss'  )
plt.legend()

plt.show()
