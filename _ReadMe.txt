Kroki do tworzenia sieci i uczenia modelu:

1. Pobranie danych
2. Przygotowanie danych:
	dane muszą być w formie tensorów float64/32 - 2D/3D/4D/5D
	muszą być w tej samej skali
	trzeba je zwektoryzować
3. Tworzenie modelu:
	model = models.sequential()
	model = models.add(layer.dense(n, activation='relu', inputshape=(m,))) 
	n - głębokość warstwy, activation - rodzaj operacji na tensorach, w pośrednich najczęściej relu
	model = models.add(layer.dense(n, activation=' ')) -> warstwa ostatnia, wymiar i aktywacja zmienna w zależności od danych!
	Dla wyniku binarnego: n=1, activation='sigmoid'
	Dla klasyfikacji obrazów n->różne, activation='softmax' https://keras.io/activations/ https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/
	
4. Kompilacja modelu:
	model.compile(optimizer='rmsprop', loss=' ', metrics=['accuracy'])
	matryka zazwyczaj accuracy, optymizer i loss różne w zależności do danych -> https://keras.io/losses/ https://keras.io/optimizers/
	https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
	dla binary- binary_crossentropy, dla kategorii np. categorical_crossentropy
	 
5. Uczenie:
	model.fit(train,labels,epochs,batch_size)
	ważny jest dopór epoch i batch -> overfitting
	
	
Tips:
warstwy pośrednie nigdy nie krótsze od ostatniej!


Typy uczenia maszynowego:

1. Supervised learning
	- mamy dane i anotacje
	- typowe
2. Unsupervised learning
	- do alalizy danych i procesingu
3. Self supervised learning


Dzielenie danych: 
- Train, Eval, Test..


Przygotowywanie danych: 

1. Vektoryzacja danych
- dane jako tensory float (0-1)(czasami int)
2. Normalizacja
- wyrównanie skali, zakresu -> 0-1
- 

Walka z overfitting (regularization)

1. Więcej danych...
2. Zmniejszenie rozmiaru sieci (ilośc uczonych parametrów zależy od ilości warstw i ich wielkości)
3. Prostrzy model wolniej oferfituje
4. Weight regularization - dodawany do warstwy kernel_regulizer=regulizers.l1/l2 https://keras.io/regularizers/ https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
5. Dropout - wyrzucenie randomowych próbek z tensora wyjściowego warstwy (0.2 - 0.5) - w keras dodanie warstwy layers.Dropout(0.5))

Workflow:

Problem type:									Last-layer activation:			Loss function:
Binary classification;							sigmoid;						binary_crossentropy
Multi-class, single-label classification;		softmax;						categorical_crossentropy
Multi-class, multi-label classification;		sigmoid;						binary_crossentropy
Regression to arbitrary values;					None;							mse
Regression to values between 0 and 1;			sigmoid;						mse or binary_crossentropy

1. Utwórz mały model działający lepiej niż random
2. Rozszerzaj aż overfituje
3. Regularize + tuning hyperparametrów (units per layer, lerning rate of oprimizer, epochs)

Z książki:   

1) Define the problem at hand and the data you will be training on; collect this data or annotate it with labels if need be.
2) Choose how you will measure success on your problem. Which metrics will you be monitoring on your validation data?
3) Determine your evaluation protocol: hold-out validation? K-fold validation? Which portion of the data should you use for validation?
4) Develop a first model that does better than a basic baseline: a model that has "statistical power".
5) Develop a model that overfits.
6) Regularize your model and tune its hyperparameters, based on performance on the validation data.



Złożone sieci neuronowe (convolutional neural networks): (wcześniej dense-connected models)
- do analizowania obrazów (computer vision)

convnet:
-operują na tensorach 3d "feature maps" (height, width i depth) (spatial axes i channels)
	np. dla obrazu RGB debth 3 - channels: red, green, blue / dla czarnobiałego depth 1
	
conv2d i maxpolling2d layers
model.add(layers.Conv2D(64, (3,3), activation = 'relu')) # (3,3) - rozmiar okna do uczenia
model.add(layers.MaxPooling2D((2,2)))



	convlavers:				vs 							denselayers:
	- warstwy uczą się wzorców lokalnych 				- uczą się wzorców globalnych
	- szuka wzorców w małych oknach						- ten sam wzorzec wykryje tylko w tej samej części obrazu
	- wykrywa te same wzorce w różnych miejscach
		(translation-invariant)
	- małe wzorce łączy w duży obraz
		(spatial hierarchies of patterns)
	




