import  keras.optimizers as optim
from keras import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import initializers


def get_model(num_of_output_classes,input_dim, lr=0.01):
	initializer = initializers.HeNormal()
	model = Sequential()
	model.add(Dense(15, input_dim=input_dim, activation='relu', kernel_initializer=initializer))
	model.add(BatchNormalization())
	model.add(Dense(15, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(num_of_output_classes, activation='softmax'))
	optimizer = optim.Adam(learning_rate=lr)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

def get_trained_model(trainX, trainy, n_epochs,num_of_output_classes):
	model = get_model(num_of_output_classes=num_of_output_classes,input_dim=122)
	# train model
	model.fit(trainX, trainy, epochs=n_epochs, verbose=0)
	return model

