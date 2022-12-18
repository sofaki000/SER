import keras.optimizers as optim
from keras import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model


def get_model(num_of_output_classes,input_dim, lr=0.01):
	model = Sequential()
	model.add(Dense(128, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(num_of_output_classes, activation='softmax'))
	#optimizer = optim.Adam(learning_rate=lr)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# plot the model
#	plot_model(model, 'dense_model.png', show_shapes=True)
	return model

get_model(num_of_output_classes=7,input_dim=123, lr=0.01)
def get_trained_model(trainX, trainy, n_epochs,num_of_output_classes):
	model = get_model(num_of_output_classes=num_of_output_classes,input_dim=40)
	# train model
	model.fit(trainX, trainy, epochs=n_epochs, verbose=0)
	return model