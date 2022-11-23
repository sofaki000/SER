# evaluate ensemble model
import numpy as np
from keras.utils import to_categorical
from numpy import array, argmax
from sklearn.metrics import accuracy_score

from models import get_model
# fit and evaluate a neural net model on the dataset
from numpy import mean, std

from models import get_trained_model

def evaluate_model(model, trainX, trainy, testX, testy, epochs):
	model.fit(trainX, trainy, epochs=epochs, verbose=0)
	# evaluate the model
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc


def get_evaluation_scores_for_same_model_for_multiple_tries(trainX, trainy, testX, testy, n_repeats, input_dim, output_dim, epochs, lr):
	# repeated evaluation
	scores = list()
	for _ in range(n_repeats):
		model = get_model(num_of_output_classes=output_dim, input_dim=input_dim, lr=lr)
		score = evaluate_model(model, trainX, trainy, testX, testy, epochs)  # to idio montelo, tyxaia pragmata
		print('> %.3f' % score)
		scores.append(score)
	return mean(scores), std(scores), scores



# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy, num_output_classes):
	# select a subset of members
	subset = members[:n_members]
	print(len(subset))
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	# calculate accuracy
	return accuracy_score(testy, to_categorical(yhat, num_classes=num_output_classes))


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# sum across ensemble members
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result

def evaluate_members(members, testX, testy):
    # make prediction
    yhat = ensemble_predictions(members, testX)
    # calculate accuracy
    return accuracy_score(testy, to_categorical(yhat))

def evaluate_ensemble_model(n_members , x_train, y_train, x_test, y_test,num_of_output_classes, n_epochs):
	members = [get_trained_model(x_train, y_train,n_epochs,num_of_output_classes) for _ in range(n_members)]
	# evaluate different numbers of ensembles
	scores = list()
	for i in range(1, n_members + 1):
		test_accuracy = evaluate_n_members(members, i, x_test, y_test, num_of_output_classes)
		print('> %.3f' % test_accuracy)
		scores.append(test_accuracy)

	return mean(scores), std(scores),scores