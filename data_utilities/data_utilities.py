import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from data_utilities.Sample import Samples
from data_utilities.data_handler import split_data, suffle_data, get_samples
from utilities.preprocessing_utilities import preprocess_all_samples

def get_transformed_data(number_of_samples_to_load=20, one_dataset = False,  load_tess=True, load_savee=False):
    # we get the samples from filesystem
    samples = get_samples(number_of_samples_to_load, encoder=OneHotEncoder,
                          one_dataset=one_dataset,  load_tess=True, load_savee=False)

    # we shuffle the samples
    samples = suffle_data(samples)

    # we split to test and train samples
    test_samples,train_samples = split_data(samples, test_percentage=0.3)

    # we rescale features to be in the same scale
    #train_samples, test_samples = preprocess_all_samples(train_samples, test_samples)

    # we return the features and labels
    trainX = Samples(train_samples).get_features()
    trainY = Samples(train_samples).get_encoded_labels()
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)


    testX = Samples(test_samples).get_features()
    testY = Samples(test_samples).get_encoded_labels()
    testX = np.asarray(testX)
    testY = np.asarray(testY)

    return trainX, trainY, testX, testY



def get_transformed_testing_data():
    X, y = make_classification(n_samples=1000, n_features=40, n_informative=10, n_redundant=90, random_state=1)
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    n_inputs = X.shape[1]
    return X_train, y_train, X_test, y_test