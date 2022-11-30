from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_utilities.data_handler import load_test_data
import numpy as np


def get_transformed_data(dataset_number_to_load=0):
    X_train, y_train, X_test, y_test = load_test_data(dataset_number_to_load) # load_feel_test()
    # preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler_test = preprocessing.StandardScaler().fit(X_test)
    y_train =np.float32(y_train.toarray())
    y_test =np.float32(y_test.toarray())
    x_train = scaler.transform(X_train)
    x_test = scaler_test.transform(X_test)
    return x_train, y_train, x_test, y_test


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