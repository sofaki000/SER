from sklearn import preprocessing
from data_handler import load_feel_test, load_test_data
import numpy as np


def get_transformed_data():
    X_train, y_train, X_test, y_test = load_test_data() # load_feel_test()

    # preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler_test = preprocessing.StandardScaler().fit(X_test)
    y_train =np.float32(y_train.toarray())
    y_test =np.float32(y_test.toarray())
    x_train = scaler.transform(X_train)
    x_test = scaler_test.transform(X_test)
    return x_train, y_train, x_test, y_test