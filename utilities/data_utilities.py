from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_utilities.data_handler import load_test_data, split_data
from keras_models.Sample import  Samples
from utilities.preprocessing_utilities import preprocess_all_samples


def get_transformed_data(dataset_number_to_load=0):
    samples = Samples(load_test_data(dataset_number_to_load))
    test_samples,train_samples = split_data(samples, test_percentage=0.3)
    train_samples, test_samples = preprocess_all_samples(train_samples, test_samples)
    return train_samples, test_samples



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