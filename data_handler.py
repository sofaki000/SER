import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from utilities.noise_utilities import augment_data, extract_mfcc
import warnings
import os
warnings.filterwarnings('ignore')

def load_feeling(feelings):
    paths = []
    labels = []
    path = 'data/test'
    # path = '../TESS Toronto emotional speech set data'
    if os.path.exists(path) is False:
        raise Exception("Can't find dataa")
    counter =0
    for dirname, _, filenames in os.walk(path):
        counter+=1
        for filename in filenames:
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            if label in feelings:
                labels.append(label.lower())
                paths.append(os.path.join(dirname, filename))
        # if len(paths) == 2800:
        #     break
        if len(paths) == 2:
            break
        if counter==3:
            return paths, labels
    print('Dataset is Loaded')
    return paths, labels

# paths, labels = load_feeling(["angry", "Sad"])
# print(paths)
def loadTestSet():
    paths = []
    labels = []
    path = 'data/test'
    if os.path.exists(path) is False:
        raise Exception("Can't find data")
    counter =0
    for dirname, _, filenames in os.walk(path):
        counter+=1
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            # print(filename)
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
        # if len(paths) == 2800:
        #     break
        if len(paths) == 2:
            break
    print('Dataset is Loaded')
    return paths, labels

def loadDataFromPathAndLabels(paths, labels, encoder=OneHotEncoder ):
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels
    samples_size = len(labels)
    # for each speech sample apply function extract_mfcc
    #features = df['speech'].apply(lambda x: augment_data_and_extract_mfcc(x))
    features = []

    #TODO: Implement features as a pandas dataframe
    for sample in df['speech']:
        augmented_data, sr = augment_data(sample) #returns all data!!
        array_with_mfcc_features = extract_mfcc(augmented_data, sr)
        array_with_rms_features = get_rms_value(augmented_data)
        array_with_zcr_data = get_zcr_data(augmented_data)
        array_with_sc_data = get_spectral_centroid(augmenteddata)
        for array in array_with_mfcc_features:
            features.append(array)
        for array in array_with_rms_features:
            features.append(array)
        for array in array_with_zcr_data:
            features.append(array)
        for array in array_with_zcr_data:
            features.append(array)



    input_features = [x for x in features]
    print(input_features)
    #TODO: Add features to the input features
    input_features = np.array(input_features)  # samples x n_features
    enc = encoder()

    actual_labels = enc.fit_transform(df[['label']])
    #TODO: we have to check the labels are correct for noisy data
    import scipy.sparse as sp
    actual_labels = sp.vstack((actual_labels, actual_labels,actual_labels,actual_labels), format='csr')

    if hasattr(actual_labels, "__len__") is False:
        actual_labels = actual_labels.toarray()
    data_split = (int)(samples_size * 0.7)

    X_test = input_features[:data_split]
    y_test = actual_labels[:data_split]
    X_train = input_features[data_split:]
    y_train = actual_labels[data_split:]
    return X_train, y_train, X_test, y_test


def load_test_data():
    print("loading test data is called")
    paths, labels = loadTestSet()
    return loadDataFromPathAndLabels(paths, labels)


def load_train_and_test_data_for_some_feelings(feelings):
    paths, labels = load_feeling(feelings)
    return loadDataFromPathAndLabels(paths, labels)


def load_feel_test():
    return load_train_and_test_data_for_some_feelings(['angry' , 'happy', 'fear'])