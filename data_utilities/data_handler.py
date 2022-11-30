import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utilities.noise_utilities import augment_data, extract_mfcc, get_zcr_data, get_rms_value, get_spectral_centroid
import configuration
import warnings
import os
warnings.filterwarnings('ignore')

def load_feeling(feelings):
    paths = []
    labels = []
    path = '../data/test'
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
    path = 'C:/Users/user/Desktop/Speech Emotion Recognition Project/github - ser/SER/data/test'
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
    #X_mfcc = df['speech'].apply(lambda x: augment_data_and_extract_mfcc(x))
    X_mfcc = []
    for sample in df['speech']:
        #gia kathe deigma kanoume augment data & extract features edw
        augmented_data, sr = augment_data(sample) #all data instances for this sample
        #apo kathe sample dld kathe seira tou augmented data kanw extract features
        array_with_mfcc_features = np.apply_along_axis(extract_mfcc, 1, augmented_data, sr)
        array_with_rms_features = np.apply_along_axis(get_rms_value, 1, augmented_data)
        array_with_zcr_data = np.apply_along_axis(get_zcr_data, 1, augmented_data)
        array_with_sc_data =np.apply_along_axis(get_spectral_centroid, 1 ,augmented_data)
        ##edw giati mou ta kanei 3d h kwlomalakia??????
        print("Data", array_with_sc_data.shape, array_with_rms_features.shape, array_with_zcr_data.shape, array_with_sc_data.shape)
        array_with_augmented_features = [] #TODO:change
        for array in array_with_augmented_features:
            X_mfcc.append(array)

    input_features = [x for x in X_mfcc]
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