import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import configuration
from data_utilities.Sample import Samples
from utilities.noise_utilities import augment_data_and_extract_mfcc
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
def loadTestSet(dataset_number_to_load=0):
    paths = []
    labels = []
    path = configuration.data_path

    if dataset_number_to_load==0: # fortwnei 5 samples ,output_classes=5
        path = f'{configuration.data_path}test'
    elif dataset_number_to_load==1:# fortwnei kamia 100 samples
        path =  path = f'{configuration.data_path}test_data'
    elif dataset_number_to_load==2:# fortwnei kamia 1000 samples, output_classes=6
        path =  path = f'{configuration.data_path}test_data2'
    elif dataset_number_to_load==3:# fortwnei kamia 5000 samples, output_classes=7
        path =  path = f'{configuration.data_path}test_data3'
    elif dataset_number_to_load==4:# fortwnei olo to tess toronto dataset
        path =  path = f'{configuration.data_path}test_data3'

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
        if len(paths) == 2:
            break
    print('Dataset is found. Loading data...')
    return paths, labels


def get_samples(paths, labels, encoder=OneHotEncoder):
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels
    enc = encoder()
    encodings = enc.fit_transform(df[['label']]).toarray()

    map_with_encodings = dict({"angry":encodings[0], "disgust":encodings[1], "fear":encodings[2], "happy":encodings[3], "neutral":encodings[4]})

    samples = []
    i = 0
    for sample in df['speech']:
        encoding = encodings[i]
        emotion_samples = augment_data_and_extract_mfcc(sample,encoding)
        samples = np.concatenate((samples, emotion_samples))
        i +=1

        if i%20==0:
            print(f'{i+1} samples loaded...')

    return samples


def split_data(samples, test_percentage=0.3):
    test_samples,train_samples = samples.split_sample(test_percentage)

    return train_samples, test_samples


def suffle_data(samples):
    samples_array = samples.get_samples_array()
    import random
    random.shuffle(samples_array)
    return Samples(samples_array)

def load_test_data(dataset_number_to_load=0):
    print("loading test data is called")
    paths, labels = loadTestSet(dataset_number_to_load)
    return get_samples(paths, labels)


def load_train_and_test_data_for_some_feelings(feelings):
    paths, labels = load_feeling(feelings)
    return get_samples(paths, labels)


def load_feel_test():
    return load_train_and_test_data_for_some_feelings(['angry' , 'happy', 'fear'])