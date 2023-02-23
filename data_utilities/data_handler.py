import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import configuration
from data_utilities.Sample import Samples
from data_utilities.all_datasets import get_dataframe_with_all_datasets, get_dataframe_with_one_dataset
from utilities.noise_utilities import get_sample_from_file, augment_data
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


def get_samples(load_tess, load_savee, load_crema, number_of_samples_to_load=20,
                encoder=OneHotEncoder, one_dataset=False, use_augmented_data = False):



    if one_dataset:
        df_all = get_dataframe_with_one_dataset(number_of_samples_to_load)
    else:
        df_all = get_dataframe_with_all_datasets(number_of_samples_to_load=number_of_samples_to_load,
                                                load_tess=load_tess, load_savee=load_savee,
                                                         load_crema=load_crema)

    enc = encoder()
    encodings = enc.fit_transform(df_all[['Emotions']]).toarray()

    samples = []


    for i in range(df_all['Emotions'].size):
        filename_for_sample = df_all['Path'].iloc[i]
        label = df_all['Emotions'].iloc[i]

        # data, pitched_data,streched_data, noisy_data,sampling_rate = augment_data(filename_for_sample)
        first_half, second_half, pitched_data_first_half, pitched_data_second_half, \
        stretched_data_first_half, stretched_data_second_half, noisy_data_first_half, \
        noisy_data_second_half, sampling_rate = augment_data(filename_for_sample)

        encoding = encodings[i]
        emotion_sample = get_sample_from_file(label, first_half, second_half, sampling_rate, encoding)
        samples.append(emotion_sample)

        if use_augmented_data:
            emotion_sample_pitched = get_sample_from_file(label, pitched_data_first_half,
                                                          pitched_data_second_half,
                                                          sampling_rate, encoding)

            emotion_sample_streched_data = get_sample_from_file(label, stretched_data_first_half,
                                                                stretched_data_second_half, sampling_rate, encoding)
            emotion_sample_noisy_data = get_sample_from_file(label, noisy_data_first_half,
                                                                    noisy_data_second_half, sampling_rate, encoding)


            samples.append(emotion_sample_pitched)
            samples.append(emotion_sample_streched_data)
            samples.append(emotion_sample_noisy_data)

        if i%20==0:
            print(f'{i+1} samples loaded...')

    return Samples(samples)


def split_data(samples, test_percentage=0.3):
    test_samples,train_samples = samples.split_sample(test_percentage)

    return train_samples, test_samples


def suffle_data(samples):
    samples_array = samples.get_samples_array()

    import random
    random.shuffle(samples_array)
    return Samples(samples_array)




def load_train_and_test_data_for_some_feelings(feelings):
    paths, labels = load_feeling(feelings)
    return get_samples(paths, labels)


def load_feel_test():
    return load_train_and_test_data_for_some_feelings(['angry' , 'happy', 'fear'])