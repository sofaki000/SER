from librosa.feature import spectral
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
from data_utilities.Sample import Sample
from utilities.data_augmentation_utilities import add_noise, pitch, stretch
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
from utilities.data_augmentation_utilities import add_noise, pitch, stretch
warnings.filterwarnings('ignore')

def draw_spectrogram_for_emotion(df, emotion):
    # px emotion = 'fear'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    show_wave(data, sampling_rate, emotion)
    show_spectrogram(data, sampling_rate, emotion)
    # Audio(path)

def augment_data(filename):

    # Load 0.05 seconds of a file, starting 0 seconds in
    first_half, sampling_rate = librosa.load(filename,duration=0.05, offset=0)
    second_half , sampling_rate = librosa.load(filename,duration=0.05, offset=0.05)


    # adding pitched data
    pitched_data_first_half = pitch(first_half, sampling_rate)
    pitched_data_second_half = pitch(second_half, sampling_rate)

    # adding streched data
    stretched_data_first_half = stretch(first_half)
    stretched_data_second_half = stretch(second_half)

    # adding noisy data
    noisy_data_first_half = add_noise(first_half)
    noisy_data_second_half = add_noise(second_half)

    return first_half, second_half,pitched_data_first_half, pitched_data_second_half, \
           stretched_data_first_half, stretched_data_second_half,noisy_data_first_half, noisy_data_second_half, sampling_rate

def show_wave(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

#TODO: check an megethos arxeioy mas ikanopoiei na einai diaforetiko
def show_spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()




def extract_mfcc(data, sampling_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    return mfcc

def get_rms_value(data):
    S, phase = librosa.magphase(librosa.stft(data))
    rms = librosa.feature.rms(S=S) #gets rms from spectrogram of data
    return rms

def get_zcr_data(data): #zero crossing rate - thelei ena extra check
    zcr_in_frame = librosa.feature.zero_crossing_rate(data)
    return zcr_in_frame

def get_spectral_centroid(data):
    S, phase = librosa.magphase(librosa.stft(data))
    sc = librosa.feature.spectral_centroid(S=S)
    return sc

def get_sample_from_file(label, data1, data2, sampling_rate, encoding):
    features_for_sample1 = get_features_for_sample(data1, sampling_rate)
    features_for_sample2 = get_features_for_sample(data2, sampling_rate)

    features = np.concatenate((features_for_sample1, features_for_sample2))

    return Sample(features=features, name=label, encoding=encoding)


def get_features_for_sample(data, sampling_rate):
    result = np.array([])
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')
    result = np.hstack((result, mfcc_delta))  # stacking horizontally

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
    result = np.hstack((result, mfcc_delta2))  # stacking horizontally

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data,  frame_length=512, hop_length=256).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sampling_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    freqs, times, D = librosa.reassigned_spectrogram(data, fill_nan=True)
    sc = np.mean(librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs))
    result = np.hstack((result, np.array([sc])))
    #feature_vector = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, np.array([zero_crossing_rate]), np.array([sc])), axis=0)
    feature_vector = np.reshape(result, (1, len(result)))
    return feature_vector