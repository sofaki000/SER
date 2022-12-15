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
    data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
    pitched_data = pitch(data, sampling_rate)
    stretched_data = stretch(data)
    noisy_data = add_noise(data)

    return data,pitched_data, stretched_data, noisy_data,sampling_rate


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

def get_sample_from_file(label, data, sampling_rate, encoding):
    features_for_sample = get_features_for_sample(data, sampling_rate)

    return Sample(features=features_for_sample, name=label, encoding=encoding)


def get_features_for_sample(data, sampling_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
    zero_crossing_rate = np.mean(sum(spectral.zero_crossing_rate(y=data, frame_length=512, hop_length=256)))
    freqs, times, D = librosa.reassigned_spectrogram(data, fill_nan=True)
    sc = np.mean(librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs))
    feature_vector = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, np.array([zero_crossing_rate]), np.array([sc])), axis=0)
    feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
    return feature_vector