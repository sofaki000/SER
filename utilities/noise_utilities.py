import numpy as np # linear algebra
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
from utilities.data_augmentation_utilities import add_noise, pitch, stretch
warnings.filterwarnings('ignore')

def draw_spectogram_for_emotion(df, emotion):
    # px emotion = 'fear'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    show_wave(data, sampling_rate, emotion)
    show_spectogram(data, sampling_rate, emotion)
    # Audio(path)

def extract_mfcc(filename):
    data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    return mfcc

def augment_data_and_extract_mfcc(filename):
    data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)

    pitched_data = pitch(data, sampling_rate)
    streched_data = stretch(data, sampling_rate)
    noisy_data = add_noise(data)

    mfcc = np.mean(librosa.feature.mfcc(y=noisy_data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    mfcc1 = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    mfcc2 = np.mean(librosa.feature.mfcc(y=pitched_data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    mfcc3 = np.mean(librosa.feature.mfcc(y=streched_data, sr=sampling_rate, n_mfcc=40).T, axis=0)

    mfcc_features = np.vstack((mfcc,mfcc2, mfcc3, mfcc1))

    return  mfcc_features

def show_wave(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def show_spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()