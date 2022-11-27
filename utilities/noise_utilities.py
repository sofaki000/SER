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

#def extract_mfcc(filename):
 #   data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
 #   mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
 #   return mfcc

def augment_data(filename):
    data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
    pitched_data = pitch(data, sampling_rate)
    stretched_data = stretch(data)
    noisy_data = add_noise(data)
    augmented_data = np.vstack(data, pitched_data, stretched_data, noisy_data)
    return augmented_data, sampling_rate

def extract_mfcc(data, sampling_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    return mfcc


def show_wave(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def show_spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

#More features of .wav files

def get_rms_value(data):
    S, phase = librosa.magphase(librosa.stft(data))
    rms = librosa.feature.rms(S=S) #gets rms from spectrogram of data
    return rms

def get_zcr_data(data): #zero crossing rate
    zcr_in_frame = librosa.feature.zero_crossing_rate(data)
    zcr = sum(librosa.zero_crossings(data))
    return zcr_in_frame, zcr

def get_spectral_centroid(data):
    S, phase = librosa.magphase(librosa.stft(data))
    sc = librosa.feature.spectral_centroid(S=S)
    return sc

