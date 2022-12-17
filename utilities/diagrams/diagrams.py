import numpy as np # linear algebra
import librosa
import librosa.display as ld
import librosa.display
from IPython.display import Audio,display
import matplotlib.pyplot as plt
import warnings
from utilities.data_augmentation_utilities import add_noise, pitch, stretch, shift
warnings.filterwarnings('ignore')

#def draw_spectrogram_for_emotion(df, emotion):
    # px emotion = 'fear'
#    path = np.array(df['speech'][df['label'] == emotion])[0]
#    data, sampling_rate = librosa.load(path)
#    show_wave(data, sampling_rate, emotion)
#    show_spectrogram(data, sampling_rate, emotion)
#    Audio(path)

#def extract_mfcc(filename):
 #   data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
 #   mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
 #   return mfcc

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 4))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.savefig("waveplot.png")
    plt.show()

def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    #plt.figure(figsize=(12, 3))
    plt.figure(figsize=(10, 4))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+02.0f dB')
    plt.savefig("Spectogram.png")

import numpy as np
def spectrogram(wav):
    D = librosa.stft(wav, n_fft=256, hop_length=128, window='hamming')
    spect, phase = librosa.magphase(D, power=1)
    phase_angle = np.angle(phase)
    return spect

def create_log_spectrogram(data, sr, e):
    #spec=create_spectrogram(data,sr,e)
    spec = spectrogram(data)
    log_spec = librosa.power_to_db(spec ** 2, ref=np.max)
    plt.figure(figsize=(10,10))
    plt.title('Log Spectrogram for audio with {} emotion'.format(e), size=15)
    #librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    librosa.display.specshow(log_spec, y_axis='log', x_axis='time', sr=sr)
    plt.ylabel('Frequency (dB)')
    #**
    plt.imshow(log_spec, origin='lower',vmin=-60)
    plt.colorbar(format='%+02.0f dB')
    plt.show()
    plt.hist(log_spec)
    plt.title('Histogram for audio with {} emotion'.format(e), size=15)
    plt.xlabel('Magnitude (dB)')
    plt.show()
    plt.savefig("Log_spectogram.png")


def create_Mel_spectrogram(data, sr,e):
    # number of mel frequency bands
    S = librosa.feature.melspectrogram(data, sr=sr, n_fft=256, hop_length=128, n_mels=64)
    librosa.display.specshow(S, sr=sr, hop_length=128, x_axis='time', y_axis='mel');
    plt.title('Mel Spectrogram for audio with {} emotion'.format(e), size=15)
    plt.colorbar(format='%+2.0f dB');
    plt.savefig("Mel_spectogram.png")
    #S = librosa.feature.melspectrogram(data, sr=sr, n_fft=512, hop_length=256, n_mels=64)
    #S = librosa.feature.melspectrogram(data, sr=sr, n_fft=2048, hop_length=1024, n_mels=64)


def create_Log_Mel_spectrogram(data, sr,e, filename="Log_Mel_spectogram.png"):
    #mel_spec = librosa.feature.melspectrogram(data, sr=sr, n_fft=256, hop_length=128, n_mels=64)
    #mel_spec_dB = librosa.power_to_db(mel_spec ** 2, ref=np.max)
    S = librosa.feature.melspectrogram(data, sr=sr, n_fft=256, hop_length=128, n_mels=64)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.title('Log Mel Spectrogram for audio with {} emotion'.format(e), fontsize=12, size=15)
    librosa.display.specshow(S_DB, sr=sr, hop_length=128, x_axis='time', y_axis='mel');
    plt.colorbar(format='%+2.0f dB');
    plt.savefig(filename)


#def augment_data(filename):
#    data, sampling_rate = librosa.load(filename, duration=3, offset=0.5)
#    pitched_data = pitch(data, sampling_rate)
#    stretched_data = stretch(data)
#    noisy_data = add_noise(data)
#    shifted_data = shift(data)
#    #print(data.shape, "pitched",pitched_data.shape ,"/nstr", stretched_data.shape, "/nnoisy dt",noisy_data.shape)
#    #input = (data, pitched_data, stretched_data, noisy_data)
#    #TODO: return 4 matrices
#    return data,pitched_data, stretched_data, noisy_data, sampling_rate, shifted_data

def simple_audio(data,sr,e):
    plt.figure(figsize=(10, 4))
    plt.title('Simple Audio for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(y=data, sr=sr)
    plt.savefig("Data before augmentation .png")
    plt.show()
    Audio(path)

def noise_injection_audio(data,sr,e):
    x = add_noise(data)
    plt.figure(figsize=(10, 4))
    plt.title('Audio with {} emotion after noise injection '.format(e), size=15)
    librosa.display.waveshow(y=x, sr=sr)
    plt.savefig("Data after noise injection.png")
    plt.show()
    Audio(x, rate=sr)

def stretching_audio(data,sr,e):
    x = stretch(data,sr)
    plt.figure(figsize=(10, 4))
    plt.title('Audio with {} emotion after stretching '.format(e), size=15)
    librosa.display.waveshow(y=x, sr=sr)
    plt.savefig("Data after noise stretching.png")
    plt.show()
    Audio(x, rate=sr)

    #   Bgainei eutheia **?????

def shift_audio(data,sr,e):
    x = shift(data)
    plt.figure(figsize=(10, 4))
    plt.title('Audio with {} emotion after shifting '.format(e), size=15)
    librosa.display.waveshow(y=x, sr=sr)
    plt.savefig("Data after noise shifting.png")
    plt.show()
    Audio(x, rate=sr)

def pitch_audio(data,sr,e):
    x = pitch(data, sr)
    plt.figure(figsize=(10, 4))
    plt.title('Audio with {} emotion after pitching '.format(e), size=15)
    librosa.display.waveshow(y=x, sr=sr)
    plt.savefig("Data after noise pitching.png")
    plt.show()
    Audio(x, rate=sr)

emotion='disgust'
#path = "C:\\Users\\user\\Documents\\SpeechEmotionRecognition-ERGASIA-2022\\code\\SER\\data\\test\\OAF_back_angry.wav"
#path = "C:\\Users\\user\\Documents\\AVT_SER__Êþäéêáò_×ñýóáò\\SER_xrysas\\data\\test\\OAF_back_angry.wav"
path = "C:\\Users\\Lenovo\\Desktop\\ser\\SER\\data\\TESS Toronto emotional speech set data\\OAF_disgust\\OAF_back_disgust.wav"
#data, sampling_rate = librosa.load(path)
data, sampling_rate = librosa.load(path, duration=3, offset=0.5)
#data, sampling_rate = librosa.load('OAF_back_angry.wav')
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
create_log_spectrogram(data,sampling_rate, emotion)
create_Mel_spectrogram(data,sampling_rate, emotion)
create_Log_Mel_spectrogram(data,sampling_rate, emotion, f"{emotion}_spectogram.png")
simple_audio(data,sampling_rate,emotion)
noise_injection_audio(data,sampling_rate,emotion)
stretching_audio(data,sampling_rate,emotion)
shift_audio(data,sampling_rate,emotion)
pitch_audio(data,sampling_rate,emotion)

#Audio(path)
#display(Audio(path))
#display(Audio(data,sampling_rate))
Audio(data,rate=sampling_rate)

#emotion='disgust'
#path = "C:\\Users\\user\\Documents\\AVT_SER__Êþäéêáò_×ñýóáò\\SER_xrysas\\data\\test\\OAF_back_disgust.wav"
#data, sampling_rate = librosa.load(path)
#create_waveplot(data, sampling_rate, emotion)
#create_spectrogram(data, sampling_rate, emotion)
#create_log_spectrogram(data,sampling_rate, emotion)
#create_Mel_spectrogram(data,sampling_rate, emotion)
#create_Log_Mel_spectrogram(data,sampling_rate, emotion)
#Audio(path)
#display(Audio(path))

#emotion='fear'
#path = "C:\\Users\\user\\Documents\\AVT_SER__Êþäéêáò_×ñýóáò\\SER_xrysas\\data\\test\\OAF_back_fear.wav"
#data, sampling_rate = librosa.load(path)
#create_waveplot(data, sampling_rate, emotion)
#create_spectrogram(data, sampling_rate, emotion)
#create_log_spectrogram(data,sampling_rate, emotion)
#create_Mel_spectrogram(data,sampling_rate, emotion)
#create_Log_Mel_spectrogram(data,sampling_rate, emotion)
#Audio(path)
#display(Audio(path))

#emotion='happy'
#path = "C:\\Users\\user\\Documents\\AVT_SER__Êþäéêáò_×ñýóáò\\SER_xrysas\\data\\test\\OAF_back_happy.wav"
#data, sampling_rate = librosa.load(path)
#create_waveplot(data, sampling_rate, emotion)
#create_spectrogram(data, sampling_rate, emotion)
#create_log_spectrogram(data,sampling_rate, emotion)
#create_Mel_spectrogram(data,sampling_rate, emotion)
#create_Log_Mel_spectrogram(data,sampling_rate, emotion)
#Audio(path)
#display(Audio(path))

#emotion='neutral'
#path = "C:\\Users\\user\\Documents\\AVT_SER__Êþäéêáò_×ñýóáò\\SER_xrysas\\data\\test\\YAF_knock_neutral.wav"
#data, sampling_rate = librosa.load(path)
#create_waveplot(data, sampling_rate, emotion)
#create_spectrogram(data, sampling_rate, emotion)
#create_log_spectrogram(data,sampling_rate, emotion)
#create_Mel_spectrogram(data,sampling_rate, emotion)
#create_Log_Mel_spectrogram(data,sampling_rate, emotion)
#Audio(path)
#display(Audio(path))

#def show_wave(data, sr, emotion):
#    plt.figure(figsize=(10, 4))
#    plt.title(emotion, size=20)
#    librosa.display.waveshow(data, sr=sr)
#    plt.show()
#TODO: check an megethos arxeioy mas ikanopoiei na einai diaforetiko
#def show_spectrogram(data, sr, emotion):
#    x = librosa.stft(data)
#    xdb = librosa.amplitude_to_db(abs(x))
#    plt.figure(figsize=(11, 4))
#    plt.title(emotion, size=20)
#    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
#    plt.colorbar()



def extract_features(data, sampling_rate):
    from librosa.feature import spectral
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    mfcc_delta = librosa.feature.delta(mfcc, order=1, mode='nearest')
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
    zero_crossing_rate = np.mean(sum(spectral.zero_crossing_rate(y=data, frame_length=512, hop_length=256)))
    freqs, times, D = librosa.reassigned_spectrogram(data, fill_nan=True)
    sc = np.mean(librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs))
    feature_vector = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, np.array([zero_crossing_rate]), np.array([sc])), axis=0)
    feature_vector = np.reshape(feature_vector, (1, len(feature_vector)))
    return feature_vector

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