import numpy as np
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

def add_noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


# for testing
# data, sampling_rate =  librosa.load('data/test/OAF_back_angry.wav', duration=3, offset=0.5)
# fig, ax = plt.subplots(nrows=1, ncols=1)
# librosa.display.waveshow(y=data, sr=sampling_rate)
# Audio('data/test/OAF_back_angry.wav')
# fig.savefig("without_noise.png")
#
# fig, ax = plt.subplots(nrows=1, ncols=1)
# x = add_noise(data)
# librosa.display.waveshow(y=x, sr=sampling_rate)
# fig.savefig("with_noise.png")