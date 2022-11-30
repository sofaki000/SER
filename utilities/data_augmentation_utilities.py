import librosa
import numpy as np
from IPython.lib.display import Audio
from matplotlib import pyplot as plt
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

def add_noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    #returns a smaller array - must transform to return an array with the same size
    #as the other data augmentation processes
    stretched_data =  librosa.effects.time_stretch(data, rate)
    # if len(stretched_data) < len(data):
    #     zeros_appended_size = len(data) - len(stretched_data)
    #     zeros_appended = np.zeros(zeros_appended_size)
    #     stretched_data.append(zeros_appended)
    # elif len(stretched_data) > len(data):
    #     stretched_data = stretched_data[0:len(data)]
    return stretched_data

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