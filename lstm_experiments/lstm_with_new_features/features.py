import librosa
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from data_utilities.Sample import Samples, Sample
from data_utilities.all_datasets import get_dataframe_with_all_datasets
from data_utilities.data_handler import split_data, suffle_data

output_classes = 7



def extract_features(load_tess,  load_savee, load_crema, return_all=False):
    df_all = get_dataframe_with_all_datasets(load_tess=load_tess, load_savee=load_savee, load_crema=load_crema)
    duration = 3
    sr = 22050
    n_mels = 128
    num_frames = 563

    enc = OneHotEncoder()
    encodings = enc.fit_transform(df_all[['Emotions']]).toarray()
    features = []
    labels = []

    for i in range(df_all['Emotions'].size):
        filename_for_sample = df_all['Path'].iloc[i]

        label_encoded = encodings[i]
        labels.append(label_encoded)
        y, sr = librosa.load(filename_for_sample, sr=None, mono=True)

        # Extract Mel spectrogram feature
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=512)

        log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

        # Pad or truncate to fixed number of frames
        if log_mel_spec.shape[1] < num_frames:
            pad_width = num_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif log_mel_spec.shape[1] > num_frames:
            log_mel_spec = log_mel_spec[:, :num_frames]

        # Append to feature list
        features.append(log_mel_spec)

    # Convert features to numpy array
    features = np.array(features)

    # Reshape features to have batch size as first dimension
    features = features.reshape((features.shape[0], num_frames, n_mels))

    if return_all:
        return features, labels
    split_on_index = int(0.8*len(features))
    train_features = features[0:split_on_index ,:,:]
    test_features = features[split_on_index:,:,:]
    train_labels =labels[0:split_on_index]
    test_labels = labels[split_on_index:]
    return train_features, test_features, train_labels, test_labels