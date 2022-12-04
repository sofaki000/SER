import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
import os
from PIL import Image
from scipy.fftpack import fft
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os

from utilities.data_utilities import get_transformed_data


def file_search(dirname, ret, list_avoid_dir=[]):
    filenames = os.listdir(dirname)

    for filename in filenames:
        full_filename = os.path.join(dirname, filename)

        if os.path.isdir(full_filename):
            if full_filename.split('/')[-1] in list_avoid_dir:
                continue
            else:
                file_search(full_filename, ret, list_avoid_dir)

        else:
            ret.append(full_filename)

list_files = []
for x in range(5):
    sess_name = 'Session' + str(x+1)
    path = '/content/IEMOCAP_full_release/'+ sess_name + '/sentences/wav/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    print (sess_name + ", #sum files: " + str(len(list_files)))

#extract_feature( list_files, out_file )

def audio2spectrogram(filepath):
    # fig = plt.figure(figsize=(5,5))
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    # print('samplerate',samplerate)
    _, spectrogram = log_specgram(test_sound, samplerate)
    # print(spectrogram.shape)
    # print(type(spectrogram))
    # plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    return spectrogram


def audio2wave(filepath):
    fig = plt.figure(figsize=(5, 5))
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    plt.plot(test_sound)


def log_specgram(audio, sample_rate, window_size=40,
                 step_size=20, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    # print('noverlap',noverlap)
    # print('nperseg',nperseg)
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


N_CHANNELS = 3


def get_3d_spec(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
         delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)


import pandas as pd

no_rows=len(list_files)
index=0
sprectrogram_shape=[]

bookmark=0
extraLabel=0
# for everyFile in list_files:
#   if(everyFile.split('/')[-1].endswith('.wav')):
#     filename=everyFile.split('/')[-1].strip('.wav')
#     # lable=df.loc[df['sessionID']==filename]['label'].values[0]
#     #print('label',lable)
#     if True: #(lable!=-1):
#       #sprectrogram_shape.append(audio2spectrogram(everyFile))
#       spector=audio2spectrogram(everyFile)
#       spector=get_3d_spec(spector)
#       npimg = np.transpose(spector,(2,0,1))
#       input_tensor=torch.tensor(npimg)
#       input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
#       #X, sample_rate = librosa.load(everyFile, res_type='kaiser_fast',sr=22050*2)
#       #sample_rate = np.array(sample_rate)
#       #mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=0)
#       #feature = mfccs
#       docs.append({
#          'fileName':everyFile.split('/')[-1].strip('.wav'),
#          #'feature_mfcc':feature,
#          'sprectrome':input_batch,
#          'label':lable
#               })
#       index+=1
#       print('index',index)
#     else:
#       extraLabel=extraLabel+1
#       print('extraLabel',extraLabel)

import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((12, 12))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print('features', x.shape)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ModifiedAlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        # print('features',x.shape)
        x = torch.flatten(x, start_dim=2)  # a1,a2,a3......al{a of dim c}
        x = torch.sum(x, dim=2)  # a1*alpha1+a2*alpha2+.......+al*alphal
        x = self.classifier(x)
        return x


def modifiedAlexNet(pretrained=False, progress=True, **kwargs):
    model_modified = ModifiedAlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model_modified.load_state_dict(state_dict)
    return model_modified


original_model = alexnet(pretrained=True)
original_dict = original_model.state_dict()
modifiedAlexNet = modifiedAlexNet(pretrained=False)
modified_model_dict = modifiedAlexNet.state_dict()
pretrained_modified_model_dict = {k: v for k, v in original_dict.items() if k in modified_model_dict}
modifiedAlexNet.to('cuda')

x = audio2spectrogram(list_files[40])
x = get_3d_spec(x)
npimg = np.transpose(x, (2, 0, 1))
input_tensor = torch.tensor(npimg)

input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

with torch.no_grad():
    output = modifiedAlexNet(input_batch)
    print(output)
import random

x_train, y_train, x_test, y_test = get_transformed_data(dataset_number_to_load=0)
output_classes = 5
total_length= x_train.shape[0]
train_length=int(.9*total_length)
train_list=x_train
test_list=x_test
print('no of items for train ',len(train_list))
print('no of items for test ',len(test_list))

for name, param in modifiedAlexNet.named_parameters():
    if (param.requires_grad):
        print(name)
    else:
        print('no grad', name)
import torch.optim as optim
from transformers import AdamW
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(modifiedAlexNet.parameters(), lr =  2e-4, eps = 1e-8 )
from transformers import get_linear_schedule_with_warmup

NUM_EPOCHS=16

writer = SummaryWriter(log_dir='/content/')
total_steps = len(train_list) * NUM_EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

total_steps = 1

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

for epoch in range(NUM_EPOCHS):
    modifiedAlexNet.train()
    for every_trainlist in train_list:
        label1 = every_trainlist['label']
        label1 = torch.tensor([label1])
        sprectrome = every_trainlist['sprectrome']
        if (sprectrome.shape[2] > 65):
            optimizer.zero_grad()
            sprectrome = sprectrome.to('cuda')
            label1 = label1.to('cuda')
            modifiedAlexNet.zero_grad()
            output = modifiedAlexNet(sprectrome)
            # print('softmax output ',output)
            loss = criterion(output, label1)
            # print('label1',label1)
            # print('loss',loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modifiedAlexNet.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == label1)
            # print('accuracy.item()',accuracy.item())
            # print('preds',preds)
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == label1)
                    # print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'.format(epoch + 1, total_steps, loss.item(), accuracy.item()))

            total_steps += 1
torch.save(modifiedAlexNet, '/alexnetmodified.pt')
model = torch.load('/alexnetmodified.pt')
model.eval()
model.to('cpu')

y_actu = []
y_pred = []
for every_test_list in test_list:
    label1 = every_test_list['label']
    label1 = torch.tensor([label1])
    sprectrome = every_test_list['sprectrome']
    with torch.no_grad():
        if (sprectrome.shape[2] > 65):
            # sprectrome = sprectrome.to('cuda')
            # label1=label1.to('cuda')
            output = model(sprectrome)
            _, preds = torch.max(output, 1)
            y_actu.append(label1.numpy()[0])
            y_pred.append(preds.numpy()[0])


from sklearn.metrics import confusion_matrix
confusion_matrix(y_actu, y_pred)