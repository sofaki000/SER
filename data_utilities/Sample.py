import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Sample():
    def __init__(self, name, features, encoding):
        self.name = name
        self.features = features
        self.encoding = encoding
    def get_name(self):
        return self.name
    def get_encoding(self):
        return self.encoding
    def get_features(self):
        return self.features.reshape(283)


class Samples():
    def __init__(self, samples):
        # array with Sample
        self.samples = samples
    def get_samples_array(self):
        return self.samples
    def split_sample(self, split_percentage):
        data_split = int(len(self.samples)*split_percentage)
        return self.samples[data_split:], self.samples[:data_split]
    def get_labels(self):
        labels = []
        for sample in self.samples:
            labels.append(sample.get_name())
        return labels
    def get_encoded_labels(self):
        encodings = []
        for sample in self.samples:
            encodings.append(sample.get_encoding())

        return encodings
    def get_features(self):
        features = []
        for sample in self.samples:
            features.append(sample.get_features().reshape(283))

        return features
    def get_size(self):
        return len(self.samples)
    # to make class iterable:
    def __iter__(self):
        return SampleIterator(self.samples)


class SampleIterator:
   ''' Iterator class '''
   def __init__(self, samples):
       # Team object reference
       self._samples = samples
       # member variable to keep track of current index
       self._index = 0
   def __next__(self): #''''Returns the next value from team object's lists '''
       if self._index < len(self._samples):
           result = self._samples[self._index]
           self._index +=1
           return result
       return None