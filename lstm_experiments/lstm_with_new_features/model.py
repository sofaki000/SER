from keras.layers import Flatten, Concatenate, Reshape, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Permute, Dot, Activation, Lambda
from tensorflow.keras import backend as K

output_classes = 7  # len(np.unique(labels))


def get_lstm_model(features):
    lstm_units = 256
    # Build the model
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(features.shape[1], features.shape[2])))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(output_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Predict the output
    #output = model.predict(features)

    return model


from keras import backend as K
from keras.layers import Layer, Input, LSTM, Dense, Concatenate, dot, Activation
from keras.models import Model

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.dot(x, self.W) + self.b
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def get_attention_model(features):
    inputs = Input(shape=(features.shape[1], features.shape[2]))
    lstm_out = LSTM(256, return_sequences=True)(inputs)
    attention_out = AttentionLayer()(lstm_out)
    outputs = Dense(output_classes, activation='softmax')(attention_out)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model


def create_multihead_attention_model(features, num_heads=4, d_model=128, dff=512, dropout_rate=0.1):
    inputs = Input(shape=(features.shape[1], features.shape[2]))

    x = LSTM(128, return_sequences=True)(inputs)
    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = Concatenate()([x, inputs])
    x = Dropout(0.1)(x)
    x = LSTM(128)(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model







