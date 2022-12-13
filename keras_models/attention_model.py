import  keras.optimizers as optim
import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import Reshape, Dropout, Embedding, BatchNormalization
from keras.callbacks import Callback
from keras.layers import Layer, Lambda, Dot, Activation, Concatenate, LSTM
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Sequential
from keract import get_activations
from matplotlib import pyplot as plt
import os

from utilities.data_utilities import get_transformed_data


def create_argmax_mask(x):
    mask = np.zeros_like(x)
    for i, m in enumerate(x.argmax(axis=1)):
        mask[i, m] = 1
    return mask

class VisualizeAttentionMap(Callback):

    def __init__(self, model, x):
        super().__init__()
        self.model = model
        self.x = x

    def on_epoch_begin(self, epoch, logs=None):
        attention_map = get_activations(self.model, self.x, layer_names='attention_weight')['attention_weight']
        x = self.x[..., 0]
        plt.close()
        fig, axes = plt.subplots(nrows=3, figsize=(10, 8))
        maps = [attention_map, create_argmax_mask(attention_map), create_argmax_mask(x)]
        maps_names = ['attention layer (continuous)', 'attention layer - argmax (discrete)', 'ground truth (discrete)']
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(maps[i], interpolation='none', cmap='jet')
            ax.set_ylabel(maps_names[i] + '\n#sample axis')
            ax.set_xlabel('sequence axis')
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        cbar_ax = fig.add_axes([0.75, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle(f'Epoch {epoch} - training\nEach plot shows a 2-D matrix x-axis: sequence length * y-axis: '
                     f'batch/sample axis. \nThe first matrix contains the attention weights (softmax).'
                     f'\nWe manually apply argmax on the attention weights to see which time step ID has '
                     f'the strongest weight. \nFinally, the last matrix displays the ground truth. The task '
                     f'is solved when the second and third matrix match.')
        plt.draw()
        plt.pause(0.001)

        plt.show()




debug_flag = int(os.environ.get('KERAS_ATTENTION_DEBUG', 0))


# https://github.com/philipperemy/keras-attention-mechanism
class Attention(object if debug_flag else Layer):

    def __init__(self, units=128, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope(self.name if not debug_flag else 'attention'):
            self.attention_score_vec = Dense(input_dim, use_bias=False, name='attention_score_vec')
            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
            self.attention_score = Dot(axes=[1, 2], name='attention_score')
            self.attention_weight = Activation('softmax', name='attention_weight')
            self.context_vector = Dot(axes=[1, 1], name='context_vector')
            self.attention_output = Concatenate(name='attention_output')
            self.attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        if not debug_flag:
            # debug: the call to build() is done in call().
            super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            return self.call(inputs, training, **kwargs)
        else:
            return super(Attention, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @param training: not used in this layer.
        @return: 2D tensor with shape (batch_size, units)
        @author: felixhao28, philipperemy.
        """
        if debug_flag:
            self.build(inputs.shape)
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = self.attention_score_vec(inputs)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(Attention, self).get_config()
        config.update({'units': self.units})
        return config

class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)

        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


# # Create a traditional RNN network
# def create_RNN(hidden_units, output_classes, input_shape, activation):
#     # model = Sequential()
#     # model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
#     # model.add(Dense(units=dense_units, activation=activation[1]))
#     # model.compile(loss='mse', optimizer='adam',metrics='accuracy')
#     # return model
#     # x = Input(shape=input_shape)
#     # RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
#     # outputs = Dense(dense_units, trainable=True, activation='softmax')(RNN_layer)
#     # model = Model(x, outputs)
#     # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
#     # return model
#
#     model = Sequential()
#     # Add an Embedding layer expecting input vocab of size 1000, and output embedding dimension of size 64.
#     model.add(SimpleRNN(128, input_shape=input_shape))
#     # Add a Dense layer with 10 units.
#     model.add(Dense(output_classes))
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics='accuracy')
#     return model



def get_dense_model(num_of_output_classes,input_dim, lr=0.01):
	model = Sequential()
	model.add(Dense(128, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(num_of_output_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
	return model


def get_model_with_attention(num_of_output_classes,input_dim, lr=0.01):
	model = Sequential()
	model.add(Dense(128, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(attention())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, input_dim=input_dim, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(num_of_output_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# ######## ADDING ATTENTION TO MODEL
def create_RNN_with_attention_working(hidden_units, dense_units, input_shape, activation):
    x = Input(shape=input_shape)
    RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(RNN_layer)
    outputs = Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    model = Model(x, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics='accuracy')
    return model


def create_model_with_additive_attention(input_shape, output_classes):
    model_input = Input(shape=input_shape)
    model_input = Reshape(target_shape=(1,40))(model_input)
    x = LSTM(64, return_sequences=True)(model_input)
    x = Attention(units=32)(x)
    x = Dense(output_classes)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam',metrics='accuracy')
    return model


from sklearn import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate
from keras.saving.save import load_model
import numpy as np
from keras import Model
from keras.layers import Layer, Lambda, Dot, Activation, Concatenate, LSTM
import keras.backend as K
from keras.layers import Input, Dense
from matplotlib import pyplot as plt
import os

debug_flag = int(os.environ.get('KERAS_ATTENTION_DEBUG', 0))
class Attention(object if debug_flag else Layer):

    def __init__(self, units=128, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope(self.name if not debug_flag else 'attention'):
            self.attention_score_vec = Dense(input_dim, use_bias=False, name='attention_score_vec')
            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
            self.attention_score = Dot(axes=[1, 2], name='attention_score')
            self.attention_weight = Activation('softmax', name='attention_weight')
            self.context_vector = Dot(axes=[1, 1], name='context_vector')
            self.attention_output = Concatenate(name='attention_output')
            self.attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        if not debug_flag:
            # debug: the call to build() is done in call().
            super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            return self.call(inputs, training, **kwargs)
        else:
            return super(Attention, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @param training: not used in this layer.
        @return: 2D tensor with shape (batch_size, units)
        @author: felixhao28, philipperemy.
        """
        if debug_flag:
            self.build(inputs.shape)
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = self.attention_score_vec(inputs)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(Attention, self).get_config()
        config.update({'units': self.units})
        return config


#works ok
def get_model_with_attention_v3(time_steps, input_dim):
    model_input = Input(shape=(time_steps, input_dim))
    x = LSTM(64, return_sequences=True, name="lstm")(model_input)
    x = Attention(units=32)(x)
    x = Reshape(target_shape=(1,32))(x)
    x = LSTM(64, return_sequences=True, name="lstm_2")(x)
    x = Attention(units=32)(x)
    x = Dense(1)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam', metrics='accuracy')
    return model

#works ok
def get_model_without_attention_v3(time_steps, input_dim):
    model_input = Input(shape=(time_steps, input_dim))
    x = LSTM(64, return_sequences=True, name="lstm")(model_input)
    x = LSTM(64, return_sequences=True, name="lstm_2")(x)
    x = Dense(1)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam', metrics='accuracy')
    return model


def get_model_with_attention_v2(samples, time_steps, input_dim):
    model_input = Input(shape=(samples,time_steps, input_dim))
    model_input = Dense(64)(model_input)
    reshaped_input = Reshape((-1, 40, 60))
    # model_input = tf.reshape(model_input, [-1,])
    x = LSTM(64, return_sequences=True, name="predictions")(reshaped_input)
    x = Attention(units=32)(x)
    x = Dense(5)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam')
    return model