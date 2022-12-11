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

from utilities.data_utilities import get_transformed_data

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



def get_model_with_attention(time_steps, input_dim):
    model_input = Input(shape=(time_steps, input_dim))
    x = LSTM(64, return_sequences=True)(model_input)
    x = Attention(units=32)(x)
    x = Dense(5)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam')

    return model

def main():

    # data_x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    # data_y = np.random.uniform(size=(num_samples, output_dim))

    data_x, data_y, testX, testY ,actual_labels= get_transformed_data(dataset_number_to_load=0)
    n_samples = data_x.shape[0]
    n_inputs = data_x.shape[1]  # number of features
    time_steps, input_dim, output_dim =  n_inputs, 1, 1
    # Define/compile the model.
    model = get_model_with_attention(time_steps, input_dim)

    # train.
    model.fit(data_x, data_y, epochs=10)

    # test save/reload model.
    pred1 = model.predict(data_x)
    model_2 = "rnn_model_with_attention.h5"
    model_1 = 'test_model.h5'
    model.save(model_1)
    model_h5 = load_model('test_model.h5', custom_objects={'Attention': Attention})
    model = Model(inputs=model_h5.input, outputs=[model_h5.output, model_h5.get_layer('attention').output])
    pred2, attention_weights = model.predict(data_x)
    np.testing.assert_almost_equal(pred1, pred2)

    import seaborn as sns
    cmap = sns.color_palette("coolwarm", 128)

    plt.xlabel("Training emotions")
    plt.ylabel("Validation emotions")

    yticklabels = []
    for sample in actual_labels:
        name = sample.get_name()
        feats = sample.get_features()

        # we find the corresponding features
        for features in data_x:
            if np.equal(features, feats).all():
                yticklabels.append(name)

    plt.figure(figsize=(30, 10))
    ax1 = sns.heatmap(attention_weights,  cmap=cmap, yticklabels=yticklabels)
    plt.savefig("attention.png")
    print('Success.')


if __name__ == '__main__':
    main()