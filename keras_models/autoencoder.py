from keras import Input, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.utils import plot_model


def get_autoencoder_model(n_inputs, compress=True ):
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    if compress: # with bottleneck: with compression
        n_bottleneck = round( float(n_inputs) / 2.0)
    else: # without bottle neck: n_inputs (using same number of nodes), without compression
        n_bottleneck = n_inputs
    bottleneck = Dense(n_bottleneck)(e)
    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs * 2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    #with mean squared error: model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # plot the autoencoder
    plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
    return model

