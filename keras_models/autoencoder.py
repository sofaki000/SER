from keras import Input, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Conv1D, MaxPooling1D, Conv1DTranspose
import tensorflow as tf

import configuration
import experiments_configuration.autoencoder_exp_config as autoencoder_config
from data_utilities.data_utilities import get_transformed_data
from utilities.plot_utilities import plot_validation_and_train_loss, plot_validation_and_train_acc
from utilities.train_utilities import get_callbacks_for_training

f = open(f"{configuration.experiments_results_text_path}/test_results_for_autoencoder.txt", "a")

f.write("------------ new experiment----------------\n")

def get_cnn_autoencoder(n_inputs):
    input = Input(shape=(n_inputs,))
    # Encoder
    x = tf.expand_dims(input, axis=-1)
    x = Conv1D(32, 2, activation="relu", padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    # x = Conv1D(64, 2, activation="relu", padding="same")(x)
    # x = MaxPooling1D(2, padding="same")(x)

    # Decoder
    x = Conv1DTranspose(32, 2, strides=2, activation="relu", padding="same")(x)
    # x = Conv1DTranspose(64, 2 , strides=2, activation="relu", padding="same")(x)
    x = Conv1D(1, 2, activation="sigmoid", padding="same")(x)
    # decoded = Dense(n_inputs, activation='sigmoid')(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return autoencoder

def get_simple_autoencoder(n_inputs):
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = Input(shape=(n_inputs,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(n_inputs, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return autoencoder

def get_autoencoder(n_inputs):
    visible = Input(shape=(n_inputs,))
    e = Dense(n_inputs * 2)(visible)
    bottleneck = Dense(512, activation='linear')(e)

    d = Dense(n_inputs, activation='linear')(bottleneck)

    d = Dense(n_inputs * 2, activation='linear')(d)

    output = Dense(n_inputs, activation='softmax')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_autoencoder_model(n_inputs, compress=True ):
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    #e = Dense(n_inputs)(e)
    e = Dense(512, activation='linear')(e)
    e = BatchNormalization()(e)
    #e = LeakyReLU()(e)
    # bottleneck
    if compress: # with bottleneck: with compression
        n_bottleneck = round( float(n_inputs) / 2.0)
    else: # without bottle neck: n_inputs (using same number of nodes), without compression
        n_bottleneck = n_inputs
    # bottleneck = Dense(n_bottleneck)(e)
    # bottleneck = Dense(n_bottleneck, activation='tanh')(e)
    bottleneck = Dense(128, activation='linear')(e)
    # define decoder, level 1
    d = Dense(n_inputs, activation='linear')(bottleneck)
    d = BatchNormalization()(d)
    #d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs * 2, activation='linear')(d)
    d = BatchNormalization()(d)
    #d = LeakyReLU()(d)
    # output layer
    # output = Dense(n_inputs, activation='linear')(d)
    output = Dense(n_inputs, activation='softmax')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    #with mean squared error: model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # plot the autoencoder
    # plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
    return model


# this method tests how good our autoencoder models are
def test_autoencoder():
    ##### to see how well autoencoder performs alone:
    x_train, y_train, x_test, y_test = get_transformed_data(dataset_number_to_load=2)
    output_classes = 5 # how many classes we have to classify
    n_samples = x_train.shape[0]
    n_inputs = x_train.shape[1] # number of features

    ############################# DEFINING AND TRAINING THE ENCODER
    autoencoder_model = get_cnn_autoencoder(n_inputs)
    training_callbacks_autoencoder = get_callbacks_for_training(best_model_name="best_autoencoder_model")

    # fit the autoencoder model to reconstruct input
    autoencoder_history = autoencoder_model.fit(x_train,  x_train,
                                    epochs=autoencoder_config.n_epochs,
                                    verbose=2,
                                    validation_split=autoencoder_config.validation_split,
                                    callbacks=training_callbacks_autoencoder)

    epoch_training_stopped_for_model_with_encoder = training_callbacks_autoencoder[0].stopped_epoch
    if epoch_training_stopped_for_model_with_encoder==0:
        epoch_training_stopped_for_model_with_encoder = autoencoder_config.n_epochs

    # make predictions on the test set and calculate classification accuracy
    _, acc_test_autoencoder = autoencoder_model.evaluate(x_test, x_test)
    _, acc_train_autoencoder = autoencoder_model.evaluate(x_train, x_train)
    acc_test_autoencoder_model_content = f'Test Accuracy autoencoder:{acc_test_autoencoder}\n'
    train_acc_autoencoder_model_content = f'Train Accuracy autoencoder:{acc_train_autoencoder}\n'
    # saving results to file
    f.write(acc_test_autoencoder_model_content)
    f.write(train_acc_autoencoder_model_content)
    # printing results
    print(acc_test_autoencoder_model_content)
    print(train_acc_autoencoder_model_content)

    title_loss_with_autoencoder = f"autoencoder loss ,lr:{autoencoder_config.learning_rate},Samples:{n_samples}, Epochs:{epoch_training_stopped_for_model_with_encoder}, Test acc:{acc_test_autoencoder:.3f}, Train acc:{acc_train_autoencoder:.3f}"
    title_acc_with_autoencoder = f"autoencoder accuracy, Test acc:{acc_test_autoencoder:.3f}, Train acc:{acc_train_autoencoder:.3f}"

    plot_validation_and_train_loss("autoencoder_loss_cnn_less_neurons_d2.png",
                                   title_loss_with_autoencoder,
                                   autoencoder_history)

    plot_validation_and_train_acc("autoencoder_acc_cnn_less_neurons_d2.png",
                                  title_acc_with_autoencoder,
                                  autoencoder_history)

    f.close()