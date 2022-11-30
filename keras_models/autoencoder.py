from keras import Input, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.utils import plot_model
import experiments_configuration.autoencoder_exp_config as autoencoder_config
from utilities.data_utilities import get_transformed_data
from utilities.plot_utilities import plot_validation_and_train_loss, plot_validation_and_train_acc
from utilities.train_utilities import get_callbacks_for_training

f = open("test_results_for_autoencoder.txt", "a")
f.write("------------ new experiment----------------\n")
def get_autoencoder_model(n_inputs, compress=True ):
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs * 2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    #e = Dense(n_inputs)(e)
    e = Dense(512, activation='relu')(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    if compress: # with bottleneck: with compression
        n_bottleneck = round( float(n_inputs) / 2.0)
    else: # without bottle neck: n_inputs (using same number of nodes), without compression
        n_bottleneck = n_inputs
    bottleneck = Dense(n_bottleneck, activation='relu')(e)
    # define decoder, level 1
    d = Dense(n_inputs, activation='relu')(bottleneck)
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
    # plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
    return model



##### to see how well autoencoder performs alone:
autoencoder_saved_with_compress_path = f'{autoencoder_config.saved_models_path}autoencoder.h5'

x_train, y_train, x_test, y_test = get_transformed_data(dataset_number_to_load=0)
output_classes = 5 # how many classes we have to classify
n_samples = x_train.shape[0]
n_inputs = x_train.shape[1] # number of features

############################# DEFINING AND TRAINING THE ENCODER
autoencoder_model = get_autoencoder_model(n_inputs, compress=False)
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

plot_validation_and_train_loss("autoencoder_loss.png",
                               title_loss_with_autoencoder,
                               autoencoder_history)
plot_validation_and_train_acc("autoencoder_acc.png",
                              title_acc_with_autoencoder,
                              autoencoder_history)

f.close()