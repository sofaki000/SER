#from keras.saving.save import load_model
from tensorflow.python.keras.saving.save import load_model

import configuration
import experiments_configuration.autoencoder_exp_config as autoencoder_config
from keras_models.autoencoder import get_simple_autoencoder
from keras_models.models import get_model
from data_utilities.data_utilities import get_transformed_data
from utilities.plot_utilities import plot_validation_and_train_loss, plot_validation_and_train_acc
from utilities.train_utilities import get_callbacks_for_training

autoencoder_saved_with_compress_path = f'{autoencoder_config.saved_models_path}autoencoder_with_compress.h5'
f = open(f"{configuration.experiments_results_text_path}/test_results.txt", "a")


x_train, y_train, x_test, y_test = get_transformed_data(number_of_samples_to_load=-1)

output_classes = 7 # how many classes we have to classify
n_samples = x_train.shape[0]
n_inputs = x_train.shape[1] # number of features

# this method runs one model. Initially we pass the train data from an autoencoder
# and then pass it to our original model. Then we pass the train data immediately from the
# original model. This is to compare if passing the train data from an autoencoder yields
# better results than without using autoencoder

def run_model_with_autoencoded_features(model):
    ############################# DEFINING AND TRAINING THE ENCODER
    autoencoder_model = get_simple_autoencoder(n_inputs)
    training_callbacks_autoencoder = get_callbacks_for_training(best_model_name="best_autoencoder_model")
    # fit the autoencoder model to reconstruct input
    autoencoder_model.fit(x_train, x_train,  epochs=autoencoder_config.n_epochs,
                                    verbose=2,  validation_split=0.3,  callbacks=training_callbacks_autoencoder)
    autoencoder_model.save(autoencoder_saved_with_compress_path)


    # load the autoencoder model from file
    autoencoder = load_model(autoencoder_saved_with_compress_path)
    X_train_encode = autoencoder.predict(x_train)
    X_test_encode = autoencoder.predict(x_test)

    # ############################# ADDING AUTOENCODER TO EXISTING MODEL:

    # fit the model on the training set
    training_callbacks_model = get_callbacks_for_training(best_model_name="best_model_with_autoencoder")
    history_model_with_autoencoder = model.fit(X_train_encode, y_train, epochs= autoencoder_config.n_epochs,
                                                                validation_split=autoencoder_config.validation_split,callbacks=training_callbacks_model)

    epoch_training_stopped_for_model_with_encoder = training_callbacks_model[0].stopped_epoch
    if epoch_training_stopped_for_model_with_encoder==0:
        epoch_training_stopped_for_model_with_encoder = autoencoder_config.n_epochs

    # make predictions on the test set and calculate classification accuracy
    _, acc_test_with_autoencoder = model.evaluate(X_test_encode, y_test)
    _, acc_train_with_autoencoder = model.evaluate(X_train_encode, y_train)
    acc_test_autoencoder_model_content = f'Test Accuracy with autoencoder:{acc_test_with_autoencoder}\n'
    train_acc_autoencoder_model_content = f'Train Accuracy with autoencoder:{acc_train_with_autoencoder}\n'
    # saving results to file
    f.write(acc_test_autoencoder_model_content)
    f.write(train_acc_autoencoder_model_content)
    # printing results
    print(acc_test_autoencoder_model_content)
    print(train_acc_autoencoder_model_content)

    title_loss_with_autoencoder = f"Model loss with autoencoder,lr:{autoencoder_config.learning_rate},Samples:{n_samples}, Epochs:{epoch_training_stopped_for_model_with_encoder}, Test acc:{acc_test_with_autoencoder:.3f}, Train acc:{acc_train_with_autoencoder:.3f}"
    title_acc_with_autoencoder = f"Model accuracy with autoencoder, Test acc:{acc_test_with_autoencoder:.3f}, Train acc:{acc_train_with_autoencoder:.3f}"

    plot_validation_and_train_loss(autoencoder_config.loss_file_name_autoencoder,
                                   title_loss_with_autoencoder,
                                   history_model_with_autoencoder)
    plot_validation_and_train_acc(autoencoder_config.accuracy_file_name_autoencoder,
                                  title_acc_with_autoencoder,
                                  history_model_with_autoencoder)
    del model
    del X_test_encode
    del X_train_encode

def run_plain_model(model):
    ############################# Without adding autoencoder in model

    # fit the model on the training set
    training_callbacks_model_without_autoencoder = get_callbacks_for_training( best_model_name="best_model_without_autoencoder")
    history_without_autoencoder = model.fit(x_train, y_train, epochs=autoencoder_config.n_epochs,
                                                                validation_split=autoencoder_config.validation_split,
                                                                callbacks=training_callbacks_model_without_autoencoder)

    epoch_training_stopped_for_model_without_autoencoder = training_callbacks_model_without_autoencoder[0].stopped_epoch
    if epoch_training_stopped_for_model_without_autoencoder == 0:
        epoch_training_stopped_for_model_without_autoencoder = autoencoder_config.n_epochs

    # make predictions on the test set and calculate classification accuracy
    _, acc_test_without_autoencoder = model.evaluate(x_test, y_test)

    _, acc_train_without_autoencoder = model.evaluate(x_train, y_train)

    test_acc_model = f'Test Accuracy without autoencoder:{acc_test_without_autoencoder}\n'
    train_acc_model = f'Train Accuracy without autoencoder:{acc_train_without_autoencoder}\n'
    f.write(test_acc_model)
    f.write(train_acc_model)
    print(test_acc_model)
    print(train_acc_model)

    title_loss_without_autoencoder = f"Model loss without autoencoder, lr:{autoencoder_config.learning_rate},Samples:{n_samples}, Epochs:{epoch_training_stopped_for_model_without_autoencoder}, Test acc:{acc_test_without_autoencoder:.3f}, Train acc:{acc_train_without_autoencoder:.3f}"
    title_acc_without_autoencoder = f"Model accuracy without autoencoder, Test acc:{acc_test_without_autoencoder:.3f}, Train acc:{acc_train_without_autoencoder:.3f}"

    plot_validation_and_train_loss(autoencoder_config.loss_file_name_without_autoencoder,
                                   title_loss_without_autoencoder,
                                   history_without_autoencoder)

    plot_validation_and_train_acc(autoencoder_config.accuracy_file_name_without_autoencoder,
                                  title_acc_without_autoencoder,
                                  history_without_autoencoder)


model_with_autoencoder2 = get_model(num_of_output_classes=output_classes, input_dim=n_inputs,  lr=autoencoder_config.learning_rate)
run_model_with_autoencoded_features(model_with_autoencoder2)


model_without_autoencoder3 = get_model(num_of_output_classes=output_classes, input_dim=n_inputs, lr=autoencoder_config.learning_rate)
run_plain_model(model_without_autoencoder3)

f.close()