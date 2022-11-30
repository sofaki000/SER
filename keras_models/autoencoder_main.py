from keras.saving.save import load_model

import experiments_configuration.autoencoder_exp_config as autoencoder_config
from autoencoder import get_autoencoder_model
from keras_models.models import get_model
from utilities.data_utilities import get_transformed_data
from utilities.plot_utilities import plot_validation_and_train_loss, plot_validation_and_train_acc
from utilities.train_utilities import get_callbacks_for_training

autoencoder_saved_with_compress_path = f'{autoencoder_config.saved_models_path}autoencoder_with_compress.h5'
f = open("test_results.txt", "a")

x_train, y_train, x_test, y_test = get_transformed_data(dataset_number_to_load=4)
output_classes = 7 # how many classes we have to classify
n_samples = x_train.shape[0]
n_inputs = x_train.shape[1] # number of features

############################# DEFINING AND TRAINING THE ENCODER
autoencoder_model = get_autoencoder_model(n_inputs, compress=False)
training_callbacks_autoencoder = get_callbacks_for_training(best_model_name="best_autoencoder_model")
# fit the autoencoder model to reconstruct input
autoencoder_history = autoencoder_model.fit(x_train, x_train,
                                epochs=autoencoder_config.n_epochs,
                                verbose=2,
                                validation_split=autoencoder_config.validation_split,
                                callbacks=training_callbacks_autoencoder)
autoencoder_model.save(autoencoder_saved_with_compress_path)

#
# load the autoencoder model from file
autoencoder = load_model(autoencoder_saved_with_compress_path)
X_train_encode = autoencoder.predict(x_train)
X_test_encode = autoencoder.predict(x_test)

# ############################# ADDING AUTOENCODER TO EXISTING MODEL:
model_with_autoencoder = get_model(num_of_output_classes=output_classes, input_dim=40,  lr=autoencoder_config.learning_rate)

# fit the model on the training set
training_callbacks_model = get_callbacks_for_training(best_model_name="best_model_with_autoencoder")
history_model_with_autoencoder = model_with_autoencoder.fit(X_train_encode, y_train,
                                                            epochs= autoencoder_config.n_epochs,
                                                            validation_split=autoencoder_config.validation_split,
                                                            callbacks=training_callbacks_model)

epoch_training_stopped_for_model_with_encoder = training_callbacks_model[0].stopped_epoch
if epoch_training_stopped_for_model_with_encoder==0:
    epoch_training_stopped_for_model_with_encoder = autoencoder_config.n_epochs

# make predictions on the test set and calculate classification accuracy
_, acc_test_with_autoencoder = model_with_autoencoder.evaluate(X_test_encode, y_test)
_, acc_train_with_autoencoder = model_with_autoencoder.evaluate(X_train_encode, y_train)
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
del model_with_autoencoder
del X_test_encode
del X_train_encode

############################# Without adding autoencoder in model
model_without_autoencoder = get_model(num_of_output_classes=output_classes,
                                      input_dim=n_inputs,
                                      lr=autoencoder_config.learning_rate)
# fit the model on the training set
training_callbacks_model_without_autoencoder = get_callbacks_for_training(best_model_name="best_model_without_autoencoder")
history_without_autoencoder = model_without_autoencoder.fit(x_train,
                                                            y_train,
                                                            epochs= autoencoder_config.n_epochs,
                                                            validation_split=autoencoder_config.validation_split,
                                                            callbacks=training_callbacks_model_without_autoencoder)


epoch_training_stopped_for_model_without_autoencoder = training_callbacks_model_without_autoencoder[0].stopped_epoch
if epoch_training_stopped_for_model_without_autoencoder==0:
    epoch_training_stopped_for_model_without_autoencoder = autoencoder_config.n_epochs

# make predictions on the test set and calculate classification accuracy
_, acc_test_without_autoencoder = model_without_autoencoder.evaluate(x_test, y_test)

_, acc_train_without_autoencoder = model_without_autoencoder.evaluate(x_train,y_train)


test_acc_model = f'Test Accuracy without autoencoder:{acc_test_without_autoencoder}\n'
train_acc_model = f'Train Accuracy without autoencoder:{acc_train_without_autoencoder}\n'
f.write(test_acc_model)
f.write(train_acc_model)
print(test_acc_model )
print(train_acc_model )

title_loss_without_autoencoder = f"Model loss without autoencoder, lr:{autoencoder_config.learning_rate},Samples:{n_samples}, Epochs:{epoch_training_stopped_for_model_without_autoencoder}, Test acc:{acc_test_without_autoencoder:.3f}, Train acc:{acc_train_without_autoencoder:.3f}"
title_acc_without_autoencoder= f"Model accuracy without autoencoder, Test acc:{acc_test_without_autoencoder:.3f}, Train acc:{acc_train_without_autoencoder:.3f}"

plot_validation_and_train_loss(autoencoder_config.loss_file_name_without_autoencoder,
                               title_loss_without_autoencoder,
                               history_without_autoencoder)

plot_validation_and_train_acc(autoencoder_config.accuracy_file_name_without_autoencoder,
                              title_acc_without_autoencoder,
                              history_without_autoencoder)


f.close()