from keras.callbacks import EarlyStopping, ModelCheckpoint

import configuration
from experiments_configuration import attention_exp_config
from utilities.plot_utilities import plot_validation_and_train_loss, plot_validation_and_train_acc


def train_model_and_save_results(model, accuracy_file_name_model,
                                 loss_file_name_model, trainX, trainY, testX,testY, file,best_model_name):

    training_callbacks = get_callbacks_for_training(best_model_name=best_model_name)
    history = model.fit(trainX, trainY, validation_split=0.5,  callbacks=training_callbacks,
                                        epochs=attention_exp_config.epochs,
                                        batch_size=10,
                                        verbose=2)

    train_loss, train_acc = model.evaluate(trainX, trainY)
    test_loss, test_acc = model.evaluate(testX, testY)

    content_train = f'Train: Loss:{train_loss:.2f}, acc:{train_acc:.2f}\n'
    content_test = f'Test: Loss:{test_loss:.2f}, acc:{test_acc:.2f}\n'
    file.write(content_train)
    file.write(content_test)
    print(content_train)
    print(content_test)

    epoch_stopped = training_callbacks[0].stopped_epoch

    plot_validation_and_train_acc(file_name=accuracy_file_name_model,
                                  title=f"Accuracy, epoch stopped:{epoch_stopped}",
                                  history=history)

    plot_validation_and_train_loss(file_name=loss_file_name_model,
                                   title=f"Loss, epoch stopped:{epoch_stopped}",
                                   history=history)

    return history

def get_callbacks_for_training(best_model_name):
    # patience: how many epochs we wait with no improvement before we stop training
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    #{configuration.saved_models_path}
    mc = ModelCheckpoint(f'{best_model_name}.h5',
                         monitor='val_loss',
                         mode='min',
                         save_best_only=True,
                         verbose=1)  # callback to save best model
    cb_list = [es, mc]
    return cb_list


def train_model(model, x_train, y_train, x_test, y_test,n_epochs):
    # these two callbacks are responsible for saving best model and doing early stopping method
    # to avoid overfitting
    # callbacks_for_early_stopping = get_callbacks_for_training()

    # training the model
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=n_epochs,
                        verbose=1)
                        #callbacks=callbacks_for_early_stopping)

    # we print the epoch we stopped at (because we are doing early stopping, we care to see
    # how much we actually trained
    #print(f'Stopped at epoch {callbacks_for_early_stopping[0].stopped_epoch}')
    return history