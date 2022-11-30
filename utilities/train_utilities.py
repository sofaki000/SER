from keras.callbacks import EarlyStopping, ModelCheckpoint

import configuration


def get_callbacks_for_training(best_model_name="best_model"):
    # patience: how many epochs we wait with no improvement before we stop training
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    mc = ModelCheckpoint(f'{configuration.saved_models_path}{best_model_name}.h5',
                         monitor='val_loss',
                         mode='min',
                         save_best_only=True,
                         verbose=1)  # callback to save best model

    cb_list = [es, mc]
    return cb_list


def train_model(model, x_train, y_train, x_test, y_test,n_epochs):
    # these two callbacks are responsible for saving best model and doing early stopping method
    # to avoid overfitting
    callbacks_for_early_stopping = get_callbacks_for_training()

    # training the model
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks_for_early_stopping)

    # we print the epoch we stopped at (because we are doing early stopping, we care to see
    # how much we actually trained
    print(f'Stopped at epoch {callbacks_for_early_stopping[0].stopped_epoch}')
    return history