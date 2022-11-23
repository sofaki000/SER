from keras.callbacks import EarlyStopping, ModelCheckpoint


def get_callbacks_for_training():
    # patience: how many epochs we wait with no improvement before we stop training
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint('../models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)  # callback to save best model

    cb_list = [es, mc]
    return cb_list