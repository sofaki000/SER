import numpy as np
from keras.utils.np_utils import to_categorical

from lstm_experiments.lstm_with_new_features.final_lstm import extract_features, get_lstm_model
from utilities.plot_utilities import plot_validation_and_train_acc, plot_validation_and_train_loss
from utilities.train_utilities import get_callbacks_for_training

features, labels = extract_features()

model = get_lstm_model(features)

training_callbacks = get_callbacks_for_training(best_model_name='best_model_lstm')
epochs = 10

labels =  np.array(labels)
history = model.fit(features, labels , validation_split=0.3, callbacks=training_callbacks,
                                        epochs= epochs,
                                        batch_size=10,
                                        verbose=2)


epoch_stopped = training_callbacks[0].stopped_epoch
train_loss, train_acc = model.evaluate(features, labels)
test_loss, test_acc = model.evaluate(features, labels)
if epoch_stopped ==0:
    epoch_stopped = epochs

accuracies_content = f'test acc:{test_acc:.2f}, train acc:{train_acc:.2f}'
experiment_name = "splitting_by_mel_spectograms_frames"
plot_validation_and_train_acc(file_name=f'{experiment_name}_acc', title=f"Accuracy, epoch stopped:{epoch_stopped},{accuracies_content}",
                              history=history)

plot_validation_and_train_loss(file_name=f'{experiment_name}_loss', title=f"Loss, epoch stopped:{epoch_stopped}", history=history)