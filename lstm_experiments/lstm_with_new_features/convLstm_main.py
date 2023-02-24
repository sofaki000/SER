import numpy as np
from keras.utils.vis_utils import plot_model

from lstm_experiments.evaluation_utilities import get_evaluation_for_model
from lstm_experiments.lstm_with_new_features.features import extract_features
from lstm_experiments.lstm_with_new_features.model import get_ConvLSTM_model
from utilities.train_utilities import get_callbacks_for_training

train_features, test_features, train_labels, test_labels = extract_features(load_tess=True,
                                                                            load_savee=False,
                                                                            load_crema=False)

# model with ConvLstm layer
model = get_ConvLSTM_model(train_features)
plot_model(model, to_file='convlstm_model.png', show_shapes=True, show_layer_names=True)

f = open(f"conv_lstm_experiments_final.txt", "a")
f.write(f'\n\n------------------------------\n\n')
experiment_name = "model_with_convlstm"
f.write(experiment_name)
f.write(f'\n\ntrain size:{train_features.shape},test_size: {test_features.shape} \n')

training_callbacks = get_callbacks_for_training(best_model_name='best_model_convlstm')
epochs = 30

train_labels =  np.array(train_labels)
test_labels = np.array(test_labels)
rows = 64
cols = 2
num_channels = 1
train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], rows, cols, num_channels)
test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], rows, cols, num_channels)

# train_features = train_features.reshape(len(train_labels), 563, 1 , 64,2)
history = model.fit(train_features, train_labels ,
                    validation_split=0.3,
                    callbacks=training_callbacks,
                    epochs= epochs,
                    batch_size=10,
                    verbose=2)


experiment_name = "tess_model_with_convlstm"

file_name = f"C:\\Users\\Lenovo\Desktop\\τεχνολογία ήχου και εικόνας\\FINAL\\SER\\lstm_experiments\\lstm_with_new_features\\{experiment_name}_"
get_evaluation_for_model(history,f,
                         experiment_name,
                         epochs,
                         file_name,
                         training_callbacks,
                         model,
                         train_features,
                         train_labels,
                         test_features,
                         test_labels)