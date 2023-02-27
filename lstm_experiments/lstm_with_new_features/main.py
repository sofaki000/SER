import numpy as np
from keras.utils.vis_utils import plot_model

from lstm_experiments.evaluation_utilities import get_evaluation_for_model
from lstm_experiments.lstm_with_new_features.features import extract_features
from lstm_experiments.lstm_with_new_features.model import get_lstm_model, get_attention_model, \
    create_multihead_attention_model, get_lstm_model_with_timeseries_layer, \
    get_lstm_model_with_timeseries_layer_with_additive_attention
from utilities.plot_utilities import plot_validation_and_train_acc, plot_validation_and_train_loss
from utilities.train_utilities import get_callbacks_for_training

train_features, test_features, train_labels, test_labels = extract_features(load_tess=False,
                                                                            load_savee=False,
                                                                            load_crema=True)
#model = get_lstm_model(features)

# model with additive attention
model = get_attention_model(train_features)

# model with multihead attention
#model = create_multihead_attention_model(train_features)
# model = get_lstm_model_with_timeseries_layer(train_features)

#model = get_lstm_model_with_timeseries_layer_with_additive_attention(train_features)
plot_model(model, to_file='model_with_additive_attention_crema.png', show_shapes=True, show_layer_names=True)

f = open(f"lstm_experiments_final.txt", "a")
f.write(f'\n\n------------------------------\n\n')
experiment_name = "model_with_attention_crema"
f.write(experiment_name)
f.write(f'\n\ntrain size:{train_features.shape},test_size: {test_features.shape} \n')

training_callbacks = get_callbacks_for_training(best_model_name='best_model_with_timedistributed_layer_with_attention')
epochs = 30

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
history = model.fit(train_features, train_labels ,
                    validation_split=0.3,
                    callbacks=training_callbacks,
                    epochs= epochs,
                    batch_size=10,
                    verbose=2)


experiment_name = "model_with_attention_crema"

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