from keras.models import load_model

import configuration
from data_utilities.data_utilities import get_transformed_data
from keras_models.attention_model import Attention
from keras_models.gru_models import get_gru_model
from utilities.train_utilities import train_model_and_save_results

trainX, trainY, testX, testY = get_transformed_data(dataset_number_to_load=4)
output_classes = 7
n_samples = len(trainX)
n_test_samples = len(testX)
n_features = 40
trainX = trainX.reshape(n_samples,n_features,1) # we reshape so it is lstm friendly
testX = testX.reshape(n_test_samples,n_features,1)


##################### without attention
#lstm_model = get_lstm_model(time_steps=1, input_dim=40)
# lstm_model = get_gru_model(output_classes=output_classes)
f = open(f"{configuration.models_experiments_results_text_path}\\pretrained_models.txt", "a")

model = load_model('C:\\Users\\Lenovo\\Desktop\\ser\\SER\\model_without_attention.h5')
#model = load_model('C:\\Users\\Lenovo\\Desktop\\ser\\SER\\lstm_best_model.h5', custom_objects={'attention': Attention})

experiment_name = "pretrained_gru"
accuracy_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_acc_{experiment_name}.png'
loss_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_loss_{experiment_name}.png'

history_without_attention= train_model_and_save_results(model=model,
                             accuracy_file_name_model=accuracy_file_name_model,
                             loss_file_name_model=loss_file_name_model,
                             trainX=trainX, trainY=trainY, testX=testX, testY=testY,
                                                        file=f,best_model_name="pretrained_model")
