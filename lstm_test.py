import configuration
from keras_models.gru_models import get_gru_model_with_attention, get_gru_model
from data_utilities.data_utilities import get_transformed_data
from keras_models.lstm_models import get_lstm_model_with_dropout_and_attention, get_lstm_model_with_dropout
from utilities.plot_utilities import plot_validation_and_train_acc_2_models
from utilities.train_utilities import train_model_and_save_results

trainX, trainY, testX, testY = get_transformed_data(dataset_number_to_load=4)
output_classes = 7
n_samples = len(trainX)
n_test_samples = len(testX)
n_features = 40
trainX = trainX.reshape(n_samples,n_features,1) # we reshape so it is lstm friendly
testX = testX.reshape(n_test_samples,n_features,1)

f = open(f"{configuration.models_experiments_results_text_path}\\models_lstm_architecture.txt", "a")
experiment_name = "lstm_256_256_512_512neurons"
f.write(f'{experiment_name}\n')

##################### without attention
# lstm_model = get_gru_model(output_classes=output_classes)
# lstm_model_with_attention = get_gru_model_with_attention(output_classes=output_classes)

lstm_model_with_attention = get_lstm_model_with_dropout_and_attention(output_classes=output_classes)
lstm_model = get_lstm_model_with_dropout(output_classes=output_classes)
# lstm layer expects input in shape: samples, time series, features

#lstm_model = get_lstm_model_with_attention(time_steps=1, input_dim=40)

# for model without attention
accuracy_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_acc_{experiment_name}.png'
loss_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_loss_{experiment_name}.png'
history_without_attention= train_model_and_save_results(model=lstm_model,
                             accuracy_file_name_model=accuracy_file_name_model,
                             loss_file_name_model=loss_file_name_model,
                             trainX=trainX, trainY=trainY,testX=testX,testY=testY, file=f,
                             best_model_name="model_lstm_without_attention")

# for model with attention
accuracy_file_name_model_with_attention = f'{configuration.attention_experiments_results_plots_path}\\attention_acc{experiment_name}.png'
loss_file_name_model_with_attention = f'{configuration.attention_experiments_results_plots_path}\\attention_loss{experiment_name}.png'
history_with_attention = train_model_and_save_results(lstm_model_with_attention, accuracy_file_name_model_with_attention,
                             loss_file_name_model_with_attention,
                             trainX, trainY, testX,testY, f,
                             best_model_name="model_lstm_attention")

f.close()


plot_validation_and_train_acc_2_models(file_name="Comparison_gru_with_attention",
                                       title="Comparing accuracy for model with and without attention",
                                       history1=history_without_attention,
                                       history2=history_with_attention)