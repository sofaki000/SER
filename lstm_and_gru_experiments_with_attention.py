import configuration
from data_utilities.data_utilities import get_transformed_data
from keras_models.attention_model import get_model_with_additive_attention
from keras_models.gru_models import get_gru_model, get_gru_model_with_attention, get_gru_model_with_more_layers, \
    get_gru_model_with_attention_and_more_layers
from keras_models.lstm_models import get_lstm_model_with_dropout_and_attention, get_lstm_model_with_dropout, \
    get_lstm_model_with_dropout_more_layers, get_lstm_model_with_dropout_and_attention_more_layers
from utilities.plot_utilities import plot_validation_and_train_acc_2_models
from utilities.train_utilities import train_model_and_save_results

trainX, trainY, testX, testY = get_transformed_data(number_of_samples_to_load=-1)
output_classes = 7
n_samples = len(trainX)
n_test_samples = len(testX)
n_features = 283

trainX = trainX.reshape(n_samples,n_features,1) # we reshape so it is lstm friendly
testX = testX.reshape(n_test_samples,n_features,1)

f = open(f"{configuration.models_experiments_results_text_path}\\models_lstm_architecture.txt", "a")
experiment_name = "_tess_crema_lstm_and_gru_v2"
f.write(f'{experiment_name}\n')

##################### without attention
# gru_model = get_gru_model(num_features=n_features,output_classes=output_classes)
#gru_model_with_attention = get_gru_model_with_attention(num_features=n_features,output_classes=output_classes)

# with additive attention:
model_with_additive_attention = get_model_with_additive_attention(num_of_output_classes=output_classes, input_dim=n_features, lr=0.01)

gru_model = get_gru_model_with_more_layers(num_features=n_features,
                                           output_classes=output_classes)
gru_model_with_attention = get_gru_model_with_attention_and_more_layers(num_features=n_features,
                                                                        output_classes=output_classes)

lstm_model_with_attention = get_lstm_model_with_dropout_and_attention_more_layers(num_features=n_features,
                                                                                  output_classes=output_classes)
lstm_model = get_lstm_model_with_dropout_more_layers(num_features=n_features,
                                                     output_classes=output_classes)
# lstm layer expects input in shape: samples, time series, features

#lstm_model = get_lstm_model_with_attention(time_steps=1, input_dim=40)

# for model without attention
accuracy_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_acc_{experiment_name}.png'
loss_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_loss_{experiment_name}.png'
lstm_history_without_attention= train_model_and_save_results(model=lstm_model,
                             accuracy_file_name_model=accuracy_file_name_model,
                             loss_file_name_model=loss_file_name_model,
                             trainX=trainX, trainY=trainY,testX=testX,testY=testY, file=f,
                             best_model_name="model_lstm_without_attention")

# for model with attention
accuracy_file_name_model_with_attention = f'{configuration.attention_experiments_results_plots_path}\\attention_acc{experiment_name}.png'
loss_file_name_model_with_attention = f'{configuration.attention_experiments_results_plots_path}\\attention_loss{experiment_name}.png'
lstm_history_with_attention = train_model_and_save_results(lstm_model_with_attention,
                                                      accuracy_file_name_model_with_attention,
                                                      loss_file_name_model_with_attention,
                                                      trainX, trainY, testX,testY, f,
                                                      best_model_name="model_lstm_attention")

plot_validation_and_train_acc_2_models(file_name= f"Comparison_lstm_with_attention_{experiment_name}",
                                       title= "Comparing accuracy for model with and without attention",
                                       history1= lstm_history_without_attention,
                                       history2= lstm_history_with_attention)

################## GRU MODELS
#with attention
gru_accuracy_file_name_model_with_attention = f'{configuration.attention_experiments_results_plots_path}\\attention_acc_gru_{experiment_name}.png'
gru_loss_file_name_model_with_attention = f'{configuration.attention_experiments_results_plots_path}\\attention_loss_gru_{experiment_name}.png'
gru_history_with_attention = train_model_and_save_results(gru_model_with_attention,
                                                      gru_accuracy_file_name_model_with_attention,
                                                     gru_loss_file_name_model_with_attention,
                                                     trainX, trainY, testX,testY, f,
                                                     best_model_name="gru_attention_model")

#without attention
gru_accuracy_file_name_model  = f'{configuration.attention_experiments_results_plots_path}\\acc_gru_{experiment_name}.png'
gru_loss_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\loss_gru_{experiment_name}.png'
gru_history_without_attention = train_model_and_save_results(gru_model,
                                                      gru_accuracy_file_name_model,
                                                      gru_loss_file_name_model,
                                                      trainX, trainY, testX,testY, f,
                                                      best_model_name="gru_model")

plot_validation_and_train_acc_2_models(file_name=f"Comparison_gru_with_attention_{experiment_name}",
                                       title="Comparing accuracy for model with and without attention",
                                       history1=gru_history_without_attention,
                                       history2=gru_history_with_attention)


############ model with additive attention #####################
accuracy_file_name_additive_attention = f'{configuration.attention_experiments_results_plots_path}\\additive_attention_acc{experiment_name}.png'
loss_file_name_additive_attention = f'{configuration.attention_experiments_results_plots_path}\\additive_attention_loss{experiment_name}.png'
lstm_history_with_attention = train_model_and_save_results(model_with_additive_attention,
                                                      accuracy_file_name_additive_attention,
                                                      loss_file_name_additive_attention,
                                                      trainX, trainY, testX,testY, f,
                                                      best_model_name="model_dense_additive_attention")

f.close()