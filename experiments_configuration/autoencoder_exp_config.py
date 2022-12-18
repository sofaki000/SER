
import os
import configuration

experiments_folder = 'C:\\Users\\user\\Desktop\\Speech Emotion Recognition Project\\github - ser\\SER\\experiments_results_graphs\\autoencoder_experiments\\final\\WHOLE_DS\\'

os.makedirs(experiments_folder, exist_ok=True)

n_epochs = 200
learning_rate = 0.001
batch_size = 1
validation_split = 0.5
loss_file_name_autoencoder = f"{experiments_folder}_autoencoder_tess_crema_loss.png"
accuracy_file_name_autoencoder =  f"{experiments_folder}_autoencoder_tess_cremal_acc.png"

loss_file_name_without_autoencoder =  f"{experiments_folder}model_tess_crema_loss.png"
accuracy_file_name_without_autoencoder = f"{experiments_folder}model_tess_crema_acc.png"

saved_models_path = configuration.saved_models_path