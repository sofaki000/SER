
import os
import configuration

experiments_folder  = 'C:\\Users\\Lenovo\\Desktop\\ser\\SER\\experiments_results\\autoencoder_experiments\\model_more_neurons\\'
os.makedirs(experiments_folder, exist_ok=True)
n_epochs = 200
learning_rate = 0.001
batch_size = 128
loss_file_name_autoencoder = f"{experiments_folder}model_with_autoencoder_loss.png"
accuracy_file_name_autoencoder =  f"{experiments_folder}model_with_autoencoder_acc.png"

loss_file_name_without_autoencoder =  f"{experiments_folder}model_without_autoencoder_loss.png"
accuracy_file_name_without_autoencoder = f"{experiments_folder}model_without_autoencoder_acc.png"

saved_models_path = configuration.saved_models_path