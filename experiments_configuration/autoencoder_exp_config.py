
import os
import configuration

experiments_folder = 'C:\\Users\\Lenovo\\Desktop\\ser\\SER\\experiments_results_graphs\\autoencoder_experiments\\final\\'

os.makedirs(experiments_folder, exist_ok=True)

n_epochs = 200
learning_rate = 0.001
batch_size = 1
validation_split = 0.5
loss_file_name_autoencoder = f"{experiments_folder}model_autoencoder_loss_small_ds.png"
accuracy_file_name_autoencoder =  f"{experiments_folder}model_autoencoder_small_ds.png"

loss_file_name_without_autoencoder =  f"{experiments_folder}model_loss_small_ds.png"
accuracy_file_name_without_autoencoder = f"{experiments_folder}model_acc_small_ds.png"

saved_models_path = configuration.saved_models_path