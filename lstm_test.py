import numpy as np
from sklearn.preprocessing import OneHotEncoder

from keras_models.attention_model import get_model_with_attention_v3, get_model_without_attention_v3
from utilities.data_utilities import get_transformed_data

train_samples, test_samples = get_transformed_data(dataset_number_to_load=0)

trainX = train_samples.get_features()
trainY = train_samples.get_encoded_labels()

# lstm layer expects input in shape: samples, time series, features
n_samples = len(trainX)
trainX = np.asarray(trainX) #trainX.reshape(n_samples,1,40)
trainY = np.asarray(trainY)

lstm_model = get_model_with_attention_v3(time_steps=1, input_dim=40)
modelRnn_history = lstm_model.fit(trainX, trainY, epochs=10,  batch_size=1, verbose=2)
# 2dataset: Train: Loss:0.17, acc:0.83
train_loss, train_acc = lstm_model.evaluate(trainX, trainY)
print(f'Train: Loss:{train_loss:.2f}, acc:{train_acc:.2f}')


##################### without attention
lstm_model = get_model_without_attention_v3(time_steps=1, input_dim=40)

# lstm layer expects input in shape: samples, time series, features
modelRnn_history = lstm_model.fit( trainX, trainY, epochs=10, batch_size=1, verbose=2)
train_loss, train_acc = lstm_model.evaluate(trainX, trainY)
print(f'Train: Loss:{train_loss:.2f}, acc:{train_acc:.2f}')