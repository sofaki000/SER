
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from experiments_configuration import attention_exp_config
from keras_models.attention_model import create_RNN_with_attention, create_RNN
from utilities.data_utilities import get_transformed_data
from utilities.train_utilities import get_callbacks_for_training


def get_fib_seq(n, scale_data=True):
    # Get the Fibonacci sequence
    seq = np.zeros(n)
    fib_n1 = 0.0
    fib_n = 1.0
    for i in range(n):
        seq[i] = fib_n1 + fib_n
        fib_n1 = fib_n
        fib_n = seq[i]
    scaler = []
    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        seq = np.reshape(seq, (n, 1))
        seq = scaler.fit_transform(seq).flatten()
    return seq, scaler




def get_fib_XY(total_fib_numbers, time_steps, train_percent, scale_data=True):
    dat, scaler = get_fib_seq(total_fib_numbers, scale_data)
    Y_ind = np.arange(time_steps, len(dat), 1)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[0:rows_x]
    for i in range(time_steps - 1):
        temp = dat[i + 1:rows_x + i + 1]
        X = np.column_stack((X, temp))
    # random permutation with fixed seed
    rand = np.random.RandomState(seed=13)
    idx = rand.permutation(rows_x)
    split = int(train_percent * rows_x)
    train_ind = idx[0:split]
    test_ind = idx[split:]
    trainX = X[train_ind]
    trainY = Y[train_ind]
    testX = X[test_ind]
    testY = Y[test_ind]
    trainX = np.reshape(trainX, (len(trainX), time_steps, 1))
    testX = np.reshape(testX, (len(testX), time_steps, 1))
    return trainX, trainY, testX, testY, scaler


# Generate the dataset
#trainX, trainY, testX, testY, scaler = get_fib_XY(1200, attention_exp_config.time_steps, 0.7)
trainX, trainY, testX, testY = get_transformed_data(dataset_number_to_load=3)
n_samples = trainX.shape[0]
n_inputs = trainX.shape[1] # number of features



model_RNN = create_RNN(hidden_units=attention_exp_config.hidden_units,
                       dense_units=1,
                       input_shape=(n_inputs, 1),
                       activation=['tanh', 'tanh'])

training_callbacks = get_callbacks_for_training(best_model_name="rnn_model")
model_RNN.fit(trainX, trainY, epochs=attention_exp_config.epochs, batch_size=1, verbose=2, callbacks=training_callbacks)

# Evalute model
train_mse = model_RNN.evaluate(trainX, trainY)
test_mse = model_RNN.evaluate(testX, testY)

# Print error
print("Train set MSE = ", train_mse)
print("Test set MSE = ", test_mse)

####### model with attention
model_attention = create_RNN_with_attention(hidden_units=attention_exp_config.hidden_units,
                                            dense_units=1,
                                            input_shape=(attention_exp_config.time_steps, 1),
                                            activation='tanh')


training_callbacks_attention = get_callbacks_for_training(best_model_name="rnn_model_with_attention")
model_attention.fit(trainX, trainY,
                    epochs=attention_exp_config.epochs,
                    batch_size=1,
                    verbose=2,
                    callbacks=training_callbacks_attention)

# Evalute model
train_mse_attn = model_attention.evaluate(trainX, trainY)
test_mse_attn = model_attention.evaluate(testX, testY)

# Print error
print("Train set MSE with attention = ", train_mse_attn)
print("Test set MSE with attention = ", test_mse_attn)
