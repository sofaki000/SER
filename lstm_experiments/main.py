from keras import Input

import configuration
from data_utilities.data_utilities import get_transformed_data
from keras.layers import Layer, LSTM, Dense
from keras.layers import Reshape,BatchNormalization
from keras.callbacks import Callback
from keras.layers import SimpleRNN
from keras.models import Sequential, Model
from keract import get_activations
import os

from keras_models.attention_model import Attention
from utilities.plot_utilities import plot_validation_and_train_acc, plot_validation_and_train_loss
from utilities.train_utilities import get_callbacks_for_training


def create_model_with_additive_attention(input_shape, output_classes):
    # model_input = Input(shape=(2,input_shape))
    # model_input = Reshape(target_shape=(2,283))(model_input)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True))
    model.add(Attention(units=32))
    model.add(Dense(output_classes))
    model.compile(loss='mae', optimizer='adam',metrics='accuracy')
    return model

epochs = 1
output_classes = 5
n_features = 283

trainX, trainY, testX, testY = get_transformed_data(number_of_samples_to_load=-1)
n_samples = len(trainX)
n_test_samples = len(testX)

model = create_model_with_additive_attention(input_shape=n_features, output_classes=output_classes)
training_callbacks = get_callbacks_for_training(best_model_name='best_model_lstm')
history = model.fit(trainX, trainY, validation_split=0.5,  callbacks=training_callbacks,
                                        epochs= epochs,
                                        batch_size=10,
                                        verbose=2)

train_loss, train_acc = model.evaluate(trainX, trainY)
test_loss, test_acc = model.evaluate(testX, testY)

f = open(f"lstm_experiments.txt", "a")
content_train = f'Train: Loss:{train_loss:.2f}, acc:{train_acc:.2f}\n'
content_test = f'Test: Loss:{test_loss:.2f}, acc:{test_acc:.2f}\n'
f.write(content_train)
f.write(content_test)
print(content_train)
print(content_test)

accuracies_content = f'test acc:{test_acc:.2f}, train acc:{train_acc:.2f}'

epoch_stopped = training_callbacks[0].stopped_epoch

plot_validation_and_train_acc(file_name='acc',
                                  title=f"Accuracy, epoch stopped:{epoch_stopped},{accuracies_content}", history=history)

plot_validation_and_train_loss(file_name='loss', title=f"Loss, epoch stopped:{epoch_stopped}", history=history)