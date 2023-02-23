import numpy as np
from keras.models import load_model
from data_utilities.data_utilities import get_transformed_data
from keras_models.attention_model import attention, Attention

model_name = 'best_model_lstm_bad_trained_on_tess.h5'
model = load_model(model_name, custom_objects={'Attention': Attention})

trainX, trainY, testX, testY = get_transformed_data(number_of_samples_to_load=-1,
                                                    load_tess=False,
                                                    load_savee=False,
                                                    load_crema=True)

data = np.concatenate((trainX,testX))
labels = np.concatenate((trainY,testY))
# data = np.append(data, data[0].reshape(1,2,283), axis=0)
# labels = np.append(labels, labels[0].reshape(1,7),axis=0)

f = open(f"model_trained_on_tess_on_other_datasets.txt", "a")
f.write(f'\n\n------------------------------\n\n')

experiment_name = "crema"
f.write(experiment_name)
f.write(f'test size:{testX.shape[0]}\n')
test_loss, test_acc = model.evaluate(data, labels)
content_test = f'Test: loss:{test_loss:.2f}, accuracy:{test_acc:.2f}\n'
f.write(content_test)