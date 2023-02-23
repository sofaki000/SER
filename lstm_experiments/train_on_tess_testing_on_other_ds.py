from keras.models import load_model

from data_utilities.data_utilities import get_transformed_data
from keras_models.attention_model import attention, Attention

model_name = 'best_model_lstm_bad_trained_on_tess.h5'
model = load_model(model_name, custom_objects={'Attention': Attention})

trainX, trainY, testX, testY = get_transformed_data(number_of_samples_to_load=-1,
                                                    load_tess=False,
                                                    load_savee=False)

f = open(f"model_trained_on_tess_on_other_datasets.txt", "a")
f.write(f'\n\n------------------------------\n\n')
f.write("testing on crema")
experiment_name = "augmentation"
f.write(experiment_name)
f.write(f'test size:{testX.shape[0]}\n')
test_loss, test_acc = model.evaluate(testX, testY)
content_test = f'Έλεγχος: Απώλεια:{test_loss:.2f}, Ακρίβεια:{test_acc:.2f}\n'
f.write(content_test)