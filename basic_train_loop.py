from keras_models.models import get_model
from utilities.data_utilities import get_transformed_data
from utilities.plot_utilities import plot_validation_and_train_acc

x_train, y_train, x_test, y_test = get_transformed_data()
n_epochs = 200
learning_rate=0.01

model = get_model(num_of_output_classes= 5,input_dim=40, lr=learning_rate)
history = model.fit(x_train , y_train, validation_data=(x_test, y_test), epochs=n_epochs, verbose=0)

# evaluate the 1 model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
title=f'Epochs:{n_epochs}, lr:{learning_rate}'

plot_validation_and_train_acc("experiments_results/one_model_accuracy", title, history)