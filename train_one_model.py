from utilities.data_utilities import get_transformed_data
from models import get_model
from utilities.train_utilities import get_callbacks_for_training

# these two callbacks are responsible for saving best model and doing early stopping method
# to avoid overfitting
callbacks_for_early_stopping = get_callbacks_for_training()

# loading train and test data
x_train, y_train, x_test, y_test = get_transformed_data()

n_epochs = 400
learning_rate=0.01

# creating the model
model = get_model(num_of_output_classes=5 ,input_dim=40, lr=learning_rate)

# training the model
history = model.fit(x_train , y_train, validation_data=(x_test, y_test), epochs=n_epochs, verbose=1,callbacks=callbacks_for_early_stopping)

# we print the epoch we stopped at (because we are doing early stopping, we care to see
# how much we actually trained
print(f'Stopped at epoch {callbacks_for_early_stopping[0].stopped_epoch}')