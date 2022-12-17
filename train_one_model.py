import configuration
from keras_models.models import get_model
from data_utilities.data_utilities import get_transformed_data

# loading train and test data
from utilities.plot_utilities import plot_confusion_matrix

x_train, y_train, x_test, y_test = get_transformed_data()


# creating the model
model = get_model(num_of_output_classes=5,
                  input_dim=122,
                  lr=configuration.learning_rate)


history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    verbose=1)

plot_confusion_matrix(model, x_test, y_test)