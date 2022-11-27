import configuration
from utilities.data_utilities import get_transformed_data
from models import get_model
from utilities.train_utilities import get_callbacks_for_training, train_model

# loading train and test data
x_train, y_train, x_test, y_test = get_transformed_data()

# creating the model
model = get_model(num_of_output_classes=5,
                  input_dim=40,
                  lr=configuration.learning_rate)

history = train_model(model, x_train, y_train, x_test, y_test,configuration.n_epochs)

