from keras.saving.save import load_model
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from keras_models.attention_model import Attention, get_model_with_attention_v2
from utilities.data_utilities import get_transformed_data
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

data_x, data_y, testX, testY, actual_labels = get_transformed_data(dataset_number_to_load=0)
n_samples = data_x.shape[0]
n_inputs = data_x.shape[1]  # number of features
time_steps, input_dim, output_dim = n_inputs, 1, 1

model = get_model_with_attention_v2(n_samples, time_steps, input_dim)
model.fit(data_x, data_y, epochs=10)

print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
output_class = [5]

losses = [
    (ActivationMaximization(layer_dict[layer_name], output_class), 2),
    (LPNorm(model.input), 10),
    (TotalVariation(model.input), 10)
]

opt = Optimizer(model.input, losses)
opt.minimize(max_iter=500, verbose=True, callbacks=[GifGenerator('opt_progress')])