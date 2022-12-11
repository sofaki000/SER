from keras.saving.save import load_model
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from keras_models.attention_model import attention

model_name = 'rnn_model_with_attention.h5'
model = load_model(model_name, custom_objects={'attention': attention})
# # Build the VGG16 network with ImageNet weights
# model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
output_class = [20]

losses = [
    (ActivationMaximization(layer_dict[layer_name], output_class), 2),
    (LPNorm(model.input), 10),
    (TotalVariation(model.input), 10)
]

opt = Optimizer(model.input, losses)
opt.minimize(max_iter=500, verbose=True, callbacks=[GifGenerator('opt_progress')])