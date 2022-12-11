from keras import Model
from keras.saving.save import load_model
from matplotlib import pyplot as plt

import configuration
from experiments_configuration import attention_exp_config
from keras_models.attention_model import create_RNN_with_attention, create_RNN, VisualizeAttentionMap, \
    create_model_with_additive_attention, attention
from utilities.data_utilities import get_transformed_data
from utilities.train_utilities import get_callbacks_for_training

f = open(f"{configuration.experiments_results_text_path}/test_results_attention.txt", "a")

# Generate the dataset
trainX, trainY, testX, testY = get_transformed_data(dataset_number_to_load=0)
n_samples = trainX.shape[0]
n_inputs = trainX.shape[1] # number of features

# model_RNN = create_RNN(hidden_units=attention_exp_config.hidden_units,
#                        dense_units=1,
#                        input_shape=(n_inputs, 1),
#                        activation=['tanh', 'tanh'])
#
# training_callbacks = get_callbacks_for_training(best_model_name="rnn_model")
# model_RNN.fit(trainX, trainY, epochs=attention_exp_config.epochs, batch_size=1, verbose=2, callbacks=training_callbacks)

# Evalute model
# train_mse = model_RNN.evaluate(trainX, trainY)
# test_mse = model_RNN.evaluate(testX, testY)

# f.write(f"Train set MSE = {train_mse}")
# f.write(f"Test set MSE ={test_mse}" )
# f.close()
# # Print error
# print(f"Train set MSE = {train_mse}")
# print(f"Test set MSE ={test_mse}" )

####### model with attention
model_attention = create_RNN_with_attention(hidden_units=attention_exp_config.hidden_units,
                                            dense_units=1,
                                            input_shape=(attention_exp_config.time_steps, 1),
                                            activation='tanh')
#
#
# # model_attention = create_RNN_with_attention(input_shape=40, output_classes=5)
#
# training_callbacks_attention = get_callbacks_for_training(best_model_name="wohoo")
#
# visualize_attention = VisualizeAttentionMap(model_attention, trainX)
#
# model_attention.fit(trainX, trainY,
#                     epochs=attention_exp_config.epochs,
#                     batch_size=1,
#                     verbose=2,
#                     callbacks=training_callbacks_attention.append(visualize_attention))
#
#
# # Evalute model
# train_mse_attn = model_attention.evaluate(trainX, trainY)
# test_mse_attn = model_attention.evaluate(testX, testY)
#
# # Print error
# print("Train set MSE with attention = ", train_mse_attn)
# print("Test set MSE with attention = ", test_mse_attn)
#
#
model_name = 'rnn_model_with_attention.h5'
#model_attention.save(model_name)
# model_attention.save_weights('rnn_model_with_attention')


model = load_model(model_name, custom_objects={'attention': attention})
model = Model(inputs=model.input, outputs=[model.output, model.get_layer('attention').output])

outputs = model.predict(testX)
model_outputs = outputs[0]
attention_outputs = outputs[1]

import seaborn as sns
sns.heatmap((attention_outputs[0],attention_outputs[1]))# ,  xlabel='Keys', ylabel='Queries'
plt.show()


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """Show heatmaps of matrices."""

    num_rows, num_cols = len(matrices), len(matrices[0])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,  sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);

class CharVal(object):
    def __init__(self, char, val):
        self.char = char
        self.val = val

    def __str__(self):
        return self.char

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def color_charvals(s):
    r = 255-int(s.val*255)
    color = rgb_to_hex((255, r, r))
    return 'background-color: %s' % color

# if you are using batches the outputs will be in batches
# get exact attentions of chars
an_attention_output = attention_outputs[0][-len(testX):]


# before the prediction i supposed you tokenized text
# you need to match each char and attention
char_vals = [CharVal(c, v) for c, v in zip(testX, attention_outputs)]

import pandas as pd
char_df = pd.DataFrame(char_vals).transpose()
# apply coloring values
char_df = char_df.style.applymap(color_charvals)

#
# plt.pcolormesh(attention_outputs)
# plt.title('Attention')
# plt.show()