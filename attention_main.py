import configuration
from experiments_configuration import attention_exp_config
from keras_models.attention_model import VisualizeAttentionMap, \
    get_dense_model, get_model_with_attention
from data_utilities.data_utilities import get_transformed_data
from utilities.plot_utilities import plot_validation_and_train_acc, plot_validation_and_train_loss
from utilities.train_utilities import get_callbacks_for_training

f = open(f"{configuration.experiments_results_text_path}/test_results_attention.txt", "a")

# Generate the dataset
trainX, trainY, testX, testY, scaled_samples = get_transformed_data(dataset_number_to_load=1)

n_samples = trainX.shape[0]
n_inputs = trainX.shape[1] # number of features
output_size = 5
model_RNN= get_dense_model(num_of_output_classes=output_size,input_dim=40, lr=0.01)
# model_RNN = get_dense_model(hidden_units=attention_exp_config.hidden_units,
#                        output_classes=output_size,
#                        input_shape=(1, attention_exp_config.number_of_features),
#                        activation='tanh')

#trainX= trainX.reshape(1,20,40) # to rnn ta perimenei se morfh: samples, time steps, and features.
training_callbacks = get_callbacks_for_training(best_model_name="rnn_model_without_attention")

modelRnn_history = model_RNN.fit(trainX, trainY,
                validation_split=0.2,
                epochs=attention_exp_config.epochs,
                batch_size=1,
                verbose=2,
                callbacks=training_callbacks)

#Evalute model
train_loss, train_acc = model_RNN.evaluate(trainX, trainY)
test_loss, test_acc  = model_RNN.evaluate(testX, testY)

model_content_train= f"Dense model: train loss={train_loss:.2f}, train accuracy:{train_acc:.2f}\n"
model_content_test = f"Dense model: test loss={test_loss:.2f}, test accuracy:{test_acc:.2f}\n"
f.write(model_content_train)
f.write(model_content_test)
# Print error
print(model_content_train)
print(model_content_test)
accuracy_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_acc.png'
loss_file_name_model = f'{configuration.attention_experiments_results_plots_path}\\simple_model_loss.png'

plot_validation_and_train_acc(file_name=accuracy_file_name_model, title="Accuracy model with attention", history=modelRnn_history)
plot_validation_and_train_loss(file_name=loss_file_name_model, title="Loss model with attention", history=modelRnn_history)
####### model with attention ##################################

model_attention= get_model_with_attention(num_of_output_classes=output_size,input_dim=40, lr=0.01)
# model_attention = create_RNN_with_attention(hidden_units=attention_exp_config.hidden_units,
#                                             dense_units=output_size,
#                                             input_shape=(1,attention_exp_config.number_of_features),
#                                             activation='tanh')

training_callbacks_attention = get_callbacks_for_training(best_model_name="best_model_with_attention")

visualize_attention = VisualizeAttentionMap(model_attention, trainX)

attention_history = model_attention.fit(trainX, trainY,
                    epochs=attention_exp_config.epochs,
                    batch_size=1,
                    verbose=2,
                    validation_split=0.2,
                    callbacks=training_callbacks_attention)


accuracy_file_name = f'{configuration.attention_experiments_results_plots_path}\\attention_acc.png'
loss_file_name = f'{configuration.attention_experiments_results_plots_path}\\attention_loss.png'

plot_validation_and_train_acc(file_name=accuracy_file_name, title="Accuracy model with attention", history=attention_history)
plot_validation_and_train_loss(file_name=loss_file_name, title="Loss model with attention", history=attention_history)

# Evalute model
train_loss, train_acc = model_attention.evaluate(trainX, trainY)
test_loss, test_acc = model_attention.evaluate(testX, testY)

# Print error
content_train= f"Model with attention: train loss={train_loss:.2f}, train accuracy:{train_acc:.2f}\n"
content_test = f"Model with attention: test loss={test_loss:.2f}, test accuracy:{test_acc:.2f}\n"

print(content_train)
print(content_test)
f.write(content_train)
f.write(content_test)

f.close()

# #
# model_name = 'rnn_model_with_attention.h5'
# #model_attention.save(model_name)
# # model_attention.save_weights('rnn_model_with_attention')
#
#
# model = load_model(model_name, custom_objects={'attention': attention})
# model = Model(inputs=model.input, outputs=[model.output, model.get_layer('attention').output])
#
# outputs = model.predict(testX)
# model_outputs = outputs[0]
# attention_outputs = outputs[1]
#
# import seaborn as sns
# sns.heatmap((attention_outputs[0],attention_outputs[1]))# ,  xlabel='Keys', ylabel='Queries'
# plt.show()
#
#
# def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
#     """Show heatmaps of matrices."""
#
#     num_rows, num_cols = len(matrices), len(matrices[0])
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,  sharex=True, sharey=True, squeeze=False)
#     for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
#         for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
#             pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
#             if i == num_rows - 1:
#                 ax.set_xlabel(xlabel)
#             if j == 0:
#                 ax.set_ylabel(ylabel)
#             if titles:
#                 ax.set_title(titles[j])
#     fig.colorbar(pcm, ax=axes, shrink=0.6);
#
# class CharVal(object):
#     def __init__(self, char, val):
#         self.char = char
#         self.val = val
#
#     def __str__(self):
#         return self.char
#
# def rgb_to_hex(rgb):
#     return '#%02x%02x%02x' % rgb
#
# def color_charvals(s):
#     r = 255-int(s.val*255)
#     color = rgb_to_hex((255, r, r))
#     return 'background-color: %s' % color
#
# # if you are using batches the outputs will be in batches
# # get exact attentions of chars
# an_attention_output = attention_outputs[0][-len(testX):]
#
#
# # before the prediction i supposed you tokenized text
# # you need to match each char and attention
# char_vals = [CharVal(c, v) for c, v in zip(testX, attention_outputs)]
#
# import pandas as pd
# char_df = pd.DataFrame(char_vals).transpose()
# # apply coloring values
# char_df = char_df.style.applymap(color_charvals)
#
# #
# # plt.pcolormesh(attention_outputs)
# # plt.title('Attention')
# # plt.show()