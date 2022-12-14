from keras.saving.save import load_model
from keras import Model
from matplotlib import pyplot as plt, cm
import numpy as np
from keras_models.attention_model import get_model_with_attention, Attention
from data_utilities.data_utilities import get_transformed_data


def plot3DHeatMap(x, y, z):
    colo = [x + y + z]

    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # setting color bar
    color_map = cm.ScalarMappable(cmap=cm.Greens_r)
    color_map.set_array(colo)

    # creating the heatmap
    img = ax.scatter(x, y, z, marker='s', s=200, color='green')
    plt.colorbar(color_map)
    # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis, Samples')
    ax.set_ylabel('Y-axis, Features')
    ax.set_zlabel('Z-axis, Units')
    # displaying plot
    plt.show()



def main():
    # data_x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    # data_y = np.random.uniform(size=(num_samples, output_dim))

    data_x, data_y, testX, testY ,actual_labels= get_transformed_data(dataset_number_to_load=0)
    n_samples = data_x.shape[0]
    n_inputs = data_x.shape[1]  # number of features
    time_steps, input_dim, output_dim =  n_inputs, 1, 1
    # Define/compile the model.
    model = get_model_with_attention(time_steps, input_dim)

    # train.
    model.fit(data_x, data_y, epochs=10)
    # test save/reload model.
    pred1 = model.predict(data_x)
    model_1 = 'test_model.h5'
    model.save(model_1)
    model_h5 = load_model('test_model.h5', custom_objects={'Attention': Attention})
    model = Model(inputs=model_h5.input, outputs=[model_h5.output,
                                                  model_h5.get_layer('attention').output,
                                                  model_h5.get_layer('lstm').output])
    pred2, attention_weights,lstm_weights = model.predict(data_x)
    np.testing.assert_almost_equal(pred1, pred2)

    import seaborn as sns
    cmap = sns.color_palette("coolwarm", 128)
    yticklabels = []
    for sample in actual_labels:
        name = sample.get_name()
        feats = sample.get_features()
        # we find the corresponding features
        for features in data_x:
            if np.equal(features, feats).all():
                yticklabels.append(name)

    plt.figure(figsize=(30, 10))
    ax1 = sns.heatmap(attention_weights,  cmap=cmap, yticklabels=yticklabels)
    plt.savefig("attention2.png")

    #lstm_weights: samples, features, units


    # importing required libraries
    samples_x_axis = [i for i in range(20)]
    features_y_axis = []
    # plot3DHeatMap(lstm_weights[0], lstm_weights[0][:], lstm_weights[0][0][:])

    print('Success.')


if __name__ == '__main__':
    main()