import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from data_utilities.Sample import Samples

def plot_metric_from_history_for_model_comparison(history_model1,
                                                  history_model2,
                                                  train_metric_name,
                                                  validation_metric_name,
                                                  label_train_metric1,
                                                  label_validation_metric1,
                                                  label_train_metric2,
                                                  label_validation_metric2,
                                                  file_name,
                                                  title,
                                                  y_label):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history_model1.history[train_metric_name], color='r', label=label_train_metric1)
    ax.plot(history_model1.history[validation_metric_name], color='g', label=label_validation_metric1)


    ax.plot(history_model2.history[train_metric_name], '--', color='r', label=label_train_metric2)
    ax.plot(history_model2.history[validation_metric_name], '--', color='g', label=label_validation_metric2)

    fig.suptitle(title, fontsize=7)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend()
    fig.savefig(file_name)

def plot_validation_and_train_metric_from_history(history,
                                                  train_metric_name,
                                                  validation_metric_name,
                                                  label_train_metric,
                                                  label_validation_metric,
                                                  file_name,
                                                  title,
                                                  y_label):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(history.history[train_metric_name], color='r', label=label_train_metric)
    ax.plot(history.history[validation_metric_name], color='g', label=label_validation_metric)
    fig.suptitle(title, fontsize=7)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.legend()
    fig.savefig(file_name)

def plot_validation_and_train_loss(file_name,title, history):
    train_metric_name = 'loss'
    validation_metric_name='val_loss'
    label_train_metric = 'Train loss'
    label_validation_metric = 'Validation loss'
    y_label = 'Loss'
    plot_validation_and_train_metric_from_history(history,
                                                  train_metric_name,
                                                  validation_metric_name,
                                                  label_train_metric,
                                                  label_validation_metric,
                                                  file_name,
                                                  title,
                                                  y_label)

def plot_validation_and_train_acc_2_models(file_name, title,history1, history2):
    train_metric_name = 'accuracy'
    validation_metric_name='val_accuracy'
    # labels:
    label_train_metric1 = 'Train accuracy'
    label_validation_metric1 = 'Validation accuracy'
    label_validation_metric2 = 'Validation accuracy, attention model'
    label_train_metric2 = 'Train accuracy, attention model'
    y_label = 'Accuracy'
    plot_metric_from_history_for_model_comparison(history_model1=history1,
                                                  history_model2=history2,
                                                  train_metric_name=train_metric_name,
                                                  validation_metric_name=validation_metric_name,
                                                  label_train_metric1=label_train_metric1,
                                                  label_validation_metric1=label_validation_metric1,
                                                  label_train_metric2=label_train_metric2,
                                                  label_validation_metric2=label_validation_metric2,
                                                  file_name=file_name,
                                                  title=title,
                                                  y_label=y_label)

def plot_validation_and_train_acc(file_name, title, history):
    train_metric_name = 'accuracy'
    validation_metric_name='val_accuracy'
    label_train_metric = 'Train accuracy'
    label_validation_metric = 'Validation accuracy'
    y_label = 'Accuracy'
    plot_validation_and_train_metric_from_history(history,
                                                  train_metric_name,
                                                  validation_metric_name,
                                                  label_train_metric,
                                                  label_validation_metric,
                                                  file_name,
                                                  title,
                                                  y_label)

def plot_test_acc(file_name,title, accuracy):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(accuracy, color='g', label='Validation accuracy')
    fig.suptitle(title, fontsize=10)
    plt.xlabel('Epochs', fontsize=10)
    plt.legend()
    plt.ylabel('Accuracy', fontsize=10)
    fig.savefig(file_name)


def plot_hist_and_distribution(file_name1, file_name2, scores ):
    # histogram
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # histogram to show the shape of the distribution
    # A histogram of the accuracy scores is also created
    plt.hist(scores, bins=10)
    fig.savefig(file_name1)

    # boxplot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # boxplot of distribution,box and whisker plot to show the spread and body of the distribution
    plt.boxplot(scores)
    fig.savefig(file_name2)


def plot_sensitivity_analysis_on_number_of_ensemble_members(n_members, scores, file_name):
    # score in ensemble members
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # plot score vs number of ensemble members
    x_axis = [i for i in range(1, n_members + 1)]
    plt.plot(x_axis, scores)
    fig.savefig(file_name)


def plot_confusion_matrix(model1, x_test, y_test):
    # confusion matrix
    # model1 = get_trained_model(x_train, y_train, n_epochs, 5)
    # model1.fit(x_train, y_train)

    y_pred = model1.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def plot_correlation(df):
    correlation = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.title('Correlation between different features')
    plt.show()

def plot_PCA(X):
    X_std = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_std)
    plt.plot(np.cumsum(pca.explained_varianceratio))
    plt.xlim(0, len(X_std), 1)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    
# Create scaler
# scaler = StandardScaler()
# train_samples = Samples(load_test_data(dataset_number_to_load=4))
# X = train_samples.get_features()
# y = train_samples.get_labels()
# # Rescale feature matrix
# X_rescaled = scaler.fit_transform(X)
# # Create TSNE tranformer
tsne = TSNE(n_components=2, verbose=1, random_state=0)

# Define a function to apply and plot T-SNE:
def plot_tsne(rescaled_data, target, title):
    # Apply T-SNE transformer on rescaled data
    tsne_2 = tsne.fit_transform(rescaled_data)

    # Create a legend for pre-converted target data
    legend = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    df_tess = pd.DataFrame({'emotion': target}) #bug: exei emotion: ps
    df_tess['emotion'].replace({'angry': 0, 'disgust': 1,
                                'fear': 2, 'happy': 3,
                                'neutral': 4, 'sad': 5,
                                'surprise': 6,
                                'ps':6},
                               inplace=True)

    y = [legend[x] for x in df_tess.emotion.values]

    # Store T-SNE data in a DataFrame
    tsne2 = pd.DataFrame()
    tsne2['y'] = y
    tsne2['1st component'] = tsne_2[:, 0]
    tsne2['2nd component'] = tsne_2[:, 1]

    # Plot data
    plt.figure(figsize=(20, 10))
    sns.scatterplot(x='1st component',
                    y='2nd component',
                    hue=tsne2.y.tolist(),
                    palette=sns.color_palette('hls', 7),
                    data=tsne2, s=20, alpha=0.7)

    plt.title(title, size=15)
    plt.savefig("T_SNE_components")


# Apply function
# plot_tsne(X_rescaled, y, '2D-plot of the first and second T-SNE components')