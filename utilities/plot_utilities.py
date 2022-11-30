

# plot history
from matplotlib import pyplot as plt

def plot_validation_and_train_metric_from_history(history, train_metric_name, validation_metric_name,
                                                  label_train_metric, label_validation_metric, file_name, title, y_label):
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
