from utilities.data_utilities import get_transformed_data
from utilities.evaluation_utilities import get_evaluation_scores_for_same_model_for_multiple_tries
from methods.ensemble_method import do_ensemble_method_and_plot_test_accuracies
from utilities.plot_utilities import plot_hist_and_distribution

x_train, y_train, x_test, y_test = get_transformed_data()
n_epochs = 20
learning_rate =0.01

def evaluate_same_model_many_times():
    # repeated evaluation of same model in order to see it's variance
    n_repeats = 15
    mean, std, scores = get_evaluation_scores_for_same_model_for_multiple_tries(x_train, y_train, x_test, y_test, n_repeats=n_repeats, input_dim=40, output_dim=5, epochs=2, lr= learning_rate)
    # summarize the distribution of scores
    print(f'Scores Mean: {mean:.3f}, Standard Deviation: {std:.3f}')
    plot_hist_and_distribution("results/hist_of_test_acc_scores_for_one_model", "results/box_and_whisker_of_test_acc_scores_for_one_model",  scores)

do_ensemble_method_and_plot_test_accuracies(x_train, y_train, x_test, y_test,5, n_epochs)