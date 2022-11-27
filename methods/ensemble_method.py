from utilities.evaluation_utilities import evaluate_ensemble_model
from utilities.plot_utilities import plot_test_acc, plot_sensitivity_analysis_on_number_of_ensemble_members


def do_ensemble_method_and_plot_test_accuracies(x_train, y_train, x_test, y_test,num_of_output_classes, n_epochs):
    # how many saved_models we will train for the ensemble method
    n_members = 15

    mean, std, scores= evaluate_ensemble_model(n_members , x_train, y_train, x_test, y_test,num_of_output_classes, n_epochs)
    # summarize the distribution of scores
    print(f'Ensemble: Scores Mean: {mean:.3f}, Standard Deviation: {std:.3f}')
    plot_test_acc("results/ensemble_models_test_acc","Test accuracies for each number of saved_models used", scores)
    plot_sensitivity_analysis_on_number_of_ensemble_members(n_members, scores, "results/scores_for_different_ensemble_nums")