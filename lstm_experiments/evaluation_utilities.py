import os

from sklearn.metrics import classification_report

from utilities.plot_utilities import plot_validation_and_train_acc, plot_validation_and_train_loss


def get_evaluation_for_model(history,f, experiment_name,epochs,file_name,
                             training_callbacks, model, features, labels, test_features, test_labels):
    f.write(experiment_name)
    #os.makedirs(file_name, exist_ok=True)  # succeeds even if directory exists.
    epoch_stopped = training_callbacks[0].stopped_epoch
    train_loss, train_acc = model.evaluate(features, labels)
    test_loss, test_acc = model.evaluate(test_features, test_labels)

    content_train = f'train: loss:{train_loss:.2f}, acc:{train_acc:.2f}\n'
    content_test = f'test: loss:{test_loss:.2f}, acc:{test_acc:.2f}\n'
    f.write(content_train)
    f.write(content_test)
    print(content_train)
    print(content_test)

    if epoch_stopped == 0:
        epoch_stopped = epochs

    accuracies_content = f'test acc:{test_acc:.2f}, train acc:{train_acc:.2f}'
    plot_validation_and_train_acc(file_name=f'{file_name}_acc',
                                  title=f"Accuracy, epoch stopped:{epoch_stopped},{accuracies_content}",
                                  history=history)

    plot_validation_and_train_loss(file_name=f'{file_name}_loss',
                                   title=f"Loss, epoch stopped:{epoch_stopped}",
                                   history=history)


    # # metrikes
    # y_pred = model.predict(test_features)
    # classification_report_content = classification_report(test_labels, y_pred)
    # print(classification_report_content)
    # f.write(classification_report_content)