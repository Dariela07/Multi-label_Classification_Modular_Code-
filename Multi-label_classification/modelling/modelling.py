from model.randomforest import RandomForest
from model.knn import KNN
from model.DNN import DNN


def model_predict(data):
    """
    Trains a DNN model with the given training data and returns the trained model and its predictions on the test data.
    """
    model = DNN("DNN", data.X_train, data.y_train, data.count_classes)
    model.train(data)
    predictions = model.predict(data.X_test)
    return model, predictions
def model_evaluate(model, data):
    """
    Prints examples of predicted vs. true categories in the test data and displays both testing and training accuracy.
    """
    model.print_predicted_and_true_categories(data.X_test, data.y_test, data, n_examples=3)
    model.print_testing_accuracy(data.X_test, data.y_test, data)
    model.print_training_accuracy(data.X_train, data.y_train, data)

