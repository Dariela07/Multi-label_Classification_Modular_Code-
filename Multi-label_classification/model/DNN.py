import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization, LSTM, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from Config import *
from model.base import BaseModel


class DNN(BaseModel):
    def __init__(self,
                 model_name: str,
                 X: np.ndarray,
                 y: np.ndarray,
                 count_classes: list    # It is a list of number of categories under each Type
                  ) -> None:
        """
        Initializes the model, sets attributes and constructs the neural network layers.
        """
        super(DNN, self).__init__()
        self.X = X
        self.model_name = model_name
        self.y = y
        self.count_classes = count_classes
        self.predict_prob = None
        self.predict_binary = None
        self.mdl = Sequential([
            Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
            Dropout(0.5),
            Dense(sum(self.count_classes), activation='sigmoid')
        ])
        self.early_stopping = EarlyStopping(monitor='val_loss',patience=3)


    def train(self, data) -> None:
        """
        Compiles and trains the model using training data.
        """
        self.mdl.compile(loss='binary_crossentropy',
              optimizer='Adam',
              # optimizer='Adam',
              metrics = ['binary_accuracy'])
        history = self.mdl.fit(data.X_train, data.y_train, epochs=Config.EPOCHS, verbose=Config.VERBOSE, callbacks=[self.early_stopping])


    def predict(self, X_test: pd.Series):
        """
        Generates binary predictions for the input test data.
        """
        self.predict_prob = self.mdl.predict(X_test)
        self.predict_binary = np.where(self.predict_prob >0.5, 1, 0)
        # print(self.predict_binary)
        return self.predict_binary


    def calculate_accuracy_per_sample(self, y_test, y_predict, data):
        """
        Calculates accuracy score for one sample.
        """
        score = 0  # The default accuracy is 0
        # Stamps are defined to slice the classes in Type  2, 3, 4,
        # data.count_classes = [2, 9, 14], so there are 2, 9, 14 categories in Type2, Type3 and Type4 respectively.
        stamp1 = data.count_classes[0]
        stamp2 = data.count_classes[0] + data.count_classes[1]
        y_test1 = y_test[:stamp1]  # Type two
        y_predict1 = y_predict[:stamp1]
        y_test2 = y_test[stamp1:stamp2]  # Type three
        y_predict2 = y_predict[stamp1:stamp2]
        y_test3 = y_test[stamp2:]  # Type four
        y_predict3 = y_predict[stamp2:]
        if np.array_equal(y_test1, y_predict1):  # Check Type2 prediction. If it is not correct, error variable is not updated, default to 0
            if not np.array_equal(y_test2, y_predict2): # If predicting and actual results are the same in Type 2, Check Type 3
                score = 1 / 3   # If Type 3 is not correct, accuracy is 1/3
            else:
                if not np.array_equal(y_test3, y_predict3):  # If Type 3 is also correct, check Type 4
                    score = 2 / 3    # If Type 4 is not predicted correctly, then accuracy is 67%
                else:
                    score = 1     # If Type 4 is also predicted correctly, then the accuracy is 1
        return score


    def calculate_average_accuracy(self, y_tests, y_predicts, data):
        """
        Computes the average accuracy across all samples.
        """
        all_scores = []
        for y_test, y_predict in zip(y_tests, y_predicts):      # Loop through all the samples
            score = self.calculate_accuracy_per_sample(y_test, y_predict, data)
            all_scores.append(score)        # accuracy for each email classification is appended in a list
        # print(all_scores)
        Average_accuracy = sum(all_scores) / len(all_scores)    # The average accuracy is computed
        return Average_accuracy


    def print_predicted_and_true_categories(self, X_test, y_test, data, n_examples ) -> None:
        """
        Example Printer: Prints predicted and true categories, and its corresponding accuracy score for a subset of examples in the testing data.
        """
        predict_binary = self.predict(X_test)  # <class 'numpy.ndarray'>
        random_indices = np.random.choice(y_test.shape[0], n_examples, replace=False)
        for predict, actual,i in zip(predict_binary[random_indices], y_test[random_indices], range(n_examples)):
            # print(predict, actual)
            print(f"Testing example {i+1}:")
            actual_categories = [x for x, i in zip(data.classes, actual) if i ==1]
            print(f"Actual categories: {actual_categories}")
            predict_categories = [x for x, i in zip(data.classes, predict) if i ==1]
            print(f"Predicted categories: {predict_categories}")
            print(f"Accuracy is: {self.calculate_accuracy_per_sample(actual, predict, data)}")
            print("")


    def print_testing_accuracy(self, x_test, y_test, data):
        """
        Results Printer: prints the accuracy score on the test data.
        """
        y_predict = self.predict(x_test)
        accuracy = self.calculate_average_accuracy(y_test, y_predict, data)
        print('Accuracy on test set: {:.2f}%'.format(accuracy * 100))


    def print_training_accuracy(self, x_train, y_train, data):
        """
        Results Printer: prints the accuracy score on the training data.
        """
        y_predict = self.predict(x_train)
        accuracy = self.calculate_average_accuracy(y_train, y_predict, data)
        print('Accuracy on training set: {:.2f}%'.format(accuracy * 100))


