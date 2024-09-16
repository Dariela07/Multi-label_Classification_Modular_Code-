import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *


# data_model.py
class Data():
    """
    The Data class preprocesses input data, binarizes class labels, splits the data into training and testing sets,
    and stores class information and embeddings
    """
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 test_size) -> None:

        # The provided original code is removed because in preprocess.py, under noise_remover function, any class in y1 with less than 10 records, and any class in y2, y3, y4 with less than 2 records are already removed.

        y_columns = df[Config.CLASS_COL]
        classes_dic = {}  # a dictionary to store each Type and its corresponding classes
        for c in Config.CLASS_COL:
            classes_dic[f'{c}_classes'] = df[c].unique()

        classes_n = []  # Store the number of classes for each Type
        classes_names = []  # All classes name for all Type are appended in one list
        for i in classes_dic.values():
            classes_n.append(len(i))
            classes_names = classes_names + list(i)

        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=classes_names)
        y_m = mlb.fit_transform(y_columns.values.astype('str'))
        self.X = X
        self.y = y_m
        self.test_size = 0.3
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=2, stratify=y_m)
        self.classes = classes_names
        self.count_classes = classes_n
        self.embeddings = X


    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_embeddings(self):
        return self.embeddings


