from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score


from model.base import BaseModel


class KNN(BaseModel):
    def __init__(self,
                 k: int,
                 model_name: str,
                 X: np.ndarray,
                 df:pd.DataFrame,
                 y: np.ndarray) -> None:
        class_y2 = df['y2'].unique()
        class_y3 = df['y3'].unique()
        class_y4 = df['y4'].unique()
        #class_length = [len(class_y2), len(class_y3), len(class_y4)]
        class_names = list(class_y2) + list(class_y3) + list(class_y4)
        super(KNN, self).__init__()
        self.X = X
        self.model_name = model_name
        self.mlb = MultiLabelBinarizer(classes=class_names)
        self.y_m = self.mlb.fit_transform(y.values.astype('str'))
        self.y = y
        self.mdl = KNeighborsClassifier(n_neighbors=k)
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl.fit(self.X_train, self.y_train)

    def predict(self, X_test: pd.Series):
        self.y_pred = self.mdl.predict(X_test)

    def print_results(self, data):
        print(accuracy_score(self.y_test, self.y_pred))
        print("##")

        print("##")
        print(classification_report(self.y_test, self.y_pred, target_names=self.mlb.classes_,zero_division=1))


    def data_transform(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_m, test_size=0.2, random_state=0)


