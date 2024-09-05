import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# #read preprocessed data
# data = pd.read_csv('data\\preprocessed_imgs.csv', header=None)

# X = data.iloc[1:, 1:-1].values
# y = data.iloc[1:, -1]


class Classifier:
    def __init__(self, X, y, random_state=29):
        self.random_state = random_state
        self.X = X
        self.y = y
        self.enconded_label()
        self.split_train_test()
        self.scaling_data()


    #Encoded Label from string to int
    def enconded_label(self):
        self.label_encoder = LabelEncoder()
        self.encoded_y = self.label_encoder.fit_transform(self.y)
    
    
    #split train test data
    def split_train_test(self):
        random_state = 29
        test_size = 0.3
        is_shuffle = True

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.encoded_y,
            test_size=test_size,
            random_state=random_state,
            shuffle=is_shuffle
        )
        
        
    #scaling data
    def scaling_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)


    #training SVC model
    def train_model(self):
        self.model = SVC(
            kernel='rbf',
            random_state=self.random_state,
            probability=True,
            C=0.5
        )
        self.model.fit(self.X_train, self.y_train)
        
        
    #evaluate model   
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred)
        clf_rp = classification_report(self.y_test, y_pred)

        print('Acc:', score)
        print('Report: ', clf_rp)


# clf = Classifier(X, y)
# clf.train_model()
# clf.evaluate()

