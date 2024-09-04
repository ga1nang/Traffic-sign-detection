import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


#read preprocessed data
data = pd.read_csv('data\\preprocessed_imgs.csv', header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1]


#Encoded Label from string to int
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)


#split train test data
random_state = 29
test_size = 0.3
is_shuffle = True

X_train, X_test, y_train, y_test = train_test_split(
    X, encoded_y,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)


#scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#training SVC model
clf = SVC(
    kernel='rbf',
    random_state=random_state,
    probability=True,
    C=0.5
)
clf.fit(X_train, y_train)


#evaluate model
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
clf_rp = classification_report(y_test, y_pred)

print('Acc:', score)
print('Report: ', clf_rp)

