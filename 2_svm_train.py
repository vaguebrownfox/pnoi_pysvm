# import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pickle

# get dataset (feature vector)
dataset = pd.read_csv('./features_svm.csv')
X = dataset.iloc[:, 0:-1]
y = dataset.iloc[:, 72].values

# split dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1)

# scale data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# classifier instance
classifier = SVC(kernel='linear', random_state=0)

# fit with training data
classifier.fit(X_train, y_train)

# predict with test data
y_pred = classifier.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

# write the model to a file
with open('model.sav', 'wb') as m:
    pickle.dump(classifier, m)
