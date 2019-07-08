import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys


def main():
    data = pd.read_csv(sys.argv[1])
    X = data.drop('city', 1).values #remove first label
    y = data['city'].values

    # Divide the data into train and test
    X_train, X_valid, y_train, y_valid = train_test_split(X, y) #, test_size=0.33)
    
    # Bayesian model
    bayes_model = make_pipeline(
        MinMaxScaler(),
        GaussianNB()
        )
    bayes_model.fit(X_train, y_train)

    # k-nearest neighbours (scaled)
    n_nbs = 5
    knn_model = make_pipeline(
        MinMaxScaler(),
        KNeighborsClassifier(n_neighbors=n_nbs)
        )
    knn_model.fit(X_train, y_train)

    # Support Vector machines
    svc_model = make_pipeline(
        MinMaxScaler(),
        SVC(kernel='rbf', C=5, gamma='scale', decision_function_shape='ovr')
        )
    svc_model.fit(X_train, y_train)

    print("bayes model score: " + str(bayes_model.score(X_valid, y_valid)))
    print("knn model score: " + str(knn_model.score(X_valid, y_valid)))
    print("svc model score: " + str(svc_model.score(X_valid, y_valid)))

    data_unlabelled = pd.read_csv(sys.argv[2])
    X_unlabelled = data_unlabelled.drop('city', 1).values

    bayes_pred = np.array(bayes_model.predict(X_unlabelled))
    knn_pred = np.array(knn_model.predict(X_unlabelled))
    svc_pred = np.array(svc_model.predict(X_unlabelled))

    print("svc prediction results: " + str(svc_pred))
    
if __name__ == '__main__':
    main()
