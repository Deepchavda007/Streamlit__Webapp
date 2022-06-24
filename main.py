import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

st.title("Machine Learning Algorithms")

st.write("""
         # Explore Different Classifier
         """)

dataset_name = st.sidebar.selectbox(
    "Select Dataset", ["Iris Dataset", "Wine Dataset"])


classifier_name = st.sidebar.selectbox(
    "Select Classifier", ["KNN", "SVM", "Random Forest"])


def get_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        data = datasets.load_iris()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X, y


X, y = get_dataset(dataset_name)
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 20)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

    else:
        Max_depth = st.sidebar.slider("Max_depth", 2, 15)
        N_estimators = st.sidebar.slider("N_estimators", 1, 100)
        params["Max_depth"] = Max_depth
        params["N_estimators"] = N_estimators

    return params


params = add_parameter_ui(classifier_name)


def get_classififier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(C=params["C"])

    else:
        clf = RandomForestClassifier(n_estimators=params["N_estimators"],
                                     max_depth=params["Max_depth"], random_state=2316)

    return clf


clf = get_classififier(classifier_name, params)


# classification

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2316)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

Accuracy = accuracy_score(y_test, y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {Accuracy}")


# Plot

pca = PCA(2)
x_projected = pca.fit_transform(X)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
