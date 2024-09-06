# model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_model():
    data = load_iris()
    X, y = data.data, data.target
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

