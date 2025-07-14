import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

MIN_ACCURACY = 0.35
MODEL_PATH = "model.pkl"

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Training Accuracy: {acc:.2%}")
    if acc >= MIN_ACCURACY:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        return model, True
    return None, False

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)