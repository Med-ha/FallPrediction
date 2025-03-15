from sklearn.ensemble import RandomForestClassifier
from joblib import dump

class Trainer:
    @staticmethod
    def train_classifier(X, y):
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X, y)
        return model

    @staticmethod
    def save_model(model, path):
        dump(model, path)
