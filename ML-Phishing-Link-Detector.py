#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

class DatasetLoader:
    
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)

    def preprocess(self):
        
        
        self.data = self.data.drop(['FILENAME', 'URL', 'Domain', 'Title'], axis=1)

        features = self.data.drop('label', axis=1)

        features = features.apply(lambda x: x / x.max(), axis=0)

        self.data = pd.concat([features, self.data['label']], axis=1)


    def split_dataset(self):
        train_data, remaining_data = train_test_split(self.data, test_size=0.30, random_state=42)
        dev_data, test_data = train_test_split(remaining_data, test_size=0.50, random_state=42)
        return train_data, dev_data, test_data
        
        
class LRClassifier:

    def __init__(self, train_data, dev_data, test_data):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = LogisticRegression()

    def fit(self):
        features = self.train_data.iloc[:, :-1]
        targets = self.train_data.iloc[:, -1]
        self.model.fit(features, targets)


    def evaluate(self):
        while True:
            type = input("Would you like to evaluate this model's performance on the training, development or test_data? (train/dev/test): ").lower().strip()

            if (type == "train"):
                X = self.train_data.iloc[:, :-1]
                Y = self.train_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred)
                print(f"Training Set Accuracy (Logistic Regression): {accuracy: .6f}")
                break
            elif (type == "dev"):
                X = self.dev_data.iloc[:, :-1]
                Y = self.dev_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred)
                print(f"Development Set Accuracy (Logistic Regression): {accuracy: .6f}")
                break
            elif (type == "test"):
                X = self.test_data.iloc[:, :-1]
                Y = self.test_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred)
                print(f"Test Set Accuracy (Logistic Regression): {accuracy: .6f}")
                break
            else:
                print("Invalid Dataset Type...\n\n")


class SVMClassifier:
    def __init__(self, train_data, dev_data, test_data):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = SVC()

    def train(self):
        X = self.train_data.iloc[:, :-1]
        Y = self.train_data.iloc[:, -1]
        self.model.fit(X, Y)

    def evaluate(self):
        while True:
            type = input("Would you like to evaluate this model's performance on the training, development or test_data? (train/dev/test): ").lower().strip()

            if (type == "train"):
                X = self.train_data.iloc[:, :-1]
                Y = self.train_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred)
                print(f"Training Set Accuracy (SVM): {accuracy: .6f}")
                break
            elif (type == "dev"):
                X = self.dev_data.iloc[:, :-1]
                Y = self.dev_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred)
                print(f"Development Set Accuracy (SVM): {accuracy: .6f}")
                break
            elif (type == "test"):
                X = self.test_data.iloc[:, :-1]
                Y = self.test_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred)
                print(f"Test Set Accuracy (SVM): {accuracy: .6f}")
                break
            else:
                print("Invalid Dataset Type...\n\n")

class KNNClassifier:
    def __init__(self, train_data, dev_data):
        self.train_data = train_data
        self.dev_data = dev_data
        self.k = 3  # Default k value for KNN

    def train(self):
        # Implement KNN logic here
        pass

    def evaluate(self, dataset, dataset_name):
        # Implement KNN evaluation logic here
        pass

class DummyModel:
    def __init__(self, strategy):
        self.model = DummyClassifier(strategy=strategy)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, dataset, dataset_name):
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"{dataset_name} Accuracy (Dummy Classifier, {self.model.strategy}): {accuracy:.4f}")

    




def main():
    print("\n\n************** COMP 379 HW2 | Lueders, Sebastian **************\n\n\n")
    
    
if __name__ == "__main__":
    
    main()




