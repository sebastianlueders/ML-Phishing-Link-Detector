#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from collections import Counter
from sklearn.dummy import DummyClassifier

class DatasetLoader:
    
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)

    def preprocess(self, scale=True):
        
        self.data = self.data.drop(['FILENAME', 'URL', 'Domain', 'TLD', 'Title'], axis=1)

        if (scale):
            features = self.data.drop('label', axis=1)

            features = (features - features.min()) / (features.max() - features.min())

            self.data = pd.concat([features, self.data['label']], axis=1)


    def split_dataset(self):
        train_data, remaining_data = train_test_split(self.data, test_size=0.30, random_state=42)
        dev_data, test_data = train_test_split(remaining_data, test_size=0.50, random_state=42)
        return train_data, dev_data, test_data
        
        
class LRClassifier:

    def __init__(self, train_data, dev_data, test_data, C=1.0, epochs=100, penalty='l2', random_state=42):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = LogisticRegression(C=C, max_iter=epochs, penalty=penalty, random_state=random_state)

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
                accuracy = accuracy_score(Y, y_pred) * 100
                print(f"Training Set Accuracy (Logistic Regression): {accuracy: .4f}%\n")
                break
            elif (type == "dev"):
                X = self.dev_data.iloc[:, :-1]
                Y = self.dev_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred) * 100
                print(f"Development Set Accuracy (Logistic Regression): {accuracy: .4f}%\n")
                break
            elif (type == "test"):
                X = self.test_data.iloc[:, :-1]
                Y = self.test_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred) * 100
                print(f"Test Set Accuracy (Logistic Regression): {accuracy: .4f}%\n")
                break
            else:
                print("Invalid Dataset Type...\n\n")


class SVMClassifier:
    def __init__(self, train_data, dev_data, test_data, C=1.0, kernel='rbf', epochs=-1, random_state=42):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = SVC(C=C, kernel=kernel, max_iter=epochs, random_state=random_state)

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
                accuracy = accuracy_score(Y, y_pred) * 100
                print(f"Training Set Accuracy (SVM): {accuracy: .4f}%\n")
                break
            elif (type == "dev"):
                X = self.dev_data.iloc[:, :-1]
                Y = self.dev_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred) * 100
                print(f"Development Set Accuracy (SVM): {accuracy: .4f}%\n")
                break
            elif (type == "test"):
                X = self.test_data.iloc[:, :-1]
                Y = self.test_data.iloc[:, -1]
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(Y, y_pred) * 100
                print(f"Test Set Accuracy (SVM): {accuracy: .4f}%\n")
                break
            else:
                print("Invalid Dataset Type...\n\n")

class KNNClassifier:
    def __init__(self, train_data, test_data, k=3, p=2):
        self.Xtrain = train_data.iloc[:, :-1].values  
        self.Ytrain = train_data.iloc[:, -1].values   
        self.Xtest = test_data.iloc[:, :-1].values    
        self.Ytest = test_data.iloc[:, -1].values 
        self.k = k
        self.p = p

    def distance(self, row1, row2):
        return np.sum(np.abs(row1 - row2) ** self.p, axis=-1) ** (1 / self.p)

    def evaluate(self):
        predictions = np.array([self.predict(X) for X in self.Xtest])
        accuracy = np.mean(predictions == self.Ytest) * 100
        print(f"KNN Accuracy: {accuracy:.4f}%")

    def predict(self, X):
        # Compute distances between the test sample X and all training samples
        distances = np.array([self.distance(X, train_row) for train_row in self.Xtrain])
        # Get the indices of the k-nearest neighbors
        nearest_indices = np.argpartition(distances, self.k)[:self.k]
        # Get the labels of the nearest neighbors
        nearest_labels = self.Ytrain[nearest_indices]
        # Find the most common label among the k-nearest neighbors
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]

class DummyModel:
    def __init__(self, train_data, test_data, strategy):
        self.strategy = strategy
        self.model = DummyClassifier(strategy=strategy)
        self.train_data = train_data
        self.test_data = test_data

    def train(self, X, y):
        self.Xtrain_ = self.train_data.iloc[:, :-1]
        self.Ytrain_ = self.train_data.iloc[:, -1]
        self.train_data = None  
        self.model.fit(self.Xtrain_, self.Ytrain_)

    def evaluate(self):
        self.Xtest_ = self.test_data.iloc[:, :-1]
        self.Ytest_ = self.test_data.iloc[:, -1]
        self.test_data = None
        predictions = self.model.predict(self.Xtest_)
        accuracy = accuracy_score(self.Ytest_, predictions) * 100
        print(f"Accuracy (Dummy Classifier, {self.strategy}): {accuracy:.4f}%\n")



def main():
    print("\n\n************** COMP 379 HW2 | Lueders, Sebastian **************\n\n\n")

    while True:
        print("Available Predictive Models:\n")
        print("1. Logistic Regression Classification\n")
        print("2. Support Vector Classification\n")
        print("3. K Nearest Neighbors Classification\n")
        print("4. Dummy Classifier\n")
        model = input("Please enter the number corresponding to the model you want to use: ")
        if (int(model) < 5 and int(model) > 0):
            model_num = int(model)
            break
    
    file = input("Please provide the csv file name of your entire dataset (leave blank if using PhiUSIIL_Phishing_URL_Dataset.csv): ").strip()

    if file == "":
        file = "PhiUSIIL_Phishing_URL_Dataset.csv"


    print("Data Loading...\n")

    data = DatasetLoader(file)
    data.preprocess()
    train, dev, test = data.split_dataset()

    print("Data Loaded Successfully\n")

    if model_num == 1:
        c_value = input("What would you like to set the C value to (leave blank for default): ")
        if c_value == "":
            c_value = 1.0
        else:
            c_value = float(c_value)
        epochs = input("How many epochs? (Leave blank for default): ")
        if epochs == "":
            epochs = 100
        else:
            epochs = int(epochs)
        regularization = input("What type of regularization would you like to use? (l2/l1/elasticnet/none): ").strip().lower()
        if regularization not in ['l1', 'elasticnet', 'none']:
            regularization = "l2"
        rand_seed = input("Enter a random seed value for splitting the datasets (Leave blank for default of 42): ")
        if rand_seed == "":
            rand_seed = 42
        else:
            rand_seed = int(rand_seed)
        print("Training Model...\n")
        model = LRClassifier(train, dev, test, C=c_value, epochs=epochs, penalty=regularization, random_state=rand_seed)
        model.fit()
        print("Finished Training\n")
        model.evaluate()
        while True:
            next_instruction = input("Would you like to evaluate the model's performance on another dataset? (y/n): ").strip().lower()
            if next_instruction == 'y':
                model.evaluate()
            elif next_instruction == 'n':
                break
            else:
                print("Invalid Selection\n")
            
    elif model_num == 2:
        c_value = input("What would you like to set the C value to (leave blank for default): ")
        if c_value == "":
            c_value = 1.0
        else:
            c_value = float(c_value)
        epochs = input("How many epochs? (Leave blank for default): ")
        if epochs == "":
            epochs = -1
        else:
            epochs = int(epochs)
        kernel = input("What type of kernel function would you like to use? (rbf/linear/poly/sigmoid): ").strip().lower()
        if kernel not in ['linear', 'poly', 'sigmoid']:
            kernel = "rbf"
        rand_seed = input("Enter a random seed value for splitting the datasets (Leave blank for default of 42): ")
        if rand_seed == "":
            rand_seed = 42
        else:
            rand_seed = int(rand_seed)
        print("Training Model...\n")
        model = SVMClassifier(train, dev, test, C=c_value, epochs=epochs, kernel=kernel, random_state=rand_seed)
        model.train()
        print("Finished Training\n")
        model.evaluate()
        while True:
            next_instruction = input("Would you like to evaluate the model's performance on another dataset? (y/n): ").strip().lower()
            if next_instruction == 'y':
                model.evaluate()
            elif next_instruction == 'n':
                break
            else:
                print("Invalid Selection\n")

    elif model_num == 3:
        k = input("What would you like to set the k value to (leave blank for default of 3): ")
        if k == "":
            k = 3
        else:
            k = int(k)
        p = input("What would you like to set p to for the distance calculation? (Leave blank for the default of Euclidean distance): ")
        if p == "":
            p = 2.0
        else:
            p = float(p)
        eval = input("Would you like to evaluate this model's performance on the dev or test set? (dev/test): ").strip().lower()
        
        print("Training Model...\n")
        if eval == "test":
            model = KNNClassifier(train, test, k=k, p=p)
        else:
            model = KNNClassifier(train, dev, k=k, p=p)
        print("Finished Training... Proceeding to Model Evaluation\n")
        model.evaluate()

    elif model_num == 4:
        strat = input("What would you like to set the strategy to for the Dummy Classifier? (most_frequent/stratified/uniform/prior/constant): ")
        if strat not in ['stratified', 'uniform', 'prior', 'constant']:
            strat = "most_frequent"
        eval = input("Would you like to evaluate this model's performance on the dev or test set? (dev/test): ").strip().lower()
        print("Training Model...\n")
        if eval == "test":
            model = DummyModel(train, test, strategy=strat)
        else:
            model = DummyModel(train, dev, strategy=strat)
        model.train()
        print("Finished Training\n")
        model.evaluate()
    else:
        print("Invalid Model Selected\n")


    new_model = input("Would you like to train another model? (y/n): ").strip().lower()

    if new_model == 'y':
        main()
    else:
        print("Quitting Program...\n")
        


        
    
if __name__ == "__main__":
    
    main()




