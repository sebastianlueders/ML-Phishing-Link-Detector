#!/usr/bin/env python3

from sklearn.model_selection import train_test_split


class PhishingDetector:
    
    def __init__(self, lr=0.01, epochs=100, rseed=1, model_type):
        self.lr = lr
        self.epochs = epochs
        self.rseed = rseed
    
    
    




class Perceptron(PhishingDetector):
    
    def fit(self, t_data, targets):
        n_samples, n_features = t_data.shape
        self.initialize_weights(t_data) # Initializes a corresponding number of weights to the number of features in a dataset
        self.clear_weights_log() # Clears weights.csv data before training a new instance

        self.clear_costs_log() #Clears costs.csv data before training a new instance
        self.costs_ = [] #Initializes instance field to store cost value of each epoch iteration

        for i in range(self.epochs):
            net_input = self.net_input(t_data) #Returns an array of net input result of each row
            output = self.activation(net_input) #For Adaline, returns the net input vector unchanged (identity activation)
            errors = targets - output #Creates a vector to store the difference between actual value and net input
            self.weights_[1:] += self.lr * t_data.T.dot(errors) / n_samples #Updates the weight by multiplying lr by error by feature value & adding to original weight value
            self.weights_[0] += self.lr * errors.sum() / n_samples #Updates the bias by multiplying lr by the sum of all sample errors & adding to current bias value
            self.log_weights(self.weights_) #Updates weights.csv with weight value after each epoch
            cost = (errors ** 2).sum() / 2.0 #Calculates the cost/loss of this epoch iteration
            self.log_costs(cost) #Updates costs.csv with weight value after each epoch
            self.costs_.append(cost) #Adds this epoch's cost value to the costs vector
        return self
    


class Adaline(PhishingDetector):

    def fit(self, t_data, targets):
        self.initialize_weights(t_data) # Initializes a corresponding number of weights to the number of features in a dataset
        self.clear_weights_log() # Clears weights.csv data before training a new instance
        
        self.clear_costs_log() # Clears costs.csv data since perceptron dosen't include a cost function
        self.costs_ = [] # To store cost values

        for i in range(self.epochs):
            errors = 0 # Resets error value each epoch
            for xi, target in zip(t_data, targets):
                update = self.lr * (target - self.predict(xi)) #Calculates error times lr for every sample iteratively
                if update != 0.0:  # If an update is needed, the prediction was incorrect so we increment errors
                    errors += 1
                self.weights_[1:] += update * xi #Accounts for multiplying by xi before incrementing the weights
                self.weights_[0] += update #Since x0 would equal 1, updates bias without needing x0 coefficient
            self.log_weights(self.weights_) # Logs weights to weights.csv
            self.log_costs(errors)  # Logs the number of errors in each epoch (the cost of a perceptron model)
            self.costs_.append(errors) # adds to instance cost array
        return self


class LogisticRegression(PhishingDetector):

    def fit(self, t_data, targets):



    

def welcome_message():
    print("\n\n************** COMP 379 HW2 | Lueders, Sebastian **************\n\n\n")


if __name__ == "__main__":
    
    welcome_message()




