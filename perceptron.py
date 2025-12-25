import os
import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, eta, n_iter, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self._initwb_(X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                delta = self.eta * (target - self.predict(xi))
                if delta != 0.0:
                    self.w_ += delta * xi
                    self.b_ += delta
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                print(f'End of training, converged after {i + 1} iterations.')
                break
        else:
            print(f'End of training, max n of iterations ({self.n_iter}) reached.')    
        return self
                
    def _net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return np.where(self._net_input(X) < 0.0, 0, 1)

    def _initwb_(self, n_features):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = np.float_(0.0)

df = pd.read_csv("iris.data.csv", header=None, encoding='utf-8')
df.tail()