import numpy as np

class Perceptron:
    """A simple class implementation of a Perceptron classifier

    Parameters
    -----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of epochs (number of iterations over the training 
        dataset)
    random_state: int
        Seed for the random initialization of the weights and bias.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Trains the Perceptron to fit the training data

        Parameters
        -----------
        X: [array-like], shape = [n_samples, n_features]
            Training vectors. n_features is the number of predictors
            per sample, while n_sample is the number of samples in 
            the training dataset
        y: array-like, shape = [n_samples]
            Vector with the corresponding class for each of the
            training samples in the training dataset

        Returns
        --------
        self:  object
        """
        self._initwb(X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                delta = self.eta * (target - self.predict(xi))
                self._w += delta * xi
                self._b += delta
                errors += int(delta != 0)
            self.errors_.append(errors)
            if errors == 0:
                print(f'End of training, converged after {i + 1} iterations.')
                break
        else:
            print(f'End of training, max n of iterations ({self.n_iter}) reached.')    
        return self
                
    def _net_input(self, X):
        """Calculate the weighted sum input to the neuron"""
        return np.dot(X, self._w) + self._b
    
    def predict(self, X):
        """Calculate the activation step function and return class prediction"""
        net_input = self._net_input(X)
        return np.where(net_input >= 0.0, 1, 0)

    def _initwb(self, n_features):
        """Initialize weights and bias of the Perceptron"""
        self._rgen = np.random.RandomState(self.random_state)
        self._w = self._rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self._b = np.float_(0.)