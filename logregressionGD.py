import numpy as np

class LogisticRegressionGD:
    """A simple class implementation of the Logistic Regression classifier trained with 
    full batch gradient descent.
    
    Parameters
    -----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of epochs (number of iterations over the training 
        dataset)
    random_state: int
        Seed for the random initialization of the weights and bias.
    tol: float
        Loss value threshold below which the training stops, even if less than
        n_iter epochs were performed.
    """

    def __init__(self, eta=0.1, n_iter=50, random_state=1, tol=0.01):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y):
        """Trains the Logistic Regression classifier to fit the training data

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
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self._net_input(X)
            output  = self._activation(net_input)
            errors = (y - output)
            loss = self._cross_entropy(y, output)
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            self.losses_.append(loss)
            if loss < self.tol:
                print(f'End of training, converged after {i + 1} iterations.')
                break       
        else:
            print(f'End of training, max n of iterations ({self.n_iter}) reached.')    
        return self
                
    def _net_input(self, X):
        """Calculate the weighted sum input to the neuron"""
        return np.dot(X, self.w_) + self.b_
    
    def _activation(self, z):
        """Sigmoid activation function"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def _cross_entropy(self, y, output):
        """Cross entropy loss function"""
        return  (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1-output)))) / len(y)
    
    def predict(self, X):
        """Calculate the output of the Logistic Regression classifier and return class prediction"""
        activation = self._activation(self._net_input(X))
        return np.where(activation >= 0.5, 1, 0)

    def _initwb(self, n_features):
        """Initialize weights and bias of the Logistic Regression classifier"""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = np.float_(0.0)