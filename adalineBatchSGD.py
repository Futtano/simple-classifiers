import numpy as np

class AdalineBatchSGD:
    """A simple class implementation of the ADAptive LIner Neuron classifier using 
    mini-batch SGD for training.
    
    Parameters
    -----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Number of epochs (number of iterations over the training 
        dataset)
    batch_size: int
        Number of samples per batch
    shuffle: bool
        Set to True to shuffle randomly the samples for each epoch
    random_state: int
        Seed for the random initialization of the weights and bias.
    tol: float
        Loss value threshold below which the training stops, even if less than
        n_iter epochs were performed.
    """
    def __init__(self, eta=0.01, n_iter=50, batch_size=20, shuffle=True, random_state=1, tol=0.01 ):
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y):
        """Trains the Adaline classifier to fit the training data using mini-batch SGD

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

        n_samples = X.shape[0]

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            batch_losses = []
            for start_index in range(0, n_samples, self.batch_size):
                stop_index = start_index + self.batch_size
                X_b, y_b = X[start_index : stop_index], y[start_index : stop_index] 
                outputs = self._activation(self._netinput(X_b))
                errors = y_b - outputs
                self._w += self.eta * 2.0 * X_b.T.dot(errors) / X_b.shape[0]
                self._b += self.eta * 2.0 * errors.mean()
                batch_losses.append(np.mean(errors ** 2))

            avg_loss = np.mean(batch_losses)
            self.losses_.append(avg_loss)
            if avg_loss < self.tol:
                print(f'End of training, converged after {i + 1} iterations.')
                break
        else:
            print(f'End of training, max n of iterations ({self.n_iter}) reached.')    
        return self  
            
    def predict(self, X):
        """Calculate the output of the Adaline classifier and return class prediction"""
        net_input = self._netinput(X)
        return np.where(net_input >= 0.5, 1, 0)

    def _initwb(self, n_features):
        """Initialize weights and bias of the Adaline classifier"""
        self._rgen = np.random.default_rng(self.random_state)
        self._w = self._rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self._b = np.float_(0.)

    def _netinput(self, X):
        """Calculate the weighted sum input to the neuron"""
        return X @ self._w + self._b

    def _activation(self, X):
        """Adaline activation function"""
        return X

    def _shuffle(self, X, y):
        """Shuffle the training data to avoid bias"""
        p = self._rgen.permutation(len(y))
        return X[p], y[p]