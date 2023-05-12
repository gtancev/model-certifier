import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from metas_unclib import *

class ModelCertifier:
    def __init__(self, 
                 n_folds=20, 
                 n_iterations=10000,
                 n_jobs=-1):
        """
        Initialize certifier class.

        n_points: number of samples in space for MC simulation
        n_folds: number of models
        n_iterations: number of iterations for MC simuation
        """
        self.n_folds = n_folds  # number of bootstrap folds
        self.n_iterations = n_iterations  # number of MC iterations
        self.n_jobs = n_jobs  # number of jobs
        self.models = []  # list to store fitted base models
        self.u_x = None  # stored standard uncertainty of x

    @ staticmethod
    def unclib_to_numpy(variables):
        """
        Function that transforms a collection of unclib variables
        in a list to two numpy arrays containing the values and the
        uncertainties.
        """
        values = []
        uncertainties = []
        for var in variables:
            values.append(var.value)
            uncertainties.append(var.stdunc)
        return np.asarray(values), np.asarray(uncertainties)
    
    @staticmethod
    def numpy_to_unclib(values, uncertainties, desc="o_"):
        """
        Function that transforms two numpy array of values and
        uncertainties to a list with the corresponding
        unclib elements.
        """
        assert values.shape[0] == uncertainties.shape[0]
        n = values.shape[0]
        elements = []
        for i in range(n):
            elements.append(ufloat(values[i], 
                                   uncertainties[i], 
                                   desc=desc+str(i)))
        return elements
    
    @staticmethod
    def distributed_training(X, y, model):
        """
        Function for distributed training.

        Input
        -----
        X, y: data set
        model: machine learning model with "fit" method
        """
        n = X.shape[0]
        samples = np.random.randint(0, n, n)
        X_b, y_b = X[samples], y[samples]
        model.fit(X_b, y_b)
        return model
    
    @staticmethod
    def distributed_noise_estimate(X, y, model, robust=False):
        """
        Function to compute the noise.

        Input
        -----
        X, y: data set
        model: machine learning model with "fit" method
        """
        y_pred = model.predict(X)
        sigma = np.sqrt(np.mean((y - y_pred)**2))
        if robust:
            sigma = np.sqrt(np.median((y - y_pred) ** 2))
        return sigma
    
    @staticmethod
    def distributed_Monte_Carlo(X, model, u_x, sigma, include_noise=True):
        """
        Function to perform distributed Monte Carlo.

        Input
        -----
        X: data set
        model: machine learning model with "fit" method
        u_x: uncertainty vector
        sigma: noise
        """
        n, p = X.shape
        eps_1 = np.random.normal(0, 1, (n, p))
        eps_2 = np.random.normal(0, 1, n)
        if (X.shape == u_x.shape):
            X_i = X + np.multiply(eps_1, u_x)
        else:
            X_i = X + np.dot(eps_1, u_x)
        y_i = model.predict(X_i)
        if include_noise:
            y_i = y_i + np.dot(eps_2, sigma)
        return y_i

    @staticmethod
    def get_min_max(X, axis=0):
        """
        Function to get min. and max. values of an array.

        Input
        -----
        X: design matrix
        """
        X_min, X_max = np.min(X, axis=axis), np.max(X, axis=axis)
        return X_min, X_max
    
    @staticmethod
    def sample_uniform(low, high, size):
        """
        Function to sample uniformly between
        low and high.
        """
        return np.random.uniform(low, high, size)
    
    @staticmethod
    def sample_normal(loc, scale, size):
        """
        Function to sample from normal distribution.
        """
        return np.random.normal(loc, scale, size)
    
    @staticmethod
    def sample_bootstrap(n):
        """
        Sample indicies for a bootstrap fold.
        """
        return np.random.randint(0, n, n)
    
    def create_test_data(self, X):
        """
        Method to create a test data set, which does not respect 
        the original data distribution (the resulting variables
        will be orthogonal).

        Input
        -----
        X: design matrix or vector
        """
        n, p = X.shape
        X_min, X_max = self.get_min_max(X)
        X_test = self.sample_uniform(X_min, X_max, (n, p))
        return X_test

    def make_prediction(self, X, single_sample=True):
        """
        Function that performs predictions
        but also reshapes input if necessary.

        Input
        -----
        X: design matrix or vector
        single sample: whether X contains only one sample
        """
        n_models = len(self.models)
        if n_models == 0:
            return print("Estimate uncertainty first.")
        if (len(X.shape) == 1):
            if single_sample:
                X = X.reshape(1, -1)
            else:
                X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, n_models))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X).ravel()
        mu, std = np.mean(predictions, axis=1), np.std(predictions, axis=1)
        return mu, std, predictions
    
    def estimate_uncertainty(self, X, y, base_model, u_x, X_test=None, bootstrap=True, include_noise=True, robust=False):
        """
        Function that performs black-box uncertainty estimation.

        Input
        -----
        X, y: input data set D of shape (n, p) and (n, )
        base_model: model class to be evaluated
        u_x: standard uncertainty for X of shape (1) or (p, ) or shape of X_test
        X_test: test data set; will be created if not provided
        bootstrap: wether to use bootstrapping to estimate model uncertainty
        include_noise: whether to add noise to the predictions
        robust: wether to use robust statistics (i.e., median instead of mean)

        Output
        ------
        X_test, predictions: matrix of MC samples at evaluation points X_test
        """
        # Create data if no test dataset has been provided.
        if X_test is None:
            X_test = self.create_test_data(X)
        n_samples, _ = X_test.shape

        # Allocate memory.
        predictions = np.zeros((n_samples, self.n_folds * self.n_iterations))
        self.u_x = np.array(u_x)
        if (X_test.shape != self.u_x.shape):
            self.u_x = np.diag(self.u_x)

        # Distributed training to estimate epistemic uncertainty.
        if bootstrap:
            base_model = deepcopy(base_model)
            self.models = Parallel(n_jobs=-1)(delayed(self.distributed_training)(X, 
                                                                                 y, 
                                                                                 base_model) for _ in range(self.n_folds))
        else:
            self.models.append(base_model)

        # Estimate average noise across models.
        sigma = Parallel(n_jobs=-1)(delayed(self.distributed_noise_estimate)(X, 
                                                                             y, 
                                                                             model,
                                                                             robust) for model in self.models)
        sigma = np.mean(np.array(sigma)) # this could be a biased estimate...
    
        # Simulate uncertainty in input / output.
        for b, model in enumerate(self.models):
            y_i = Parallel(n_jobs=-1)(delayed(self.distributed_Monte_Carlo)(X_test, 
                                                                            model, 
                                                                            self.u_x, 
                                                                            sigma, 
                                                                            include_noise) for _ in range(self.n_iterations))
            predictions[:, b * self.n_iterations : (b + 1) * self.n_iterations] = np.array(y_i).T
        return X_test, predictions
