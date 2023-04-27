from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X)  # Numpy function to calculate mean of normal random variable

        if not self.biased_:  # If unbiased
            self.var_ = np.var(X, ddof=1)  # We have to divide by m-1 so we put ddof=1
        else:
            self.var_ = np.var(X)  # Else: anything but ddof=1

        self.fitted_ = True  # After we fit the variables, flag = True

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:  # Because we have to fit the variables before calculating PDF
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # Based on pdf formula for normal random variables:
        pdf_numerator = np.exp(np.power(X-self.mu_, 2) / (-2 * self.var_))
        pdf_denominator = np.sqrt(2 * np.pi * self.var_)

        return pdf_numerator / pdf_denominator

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        # Based on likelihood formula for normal random variables:
        likelihood_numerator = np.exp((-1 / (2 * sigma)) * (np.sum(np.power(X-mu, 2))))
        likelihood_denominator = np.power((2 * np.pi * sigma), np.size(X) / 2)

        return np.log(likelihood_numerator / likelihood_denominator)

class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X, axis=0)  # Calculates mean of each column and puts in an array
        self.cov_ = np.cov(X, rowvar=False)  # Rowvar = False because we want by column not by row

        self.fitted_ = True  # After we fit the variables, flag = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:  # Because we have to fit the variables before calculating PDF
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # We will play a bit with the multiplications of matrix, with their dimensions and get the formula:
        pdf_numerator = np.exp(-1 * np.sum((((X - self.mu_) @ inv(self.cov_)) * (X - self.mu_) / 2), axis=1))

        # Size of mu is the d in the formula
        pdf_denomenator = np.sqrt(np.power((2 * np.pi), np.size(self.mu_)) * det(self.cov_))

        return pdf_numerator / pdf_denomenator

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        d = np.size(mu)
        m = np.size(X, 0)  # Number of samples

        # We will calculate the log likelihood according to the formula we got in question 13 in the HW:
        res = (-m/2) * (d * np.log((2 * np.pi)) + slogdet(cov)[1]) + \
            np.sum((((X - mu) @ inv(cov)) * (X - mu)) / -2)

        return res


