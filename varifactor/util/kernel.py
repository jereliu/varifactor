# adapted heavily from package kgof/kernel by Wittawat Jitkrittum
import logging
import numpy as np

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kernel")


class kernel(object):
    """
    base class for differentiable kernel functions
    """

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        raise NotImplementedError()

    def grad(self, X, Y):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        raise NotImplementedError()

    def hess(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        raise NotImplementedError()


class RBF(kernel):
    """
    Gaussian rbf kernel with median heuristics
    """
    def __init__(self, sigma2=None):
        """
        :param sigma: wave length of gaussian kernel, if None then use median heuristics
        """
        self.sigma2 = sigma2

    def eval(self, X, Y=None):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        X, Y = self.check_input(X, Y)

        sumx2 = np.atleast_2d(np.sum(X ** 2, 1)).T
        sumy2 = np.atleast_2d(np.sum(Y ** 2, 1))
        D2 = sumx2 - 2 * np.dot(X, Y.T) + sumy2

        K = np.exp(-D2/(2.0 * self.sigma2))

        return K

    def grad(self, X, Y=None, dim=0, wrt="x"):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        X, Y = self.check_input(X, Y)

        # compute
        K = self.eval(X, Y)
        Diff = X[:, [dim]] - Y[:, [dim]].T
        G = -K * Diff/self.sigma2

        if wrt == "x":
            return G
        elif wrt == "y":
            return -G

    def hess(self, X, Y=None):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(x, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        X, Y = self.check_input(X, Y)

        #
        d = X.shape[1]
        sigma2 = self.sigma2

        D2 = np.sum(X**2, 1)[:, np.newaxis] - 2*np.dot(X, Y.T) + np.sum(Y**2, 1)
        K = np.exp(-D2/(2.0*sigma2))
        G = K/sigma2*(d - D2/sigma2)

        return G

    def set_sigma(self, X, Y=None):
        """
        Set sigma2 parameter using median heuristics
        :param X:
        :param Y:
        """
        X, Y = self.check_input(X, Y, check_sigma=False)

        sumx2 = np.atleast_2d(np.sum(X**2, 1)).T
        sumy2 = np.atleast_2d(np.sum(Y**2, 1))
        D2 = sumx2 - 2 * np.dot(X, Y.T) + sumy2

        sigma2 = np.median(D2)

        return sigma2

    def check_input(self, X, Y, check_sigma=True):
        if Y is None:
            Y = X
        else:
            (n1, d1) = X.shape
            (n2, d2) = Y.shape
            assert d1 == d2, 'Dimensions of the two inputs must be the same'
        if check_sigma:
            assert self.sigma2 is not None, 'sigma2 is empty, please run .set_sigma'
        return X, Y

