import numpy as np
from numpy.linalg import cholesky, lstsq
from scipy.linalg import cho_factor, cho_solve
from .kernel_function import CustomKernelFunctions


def nll_func(x_train, y_train, theta):
    """
    Returns a numerically stable function implementation of the negative log likelihood using the cholesky
    decomposition of the kernel matrix. http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section 2.2,
    Algorithm 2.1.
    :param x_train: Array of m points (m x d).
    :param y_train: Array of m points (m x 1)
    :return: negative log likelihood (scalar) computing function
    """

    l_params = np.exp(theta[:-2])
    sigma_f = np.exp(theta[-2])
    sigma_n = np.exp(theta[-1])

    kernel = CustomKernelFunctions(params={"l": l_params, "sigma_f": sigma_f})
    k_train = kernel(x_train, x_train) + sigma_n ** 2 * np.eye(len(x_train))
    l_mat, _ = fac = cho_factor(k_train)
    nll = (
        np.sum(np.log(np.diagonal(l_mat)))
        + 0.5 * y_train.T @ cho_solve(fac, y_train)
        + 0.5 * len(x_train) * np.log(2 * np.pi)
    )
    return nll
