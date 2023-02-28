# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:37:13 2022

@author: LABadmin
"""

import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:37:13 2022

@author: LABadmin
"""


def linearAnalyticalSolution(x, y, noIntercept=False):
    """
    Fits a robust line to data by using the least squares function.

    Parameters
    ----------
    x : np.ndarray
        The values along the x axis.
    y : np.ndarray
        The values along the y axis.
    noIntercept : bool, optional
        Whether to not use the intercept. The default is False (intercept used).

    Returns
    -------
    a : float64
        The intercept of the fitted line.
    b : float64
        The slope of the fitted line.
    mse : float64
        The mean square error of the fit.

    """
    n = len(x)
    a = (np.sum(y) * np.sum(x ** 2) - np.sum(x) * np.sum(x * y)) / (
        n * np.sum(x ** 2) - np.sum(x) ** 2
    )
    b = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
        n * np.sum(x ** 2) - np.sum(x) ** 2
    )
    if noIntercept:
        b = np.sum(x * y) / np.sum(x ** 2)
    mse = (np.sum((y - (a + b * x)) ** 2)) / n
    return a, b, mse
