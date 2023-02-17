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
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    noIntercept : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    mse : TYPE
        DESCRIPTION.

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
