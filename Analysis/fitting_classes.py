# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:03:33 2023

@author: Liad
"""
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
import numpy as np
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl
import sys
import dask.array as da
import pandas as pd
import re
import traceback
from numba import jit, cuda

from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

import os
import glob
import pickle
from numba import jit

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import acf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf

import traceback
from abc import ABC, abstractmethod
import inspect


# a base class for all fitting classes
class BaseTuner(ABC):

    # separator between groups
    sep = None
    func = None
    p0 = None
    bounds = None
    removed = 0
    state = 0
    fitScore = 0

    def __init__(self, sep=None, xtol=10**-10, max_nfev=5000):
        """
        Initialising the class

        Parameters
        ----------
        sep : int, optional
            When fitting two different datasets, the separation point between them (how long the first dataset is). The default is None.
        xtol : float, optional
            tolerance of fitting function (recommended to leave at default). The default is 10**-10.
        max_nfev : int, optional
            maximum evaluations (leave at default). The default is 5000.

        Returns
        -------
        None.

        """
        self.sep = sep
        self.removed = 0
        self.xtol = xtol
        self.xtol = xtol
        self.max_nfev = max_nfev
        pass

    @abstractmethod
    def set_function(self, *args):
        """
        Implement this function when building a fitting class.
        follow roughly the structure of this model
        this function sets the correct function for the class to use in fitting

        Parameters
        ----------
        *args : any
            use this to decide how to select a function to use.

        Returns
        -------
        None.

        """
        self.func = np.nanmean

    @abstractmethod
    def set_bounds_p0(self, x, y, *args):
        """
        Implement this method whenwhen building a fitting class.
        This function sets the bounds and preliminary guess for the fitter.
        format is tuples.
        bounds is ((low1,low2,low3,...),(up1,up2,up3,...))
        p0 is (guess1,guess2,guess3,...)
        Parameters
        ----------
        x : array of float
            the xs to fit.
        y : array of float
            the ys of the fit.
        *args : any
            anything else one might want to use to make bounds.

        Returns
        -------
        None.

        """
        self.bounds = None
        self.p0 = None

    def predict(self, x):
        """
        predicts the ys from given xs

        Parameters
        ----------
        x : array of float
            the xs to guess from.

        Returns
        -------
        float
            the guess.

        """
        return self.func(x, *self.props)

    # @jit(target_backend="cuda")
    def fit_(self, x, y, func, p0, bounds):
        try:
            props = np.nan
            props, _ = sp.optimize.curve_fit(
                func,
                x,
                y,
                p0=p0,
                bounds=bounds,
                xtol=self.xtol,
                # max_nfev=self.max_nfev,
                method="trf",
                loss="soft_l1",
            )
            return props
        except:
            print(traceback.format_exc())
            return np.nan

    def fit(self, x, y, save=True):
        """
        fits the function given the parameters.
        and (optionally) saves the results

        Parameters
        ----------
        x : array of float
            the xs to fit.
        y : array of float
            the ys of the fit.
        save : bool, optional
            saves the fitted parameters in the class. The default is True.

        Returns
        -------
        props : float
            the fitted parameters.

        """
        props = np.nan
        p0, bounds = self.set_bounds_p0(x, y)

        props = self.fit_(x, y, self.func, p0, bounds)

        if save:
            self.props = props
            self.p0 = p0
            self.bounds = bounds
            if not (np.all(np.isnan(props))):
                preds = self.predict(x)
            else:
                preds = np.ones_like(x) * np.nan
            if not np.any(np.isnan(props)):
                self.fitScore = self.score(preds, y)
            else:
                print("could not fit")
                self.fitScore = 0
        return props

    def predict_constant(self, x, y):
        return np.nanmean(y)

    def split_cv(self, x, y, split=0.1):
        N = len(x)
        p0, bounds = self.set_bounds_p0(x, y)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=split
        )
        props = self.fit_(X_train, y_train, self.func, p0, bounds)
        R2 = np.nan
        if not np.any(np.isnan(props)):
            preds = self.func(X_test, *props)
        R2 = self.score(preds, y_test)
        return R2

    def split_cv_constant(self, x, y, split=0.1):
        N = len(x)
        p0, bounds = self.set_bounds_p0(x, y)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=split
        )
        preds = np.repeat(np.nanmean(y_test), len(y_test))
        R2 = self.score(preds, y_test)
        return R2

    def loo_constant(self, x, y):
        N = len(x)
        preds = np.ones_like(y) * np.nan
        p0, bounds = self.set_bounds_p0(x, y)
        for i in range(N):
            preds[i] = np.nanmean(np.delete(y, i, axis=0))
        R2 = self.score(preds, y)
        return R2

    # leave one out
    # @jit(target_backend="cuda")
    def loo(self, x, y):
        """
        Leave One Out scoring.

        Parameters
        ----------
        x : array of float
            the xs to fit.
        y : array of float
            the ys of the fit.

        Returns
        -------
        R2 : float
            how well the function performed.

        """
        N = len(x)
        preds = np.ones_like(y) * np.nan
        p0, bounds = self.set_bounds_p0(x, y)
        for i in range(N):
            l = x[i]
            try:
                if not (self.sep is None):
                    self.sep -= 1
                props = self.fit_(
                    x[np.setdiff1d(range(x.shape[0]), i)],
                    y[np.setdiff1d(range(y.shape[0]), i)],
                    self.func,
                    p0,
                    bounds,
                )
                if not (self.sep is None):
                    self.sep += 1
                    self.state = i
                if np.any(np.isnan(props)):
                    preds[i] = np.nan
                else:
                    preds[i] = self.func(l, *props)
            except:
                print(traceback.format_exc())
        R2 = self.score(preds, y)
        return R2

    @abstractmethod
    def predict_split(self, x, state):
        """
        Implement this function when implementing a new fitting class
        function used to predict the values given x in different state.
        uses the state toggle to signal what state to use/

        Parameters
        ----------
        x : array of float
            the xs to fit.
        state : any
            used to tell the function what state to fit (e.g. quiet/active).

        Returns
        -------
        float
            the guesses given the x.

        """
        return np.nan

    def auc_diff(self, x):
        valRange = np.arange(np.min(x), np.max(x), 0.01)
        pred1 = self.predict_split(valRange, 0)
        pred2 = self.predict_split(valRange, 1)
        return np.trapz(np.abs(pred2 - pred1))

    def shuffle_split(self, x, y, nshuff=500):
        """
        Creates a null distribution for the AUC of the difference between states.

        Parameters
        ----------
        x : array of float
            the xs to fit.
        y : array of float
            the ys of the fit.
        nshuff : int, optional
            The number of shuffles. The default is 500.

        Returns
        -------
        dist : array of float
            the null distribution.

        """
        dist = np.zeros(nshuff)
        valRange = np.arange(np.min(x), np.max(x), 0.01)
        vrn = len(valRange)
        vals = np.append(valRange, valRange)
        sep_save = self.sep
        self.sep = vrn
        for i in range(nshuff):
            ind_surr = np.random.permutation(len(x))
            x_surr = x.copy()[ind_surr]
            y_surr = y.copy()[ind_surr]
            props = self.fit(x_surr, y_surr, save=False)
            if not (np.any(np.isnan(props))):
                preds = self.func(vals, *props)
                pred1 = preds[:vrn]
                pred2 = preds[vrn:]
                diff_auc = np.trapz(np.abs(pred2 - pred1))
                dist[i] = diff_auc
            else:
                dist[i] = 0
        self.sep = sep_save
        return dist

    # @jit(target_backend="cuda", forceobj=True)
    def shuffle(self, x, y, nshuff=200):
        """
        Creates a null distribution for the single fit

        Parameters
        ----------
        x : array of float
            the xs to fit.
        y : array of float
            the ys of the fit.
        nshuff : int, optional
            The number of shuffles. The default is 200.

        Returns
        -------
        evDist : array of float
            the null disribution of the explained variance.
        propsDist : TYPE
            the null disribution of the parameters.

        """
        evDist = np.zeros(nshuff)
        propsDist = np.zeros((nshuff, self.func.__code__.co_argcount - 2))
        for i in range(nshuff):
            x_surr = np.random.permutation(x)
            props = self.fit(x_surr, y, save=False)
            if not (np.any(np.isnan(props))):
                preds = self.func(x_surr, *props)
                sc = self.score(preds, y)
                evDist[i] = sc
                propsDist[i, :] = props
            else:
                evDist[i] = 0
                propsDist[i, :] = np.nan
        return evDist, propsDist

    def score(self, pred, true):
        """
        Returns the score of the fit - measured with explained variance,
        by comparing the predicitons and true values

        Parameters
        ----------
        pred : array of float
            The predicted values.
        true : array of float
            The actual values.

        Returns
        -------
        varExplained : float
            explained variance.

        """
        u = np.nansum((np.array(pred) - true) ** 2)
        v = np.nansum((true - true.mean()) ** 2)
        varExplained = 1 - u / (v + 0.000001)
        return varExplained

    def score_mean(self, true):
        return self.score(np.repeat(np.nanmean(true), len(true)), true)

    def get_parameters(self):
        """
        returns a text version of the function parameters

        Returns
        -------
        string
            the text of the fit parameters.

        """
        return inspect.signature(self.func)


# fit using log - gaussian
class FrequencyTuner(BaseTuner):
    def __init__(self, funcName, sep=None):
        BaseTuner.__init__(self, sep)
        self.set_function(funcName)

    # parameters: name of function
    def set_function(self, *args):
        if args[0] == "gauss":
            self.func = self.gauss
        if args[0] == "gauss_split":
            self.func = self.gauss_split

    def _make_prelim_guess(self, x, y):
        # get average per ori
        xu = np.unique(x)
        avgy = np.zeros_like(xu, dtype=float)
        for xi, xuu in enumerate(xu):
            avgy[xi] = np.nanmedian(y[x == xuu])
        return (
            np.nanmin(y),
            np.nanmax(avgy),
            xu[np.nanargmax(avgy)],
            1,
        )

    def set_bounds_p0(self, x, y, func=None):

        p0 = self._make_prelim_guess(x, y)
        xu = np.unique(x)
        minDiff = np.min(np.diff(xu, axis=0))
        maxRange = x[-1] - x[0]
        bounds = (
            (np.nanmin(y), np.nanmin(y), np.min(xu), 1),
            (np.nanmax(y), np.nanmax(y), np.max(xu), 10),
        )
        if ((func is None) & (self.func == self.gauss)) | (
            (not (func is None)) & (func == self.gauss)
        ):
            return p0, bounds  # just take default params

        if ((func is None) & (self.func == self.gauss_split)) | (
            (not (func is None)) & (func == self.gauss_split)
        ):
            try:
                meanVals = pd.DataFrame({"x": x, "y": y}).groupby("x").mean()
                x_mean = meanVals.index.to_numpy()
                y_mean = meanVals["y"].to_numpy()
                p0_, _ = sp.optimize.curve_fit(
                    self.gauss,
                    x_mean,
                    y_mean,
                    p0=p0,
                    bounds=bounds,
                    xtol=self.xtol,
                    max_nfev=self.max_nfev,
                    method="trf",
                    loss="soft-l1",
                    f_scale=1,
                    x_scale="jac",
                )

                p0 = (
                    p0_[0],
                    p0_[0],
                    p0_[1],
                    p0_[1],
                    p0_[2],
                    p0_[2],
                    p0_[3],
                    p0_[3],
                )

                bounds = (
                    (
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        np.min(xu),
                        np.min(xu),
                        1,
                        1,
                    ),
                    (
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        np.max(xu),
                        np.max(xu),
                        10,
                        10,
                    ),
                )
            except:
                p0 = (p0[0], p0[0], p0[1], p0[1], p0[2], p0[2], p0[3], p0[3])

                bounds = (
                    (
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        -np.inf,
                        np.min(xu),
                        np.min(xu),
                        minDiff,
                        minDiff,
                    ),
                    (
                        np.inf,
                        np.inf,
                        np.inf,
                        np.inf,
                        np.max(xu),
                        np.max(xu),
                        10,
                        10,
                    ),
                )
            return p0, bounds

    def gauss(self, f, bl, A, f0, sig):
        res = bl + A * np.exp(
            (-((np.log2(f + np.finfo(float).eps) - np.log2(f0)) ** 2))
            / (2 * sig**2 + np.finfo(float).eps)
        )
        return res

    def gauss_split(self, f, blq, bla, Aq, Aa, f0q, f0a, sigq, siga):
        sep = self.sep
        # have one state only to predict
        if len(np.atleast_1d(f)) == 1:
            if self.state <= sep:
                # quiet
                y = self.gauss(f, blq, Aq, f0q, sigq)
            else:
                # active
                y = self.gauss(f, bla, Aa, f0a, siga)
            return y

        if not (sep is None):
            quiet = f[:sep]
            active = f[sep:]
            yq = self.gauss(quiet, blq, Aq, f0q, sigq)
            ya = self.gauss(active, bla, Aa, f0a, siga)
            return np.append(yq, ya)
        else:
            return np.nan

    def predict_split(self, x, state):
        if state == 0:
            # quiet
            return self.gauss(x, *self.props[::2])
        else:
            return self.gauss(x, *self.props[1::2])


# fit using warpped gaussian estimate
class OriTuner(BaseTuner):
    def __init__(self, funcName, sep=None):
        BaseTuner.__init__(self, sep)
        self.set_function(funcName)

    # parameters: name of function
    def set_function(self, *args):
        if args[0] == "gauss":
            self.func = self.wrapped_gauss
        if args[0] == "gauss_split":
            self.func = self.wrapped_gauss_split

    def _make_prelim_guess(self, x, y):
        # get average per ori
        xu = np.unique(x)
        avgy = np.zeros_like(xu, dtype=float)
        for xi, xuu in enumerate(xu):
            avgy[xi] = np.nanmedian(y[x == xuu])
        return (
            np.nanmin(y),
            np.nanmax(avgy),
            0.5,
            xu[np.nanargmax(avgy)],
            np.abs(xu[np.nanargmax(avgy)] - xu[np.nanargmin(avgy)]),
        )

    def set_bounds_p0(self, x, y, func=None):

        p0 = self._make_prelim_guess(x, y)
        bounds = (
            (np.nanmin(y), np.nanmin(y), 0, 0, 0.5),
            (np.nanmax(y), np.nanmax(y), 1, 360, 360),
        )
        if ((func is None) & (self.func == self.wrapped_gauss)) | (
            (not (func is None)) & (func == self.wrapped_gauss)
        ):
            return p0, bounds  # just take default params

        if ((func is None) & (self.func == self.wrapped_gauss_split)) | (
            (not (func is None)) & (func == self.wrapped_gauss_split)
        ):
            try:

                meanVals = pd.DataFrame({"x": x, "y": y}).groupby("x").mean()
                x_mean = meanVals.index.to_numpy()
                y_mean = meanVals["y"].to_numpy()
                p0_, _ = sp.optimize.curve_fit(
                    self.wrapped_gauss,
                    x_mean,
                    y_mean,
                    p0=p0,
                    bounds=bounds,
                    xtol=self.xtol,
                    max_nfev=self.max_nfev,
                    method="trf",
                    loss="soft_l1",
                    f_scale=1,
                )

                # p0_, _ = sp.optimize.curve_fit(
                #     self.wrapped_gauss,
                #     x,
                #     y,
                #     p0=p0,
                #     bounds=bounds,
                #     xtol=self.xtol,
                #     max_nfev=self.max_nfev,
                #     method="trf",
                #     loss="soft_l1",
                #     f_scale=1,
                # )

                p0 = (p0_[0], p0_[0], p0_[1], p0_[1], p0_[2], p0_[3], p0_[4])

                bounds = (
                    (
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        0,
                        0,
                        0.5,
                    ),
                    (
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        1,
                        360,
                        360,
                    ),
                )
            except:
                p0 = (p0[0], p0[0], p0[1], p0[1], p0[2], p0[3], p0[4])

                bounds = (
                    (
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        0,
                        0,
                        0.5,
                    ),
                    (
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        1,
                        360,
                        360,
                    ),
                )
            return p0, bounds

    def wrapped_gauss(self, oris, R0, Rp, DI, dp, s):

        Rn = (Rp - DI * Rp) / (1 + DI)

        n = 5
        gauss1 = Rp * np.sum(
            np.exp(
                -0.5
                * (
                    (np.tile(oris - dp, (2 * n, 1)).T + 360 * np.arange(-n, n))
                    / s
                )
                ** 2
            ),
            1,
        )
        gauss2 = Rn * np.sum(
            np.exp(
                -0.5
                * (
                    (
                        np.tile(oris - dp + 180, (2 * n, 1)).T
                        + 360 * np.arange(-n, n)
                    )
                    / s
                )
                ** 2
            ),
            1,
        )
        gauss = R0 + gauss1 + gauss2
        return gauss

    def wrapped_gauss_split(self, oris, R0a, R0q, Rpa, Rpq, DI, dp, s):
        sep = self.sep
        # have one state only to predict
        if len(np.atleast_1d(oris)) == 1:
            if self.state <= sep:
                # quiet
                y = self.wrapped_gauss(oris, R0q, Rpq, DI, dp, s)
            else:
                # active
                y = self.wrapped_gauss(oris, R0a, Rpa, DI, dp, s)
            return y

        if not (sep is None):
            quiet = oris[:sep]
            active = oris[sep:]
            yq = self.wrapped_gauss(quiet, R0q, Rpq, DI, dp, s)
            ya = self.wrapped_gauss(active, R0a, Rpa, DI, dp, s)
            return np.append(yq, ya)
        else:
            return np.nan

    def predict_split(self, x, state):
        if state == 0:
            return self.wrapped_gauss(x, *self.props[[1, 3, 4, 5, 6]])
        else:
            return self.wrapped_gauss(x, *self.props[[0, 2, 4, 5, 6]])


# fit using hyperbolic ratio function
class ContrastTuner(BaseTuner):
    def __init__(self, funcName, sep=None):
        BaseTuner.__init__(self, sep)
        self.set_function(funcName)

    def set_function(self, *args):
        if args[0] == "contrast":
            self.func = self.hyperbolic
        if args[0] == "contrast_split":
            self.func = self.hyperbolic_split

    def _make_prelim_guess(self, x, y):
        # get average per ori
        xu = np.unique(x)
        avgy = np.zeros_like(xu, dtype=float)
        for xi, xuu in enumerate(xu):
            avgy[xi] = np.nanmedian(y[x == xuu])
        return (
            np.nanmin(y),
            np.nanmax(avgy),
            0.5,
            2,
        )

    def set_bounds_p0(self, x, y, func=None):

        p0 = self._make_prelim_guess(x, y)
        bounds = (
            (np.nanmin(y), np.nanmin(y), 0.01, 0),
            (np.nanmax(y), np.nanmax(y), 1, 10),
        )
        if ((func is None) & (self.func == self.hyperbolic)) | (
            (not (func is None)) & (func == self.hyperbolic)
        ):
            return p0, bounds  # just take default params

        if ((func is None) & (self.func == self.hyperbolic_split)) | (
            (not (func is None)) & (func == self.hyperbolic_split)
        ):
            try:

                meanVals = pd.DataFrame({"x": x, "y": y}).groupby("x").mean()
                x_mean = meanVals.index.to_numpy()
                y_mean = meanVals["y"].to_numpy()
                p0_, _ = sp.optimize.curve_fit(
                    self.hyperbolic,
                    x_mean,
                    y_mean,
                    p0=p0,
                    bounds=bounds,
                    xtol=self.xtol,
                    max_nfev=self.max_nfev,
                    method="trf",
                    loss="soft_l1",
                    f_scale=1,
                )

                # p0_, _ = sp.optimize.curve_fit(
                #     self.wrapped_gauss,
                #     x,
                #     y,
                #     p0=p0,
                #     bounds=bounds,
                #     xtol=self.xtol,
                #     max_nfev=self.max_nfev,
                #     method="trf",
                #     loss="soft_l1",
                #     f_scale=1,
                # )

                p0 = (p0_[0], p0_[0], p0_[1], p0_[1], p0_[2], p0_[3])

                bounds = (
                    (
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        0.01,
                        0,
                    ),
                    (
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        1,
                        10,
                    ),
                )
            except:
                p0 = (p0[0], p0[0], p0[1], p0[1], p0[2], p0[3])

                bounds = (
                    (
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        np.nanmin(y),
                        0.01,
                        0,
                    ),
                    (
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        np.nanmax(y),
                        1,
                        10,
                    ),
                )
            return p0, bounds

    def predict_split(self, x, state):
        if state == 0:
            return self.hyperbolic(x, *self.props[[0, 2, 4, 5]])
        else:
            return self.hyperbolic(x, *self.props[[1, 3, 4, 5]])

    def hyperbolic(self, c, R0, R, c50, n):
        return R * (c**n / (c50**n + c**n)) + R0

    def hyperbolic_split(self, c, R0q, R0a, Rq, Ra, c50, n):
        sep = self.sep
        # have one state only to predict
        if len(np.atleast_1d(c)) == 1:
            if self.state <= sep:
                # quiet
                y = self.hyperbolic(c, R0q, Rq, c50, n)
            else:
                # active
                y = self.hyperbolic(c, R0a, Ra, c50, n)
            return y

        if not (sep is None):
            quiet = c[:sep]
            active = c[sep:]
            yq = self.hyperbolic(quiet, R0q, Rq, c50, n)
            ya = self.hyperbolic(active, R0a, Ra, c50, n)
            return np.append(yq, ya)
        else:
            return np.nan


class GammaTuner(BaseTuner):
    def __init__(self, funcName, sep=None):
        BaseTuner.__init__(self, sep)
        self.set_function(funcName)

    def set_function(self, *args):
        if args[0] == "gamma":
            self.func = self.gamma
        if args[0] == "gamma_split":
            self.func = self.gamma_split

    def _make_prelim_guess(self, x, y):
        # get average per ori
        xu = np.unique(x)
        avgy = np.zeros_like(xu, dtype=float)
        for xi, xuu in enumerate(xu):
            avgy[xi] = np.nanmean(y[x == xuu])
        return (
            np.nanmin(avgy),
            np.nanmax(avgy),
            1,
            0,
        )

    def set_bounds_p0(self, x, y, func=None):
        p0 = self._make_prelim_guess(x, y)
        bounds = (
            (
                np.nanmin(y),
                np.nanmin(y),
                0.01,
                0,
            ),
            (
                np.nanmax(y),
                np.nanmax(y),
                100,
                100,
            ),  # np.nanmax(y),  # np.nanmax(y),
        )
        if ((func is None) & (self.func == self.gamma)) | (
            (not (func is None)) & (func == self.gamma)
        ):
            return p0, bounds  # just take default params

        if ((func is None) & (self.func == self.gamma_split)) | (
            (not (func is None)) & (func == self.gamma_split)
        ):
            try:

                meanVals = pd.DataFrame({"x": x, "y": y}).groupby("x").mean()
                x_mean = meanVals.index.to_numpy()
                y_mean = meanVals["y"].to_numpy()
                p0_, _ = sp.optimize.curve_fit(
                    self.gamma,
                    x_mean,
                    y_mean,
                    p0=p0,
                    bounds=bounds,
                    xtol=self.xtol,
                    max_nfev=self.max_nfev,
                    method="trf",
                    loss="soft_l1",
                    f_scale=1,
                )

                # p0_, _ = sp.optimize.curve_fit(
                #     self.wrapped_gauss,
                #     x,
                #     y,
                #     p0=p0,
                #     bounds=bounds,
                #     xtol=self.xtol,
                #     max_nfev=self.max_nfev,
                #     method="trf",
                #     loss="soft_l1",
                #     f_scale=1,
                # )

                p0 = (
                    p0_[0],
                    p0_[0],
                    p0_[1],
                    p0_[1],
                    p0_[2],
                    p0_[2],
                    p0_[3],
                    p0_[3],
                )

                bounds = (
                    (
                        bounds[0][0],
                        bounds[0][0],
                        bounds[0][1],
                        bounds[0][1],
                        bounds[0][2],
                        bounds[0][2],
                        bounds[0][3],
                        bounds[0][3],
                    ),
                    (
                        bounds[1][0],
                        bounds[1][0],
                        bounds[1][1],
                        bounds[1][1],
                        bounds[1][2],
                        bounds[1][2],
                        bounds[1][3],
                        bounds[1][3],
                    ),
                )
            except:
                p0 = (
                    p0[0],
                    p0[0],
                    p0[1],
                    p0[1],
                    p0[2],
                    p0[2],
                    p0[3],
                    p0[3],
                )

                bounds = (
                    (
                        bounds[0][0],
                        bounds[0][0],
                        bounds[0][1],
                        bounds[0][1],
                        bounds[0][2],
                        bounds[0][2],
                        bounds[0][3],
                        bounds[0][3],
                    ),
                    (
                        bounds[1][0],
                        bounds[1][0],
                        bounds[1][1],
                        bounds[1][1],
                        bounds[1][2],
                        bounds[1][2],
                        bounds[1][3],
                        bounds[1][3],
                    ),
                )
            return p0, bounds

    def predict_split(self, x, state):
        if state == 0:
            return self.gamma(x, *self.props[[0, 2, 4, 6, 8]])
        else:
            return self.gamma(x, *self.props[[1, 3, 5, 7, 9]])

    def gamma(self, s, r0, A, a, n):
        # r0 = np.float32(r0)
        # A = np.float32(A)
        # a = np.float32(a)
        tau = 0
        # n = int(n)
        res = r0 + A * (
            (((a * (s - tau)) ** n) * np.exp(-a * (s - tau)))
            / ((n**n) * np.exp(-n))
        )

        return res

    def gamma_split(self, s, r0q, r0a, Aq, Aa, aq, aa, nq, na):
        sep = self.sep

        # have one state only to predict
        if len(np.atleast_1d(c)) == 1:
            if self.state <= sep:
                # quiet
                y = self.gamma(s, r0q, Aq, aq, nq)
            else:
                # active
                y = self.gamma(s, r0a, Aa, aa, na)
            return y

        if not (sep is None):
            quiet = c[:sep]
            active = c[sep:]
            yq = self.gamma(s, r0q, Aq, aq, nq)
            ya = self.gamma(s, r0a, Aa, aa, na)
            return np.append(yq, ya)
        else:
            return np.nan

        return res

    # def gamma(self, s, r0, A, a, tau, n):
    #     # r0 = np.float32(r0)
    #     # A = np.float32(A)
    #     # a = np.float32(a)
    #     n = int(n)
    #     res = r0 + A * (
    #         (((a * (s - tau)) ** n) * np.exp(-a * (s - tau)))
    #         / ((n**n) * np.exp(-n))
    #     )

    #     return res

    # def gamma_split(self, s, r0q, r0a, Aq, Aa, aq, aa, tauq, taua, nq, na):
    #     sep = self.sep
    #     # have one state only to predict
    #     if len(np.atleast_1d(c)) == 1:
    #         if self.state <= sep:
    #             # quiet
    #             y = self.gamma(s, r0q, Aq, aq, tauq, nq)
    #         else:
    #             # active
    #             y = self.gamma(s, r0a, Aa, aa, taua, na)
    #         return y

    #     if not (sep is None):
    #         quiet = c[:sep]
    #         active = c[sep:]
    #         yq = self.gamma(s, r0q, Aq, aq, tauq, nq)
    #         ya = self.gamma(s, r0a, Aa, aa, taua, na)
    #         return np.append(yq, ya)
    #     else:
    #         return np.nan

    #     return res


class Gauss2DTuner(BaseTuner):
    def __init__(self, funcName, bestSpot, sep=None):
        BaseTuner.__init__(self, sep)
        self.set_function(funcName)
        self.maxSpot = bestSpot

    def set_function(self, *args):
        if args[0] == "gauss":
            self.func = self.gauss_2d
        if args[0] == "gauss_split":
            self.func = self.gauss2d_split

    def _make_prelim_guess(self, x, y):
        # get average per ori
        xdiff = np.nanmedian(np.diff(np.unique(x[:, 0])))
        ydiff = np.nanmedian(np.diff(np.unique(x[:, 1])))

        df = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "resp": y})
        means = df.groupby(["x", "y"]).mean().reset_index()
        maxValInd = np.argmax(means["resp"])
        maxVal = means.iloc[maxValInd]["resp"]
        maxX = self.maxSpot[1]  # means.iloc[maxValInd]["x"]
        maxY = self.maxSpot[0]  # means.iloc[maxValInd]["y"]
        p0 = (
            maxVal,
            maxX,
            maxY,
            xdiff,
            ydiff,
            0,
            0,
        )
        return p0

    def set_bounds_p0(self, x, y, func=None):

        p0 = self._make_prelim_guess(x, y)
        maxX = np.nanmax(x[:, 0])
        minX = np.nanmin(x[:, 0])
        maxY = np.nanmax(x[:, 1])
        minY = np.nanmin(x[:, 1])
        # 1 sd cannot exceed boarders
        maxA = np.max([np.abs(maxX - p0[1]), np.abs(minX - p0[1])])
        maxB = np.max([np.abs(maxY - p0[2]), np.abs(minY - p0[2])])
        maxSd = np.max([maxA, maxB])

        xu = np.unique(x[:, 0])
        yu = np.unique(x[:, 1])
        xdiff = np.nanmean(np.diff(xu))
        ydiff = np.nanmean(np.diff(yu))
        minDiff = np.nanmin([xdiff, ydiff])

        possibleMaxX = np.nanmax(x[:, 0]) - np.nanmin(x[:, 0])
        possibleMaxY = np.nanmax(x[:, 1]) - np.nanmin(x[:, 1])

        bounds = (
            (
                0,
                self.maxSpot[1] - 5,  # ,np.nanmin(x[:, 0])
                self.maxSpot[0] - 5,  # np.nanmin(x[:, 1])
                0.5 / 2,
                0.5 / 2,
                0,
                0,
            ),
            (
                1,
                self.maxSpot[1] + 5,  # ,np.nanmax(x[:, 0])
                self.maxSpot[0] + 5,  # np.nanmax(x[:, 1])
                maxSd,  # possibleMaxX,
                maxSd,  # possibleMaxY,
                np.pi,
                1,
            ),
        )

        if ((func is None) & (self.func == self.gauss_2d)) | (
            (not (func is None)) & (func == self.gauss_2d)
        ):
            return p0, bounds  # just take default params

        if ((func is None) & (self.func == self.gauss_2d_split)) | (
            (not (func is None)) & (func == self.gauss_2d_split)
        ):

            p0 = (
                p0[0],
                p0[0],
                p0[1],
                p0[2],
                p0[3],
                p0[3],
                p0[4],
                p0[4],
                p0[5],
                p0[6],
                p0[6],
            )

            bounds = (
                (
                    bounds[0][0],
                    bounds[0][0],
                    bounds[0][1],
                    bounds[0][2],
                    bounds[0][3],
                    bounds[0][3],
                    bounds[0][4],
                    bounds[0][4],
                    bounds[0][5],
                    bounds[0][6],
                    bounds[0][6],
                ),
                (
                    bounds[1][0],
                    bounds[1][0],
                    bounds[1][1],
                    bounds[1][2],
                    bounds[1][3],
                    bounds[1][3],
                    bounds[1][4],
                    bounds[1][4],
                    bounds[1][5],
                    bounds[1][6],
                    bounds[1][6],
                ),
            )
            return p0, bounds

    def predict_split(self, x, state):
        if state == 0:
            return self.gauss_2d(x, *self.props[[0, 2, 3, 4, 6, 8, 9]])
        else:
            return self.gauss_2d(x, *self.props[[1, 2, 3, 5, 7, 8, 10]])

    def gauss_2d(self, cors, A, xo, yo, a, b, theta, G):
        cors = np.atleast_2d(cors)
        x = cors[:, 0]
        y = cors[:, 1]
        xx = (x - xo) * np.cos(theta) - (y - yo) * np.sin(theta)
        yy = (x - xo) * np.sin(theta) + (y - yo) * np.cos(theta)

        res = G + (A) * np.exp(
            -(((xx) ** 2) / (2 * a**2) + ((yy) ** 2) / (2 * b**2))
        )
        return res

    def gauss_2d_split(
        self, cors, Aq, Aa, xo, yo, aq, aa, bq, ba, theta, Gq, Ga
    ):
        sep = self.sep
        # have one state only to predict
        if len(np.atleast_1d(c)) == 1:
            if self.state <= sep:
                # quiet
                y = self.gauss_2d(cors, Aq, xo, yo, aq, bq, theta, Gq)
            else:
                # active
                y = self.gauss_2d(s, Aa, xo, yo, aa, ba, theta, Ga)
            return y

        if not (sep is None):
            quiet = c[:sep]
            active = c[sep:]
            yq = self.gauss_2d(cors, Aq, xo, yo, aq, bq, theta, Gq)
            ya = self.gauss_2d(s, Aa, xo, yo, aa, ba, theta, Ga)
            return np.append(yq, ya)
        else:
            return np.nan

        return res
