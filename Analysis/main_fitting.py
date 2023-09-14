# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:06:33 2023

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

from alignment_functions import get_calcium_aligned
from support_functions import (
    load_grating_data,
    get_trial_classification_running,
    run_complete_analysis,
    get_directory_from_session,
)
from user_defs import directories_to_fit, create_fitting_ops
import traceback
from abc import ABC, abstractmethod
import inspect

#%%
sessions = directories_to_fit()

ops = create_fitting_ops()

saveDir = ops["save_dir"]
for currSession in sessions:

    print(f"starting to run session: {currSession}")
    # get data for current session
    di = get_directory_from_session("Z:\\ProcessedData\\", currSession)
    data = load_grating_data(di)

    # make save dir for later
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    print("getting aligned signal")
    gratingRes, ts = get_calcium_aligned(
        data["sig"],
        data["calTs"].reshape(-1, 1),
        data["gratingsSt"],
        np.array([-1, 4]).reshape(-1, 1).T,
        data["planes"].reshape(-1, 1),
        data["planeDelays"].reshape(1, -1),
    )

    print("getting trial classification")
    quietI, activeI = get_trial_classification_running(
        data["wheelVelocity"],
        data["wheelTs"],
        data["gratingsSt"],
        data["gratingsEt"],
        activeQuantile=ops["active_quantile"],
        quietQuantile=ops["quiet_quantile"],
    )

    respP = np.zeros(gratingRes.shape[-1])
    respDirection = np.zeros(gratingRes.shape[-1])

    paramsOri = np.zeros((5, gratingRes.shape[-1]))
    paramsOriSplit = np.zeros((7, gratingRes.shape[-1]))
    varsOri = np.zeros((3, gratingRes.shape[-1]))
    pvalOri = np.zeros(gratingRes.shape[-1])
    TunersOri = np.empty((2, gratingRes.shape[-1]), dtype=object)

    paramsTf = np.zeros((4, gratingRes.shape[-1]))
    paramsTfSplit = np.zeros((8, gratingRes.shape[-1]))
    varsTf = np.zeros((3, gratingRes.shape[-1]))
    pvalTf = np.zeros(gratingRes.shape[-1])
    TunersTf = np.empty((2, gratingRes.shape[-1]), dtype=object)

    paramsSf = np.zeros((4, gratingRes.shape[-1]))
    paramsSfSplit = np.zeros((8, gratingRes.shape[-1]))
    varsSf = np.zeros((3, gratingRes.shape[-1]))
    pvalSf = np.zeros(gratingRes.shape[-1])
    TunersSf = np.empty((2, gratingRes.shape[-1]), dtype=object)

    paramsCon = np.zeros((4, gratingRes.shape[-1]))
    paramsConSplit = np.zeros((6, gratingRes.shape[-1]))
    varsCon = np.zeros((3, gratingRes.shape[-1]))
    pvalCon = np.zeros(gratingRes.shape[-1])
    TunersCon = np.empty((2, gratingRes.shape[-1]), dtype=object)

    fittingRange = range(0, gratingRes.shape[-1])
    # check if want to run only some neurons
    if len(currSession["SpecificNeurons"]) > 0:
        fittingRange = currSession["SpecificNeurons"]
        # assume to wants to redo only those, so try reloading existing data first
        try:
            respP = np.load(os.path.join(saveDir, "resp.pval.npy"))
            respDirection = np.load(
                os.path.join(saveDir, "resp.direction.npy")
            )

            paramsOri = np.load(os.path.join(saveDir, "fit.ori.params.npy"))
            paramsOriSplit = np.load(
                os.path.join(saveDir, "fit.ori.split.params.npy")
            )
            varsOri = np.load(os.path.join(saveDir, "fit.ori.vars.npy"))
            pvalOri = np.load(os.path.join(saveDir, "fit.ori.pval.npy"))

            paramsTf = np.load(os.path.join(saveDir, "fit.tf.params.npy"))
            paramsTfSplit = np.load(
                os.path.join(saveDir, "fit.tf.split.params.npy")
            )
            varsTf = np.load(os.path.join(saveDir, "fit.tf.vars.npy"))
            pvalTf = np.load(os.path.join(saveDir, "fit.tf.pval.npy"))

            paramsSf = np.load(os.path.join(saveDir, "fit.sf.params.npy"))
            paramsSfSplit = np.load(
                os.path.join(saveDir, "fit.sf.split.params.npy")
            )
            varsSf = np.load(os.path.join(saveDir, "fit.sf.vars.npy"))
            pvalSf = np.load(os.path.join(saveDir, "fit.sf.pval.npy"))

            paramsCon = np.load(os.path.join(saveDir, "fit.con.params.npy"))
            paramsConSplit = np.load(
                os.path.join(saveDir, "fit.con.split.params.npy")
            )
            varsCon = np.load(os.path.join(saveDir, "fit.con.vars.npy"))
            pvalCon = np.load(os.path.join(saveDir, "fit.con.pval.npy"))
        except:
            pass

    for n in fittingRange:

        sig, res_ori, res_freq, res_spatial, res_con = run_complete_analysis(
            gratingRes, data, ts, quietI, activeI, n
        )

        respP[n] = sig[0]
        respDirection[n] = sig[1]

        paramsOri[:, n] = res_ori[0]
        paramsOriSplit[:, n] = res_ori[1]
        varsOri[:, n] = res_ori[2:5]
        pvalOri[n] = res_ori[6]
        TunersOri[:, n] = res_ori[7:]

        paramsTf[:, n] = res_freq[0]
        paramsTfSplit[:, n] = res_freq[1]
        varsTf[:, n] = res_freq[2:5]
        pvalTf[n] = res_freq[6]
        TunersTf[:, n] = res_freq[7:]

        paramsSf[:, n] = res_spatial[0]
        paramsSfSplit[:, n] = res_spatial[1]
        varsSf[:, n] = res_spatial[2:5]
        pvalSf[n] = res_spatial[6]
        TunersSf[:, n] = res_spatial[7:]

        paramsCon[:, n] = res_con[0]
        paramsConSplit[:, n] = res_con[1]
        varsCon[:, n] = res_con[2:5]
        pvalCon[n] = res_con[6]
        TunersCon[:, n] = res_con[7:]

    np.save(os.path.join(saveDir, "resp.pval.npy"), respP)
    np.save(os.path.join(saveDir, "resp.direction.npy"), respDirection)

    np.save(os.path.join(saveDir, "fit.ori.params.npy"), paramsOri)
    np.save(os.path.join(saveDir, "fit.ori.split.params.npy"), paramsOriSplit)
    np.save(os.path.join(saveDir, "fit.ori.vars.npy"), varsOri)
    np.save(os.path.join(saveDir, "fit.ori.pval.npy"), pvalOri)

    np.save(os.path.join(saveDir, "fit.tf.params.npy"), paramsTf)
    np.save(os.path.join(saveDir, "fit.tf.split.params.npy"), paramsTfSplit)
    np.save(os.path.join(saveDir, "fit.tf.vars.npy"), varsTf)
    np.save(os.path.join(saveDir, "fit.tf.pval.npy"), pvalTf)

    np.save(os.path.join(saveDir, "fit.sf.params.npy"), paramsSf)
    np.save(os.path.join(saveDir, "fit.sf.split.params.npy"), paramsSfSplit)
    np.save(os.path.join(saveDir, "fit.sf.vars.npy"), varsSf)
    np.save(os.path.join(saveDir, "fit.sf.pval.npy"), pvalSf)

    np.save(os.path.join(saveDir, "fit.con.params.npy"), paramsCon)
    np.save(os.path.join(saveDir, "fit.con.split.params.npy"), paramsConSplit)
    np.save(os.path.join(saveDir, "fit.con.vars.npy"), varsCon)
    np.save(os.path.join(saveDir, "fit.con.pval.npy"), pvalCon)

#%%
