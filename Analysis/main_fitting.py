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

# note: probably need to set your path if this gives a ModuleNotFoundError
from alignment_functions import get_calcium_aligned
from support_functions import (
    load_grating_data,
    get_trial_classification_running,
    run_complete_analysis,
    get_directory_from_session,
)
from plotting_functions import print_fitting_data
from user_defs import directories_to_fit, create_fitting_ops
import traceback
from abc import ABC, abstractmethod
import inspect

#%%
# Note: go to user_defs and change the inputs to directories_to_fit() and create_fitting_ops().
sessions = directories_to_fit()

ops = create_fitting_ops()
# Loads the save directory from the fitting_ops in user_defs.
saveDir = ops["save_dir"]
for currSession in sessions:

    print(f"starting to run session: {currSession}")
    # Gets data for current session from the Preprocessed folder.
    di = get_directory_from_session("Z:\\ProcessedData\\", currSession)
    # Creates a dictionary with all the output from main_preprocess. 
    data = load_grating_data(di)

    # Makes save dir for later.
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

    paramsOri = np.zeros((gratingRes.shape[-1],5))
    # paramsOriSplit = np.zeros((7, gratingRes.shape[-1]))
    paramsOriSplit = np.zeros((gratingRes.shape[-1],5,2))
    # varsOri = np.zeros((3, gratingRes.shape[-1]))
    varOriConst = np.zeros(gratingRes.shape[-1])
    varOriOne = np.zeros(gratingRes.shape[-1])
    varOriSplit = np.zeros(gratingRes.shape[-1])
    pvalOri = np.zeros(gratingRes.shape[-1])
    TunersOri = np.empty((gratingRes.shape[-1],2), dtype=object)

    paramsTf = np.zeros((gratingRes.shape[-1],4))
    # paramsTfSplit = np.zeros((8, gratingRes.shape[-1]))
    paramsTfSplit = np.zeros((gratingRes.shape[-1]4,2))
    # varsTf = np.zeros((3, gratingRes.shape[-1]))
    varTfConst = np.zeros(gratingRes.shape[-1])
    varTfOne = np.zeros(gratingRes.shape[-1])
    varTfSplit = np.zeros(gratingRes.shape[-1])
    pvalTf = np.zeros(gratingRes.shape[-1])
    TunersTf = np.empty((gratingRes.shape[-1],2), dtype=object)

    paramsSf = np.zeros((gratingRes.shape[-1],4))
    # paramsSfSplit = np.zeros((8, gratingRes.shape[-1]))
    paramsSfSplit = np.zeros((gratingRes.shape[-1],4,2))
    # varsSf = np.zeros((3, gratingRes.shape[-1]))
    varSfConst = np.zeros(gratingRes.shape[-1])
    varSfOne = np.zeros(gratingRes.shape[-1])
    varSfSplit = np.zeros(gratingRes.shape[-1])
    pvalSf = np.zeros(gratingRes.shape[-1])
    TunersSf = np.empty((gratingRes.shape[-1],2), dtype=object)

    paramsCon = np.zeros((gratingRes.shape[-1],4))
    # paramsConSplit = np.zeros((6, gratingRes.shape[-1]))
    paramsConSplit = np.zeros((gratingRes.shape[-1],4,2))
    # varsCon = np.zeros((3, gratingRes.shape[-1]))
    varConConst = np.zeros(gratingRes.shape[-1])
    varConOne = np.zeros(gratingRes.shape[-1])
    varConSplit = np.zeros(gratingRes.shape[-1])
    pvalCon = np.zeros(gratingRes.shape[-1])
    TunersCon = np.empty((gratingRes.shape[-1],2), dtype=object)

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
            gratingRes, data, ts, quietI, activeI, n,ops["fitOri"],ops["fitTf"],ops["fitSf"],ops["fitContrast"]
        )

        respP[n] = sig[0]
        respDirection[n] = sig[1]

        paramsOri[ n,:] = res_ori[0]        
        paramsOriSplit[n,:,0] = res_ori[1][[0, 2, 4, 5, 6]]
        paramsOriSplit[n,:,1] = res_ori[1][[1, 3, 4, 5, 6]]
        # varsOri[:, n] = res_ori[2:5]
        varOriConst[n] = res_ori[2]
        varOriOne[n] = res_ori[3]
        varOriSplit[n] = res_ori[4]
        pvalOri[n] = res_ori[6]
        TunersOri[n,:] = res_ori[7:]

        paramsTf[n,:] = res_freq[0]        
        paramsTfSplit[n,:,0] = res_freq[1][::2]
        paramsTfSplit[n,:,1] = res_freq[1][1::2]
        # varsTf[:, n] = res_freq[2:5]
        varTfConst[n] = res_freq[2]
        varTfOne[n] = res_freq[3]
        varTfSplit[n] = res_freq[4]
        pvalTf[n] = res_freq[6]
        TunersTf[n,:] = res_freq[7:]

        paramsSf[n,:] = res_spatial[0]        
        paramsSfSplit[n,:,0] = res_spatial[1][::2]
        paramsSfSplit[n,:,1] = res_spatial[1][1::2]
        # varsSf[:, n] = res_spatial[2:5]
        varSfConst[n] = res_spatial[2]
        varSfOne[n] = res_spatial[3]
        varSfSplit[n] = res_spatial[4]
        pvalSf[n] = res_spatial[6]
        TunersSf[n,:] = res_spatial[7:]

        paramsCon[n,:] = res_con[0]        
        paramsConSplit[n,:,0] = res_con[1][[0, 2, 4, 5]]
        paramsConSplit[n,:,1] = res_con[1][[1, 3, 4, 5]]
        # varsCon[:, n] = res_con[2:5]
        varConConst[n] = res_con[2]
        varConOne[n] = res_con[3]
        varConSplit[n] = res_con[4]
        pvalCon[n] = res_con[6]
        TunersCon[n,:] = res_con[7:]
        
        


    np.save(os.path.join(saveDir, "gratingResp.pVal.npy"), respP)
    np.save(os.path.join(saveDir, "gratingResp.direction.npy"), respDirection)
    
    if (ops["fitOri"]):
        np.save(os.path.join(saveDir, "gratingOriTuning.params.params.npy"), paramsOri)
        np.save(os.path.join(saveDir, "gratingOriTuning.paramsRunning.npy"), paramsOriSplit)    
        np.save(os.path.join(saveDir, "gratingOriTuning.expVar.constant.npy"), varOriConst)
        np.save(os.path.join(saveDir, "gratingOriTuning.expVar.noSplit.npy"), varOriOne)
        np.save(os.path.join(saveDir, "gratingOriTuning.expVar.runningSplit.npy"), varOriSplit)
        np.save(os.path.join(saveDir, "gratingOriTuning.pVal.runningSplit.npy"), pvalOri)
    
    if (ops["fitTf"]):
        np.save(os.path.join(saveDir, "gratingTfTuning.params.params.npy"), paramsTf)
        np.save(os.path.join(saveDir, "gratingTfTuning.paramsRunning.npy"), paramsTfSplit)    
        np.save(os.path.join(saveDir, "gratingTfTuning.expVar.constant.npy"), varTfConst)
        np.save(os.path.join(saveDir, "gratingTfTuning.expVar.noSplit.npy"), varTfOne)
        np.save(os.path.join(saveDir, "gratingTfTuning.expVar.runningSplit.npy"), varTfSplit)
        np.save(os.path.join(saveDir, "gratingTfTuning.pVal.runningSplit.npy"), pvalTf)  
    
    if (ops["fitSf"]):
        np.save(os.path.join(saveDir, "gratingSfTuning.params.params.npy"), paramsSf)
        np.save(os.path.join(saveDir, "gratingSfTuning.paramsRunning.npy"), paramsSfSplit)    
        np.save(os.path.join(saveDir, "gratingSfTuning.expVar.constant.npy"), varSfConst)
        np.save(os.path.join(saveDir, "gratingSfTuning.expVar.noSplit.npy"), varSfOne)
        np.save(os.path.join(saveDir, "gratingSfTuning.expVar.runningSplit.npy"), varSfSplit)
        np.save(os.path.join(saveDir, "gratingSfTuning.pVal.runningSplit.npy"), pvalSf)  
    
    if (ops["fitContrast"]):
        np.save(os.path.join(saveDir, "gratingContrastTunin.params.params.npy"), paramsCon)
        np.save(os.path.join(saveDir, "gratingContrastTunin.paramsRunning.npy"), paramsConSplit)    
        np.save(os.path.join(saveDir, "gratingContrastTunin.expVar.constant.npy"), varConConst)
        np.save(os.path.join(saveDir, "gratingContrastTunin.expVar.noSplit.npy"), varConOne)
        np.save(os.path.join(saveDir, "gratingContrastTunin.expVar.runningSplit.npy"), varConSplit)
        np.save(os.path.join(saveDir, "gratingContrastTunin.pVal.runningSplit.npy"), pvalCon)
    

#%%plotting

    
for n in fittingRange:
         try:
     
             print_fitting_data(gratingRes,ts, quietI, activeI, data, paramsOri, 
         paramsOriSplit, varsOri, pvalOri, paramsTf, paramsTfSplit, varsTf, 
         pvalTf, paramsSf, paramsSfSplit, varsSf, pvalSf, n, respP,None, saveDir)
             plt.close()
         except Exception:
             print("fail "+ str(n))
    
 
