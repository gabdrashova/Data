import numpy as np
from matplotlib import pyplot as plt
import random
import sklearn
import seaborn as sns
import scipy as sp
from matplotlib import rc
import matplotlib.ticker as mtick
import matplotlib as mpl
import pandas as pd
import os
import glob
import pickle
import traceback
from Data.TwoP.runners import *
from Data.Bonsai.extract_data import *
from Data.TwoP.folder_defs import *


# %% load directories and processing ops

# Please change the values in define_directories and create_processing_ops in module folder_defs.
csvDir, s2pDir, zstackDir, metadataDir = define_directories()
pops = create_processing_ops()

# %% read database
# Read database from the preprocess file.
# In the file the values should be Name, Date, Zstack dir number, planes to ignore
# and save directory (if none default is wanted) and process (True,False)
database = pd.read_csv(
    csvDir,
    dtype={
        "Name": str,
        "Date": str,
        "Zstack": str,
        "IgnorePlanes": str,
        "SaveDir": str,
        "Process": bool,
    },
)


# %% run over data base
for i in range(len(database)):
    # This if loop goes through the pandas dataframe called database created above and 
    # if True in the column "Process", the processing continues
    if database.loc[i]["Process"]:
        try:
            print("reading directories")
            (
                s2pDirectory,
                zstackPath,
                metadataDirectory,
                saveDirectory,
            ) = read_csv_produce_directories(
                database.loc[i], s2pDir, zstackDir, metadataDir
            )
            ignorePlanes = np.atleast_1d(
                np.array(database.loc[0]["IgnorePlanes"]).astype(int)
            )
            ops = get_ops_file(s2pDirectory)
            print("getting piezo data")
            planePiezo = get_piezo_data(ops)
            print("processing suite2p data")
            fc = process_s2p_directory(
                s2pDirectory,
                pops,
                planePiezo,
                zstackPath,
                saveDirectory=saveDirectory,
                ignorePlanes=ignorePlanes,
                debug=pops["debug"],
            )
            print("reading bonsai data")
            process_metadata_directory(
                metadataDirectory, ops, pops, saveDirectory
            )
        except Exception:
            print("Could not process due to errors, moving to next batch.")
            print(traceback.format_exc())
    # if False in the column "Process", the processing of those experiments is skipped
    else:
        print("skipping " + str(database.loc[i]))

    # %%
