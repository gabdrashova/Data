# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:36:28 2023

@author: LABadmin
"""
from suite2p import default_ops
import pandas as pd


######### Ops settings for the boutton registration script


def create_global_defs():
    s2pDir = "Z:\\RawData\\"
    return s2pDir


def create_ops_boutton_registration(filePath):
    ops = default_ops()
    ops["data_path"] = filePath[1:]

    ops["look_one_level_down"] = False
    ops["ignore_flyback"] = [0]
    ops["nchannels"] = 1
    ops["nplanes"] = 8
    ops["functional_chan"] = 1

    # registration ops
    ops["keep_movie_raw"] = True
    ops["align_by_chan"] = 1

    ops["block_size"] = [64, 256]
    # run for only X number frames
    # ops['frames_include'] = 1000

    # set save folder
    ops["save_path0"] = filePath[0]

    # localised optioed
    ops["delete_extra_frames"] = False

    return ops


def directories_to_register():
    dirDefs = [
        {
            "Name": "Io",
            "Date": "2023-02-01",
            "Experiments": [1, 2, 3, 4, 5],
        },
    ]
    return pd.DataFrame(dirDefs)
