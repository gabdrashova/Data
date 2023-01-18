# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:35:05 2022

@author: LABadmin
"""

# define directories # Change to a different file
def define_directories():
    csvDir = "D:\\preprocess.csv"
    s2pDir = "Z:/Suite2Pprocessedfiles/"
    zstackDir = "Z:\\RawData\\"
    metadataDir = "Z:\\RawData\\"

    return csvDir, s2pDir, zstackDir, metadataDir


def create_processing_ops():
    pops = {
        "debug": True,
        "plot": False,
        "f0_percentile": 8,
        "f0_window": 60,
        "zcorrect_mode": "Stack",
        "remove_z_extremes": True,
    }
    return pops
