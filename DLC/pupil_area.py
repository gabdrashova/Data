# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:45:03 2023

@author: maria
"""

# DLC pupil area calculation

import numpy as np
import pandas as pd
import seaborn as sns
import os
import scipy
import matplotlib.pyplot as plt
import Analysis_pipeline.plotting_functions as plotfun
import Analysis_pipeline.analysis_functions as afun
import Analysis_pipeline.behaviour_analysis_functions as bfun
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%testing function

Drive = "Z"
Subfolder = "Suite2Pprocessedfiles" 
animal = "Ladon"
date = "2023-04-04"
experiment = 2
video_number = 0
network = "DLC_resnet101_FaceNov22shuffle1_99000"
csv_path = r""+Drive+":\\Suite2Pprocessedfiles\\"+ animal+"\\"+date+"\\DLC\\Video"+str(video_number)+network+".csv"

#%%
def get_pupil_area(csv_path, save_path):
    """
    Calculates the area of the pupil in each frame of one face video. 
    Based on coordinates given by DeepLabCut.
    Note assumes csv format with the first three rows to be something like
    [0]: DLC_resnet101_FaceNov22shuffle1_99000
    [1]: bodyparts (so the label names such as top,bottom etc)
    [2]: coords
    (The function reorganises the csv to be easier to handle)

    Parameters
    ----------
    csv_path : str
        The path to the csv file for the video.

    Returns
    -------
    Area : np.array[frames]
        The pupil area.

    """
    df = pd.read_csv(csv_path)
    
    # remove unnecessary rows (the network name etc)
    df = df.drop([1], axis=0)
    df.columns = df.iloc[0]
    df = df.drop(0)
    df = df.reset_index(drop=True)
    
    
    # # rename headings to something meaningful (the label names)
    
    df.columns.values[[ 1, 2, 3, 4, 5, 6, 7,8, 9,10,11,12]] = ['top_x', 
    'top_y','top_conf','bottom_x', 'bottom_y', 'bottom_conf', 'left_x', 
    'left_y', 'left_conf', 'right_x', 'right_y', 'right_conf']
    
    
    # Gets columns from df which correspond to pupil points.
    # On vertical axis.
    x_vertical = df.loc[:, ['top_x','bottom_x']]
    y_vertical = df.loc[:,['top_y', 'bottom_y']]
    
    # Gets confidence level to exclude points where network isn't confident of label position.
    confidence_top = df.loc[:,['top_conf']]
    confidence_bottom = df.loc[:,['bottom_conf']]
    exclude_vertical = np.where((np.array(confidence_bottom).astype(float)<0.9
                        ) | (np.array(confidence_top).astype(float)<0.9))[0]
    
    # Calculates the distance between the top and bottom pupil labels using the x and y coordinates.
    distance_all_vertical = np.zeros(df.shape[0])
    for n in range(df.shape[0]):
        distance_vertical = math.sqrt((float(x_vertical.iloc[n,0]) - 
                                       float(x_vertical.iloc[n,1]))**2 + (
            float(y_vertical.iloc[n,0]) - float(y_vertical.iloc[n,1]))**2)
        distance_all_vertical[n] = distance_vertical
    
    
    distance_all_vertical[exclude_vertical]= np.nan
    
    # On horizontal axis  .  
    x_horizontal = df.loc[:, ['left_x','right_x']]
    y_horizontal = df.loc[:,['left_y', 'right_y']]
    
    distance_all_horizontal = np.zeros(df.shape[0])
    
    # Gets confidence level to exclude points where network isn't confident of label position.
    confidence_left = df.loc[:,['left_conf']]
    confidence_right = df.loc[:,['right_conf']]
    exclude_horizontal = np.where((np.array(confidence_left).astype(float)<0.9
                        ) | (np.array(confidence_right).astype(float)<0.9))[0]
    
    for n in range(df.shape[0]):
        distance_horizontal = math.sqrt((float(x_horizontal.iloc[n,0]) -
                                         float(x_horizontal.iloc[n,1]))**2 + (
            float(y_horizontal.iloc[n,0]) - float(y_horizontal.iloc[n,1]))**2)
        distance_all_horizontal[n] = distance_horizontal
    
    distance_all_horizontal[exclude_horizontal]= np.nan
    
    # Calculates based on these coordinates whhat the area is per each frame.
    
    Area = (distance_all_vertical/2)*(distance_all_horizontal/2)*math.pi
    
    # Saves the area as an npy file.
    np.save(save_path, Area)
    
    return Area

#%%
npy_path = ""+Drive+":\\Suite2Pprocessedfiles\\"+ animal+"\\"+date+"\\DLC\\Video"+str(video_number)+"pupil_area.npy"
Atest = get_pupil_area(csv_path)
np.save(npy_path, Atest)