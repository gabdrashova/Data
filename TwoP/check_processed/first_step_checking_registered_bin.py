# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:43:55 2023

@author: maria
"""
import numpy as np
import os
Drive = "Z"
Subfolder = "ProcessedData" 
#Subfolder = "Suite2Pprocessedfiles" 
animal = "Lotho"
date = "2023-04-12"

Directory = ""+Drive+":\\"+Subfolder+"\\"+ animal+"\\"+date+"\\suite2p\\combined\\"
ops = np.load(os.path.join(Directory, "ops.npy"), allow_pickle = True)
ops = ops.item()
frames = ops["frames_per_folder"]
all_exp = np.zeros((frames.shape[0],1))
for n in range(frames.shape[0]):
    length = np.sum(frames[0:n+1])
    all_exp[n] = length

print("exp lengths: " + "\n" + str(all_exp))

experiment_num = 2

# Initialize the first range as "0-100"
ranges = ["0-100"]

# Calculate the ranges based on the input array
for num in all_exp:
    num = int(num)
    range_str_end = f"{num - 100}-{num}"
    range_str_start = f"{num}-{num+100}"
    range_tuple = (range_str_end, range_str_start)
    ranges.append(range_tuple)

# Iterate through and print the derived ranges
for i, range_str in enumerate(ranges):
    print(f" {i}: {range_str}")

# creating folders where the average tiffs will be saved, calling it AVG_tiffs
root_directory = ""+Drive+":\\"+Subfolder+"\\"+ animal+"\\"+date+"\\suite2p\\"

for dirpath, dirnames, filenames in os.walk(root_directory):
    for dirname in dirnames:
        # Create the "AVG_tiffs" folder within each subdirectory
        avg_tiffs_folder = os.path.join(dirpath, dirname, "AVG_tiffs")
        os.makedirs(avg_tiffs_folder, exist_ok=True)

print("AVG_tiffs folders created in all subdirectories.")