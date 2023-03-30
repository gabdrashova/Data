# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:05:10 2022

@author: maria
"""
"""
From Carsen's email to Sylvia:
What the code does:
-compute reference image from each plane
-align reference images to each other
-align each frame to each reference image and choose shifts based on best correlation
It returns the best correlation reference image index as ops['zpos_registration'].

There were some changes to the main suite2p code. To use this clone the repository then 

git checkout refactor​
pip install -e .​

Then you'll get this version of suite2p installed. I will add it to the pip though sooner rather than later since there are some outstanding bugs. '
"""

import time, os
import numpy as np
from suite2p.registration import register, rigid, bidiphase
from suite2p.io import tiff_to_binary, BinaryRWFile
from suite2p import io
from suite2p import default_ops
from tifffile import imread
import matplotlib.pyplot as plt
from natsort import natsorted
import imp
from suite2p.registration import utils, rigid
from registration_defs import *
from runners import read_directory_dictionary
from os import path


def run_single_registration(dataEntry):
    """
    This is a function meant to be used in parallel processing that runs the registration function on a single data entry

    Parameters
    ----------
    dataEntry : a data entry given from a database

    Returns
    -------
    None.

    """
    s2pDir = create_global_defs()
    filePath = read_directory_dictionary(dataEntry, s2pDir)
    ops = create_ops_boutton_registration(filePath)
    z_register_one_file(ops)


def z_register_one_file(ops):
    """
    This function uses suite2p features to register an imaging stack dynamically
    by fittng each frame to its best matching plane

    Parameters
    ----------
    ops : an ops file created in the file registration_defs.py

    Returns
    -------
    None.

    """

    # convert tiffs to binaries
    if "save_folder" not in ops or len(ops["save_folder"]) == 0:
        ops["save_folder"] = "suite2p"
    save_folder = os.path.join(ops["save_path0"], ops["save_folder"])
    os.makedirs(save_folder, exist_ok=True)

    ops = tiff_to_binary(ops)

    # get plane folders
    plane_folders = natsorted(
        [
            f.path
            for f in os.scandir(save_folder)
            if f.is_dir() and f.name[:5] == "plane"
        ]
    )
    ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
    nplanes = len(ops_paths)

    # compute reference image
    refImgs = []

    for ipl, ops_path in enumerate(ops_paths):
        if ipl in ops["ignore_flyback"]:
            print(">>>> skipping flyback PLANE", ipl)
            continue

        ops = np.load(ops_path, allow_pickle=True).item()
        align_by_chan2 = ops["functional_chan"] != ops["align_by_chan"]
        raw = ops["keep_movie_raw"]
        reg_file = ops["reg_file"]
        raw_file = ops.get("raw_file", 0) if raw else reg_file
        if ops["nchannels"] > 1:
            reg_file_chan2 = ops["reg_file_chan2"]
            raw_file_chan2 = (
                ops.get("raw_file_chan2", 0) if raw else reg_file_chan2
            )
        else:
            reg_file_chan2 = reg_file
            raw_file_chan2 = reg_file

        align_file = reg_file_chan2 if align_by_chan2 else reg_file
        align_file_raw = raw_file_chan2 if align_by_chan2 else raw_file
        Ly, Lx = ops["Ly"], ops["Lx"]

        # M:this part of the code above just does registration etc (what is done with the GUI usually)
        # grab frames
        with BinaryRWFile(Ly=Ly, Lx=Lx, filename=align_file_raw) as f_align_in:
            n_frames = f_align_in.shape[0]
            frames = f_align_in[
                np.linspace(
                    0,
                    n_frames,
                    1 + np.minimum(ops["nimg_init"], n_frames),
                    dtype=int,
                )[:-1]
            ]

        # M: this is done to adjust bidirectional shift occuring due to line scanning
        # compute bidiphase shift
        if (
            ops["do_bidiphase"]
            and ops["bidiphase"] == 0
            and not ops["bidi_corrected"]
        ):
            bidiphase = bidiphase.compute(frames)
            print(
                "NOTE: estimated bidiphase offset from data: %d pixels"
                % bidiphase
            )
            ops["bidiphase"] = bidiphase
            # shift frames
            if bidiphase != 0:
                bidiphase.shift(frames, int(ops["bidiphase"]))
        else:
            bidiphase = 0

        # compute reference image
        refImgs.append(register.compute_reference(frames))

    # align reference frames to each other
    frames = np.array(refImgs).copy()
    for frame in frames:
        rmin, rmax = np.int16(np.percentile(frame, 1)), np.int16(
            np.percentile(frame, 99)
        )
        frame[:] = np.clip(frame, rmin, rmax)

    refImg = frames.mean(axis=0)
    # M: the below section is just the usual xy registration
    niter = 8
    for iter in range(0, niter):
        # rigid registration
        ymax, xmax, cmax = rigid.phasecorr(
            data=rigid.apply_masks(
                frames,
                *rigid.compute_masks(
                    refImg=refImg,
                    maskSlope=ops["spatial_taper"]
                    if ops["1Preg"]
                    else 3 * ops["smooth_sigma"],
                )
            ),
            cfRefImg=rigid.phasecorr_reference(
                refImg=refImg, smooth_sigma=ops["smooth_sigma"]
            ),
            maxregshift=ops["maxregshift"],
            smooth_sigma_time=ops["smooth_sigma_time"],
        )
        dys = np.zeros(len(frames), "int")
        dxs = np.zeros(len(frames), "int")
        for i, (frame, dy, dx) in enumerate(zip(frames, ymax, xmax)):
            frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)
            dys[i] = dy
            dxs[i] = dx

    print("shifts of reference images: (y,x) = ", dys, dxs)

    # frames = smooth_reference_stack(frames, ops)

    refImgs = list(frames)

    # register and choose the best plane match at each time point,
    # in accordance with the reference image of each plane

    imp.reload(utils)
    imp.reload(rigid)
    imp.reload(register)

    ops["refImg"] = refImgs

    # Get the correlation between the reference images
    corrs_all = get_reference_correlation(frames, ops)
    smooth_images_by_correlation(ops_paths, corrs_all)

    for ipl, ops_path in enumerate(ops_paths):
        if ipl in ops["ignore_flyback"]:
            print(">>>> skipping flyback PLANE", ipl)
            continue
        else:
            print(">>>> registering PLANE", ipl)
        ops = np.load(ops_path, allow_pickle=True).item()
        ops = register.register_binary(ops, refImg=refImgs)
        np.save(ops["ops_path"], ops)

        # At this point the files are registered properly according to where they are
        # now we need to go over each zposition and replace the frame on the channel with a weighted
        # frame on the plane it actually is
        replace_frames_by_zpos(ops, ops_paths, ipl)


def replace_frames_by_zpos(ops, ops_paths, plane):

    reg_file = ops["reg_file"]
    zpos = ops["zpos_registration"]
    batch_size = ops["batch_size"]
    possible_z = np.unique(zpos)

    with BinaryRWFile(
        Ly=ops["Ly"], Lx=ops["Lx"], filename=reg_file
    ) as f_align_in:
        n_frames = f_align_in.shape[0]
        # go through frames and replace places where a jump occurred from the weighted frame from the right plane
        for b in range(0, n_frames, batch_size):
            zpos_t = zpos[b : min(b + batch_size, n_frames)]
            frames = f_align_in[b : min(b + batch_size, n_frames)]
            changeInds = np.where(zpos_t != plane)[0]
            if len(changeInds) > 0:
                uniqueZ = np.unique(zpos_t[changeInds])
                for z in uniqueZ:
                    if z in ops["ignore_flyback"]:
                        continue
                    ops_alt1 = np.load(ops_paths[z], allow_pickle=True).item()
                    reg_file_alt = ops_alt1["reg_file"]
                    with BinaryRWFile(
                        Ly=ops["Ly"], Lx=ops["Lx"], filename=reg_file_alt
                    ) as f_alt:
                        frames_alt = f_alt[b : min(b + batch_size, n_frames)]
                        frames[changeInds] = frames_alt[changeInds]
            f_align_in[b : min(b + batch_size, n_frames)] = frames


def get_reference_correlation(refImgs, ops):
    """
    returns the correlation of each plane with the neighbouring planes as a weight

    Parameters
    ----------
    refImgs : the stack of refImgs
    ops: the ops file

    Returns
    -------
    corrs: average correlations of each plane with the others.

    """
    nZ = refImgs.shape[0]
    corrs_all = []

    for z in range(nZ):
        refImg = refImgs[z, :, :]
        ymax, xmax, cmax = rigid.phasecorr(
            data=rigid.apply_masks(
                refImgs,
                *rigid.compute_masks(
                    refImg=refImg,
                    maskSlope=ops["spatial_taper"]
                    if ops["1Preg"]
                    else 3 * ops["smooth_sigma"],
                )
            ),
            cfRefImg=rigid.phasecorr_reference(
                refImg=refImg, smooth_sigma=ops["smooth_sigma"]
            ),
            maxregshift=ops["maxregshift"],
            smooth_sigma_time=ops["smooth_sigma_time"],
        )

        corrs = cmax[np.max([0, z - 1]) : np.min([nZ, z + 2])]
        corrs /= np.sum(corrs)
        corrs_all.append(corrs)
    return corrs_all


def smooth_images_by_correlation(ops_paths, corrs_all):
    for ipl, ops_path in enumerate(ops_paths):
        corrs = corrs_all[ipl]
        ops = np.load(ops_paths[ipl], allow_pickle=True).item()
        reg_file = ops["reg_file"]
        batch_size = ops["batch_size"]
        with BinaryRWFile(
            Ly=ops["Ly"], Lx=ops["Lx"], filename=reg_file
        ) as f_main:
            n_frames, Ly, Lx = f_main.shape

            # do edges separately
            if ipl == 0:
                ops_alt1 = np.load(
                    ops_paths[ipl + 1], allow_pickle=True
                ).item()
                with BinaryRWFile(
                    Ly=ops_alt1["Ly"],
                    Lx=ops_alt1["Lx"],
                    filename=ops_alt1["reg_file"],
                ) as f_plus:
                    for b in range(0, n_frames, batch_size):
                        f_main[b : min(b + batch_size, n_frames)] = (
                            f_main[b : min(b + batch_size, n_frames)]
                            * corrs[0]
                            + f_plus[b : min(b + batch_size, n_frames)]
                            * corrs[1]
                        )
            elif ipl == len(ops_paths):
                ops_alt1 = np.load(
                    ops_paths[ipl - 1], allow_pickle=True
                ).item()
                with BinaryRWFile(
                    Ly=ops_alt1["Ly"],
                    Lx=ops_alt1["Lx"],
                    filename=ops_alt1["reg_file"],
                ) as f_minus:
                    for b in range(0, n_frames, batch_size):
                        f_main[b : min(b + batch_size, n_frames)] = (
                            f_main[b : min(b + batch_size, n_frames)]
                            * corrs[1]
                            + f_minus[b : min(b + batch_size, n_frames)]
                            * corrs[0]
                        )
            else:
                ops_alt1 = np.load(
                    ops_paths[ipl - 1], allow_pickle=True
                ).item()
                ops_alt2 = np.load(
                    ops_paths[ipl + 1], allow_pickle=True
                ).item()
                with BinaryRWFile(
                    Ly=ops_alt1["Ly"],
                    Lx=ops_alt1["Lx"],
                    filename=ops_alt1["reg_file"],
                ) as f_minus:
                    with BinaryRWFile(
                        Ly=ops_alt2["Ly"],
                        Lx=ops_alt2["Lx"],
                        filename=ops_alt2["reg_file"],
                    ) as f_plus:
                        for b in range(0, n_frames, batch_size):
                            f_main[b : min(b + batch_size, n_frames)] = (
                                f_main[b : min(b + batch_size, n_frames)]
                                * corrs[1]
                                + f_minus[b : min(b + batch_size, n_frames)]
                                * corrs[0]
                                + f_plus[b : min(b + batch_size, n_frames)]
                                * corrs[2]
                            )
