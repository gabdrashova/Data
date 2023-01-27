# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:39:57 2022

@author: LABadmin
"""

"""Runner functions"""

from suite2p.registration.zalign import compute_zpos
from joblib import Parallel, delayed
import numpy as np
import time
import traceback
import io
import os
import skimage.io
import glob
import pickle
import scipy as sp
import warnings

from Data.TwoP.process_tiff import *
from Data.TwoP.preprocess_traces import *
from Data.Bonsai.extract_data import *
from Data.TwoP.general import *
from Data.TwoP.folder_defs import create_processing_ops


def _process_s2p_singlePlane(
    pops, planeDirs, zstackPath, saveDirectory, piezo, plane
):
    currDir = planeDirs[plane]

    F = np.load(os.path.join(currDir, "F.npy"), allow_pickle=True).T
    N = np.load(os.path.join(currDir, "Fneu.npy")).T
    isCell = np.load(os.path.join(currDir, "iscell.npy")).T
    stat = np.load(os.path.join(currDir, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(currDir, "ops.npy"), allow_pickle=True).item()
    processing_metadata = {}

    fs = ops["fs"]
    F = F[:, isCell[0, :].astype(bool)]
    N = N[:, isCell[0, :].astype(bool)]
    stat = stat[isCell[0, :].astype(bool)]

    cellLocs = np.zeros((len(stat), 3))
    ySpan = ops["refImg"].shape[1]

    F = zero_signal(F)
    N = zero_signal(N)

    # Get cell locations
    for i, s in enumerate(stat):
        relYpos = s["med"][1] / ySpan
        piezoInd = int(np.round((len(piezo) - 1) * relYpos))
        zPos = piezo[piezoInd]
        cellLocs[i, :] = np.append(s["med"], zPos)

    # FCORR stuff
    Fc, regPars, F_binValues, N_binValues = correct_neuropil(F, N, fs)
    F0 = get_F0(
        Fc, fs, prctl_F=pops["f0_percentile"], window_size=pops["f0_window"]
    )
    dF = get_delta_F_over_F(Fc, F0)

    zprofiles = None
    zTrace = None
    # hack to avoid random reg directories
    ops["reg_file"] = os.path.join(currDir, "data.bin")
    ops["ops_path"] = os.path.join(currDir, "ops.npy")
    if not (zstackPath is None):
        try:
            refImg = ops["refImg"]
            zFileName = os.path.join(
                saveDirectory, "zstackAngle_plane" + str(plane) + ".tif"
            )
            if not (os.path.exists(zFileName)):
                zstack = register_zstack(
                    zstackPath, spacing=1, piezo=piezo, target_image=refImg
                )
                skimage.io.imsave(zFileName, zstack)
                _, zcorr = compute_zpos(zstack, ops)
            elif not ("zcorr" in ops.keys()):
                zstack = skimage.io.imread(zFileName)

                ops, zcorr = compute_zpos(zstack, ops)
                np.save(ops["ops_path"], ops)
            else:
                zstack = skimage.io.imread(zFileName)
                zcorr = ops["zcorr"]
            zTrace = np.argmax(zcorr, 0)
            zprofiles = extract_zprofiles(
                currDir,
                zstack,
                neuropil_correction=regPars[1, :],
                metadata=processing_metadata,
                smooting_factor=2,
            )

            Fcz = correct_zmotion(
                dF,
                zprofiles,
                zTrace,
                ignore_faults=pops["remove_z_extremes"],
                metadata=pops,
            )
        except:
            print(currDir + ": Error in correcting z-motion")
            print(traceback.format_exc())
            Fcz = dF
    else:
        Fcz = dF
    results = {
        "dff": dF,
        "dff_zcorr": Fcz,
        "zProfiles": zprofiles,
        "zTrace": zTrace,
        "locs": cellLocs,
    }

    if pops["plot"]:
        for i in range(dF.shape[-1]):
            # Print full
            plotArrangement = [
                ["profile", "f"],
                ["profile", "corr"],
                ["profile", "zcorr"],
                ["profile", "trace"],
            ]
            f, ax = plt.subplot_mosaic(plotArrangement)
            ax["f"].plot(F[:, i], "b")
            ax["f"].plot(N[:, i], "r")
            ax["f"].legend(
                ["Fluorescence", "Neuropil"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["corr"].plot(Fc[:, i], "k")
            ax["corr"].plot(F0[:, i], "b", linewidth=4, zorder=10)
            ax["corr"].legend(
                ["Corrected F", "F0"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )

            ax["zcorr"].plot(Fcz[:, i], "k")
            ax["zcorr"].plot(dF[:, i], "b--", linewidth=3)
            ax["zcorr"].legend(
                ["dF/F", "dF/F z-zcorrected"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["zcorr"].set_xlabel("time (frames)")
            if not zTrace is None:
                ax["trace"].plot(zTrace)
                ax["trace"].legend(
                    ["Z trace"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
            if not zprofiles is None:
                ax["profile"].plot(zprofiles[:, i], range(zprofiles.shape[0]))
                ax["profile"].legend(
                    ["Z profile"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
                ax["profile"].set_xlabel("fluorescence")
                ax["profile"].set_xlabel("depth")

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            plt.savefig(
                os.path.join(
                    saveDirectory,
                    "Plane" + str(plane) + "Neuron" + str(i) + ".png",
                ),
                format="png",
            )

            with open(
                os.path.join(
                    saveDirectory,
                    "Plane" + str(plane) + "Neuron" + str(i) + ".fig.pickle",
                ),
                "wb",
            ) as file:
                pickle.dump(f, file)

            # Print Part
            f, ax = plt.subplot_mosaic(plotArrangement)
            ax["f"].plot(F[1:500, i], "b")
            ax["f"].plot(N[1:500, i], "r")
            ax["f"].legend(
                ["Fluorescence", "Neuropil"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["corr"].plot(Fc[1:500, i], "k")
            ax["corr"].plot(F0[1:500, i], "b", linewidth=4)
            ax["corr"].legend(
                ["Corrected F", "F0"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )

            ax["zcorr"].plot(Fcz[1:500, i], "k")
            ax["zcorr"].plot(dF[1:500, i], "b--", linewidth=3)
            ax["zcorr"].legend(
                ["dF/F", "dF/F z-zcorrected"],
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
            )
            ax["zcorr"].set_xlabel("time (frames)")
            if not zTrace is None:
                ax["trace"].plot(zTrace[1:500])
                ax["trace"].legend(
                    ["Z trace"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
            if not zprofiles is None:
                ax["profile"].plot(zprofiles[:, i], range(zprofiles.shape[0]))
                ax["profile"].legend(
                    ["Z profile"], bbox_to_anchor=(1.01, 1), loc="upper left"
                )
                ax["profile"].set_xlabel("fluorescence")
                ax["profile"].set_xlabel("depth")

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            plt.savefig(
                os.path.join(
                    saveDirectory,
                    "Plane" + str(plane) + "Neuron" + str(i) + "_zoom.png",
                ),
                format="png",
            )

            with open(
                os.path.join(
                    saveDirectory,
                    "Plane"
                    + str(plane)
                    + "Neuron"
                    + str(i)
                    + "_zoom.fig.pickle",
                ),
                "wb",
            ) as file:
                pickle.dump(f, file)

            plt.close("all")
    return results


def process_s2p_directory(
    suite2pDirectory,
    pops=create_processing_ops(),
    piezoTraces=None,
    zstackPath=None,
    saveDirectory=None,
    ignorePlanes=None,
    debug=False,
):
    """
    This function runs over a suite2p directory and pre-processes the data in each plane
    the pre processing includes:
        neuropil correction
        z-trace extraction and correction according to profile
        at the function saves all the traces together

    Parameters
    ----------
    suite2pDirectory : TYPE
        the suite2p parent directory, where the plane directories are.
    piezoTraces : [time X plane] um
        a metadata directory for the piezo trace.
    zstackPath : TYPE
        the path of the acquired z-stack.
    saveDirectory : TYPE, optional
        the directory where the processed data will be saved. If None will add a ProcessedData directory to the suite2pdir. The default is None.

    Returns
    -------
    None.

    """

    if saveDirectory is None:
        saveDirectory = os.path.join(suite2pDirectory, "ProcessedData")
    if not os.path.isdir(saveDirectory):
        os.makedirs(saveDirectory)
    planeDirs = glob.glob(os.path.join(suite2pDirectory, "plane*"))
    combinedDir = glob.glob(os.path.join(suite2pDirectory, "combined*"))

    ops = np.load(
        os.path.join(combinedDir[0], "ops.npy"), allow_pickle=True
    ).item()
    numPlanes = ops["nplanes"]

    planeRange = np.arange(numPlanes)
    if not (ignorePlanes is None):
        ignorePlanes = np.intersect1d(planeRange, ignorePlanes)
        planeRange = np.delete(planeRange, ignorePlanes)
    preTime = time.time()
    # TODO: extract planes
    if not debug:
        jobnum = 4
    else:
        jobnum = 1
    results = Parallel(n_jobs=jobnum, verbose=5)(
        delayed(_process_s2p_singlePlane)(
            pops, planeDirs, zstackPath, saveDirectory, piezoTraces[:, p], p
        )
        for p in planeRange
    )
    # signalList = _process_s2p_singlePlane(planeDirs,zstackPath,saveDirectory,piezoTraces[:,0],1)
    postTime = time.time()
    print("Processing took: " + str(postTime - preTime) + " ms")
    planes = np.array([])

    signalList = []
    signalLocs = []
    zTraces = []
    zProfiles = []
    for i in range(len(results)):
        signalList.append(results[i]["dff_zcorr"])
        signalLocs.append(results[i]["locs"])
        zTraces.append(results[i]["zTrace"])
        zProfiles.append(results[i]["zProfiles"])
        res = signalList[i]
        planes = np.append(planes, np.ones(res.shape[1]) * planeRange[i])
    # TODO: combine results
    # check that all signals are the same length
    minLength = 10**10
    for i in range(len(signalList)):
        minLength = np.min((signalList[i].shape[0], minLength))
    for i in range(len(signalList)):
        signalList[i] = signalList[i][:minLength, :]
        if not zTraces[i] is None:
            zTraces[i] = zTraces[i][:minLength]
    signals = np.hstack(signalList)
    locs = np.vstack(signalLocs)
    zProfile = np.hstack(zProfiles)
    zTrace = np.vstack(zTraces)

    # save stuff
    np.save(os.path.join(saveDirectory, "calcium.dff.npy"), signals)
    np.save(os.path.join(saveDirectory, "calcium.planes.npy"), planes)
    np.save(os.path.join(saveDirectory, "rois.xyz.npy"), locs)
    np.save(os.path.join(saveDirectory, "rois.zprofiles.npy"), zProfile)
    np.save(os.path.join(saveDirectory, "planes.zTrace"), zTrace)


# bonsai + arduino
def process_metadata_directory(
    bonsai_dir, ops, pops=create_processing_ops, saveDirectory=None
):

    if saveDirectory is None:
        saveDirectory = os.path.join(suite2pDirectory, "ProcessedData")
    # metadataDirectory_dirList = glob.glob(os.path.join(metadataDirectory,'*'))
    metadataDirectory_dirList = ops["data_path"]

    fpf = ops["frames_per_folder"]
    planes = ops["nplanes"]
    lastFrame = 0

    frameTimes = []
    wheelTimes = []
    faceTimes = []
    bodyTimes = []

    velocity = []

    sparseSt = []
    sparseEt = []
    sparseMaps = []

    retinalSt = []
    retinalEt = []
    retinalStim = []

    gratingsSt = []
    gratingsEt = []
    gratingsOri = []
    gratingsSfreq = []
    gratingsTfreq = []
    gratingsContrast = []
    gratingsReward = []

    circleSt = []
    circleEt = []
    circleX = []
    circleY = []
    circleDiameter = []
    circleWhite = []
    circleDuration = []

    for dInd, di in enumerate(metadataDirectory_dirList):
        if len(os.listdir(di)) == 0:
            continue
        # move on if not a directory (even though ideally all should be a dir)
        # if (not(os.path.isdir(di))):
        #     continue
        expDir = os.path.split(di)[-1]

        # if folder is not selected for analysis move on
        # if not(expDir.isnumeric()) or not (int(expDir) in folder_numbers):
        #     continue

        # frame_in_file = fpf[int(expDir) - 1]
        frame_in_file = fpf[dInd]

        try:
            nidaq, chans, nt = get_nidaq_channels(di, plot=pops["plot"])
        except Exception as e:
            print("Error is directory: " + di)
            print("Could not load nidaq data")
            print(e)
        try:
            frameclock = nidaq[:, chans == "frameclock"]
            frames = assign_frame_time(frameclock, plot=pops["plot"])
            # take only first frames of each go
            frameDiffMedian = np.median(np.diff(frames))
            firstFrames = frames[::planes]
            imagedFrames = np.zeros(frame_in_file) * np.nan
            imagedFrames[: len(firstFrames)] = firstFrames
            planeTimeDelta = np.arange(planes) * frameDiffMedian
        except:
            print("Error is directory: " + di)
            print("Could not extract frames, filling up with NaNs")
            frameTimes.append(np.zeros(frame_in_file) * np.nan)
            continue
        frameTimes.append(imagedFrames + lastFrame)

        sparseFile = glob.glob(os.path.join(di, "SparseNoise*"))
        propsFile = glob.glob(os.path.join(di, "props*.csv"))
        propTitles = np.loadtxt(
            propsFile[0], dtype=str, delimiter=",", ndmin=2
        ).T

        try:
            photodiode = nidaq[:, chans == "photodiode"]
            frameChanges = detect_photodiode_changes(
                photodiode, plot=pops["plot"]
            )
            frameChanges += lastFrame

            # TODO: Have one long st and et list with different identities so a
            # list of st,et and a list with the event type

            # Treat as sparse noise
            if len(sparseFile) != 0:
                sparseMap = get_sparse_noise(di)
                sparseMap = sparseMap[: len(frameChanges), :, :]

                # calculate the end of the final frame
                sparse_et = np.append(
                    frameChanges[1::],
                    frameChanges[-1] + np.median(np.diff(frameChanges)),
                )

                sparseSt.append(frameChanges.reshape(-1, 1).copy())
                sparseEt.append(sparse_et.reshape(-1, 1).copy())
                sparseMaps.append(sparseMap.copy())

                # np.save(os.path.join(saveDirectory,'sparse.st.npy'),frameChanges)
            if propTitles[0] == "Retinal":

                retinal_et = np.append(
                    frameChanges[1::],
                    frameChanges[-1] + (frameChanges[14] - frameChanges[13]),
                )
                retinal_stimType = np.empty(
                    (len(frameChanges), 1), dtype=object
                )
                # retinal_stimType[::13] = "Off"
                # retinal_stimType[1::13] = "On"
                # retinal_stimType[2::13] = "Off"
                # retinal_stimType[3::13] = "Grey"
                # retinal_stimType[4::13] = "ChirpF"
                # retinal_stimType[5::13] = "Grey"
                # retinal_stimType[6::13] = "ChirpC"
                # retinal_stimType[7::13] = "Grey"
                # retinal_stimType[8::13] = "Off"
                # retinal_stimType[9::13] = "Blue"
                # retinal_stimType[10::13] = "Off"
                # retinal_stimType[11::13] = "Green"
                # retinal_stimType[12::13] = "Off"

                retinal_stimType[12::13] = "Off"
                retinal_stimType[0::13] = "On"
                retinal_stimType[1::13] = "Off"
                retinal_stimType[2::13] = "Grey"
                retinal_stimType[3::13] = "ChirpF"
                retinal_stimType[4::13] = "Grey"
                retinal_stimType[5::13] = "ChirpC"
                retinal_stimType[6::13] = "Grey"
                retinal_stimType[7::13] = "Off"
                retinal_stimType[8::13] = "Blue"
                retinal_stimType[9::13] = "Off"
                retinal_stimType[10::13] = "Green"
                retinal_stimType[11::13] = "Off"

                retinalSt.append(frameChanges.reshape(-1, 1).copy())
                retinalEt.append(retinal_et.reshape(-1, 1).copy())
                retinalStim.append(retinal_stimType.copy())

            if len(propTitles) >= 3:
                if propTitles[2] == "Diameter":
                    stimProps = get_stimulus_info(di)
                    circle_et = np.append(
                        frameChanges[1::],
                        frameChanges[-1] + np.median(np.diff(frameChanges)),
                    )

                    circleSt.append(frameChanges.reshape(-1, 1).copy())
                    circleEt.append(circle_et.reshape(-1, 1).copy())

                    circleX.append(
                        stimProps.X.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleY.append(
                        stimProps.Y.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleDiameter.append(
                        stimProps.Diameter.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleWhite.append(
                        stimProps.White.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

                    circleDuration.append(
                        stimProps.Dur.to_numpy()
                        .reshape(-1, 1)
                        .astype(float)
                        .copy()
                    )

            if propTitles[0] == "Ori":
                stimProps = get_stimulus_info(di)

                st = frameChanges[::2].reshape(-1, 1).copy()
                et = frameChanges[1::2].reshape(-1, 1).copy()

                if len(stimProps) != len(st):
                    # raise ValueError(
                    #     "Number of frames and stimuli do not match. Skpping"
                    # )
                    warnings.warn("Number of frames and stimuli do not match")

                gratingsSt.append(st)
                gratingsEt.append(et)
                gratingsOri.append(
                    stimProps.Ori.to_numpy().reshape(-1, 1).astype(int).copy()
                )
                gratingsSfreq.append(
                    stimProps.SFreq.to_numpy()
                    .reshape(-1, 1)
                    .astype(float)
                    .copy()
                )
                gratingsTfreq.append(
                    stimProps.TFreq.to_numpy()
                    .reshape(-1, 1)
                    .astype(float)
                    .copy()
                )
                gratingsContrast.append(
                    stimProps.Contrast.to_numpy()
                    .reshape(-1, 1)
                    .astype(float)
                    .copy()
                )
                if "Reward" in stimProps.columns:
                    gratingsReward.append(
                        np.array(
                            [x in "True" for x in np.array(stimProps.Reward)]
                        )
                        .reshape(-1, 1)
                        .astype(bool)
                        .copy()
                    )
                else:
                    gratingsReward.append(np.zeros_like(st) * np.nan)

        except:
            print("Error in stimulus processing in directory: " + di)
            print(traceback.format_exc())
        # arduino handling
        try:
            ardData, ardChans, at = get_arduino_data(di)
            nidaqSync = nidaq[:, chans == "sync"][:, 0]
            ardSync = ardData[:, ardChans == "sync"][:, 0]
            at_new = arduino_delay_compensation(nidaqSync, ardSync, nt, at)

            movement1 = ardData[:, ardChans == "rotary1"][:, 0]
            movement2 = ardData[:, ardChans == "rotary2"][:, 0]
            v, d = detect_wheel_move(movement1, movement2, at_new)

            wheelTimes.append(at_new + lastFrame)
            velocity.append(v)

            camera1 = ardData[:, ardChans == "camera1"][:, 0]
            camera2 = ardData[:, ardChans == "camera2"][:, 0]
            cam1Frames = assign_frame_time(camera1, fs=1, plot=False)
            cam2Frames = assign_frame_time(camera2, fs=1, plot=False)
            cam1Frames = at_new[cam1Frames.astype(int)]
            cam2Frames = at_new[cam2Frames.astype(int)]

            faceTimes.append(cam1Frames + lastFrame)
            bodyTimes.append(cam2Frames + lastFrame)
        except:
            print("Error in arduino processing in directory: " + di)
            print(traceback.format_exc())
        lastFrame = nt[-1] + lastFrame
    np.save(
        os.path.join(saveDirectory, "calcium.timestamps.npy"),
        np.hstack(frameTimes).reshape(-1, 1),
    )
    np.save(
        os.path.join(saveDirectory, "planes.delay.npy"),
        planeTimeDelta.reshape(-1, 1),
    )

    if len(sparseMaps) > 0:
        np.save(
            os.path.join(saveDirectory, "sparse.map.npy"),
            np.vstack(sparseMaps),
        )
        np.save(
            os.path.join(saveDirectory, "sparse.st.npy"), np.vstack(sparseSt)
        )
        np.save(
            os.path.join(saveDirectory, "sparse.et.npy"), np.vstack(sparseEt)
        )
    if len(retinalStim) > 0:
        np.save(
            os.path.join(saveDirectory, "retinal.st.npy"), np.vstack(retinalSt)
        )
        np.save(
            os.path.join(saveDirectory, "retinal.et.npy"), np.vstack(retinalEt)
        )
        np.save(
            os.path.join(saveDirectory, "retinal.stim.npy"),
            np.vstack(retinalStim),
        )
    if len(gratingsSt) > 0:
        np.save(
            os.path.join(saveDirectory, "gratings.st.npy"),
            np.vstack(gratingsSt),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.et.npy"),
            np.vstack(gratingsEt),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.ori.npy"),
            np.vstack(gratingsOri),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.spatialF.npy"),
            np.vstack(gratingsSfreq),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.temporalF.npy"),
            np.vstack(gratingsTfreq),
        )
        np.save(
            os.path.join(saveDirectory, "gratings.contrast.npy"),
            np.vstack(gratingsContrast),
        )

    if len(circleSt) > 0:
        np.save(
            os.path.join(saveDirectory, "circles.st.npy"),
            np.vstack(circleSt),
        )
        np.save(
            os.path.join(saveDirectory, "circles.et.npy"),
            np.vstack(circleEt),
        )
        np.save(
            os.path.join(saveDirectory, "circles.x.npy"),
            np.vstack(circleX),
        )
        np.save(
            os.path.join(saveDirectory, "circles.y.npy"),
            np.vstack(circleY),
        )
        np.save(
            os.path.join(saveDirectory, "circles.diameter.npy"),
            np.vstack(circleDiameter),
        )
        np.save(
            os.path.join(saveDirectory, "circles.isWhite.npy"),
            np.vstack(circleWhite),
        )
        np.save(
            os.path.join(saveDirectory, "circles.duration.npy"),
            np.vstack(circleDuration),
        )

    if len(gratingsReward) > 0:
        np.save(
            os.path.join(saveDirectory, "gratings.reward.npy"),
            np.vstack(gratingsReward),
        )
    if len(wheelTimes) > 0:
        np.save(
            os.path.join(saveDirectory, "wheel.timestamps.npy"),
            np.hstack(wheelTimes).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "wheel.velocity.npy"),
            np.hstack(velocity).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "face.timestamps.npy"),
            np.hstack(faceTimes).reshape(-1, 1),
        )
        np.save(
            os.path.join(saveDirectory, "body.timestamps.npy"),
            np.hstack(bodyTimes).reshape(-1, 1),
        )


def read_csv_produce_directories(dataEntry, s2pDir, zstackDir, metadataDir):
    """
    Gets all the base directories (suite2p, z Stack, metadata, save directory)
    and composes these directories for each experiment.


    Parameters
    ----------
    dataEntry : pandas DataFrame [amount of experiments, 6]
        The data from the preprocess.csv file in a pandas dataframe. This should have been
        created in the main_preprocess file; assumes these columns are included:
            - Name
            - Date
            - Zstack
            - IgnorePlanes
            - SaveDir
            - Process
    s2pDir : string
        Filepath to the Suite2P processed folder. For more details on what this should contain
        please look at the define_directories function definition in folder_defs.
    zstackDir : string
        Filepath to the Z stack.For more details on what this should contain
        please look at the define_directories function definition in folder_defs.
    metadataDir : string
        Filepath to the metadata directory.For more details on what this should contain
        please look at the define_directories function definition in folder_defs.

    Returns
    -------
    s2pDirectory : string [s2pr\Animal\Date\suite2p]
        The concatenated Suite2P directory.
    zstackPath : string [zstackDir\Animal\Date\Z stack value from dataEntry\Z_stack_file.tif]
        The concatenated Z stack directory.
    metadataDirectory : string [metadataDir\Animal\Date]
        The concatenated metadata directory.
    saveDirectory : string [SaveDir from dataEntry or ]
        The save directory where all the processed files are saved. If not specified,
        will be saved in the suite2p folder.

    """
    # The data from each  dataEntry column is placed into variables.
    name = dataEntry.Name
    date = dataEntry.Date
    zstack = dataEntry.Zstack
    ignorePlanes = np.fromstring(str(dataEntry.IgnorePlanes), sep=",")
    saveDirectory = dataEntry.SaveDir
    process = dataEntry.Process

    # Joins suite2p directory with the name and the date.
    s2pDirectory = os.path.join(s2pDir, name, date, "suite2p")

    # If this path doesn't exist, returns a ValueError.
    if not os.path.exists(s2pDirectory):
        raise ValueError(
            "suite 2p directory " + s2pDirectory + "was not found."
        )
    # Checks if zStack directory number has the right shape (is not a float or a NaN).
    if (type(zstack) is float) and (np.isnan(zstack)):
        zstackPath = None
        zstackDirectory = None
    else:
        # Creates the Z Stack directory.
        zstackDirectory = os.path.join(zstackDir, name, date, str(zstack))
        try:
            # Returns a path to the tif file with the Z stack within the specified zstackDirectory.
            zstackPath = glob.glob(os.path.join(zstackDirectory, "*.tif"))[0]
        except:
            # If no Z stack directory was specified in the preprocess file, returns a ValueError.
            # Note that the Z stack is essential for performing the Z correction!
            raise ValueError(
                "Z stack Directory not found. Please check the number in the processing csv"
            )
    # Joins suite2p directory with the name and the date.
    metadataDirectory = os.path.join(metadataDir, name, date)

    # If metadata directory does not exist, returns this ValueError.
    if not os.path.exists(metadataDirectory):
        raise ValueError(
            "metadata directory " + metadataDirectory + "was not found."
        )

    if not type(saveDirectory) is str:
        # If the saveDirectory is not a string, saves files created here
        # in a folder called PreProcessedFiles. This exists in the s2pDirectory.
        # This also means this folder is not created if the saveDirectory is specified.
        saveDirectory = os.path.join(s2pDirectory, "PreprocessedFiles")
    # Creates the folder Preprocessedfiles if it doesn't exist yet
    if not os.path.isdir(saveDirectory):
        os.makedirs(saveDirectory)
    return s2pDirectory, zstackPath, metadataDirectory, saveDirectory
