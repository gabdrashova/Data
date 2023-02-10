"""Pre-process calcium traces extracted from tiff files."""
import numpy as np
from scipy import optimize
from Data.TwoP.general import linearAnalyticalSolution
import pandas as pd


def GetCalciumAligned(signal, time, eventTimes, window, planes, delays):
    aligned = []
    run = 0
    ps = np.unique(planes).astype(int)
    for p in range(len(ps)):
        aligned_tmp, t = AlignStim(
            signal[:, np.where(planes == ps[p])[0]],
            time + delays[0, ps[p]],
            eventTimes,
            window,
        )
        if run == 0:
            aligned = aligned_tmp
            run += 1
        else:
            aligned = np.concatenate((aligned, aligned_tmp), axis=2)
    return np.array(aligned), t


def correct_neuropil(
    F: np.ndarray,
    N: np.ndarray,
    fs,
    numN=20,
    minNp=10,
    maxNp=90,
    prctl_F=5,
    verbose=True,
):
    """
    Estimates the correction factor r for neuropil correction, so that:
        C = S - rN
        with C: actual signal from the ROI, S: measured signal, N: neuropil

    Parameters
    ----------
    F : np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.
    N : np.ndarray [t x nROIs]
        Neuropil traces of ROIs.
    numN : int, optional
        Number of bins used to partition the distribution of neuropil values. 
        Each bin will be associated with a mean neuropil value and a mean 
        signal value. The default is 20.
    minNp : int, optional
        Minimum values of neuropil considered, expressed in percentile.
        0 < minNp < 100. The default is 10.
    maxNp : int, optional
        Maximum values of neuropil considered, expressed in percentile.
        0 < maxNp < 100, minNp < maxNp. The
        default is 90.
    prctl_F : int, optional
        Percentile of the measured signal that will be matched to neuropil. 
        The default is 5.
    verbose : boolean, optional
        Feedback on fitting. The default is True.

    Returns
    -------
    signal : np.ndarray [t x nROIs]
        Neuropil corrected calcium traces.
    regPars : np.ndarray [2 x nROIs], each row: [intercept, slope]
        Intercept and slope of linear fits of neuropil (N) to measured calcium 
        traces (F)
    F_binValues : np.array [numN, nROIs]
        Low percentile (prctl_F) values for each calcium trace bin. These
        values were used for linear regression.
    N_binValues : np.array [numN, nROIs]
        Values for each neuropil bin. These values were used for linear 
        regression.

    Based on Matlab function estimateNeuropil (in +preproc) written by Mario 
    Dipoppa and Sylvia Schroeder
    """

    [nt, nROIs] = F.shape
    N_binValues = np.ones((numN, nROIs)) * np.nan
    F_binValues = np.ones((numN, nROIs)) * np.nan
    regPars = np.ones((2, nROIs)) * np.nan
    signal = np.ones((nt, nROIs)) * np.nan

    
    # Computes the F0 traces for Calcium traces and Neuropil traces respectively.
    # Refer to the function for further details on how this is done.
    F0 = get_F0(F, fs)
    N0 = get_F0(N, fs)
    # Corrects for slow drift by subtracting F0 from F and N traces.
    Fc = F - F0 
    Nc = N - N0
    

    # Determines where the minimum normalised difference between F0 and N0.
    ti = np.nanargmin((F0 - N0) / N0, 0) 

    lowActivity = np.nanpercentile(F, 50, 0) # Calculates the F at the 50th percentile. 
    for iROI in range(nROIs):
        # TODO: verbose options
        # Adds the value of F0 at which point the difference between F and N baselines is minimal.
        #TODO: WHY
        Fc[:, iROI] += F0[ti[iROI], iROI]
        Nc[:, iROI] += N0[ti[iROI], iROI]
        # Gets the current F and N trace.
        iN = Nc[:, iROI]
        iF = Fc[:, iROI]

        # Gets low and high percentile of neuropil trace.
        N_prct = np.nanpercentile(iN, np.array([minNp, maxNp]), axis=0)
        # Divides neuropil values into numN groups.
        binSize = (N_prct[1] - N_prct[0]) / numN
        # Gets neuropil values regularly spaced across range between minNp and maxNp.
        N_binValues[:, iROI] = N_prct[0] + (np.arange(0, stop=numN)) * binSize

        # Discretize values of neuropil between minN and maxN, with numN elements
        # N_ind contains values: 0...binSize for N values within minNp and maxNp.
        # Done to determine in which bin each data point belongs to.
        N_ind = np.floor((iN - N_prct[0]) / binSize)

        # Finds the matching (low percentile) value from F trace for each neuropil bin.
        # This is to determine values of F that are relatively low as these are unlikely to reflect neural spiking.
        for Ni in range(numN):
            tmp = np.ones_like(iF) * np.nan
            tmp[N_ind == Ni] = iF[N_ind == Ni]
            F_binValues[Ni, iROI] = np.nanpercentile(tmp, prctl_F, 0)
        # Fits only non-nan values.
        noNan = np.where(
            ~np.isnan(F_binValues[:, iROI]) & ~np.isnan(N_binValues[:, iROI])
        )[0]
        #TODO: CONTINUE
        # perform linear regression between neuropil and signal bins under constraint that 0<slope<2
        # res, _ = optimize.curve_fit(_linear, N_binValues[noNan, iROI], F_binValues[noNan, iROI],
        #                             p0=(np.nanmean(F_binValues[:, iROI]), 0), bounds=([-np.inf, 0], [np.inf, 2]))
        # Finds analytical solution to determine the correction factor
        #TODO: WHY these equations specifically
        a, b, mse = linearAnalyticalSolution(
            N_binValues[noNan, iROI], F_binValues[noNan, iROI], False
        )
        # regPars[:, iROI] = res
        regPars[:, iROI] = (a, b)
        #TODO: WHY is b (the slope of the linear fit used)
        ## avoid over correction
        # b = min(b, 1)
        corrected_sig = iF - b * iN

        # determine neuropil correct signal
        signal[:, iROI] = corrected_sig.copy()
    return signal, regPars, F_binValues, N_binValues


# TODO
def correct_zmotion(F, zprofiles, ztrace, ignore_faults=True, metadata={}):
    """
    Corrects changes in fluorescence due to brain movement along z-axis (depth). Method is based on algorithm
    described in Ryan, ..., Lagnado (J Physiol, 2020)

    Parameters
    ----------
    F : np.array [t x nROIs]
        Calcium traces (measured signal) of ROIs from a single(!) plane. It is assumed that these are neuropil corrected!
    zprofiles : np.array [slices x nROIs]
        Fluorescence profiles of ROIs across depth of z-stack. These profiles are assumed to be neuropil corrected!
    ztrace : np.array [t]
        Depth of each frame of the imaged plane. Indices in this array refer to slices in zprofiles.

    Returns
    -------
    signal : np.array [t x nROIs]
        Z-corrected calcium traces.
    """

    """
    Steps
    1) Smooth z-profile of each ROI using Moffat function.
    2) Create correction vector based on z-profiles and ztrace.
    3) Correct calcium traces using correction vector.
    """

    # Step 1 - Consider smoothing instead of a Moffat function. Considering how our stacks look

    # find correction factor

    referenceDepth = int(np.round(np.median(ztrace)))
    # zprofiles = zprofiles - np.min(zprofiles, 0)
    correctionFactor = zprofiles / zprofiles[referenceDepth, :]

    # Step 2 - If taking the raw data from ops need to get the correct frame ...
    # by taking the max correlation for each time point
    correctionMatrix = correctionFactor[ztrace, :]
    # Step 3 - Correct
    signal = F / correctionMatrix

    if ignore_faults:
        signal = remove_zcorrected_faults(
            ztrace, correctionFactor, signal, metadata
        )
    return signal


# TODO
def register_zaxis():
    None


# TODO
def get_F0(Fc, fs, prctl_F=8, window_size=60, verbose=True):
    """
    Determines the baseline fluorescence to use for computing deltaF/F.
   

    Parameters
    ----------
    Fc : np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.
    fs : float
        The frame rate (frames/second/plane).
    prctl_F : int, optional
        The percentile from which to take F0. The default is 8.
    window_size : int, optional
        The rolling window over which to calculate F0. The default is 60.
    verbose : bool, optional
        Whether or not to provide detailed processing information. 
        The default is True.

    Returns
    -------
    F0 : np.ndarray [t x nROIs]
        The baseline fluorescence (F0) traces for each ROI.

    """
    # Translates the window size from seconds into frames.
    window_size = int(round(fs * window_size))
    # Creates an array with the shape of Fc where the F0 values will be placed.
    F0 = np.zeros_like(Fc)
    # Converts Fc into a pandas array and pads the array using the window size
    # as a pad width.
    Fc_pd = pd.DataFrame(
        np.pad(
            Fc,
            [
                (window_size, 0),
                (0, 0),
            ],
            mode="median",
        )
    )
    # Calculate F0 by checking the percentile specified from the rolling window.
    F0 = np.array(
        Fc_pd.rolling(window_size).quantile(
            prctl_F * 0.01, interpolation="midpoint"
        )
    )
    # Removes the padded timepoints at the beginning.
    F0 = F0[window_size:]
    # for t in range(0, Fc.shape[0]):
    #     rng = np.arange(t, np.min([len(Fc), t + window_size]))
    #     F0t = np.nanpercentile(Fc[rng, :], prctl_F, 0)
    #     # F0[t, :] = np.tile(
    #     #     F0t, (Fc[rng, :].shape[0], 1)
    #     F0[t, :] = F0t
    return F0


# TODO: understand why np.fmax is used
def get_delta_F_over_F(Fc, F0):
    """
    Calculates delta F over F by subtracting F0 from Fc and dividing this by
    the maximum of the mean F of F0.

    Parameters
    ----------
    Fc :np.ndarray [t x nROIs]
        Calcium traces (measured signal) of ROIs.
    F0 : np.ndarray [t x nROIs]
        The baseline fluorescence (F0) traces of ROIs.

    Returns
    -------
    np.ndarray [t x nROIs]
    Normalised fluorescence traces (dF/F) of ROIs.

    """
    return (Fc - F0) / np.fmax(1, np.nanmean(F0, 0))


def remove_zcorrected_faults(ztrace, zprofiles, signals, metadata={}):
    """
    This functions cleans timepoints in the trace where the imaging takes place
    in a plane that is meaningless as to cell activity.
    This is defined as times when there are two peaks or slopes in the imaging
    region and the imaging plane is in the second slop.

    Parameters
    ----------
    ztrace : TYPE
        the imaging plane on the z-axis
    zprofiles : TYPE
        the z-profiles of all the cells.
    signals : TYPE
        the signal traces.

    Returns
    -------
    signals: the corrected sigals with the faulty timepoints removed.

    """

    zp_focused = zprofiles[min(ztrace) : max(ztrace), :]
    ztrace -= min(ztrace)
    dif = np.diff(zp_focused, 1, axis=0)
    zero_crossings_inds = np.where(np.diff(np.signbit(dif), axis=0))
    zero_crossing = zero_crossings_inds[0]
    cells = zero_crossings_inds[1]
    imagingPlane = np.median(ztrace)  # -min(ztrace)
    metadata["removedIndex"] = []
    for i in range(signals.shape[1]):
        cellInds = np.where(cells == i)[0]
        # no zero crossings of the derivative means imaging is on a monotonous slope
        if len(cellInds) == 0:
            continue
        zc = zero_crossing[cellInds]
        distFromPlane = imagingPlane - zc
        planeCrossingInd = np.argmin(abs(distFromPlane))
        planeCrossing = zc[planeCrossingInd]
        # if there are many crossings, find out what is the closest one to
        # the normal plane
        if len(zc) > 1:

            # the imaging plane has the first peak/trough - discardanythin that comes after the rest
            if planeCrossingInd == 0:
                signals[np.where(ztrace > zc[1]), i] = np.nan
            else:
                signals[
                    np.where(ztrace < zc[planeCrossingInd - 1]), i
                ] = np.nan
        # Check if differential is positive before crossing
        # If that's the case we're golden, the problem is if negative
        # Then it's a trough and if depends on what side of the trough we are
        # imaging
        if dif[planeCrossingInd, i] < 0:
            if distFromPlane[planeCrossingInd] < 0:
                removeInd = np.where(ztrace > planeCrossing + 1)[0]
            else:
                removeInd = np.where(ztrace < planeCrossing + 1)[0]
            signals[removeInd, i] = np.nan
        metadata["removedIndex"].append(np.where(np.isnan(signals))[0])
    return signals


def _linear(x, a, b):
    return a + b * x


def zero_signal(F):
    """
   This function adds the value 19520 to all ROIs across time.This value represents the absolute zero signal 
   and was obtained by averaging the darkest frame over many imaging sessions. It is important to note 
   that the absolute zero value is arbitrary and depends on the voltage range of the PMTs.
   
   

   Parameters
   ----------
   F : np.ndarray [t x nROIs]
   Calcium traces (measured signal) of ROIs.

   Returns
   -------
   F : np.ndarray [t x nROIs]
   Calcium traces (measured signal) of ROIs with the addition of the absolute
   zero signal.

   """
    return F + 19520
