"""Pre-process tiff files."""


import os
import numpy as np
import pandas as pd
import skimage
import scipy as sp
from skimage import io
from skimage import data
from skimage import metrics
from skimage.util import img_as_float
import tifftools as tt
import pandas as pd

# from pystackreg import StackReg
from suite2p.extraction.extract import extract_traces
from suite2p.extraction.masks import create_masks
from suite2p.registration.register import register_frames, compute_reference
from suite2p.registration import rigid
from Data.TwoP.preprocess_traces import correct_neuropil
from suite2p.default_ops import default_ops
from numba import jit

from Data.TwoP.preprocess_traces import zero_signal


@jit(forceobj=True)
def _fill_plane_piezo(stack, piezoNorm, i, spacing=1):
    """


    Parameters
    ----------
    stack : array [z,y,X]
        the registered image stack.
    piezoNorm : array [t]
        the piezo depth normalised to the stack spacing.
    i: int
        the frame to create
    Returns
    -------
    a normalised frame.

    """
    # Normalised the piezo trace to the current depth
    piezoNorm -= piezoNorm[0]
    piezoNorm += i

    planes = stack.shape[0]
    resolutionx = stack.shape[1]
    resolutiony = stack.shape[2]
    currPixelY = 0
    currDepth = 0
    slantImg = np.zeros(stack.shape[1:])

    pixelsPerMoveY = np.ones(len(piezoNorm)) * resolutiony

    # Nmber of pixels per piezo step
    numPixelsY = np.round(pixelsPerMoveY / len(piezoNorm)).astype(int)

    # correct in case of rounding error
    Yerr = resolutiony - sum(numPixelsY)
    numPixelsY[-1] += Yerr

    # The end points of each time bit
    pixelsY = np.cumsum(numPixelsY).astype(int)

    interp = sp.interpolate.RegularGridInterpolator(
        (
            np.arange(0, planes, spacing),
            np.arange(0, resolutiony),
            np.arange(0, resolutionx),
        ),
        stack,
        fill_value=None,
    )
    for d in range(len(piezoNorm)):
        endPointY = pixelsY[d]
        depth = piezoNorm[d]

        # If beyond the depth take the final frame
        if depth > planes - 1:
            depth = planes - 1
        # If below the topmost frame take the first one
        if depth < 0:
            depth = 0
        for yt in np.arange(currPixelY, endPointY):
            # print (depth,yt)
            line = interp(
                (
                    depth,
                    yt,
                    np.arange(0, resolutionx),
                )
            )
            slantImg[yt, 0:resolutionx] = line
        currPixelY += numPixelsY[d]
        currDepth += depth
    return slantImg


def _register_swipe(zstack, start, finish, progress):
    # print(str(start)+', finish:' + str(finish)+', progress:'+str(progress))
    for i in range(start, finish, progress):
        if i == 0:
            stackRange = range(i, i + 2)
        elif i == zstack.shape[0] - 1:
            stackRange = range(i - 1, i + 1)
        else:
            stackRange = range(i - 1, i + 2)
        # print(str(i))
        miniStack = zstack[stackRange]
        res = register_frames(
            miniStack[1, :, :], miniStack[:, :, :].astype(np.int16)
        )
        zstack[stackRange, :, :] = res[0]
    return zstack


def register_zstack_frames(zstack):
    #### Start from centre take triples and align them
    centreFrame = int(np.floor(zstack.shape[0] / 2))
    tempStack = np.zeros_like(zstack)
    # swipe up
    zstack = _register_swipe(zstack, centreFrame, 0, -1)
    # swipe down
    zstack = _register_swipe(zstack, centreFrame, zstack.shape[0], 1)
    # top to bottom
    zstack = _register_swipe(zstack, centreFrame, zstack.shape[0], 1)
    return zstack


def registerStacktoRef(zstack, refImg, ops=default_ops()):
    ref = rigid.phasecorr_reference(refImg, ops["smooth_sigma"])
    data = rigid.apply_masks(
        zstack.astype(np.int16),
        *rigid.compute_masks(
            refImg=refImg,
            maskSlope=ops["spatial_taper"]
            if ops["1Preg"]
            else 3 * ops["smooth_sigma"],
        )
    )
    corrRes = rigid.phasecorr(
        data,
        ref.astype(np.complex64),
        ops["maxregshift"],
        ops["smooth_sigma_time"],
    )
    maxCor = np.argmax(corrRes[-1])
    dx = corrRes[1][maxCor]
    dy = corrRes[0][maxCor]
    zstackCorrected = np.zeros_like(zstack)
    for z in range(zstack.shape[0]):
        frame = zstack[z, :, :]
        zstackCorrected[z, :, :] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)
    return zstackCorrected


# TODO. Also: return new planes following a certain angle through the z-stack. New plane should follow movement trace of
# piezo.
def register_zstack(tiff_path, spacing=1, piezo=None, target_image=None):
    """
    Loads tiff file containing imaged z-stack, aligns all frames to each other, averages across repetitions, and (if
    piezo not None) reslices the 3D z-stack so that slant/orientation of the new slices matches the slant of the frames
    imaged during experiments (slant given by piezo trace).

    Parameters
    ----------
    tiff_path : String
        Path to tiff file containing z-stack.
    piezo : np.array [t]
        Movement of piezo across z-axis for one plane. Unit: microns. Raw taken from niDaq
    [Note: need to add more input arguments depending on how registration works. Piezo movement might need to provided
    in units of z-stack slices if tiff header does not contain information about depth in microns]
    spacing: distance between planes (in microns)
    target_image : np.array [x x y]
        Image used by suite2p to align frames to. Is needed to align z-stack to this image and then apply masks at
        correct positions.

    Returns
    -------
    zstack : np.array [x x y x z]
        Registered (and resliced) z-stack.
    """

    image = skimage.io.imread(tiff_path)

    planes = image.shape[0]
    resolutionx = image.shape[2]
    resolutiony = image.shape[3]
    zstack = np.zeros((planes, resolutionx, resolutiony))
    for i in range(planes):
        # sr = StackReg(StackReg.TRANSLATION)
        # reg_arrays = sr.register_transform_stack(image[i,:,:,:], reference=reference)
        res = register_frames(
            image[i, 0, :, :], image[i, :, :, :].astype(np.int16)
        )
        # zstack[i,:,:] = np.mean(reg_arrays, axis=0)
        zstack[i, :, :] = np.mean(res[0], axis=0)
    zstack = register_zstack_frames(zstack)

    if not (piezo is None):
        piezoNorm = piezo / spacing
        depthDiff = np.diff(piezo)
        totalDepthTravelled = sum(depthDiff)
        proportionTavelled = depthDiff / totalDepthTravelled
        zstackTmp = np.zeros(zstack.shape)
        for p in range(planes):
            # print(p)
            zstackTmp[p, :, :] = _fill_plane_piezo(zstack, piezoNorm, p)
        zstack = zstackTmp
    # if (save):
    #     savePath = os.path.splitext(tiff_path)[0]+'_angled'
    #     svTmp = savePath
    #     i = 0
    #     while (os.path.exists(savePath+'.tif')):
    #         savePath = svTmp+str(i)
    #         i+=1
    #     savePath += '.tif'
    #     io.imsave(savePath, zstack)

    if not (target_image is None):
        zstack = registerStacktoRef(zstack, target_image)
    return zstack


def _moffat(r, B, A, alpha, beta):
    return B + A * (1 + (((r) ** 2) / alpha**2)) ** -beta


def _gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


# TODO


def extract_zprofiles(
    extraction_path,
    zstack,
    neuropil_correction=None,
    ROI_masks=None,
    neuropil_masks=None,
    smooting_factor=None,
    metadata={},
):
    """
    Extracts fluorescence of ROIs across depth of z-stack.

    Parameters
    ----------
    ROI_masks : np.array [x x y x nROIs]
        (output of suite2p so need to check the format of their ROI masks)
        Pixel masks of ROIs in space (x- and y-axis).
    neuropil_masks : np.array [x x y x nROIs]
        (this assumes that suite2p actually uses masks for neuropil. I'm pretty sure there is this option to use masks
        but we need to use this option instead of using "basis functions".)
        Pixel masks of ROI's neuropil in space (x- and y-axis).
    zstack : np.array [x x y x z]
        Registered z-stack where slices are oriented the same way as imaged planes (output of register_zstack).
    target_image : np.array [x x y]
        Image used by suite2p to align frames to. Is needed to align z-stack to this image and then apply masks at
        correct positions.
    #All of these factors can be replaced by the plane directory - Liad 13/07/2022
    neuropil_correction : np.array [nROIs]
        Correction factors determined by preprocess_traces.correct_neuropil.

    Returns
    -------
    zprofiles : np.array [z x nROIs]
        Depth profiles of ROIs.
    """

    """
    Steps
    1) Register z-stack to target image.
    2) Extract fluorescence within ROI masks across all slices of z-stack.
    3) Extract fluorescence within neuropil masks across all slices of z-stack.
    4) Perform neuropil correction on ROI traces using neuropil traces and correction factors.
    
    Notes (useful functions in suite2p);
    - neuropil masks are created in /suite2p/extraction/masks.create_neuropil_masks called from masks.create_masks
    - ROI and neuropil traces extracted in /suite2p/extraction/extract.extract_traces called from 
      extract.extraction_wrapper
    - to register frames, see line 285 (rigid registration) in /suite2p/registration/register for rigid registration
    """

    stat = np.load(
        os.path.join(extraction_path, "stat.npy"), allow_pickle=True
    )
    ops = np.load(
        os.path.join(extraction_path, "ops.npy"), allow_pickle=True
    ).item()
    isCell = np.load(os.path.join(extraction_path, "iscell.npy")).astype(bool)

    ### Step 1
    #
    # X = ops['Lx']
    # Y = ops['Ly']
    # if (target_image is None):
    #     refImg = ops['refImg']
    # else:
    X = zstack.shape[1]
    Y = zstack.shape[2]

    # zstack_reg = registerStacktoRef(zstack,refImg,ops)

    if (ROI_masks is None) and (neuropil_masks is None):
        rois, npils = create_masks(stat, Y, X, ops) # Suite2P function.
    zProfile, Fneu = extract_traces(zstack, rois, npils, 1)
    zProfile = zero_signal(zProfile)
    Fneu = zero_signal(Fneu)
    zProfile = zProfile[isCell[:, 0], :].T
    Fneu = Fneu[isCell[:, 0], :].T

    zprofileRaw = zProfile.T.copy()
    # Perform neuropil correction
    if not (neuropil_correction is None):
        zProfile = zProfile - neuropil_correction.reshape(1, -1) * Fneu
        # zProfile = np.fmax(zProfile,np.ones(zProfile.shape)*Fbl.reshape(-1,1))
        # zProfile = zProfile.T
    # zProfileC = np.zeros(zProfile.shape)

    if not (smooting_factor is None):
        zProfile = sp.ndimage.gaussian_filter1d(
            zProfile, smooting_factor, axis=0
        )
    depths = np.arange(
        -(zstack.shape[0] - 1) / 2, (zstack.shape[0] - 1) / 2 + 1
    )
    metadata["zprofiles_raw"] = zprofileRaw
    metadata["zprofiles_neuropil"] = Fneu.T

    return zProfile
