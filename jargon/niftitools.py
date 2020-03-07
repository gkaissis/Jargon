'''
Functions to work with nifti files

'''

import nibabel
import matplotlib.pyplot as plt
import math

def nii_to_np(path_to_nii, squeeze_empty=False):
    '''
    Convert a nifti into a numpy.ndarray

    Parameters
    ----------
    path_to_nii (str or pathlike object): Path to the nifti file
    squeeze_empty (bool, default: False): Whether to squeeze empty
    array dimensions

    Returns
    ----------
    numpy.ndarray
    '''

    nii = nibabel.load(path_to_nii).get_fdata()
    if squeeze_empty:
        nii = nii.squeeze()
    print(f"Generated array of shape: {nii.shape}")
    return nii



def show_nifti(path_to_nii, cmap="gray", figsize=(5,5)):
    '''
    Show a nifti file as a panel of figures

    Parameters
    ----------
    path_to_nii (str or pathlike object): Path to the nifti file
    cmap (str or matplotlib.cmap): Color map to use
    figsize (tuple): Size of the figure

    Returns
    ----------
    matplotlib figure
    '''
    img = nibabel.load(path_to_nii).get_fdata().squeeze()

    fig, ax = plt.subplots(math.ceil(img.shape[2]/5), 5, figsize=(15,15))
    try:
        for i, axi in enumerate(ax.flat):
            axi.imshow(np.fliplr(np.rot90(img[...,i], axes=(1,0))), cmap="gray")
            axi.set_title(i)
    except Exception as e:
        print(f"Error caught: {e}")
        pass
