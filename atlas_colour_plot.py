# -*- coding: utf-8 -*-
"""
Plot values in parcellated brain regions, copy and paste into script.

@author: Sebastian C. Coleman
"""

from nilearn import plotting, image, datasets
from nibabel import nifti1, Nifti1Image
import numpy as np
from scipy.stats import zscore

def atlas_plotter(atlas_file, values, threshold=50, cmap='cold_hot'):
    
    # load fsaverage and atlas   
    atlas_img = image.load_img(atlas_file)
    atlas_data = atlas_img.get_fdata()
    indices = np.sort(np.unique(atlas_data[atlas_data>0]))
    
    # empty image
    atlas_new = np.zeros(np.shape(atlas_data))
    
    # place values in each parcel region
    for reg in range(len(values)):
        reg_i = atlas_data==indices[reg]
        atlas_new[reg_i] = values[reg]
    
    # make image from new atlas data
    new_img = nifti1.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # find threshold for image 
    threshold = (threshold/100) * np.max(np.abs(values))
    
    # plot
    plotting.plot_img_on_surf(new_img, views=['lateral', 'medial'],
                              hemispheres=['left', 'right'], darkness=0.7,
                              colorbar=True, cmap=cmap, bg_on_data=True,
                              threshold=threshold)


def atlas_4d_volume_plotter(atlas_file, values, z_threshold=1.5, display_mode='ortho', cmap='cold_hot'):
    """
    Plots z-scored scalar values in parcellation, overlaid on MNI brain template.

    Parameters:
    ----------
    atlas_file : str
        Path to the 4D atlas file in Nifti format.
    values : array-like
        List or array of values corresponding to each region in the atlas.
    z_threshold : float, optional
        Threshold value for displaying z-scores (default is 1.5).
    display_mode : {'ortho', 'x', 'y', 'z'}, optional
        Specifies the direction of the slices ('ortho' for three views or 'x', 'y', 'z' for tiled single view).
    cmap : str, optional
        Colormap to use for displaying the values (default is 'cold_hot').

    Returns:
    -------
    None
        Displays the 4D atlas plot.

    Notes:
    ------
    - The parcellation must be 4D, i.e., each 3D volume is a binary mask for a parcel.
    - Uses MNI152 template for visualization.
    - Requires Nilearn library for image handling and plotting.
    """
    
    # z-score values
    values = zscore(values)
    vmax = 1.3 * np.max(np.abs(values))
    vmin = -vmax
    
    # load fsaverage and atlas   
    mni = datasets.load_mni152_template()
    atlas_img = image.load_img(atlas_file)
    atlas_data = atlas_img.get_fdata()

    # different procedure for 3D vs 4D parcellation files
    if len(atlas_img.shape)==4:
        
        # empty image
        single_vol = image.index_img(atlas_img, 0)
        atlas_new = np.zeros(np.shape(atlas_data))
        
        # place values in each parcel region
        for reg in range(len(values)):
            atlas_new[:, :, :, reg] = atlas_data[:, :, :, reg] * values[reg]
        
        # mean over fourth dimension
        atlas_new = np.mean(atlas_new, 3) * len(values)
        
        # make image from new atlas data
        new_img = Nifti1Image(atlas_new, single_vol.affine, single_vol.header)

    elif len(atlas_img.shape)==3:

        # empty image
        atlas_new = np.zeros(np.shape(atlas_data))

        # place values in each parcel region
        for reg in range(len(values)):
            reg_mask = atlas_data == reg
            atlas_new[reg_mask] = values[reg]

        # make image from new atlas data
        new_img = Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # interpolate image
    img_interp = image.resample_img(new_img, mni.affine)
    
    # plot the interpolated image
    plotting.plot_stat_map(img_interp, display_mode=display_mode, threshold=z_threshold, 
                           vmin=vmin, vmax=vmax, symmetric_cbar=True, annotate=False,
                           draw_cross=False)
