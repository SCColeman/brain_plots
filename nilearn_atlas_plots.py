# -*- coding: utf-8 -*-
"""
Plot values in parcellated brain regions, copy and paste into script.

@author: Sebastian C. Coleman
"""

from nilearn import plotting, image, datasets
from nibabel import Nifti1Image
import numpy as np

def atlas_surface_plotter(atlas_file, values, threshold=0, cmap='cold_hot', symmetric_cbar=True):
    """
    Plots values in parcellation, overlaid on fsaverage5 surface.

    Parameters:
    ----------
    atlas_file : str
        Path to the atlas file in Nifti format.
    values : array-like
        List or array of values corresponding to each region in the atlas.
    threshold : float
        Threshold value for displaying values.
    cmap : str, optional
        Colormap to use for displaying the values (default is 'cold_hot').

    Returns:
    -------
    None
        Displays the surface plot.

    Notes:
    ------
    - Uses MNI152 template for visualization.
    - Requires Nilearn library for image handling and plotting.
    """
    
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
        indices = np.unique(atlas_data[atlas_data>0])
        for reg in range(len(values)):
            reg_mask = atlas_data == indices[reg]
            atlas_new[reg_mask] = values[reg]

        # make image from new atlas data
        new_img = Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # interpolate image
    img_interp = image.resample_img(new_img, mni.affine)

    # plot interpolated image on surface
    fig = plotting.plot_img_on_surf(img_interp, views=['lateral', 'medial'],
                              hemispheres=['left', 'right'],
                              colorbar=True, cmap=cmap, bg_on_data=True,
                              threshold=threshold, symmetric_cbar=symmetric_cbar,
                              surf_mesh='fsaverage5', inflate=False)
    
    return fig

### example usage, AAL atlas (3D)
atlas = datasets.fetch_atlas_aal()
labels = atlas.labels
values = np.random.randn(len(labels))
atlas_surface_plotter(atlas.maps, values)

### example usage, glasser atlas (4D)
atlas_file = '/d/gmi/1/sebastiancoleman/atlases/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz';
values= np.random.randn(52)
atlas_surface_plotter(atlas_file, values)


def atlas_volume_plotter(atlas_file, values, threshold=0, display_mode='ortho', cmap='cold_hot', symmetric_cbar=True):
    """
    Plots values in parcellation, overlaid on MNI brain template.

    Parameters:
    ----------
    atlas_file : str
        Path to the atlas file in Nifti format.
    values : array-like
        List or array of values corresponding to each region in the atlas.
    threshold : float, optional
        Threshold value for displaying values (default is 0).
    display_mode : {'ortho', 'x', 'y', 'z'}, optional
        Specifies the direction of the slices ('ortho' for three views or 'x', 'y', 'z' for tiled single view).
    cmap : str, optional
        Colormap to use for displaying the values (default is 'cold_hot').

    Returns:
    -------
    None
        Displays the atlas volume plot.

    Notes:
    ------
    - Uses MNI152 template for visualization.
    - Requires Nilearn library for image handling and plotting.
    """
    
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
        indices = np.unique(atlas_data[atlas_data>0])
        for reg in range(len(values)):
            reg_mask = atlas_data == indices[reg]
            atlas_new[reg_mask] = values[reg]

        # make image from new atlas data
        new_img = Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # interpolate image
    img_interp = image.resample_img(new_img, mni.affine)
    
    # plot the interpolated image
    plotting.plot_stat_map(img_interp, display_mode=display_mode, threshold=threshold, 
                           symmetric_cbar=symmetric_cbar, annotate=False,
                           draw_cross=False)
    
### example usage, AAL atlas (3D)
atlas = datasets.fetch_atlas_aal()
labels = atlas.labels
values = np.random.randn(len(labels))
atlas_volume_plotter(atlas.maps, values)

### example usage, glasser atlas (4D)
atlas_file = '/d/gmi/1/sebastiancoleman/atlases/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz';
values= np.random.randn(52)
atlas_volume_plotter(atlas_file, values)
