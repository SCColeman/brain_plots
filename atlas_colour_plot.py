# -*- coding: utf-8 -*-
"""
Plot values in parcellated brain regions, copy and paste into script.

@author: Sebastian C. Coleman
"""

from nilearn import plotting, image
from nibabel import nifti1
import numpy as np


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
    
