#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fancy surface plots using surfplot

@author: sebastiancoleman
"""

from neuromaps.datasets import fetch_fslr
from neuromaps import transforms
from surfplot import Plot, utils
import nibabel as nib
from matplotlib import pyplot as plt
import matplotlib as mpl

def atlas_surface_plotter2(atlas_file, values, threshold=0, cmap='YlOrRd_r'):
    
    # make volume image of strength  
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

    # save temporary image
    nib.save(new_img, '/d/gmi/1/sebastiancoleman/fancy_plot_dir/fig.nii.gz')

    # transform to fsaverage surface
    fslr = transforms.mni152_to_fslr('/d/gmi/1/sebastiancoleman/fancy_plot_dir/fig.nii.gz')
    lh_data, rh_data = fslr[0].agg_data(), fslr[1].agg_data()
    lh_data = utils.threshold(lh_data, threshold, two_sided=True)
    rh_data = utils.threshold(rh_data, threshold, two_sided=True)

    # plot
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(surf_lh=lh, surf_rh=rh)
    vmax = np.max(np.abs(values))
    p.add_layer({'left': lh_data, 'right': rh_data}, color_range=[-vmax, vmax], cmap=cmap, cbar=False)
    fig = p.build()
    fig.show()
    
    return fig


def construct_mpl_surf_image(surf_fig, values, figsize=(5,4), cmap='YlOrRd_r', symmetric=True, cbar_label='Brain Activity'):

    # extract image data
    canvas = surf_fig.canvas
    canvas.draw()
    imflat = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    imdata = imflat.reshape(*reversed(canvas.get_width_height()), 4)
    imdata = imdata[:, :, :3]
    plt.close()
    
    # remove white space
    nonwhite_pix = (imdata != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_row[::4] = True
    nonwhite_col = nonwhite_pix.any(0)
    nonwhite_col[::4] = True
    imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
    
    # construct plot
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]
    cax = fig.add_axes([0.25, 0.15, 0.5, 0.03])  # [left, bottom, width, height]
    
    # brain
    ax.imshow(imdata_cropped)
    ax.axis("off")
    
    # colorbar
    if symmetric:
        vmax = np.max(np.abs(values))
        vmin = -vmax
    else:
        vmax = np.max(values)
        vmin = np.min(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    
    return fig, ax, cax

# example usage, glasser atlas (4D)
atlas_file = '/d/gmi/1/sebastiancoleman/atlases/Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz';
values= np.random.randn(52)

fig = atlas_surface_plotter2(atlas_file, values, threshold=0.5, cmap='cold_hot')
fig, ax, cax = construct_mpl_surf_image(fig, values, figsize=(6,5), 
                                        cmap='cold_hot', symmetric=True, 
                                        cbar_label='Random Values')
