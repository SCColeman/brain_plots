#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glass brain plot using pyvista.

@author: sebastiancoleman
"""

import numpy as np
from matplotlib import pyplot as plt
from nilearn import image, datasets, plotting, surface
import pyvista as pv
import pandas as pd
from matplotlib import pyplot as plt

def glass_brain_plot(adjacency, atlas_coords, threshold, cbar_label):
    
    import pyvista as pv
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from nilearn import surface
    
    def add_spheres(plotter, points, radius=2.0, color="black"):
        for point in points:
            sphere = pv.Sphere(radius=radius, center=point)
            plotter.add_mesh(sphere, color=color)
    
    def remove_white_space(imdata, decim=None):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        if decim:
            nonwhite_row[::decim] = True
        nonwhite_col = nonwhite_pix.any(0)
        if decim:
            nonwhite_col[::decim] = True
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped
    
    # get upper triangle only
    triu = np.triu_indices(adjacency.shape[0],1)
    mask = np.zeros((adjacency.shape[0],adjacency.shape[0]))
    mask[triu] = 1
    adjacency *= mask
    
    # get indices
    threshold = 0.3
    indices = np.argwhere(np.abs(adjacency) > threshold)
    values = adjacency[np.abs(adjacency) > threshold]
    
    # get coordinates of indices
    lines = []
    for ind in range(indices.shape[0]):
        line = np.array([atlas_coords[indices[ind,0],:], atlas_coords[indices[ind,1],:]])
        lines.append(line)
    lines = np.concatenate(lines, 0)
    
    # get colors
    vmin, vmax = -np.max(np.abs(adjacency[triu])), np.max(np.abs(adjacency[triu]))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdBu_r
    rgb_values = cmap(norm(values))
    
    # get fsaverage meshes
    fsaverage = datasets.fetch_surf_fsaverage()
    lh, rh = surface.load_surf_mesh(fsaverage.pial_left), surface.load_surf_mesh(fsaverage.pial_right)
    
    # lh mesh
    coords, faces = lh.coordinates, lh.faces
    faces_vtk = np.column_stack((np.full(faces.shape[0], 3), faces)).ravel()
    lh_mesh = pv.PolyData(coords, faces_vtk)

    # rh mesh
    coords, faces = rh.coordinates, rh.faces
    faces_vtk = np.column_stack((np.full(faces.shape[0], 3), faces)).ravel()
    rh_mesh = pv.PolyData(coords, faces_vtk)
    
    # plot brain
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(lh_mesh, color="gray", opacity=0.05)
    plotter.add_mesh(rh_mesh, color="gray", opacity=0.05)
    
    # add line
    for l, ll in enumerate(np.arange(0, lines.shape[0], 2)):
        plotter.add_lines(lines[ll:ll+2,:], color=rgb_values[l,:], width=7)
    
    # add coordinates
    add_spheres(plotter, atlas_coords, radius=2.0, color="gray")
    
    # get up view
    plotter.view_xy() # up view
    img_up = plotter.screenshot()
    img_up = remove_white_space(img_up)
    
    # get side view
    plotter.view_yz() # side view
    img_side = plotter.screenshot()
    img_side = remove_white_space(img_side)
    
    # get back view
    plotter.view_xz() # back view
    img_back = plotter.screenshot()
    img_back = remove_white_space(img_back)
    
    # plot 
    fig = plt.figure(figsize=(9,5.5))
    ax1 = fig.add_axes([0.15, 0.2, 0.4, 0.7])  # top-left
    ax2 = fig.add_axes([0.55, 0.55, 0.3, 0.35])  # top-right
    ax3 = fig.add_axes([0.55, 0.2, 0.3, 0.35])  # bottom-left
    cax = fig.add_axes([0.32, 0.12, 0.4, 0.03]) # colorbar ax
    
    # insert images
    ax1.imshow(img_up)
    ax1.axis(False)
    ax2.imshow(img_side)
    ax2.axis(False)
    ax3.imshow(img_back)
    ax3.axis(False)
    
    # add cbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16, labelpad=0)
    
    return fig

#%% example

# load atlas
atlas_file = '/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_4D.nii.gz'
atlas = image.load_img(atlas_file)
atlas_coords = np.load('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_coords.npy')
names = np.squeeze(pd.read_csv('/d/gmi/1/sebastiancoleman/atlases/MEG_atlas_38_regions_names.csv', header=None).to_numpy())

# create adjacency
adjacency = np.zeros((len(names),len(names)))
adjacency[0,:5] = np.arange(5)
adjacency[5,8:13] = np.arange(-5,0)
adjacency[:5,0] = np.arange(5)

# plot
fig = glass_brain_plot(adjacency, atlas_coords, threshold=0.3, cbar_label='Connectivity')
