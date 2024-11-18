import mne
import numpy as np
from matplotlib import pyplot as plt

def make_atlas_nifti(atlas_img, values):

    # some package imports inside function - ignore my bad practice
    from nilearn import image, datasets
    from nibabel import Nifti1Image
    
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()
    atlas_new = np.zeros(np.shape(atlas_data))
    indices = np.unique(atlas_data[atlas_data>0])
    for reg in range(len(values)):
        reg_mask = atlas_data == indices[reg]
        atlas_new[reg_mask] = values[reg]
    new_img = Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    img = image.resample_img(new_img, mni.affine)
    
    return img

def surface_brain_plot(img, subjects_dir, surf='inflated', cmap='cold_hot', symmetric=True, 
                       threshold=0, fade=True, cbar_label=None, figsize=(10,7)):
    
    # some package imports inside function - ignore my bad practice
    from nilearn import surface
    import matplotlib as mpl
    
    # make MNE stc out of nifti
    lh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'lh.pial')
    lh = surface.vol_to_surf(img, lh_surf)
    rh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'rh.pial')
    rh = surface.vol_to_surf(img, rh_surf)
    data = np.hstack([lh, rh])
    vertices = [np.arange(len(lh)), np.arange(len(rh))]
    stc = mne.SourceEstimate(data, vertices, tmin=0, tstep=1)

    # set up axes
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_axes([0, 0.60, 0.35, 0.35])  # top-left
    ax2 = fig.add_axes([0.65, 0.60, 0.35, 0.35])  # top-right
    ax3 = fig.add_axes([0.0, 0.15, 0.35, 0.35])  # bottom-left
    ax4 = fig.add_axes([0.65, 0.15, 0.35, 0.35])  # bottom-right
    ax5 = fig.add_axes([0.32, 0.3, 0.36, 0.5])  # center 
    cax = fig.add_axes([0.25, 0.1, 0.5, 0.03]) # colorbar ax
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor('none')
        ax.axis(False)
        
    # set up threshold
    if symmetric:
        vmax = np.max(np.abs(data))
        vmin = -vmax
        mid = threshold + ((vmax-threshold)/2)
        if fade:
            clim = {'kind': 'value', 'pos_lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'pos_lims':(threshold, threshold, vmax)}
    else:
        vmax = np.max(data)
        vmin = np.min(data)
        mid = threshold + ((vmax-threshold)/3)
        if fade:
            clim = {'kind': 'value', 'lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'lims':(threshold, threshold, vmax)}
        
    if surf=='inflated':
        cortex='low_contrast'
    elif surf=='pial':
        cortex=(0.6, 0.6, 0.6)
    else:
        cortex=(0.6, 0.6, 0.6)
    plot_kwargs = dict(subject='fsaverage',
                       subjects_dir=subjects_dir,
                       surface=surf,
                       cortex=cortex,
                       background='white',
                       colorbar=False,
                       time_label=None,
                       time_viewer=False,
                       transparent=True,
                       clim=clim,
                       colormap=cmap,
                       )
    
    def remove_white_space(imdata):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped

    # top left
    views = ['lat']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax1.imshow(screenshot)

    # top right
    views = ['lat']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax2.imshow(screenshot)

    # bottom left
    views = ['med']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax3.imshow(screenshot)

    # bottom right
    views = ['med']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax4.imshow(screenshot)

    # middle
    views = ['dorsal']
    hemi = 'both'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    background = np.sum(screenshot, -1) == 3*255
    alpha = np.ones(screenshot.shape[:2])  
    alpha[background] = 0
    ax5.imshow(screenshot, alpha=alpha)

    # colorbar
    cmap = plt.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16, labelpad=0)
    
    return fig
