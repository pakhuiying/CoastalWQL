import cv2
# import micasense.plotutils as plotutils
import os, glob
import json
import tqdm
# import pickle #This library will maintain the format as well
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gaussian_kde
from scipy import ndimage
from math import ceil
import preprocessing
import sunglint_correction.Hedley as Hedley
import sunglint_correction.SUGAR as sugar
import sunglint_correction.Kutser as Kutser
import sunglint_correction.Goodman as Goodman

def bboxes_to_patches(bboxes):
    if bboxes is not None:
        ((x1,y1),(x2,y2)) = bboxes
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        h = y1 - y2 # negative height as the origin is on the top left
        w = x2 - x1
        return (x1,y2), w, h
    else:
        return None
    
def compare_plots(im_list, title_list, bbox=None, save_dir = None):
    """
    :param im_list (list of np.ndarray): where the first item is always the original image
    :param title_list (list of str): the title for the first row
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    :param save_dir (str): Full filepath required. if None, no figure is saved.
    """
    rgb_bands = [28,17,7]
    crop_bands = [2,-3]
    wavelength_dict = preprocessing.bands_wavelengths()
    wavelength_dict = {i:wavelength for i,wavelength in enumerate(wavelength_dict)}
    nrow, ncol, n_bands = im_list[0].shape

    plot_width = len(im_list)*3
    if bbox is not None:
        ((x1,y1),(x2,y2)) = bbox
        coord, w, h = bboxes_to_patches(bbox)
        plot_height = 12
        plot_row = 4
    else:
        plot_height = 9
        plot_row = 3

    fig, axes = plt.subplots(plot_row,len(im_list),figsize=(plot_width,plot_height))
    if bbox is not None:
        og_avg_reflectance = [np.mean(im_list[0][y1:y2,x1:x2,band_number]) for band_number in range(n_bands)]
    else:
        og_avg_reflectance = [np.mean(im_list[0][:,:,band_number]) for band_number in range(n_bands)]
    x = list(wavelength_dict.values())
    og_y = [og_avg_reflectance[i] for i in list(wavelength_dict)]

    for i,(im, title) in enumerate(zip(im_list,title_list)): #iterate acoss column
        # plot image
        rgb_im = np.take(im,rgb_bands,axis=2)
        axes[0,i].imshow(rgb_im)
        axes[0,i].set_title(title + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
        axes[0,i].axis('off')
        if bbox is not None:
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
            axes[0,i].add_patch(rect)
        # plot original reflectance
        axes[1,i].plot(x[crop_bands[0]:crop_bands[1]],og_y[crop_bands[0]:crop_bands[1]],label=r'$R_T(\lambda)$')
        # plot corrected reflectance
        if i > 0:
            if bbox is not None:
                avg_reflectance = [np.mean(im[y1:y2,x1:x2,band_number]) for band_number in range(n_bands)]
            else:
                avg_reflectance = [np.mean(im[:,:,band_number]) for band_number in range(n_bands)]
            y = [avg_reflectance[i] for i in list(wavelength_dict)]
            axes[1,i].plot(x[crop_bands[0]:crop_bands[1]],y[crop_bands[0]:crop_bands[1]],label=r'$R_T(\lambda)\prime$')
        axes[1,i].legend(loc="upper right")
        axes[1,i].set_title(r'$R_T(\lambda)\prime$'+' in AOI')
        axes[1,i].set_xlabel('Wavelengths (nm)')
        axes[1,i].set_ylabel('Reflectance')

        # plot cropped rgb
        rgb_cropped = rgb_im[y1:y2,x1:x2,:] if bbox is not None else rgb_im
        if bbox is not None:
            axes[2,i].imshow(rgb_cropped)
            axes[2,i].set_title('AOI')
            axes[2,i].plot([0,w-1],[abs(h)//2,abs(h)//2],color="red",linewidth=3,alpha=0.5)
        
        row_idx = 3 if bbox is not None else 2
        h = nrow if bbox is None else h
        w = ncol if bbox is None else w
        # plot reflectance along red line
        for j,c in enumerate(['r','g','b']):
            axes[row_idx,i].plot(list(range(w)),rgb_cropped[h//2,:,j],c=c,alpha=0.5,label=c)
        axes[row_idx,i].set_xlabel('Image position')
        axes[row_idx,i].set_ylabel('Reflectance')
        axes[row_idx,i].legend(loc="upper right")
        axes[row_idx,i].set_title(r'$R_T(\lambda)\prime$'+' along red line')
    
    y1,y2 = axes[row_idx,0].get_ylim()
    for i in range(len(im_list)):
        axes[row_idx,i].set_ylim(y1,y2)
    
    plt.tight_layout()
    

    if save_dir is not None:
        fig.savefig('{}.png'.format(save_dir))
        plt.close()
    else:
        plt.show()
    return

def compare_sugar_algo(im_aligned,bbox=None,corrected = None, corrected_background = None, iter=3, bounds=[(1,2)], save_dir = None):
    """
    :param im_aligned (np.ndarray): reflectance image
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    :param corrected (np.ndarray): corrected for glint using SUGAR without taking into account of background
    :param corrected_background (np.ndarray): corrected for glint using SUGAR taking into account of background
    :param iter (int): number of iterations for SUGAR algorithm
    :param save_dir (str): Full filepath required. if None, no figure is saved.
    compare SUGAR algorithm, whether to take into account of background spectra
    returns a tuple (corrected, corrected_background)
    """
    if corrected is None:
        corrected = sugar.correction_iterative(im_aligned, iter=iter, bounds = bounds,estimate_background=False,get_glint_mask=False)
    if corrected_background is None:
        corrected_background = sugar.correction_iterative(im_aligned, iter=iter, bounds = bounds,estimate_background=True,get_glint_mask=False)

    if isinstance(corrected,list):
        corrected = corrected[-1]
    if isinstance(corrected_background,list):
        corrected_background = corrected_background[-1]
    
    title_list = [r'$R_T$',r'$R_T\prime$',r'$R_{T,BG}\prime$']
    im_list = [im_aligned,corrected,corrected_background]
    compare_plots(im_list, title_list, bbox, save_dir)
    return (corrected,corrected_background)

def compare_correction_algo(im_aligned,bbox,corrected_Hedley = None, corrected_Kutser = None, corrected_Goodman = None, corrected_SUGAR = None,  iter=3, save_dir=None):
    """
    :param im_aligned (np.ndarray): reflectance image
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    :param iter (int): number of iterations for SUGAR algorithm
    compare SUGAR and Hedley algorithm
    """
    if corrected_Hedley is None:
        HH = Hedley.Hedley(im_aligned,bbox)
        corrected_Hedley = HH.get_corrected_bands()
        corrected_Hedley = np.stack(corrected_Hedley,axis=2)

    if corrected_SUGAR is None:
        corrected_SUGAR = sugar.correction_iterative(im_aligned,iter=iter,bounds = [(1,2)],estimate_background=True,get_glint_mask=False,plot=False)

    if corrected_Kutser is None:
        KK = Kutser.Kutser(im_aligned,bbox)
        corrected_Kutser = KK.get_corrected_bands()
        corrected_Kutser = np.stack(corrected_Kutser,axis=2)
    
    if corrected_Goodman is None:
        GM = Goodman.Goodman(im_aligned)
        corrected_Goodman = GM.get_corrected_bands()
        corrected_Goodman = np.stack(corrected_Goodman,axis=2)

    if isinstance(corrected_SUGAR,list):
        im_list = [im_aligned,corrected_Hedley,corrected_Kutser,corrected_Goodman,corrected_SUGAR[-1]]
    elif isinstance(corrected_SUGAR, np.ndarray):
        im_list = [im_aligned,corrected_Hedley,corrected_Kutser,corrected_Goodman,corrected_SUGAR]

    title_list = ['Original','Hedley','Kutser','Goodman',f'SUGAR (iters: {iter})']

    compare_plots(im_list, title_list, bbox, save_dir = save_dir)
    return