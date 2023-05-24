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
import sunglint_correction.SUGAR as sugar

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
    
class Hedley:
    def __init__(self, im_aligned, bbox, NIR_band = 47):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        """
        self.im_aligned = im_aligned
        self.bbox = bbox
        self.NIR_band = NIR_band
        self.rgb_bands = [28,17,7]
        self.n_bands = im_aligned.shape[-1]
        self.R_min = np.percentile(self.im_aligned[:,:,self.NIR_band].flatten(),5,interpolation='nearest')
        wavelength_dict = preprocessing.bands_wavelengths()
        self.wavelength_dict = {i:wavelength for i,wavelength in enumerate(wavelength_dict)}
        self.crop_bands = [2,-3]
    
    def covariance_NIR(self,NIR,b):
        """
        NIR & b are vectors
        reflectance for band i
        """
        n = len(NIR)
        pij = np.dot(NIR,b)/n - np.sum(NIR)/n*np.sum(b)/n
        pjj = np.dot(NIR,NIR)/n - (np.sum(NIR)/n)**2
        return pij/pjj
    
    def correlation_bands_reflectance(self):
        """
        calculate correlation between NIR and other bands for reflectance
        NIR_band is 750 nm
        """
        ((x1,y1),(x2,y2)) = self.bbox
        reflectance_bands = [self.im_aligned[y1:y2,x1:x2,v].flatten() for v in range(self.n_bands)] #flattened images
        NIR_reflectance = reflectance_bands[self.NIR_band] #flattened images
        return [self.covariance_NIR(NIR_reflectance,v) for v in reflectance_bands]
    
    def get_corrected_bands(self):
        """
        correction is done in reflectance
        mode (str): pearson_corr,least_sq, covariance
        NIR_ref (str): mean, min
        """
        corr = self.correlation_bands_reflectance()
        NIR_reflectance = self.im_aligned[:,:,self.NIR_band]

        hedley_c = lambda r,b,NIR,R_min: r - b*(NIR-R_min)

        corrected_bands = []
        for band_number in range(self.n_bands): #iterate across bands
            b = corr[band_number]
            R = self.im_aligned[:,:,band_number]
            corrected_band = hedley_c(R,b,NIR_reflectance,self.R_min)
            corrected_bands.append(corrected_band)
            
        return corrected_bands
    
    def correction_stats(self):
        """
        :param bbox (tuple): ((x1,y1),(x2,y2)), where x1,y1 is the upper left corner, x2,y2 is the lower right corner
        Show corrected and original rgb image, mean reflectance
        """
        ((x1,y1),(x2,y2)) = self.bbox
        coord, w, h = bboxes_to_patches(self.bbox)
        corrected_bands = self.get_corrected_bands()
        corrected_bands = np.stack(corrected_bands,axis=2)

        fig, axes = plt.subplots(2,4,figsize=(14,10))
        for im, title, ax in zip([self.im_aligned,corrected_bands],['Original RGB','Corrected RGB'],axes[0,:2]):
            ax.imshow(np.take(im,self.rgb_bands,axis=2))
            ax.set_title(title)
            ax.axis('off')
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        for im,label in zip([self.im_aligned,corrected_bands],[r'$R_T$',r'$R_T\prime$']):
            avg_reflectance = [np.mean(im[y1:y2,x1:x2,band_number]) for band_number in range(self.im_aligned.shape[-1])]
            x = list(self.wavelength_dict.values())
            y = [avg_reflectance[i] for i in list(self.wavelength_dict)]
            axes[0,2].plot(x[self.crop_bands[0]:self.crop_bands[1]],y[self.crop_bands[0]:self.crop_bands[1]],label=label)

        axes[0,2].set_xlabel('Wavelengths (nm)')
        axes[0,2].set_ylabel('Reflectance')
        axes[0,2].legend(loc='upper right')
        axes[0,2].set_title(r'$R_T(\lambda)$ and $R_T(\lambda)\prime$')

        residual = self.im_aligned - corrected_bands
        residual = np.mean(residual[y1:y2,x1:x2,:],axis=(0,1))
        x = list(self.wavelength_dict.values())
        y = [residual[i] for i in list(self.wavelength_dict)]
        axes[0,3].plot(x[self.crop_bands[0]:self.crop_bands[1]],y[self.crop_bands[0]:self.crop_bands[1]])
        axes[0,3].set_xlabel('Wavelengths (nm)')
        axes[0,3].set_ylabel('Reflectance difference')
        axes[0,3].set_title(r'$R_T(\lambda) - R_T(\lambda)\prime$')
        
        h = y2 - y1
        w = x2 - x1

        for i, (im, title, ax) in enumerate(zip([self.im_aligned,corrected_bands],[r'$R_T$',r'$R_T\prime$'],axes[1,:2])):
            rgb_cropped = np.take(im[y1:y2,x1:x2,:],self.rgb_bands,axis=2)
            ax.imshow(rgb_cropped)
            ax.set_title(title)
            # ax.axis('off')
            ax.plot([0,w],[h//2,h//2],color="red",linewidth=3,alpha=0.5)
            for j,c in enumerate(['r','g','b']):
                axes[1,i+2].plot(list(range(w)),rgb_cropped[h//2,:,j],c=c,alpha=0.5,label=c)
            axes[1,i+2].set_xlabel('Width of image')
            axes[1,i+2].set_ylabel('Reflectance')
            axes[1,i+2].legend(loc="upper right")
            axes[1,i+2].set_title(f'{title} along red line')
        
        y1,y2 = axes[1,2].get_ylim()
        axes[1,3].set_ylim(y1,y2)
        plt.tight_layout()
        plt.show()

        return corrected_bands
    
def compare_correction_algo(im_aligned,bbox,corrected_Hedley = None, corrected_SUGAR = None, iter=3):
    """
    :param im_aligned (np.ndarray): reflectance image
    :param bbox (tuple): bbox over glint area for Hedley algorithm
    :param iter (int): number of iterations for SUGAR algorithm
    compare SUGAR and Hedley algorithm
    """
    if corrected_Hedley is None:
        HH = Hedley(im_aligned,bbox)
        corrected_Hedley = HH.get_corrected_bands()
        corrected_Hedley = np.stack(corrected_Hedley,axis=2)

    if corrected_SUGAR is None:
        corrected_bands = sugar.correction_iterative(im_aligned,iter=iter,bounds = [(1,2)],estimate_background=True,get_glint_mask=False,plot=False)
        corrected_SUGAR = corrected_bands[-1]

    rgb_bands = [28,17,7]
    fig, axes = plt.subplots(1,3,figsize=(12,5))
    im_list = [im_aligned,corrected_Hedley,corrected_SUGAR]
    title_list = ['Original','Hedley',f'SUGAR (iters: {iter})']
    for im, title, ax in zip(im_list,title_list,axes.flatten()):
        ax.imshow(np.take(im,rgb_bands,axis=2))
        ax.set_title(title + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
        ax.axis('off')

    coord, w, h = bboxes_to_patches(bbox)
    rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    plt.show()
    return