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
    
class Kutser:
    def __init__(self, im_aligned, bbox, oxy_band = 38,lower_oxy = 36, upper_oxy = 49, NIR_band = 47):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param bbox (tuple): ((x1,y1),(x2,y2)), where x1,y1 is the upper left corner, x2,y2 is the lower right corner, 
            which contains the glint region in deep water
        :param oxy_band (int): band index for oxygen absorption band, which corresponds to 760.6nm
        :param oxy_band (int): band index for outside oxygen absorption band, which corresponds to 742.39nm
        :param oxy_band (int): band index for outside oxygen absorption band, which corresponds to 860.48nm
            see Kutser, Vahtmäe and Praks
        """
        self.im_aligned = im_aligned
        self.bbox = bbox
        self.oxy_band = oxy_band
        self.lower_oxy = lower_oxy
        self.upper_oxy = upper_oxy
        self.NIR_band = NIR_band
        self.rgb_bands = [28,17,7]
        self.n_bands = im_aligned.shape[-1]
        self.R_min = np.percentile(self.im_aligned[:,:,self.NIR_band].flatten(),5,interpolation='nearest')
        wavelength_dict = preprocessing.bands_wavelengths()
        self.wavelength_dict = {i:wavelength for i,wavelength in enumerate(wavelength_dict)}
        self.crop_bands = [2,-3]
    
    def get_depth_D(self):
        """
        Assume the amount of glint is proportional to the depth of the oxygen absorption feature, D
        returns the normalised D by dividing it by the maximum D found in a deep water region
        """
        ((x1,y1),(x2,y2)) = self.bbox
        D = (self.im_aligned[:,:,self.lower_oxy] + self.im_aligned[:,:,self.upper_oxy])/2 - self.im_aligned[:,:,self.oxy_band]
        D_max = D[y1:y2,x1:x2].max() # assumed to be the maximum glint value
        return D/D_max
    
    def get_glint_G(self):
        """
        The spectral variation of glint G is found by subtracting the spectrum at the darkest (ie. lowest D) NIR deep-water pixel from the brightest
        returns G as a function of wavelength
        """
        ((x1,y1),(x2,y2)) = self.bbox

        G_list = []
        for i in range(self.n_bands):
            im = self.im_aligned[y1:y2,x1:x2,i]
            G = im.max() - im.min()
            G_list.append(G)
        return G_list
    
    def get_corrected_bands(self):
        """
        correction is done in reflectance
        """
        kutser = lambda r,g,d: r - g*d
        g_list = self.get_glint_G()
        D = self.get_depth_D()

        corrected_bands = []
        for band_number in range(self.n_bands): #iterate across bands
            G = g_list[band_number]
            R = self.im_aligned[:,:,band_number]
            corrected_band = kutser(R,G,D)
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
            ax.plot([0,w-1],[h//2,h//2],color="red",linewidth=3,alpha=0.5)
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
