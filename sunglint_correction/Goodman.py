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
    
class Goodman:
    def __init__(self, im_aligned, NIR_lower = 25, NIR_upper = 37, A = 0.000019, B = 0.1):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param NIR_lower (int): band index which corresponds to 641.93nm, closest band to 640nm
        :param NIR_upper (int): band index which corresponds to 751.49nm, closest band to 750nm
        :param A (float): the values in Goodman et al's paper, using AVIRIS reflectance (rather than radiance) data
        :param B (float): the values in Goodman et al's paper, using AVIRIS reflectance (rather than radiance) data
            see Goodman et al, which corrects each pixel independently. The NIR radiance is subtracted from the radiance at each wavelength,
            but a wavelength-independent offset is also added. 
            it is not clear how A and B were chosen, but an optimization for a case where in situ data is
            available would enable values to be found
        """
        self.im_aligned = im_aligned
        self.NIR_lower = NIR_lower
        self.NIR_upper = NIR_upper
        self.A = A
        self.B = B
        self.rgb_bands = [28,17,7]
        self.n_bands = im_aligned.shape[-1]
        wavelength_dict = preprocessing.bands_wavelengths()
        self.wavelength_dict = {i:wavelength for i,wavelength in enumerate(wavelength_dict)}
        self.crop_bands = [2,-3]

    def get_corrected_bands(self):
        goodman = lambda r,r_640,r_750, A, B: r - r_750 + (A+B*(r_640-r_750))
        corrected_bands = []
        for i in range(self.n_bands):
            R = self.im_aligned[:,:,i]
            R_640 = self.im_aligned[:,:,self.NIR_lower]
            R_750 = self.im_aligned[:,:,self.NIR_upper]
            corrected_band = goodman(R,R_640,R_750,self.A,self.B)
            corrected_bands.append(corrected_band)
        return corrected_bands
    
    def correction_stats(self,bbox):
        """
        :param bbox (tuple): ((x1,y1),(x2,y2)), where x1,y1 is the upper left corner, x2,y2 is the lower right corner
        Show corrected and original rgb image, mean reflectance
        """
        ((x1,y1),(x2,y2)) = bbox
        coord, w, h = bboxes_to_patches(bbox)
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