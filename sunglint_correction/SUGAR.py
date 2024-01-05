import cv2
# import micasense.plotutils as plotutils
import os, glob
import json
from tqdm import tqdm
# import pickle #This library will maintain the format as well
# import sunglint_correction.mutils as mutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gaussian_kde
from scipy import ndimage
from scipy.optimize import minimize_scalar
from math import ceil
import preprocessing
import sunglint_correction.Hedley as Hedley
import sunglint_correction.Goodman as Goodman
import sunglint_correction.Kutser as Kutser

# SUn-Glint-Aware Restoration (SUGAR):A sweet and simple algorithm for correcting sunglint
class SUGAR:
    def __init__(self, im_aligned,bounds=[(1,2)],sigma=1,estimate_background=True, glint_mask_method="cdf"):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param bounds (a list of tuple): lower and upper bound for optimisation of b for each band
        :param sigma (float): smoothing sigma for LoG
        :param estimate_background (bool): whether to estimate background spectra using median filtering
        :param glint_mask_method (str): choose either "cdf" or "otsu", "cdf" is set as the default
        """
        self.im_aligned = im_aligned
        self.sigma = sigma
        self.estimate_background = estimate_background
        # import wavelengths for each band
        wavelength_dict = preprocessing.bands_wavelengths()
        self.wavelength_dict = {i:wavelength for i,wavelength in enumerate(wavelength_dict)}
        self.n_bands = im_aligned.shape[-1]
        self.bounds = bounds*self.n_bands
        self.rgb_bands = [28,17,7]
        # self.NIR_band = 47 # 842.36nm
        # self.R_min = np.percentile(self.im_aligned[:,:,self.NIR_band].flatten(),5,interpolation='nearest')
        self.glint_mask_method = glint_mask_method

    def otsu_thresholding(self,im):
        """
        :param im (np.ndarray) of shape mxn. Note that it is the LoG of image
        otsu thresholding with Brent's minimisation of a univariate function
        returns the value of the threshold for input
        """
        # count,bin,_ = plt.hist(im.flatten(),bins='auto')
        auto_bins = int(0.005*im.shape[0]*im.shape[1])
        count,bin,_ = plt.hist(im.flatten(),bins=auto_bins)
        plt.close()
        
        hist_norm = count/count.sum() #normalised histogram
        Q = hist_norm.cumsum() # CDF function ranges from 0 to 1
        N = count.shape[0]
        N_negative = bin[bin<0].shape[0]
        bins = np.arange(N)
        
        def otsu_thresh(x):
            x = int(x)
            p1,p2 = np.hsplit(hist_norm,[x]) # probabilities
            q1,q2 = Q[x],Q[N-1]-Q[x] # cum sum of classes
            b1,b2 = np.hsplit(bins,[x]) # weights
            # finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
            # calculates the minimization function
            fn = v1*q1 + v2*q2
            return fn
        
        # brent method is used to minimise an univariate function
        # bounded minimisation
        # res = minimize_scalar(otsu_thresh, bounds=(1, N), method='bounded')
        # we can just limit the search to negative values since we know thresh should be negative as L<0 for glint pixels
        res = minimize_scalar(otsu_thresh, bounds=(1, N_negative), method='bounded')
        thresh = bin[int(res.x)]
        
        return thresh
    
    # def cdf_thresholding(self,im, percentile=0.05):
    #     """
    #     :param im (np.ndarray) of shape mxn
    #     :param percentile (float): lower and upper percentile values are potential glint pixels
    #     """
    #     lower_perc = percentile
    #     upper_perc = 1-percentile
    #     im_flatten = im.flatten()
    #     H,X1 = np.histogram(im_flatten, bins = int(0.005*im.shape[0]*im.shape[1]), density=True )
    #     dx = X1[1] - X1[0]
    #     F1 = np.cumsum(H)*dx
    #     F_lower = X1[1:][F1<lower_perc]
    #     F_upper = X1[1:][F1>upper_perc]
    #     while((F_lower.size == 0) or (F_upper.size == 0)):
    #         if (F_lower.size == 0):
    #             lower_perc += 0.01
    #             F_lower = X1[1:][F1<lower_perc]
    #         if (F_upper.size == 0):
    #             upper_perc -= 0.01
    #             F_upper = X1[1:][F1>upper_perc]

    #     lower_thresh = F_lower[-1]
    #     upper_thresh = F_upper[0]

    #     return lower_thresh,upper_thresh

    def cdf_thresholding(self,im,auto_bins=10):
        """
        :param im (np.ndarray) of shape mxn. Note that it is the LoG of image
        :param percentile (float): lower and upper percentile values are potential glint pixels
        """
        count,bin,_ = plt.hist(im.flatten(),bins=auto_bins)
        plt.close()
        thresh = bin[np.argmax(count)]
        return thresh

    def glint_list(self):
        """
        returns a list of np.ndarray, where each item is an extracted glint for each band based on get_glint_mask
        """
        glint_mask = self.glint_mask_list()
        extracted_glint_list = []
        for i in range(self.im_aligned.shape[-1]):
            gm = glint_mask[i]
            extracted_glint = gm*self.im_aligned[:,:,i]
            extracted_glint_list.append(extracted_glint)

        return extracted_glint_list
    
    def glint_mask_list(self):
        """
        get glint mask using laplacian of gaussian image. 
        returns a list of np.ndarray
        """
        # fig, axes = plt.subplots(10,2,figsize=(8,20))
        glint_mask_list = []
        # glint_threshold = []
        for i in range(self.im_aligned.shape[-1]):
            glint_mask = self.get_glint_mask(self.im_aligned[:,:,i])
            glint_mask_list.append(glint_mask)

        return glint_mask_list
    
    def get_glint_mask(self,im):
        """
        get glint mask using laplacian of gaussian image. 
        We assume that water constituents and features follow a smooth continuum, 
        but glint pixels vary a lot spatially and in intensities
        Note that for very extensive glint, this method may not work as well <--:TODO use U-net to identify glint mask
        returns a np.ndarray
        """
        # find the laplacian of gaussian first
        # take the absolute value of laplacian because the sign doesnt really matter, we want all edges
        # im_smooth = np.abs(ndimage.gaussian_laplace(im_copy,sigma=self.sigma))
        # im_smooth = im_smooth/np.max(im_smooth)
        LoG_im = ndimage.gaussian_laplace(im,sigma=self.sigma)
        #threshold mask
        if (self.glint_mask_method == "otsu"):
            thresh = self.otsu_thresholding(LoG_im)
            # glint_threshold.append(thresh)
            # glint_mask = np.where(LoG_im<thresh,1,0)
        elif (self.glint_mask_method == "cdf"):
            thresh = self.cdf_thresholding(LoG_im)
            # lower_thresh,upper_thresh = self.cdf_thresholding(LoG_im)
            # glint_mask = np.where((LoG_im>lower_thresh) & (LoG_im<upper_thresh),0,1)
        else:
            raise ValueError('Enter only cdf or otsu as glint_mask_method')
        # glint_mask = np.where((LoG_im>thresh) & (LoG_im<-thresh),0,1)
        glint_mask = np.where(LoG_im<thresh,1,0)
        return glint_mask
    
    def get_est_background(self, im,k_size=5):
        """ 
        :param im (np.ndarray): image of a band
        estimate background spectra
        returns a np.ndarray
        """
        # median_filtered_list = [im]
        # for i in [5,10,20,30]:
        #     y = ndimage.median_filter(im, size=i)
        #     median_filtered_list.append(y)

        # median_filtered_list = np.stack(median_filtered_list,axis=2)
        # return np.amin(median_filtered_list,axis=2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size,k_size))
        dst = cv2.erode(im, kernel)
            
        return dst

    def optimise_correction_by_band(self,im,glint_mask,R_BG,bounds):
        """
        :param im (np.ndarray): image of a band
        :param glint_mask (np.ndarray): glint mask, where glint area is 1 and non-glint area is 0
        use brent method to get the optimimum b which minimises the variation (i.e. variance) in the entire image
        returns regression slope b
        """
        hedley_c = lambda r,g,b,R_min: r - g*(r/b-R_min)
        # where R_min is the background spectra
        def optimise_b(b):
            G = glint_mask
            R = im
            # R_min = np.percentile(R,5,interpolation='nearest')
            # im_corrected = hedley_c(R,G,b,R_min)
            im_corrected = hedley_c(R,G,b,R_BG)
            # add an additional constraint to ensure global smoothness of image
            # using variance of laplacian
            # var_laplace = laplace(im_corrected).var()
            return np.var(im_corrected)#*var_laplace

        res = minimize_scalar(optimise_b, bounds=bounds, method='bounded')
        return res.x
    
    def divide_and_conquer(self):
        """
        instead of computing b_list for each window, use the previous b_list to narrow the bounds, 
        because of the strong spatial autocorrelation, we know that the b (correction magnitude) cannot diff too much
        this can optimise the run time
        """
        

    def optimise_correction(self):
        """ 
        returns a list of slope in band order i.e. 0,1,2,3,4,5,6,7,8,9 through optimisation
        """
        # glint_mask = self.get_glint_mask(plot=False)
        b_list = []
        glint_mask_list = []
        est_background_list = []
        for i in range(self.n_bands):
            glint_mask = self.get_glint_mask(self.im_aligned[:,:,i])
            glint_mask_list.append(glint_mask)
            if self.estimate_background is True:
                est_background = self.get_est_background(self.im_aligned[:,:,i])
                est_background_list.append(est_background)
            else:
                est_background = self.R_min
                est_background_list.append(est_background)
            bounds = self.bounds[i]
            b = self.optimise_correction_by_band(self.im_aligned[:,:,i],glint_mask,est_background,bounds)
            b_list.append(b)
        
        # add attributes
        self.b_list = b_list
        self.glint_mask = glint_mask_list
        self.est_background = est_background_list

        return b_list, glint_mask_list, est_background_list
    
    def get_corrected_bands(self, plot = False):
        corrected_bands = []
        for i in range(self.n_bands):
            im_copy = self.im_aligned[:,:,i].copy()
            glint_mask = self.get_glint_mask(self.im_aligned[:,:,i])
            background = self.get_est_background(self.im_aligned[:,:,i],k_size=5)
            im_copy[glint_mask == 1] = background[glint_mask == 1]
            corrected_bands.append(im_copy)

        if (plot is True):
            corrected_rgb = np.stack(corrected_bands,axis=2)
            corrected_rgb = np.take(corrected_rgb,self.rgb_bands,axis=2)
            plt.figure(figsize=(15,5))
            plt.imshow(corrected_rgb)
            plt.show()

        return corrected_bands
    # def get_corrected_bands(self,plot=False):
    #     """
    #     :param glint_normalisation (bool): whether to normalise/contrast stretch extracted glint by using histogram normalisation
    #     :param optimise_b (bool): whether to optimise b instead of calculating for b (default is True)
    #     use regression slopes to correct each bands
    #     returns a list of corrected bands in band order i.e. 0,1,2,3,4,5,6,7,8,9
    #     """
        
    #     b_list, glint_mask_list, est_background_list = self.optimise_correction()
        
    #     # where r=reflectance at pixel_i,
    #     # g= glint mask at pixel_i
    #     # b = regression slope
    #     # R_min = 5th percentile of Rmin(NIR)
    #     hedley_c = lambda r,g,b,R_min: r - g*(r/b-R_min)
    #     # hedley_c = lambda r,g,b: r - g*(r/b)

    #     corrected_bands = []
    #     # avg_reflectance = []
    #     # avg_reflectance_corrected = []

    #     # fig, axes = plt.subplots(self.n_bands,2,figsize=(10,20))
    #     for band_number in range(self.n_bands):
    #         b = b_list[band_number]
    #         R = self.im_aligned[:,:,band_number]
    #         G = glint_mask_list[band_number]
    #         R_BG = est_background_list[band_number]
    #         # corrected_band = hedley_c(R,G,b,self.R_min)
    #         # estimation of background spectra is important, 
    #         # and uncertainties is largely attribution to the estimation of background spectra
            
    #         corrected_band = hedley_c(R,G,b,R_BG)
    #         # corrected_band = hedley_c(R,G,b)
    #         corrected_bands.append(corrected_band)
        
    #     if plot is True:
    #         fig, axes = plt.subplots(self.n_bands,2,figsize=(10,20))
    #         for band_number in range(self.n_bands):
    #             axes[band_number,0].imshow(self.im_aligned[:,:,band_number],vmin=0,vmax=1)
    #             axes[band_number,1].imshow(corrected_bands[band_number],vmin=0,vmax=1)
    #             axes[band_number,0].set_title(f'Band {self.wavelength_dict[band_number]} reflectance')
    #             axes[band_number,1].set_title(f'Band {self.wavelength_dict[band_number]} reflectance corrected')
        
    #         for ax in axes.flatten():
    #             ax.axis('off')
    #         plt.tight_layout()
    #         plt.show()

    #     return corrected_bands


    def correction_stats(self,bbox):
        """
        :param bbox (tuple): ((x1,y1),(x2,y2)), where x1,y1 is the upper left corner, x2,y2 is the lower right corner
        Show corrected and original rgb image, mean reflectance
        """
        ((x1,y1),(x2,y2)) = bbox
        coord, w, h = bboxes_to_patches(bbox)
        corrected_bands = self.get_corrected_bands(plot=False)
        corrected_bands = np.stack(corrected_bands,axis=2)
        rgb_bands = [2,1,0]

        fig, axes = plt.subplots(2,4,figsize=(14,10))
        for im, title, ax in zip([self.im_aligned,corrected_bands],['Original RGB','Corrected RGB'],axes[0,:2]):
            ax.imshow(np.take(im,rgb_bands,axis=2))
            ax.set_title(title)
            ax.axis('off')
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        for im,label in zip([self.im_aligned,corrected_bands],[r'$R_T$',r'$R_T\prime$']):
            avg_reflectance = [np.mean(im[y1:y2,x1:x2,band_number]) for band_number in range(self.im_aligned.shape[-1])]
            axes[0,2].plot(list(self.wavelength_dict.values()),[avg_reflectance[i] for i in list(self.wavelength_dict)],label=label)

        axes[0,2].set_xlabel('Wavelengths (nm)')
        axes[0,2].set_ylabel('Reflectance')
        axes[0,2].legend(loc='upper right')
        axes[0,2].set_title(r'$R_T(\lambda)$ and $R_T(\lambda)\prime$')

        residual = self.im_aligned - corrected_bands
        residual = np.mean(residual[y1:y2,x1:x2,:],axis=(0,1))
        axes[0,3].plot(list(self.wavelength_dict.values()),[residual[i] for i in list(self.wavelength_dict)])
        axes[0,3].set_xlabel('Wavelengths (nm)')
        axes[0,3].set_ylabel('Reflectance difference')
        axes[0,3].set_title(r'$R_T(\lambda) - R_T(\lambda)\prime$')
        
        h = y2 - y1
        w = x2 - x1

        for i, (im, title, ax) in enumerate(zip([self.im_aligned,corrected_bands],[r'$R_T$',r'$R_T\prime$'],axes[1,:2])):
            rgb_cropped = np.take(im[y1:y2,x1:x2,:],rgb_bands,axis=2)
            ax.imshow(rgb_cropped)
            ax.set_title(title)
            ax.axis('off')
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

        return
    
    def compare_image(self,save_dir=None, filename = None, plot = True):
        """
        :param save_dir (str): specify directory to store image in. If it is None, no image is saved
        :param filename (str): filename of image u want to save e.g. 'D:\\EPMC_flight\\10thSur24Aug\\F2\\RawImg\\IMG_0192_1.tif'
        returns a figure where left figure is original image, and right figure is corrected image
        """
        corrected_bands = self.get_corrected_bands(plot=False)
        corrected_im = np.stack([corrected_bands[i] for i in [2,1,0]],axis=2)
        original_im = np.take(self.im_aligned,[2,1,0],axis=2)
        fig, axes = plt.subplots(1,2,figsize=(12,7))
        axes[0].imshow(original_im)
        axes[0].set_title('Original Image')
        axes[1].imshow(corrected_im)
        axes[1].set_title('Corrected Image')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()

        if save_dir is not None:
            save_dir = os.path.join(save_dir,"corrected_images")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            filename = get_all_dir(filename,iter=4)
            filename = os.path.splitext(filename)[0]
            full_fn = os.path.join(save_dir,filename)

            fig.suptitle(filename)
            fig.savefig('{}.png'.format(full_fn))

        if plot is True:
            plt.show()
        else:
            plt.close()
        return

def correction_iterative(im_aligned,iter=3,bounds = [(1,2)],estimate_background=True,glint_mask_method="cdf",get_glint_mask=False,plot=False,save_fp=None, termination_thresh = 20):
    """
    :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
    :param iter (int or None): number of iterations to run the sugar algorithm. If None, termination conditions are automatically applied
    :param bounds (list of tuples): to limit correction magnitude
    :param get_glint_mask (np.ndarray): 
    :param save_fp (str): full filepath of filename
    conducts iterative correction using SUGAR
    """
    glint_image = im_aligned.copy()
    corrected_images = []

    if iter is None:
        # termination conditions
        relative_difference = lambda sd0,sd1: sd1/sd0*100
        marginal_difference = lambda sd1,sd2: (sd1-sd2)/sd1*100
        relative_diff_thresh = marginal_difference_thresh = termination_thresh
        corrected_images = []
        sd_og = np.var(im_aligned)
        iter_count = 0
        sd_next = sd_og.copy() #keep track of the sd the iteration before
        while ((relative_difference(sd_og,sd_next) > relative_diff_thresh)):
            # do all the processing here
            HM = SUGAR(glint_image,bounds,estimate_background=estimate_background, glint_mask_method=glint_mask_method)
            corrected_bands = HM.get_corrected_bands()
            glint_image = np.stack(corrected_bands,axis=2)
            sd_temp = np.var(glint_image)
            corrected_images.append(glint_image)
            # save glint_mask
            # if iter_count == 0 and get_glint_mask is True:
            #     glint_mask = np.stack(HM.glint_mask,axis=2)
            if (marginal_difference(sd_next,sd_temp)<marginal_difference_thresh):
                break
            else:
                sd_next = sd_temp
            #increase count
            iter_count += 1

    else:
        for i in range(iter):
            HM = SUGAR(glint_image,bounds,estimate_background=estimate_background, glint_mask_method=glint_mask_method)
            corrected_bands = HM.get_corrected_bands()
            glint_image = np.stack(corrected_bands,axis=2)
            # save corrected bands for each iteration
            corrected_images.append(glint_image)
            # save glint_mask
            # if i == 0 and get_glint_mask is True:
            #     glint_mask = np.stack(HM.glint_mask,axis=2)

    if plot is True:
        rgb_bands = [28,17,7]
        corrected_images = [im_aligned] + corrected_images
        nrows = len(corrected_images)
        fig, axes = plt.subplots(nrows,1,figsize=(8,2*nrows))
        for i,(im, ax) in enumerate(zip(corrected_images,axes.flatten())):
            ax.set_title(f'Iter {i} ' + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
            display_im = np.take(im,rgb_bands,axis=2)
            ax.imshow(np.clip(display_im,0,1))
            ax.axis('off')
        
        # n_axis = nrows*2
        # n_del = int(n_axis - len(corrected_images))
        # for ax in axes.flatten()[-n_del:]:
        #     ax.set_axis_off()
        if (save_fp is not None):
            fn = os.path.splitext(save_fp)[0]
            fig.savefig(f'{fn}.png')
        plt.show()
        # b_list = HM.b_list
        # bounds = [(1,b*1.2) for b in b_list]
    
    return corrected_images #if get_glint_mask is False else (corrected_images,glint_mask)

class SUGARpipeline:
    def __init__(self,im_aligned,bbox,iter=3,bounds=[(1,2)],glint_mask_method='cdf',filename=None):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param bbox (tuple): bbox over glint area for Hedley algorithm
        :param filename (str): Full filepath required. if None, no figure is saved.
        :param bounds (a list of tuple): lower and upper bound for optimisation of b for each band
        :param glint_mask_method (str): choose either otsu or cdf
        """
        self.im_aligned = im_aligned
        self.bbox = bbox
        self.iter = iter
        self.filename = os.path.splitext(filename)[0] if isinstance(filename,str) else None
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = im_aligned.shape[-1]
        self.bounds = bounds*self.n_bands
        self.glint_mask_method = glint_mask_method

    def main(self,folder_name="saved_plots"):
        corrected_im_background, glint_mask = correction_iterative(self.im_aligned,
                                                                   iter=self.iter, 
                                                                   bounds = self.bounds,
                                                                   estimate_background=True,
                                                                   glint_mask_method=self.glint_mask_method,
                                                                   get_glint_mask=True)
        corrected_im = correction_iterative(self.im_aligned,
                                            iter=self.iter, 
                                            bounds = self.bounds,
                                            estimate_background=False,
                                            glint_mask_method=self.glint_mask_method,
                                            get_glint_mask=False)
        # save images
        parent_dir = os.path.join(os.getcwd(),folder_name)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        
        rgb_dir = os.path.join(parent_dir,'rgb')
        if not os.path.exists(rgb_dir):
            os.mkdir(rgb_dir)
        rgb_fp = os.path.join(rgb_dir,self.filename)

        compare_algo_dir = os.path.join(parent_dir,'compare_algo')
        if not os.path.exists(compare_algo_dir):
            os.mkdir(compare_algo_dir)
        compare_algo_fp = os.path.join(compare_algo_dir,self.filename)

        uncertainty_dir = os.path.join(parent_dir,'uncertainty')
        if not os.path.exists(uncertainty_dir):
            os.mkdir(uncertainty_dir)
        uncertainty_fp = os.path.join(uncertainty_dir,self.filename)

        # save rgb_images
        mutils.get_rgb(self.im_aligned, normalisation = False, plot=True, save_dir=rgb_fp+f'iter0')
        for i in range(len(corrected_im)):
            mutils.get_rgb(corrected_im_background[i], normalisation = False, plot=True, save_dir=rgb_fp+f'iter{i+1}_BG')
            mutils.get_rgb(corrected_im[i], normalisation = False, plot=True, save_dir=rgb_fp+f'iter{i+1}')
        # validate with other algo
        ValidateAlgo.compare_sugar_algo(self.im_aligned,bbox=self.bbox,
                                        corrected = corrected_im, corrected_background = corrected_im_background, 
                                        iter=self.iter, bounds=self.bounds, glint_mask_method=self.glint_mask_method,
                                        save_dir = compare_algo_fp+'_sugar')
        ValidateAlgo.compare_correction_algo(self.im_aligned,self.bbox,
                                             corrected_Hedley = None, corrected_Goodman = None, corrected_SUGAR = corrected_im_background, 
                                             iter=self.iter, bounds=self.bounds, glint_mask_method=self.glint_mask_method,
                                             save_dir = compare_algo_fp)
        # uncertainty estimation
        UE = uncertainty.UncertaintyEst(self.im_aligned,corrected_im_background, corrected_im,glint_mask=glint_mask)
        UE.get_uncertainty_bounds(save_dir = uncertainty_fp)
        UE.get_glint_kde(save_dir = uncertainty_fp+'_kde')
        return

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
    
def get_all_dir(fp,iter=3):
    """ get all parent sub directories up to iter (int) levels"""
    fp_temp = fp
    sub_dir_list = []
    for i in range(iter):
        base_fn, fn = os.path.split(fp_temp)
        sub_dir_list.append(fn)
        fp_temp = base_fn
    return '_'.join(reversed(sub_dir_list))