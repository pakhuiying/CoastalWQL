import cv2
# import micasense.plotutils as plotutils
import os, glob
import json
import tqdm
import pandas as pd
# import pickle #This library will maintain the format as well
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.stats import gaussian_kde
from scipy import ndimage
from scipy.optimize import curve_fit
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
            # axes[1,i].plot(x,y,label=r'$R_T(\lambda)\prime$')
        # axes[1,i].legend(loc="upper right")
        # axes[1,i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=3)
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
        # axes[row_idx,i].legend(loc="upper right")
        # axes[row_idx,i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=3)
        axes[row_idx,i].set_title(r'$R_T(\lambda)\prime$'+' along red line')
    
    y1,y2 = axes[row_idx,0].get_ylim()
    for i in range(len(im_list)):
        axes[row_idx,i].set_ylim(y1,y2)
    
    # manually add legend for the entire figure
    colors = ['white','#1f77b4', '#ff7f0e', 'white'] #blue,orange,green
    lines = [Line2D([0], [0], linewidth=3,c=c) for c in colors]
    labels = ['Row #2 legends: ',r'$R_T(\lambda)$',r'$R_T(\lambda)\prime$','']

    colors1 = ['white','b', 'r','g'] #blue,orange,green
    lines1 = [Line2D([0], [0], linewidth=3,c=c,alpha=0.5) for c in colors1]
    labels1 = ['Row #4 legends: ','b','r','g']

    handles = lines+lines1
    labels = labels+labels1

    reindex_fun = lambda nrow, ncol, idx: ncol*(idx%nrow) + idx//nrow

    handles_reindex = [handles[reindex_fun(2,4,i)] for i in range(len(handles))]
    labels_reindex = [labels[reindex_fun(2,4,i)] for i in range(len(labels))]
    fig.legend(handles=handles_reindex,labels=labels_reindex,loc='upper center', bbox_to_anchor=(0.5, 0),ncol=4)

    plt.tight_layout()
    

    if save_dir is not None:
        fig.savefig('{}.png'.format(save_dir), bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return

# def compare_plots(im_list, title_list, bbox=None, save_dir = None):
#     """
#     :param im_list (list of np.ndarray): where the first item is always the original image
#     :param title_list (list of str): the title for the first row
#     :param bbox (tuple): bbox over glint area for Hedley algorithm
#     :param save_dir (str): Full filepath required. if None, no figure is saved.
#     """
#     rgb_bands = [28,17,7]
#     crop_bands = [2,-3]
#     wavelength_dict = preprocessing.bands_wavelengths()
#     wavelength_dict = {i:wavelength for i,wavelength in enumerate(wavelength_dict)}
#     nrow, ncol, n_bands = im_list[0].shape

#     plot_width = len(im_list)*3
#     if bbox is not None:
#         ((x1,y1),(x2,y2)) = bbox
#         coord, w, h = bboxes_to_patches(bbox)
#         plot_height = 12
#         plot_row = 4
#     else:
#         plot_height = 9
#         plot_row = 3

#     fig, axes = plt.subplots(plot_row,len(im_list),figsize=(plot_width,plot_height))
#     if bbox is not None:
#         og_avg_reflectance = [np.mean(im_list[0][y1:y2,x1:x2,band_number]) for band_number in range(n_bands)]
#     else:
#         og_avg_reflectance = [np.mean(im_list[0][:,:,band_number]) for band_number in range(n_bands)]
#     x = list(wavelength_dict.values())
#     og_y = [og_avg_reflectance[i] for i in list(wavelength_dict)]

#     for i,(im, title) in enumerate(zip(im_list,title_list)): #iterate acoss column
#         # plot image
#         rgb_im = np.take(im,rgb_bands,axis=2)
#         axes[0,i].imshow(rgb_im)
#         axes[0,i].set_title(title + r'($\sigma^2_T$' + f': {np.var(im):.4f})')
#         axes[0,i].axis('off')
#         if bbox is not None:
#             rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor='red', facecolor='none')
#             axes[0,i].add_patch(rect)
#         # plot original reflectance
#         axes[1,i].plot(x[crop_bands[0]:crop_bands[1]],og_y[crop_bands[0]:crop_bands[1]],label=r'$R_T(\lambda)$')
#         # plot corrected reflectance
#         if i > 0:
#             if bbox is not None:
#                 avg_reflectance = [np.mean(im[y1:y2,x1:x2,band_number]) for band_number in range(n_bands)]
#             else:
#                 avg_reflectance = [np.mean(im[:,:,band_number]) for band_number in range(n_bands)]
#             y = [avg_reflectance[i] for i in list(wavelength_dict)]
#             axes[1,i].plot(x[crop_bands[0]:crop_bands[1]],y[crop_bands[0]:crop_bands[1]],label=r'$R_T(\lambda)\prime$')
#         axes[1,i].legend(loc="upper right")
#         axes[1,i].set_title(r'$R_T(\lambda)\prime$'+' in AOI')
#         axes[1,i].set_xlabel('Wavelengths (nm)')
#         axes[1,i].set_ylabel('Reflectance')

#         # plot cropped rgb
#         rgb_cropped = rgb_im[y1:y2,x1:x2,:] if bbox is not None else rgb_im
#         if bbox is not None:
#             axes[2,i].imshow(rgb_cropped)
#             axes[2,i].set_title('AOI')
#             axes[2,i].plot([0,w-1],[abs(h)//2,abs(h)//2],color="red",linewidth=3,alpha=0.5)
        
#         row_idx = 3 if bbox is not None else 2
#         h = nrow if bbox is None else h
#         w = ncol if bbox is None else w
#         # plot reflectance along red line
#         for j,c in enumerate(['r','g','b']):
#             axes[row_idx,i].plot(list(range(w)),rgb_cropped[h//2,:,j],c=c,alpha=0.5,label=c)
#         axes[row_idx,i].set_xlabel('Image position')
#         axes[row_idx,i].set_ylabel('Reflectance')
#         axes[row_idx,i].legend(loc="upper right")
#         axes[row_idx,i].set_title(r'$R_T(\lambda)\prime$'+' along red line')
    
#     y1,y2 = axes[row_idx,0].get_ylim()
#     for i in range(len(im_list)):
#         axes[row_idx,i].set_ylim(y1,y2)
    
#     plt.tight_layout()
    

#     if save_dir is not None:
#         fig.savefig('{}.png'.format(save_dir))
#         plt.close()
#     else:
#         plt.show()
#     return

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

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    
    return lc
class ValidateInsitu:
    """
    validate sunglint correction with in-situ data
    """
    def __init__(self,fp_list,titles,conc_index = 2, save_dir = None):
        """
        :param fp_list (list of str): where first item is the fp of the original (uncorrected R_T)
        :param titles (list of str): description of the algorithm that corresponds to fp_list
        :param save_dir (fp): directory of where to store data, if None, no data is stored
        """
        self.fp_list = fp_list
        self.titles = titles
        assert len(titles) == len(fp_list)
        self.save_dir = save_dir
        self.parent_dir = None
        if self.save_dir is not None:
            parent_dir = os.path.join(self.save_dir,'insitu_validation')
            if not os.path.exists(parent_dir):
                os.mkdir(parent_dir)
            self.parent_dir = parent_dir
        self.conc_index = conc_index
        wavelength_dict = preprocessing.bands_wavelengths()
        self.wavelength_dict = {i:wavelength for i,wavelength in enumerate(wavelength_dict)}
        self.crop_bands = [2,-3]

    def get_df_list(self):
        df_list = []
        na_idx = []
        for i in range(len(self.fp_list)):
            df = pd.read_csv(self.fp_list[i])
            df = df[(df['observation_number'] < 353) | (df['observation_number']> 402)]
            df = df.dropna()
            df_list.append(df)
            na_idx.append(set(df['observation_number'].tolist()))
        
        na_idx = na_idx[0].intersection(*na_idx[1:])
        df_list = {self.titles[i]: df[df['observation_number'].isin(list(na_idx))] for i,df in enumerate(df_list)}
        return df_list

    def plot_conc_spectral(self, cmap='Spectral_r',add_colorbar=True,axes=None):
        """ 
        :param fp_list (list of str): list of filepath to df in Extracted_Spectral_Information
        outputs individual reflectance curve mapped to TSS concentration
        """
        df_list = self.get_df_list()
        concentration = df_list[list(df_list)[0]].iloc[:,self.conc_index].tolist()
        wavelength = list(self.wavelength_dict.values())
        wavelength_array = np.array([wavelength for i in range(len(concentration))])

        df_reflectance_list = dict()
        for t,df in df_list.items():
            df_reflectance = df.filter(regex=('band.*'))
            df_reflectance.columns = [f'{w:.2f}' for w in wavelength]
            df_reflectance_list[t] = df_reflectance.values

        n = len(concentration)
        if axes is None:
            ncols = len(list(df_list))
            col_width = ncols*3
            fig, axes = plt.subplots(1,ncols,figsize=(col_width,4),sharex=True,sharey=True)
        else:
            assert len(axes) == len(list(df_reflectance_list))
        for i, ((title,df),ax) in enumerate(zip(df_reflectance_list.items(),axes)):
            x_array = wavelength_array[:,self.crop_bands[0]:self.crop_bands[1]]
            y_array = df[:,self.crop_bands[0]:self.crop_bands[1]]
            lc = multiline(x_array, y_array, concentration,ax=ax, cmap=cmap, lw=1)
            lc.set_clim(min(concentration),max(concentration))
            ax.set_title(f'{title} (N = {n})')
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(int(start), int(end), 100))
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Reflectance')

        self.cmap = lc.get_cmap()
        self.clim = lc.get_clim()
        if add_colorbar is True:
            axcb = fig.colorbar(lc)
            axcb.set_label('Turbidity (NTU)')

        if (self.parent_dir is not None) and (axes is None):
            plt.tight_layout()
            plt.show()
            fig.savefig(os.path.join(self.parent_dir,'insitu_reflectance.png'))
        return

    def plot_wavelength_conc(self,df_list,title):
        """
        :param df_list (list of pd.DataFrame): where first element is the original/uncorrected R_T, 
            and the second item is the R_T_prime
        :param title (str): title of sgc algo
        returns plot of reflectance vs concentration for every band
        """
        def func(x, a, b, c):
            return a*x**2 + b*x + c

        wavelength = list(self.wavelength_dict.values())
    
        df_reflectance_list = []
        for df in df_list:
            df_reflectance = df.filter(regex=('band.*'))
            df_reflectance.columns = [f'{w:.2f}' for w in wavelength]
            df_reflectance_list.append(df_reflectance.values)
        df_conc = df_list[0].iloc[:,self.conc_index].to_numpy()

        fig, axes = plt.subplots(8,7,figsize=(20,25))
        n = len(df_conc)
    
        title_desc = ['original',title]
        c_list = ['tab:blue','tab:orange']
        labels = [r'$R_T$',r'$R_T\prime$']
        crop_wavelengths = wavelength[self.crop_bands[0]:self.crop_bands[1]]
        RMSE_dict = {w:{'original':None,title:None} for w in crop_wavelengths}
        MAPE_dict = {w:{'original':None,title:None} for w in crop_wavelengths}
        assert len(crop_wavelengths) == len(axes.flatten())
        for i, (w, ax) in enumerate(zip(crop_wavelengths,axes.flatten())):
            for j,label in zip(range(len(df_reflectance_list)),labels):
                y = df_reflectance_list[j][:,self.crop_bands[0]:self.crop_bands[1]][:,i]
                # plot scatter
                ax.plot(df_conc,y,'o',label=labels[j],alpha=0.5,c=c_list[j])
                # fit curve
                popt, _ = curve_fit(func, df_conc, y)
                x = np.linspace(np.min(df_conc),np.max(df_conc),50)
                y_hat = func(x,*popt)
                # plot fitted line
                ax.plot(x,y_hat,linestyle='--',linewidth=2,label=f'{labels[j]}_fitted',c=c_list[j])
                # get predicted values
                Y_HAT = func(df_conc,*popt)
                #calculate rmse
                rmse = (np.sum((Y_HAT - y)**2)/len(Y_HAT))**(1/2)
                RMSE_dict[w][title_desc[j]] = rmse
                # calculate MAPE
                mape = np.sum(np.abs((y - Y_HAT)/y))/len(Y_HAT)
                MAPE_dict[w][title_desc[j]] = mape
                ax.set_title(f'{w} nm (N = {n})')
                ax.set_ylabel('Reflectance')
                ax.set_xlabel('Turbidity (NTU)')
        # fig.suptitle(title)
        plt.tight_layout()
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.05)
        fig.legend(handles, labels,loc='lower center',ncol=4,prop={'size': 16})
        plt.show()

        if self.parent_dir is not None:
            fig.savefig(os.path.join(self.parent_dir,f'{title}_insitu_reflectance.png'))

        metrics_dict = {'RMSE':RMSE_dict,'MAPE':MAPE_dict}
        metrics_df = dict()
        for metrics,dic in metrics_dict.items():
            original_r = [d['original'] for _, d in dic.items()]
            corrected_r = [d[title] for _, d in dic.items()]
            df = pd.DataFrame({'Wavelength':list(dic),'original':original_r,title:corrected_r})
            metrics_df[metrics] = df
            # if self.parent_dir is not None:
            #     df.to_csv(os.path.join(self.parent_dir,f'{title}_insitu_{metrics}.csv'),index=False)
        return metrics_df
    
    def get_metrics(self):
        df_list = self.get_df_list()
        og_df = df_list[list(df_list)[0]]
        metrics_df_list = dict()
        for i,(title,df) in enumerate(df_list.items()):
            if i > 0:
                df_2 = [og_df,df]
                metrics_df = self.plot_wavelength_conc(df_2,title)
                for metrics, df_metrics in metrics_df.items():
                    if self.parent_dir is not None:
                        df_metrics.to_csv(os.path.join(self.parent_dir,f'{i}_{title}_insitu_{metrics}.csv'),index=False)
                metrics_df_list[title] = metrics_df
        
        return metrics_df_list
    
def plot_insitu(stitch_class,image,tss_lat,tss_lon,tss_measurements,radius=1,mask = None,axis=None,**kwargs):
    """ 
    :param stitch_class (SitchHyperspectral class)
    :param image (np.ndarray): rgb image
    """
    gps_indexes = [(stitch_class.test_gps_index[i],stitch_class.test_gps_index[i+1]) for i in range(0,len(stitch_class.test_gps_index)-1,2)]
    # print(gps_indexes)
    gps_start_index, gps_end_index = gps_indexes[stitch_class.line_number] #use line number to replace 0
    nrow,ncol,n_bands = image.shape
    
    #-----------PERFORM MASKING--------------------
    if mask is not None:
        image[mask!=0] = 1

    #-----------PERFORM MASKING--------------------

    #apply get_affine_transformation to each corrected_reflectance band
    #inclusive of gps start index and gps end index
    wavelength_list = preprocessing.bands_wavelengths()#self.get_s1_bands().tolist()
    coords = np.transpose(stitch_class.unique_gps_df.iloc[[gps_start_index, gps_end_index],[1,2]].values) #1=latitude, 2=longitude
    lat = coords[0,:] #list of latitude, lat[0] = lat_start
    lon = coords[1,:] #list of lon
    lat_start, lat_end = lat
    lon_start, lon_end = lon

    #-------COMPUTE FLIGHT ANGLE (start)-------------

    # print("Flight angle is {:.2f}".format(angle))
    direction_vector = np.array([lon_end,lat_end]) - np.array([lon_start,lat_start])
    direction_vector = direction_vector/np.linalg.norm(direction_vector) #convert to unit vector
    east_vector = np.array([1,0]) #measured from the horizontal as a reference
    angle = np.arccos(np.dot(direction_vector,east_vector))/(2*np.pi)*360 #direction vector already converted to a unit vector
    
    #---cross pdt---
    #if vector is always on the left side of east_vector, cross pdt will be pointing in (i.e. -ve), otherwise it will be pointing outwards (i.e. +ve)
    if np.cross(direction_vector,east_vector) > 0: #point outwards aka to the right of the east vector
        angle = 180 - angle 

    print("Flight angle is {:.2f}".format(angle))
    #-------COMPUTE FLIGHT ANGLE (end)-------------

    #-------CONDUCT TRANSFORMATION (start)-------------
    print("Conducting georectification...")
    # transformed_imges = [self.get_affine_transformation(arr,angle) for arr in corrected_reflectance.values()] #apply transformation to all bands
    transformed_imges = stitch_class.get_affine_transformation(image,angle)
    nrow_transformed, ncol_transformed = transformed_imges.shape[0],transformed_imges.shape[1]
    # transformed_imges = [self.get_affine_transformation(arr,angle) for arr in masked_corrected_reflectance] #apply transformation to all bands
    # nrow_transformed, ncol_transformed = transformed_imges[0].shape[0], transformed_imges[0].shape[1]
    transformed_imges = np.where(transformed_imges==0,1,transformed_imges)
    
    #-------CONDUCT TRANSFORMATION (end)-------------
    

    #---------EXTRAPOLATE GPS (start)---------------
    
    # length_direction_vector = np.linalg.norm(direction_vector)
    lat_res_per_pixel = (np.max(lat) - np.min(lat))/(ncol*np.sin(angle/360*2*np.pi))
    lon_res_per_pixel = (np.max(lon) - np.min(lon))/(ncol*np.cos(angle/360*2*np.pi)) 
    #---------define coord bounding box--------------
    if angle > 90 and angle < 180:
        UPPER_LEFT_lat = np.max(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve
        UPPER_LEFT_lon = np.min(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is -ve
        ul = (UPPER_LEFT_lat,UPPER_LEFT_lon)
        LOWER_RIGHT_lat = np.min(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve
        LOWER_RIGHT_lon = np.max(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is -ve
        lr = (LOWER_RIGHT_lat,LOWER_RIGHT_lon)
        print("upper left:{}\nlower right:{}".format(ul,lr))
    else: #angle >= 0 and angle <= 90
        UPPER_LEFT_lat = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is +ve
        UPPER_LEFT_lon = np.min(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is +ve
        ul = (UPPER_LEFT_lat,UPPER_LEFT_lon)
        LOWER_RIGHT_lat = np.min(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is +ve
        LOWER_RIGHT_lon = np.max(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is +ve
        lr = (LOWER_RIGHT_lat,LOWER_RIGHT_lon)
        print("upper left:{}\nlower right:{}".format(ul,lr))
    #---------define coord rotated bounding box--------------
    if angle > 90 and angle < 180:
        #upper left corner
        UL_lon = np.min(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon is -ve
        UL_lat = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve
        upper_left = [UL_lat,UL_lon] #upper left coord of transformed img
        UR_lon = np.min(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel
        UR_lat = np.max(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel
        upper_right = [UR_lat,UR_lon] #upper right coord of transformed img
        LL_lon = np.max(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel
        LL_lat = np.min(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel
        lower_left = [LL_lat,LL_lon] #lower left coord of transformed img
        LR_lon = np.max(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel
        LR_lat = np.min(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel
        lower_right = [LR_lat,LR_lon]
    else: #angle >= 0 and angle <= 90
        UL_lon = np.max(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon is +ve
        UL_lat = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is +ve
        upper_left = [UL_lat,UL_lon] #upper left coord of transformed img
        UR_lon = np.max(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel
        UR_lat = np.max(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel
        upper_right = [UR_lat,UR_lon] #upper right coord of transformed img
        LL_lon = np.min(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel
        LL_lat = np.min(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel
        lower_left = [LL_lat,LL_lon] #lower left coord of transformed img
        LR_lon = np.min(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel
        LR_lat = np.min(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel
        lower_right = [LR_lat,LR_lon]
    # print("UL:{}\nUR:{}\nLL:{}\nLR:{}".format(upper_left,upper_right,lower_left,lower_right))
    #---------define coord rotated bounding box--------------
    #---------define function to check if coord is within rotated bounding box---------------------
    vertices = [upper_right,lower_right,lower_left] #A, B, C
    vertices = [np.flip(np.array(v)) for v in vertices] #flip to convert to lon, lat (x,y) coord sys
    def check_within_bounding_box(vertices,P):
        """ 
        ABC makes the vertices of the rectangle
        A = upper_right
        B = lower_right
        C = lower_left
        P = random point
        For points inside rectangle both distances should have the same sign - either negative or positive depending on vertices order (CW or CCW)
        and their absolute values should not be larger than lBC and lAB correspondingly
        """
        A,B,C = vertices
        lAB = np.linalg.norm(B-A)
        lBC = np.linalg.norm(C-B)
        uAB = (B-A)/lAB
        uBC = (C-B)/lBC
        BP = P-B
        ABP = np.cross(BP,uAB)
        BCP = np.cross(BP,uBC)
        if (ABP <0 and BCP<0 and abs(ABP)<= lBC and abs(BCP)<=lAB) or (ABP >0 and BCP>0 and abs(ABP)<= lBC and abs(BCP)<=lAB):
            return True
        else:
            return False

    radius = int(radius * 10) #100 pixels wide
    #---------filter tss measurements that fall within the bounding box--------------
    rows_idx = []
    cols_idx = []
    tss_idx = []
    #keys are the indexes of tss measurements
    for i in range(len(tss_lat)): #where i = number of tss observations
        # if tss_lat[i]<ul[0] and tss_lat[i]>lr[0] and tss_lon[i]<lr[1] and tss_lon[i]>ul[1]:
        P = np.array([tss_lon[i],tss_lat[i]])
        if check_within_bounding_box(vertices,P):
            vert_pixel = int((ul[0] - tss_lat[i])/np.abs(lat_res_per_pixel)) #reference from the uppermost extent-->row no.
            hor_pixel = ncol_transformed - int((lr[1]-tss_lon[i])/np.abs(lon_res_per_pixel)) #reference from the most left extent-->col no.
            rows_idx.append(vert_pixel)
            cols_idx.append(hor_pixel)
            tss_idx.append(tss_measurements[i])
    
    if axis is None:
        fig,axis = plt.subplots(figsize=(7,10))
    axis.imshow(transformed_imges)
    axis.axis('off')
    im = axis.scatter(cols_idx,rows_idx,c=tss_idx,s=2,alpha=0.5,label='in-situ sampling',**kwargs)
    axis.legend(loc='lower center',prop={'size': 16})
    # axis.set_title('In-situ sampling')
    axcb = plt.colorbar(im,ax=axis)
    axcb.set_label('Turbidity (NTU)')
    if axis is None:
        plt.show()
    
    return 

def plot_insitu_spectral(validate_insitu,stitch_class,image,tss_lat,tss_lon,tss_measurements,radius=1,mask = None,axis=None):
    nrow = len(validate_insitu.fp_list)
    ncol = 3

    fig = plt.figure(figsize=(10,11))
    gs = GridSpec(nrow,ncol,figure=fig)
    axes1 = [fig.add_subplot(gs[i,0]) for i in range(nrow)]
    validate_insitu.plot_conc_spectral(add_colorbar=False,axes=axes1)
    ylims = [ax.get_ylim() for ax in axes1]
    y_min = min([y[0] for y in ylims])
    y_max = max([y[1] for y in ylims])
    for ax in axes1:
        ax.set_ylim(y_min,y_max)
    
    axis = fig.add_subplot(gs[:,1:])
    plot_insitu(stitch_class,image,
                tss_lat,tss_lon,tss_measurements,
                radius=radius,mask = mask,
                axis=axis,vmin=validate_insitu.clim[0],vmax=validate_insitu.clim[1],cmap=validate_insitu.cmap)
    plt.tight_layout()
    plt.show()
    return