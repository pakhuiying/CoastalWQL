import numpy as np
from glob import glob
from os import listdir, mkdir
from os.path import isfile, join, exists, splitext
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_tkagg import FigureCanvasAgg,FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import colors
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import re 
from datetime import datetime, timedelta
try:
    from osgeo import gdal, osr
except:
    pass
    print("GDAL not imported, geotransformation cannot be conducted...")
import pandas as pd
import time
from PIL import Image, ImageGrab
from skimage.transform import rescale
from skimage.measure import block_reduce
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats.stats import pearsonr
import cv2
import math
import xgboost as xgb
try:
    import rasterio
except:
    pass
try:
    import requests
except:
    pass
from math import ceil
from io import BytesIO
import base64
try:
    import PySimpleGUI as sg
except:
    pass
import json
import sunglint_correction.SUGAR as sugar


def figure_to_image(figure):
    """
    Draws the previously created "figure" in the supplied Image Element

    :param element: an Image Element
    :param figure: a Matplotlib figure
    :return: The figure canvas
    """

    plt.close('all')        # erases previously drawn plots
    canv = FigureCanvasAgg(figure)
    buf = BytesIO()
    canv.print_figure(buf, format='png')
    if buf is None:
        return None
    buf.seek(0)
    return buf.read()

def convert_to_bytes(file_or_bytes, resize=None):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    *Preview only works for RGB images
    '''
    if isinstance(file_or_bytes, str):
        img = Image.open(file_or_bytes)
    else:
        try:
            img = Image.open(BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = BytesIO(file_or_bytes)
            img = Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), Image.ANTIALIAS)
    with BytesIO() as bio:
        img.save(bio, format="PNG") #one-layer image e.g. predicted image cannot be shown cus needs to be RGB images
        del img
        return bio.getvalue()

def convert_to_bytes1(img, resize=None):
    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), Image.ANTIALIAS)
    with BytesIO() as bio:
        img.save(bio, format="PNG") #one-layer image e.g. predicted image cannot be shown cus needs to be RGB images
        del img
        return bio.getvalue()

def save_element_as_file(element, filename):
    """
    Saves any element as an image file.  Element needs to have an underlyiong Widget available (almost if not all of them do)
    :param element: The element to save
    :param filename: The filename to save to. The extension of the filename determines the format (jpg, png, gif, ?)
    """
    widget = element.Widget
    # box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
    # grab = ImageGrab.grab(bbox=box)
    grab = ImageGrab.grab(bbox=None)
    grab.save(filename)

def covariates_str_to_list(covariates):
    if covariates != '':
        covariates_index = covariates.replace(' ','').split(',')
        covariates_index_list = []
        for i in covariates_index:
            if ':' in i:
                c_start, c_end = i.split(':')
                # print("c_start,c_end:",c_start, c_end)
                try:
                    covariates_index_list = covariates_index_list + list(range(int(c_start),int(c_end)+1))
                except Exception as E:
                    pass
            else:
                covariates_index_list.append(int(i))
    else:
        covariates_index_list = list(range(61))
    
    return covariates_index_list
    

def plot_predicted_image(fp):
    predicted_img = cv2.imread(fp, cv2.IMREAD_UNCHANGED) #to read a >8bit image
    fig = plt.figure()
    im = plt.imshow(predicted_img,cmap="jet")
    cax = fig.add_axes([0.91,0.3,0.03,0.4])
    fig.colorbar(im, cax=cax)
    return fig

def plot_TSS_conc_image(df):
    """ 
    input is a df with last 61 columns as wavelengths and a column called 'Concentration' that corresponds to TSS concentration
    unit (str): label for the colour map e.g. FNU or mg/l
    outputs individual reflectance curve mapped to TSS concentration
    """
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

    

    ncols = len(df.columns)
    col_names = df.columns.tolist()
    columns = ['Concentration'] + col_names[-61:]
    df_plot = df.loc[:,columns].set_index('Concentration')
    wavelength = df_plot.columns.tolist()

    concentration = df_plot.index.tolist()
    n_lines = len(concentration)
    y_array = df_plot.reset_index().iloc[:,-61:].values #reflectance
    x_array = np.array([wavelength for i in range(n_lines)])

    fig, ax = plt.subplots(figsize=(6,6))
    lc = multiline(x_array, y_array, concentration, cmap='BrBG_r', lw=1)
    ax.set_ylim(0,40)
    ax.set_xlim(450,950)
    
    axcb = fig.colorbar(lc)
    axcb.set_label('Concentration')
    ax.set_title('Reflectance of water quality variable')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    plt.rcParams["font.size"] = "8"
    return fig


def draw_gps(test_gps_index,unique_gps_df):
    # plt.figure(figsize=(20,10))
    plt.plot(unique_gps_df.iloc[:,2],unique_gps_df.iloc[:,1],'o')
    filtered_unique_gps_df = unique_gps_df.iloc[test_gps_index,:]
    gps_list = filtered_unique_gps_df.index.tolist()
    for i in range(len(gps_list)):
        
        x = filtered_unique_gps_df.iloc[i,:]['longitude']
        y = filtered_unique_gps_df.iloc[i,:]['latitude']
        time_start_stop = filtered_unique_gps_df.iloc[i,:]['datetime'].strftime('%H-%M-%S')
        label_gps = 'GPS: {}\ntime: {}'.format(gps_list[i],time_start_stop)
        plt.plot(x,y,'o',c='orange')
        plt.text(x, y, label_gps, style='italic',bbox={'facecolor': 'white', 'alpha': 0.5},fontsize="medium")
    plt.show(block=False)

def plot_GPS(unique_gps_df,gps_indices):
    plt.figure()
    lon = unique_gps_df['longitude'].tolist()
    lat = unique_gps_df['latitude'].tolist()
    plt.plot(lon, lat,'o')
    plt.plot([lon[i] for i in gps_indices],[lat[i] for i in gps_indices],'ro')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def read_gps_index(gps_index_txt):
    """ 
    :param gps_index_text (str): filepath
    """
    with open(gps_index_txt, "r") as f:
        gps_indices = f.readlines()
    gps_indices = [int(i.replace('\n','')) for i in gps_indices]
    gps_indices = sorted(gps_indices)
    return gps_indices

def convert_array_to_bytes(raster_dict):
    """ 
    list_of_np_arrays: list of images as numpy arrays
    returns a list of bytes
    """
    bytes_list = []
    for f,attr_dict in raster_dict.items():
        img_resize = raster_dict[f]['img_resize']
        img = Image.fromarray(img_resize)
        with BytesIO() as bio:
            img.save(bio, format="PNG") #one-layer image e.g. predicted image cannot be shown cus needs to be RGB images
            del img
            bytes_list.append(bio.getvalue())
            raster_dict[f]['bytes'] = bio.getvalue()

    return bytes_list

def geotransformed_images_to_arrays(raster_dict,dim_dict,scale):
    """
    updates dim_dict
    returns None
    """
    for f,attr_dict in raster_dict.items():
        r = attr_dict['rasterio']
        ncols_from_left = int((dim_dict['max_left'] - attr_dict['left'])/attr_dict['lon_res'])
        nrows_from_top = int((dim_dict['max_top'] - attr_dict['top'])/attr_dict['lat_res'])
        raster_dict[f]['ncols_from_left'] = ncols_from_left
        raster_dict[f]['nrows_from_top'] = nrows_from_top
        array_rows = nrows_from_top + attr_dict['nrows']
        array_cols = ncols_from_left + attr_dict['ncols']
        # raster_dict[f]['array_shape'] = (array_rows,array_cols)
        img = np.dstack((r.read(1),r.read(2),r.read(3)))
        # print(f'img_shape: {img.shape}')
        bckgrnd = np.zeros((dim_dict['n_rows'],dim_dict['n_cols'],3),dtype=np.uint8)
        bckgrnd[nrows_from_top:array_rows,ncols_from_left:array_cols,:] = img
        alpha = np.where(bckgrnd[:,:,1]>0,255,0)
        image_strip = np.fliplr(np.dstack((bckgrnd,alpha))).astype(np.uint8)
        img_resize = cv2.resize(image_strip,dsize=(dim_dict['n_cols']//scale,dim_dict['n_rows']//scale), interpolation=cv2.INTER_CUBIC)
        raster_dict[f]['img_resize'] = img_resize
    
    return

def geotransformed_predicted_images_to_arrays(fp_store,predicted_raster_dict, dim_dict,scale):
    """
    convert greyscale images to rgb array
    updated dim_dict
    returns mapper
    """
    max_DN = 0
    min_DN = 255
    for f in list(predicted_raster_dict):
        f = f.replace('Geotransformed_','')
        file = join(fp_store,f)
        r = rasterio.open(file)
        img = r.read(1)
        if np.max(img) > max_DN:
            max_DN = np.max(img)
        if np.min(img) < min_DN:
            min_DN = np.min(img)

    norm = colors.Normalize(vmin=min_DN, vmax=max_DN, clip=True) # data array is first mapped onto the range 0-1 
    mapper = cm.ScalarMappable(norm=norm, cmap='BrBG_r') #return mapper 
    for f,attr_dict in predicted_raster_dict.items():
        r = attr_dict['rasterio']
        ncols_from_left = int((dim_dict['max_left'] - attr_dict['left'])/attr_dict['lon_res'])
        nrows_from_top = int((dim_dict['max_top'] - attr_dict['top'])/attr_dict['lat_res'])
        predicted_raster_dict[f]['ncols_from_left'] = ncols_from_left
        predicted_raster_dict[f]['nrows_from_top'] = nrows_from_top
        array_rows = nrows_from_top + attr_dict['nrows']
        array_cols = ncols_from_left + attr_dict['ncols']
        img = r.read(1)
        bckgrnd = np.zeros((dim_dict['n_rows'],dim_dict['n_cols']),dtype=np.uint8)
        bckgrnd[nrows_from_top:array_rows,ncols_from_left:array_cols] = img
        alpha = np.where(bckgrnd>0,255,0)
        rgba_img = mapper.to_rgba(bckgrnd,bytes=True)
        rgb_img = np.delete(rgba_img, 3, 2)
        image_strip = np.fliplr(np.dstack((rgb_img,alpha))).astype(np.uint8)
        nrows,ncols,_ = image_strip.shape
        img_resize = cv2.resize(image_strip,dsize=(ncols//scale,nrows//scale), interpolation=cv2.INTER_CUBIC)
        predicted_raster_dict[f]['img_resize'] = img_resize
        grey_scale_img = np.fliplr(bckgrnd).astype(np.uint8)
        grey_scale_img = cv2.resize(grey_scale_img,dsize=(ncols//scale,nrows//scale), interpolation=cv2.INTER_CUBIC)
        predicted_raster_dict[f]['grey_scale_img'] = grey_scale_img
    return mapper


def get_dim_canvas(raster_dict):
    """ 
    raster_list: list of raster images opened by rasterio
    returns (dict): 
    """
    raster_list = [v['rasterio'] for v in raster_dict.values()]
    max_left = raster_list[0].bounds[0]
    max_top = raster_list[0].bounds[3]
    min_right = raster_list[0].bounds[2]
    min_bottom = raster_list[0].bounds[1]
    lon_res = abs(raster_list[0].bounds[0] - raster_list[0].bounds[2])/raster_list[0].read().shape[2]
    lat_res = abs(raster_list[0].bounds[1] - raster_list[0].bounds[3])/raster_list[0].read().shape[1]
    for r in raster_list:
        if r.bounds[0] > max_left:
            max_left = r.bounds[0]
        if r.bounds[3] > max_top:
            max_top = r.bounds[3]
        if r.bounds[2] < min_right:
            min_right = r.bounds[2]
        if r.bounds[1] < min_bottom:
            min_bottom = r.bounds[1]
        lon_res_temp = abs(r.bounds[0] - r.bounds[2])/r.read().shape[2]
        lat_res_temp = abs(r.bounds[1] - r.bounds[3])/r.read().shape[1]
        if lon_res_temp < lon_res:
            lon_res = lon_res_temp
        if lat_res_temp < lat_res:
            lat_res = lat_res_temp

    n_cols = ceil((max_left - min_right)/lon_res)
    n_rows = ceil((max_top - min_bottom)/lat_res)
    dist = Haversine((max_left,max_top),(min_right,max_top)).meters
    # dim_dict = {'max_top':max_top,'max_left':max_left,'lon_res':lon_res,'lat_res':lat_res,'n_cols':n_cols,'n_rows':n_rows,'dist':dist}
    return {'lon_res': lon_res,'lat_res': lat_res,'min_right':min_right,'min_bottom':min_bottom,'max_left': max_left, 'max_top': max_top,'n_cols':n_cols,'n_rows':n_rows,'dist':dist}
    # return dim_dict
    

def get_raster_dict(image_lines_list,fp_store):
    """
    image_lines_list (list of filenames)
    fp_store: directory where images are stored
    """
    raster_dict = {f:{} for f in image_lines_list}
    for f in image_lines_list:
        file = join(fp_store,f)
        try:
            r = rasterio.open(file)
            raster_dict[f]['rasterio'] = r
        except Exception as E:
            sg.popup(f'{E}\n Cannot open image file',title='Error')
            pass
        line_number = re.sub(r'^.*?image_line_','',f)[:2] #take the first two characters since they have been padded to 2 digits
        raster_dict[f]['line_number'] = line_number
        raster_dict[f]['left'] = r.bounds[0]
        raster_dict[f]['top'] = r.bounds[3]
        _, nrows, ncols = r.read().shape
        raster_dict[f]['nrows'] = nrows
        raster_dict[f]['ncols'] = ncols
        lon_res = abs(r.bounds[0] - r.bounds[2])/ncols
        lat_res = abs(r.bounds[1] - r.bounds[3])/nrows
        raster_dict[f]['lon_res'] = lon_res
        raster_dict[f]['lat_res'] = lat_res
    
    return raster_dict
#-----------------------view georeferenced images------------------------------
class ViewCanvas:
    def __init__(self,config_file,scale=20):
        self.config_file = config_file
        self.scale = scale
        self.fp_store = config_file['-PROCESSED_IMAGES-']
        self.rgb_fp_list = [join(self.fp_store,f) for f in sorted(listdir(self.fp_store)) if 'Geotransformed' not in f and 'rgb_image_line' in f]
        self.rgb_list = [np.asarray(Image.open(f)) for f in self.rgb_fp_list]
        self.gps_fp = config_file['-GPS_INDEX_TXT-']
        self.dist = None
        self.dist = None
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        try:
            with open(config_file['-GPS_INDEX_TXT-'], "r") as f:
                gps_indices = f.readlines()
            gps_indices = [int(i.replace('\n','')) for i in gps_indices]
            gps_indices.sort()
            self.gps_indices = gps_indices
        except:
            self.gps_indices = None

        self.image_folder_fp = config_file['-IMAGE_FOLDER_FILEPATH-']

        try:
            gps_df = import_gps(self.image_folder_fp)
            unique_gps_df = get_unique_df(gps_df)
            self.unique_gps_df = unique_gps_df
        except:
            self.unique_gps_df = None

        if config_file['-PREDICT_CHECKBOX-'] is True:
            self.prediction_fp_list = [join(self.fp_store,'Prediction',f) for f in sorted(listdir(join(self.fp_store,'Prediction'))) if 'Geotransformed' not in f and 'predicted_image_line' in f]
            self.prediction_list = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in self.prediction_fp_list]
        else:
            self.prediction_fp_list = None
            self.prediction_list = None

        self.wql_csv = join(config_file['-PROCESSED_IMAGES-'],'Extracted_Spectral_Information',config_file['-PREFIX-']+'_TSS_spectral_info.csv')
        try:
            wql_df = pd.read_csv(self.wql_csv)
            self.wql_df = wql_df
        except:
            self.wql_df = None
            
    
    def create_gps_image_dict(self,mode = 'rgb'):
        if mode == 'rgb':
            d = {i: {'image':self.rgb_list[i]} for i in range(len(self.rgb_list))}
        elif mode == 'prediction' and self.prediction_list is not None:
            d = {i: {'image':self.prediction_list[i]} for i in range(len(self.prediction_list))}
        number_of_lines_counter = 0
        filtered_unique_gps_df = self.unique_gps_df.iloc[self.gps_indices,:]
        for index, rows in filtered_unique_gps_df.reset_index().iterrows():
            lat = rows[2]
            lon = rows[3]
            if index %2 == 0: #start
                d[number_of_lines_counter]['start'] = {'lat':lat,'lon':lon}
            else:
                d[number_of_lines_counter]['stop'] = {'lat':lat,'lon':lon}
                number_of_lines_counter+=1
        
        return d
    
    def calculate_flight_angle(self,gps_start,gps_stop):
        """
        gps_start (tuple of float): (lon_start,lat_start)
        gps_stop (tuple of float): (lon_end,lat_end)
        """
        # print("Calculating flight angle...")
        direction_vector = np.array([gps_stop]) - np.array([gps_start])
        direction_vector = direction_vector/np.linalg.norm(direction_vector) #convert to unit vector
        east_vector = np.array([1,0]) #measured from the horizontal as a reference
        angle = np.arccos(np.dot(direction_vector,east_vector))/(2*np.pi)*360 #direction vector already converted to a unit vector
        
        if np.cross(direction_vector,east_vector) > 0: #point outwards aka to the right of the east vector
            angle = 180 - angle 
            
        return angle[0]

    def geotransform_image(self,corrected_rgb_img,angle):
        """
        returns a geotransformed corrected img
        """
        nrow,ncol = corrected_rgb_img.shape[0],corrected_rgb_img.shape[1]
        img = np.flipud(np.array(corrected_rgb_img)) #flip vertically
        center = (ncol//2,nrow//2)
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,1) #center, angle, scale
        cosofRotationMatrix = np.abs(rotation_matrix[0][0]) #scale*cos(angle)
        sinofRotationMatrix = np.abs(rotation_matrix[0][1]) #scale*sin(angle)

        newImageHeight = int((ncol * sinofRotationMatrix) +
                            (nrow * cosofRotationMatrix))
        newImageWidth = int((ncol * cosofRotationMatrix) +
                            (nrow * sinofRotationMatrix))

        rotation_matrix[0][2] += (newImageWidth/2) - center[0]
        rotation_matrix[1][2] += (newImageHeight/2) - center[1]

        rotatingimage = cv2.warpAffine(img, rotation_matrix, (newImageWidth, newImageHeight))

        return rotatingimage
    
    def get_bounding_box(self,corrected_rgb_img,geotransformed_corrected_image,angle,gps_start,gps_stop):
        """
        gets bounding box (upper left (lat,lon), lower right (lat,lon)) of a corrcted geotransformed image
        returns bbox and lat,lon resolution
        nrows, ncols are dimensions of the geotransformed image in the returned output
        """
        nrow,ncol = corrected_rgb_img.shape[0], corrected_rgb_img.shape[1]
        lon_start, lat_start = gps_start
        lon_stop, lat_stop = gps_stop
        lat = [lat_start,lat_stop]
        lon = [lon_start,lon_stop]
        lat_res_per_pixel = (np.max(lat) - np.min(lat))/(ncol*np.sin(angle/360*2*np.pi))
        lon_res_per_pixel = (np.max(lon) - np.min(lon))/(ncol*np.cos(angle/360*2*np.pi))
        #if angle < 90: lat and lon are +ve
        #elif angle > 90 and < 180: lat is +ve but lon is -ve
        if angle > 90 and angle < 180:
            UPPER_LEFT_lat = np.max(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve
            UPPER_LEFT_lon = np.min(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is -ve
            LOWER_RIGHT_lat = np.min(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve
            LOWER_RIGHT_lon = np.max(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is -ve
        else:
            UPPER_LEFT_lat = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is +ve
            UPPER_LEFT_lon = np.min(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is +ve
            LOWER_RIGHT_lat = np.min(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is +ve
            LOWER_RIGHT_lon = np.max(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is +ve
            # print("upper left:{}\nlower right:{}".format(ul,lr))
        
        nrow,ncol = geotransformed_corrected_image.shape[0], geotransformed_corrected_image.shape[1]

        return {'left':UPPER_LEFT_lon,'top':UPPER_LEFT_lat,'right':LOWER_RIGHT_lon,'bottom':LOWER_RIGHT_lat,'lat_res':abs(lat_res_per_pixel),'lon_res':abs(lon_res_per_pixel),\
            'nrows':nrow,'ncols':ncol,'geotransformed_img':geotransformed_corrected_image}
    
    def get_bckgrnd_attr(self,bbox_list):    
        min_left = bbox_list[0]['left']
        max_top = bbox_list[0]['top']
        max_right = bbox_list[0]['right']
        min_bottom = bbox_list[0]['bottom']
        lon_res = bbox_list[0]['lon_res']
        lat_res = bbox_list[0]['lat_res']

        for dict in bbox_list:
            if dict['left'] < min_left:
                min_left = dict['left']
            if dict['top'] > max_top:
                max_top = dict['top']
            if dict['right'] > max_right:
                max_right = dict['right']
            if dict['bottom'] < min_bottom:
                min_bottom = dict['bottom']
            if dict['lon_res'] < lon_res:
                lon_res = dict['lon_res']
            if dict['lat_res'] < lat_res:
                lat_res = dict['lat_res']

        bckgrnd_ncols = ceil((max_right - min_left)/lon_res)
        bckgrnd_nrows = ceil((max_top - min_bottom)/lat_res)
        dist = Haversine((max_right,max_top),(min_left,max_top)).meters
        
        self.dist = dist
        self.left = min_left
        self.right = max_right
        self.top = max_top
        self.bottom = min_bottom

        return {'min_left':min_left,'max_top':max_top,'max_right':max_right,'min_bottom':min_bottom,'lon_res':lon_res,'lat_res':lat_res,\
            'bckgrnd_ncols':bckgrnd_ncols,'bckgrnd_nrows':bckgrnd_nrows,'dist':dist}
    
    def geotransformed_images_to_base64_opt(self,bckgrnd_attr,bbox_list):
        """
        bbox is a list of dict with keys (left,right,top,bottom,lat_res,lon_res,nrows,ncols,array)
        """
        bytes_list = []
        img_resize_list = []
        for dict in bbox_list:
            img = dict['geotransformed_img']
            # print(f'img shape: {img.shape}')
            ncols_from_left = int((dict['left'] - bckgrnd_attr['min_left'])/bckgrnd_attr['lon_res'])
            nrows_from_top = int((bckgrnd_attr['max_top'] - dict['top'])/bckgrnd_attr['lat_res'])
            # print(f'ncols_from_left:{ncols_from_left},nrows_from_top:{nrows_from_top}')
            array_rows = nrows_from_top + dict['nrows']
            array_cols = ncols_from_left + dict['ncols']
            bckgrnd = np.zeros((bckgrnd_attr['bckgrnd_nrows'],bckgrnd_attr['bckgrnd_ncols'],3),dtype=np.uint8)
            bckgrnd[nrows_from_top:array_rows,ncols_from_left:array_cols,:] = img
            alpha = np.where(bckgrnd[:,:,1]>0,255,0)
            image_strip = np.dstack((bckgrnd,alpha)).astype(np.uint8)
            img_resize = image_strip
            # img_resize = cv2.resize(image_strip,dsize=(bckgrnd_attr['bckgrnd_ncols']//scale,bckgrnd_attr['bckgrnd_nrows']//scale), interpolation=cv2.INTER_CUBIC)
            img_resize_list.append(img_resize)
            b = array_to_base64(img_resize)
            bytes_list.append(b)

        return img_resize_list,bytes_list
    
    def geotransformed_prediction_to_base64_opt(self,bckgrnd_attr,bbox_list,min_DN=0,max_DN=255):
        """
        bbox is a list of dict with keys (left,right,top,bottom,lat_res,lon_res,nrows,ncols,array)
        """
        bytes_list = []
        img_resize_list = []

        norm = colors.Normalize(vmin=min_DN, vmax=max_DN, clip=True) # data array is first mapped onto the range 0-1 
        mapper = cm.ScalarMappable(norm=norm, cmap='BrBG_r') #return mapper 
        for dict in bbox_list:
            img = dict['geotransformed_img']
            # print(f'img shape: {img.shape}')
            ncols_from_left = int((dict['left'] - bckgrnd_attr['min_left'])/bckgrnd_attr['lon_res'])
            nrows_from_top = int((bckgrnd_attr['max_top'] - dict['top'])/bckgrnd_attr['lat_res'])
            # print(f'ncols_from_left:{ncols_from_left},nrows_from_top:{nrows_from_top}')
            array_rows = nrows_from_top + dict['nrows']
            array_cols = ncols_from_left + dict['ncols']
            bckgrnd = np.zeros((bckgrnd_attr['bckgrnd_nrows'],bckgrnd_attr['bckgrnd_ncols']),dtype=np.uint8)
            bckgrnd[nrows_from_top:array_rows,ncols_from_left:array_cols] = img
            alpha = np.where(bckgrnd>0,255,0)
            rgba_img = mapper.to_rgba(bckgrnd,bytes=True)
            rgb_img = np.delete(rgba_img, 3, 2)
            image_strip = np.dstack((rgb_img,alpha)).astype(np.uint8)
            img_resize = image_strip
            # img_resize = cv2.resize(image_strip,dsize=(bckgrnd_attr['bckgrnd_ncols']//scale,bckgrnd_attr['bckgrnd_nrows']//scale), interpolation=cv2.INTER_CUBIC)
            img_resize_list.append(img_resize)
            b = array_to_base64(img_resize)
            bytes_list.append(b)

        return img_resize_list,bytes_list,mapper

    def latlon_to_rowcol(self,x,bckgrnd_attr,bbox_list,mapper):
        """
        apply function to df rows
        bbox_list (list of dicts): keys (left,right,top,bottom,lat_res,lon_res,nrows,ncols,geotransformed_img)
        bckgrnd_attr (dict): 'min_left','max_top','max_right','min_bottom','lon_res','lat_res',\
            'bckgrnd_ncols','bckgrnd_nrows'
        #y differs from nrows from top because images have the origin at the upper-left corner
        """
        d = {}
        lon_res = bbox_list[x['line_number']]['lon_res']
        lat_res = bbox_list[x['line_number']]['lat_res']
        d['x'] = bckgrnd_attr['bckgrnd_ncols'] - int(abs(bckgrnd_attr['max_right'] - x['Lon'])/lon_res)
        d['y'] = bckgrnd_attr['bckgrnd_nrows'] - int(abs(bckgrnd_attr['max_top'] - x['Lat'])/lat_res)

        d['x_general'] = bckgrnd_attr['bckgrnd_ncols'] - int(abs(bckgrnd_attr['max_right'] - x['Lon'])/bckgrnd_attr['lon_res'])
        d['y_general'] = bckgrnd_attr['bckgrnd_nrows'] - int(abs(bckgrnd_attr['max_top'] - x['Lat'])/bckgrnd_attr['lat_res'])

        d['ncols_from_left'] = bckgrnd_attr['bckgrnd_ncols'] - int(abs(bckgrnd_attr['max_right'] - x['Lon'])/lon_res)
        d['nrows_from_top'] = int(abs(bckgrnd_attr['max_top'] - x['Lat'])/lat_res)
        d['hex'] = colors.rgb2hex(mapper.to_rgba(x['Concentration']))
        return pd.Series(d)
    
    def rgb_canvas(self,scale=20):
        """
        view rgb images only
        """
        gps_image_dict = self.create_gps_image_dict(mode='rgb')
        bbox_list = []
        for k,v in gps_image_dict.items():
            corrected_rgb_img = cv2.resize(v['image'],dsize=(v['image'].shape[1]//scale,v['image'].shape[0]//scale), interpolation=cv2.INTER_CUBIC)
            gps_start = v['start']['lon'],v['start']['lat']
            gps_stop = v['stop']['lon'],v['stop']['lat']
            angle = self.calculate_flight_angle(gps_start,gps_stop)
            geotransformed_corrected_image = self.geotransform_image(corrected_rgb_img,angle)
            img_attr_dict = self.get_bounding_box(corrected_rgb_img,geotransformed_corrected_image,angle,gps_start,gps_stop)
            bbox_list.append(img_attr_dict)

        bckgrnd_attr = self.get_bckgrnd_attr(bbox_list)
        img_resize_list,bytes_list = self.geotransformed_images_to_base64_opt(bckgrnd_attr,bbox_list)

        # return img_resize_list,bytes_list
        #-------add wql pts--------
        if self.wql_df is not None:
            minima = self.wql_df['Concentration'].min()
            maxima = self.wql_df['Concentration'].max()
            norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True) # data array is first mapped onto the range 0-1 
            mapper = cm.ScalarMappable(norm=norm, cmap='BrBG_r')
            lat_lon_cols = self.wql_df.apply(lambda x:self.latlon_to_rowcol(x,bckgrnd_attr,bbox_list,mapper),axis=1)
            df = pd.concat([self.wql_df,lat_lon_cols],axis=1)
            # self.wql_df = df
            sorted_wql_df = df.sort_values(by=['Concentration'])
            conc_quantile = sorted_wql_df['Concentration'].quantile(np.linspace(0,1,5),interpolation="nearest")
            legend_wql = [(q,colors.rgb2hex(mapper.to_rgba(q))) for q in conc_quantile]
            return img_resize_list,bytes_list, df,legend_wql

        else:
            return img_resize_list,bytes_list, None, None

    def prediction_canvas(self):
        """
        view prediction images
        """
        gps_image_dict = self.create_gps_image_dict(mode='prediction')
        max_DN = 0
        min_DN = 255
        for k,v in gps_image_dict.items():
            corrected_rgb_img = cv2.resize(v['image'],dsize=(v['image'].shape[1]//self.scale,v['image'].shape[0]//self.scale), interpolation=cv2.INTER_AREA)
            if np.max(corrected_rgb_img) > max_DN:
                max_DN = np.max(corrected_rgb_img)
            if np.min(corrected_rgb_img) < min_DN:
                min_DN = np.min(corrected_rgb_img)

        bbox_list = []
        for k,v in gps_image_dict.items():
            corrected_rgb_img = cv2.resize(v['image'],dsize=(v['image'].shape[1]//self.scale,v['image'].shape[0]//self.scale), interpolation=cv2.INTER_AREA)
            gps_start = v['start']['lon'],v['start']['lat']
            gps_stop = v['stop']['lon'],v['stop']['lat']
            angle = self.calculate_flight_angle(gps_start,gps_stop)
            geotransformed_corrected_image = self.geotransform_image(corrected_rgb_img,angle)
            img_attr_dict = self.get_bounding_box(corrected_rgb_img,geotransformed_corrected_image,angle,gps_start,gps_stop)
            bbox_list.append(img_attr_dict)

        bckgrnd_attr = self.get_bckgrnd_attr(bbox_list)
        img_resize_list,bytes_list,mapper = self.geotransformed_prediction_to_base64_opt(bckgrnd_attr,bbox_list,min_DN=min_DN,max_DN=max_DN)

        # return img_resize_list,bytes_list
        if self.wql_df is not None:
            lat_lon_cols = self.wql_df.apply(lambda x:self.latlon_to_rowcol(x,bckgrnd_attr,bbox_list,mapper),axis=1)
            df = pd.concat([self.wql_df,lat_lon_cols],axis=1)
            # self.wql_df = df
            sorted_wql_df = df.sort_values(by=['Concentration'])
            conc_quantile = sorted_wql_df['Concentration'].quantile(np.linspace(0,1,5),interpolation="nearest")
            legend_wql = [(q,colors.rgb2hex(mapper.to_rgba(q))) for q in conc_quantile]
            # fig = self.plot_model_performance(bbox_list,bckgrnd_attr)
            # self.plot_model_performance(mapper)
            return img_resize_list,bytes_list, df,legend_wql#, fig

        else:
            return img_resize_list,bytes_list, None, None

    def plot_model_performance(self):
        prediction_fp_list = [join(self.fp_store,'Prediction',f) for f in sorted(listdir(join(self.fp_store,'Prediction'))) if 'Geotransformed' in f and 'masked' not in f]
      
        min_DN = 255
        max_DN = 0

        def latlon_to_rowcol(x,raster_dict,mapper,scale):
            d = {}
            line_img = raster_dict[x['line_number']]
            lon_res = line_img['lon_res']
            lat_res = line_img['lat_res']
            d['x'] = line_img['ncols'] - int(abs(line_img['left'] - x['Lon'])/lon_res)
            d['y'] = line_img['nrows'] - int(abs(line_img['top'] - x['Lat'])/lat_res)
            d['ncols_from_left'] = (line_img['ncols'] - int(abs(line_img['left'] - x['Lon'])/lon_res))//scale
            d['nrows_from_top'] = int(abs(line_img['top'] - x['Lat'])/lat_res)//scale
            d['hex'] = colors.rgb2hex(mapper.to_rgba(x['Concentration']))
            return pd.Series(d)

        df_list = []
        raster_dict = {}#{i:None for i in range(len(prediction_fp_list))}
        for file in prediction_fp_list:
            line_number = int(re.sub(r'^.*?image_line_','',file)[:2]) #take the first two characters since they have been padded to 2 digits
            try:
                r = rasterio.open(file)
            except Exception as E:
                sg.popup(f'{E}\n Cannot open image file',title='Error')
                r = None
                pass
            _, nrows, ncols = r.read().shape
            lon_res = abs(r.bounds[0] - r.bounds[2])/ncols
            lat_res = abs(r.bounds[1] - r.bounds[3])/nrows
            raster_dict[line_number] = {'rasterio':r,'left':r.bounds[0],'top':r.bounds[3],'nrows':nrows,'ncols':ncols,'lon_res':lon_res,'lat_res':lat_res}
            r_min =  np.min(r.read())
            r_max =  np.max(r.read())
            if r_min < min_DN:
                min_DN = r_min
            if r_max > max_DN:
                max_DN = r_max

        norm = colors.Normalize(vmin=min_DN, vmax=max_DN, clip=True) # data array is first mapped onto the range 0-1 
        mapper = cm.ScalarMappable(norm=norm, cmap='BrBG_r') #return mapper 
            
        lat_lon_cols = self.wql_df.apply(lambda x:latlon_to_rowcol(x,raster_dict,mapper,self.scale),axis=1)
        df = pd.concat([self.wql_df,lat_lon_cols],axis=1)
        
        pred_list = []
        obs_list = []
        hex_list = []
        for l, d in raster_dict.items():
            r = d['rasterio'].read(1)
            img_resize = cv2.resize(r,dsize=(d['ncols']//self.scale,d['nrows']//self.scale), interpolation=cv2.INTER_CUBIC)
            img_resize = np.fliplr(img_resize)
            sub_df = df[df['line_number'] == l]
            row_list, col_list = sub_df['nrows_from_top'],sub_df['ncols_from_left']
            pred = img_resize[row_list,col_list].flatten().tolist()
            obs = sub_df['Concentration'].tolist()
            hex_list = hex_list + sub_df['hex'].tolist()
            pred_list = pred_list + pred
            obs_list = obs_list + obs

        RMSE = np.sqrt(np.sum(np.subtract(pred_list,obs_list)**2)/len(pred_list))
        if len(obs_list) == 0 and len(pred_list) == 0:
            fig,ax = plt.subplots(figsize=(7,7))
            ax.set_title('No water quality points available\nfor evaluation of model performance')
            return fig
        else:
            r2 = r2_score(obs_list,pred_list)

            fig,ax = plt.subplots(figsize=(7,7))
            ax.scatter(pred_list,obs_list,alpha=0.5,c=hex_list)
            ax.axline((1, 1), slope=1,ls="--",c="black",label="1:1 line")
            ax.set_ylabel('Observed data')
            ax.set_xlabel('Predicted data')
            ax.text(min(pred_list),max(obs_list),'RMSE: {:.2f}'.format(RMSE))
            ax.text(min(pred_list),0.9*max(obs_list),r'$R^2$: {:.2f}'.format(r2))
            plt.legend(loc="lower right")
            # plt.show()

            return fig
        


#-----------------------view georeferenced images------------------------------


def wql_canvas(dim_dict,df,raster_dict,scale,mapper=None):
    """
    wql df has line_number column in int, has to convert dictionary line_number to int
    """
    wql_dict = {}
    for f,attr_dict in raster_dict.items():
        wql_dict[int(attr_dict['line_number'])] = {k:v for k,v in attr_dict.items() if k != 'line_number'}

    if mapper is None:
        minima = df['Concentration'].min()
        maxima = df['Concentration'].max()
        norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True) # data array is first mapped onto the range 0-1 
        mapper = cm.ScalarMappable(norm=norm, cmap='BrBG_r')

    def latlon_to_rowcol(x,dim_dict,wql_dict,mapper,scale):
        d = {}
        lon_res = wql_dict[x['line_number']]['lon_res']
        lat_res = wql_dict[x['line_number']]['lat_res']
        d['x'] = dim_dict['n_cols'] - int(abs(dim_dict['max_left'] - x['Lon'])/lon_res)
        d['y'] = dim_dict['n_rows'] - int(abs(dim_dict['max_top'] - x['Lat'])/lat_res)
        d['ncols_from_left'] = (dim_dict['n_cols'] - int(abs(dim_dict['max_left'] - x['Lon'])/lon_res))//scale
        d['nrows_from_top'] = int(abs(dim_dict['max_top'] - x['Lat'])/lat_res)//scale

        #                     (dim_dict['n_rows'] - int(abs(max_top - x['Lat'])/lat_res))
        # d['ncols_from_left'] = (dim_dict['n_cols'] - int(abs(max_left - x['Lon'])/lon_res))#//scale
#        d['nrows_from_top'] = (dim_dict['n_rows'] - int(abs(max_top - x['Lat'])/lat_res))#//scale
        d['hex'] = colors.rgb2hex(mapper.to_rgba(x['Concentration']))
        return pd.Series(d)

    lat_lon_cols = df.apply(lambda x:latlon_to_rowcol(x,dim_dict,wql_dict,mapper,scale),axis=1)
    df = pd.concat([df,lat_lon_cols],axis=1)
    sorted_wql_df = df.sort_values(by=['Concentration'])
    conc_quantile = sorted_wql_df['Concentration'].quantile(np.linspace(0,1,5),interpolation="nearest")
    legend_wql = [(q,colors.rgb2hex(mapper.to_rgba(q))) for q in conc_quantile]
    
    return df,legend_wql

def plot_model_performance(df,predicted_raster_dict):
    """
    df (pd df)
    predicted_raster_dict (dict): keys are filenames
    """
    wql_dict = {}
    # wql_dict = {int(attr_dict['line_number']):{k:v} for k,v in attr_dict.items() if k != 'line_number'}
    for f,attr_dict in predicted_raster_dict.items():
        wql_dict[int(attr_dict['line_number'])] = {k:v for k,v in attr_dict.items() if k != 'line_number'}
    #     wql_dict[int(attr_dict['line_number'])]['filename'] = join(fp_store,f)

    model_performance = {i:{} for i in range(len(df.index))}
    for index,row in df.iterrows():
        img = wql_dict[row['line_number']]['grey_scale_img']
        x = row['ncols_from_left']
        y = row['nrows_from_top']
        obs = row['Concentration']
        pred = img[y,x]
        model_performance[index]['obs'] = obs
        model_performance[index]['pred'] = pred

    model_performance
    obs = [d['obs'] for d in model_performance.values()]
    pred = [d['pred'] for d in model_performance.values()]
    RMSE = np.sqrt(np.sum(np.subtract(pred,obs)**2)/len(pred))
    if len(obs) == 0 and len(pred) == 0:
        fig,ax = plt.subplots(figsize=(7,7))
        ax.set_title('No water quality points available\nfor evaluation of model performance')
        return fig
    else:
        r2 = r2_score(obs,pred)

        fig,ax = plt.subplots(figsize=(7,7))
        ax.scatter(pred,obs,alpha=0.5,c="k")
        ax.axline((1, 1), slope=1,ls="--",c="black",label="1:1 line")
        ax.set_ylabel('Observed data')
        ax.set_xlabel('Predicted data')
        ax.text(min(pred),max(obs),'RMSE: {:.2f}'.format(RMSE))
        ax.text(min(pred),0.9*max(obs),r'$R^2$: {:.2f}'.format(r2))
        plt.legend(loc="lower right")
        return fig


def deg_to_dms(deg, type='lat'):
    """
    converts degree to dms
    type (str): 'lat' or 'lon'
    """
    decimals, number = math.modf(deg)
    d = int(number)
    m = int(decimals * 60)
    s = (deg - d - m / 60) * 3600.00
    compass = {
        'lat': ('N','S'),
        'lon': ('E','W')
    }
    compass_str = compass[type][0 if d >= 0 else 1]
    return '{}º{}\'{:.2f}"{}'.format(abs(d), abs(m), abs(s), compass_str)

#----------------------------------live update algo-------------------------------------------
def array_to_base64(array):
    img = Image.fromarray(array)
    with BytesIO() as bio:
        img.save(bio, format="PNG") #one-layer image e.g. predicted image cannot be shown cus needs to be RGB images
        del img
        b = bio.getvalue()
    return b

class ExtendedRGB:
    """
    creates an extended RGB image that allows for real-time time correction of images
    """
    def __init__(self,image_folder_filepath,test_gps_index, height, unique_gps_df):
        self.image_folder_filepath = image_folder_filepath
        self.test_gps_index = test_gps_index
        self.unique_gps_df = unique_gps_df
        self.band_width = int(1280/61)
        self.height = height
        fp = join(self.image_folder_filepath,'RawImages')
        rawfiles = [f for f in sorted(listdir(fp)) if isfile(join(fp, f))]
        self.rawfiles = rawfiles
        self.rawfiles_attr = [f.split('_') for f in rawfiles]

    def format_datetime(self,dt,second_delay=0,millisecond_delay=0):
        """
        dt (str)
        converts string datetime into datetime object
        """
        ms = dt.split(':')[3].zfill(3)
        dt = ':'.join(dt.split(':')[:3]) + ':' + ms
        dt = datetime.strptime(dt,'%H:%M:%S:%f')
        second_delta = timedelta(seconds=second_delay,milliseconds=millisecond_delay)
        dt_offset = dt - second_delta #reduce gps time so it can match the img time which is lagging behind
        return dt_offset#,formatted_dt

    def get_overlap_ratio(self,gps_start,gps_stop,time_start,time_stop,image_index_start,image_index_end):
        """
        gps_start (tuple of float): (lon_start,lat_start)
        gps_end (tuple of float): (lon_end,lat_end)
        time_start (datetime object)
        time_stop (datetime object)
        image_index_start (int)
        image_index_end (int)
        """
        # print("Getting overlap ratio...")
        pixel_size_at_sensor = 5.3 #um
        total_pixel_of_sensor_x = self.band_width #because hyperspectral images are line images, so actual pixel_x is 20
        focal_length = 16 #mm
        actual_size_of_sensor_x = pixel_size_at_sensor*total_pixel_of_sensor_x/1000 #mm
        fov_x = 2*math.atan(actual_size_of_sensor_x/2/focal_length)*180/math.pi #deg
        total_gnd_coverage_x = 2*self.height*math.tan(math.pi*fov_x/2/180) #metres #angle is converted to radians first
        time_diff = time_stop-time_start
        time_diff = time_diff.total_seconds()
        frame_rate = int((image_index_end - image_index_start)/time_diff)
        dist = Haversine(gps_start,gps_stop).meters
        speed = dist/time_diff
        # print("Avg speed of drone: {:.2f}, Avg frame rate: {}".format(speed,frame_rate))
        d = (1/frame_rate)*speed #distance covered by drone in 1 fps
        overlap_x = total_gnd_coverage_x - d
        overlap_x_ratio = overlap_x/total_gnd_coverage_x
        return overlap_x_ratio

    def format_img_dt(self,x):
        ms = x.split('-')[3].zfill(3)
        formatted_dt = '-'.join(x.split('-')[:3]) + '-' + ms
        return datetime.strptime(formatted_dt,'%H-%M-%S-%f')

    def calculate_flight_angle(self,gps_start,gps_stop):
        """
        gps_start (tuple of float): (lon_start,lat_start)
        gps_stop (tuple of float): (lon_end,lat_end)
        """
        # print("Calculating flight angle...")
        direction_vector = np.array([gps_stop]) - np.array([gps_start])
        direction_vector = direction_vector/np.linalg.norm(direction_vector) #convert to unit vector
        east_vector = np.array([1,0]) #measured from the horizontal as a reference
        angle = np.arccos(np.dot(direction_vector,east_vector))/(2*np.pi)*360 #direction vector already converted to a unit vector
       
        if np.cross(direction_vector,east_vector) > 0: #point outwards aka to the right of the east vector
            angle = 180 - angle 
            
        return angle[0]

    
    
    def update_datetime_index_BST(self,datetime_list,gps_image_dt_dict,OG=False):
        """
        this function assigns the matched datetime img to the datetime gps using binary tree search (BST)
        rawfiles_attr (list of tuples): list of tuples based on filename (index,_,dt,_) (already sorted)
        datetime_list (list of datetime objects): list of datetime from all image lines
        gps_image_dt_dict (nested dict): 
            keys (int): line_number 
            values (dict): 
        OG (bool)
        *do not use rawfiles_attr's image index as that is not the same as the list's index. 
        *using rawfile_attr's image index therein as the list's index will retrieve the wrong matched time 
        """
        if OG is False:
            OG = ''
        else:
            OG = '_OG'
        
        n_images = len(self.rawfiles_attr)-1
        
        for i,t in enumerate(datetime_list):
            l = 0 #a tuple of index,_,dt,_
            h = n_images #a tuple of index,_,dt,_
            mid = n_images//2 #a tuple of index (0),_,dt (2),_
            while l < h:
                guess = self.rawfiles_attr[mid][2]
                guess = self.format_img_dt(guess)
                if guess > t: #if guess is too high
                    h = mid -1 
                    mid = (l + h)//2
                elif guess < t: #if guess is too low
                    l = mid+1
                    mid = (l + h)//2
                else: #if guess = t
                    break
                # print(l,h,mid)
            closest_match = self.rawfiles_attr[mid]
            if i%2 == 0: #start
                gps_image_dt_dict[i//2]['start']['datetime_img'+OG] = self.format_img_dt(closest_match[2])
                gps_image_dt_dict[i//2]['start']['index_img'+OG] = mid#int(closest_match[0])
            else:
                gps_image_dt_dict[i//2]['stop']['datetime_img'+OG] = self.format_img_dt(closest_match[2])
                gps_image_dt_dict[i//2]['stop']['index_img'+OG] = mid#int(closest_match[0])

        return gps_image_dt_dict

    def raw_to_img(self,fp,bands=(38,23,15)):
        """
        fp (str): filepath of the raw image
        bands (tuple of int): 3 tuple of list of the rgb bands e.g. ()
        outputs arrays of rgb images only
        """
        try:
            scene_infile = open(fp,'rb')
            scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1280*1024)
        except Exception as E:
            print(f'image file cannot be opened: {E}')
            pass
        reshaped_raw = scene_image_array.reshape(1024,1280)
        image_array = []
        for i in bands:
            row_start = i*self.band_width
            row_end = i*self.band_width + self.band_width
            band_array = reshaped_raw[:,row_start:row_end]
            image_array.append(band_array)
        rgb_img = np.dstack(image_array)
        return rgb_img
    
    def overlap_rgb_images(self,img_arrays,overlap_ratio,reverse=False):
        """
        method (str): Default: recursive. naive or recursive
        img_arrays (list of arrays)
        Difference between naive and recursive may not be big if fps is high, 
        but the quality of the blending is dependent on height, speed of drone because that determines the overlap ratio
        naive has much much faster performance
        """
      
        def destriping_img(img):
            """
            destriping array.shape = (61,1024)
            """
            bands = [38,23,15]
            white_fp = join(self.image_folder_filepath,"WhiteRef")
            fp_list_white = [join(white_fp,i) for i in listdir(white_fp)]
            hyperspectral_img_list_white = [raw_to_hyperspectral_img(f) for f in fp_list_white]
            hyperspectral_white = np.mean(hyperspectral_img_list_white,axis=0)
            hyperspectral_white_array = np.mean(hyperspectral_white,axis=1)
            adjust_DN = lambda x,max_DN: max_DN/x
            rgb_array = hyperspectral_white_array[:,bands]

            adjusted_rgb_array = [adjust_DN(rgb_array[:,i],np.max(rgb_array[:,i])) for i in range(len(bands))]
            adjusted_rgb_array = np.transpose(np.vstack(adjusted_rgb_array))

            nrows,ncols,c = img.shape
            repeated_DN_rgb = np.repeat(adjusted_rgb_array[:,:,np.newaxis],ncols,axis=2)
            repeated_DN_rgb = np.swapaxes(repeated_DN_rgb,1,2)
            destriped_img = repeated_DN_rgb*img
            destriped_img = np.where(destriped_img>255,255,destriped_img)
            return destriped_img.astype(np.uint8)

        # band_width = img_arrays[0].shape[1] #int(1280/61)
        overlapped_cols = int(overlap_ratio*self.band_width)
        # print(f'overlapped_cols: {overlapped_cols}')

        trimmed_initial = [img_arrays[0]]
        trimmed_subsequent = [img[:,overlapped_cols:self.band_width] for img in img_arrays[1:]]
        trimmed_image_array = trimmed_initial+trimmed_subsequent
        stitched_img = np.hstack(trimmed_image_array)
        # destriped_stitched_rgb = destriping_img(stitched_img,destriping_array)
        destriped_stitched_rgb = destriping_img(stitched_img)
        return destriped_stitched_rgb if reverse is False else np.fliplr(destriped_stitched_rgb)
        # return stitched_img if reverse is False else np.fliplr(stitched_img)

    def create_dict(self):
        filtered_unique_gps_df = self.unique_gps_df.iloc[self.test_gps_index,:] #reset index so index starts from 0 again
        # print(filtered_unique_gps_df)
        number_of_lines_counter = 0
        if len(self.test_gps_index)%2 == 0:
            number_of_lines = len(self.test_gps_index)//2
        else:
            raise ValueError("Number of indices are not even! Each line must have a start and stop point indicated")

        gps_image_dt_dict = {i:{} for i in range(number_of_lines)} #keys are line_number
        for index, rows in filtered_unique_gps_df.reset_index().iterrows():
            dt = rows[1] #datetime column
            lat = rows[2]
            lon = rows[3]
            # print(dt)
            if index %2 == 0: #start
                # datetime_std,formatted_dt = format_datetime(dt,second_delay = 1, millisecond_delay = 999)
                datetime_gps = self.format_datetime(dt,second_delay = 2, millisecond_delay = 100)
                datetime_gps_OG = self.format_datetime(dt,second_delay = 0, millisecond_delay = 0)
                # gps_image_dt_dict[number_of_lines_counter]['start'] = {'datetime_gps':datetime_std,'formatted_dt_gps':formatted_dt}
                gps_image_dt_dict[number_of_lines_counter]['start'] = {'datetime_gps':datetime_gps,'datetime_gps_OG':datetime_gps_OG,'lat':lat,'lon':lon}
            else: #stop
                # datetime_std, formatted_dt = format_datetime(dt,second_delay = 0, millisecond_delay = 0)
                datetime_gps = self.format_datetime(dt,second_delay = 0, millisecond_delay = 0)
                # gps_image_dt_dict[number_of_lines_counter]['stop'] = {'datetime_gps':datetime_std,'formatted_dt_gps':formatted_dt}
                gps_image_dt_dict[number_of_lines_counter]['stop'] = {'datetime_gps':datetime_gps,'datetime_gps_OG':datetime_gps,'lat':lat,'lon':lon}
                number_of_lines_counter+=1
        
        return gps_image_dt_dict
            # print(datetime_std)
    
    def add_reverse(self,general_dict):
        for line_number,dict in general_dict.items():
            gps_start = dict['start']['lon'],dict['start']['lat']
            gps_stop = dict['stop']['lon'],dict['stop']['lat']
            general_dict[line_number]['flight_angle'] = self.calculate_flight_angle(gps_start,gps_stop)
            if gps_start[1] > gps_stop[1]:
                general_dict[line_number]['reverse'] = True
            else:
                general_dict[line_number]['reverse'] = False
        
        return general_dict

    def get_datetime_list(self,general_dict):
        
        datetime_list = []
        datetime_list_OG = []
        for timestamps in general_dict.values():
            start_dt = timestamps['start']['datetime_gps']
            stop_dt = timestamps['stop']['datetime_gps']
            datetime_list.append(start_dt)
            datetime_list.append(stop_dt)

            start_dt_OG = timestamps['start']['datetime_gps_OG']
            stop_dt_OG = timestamps['stop']['datetime_gps_OG']
            datetime_list_OG.append(start_dt_OG)
            datetime_list_OG.append(stop_dt_OG)

        return datetime_list,datetime_list_OG

    def main(self):
        """
        outputs a dict with gps and img attributes for each start and stop points
        """
        general_dict = self.create_dict()
        general_dict = self.add_reverse(general_dict)
        datetime_list,datetime_list_OG = self.get_datetime_list(general_dict)
        # general_dict = self.update_datetime_index(datetime_list,general_dict)
        # general_dict = self.update_datetime_index(datetime_list_OG,general_dict,OG=True)
        general_dict = self.update_datetime_index_BST(datetime_list,general_dict)
        general_dict = self.update_datetime_index_BST(datetime_list_OG,general_dict,OG=True)

        for line_number, dict in general_dict.items():
            start_index = dict['start']['index_img'] #extended index
            stop_index = dict['stop']['index_img'] #extended index
            selected_filenames = self.rawfiles[start_index:stop_index+1]
            selected_filenames = [join(self.image_folder_filepath,"RawImages",f) for f in selected_filenames]
            list_of_dt = [self.format_img_dt(f[2]) for f in self.rawfiles_attr[start_index:stop_index+1]]
            general_dict[line_number]['datetime_list'] = list_of_dt
            # img_arrays = list(map(raw_to_img,selected_filenames)) #list of rgb arrays
            gps_start = (dict['start']['lon'],dict['start']['lat'])
            gps_stop = (dict['stop']['lon'],dict['stop']['lat'])
            time_start = dict['start']['datetime_img_OG'] #use img time instead of gps time because framerate depends on img time
            time_stop = dict['stop']['datetime_img_OG']
            image_index_start = dict['start']['index_img_OG'] #but using the image index is not the actual correct image index
            image_index_stop = dict['stop']['index_img_OG']
            OR_x = self.get_overlap_ratio(gps_start,gps_stop,time_start,time_stop,image_index_start,image_index_stop)
            general_dict[line_number]['overlap_ratio'] = OR_x
            img_arrays = [self.raw_to_img(f) for f in selected_filenames] #list comprehension performs same as map
            stitched_rgb = self.overlap_rgb_images(img_arrays,OR_x,reverse=dict['reverse'])
            # print(f'stitched rgb shape: {stitched_rgb.shape}')
            general_dict[line_number]['stitched_rgb'] = stitched_rgb

        return general_dict,datetime_list

class LiveCorrection:
    def __init__(self,general_dict,datetime_list,time_delay,scale=20):
        self.general_dict = general_dict
        self.datetime_list = datetime_list
        s,ms = divmod(time_delay,1000)
        self.seconds_delay = s
        self.milliseconds_delay = ms
        self.band_width = int(1280/61)
        self.scale = scale


    def get_index_datetime_list_BST(self,datetime_list,start_dt,stop_dt,seconds_delay=None,milliseconds_delay=None):
        """
        given the list of datetime_list (already sorted), and corrected start and stop times
        returns the start & stop index within datetime_list, then can help us eventually trace back the col_index of images
        """
        if seconds_delay is None and milliseconds_delay is None:
            time_delta = timedelta(seconds=self.seconds_delay,milliseconds=self.milliseconds_delay)
        else:
            time_delta = timedelta(seconds=seconds_delay,milliseconds=milliseconds_delay)
        
        start_dt_corrected,stop_dt_corrected = start_dt - time_delta, stop_dt - time_delta
        n_datetime_list = len(datetime_list)-1

        start_stop_index = []
        # start_stop_dt = []
        for dt in [start_dt_corrected,stop_dt_corrected]:
            l = 0
            h = n_datetime_list
            mid = (l+h)//2
            while l < h:
                guess = datetime_list[mid]
                if guess < dt: #if guess is too low
                    l = mid + 1
                    mid = (l+h)//2
                elif guess > dt: #if guess is too high
                    h = mid - 1
                    mid = (l+h)//2
                else:
                    break
                # print(l,h,mid)
            start_stop_index.append(mid)
            # start_stop_dt.append(datetime_list[mid])

        return start_stop_index


    def get_corrected_image_opt(self,extended_rgb_img,reverse,corrected_start_index, corrected_stop_index,overlap_ratio):
        """
        converts image index to corresponding img column index then trim images
        extended_rgb_img (array): extended rgb image that needs to be trimmed
        corected_start_index (int): index pointing to the datetime index index
        reverse (bool): whether image has been flipped
        """
        band_width = int(1280/61)
        col_index_function = lambda i,bandwidth,overlap_cols: i*(bandwidth - overlap_cols) #where i = index
        #col_index_function is a function that converts datetime index to column index
        overlap_cols = int(overlap_ratio*band_width)
        if reverse == False:
            corrected_img_start_index = col_index_function(corrected_start_index,band_width,overlap_cols)
            corrected_img_stop_index = col_index_function(corrected_stop_index,band_width,overlap_cols)
        else:
            ncols = extended_rgb_img.shape[1]
            corrected_img_stop_index = ncols - col_index_function(corrected_start_index,band_width,overlap_cols)
            corrected_img_start_index = ncols - col_index_function(corrected_stop_index,band_width,overlap_cols)
        print(f'corrected_img_start_index: {corrected_img_start_index}, corrected_img_stop_index:{corrected_img_stop_index}')
        corrected_rgb_img = extended_rgb_img[:,corrected_img_start_index:corrected_img_stop_index+1]
        print('Before correction shape:{}'.format(extended_rgb_img.shape))
        print('After correction shape:{}'.format(corrected_rgb_img.shape))
        return cv2.resize(corrected_rgb_img,dsize=(corrected_rgb_img.shape[1]//self.scale,corrected_rgb_img.shape[0]//self.scale), interpolation=cv2.INTER_CUBIC)

    def geotransform_corrected_image(self,corrected_rgb_img,angle):
        """
        returns a geotransformed corrected img
        """
        nrow,ncol,_ = corrected_rgb_img.shape
        img = np.flipud(np.array(corrected_rgb_img)) #flip vertically
        center = (ncol//2,nrow//2)
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,1) #center, angle, scale
        cosofRotationMatrix = np.abs(rotation_matrix[0][0]) #scale*cos(angle)
        sinofRotationMatrix = np.abs(rotation_matrix[0][1]) #scale*sin(angle)

        newImageHeight = int((ncol * sinofRotationMatrix) +
                            (nrow * cosofRotationMatrix))
        newImageWidth = int((ncol * cosofRotationMatrix) +
                            (nrow * sinofRotationMatrix))

        rotation_matrix[0][2] += (newImageWidth/2) - center[0]
        rotation_matrix[1][2] += (newImageHeight/2) - center[1]

        rotatingimage = cv2.warpAffine(img, rotation_matrix, (newImageWidth, newImageHeight))

        return rotatingimage

    def get_bounding_box(self,corrected_rgb_img,geotransformed_corrected_image,angle,gps_start,gps_stop):
        """
        gets bounding box (upper left (lat,lon), lower right (lat,lon)) of a corrcted geotransformed image
        returns bbox and lat,lon resolution
        nrows, ncols are dimensions of the geotransformed image in the returned output
        """
        nrow,ncol,_ = corrected_rgb_img.shape
        lon_start, lat_start = gps_start
        lon_stop, lat_stop = gps_stop
        lat = [lat_start,lat_stop]
        lon = [lon_start,lon_stop]
        lat_res_per_pixel = (np.max(lat) - np.min(lat))/(ncol*np.sin(angle/360*2*np.pi))
        lon_res_per_pixel = (np.max(lon) - np.min(lon))/(ncol*np.cos(angle/360*2*np.pi))
        #if angle < 90: lat and lon are +ve
        #elif angle > 90 and < 180: lat is +ve but lon is -ve
        if angle > 90 and angle < 180:
            UPPER_LEFT_lat = np.max(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve
            UPPER_LEFT_lon = np.min(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is -ve
            LOWER_RIGHT_lat = np.min(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve
            LOWER_RIGHT_lon = np.max(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is -ve
        else:
            UPPER_LEFT_lat = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is +ve
            UPPER_LEFT_lon = np.min(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is +ve
            LOWER_RIGHT_lat = np.min(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is +ve
            LOWER_RIGHT_lon = np.max(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel #lon_res_per_pixel is +ve
            # print("upper left:{}\nlower right:{}".format(ul,lr))
        
        nrow,ncol,_ = geotransformed_corrected_image.shape

        return {'left':UPPER_LEFT_lon,'top':UPPER_LEFT_lat,'right':LOWER_RIGHT_lon,'bottom':LOWER_RIGHT_lat,'lat_res':abs(lat_res_per_pixel),'lon_res':abs(lon_res_per_pixel),\
            'nrows':nrow,'ncols':ncol,'geotransformed_img':geotransformed_corrected_image}
        # return (ul,lr),lat_res_per_pixel,lon_res_per_pixel

    def get_bckgrnd_attr(self,bbox_list):
        
        min_left = bbox_list[0]['left']
        max_top = bbox_list[0]['top']
        max_right = bbox_list[0]['right']
        min_bottom = bbox_list[0]['bottom']
        lon_res = bbox_list[0]['lon_res']
        lat_res = bbox_list[0]['lat_res']

        for dict in bbox_list:
            if dict['left'] < min_left:
                min_left = dict['left']
            if dict['top'] > max_top:
                max_top = dict['top']
            if dict['right'] > max_right:
                max_right = dict['right']
            if dict['bottom'] < min_bottom:
                min_bottom = dict['bottom']
            if dict['lon_res'] < lon_res:
                lon_res = dict['lon_res']
            if dict['lat_res'] < lat_res:
                lat_res = dict['lat_res']

        bckgrnd_ncols = ceil((max_right - min_left)/lon_res)
        bckgrnd_nrows = ceil((max_top - min_bottom)/lat_res)
        return {'min_left':min_left,'max_top':max_top,'max_right':max_right,'min_bottom':min_bottom,'lon_res':lon_res,'lat_res':lat_res,\
            'bckgrnd_ncols':bckgrnd_ncols,'bckgrnd_nrows':bckgrnd_nrows}

   

    def geotransformed_images_to_base64_opt(self,bckgrnd_attr,bbox_list):
        """
        bbox is a list of dict with keys (left,right,top,bottom,lat_res,lon_res,nrows,ncols,array)
        """

        bytes_list = []
        img_resize_list = []
        for dict in bbox_list:
            img = dict['geotransformed_img']
            # print(f'img shape: {img.shape}')
            ncols_from_left = int((dict['left'] - bckgrnd_attr['min_left'])/bckgrnd_attr['lon_res'])
            nrows_from_top = int((bckgrnd_attr['max_top'] - dict['top'])/bckgrnd_attr['lat_res'])
            # print(f'ncols_from_left:{ncols_from_left},nrows_from_top:{nrows_from_top}')
            array_rows = nrows_from_top + dict['nrows']
            array_cols = ncols_from_left + dict['ncols']
            bckgrnd = np.zeros((bckgrnd_attr['bckgrnd_nrows'],bckgrnd_attr['bckgrnd_ncols'],3),dtype=np.uint8)
            bckgrnd[nrows_from_top:array_rows,ncols_from_left:array_cols,:] = img
            alpha = np.where(bckgrnd[:,:,1]>0,255,0)
            image_strip = np.dstack((bckgrnd,alpha)).astype(np.uint8)
            img_resize = image_strip
            # img_resize = cv2.resize(image_strip,dsize=(bckgrnd_attr['bckgrnd_ncols']//scale,bckgrnd_attr['bckgrnd_nrows']//scale), interpolation=cv2.INTER_CUBIC)
            img_resize_list.append(img_resize)
            b = array_to_base64(img_resize)
            bytes_list.append(b)

        return img_resize_list,bytes_list
    
    def main(self):
        bbox_list = []
        # geotransformed_corrected_images_dict = {k:{} for k in gps_image_dt_dict.keys()}
        for k,dict in self.general_dict.items():
            start_dt, stop_dt = dict['start']['datetime_gps_OG'],dict['stop']['datetime_gps_OG']
            # print('len(dict datetime_list):{}'.format(len(dict['datetime_list'])))
            # corrected_start_index, corrected_stop_index = self.get_index_datetime_list(dict['datetime_list'],start_dt,stop_dt)
            corrected_start_index, corrected_stop_index = self.get_index_datetime_list_BST(dict['datetime_list'],start_dt,stop_dt)
            # corrected_rgb_img = self.get_corrected_image(dict['stitched_rgb'],dict['reverse'],corrected_start_index, corrected_stop_index,dict['overlap_ratio']) #corrected means corrected for timestamps
            self.general_dict[k]['start']['index_corrected'] = corrected_start_index + self.general_dict[k]['start']['index_img']
            self.general_dict[k]['stop']['index_corrected'] = corrected_stop_index + self.general_dict[k]['start']['index_img']
            corrected_rgb_img = self.get_corrected_image_opt(dict['stitched_rgb'],dict['reverse'],corrected_start_index, corrected_stop_index,dict['overlap_ratio']) #corrected means corrected for timestamps
            angle = dict['flight_angle']
            gps_start = dict['start']['lon'],dict['start']['lat']
            gps_stop = dict['stop']['lon'],dict['stop']['lat']
            geotransformed_corrected_image = self.geotransform_corrected_image(corrected_rgb_img,angle) #conduct affine transformation
            img_attr_dict = self.get_bounding_box(corrected_rgb_img,geotransformed_corrected_image,angle,gps_start,gps_stop)
            bbox_list.append(img_attr_dict)
            # geotransformed_corrected_images_dict[line_number] ['img_attr_dict']= img_attr_dict

        bckgrnd_attr = self.get_bckgrnd_attr(bbox_list)
        # print(bckgrnd_attr)
        # img_resize_list,bytes_list = self.geotransformed_images_to_base64(bckgrnd_attr,bbox_list)
        img_resize_list,bytes_list = self.geotransformed_images_to_base64_opt(bckgrnd_attr,bbox_list)
        return img_resize_list,bytes_list,bckgrnd_attr,bbox_list

    def correct_individual_lines(self,bbox_list,bckgrnd_attr,seconds_delay,milliseconds_delay,line_number):
        """
        correct individual lines based on the specification of the line_number
        updates only 1 line and leaves the remaining lines same
        rather than reconstructing and building the entire lines of images, 
        the function returns the updated (corrected) image
        In the Tk.Canvas, we can then delete the previous line image and just draw this updated line image on top of the rest of the lines
        use the line_number to keep 
        >>> returns img_resize,b
        """
        if len(self.general_dict.keys()) != len(bbox_list):
            raise Exception("bbox list is not the same length as self.general_dict!")

        img_line = self.general_dict[line_number]
        start_dt, stop_dt = img_line['start']['datetime_gps_OG'],img_line['stop']['datetime_gps_OG']
        corrected_start_index, corrected_stop_index = self.get_index_datetime_list_BST(img_line['datetime_list'],start_dt,stop_dt,seconds_delay=seconds_delay,milliseconds_delay=milliseconds_delay)
        #datetime_list is the list of raw imges selected from start_index to stop_index
        #get_index_datetime_list_BST returns the indices within datetime_list, so we will need to add back the start_index to obtain back the raw img indices
        self.general_dict[line_number]['start']['index_corrected'] = corrected_start_index + self.general_dict[line_number]['start']['index_img']
        self.general_dict[line_number]['stop']['index_corrected'] = corrected_stop_index + self.general_dict[line_number]['start']['index_img']
        corrected_rgb_img = self.get_corrected_image_opt(img_line['stitched_rgb'],img_line['reverse'],corrected_start_index, corrected_stop_index,img_line['overlap_ratio']) #corrected means corrected for timestamps
        angle = img_line['flight_angle']
        gps_start = img_line['start']['lon'],img_line['start']['lat']
        gps_stop = img_line['stop']['lon'],img_line['stop']['lat']
        geotransformed_corrected_image = self.geotransform_corrected_image(corrected_rgb_img,angle) #conduct affine transformation
        img_attr_dict = self.get_bounding_box(corrected_rgb_img,geotransformed_corrected_image,angle,gps_start,gps_stop)
        
        img = img_attr_dict['geotransformed_img']
        # print(f'img shape: {img.shape}')
        ncols_from_left = int((img_attr_dict['left'] - bckgrnd_attr['min_left'])/bckgrnd_attr['lon_res'])
        nrows_from_top = int((bckgrnd_attr['max_top'] - img_attr_dict['top'])/bckgrnd_attr['lat_res'])
        # print(f'ncols_from_left:{ncols_from_left},nrows_from_top:{nrows_from_top}')
        array_rows = nrows_from_top + img_attr_dict['nrows']
        array_cols = ncols_from_left + img_attr_dict['ncols']
        bckgrnd = np.zeros((bckgrnd_attr['bckgrnd_nrows'],bckgrnd_attr['bckgrnd_ncols'],3),dtype=np.uint8)
        bckgrnd[nrows_from_top:array_rows,ncols_from_left:array_cols,:] = img
        alpha = np.where(bckgrnd[:,:,1]>0,255,0)
        image_strip = np.dstack((bckgrnd,alpha)).astype(np.uint8)
        img_resize = image_strip
        # img_resize = cv2.resize(image_strip,dsize=(bckgrnd_attr['bckgrnd_ncols']//scale,bckgrnd_attr['bckgrnd_nrows']//scale), interpolation=cv2.INTER_CUBIC)
        b = array_to_base64(img_resize)

        return img_resize,b

    def save_corrected_indices(self,fp):
        corrected_indices = [{'line_number':i,'start':d['start']['index_corrected'],'stop':d['stop']['index_corrected']} for i,d in self.general_dict.items()]
        with open(join(fp,'corrected_indices.json'), 'w') as fout:
            json.dump(corrected_indices , fout)

        return

def calculate_correlation_overlap(img_list):
    """
    img_list (list of np arrays) from live_correction function
    returns a tuple (correlation coefficient, p-value)
    """
    corr_coeff_list = []
    for i in range(len(img_list)-1):
        curr_img, next_img = img_list[i], img_list[i+1]
        curr_img_alpha = np.where(curr_img[:,:,1]>0,1,0)
        next_img_alpha = np.where(next_img[:,:,1]>0,1,0)
        alpha_intersect = curr_img_alpha*next_img_alpha
        alpha_intersect = np.repeat(alpha_intersect[:,:,np.newaxis],3,axis=2)
        
        curr_intersect = alpha_intersect*curr_img[:,:,:3]
        next_intersect = alpha_intersect*next_img[:,:,:3]
        # curr_intersect_corr = curr_intersect[curr_intersect!=0]
        # next_intersect_corr = next_intersect[next_intersect!=0]
        # print(curr_intersect_corr.shape,next_intersect_corr.shape)
        corr_coeff,p_value = pearsonr(curr_intersect.flatten(),next_intersect.flatten())
        corr_coeff_list.append(corr_coeff)
    return corr_coeff_list,p_value

#-----------------------------------masking algo----------------------------------------
def cut_into_512(img):
    """
    img has to be opened by cv2.imread(fp,1) #BGR
    """
    def pad_images(img):
        nrow,ncol = img.shape[0],img.shape[1]
        if ncol > 512:
            raise ValueError("Img col is > 512!")
        if len(img.shape) == 3:
            pad_img = np.zeros((512,512,3))
            pad_img[:,:ncol,:] = img
            pad_img = pad_img.astype(np.uint8)
        else:
            pad_img = np.zeros((512,512))
            pad_img[:,:ncol] = img
            pad_img = pad_img.astype(np.uint8)
            
        return pad_img

    nrow,ncol = img.shape[0],img.shape[1]
    if ncol > 512:
        cut_images = {}
        for index,i in enumerate(range(0,nrow,512)): #nrow
            for jindex,j in enumerate(range(0,ncol, 512)): #ncols
                if len(img.shape) == 3:
                    if j+512 <= ncol:
                        cut_images[(index,jindex)] = {'image':img[i:i+512,j:j+512,:],'padded':False}
                        # cut_images.append(img[i:i+512,j:j+512,:])
                    else:
                        padded_img = pad_images(img[i:i+512,j:ncol,:])
                        cut_images[(index,jindex)] = {'image':padded_img,'padded':True,'ncol':ncol}
                        # cut_images.append(padded_img)
                else:
                    if j+512 <= ncol:
                        cut_images[(index,jindex)] = {'image':img[i:i+512,j:j+512],'padded':False}
                        # cut_images.append(img[i:i+512,j:j+512])
                    else:
                        padded_img = pad_images(img[i:i+512,j:ncol])
                        cut_images[(index,jindex)] = {'image':padded_img,'padded':True,'ncol':ncol}
                        # cut_images.append(padded_img)
    else:
        cut_images = None
        print("ncol is already 512!")
    return cut_images #returns cut_dict

def load_unet_model(fp,name_checkpoint="SMI_checkpoints_1"):
    """"
    fp (str): folder path to where the model is located
    """
    checkpoints_path = join(fp,name_checkpoint)
    model = model_from_checkpoint_path(checkpoints_path)
    return model

def predict_mask(model,cut_images):
    """
    model (unet)
    cut_images (dict): with keys image, padded, ncol (if padded is True)
    mask is written to the cut_images with key ['mask']
    """
    for k,v in cut_images.items():
        out = model.predict_segmentation(
        inp=v['image'],
        # out_fname="/tmp/out.png"
        out_fname=None)
        out = np.array(out, dtype='uint8') #size is 256x256
        out_resize = cv2.resize(out, (512,512), interpolation= cv2.INTER_LINEAR) #resized mask
        cut_images[k]['mask'] = out_resize
    
    return cut_images

def mask_to_rgb(mask,img,classify=False):
    """
    classifies the objects into caissons and vessels by assigning a different colour
    """
    img_masked = img.copy()
    if classify is True:
        img_masked[mask == 2] = 128 #mask caisson
        img_masked[mask==1] = 0 #mask vessels
    else:
        img_masked[mask != 0] = 0
    return img_masked

def load_xgb_segmentation_model(fp):
    """"
    fp (str): folder path to where the model is located
    """
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(fp)
    clf._Booster = booster
    clf._le = LabelEncoder().fit([0,1,2])

    return clf

def XGBoost_segmentation(image_fp,model):
    test_tif = cv2.imread(image_fp)
    rgb_tif = cv2.cvtColor(test_tif, cv2.COLOR_BGR2RGB)
    df_dict = {c:rgb_tif[:,:,i].flatten() for c,i in zip(['r','g','b'],range(3))}
    df = pd.DataFrame.from_dict(df_dict)
    # y_pred_proba = model.predict_proba(df)
    recon_mask = model.predict(df)
    recon_mask = recon_mask.reshape(rgb_tif.shape[0],rgb_tif.shape[1])
    # masked_img = mask_to_rgb(recon_mask,rgb_tif,classify)
    #evaluate prediction
    
    return recon_mask#,masked_img


#--------------------Retrieval of environmental variables------------------------
def get_env_locations():
    locations = {'S109': 'Ang Mo Kio Avenue 5',\
        'S117': 'Banyan Road',\
        'S50': 'Clementi Road',\
        'S107': 'East Coast Parkway',\
        'S43': 'Kim Chuan Road',\
        'S108': 'Marina Gardens Drive',\
        'S44': 'Nanyang Avenue',\
        'S106': 'Pulau Ubin',\
        'S122': 'Sembawang Road',\
        'S60': 'Sentosa',\
        'S115': 'Tuas South Avenue 3',\
        'S24': 'Upper Changi Road North',\
        'S116': 'West Coast Highway',\
        'S104': 'Woodlands Avenue 9',\
        'S100': 'Woodlands Road'}
    return locations

def get_locations_metadata():
    meta_data = [{'id': 'S109', 'device_id': 'S109', 'name': 'Ang Mo Kio Avenue 5', 'location': {'latitude': 1.3764, 'longitude': 103.8492}},\
        {'id': 'S117', 'device_id': 'S117', 'name': 'Banyan Road', 'location': {'latitude': 1.256, 'longitude': 103.679}},\
        {'id': 'S50', 'device_id': 'S50', 'name': 'Clementi Road', 'location': {'latitude': 1.3337, 'longitude': 103.7768}},\
        {'id': 'S107', 'device_id': 'S107', 'name': 'East Coast Parkway', 'location': {'latitude': 1.3135, 'longitude': 103.9625}},\
        {'id': 'S43', 'device_id': 'S43', 'name': 'Kim Chuan Road', 'location': {'latitude': 1.3399, 'longitude': 103.8878}},\
        {'id': 'S108', 'device_id': 'S108', 'name': 'Marina Gardens Drive', 'location': {'latitude': 1.2799, 'longitude': 103.8703}},\
        {'id': 'S44', 'device_id': 'S44', 'name': 'Nanyang Avenue', 'location': {'latitude': 1.34583, 'longitude': 103.68166}},\
        {'id': 'S106', 'device_id': 'S106', 'name': 'Pulau Ubin', 'location': {'latitude': 1.4168, 'longitude': 103.9673}},\
        {'id': 'S122', 'device_id': 'S122', 'name': 'Sembawang Road', 'location': {'latitude': 1.41731, 'longitude': 103.8249}},\
        {'id': 'S60', 'device_id': 'S60', 'name': 'Sentosa', 'location': {'latitude': 1.25, 'longitude': 103.8279}},\
        {'id': 'S115', 'device_id': 'S115', 'name': 'Tuas South Avenue 3', 'location': {'latitude': 1.29377, 'longitude': 103.61843}},\
        {'id': 'S24', 'device_id': 'S24', 'name': 'Upper Changi Road North', 'location': {'latitude': 1.3678, 'longitude': 103.9826}},\
        {'id': 'S116', 'device_id': 'S116', 'name': 'West Coast Highway', 'location': {'latitude': 1.281, 'longitude': 103.754}},\
        {'id': 'S104', 'device_id': 'S104', 'name': 'Woodlands Avenue 9', 'location': {'latitude': 1.44387, 'longitude': 103.78538}},\
        {'id': 'S100', 'device_id': 'S100', 'name': 'Woodlands Road', 'location': {'latitude': 1.4172, 'longitude': 103.74855}}]
    return meta_data

def convert_string_to_dt(date_str,time_str):
    """
    date_str (str): YYYY-MM-DD (%Y-%m-%d)
    time_str (str): HH-MM-SS (%H-%M-%S)
    """
    dt_str = date_str + "_" + time_str
    return datetime.strptime(dt_str,'%Y-%m-%d_%H-%M-%S')

def create_nested_API_response(date_time_dict,env_params):
    """
    date_time_dict (list of dict): list of dicts, where each dict contains 'start' and 'end' keys, 
        and values are in %Y-%m-%d_%H-%M-%S
    env_params (list of env params): 'wind-direction','wind-speed','air-temperature','relative-humidity'
    """
    organise_data = {datetime.strftime(dt['start'],"%Y-%m-%d"):\
        {'time_range':{'start':dt['start'],'end':dt['end']}} for dt in date_time_dict}

    for dt, attr in organise_data.items():
        organise_data[dt]['params'] = {p:{'API_info':{'API_link':None, 'response':None},\
            'df':None} for p in env_params}
    return organise_data

def env_API_call(organise_data,API_main="https://api.data.gov.sg/v1/environment/"):
    for dt, attr in organise_data.items():
        for params in attr['params'].keys():
            link = API_main + params + '?date=' + dt #parse the main link with the date query to the link
            organise_data[dt]['params'][params]['API_info']['API_link'] = link #save the link
            req = requests.get(link) #fetch data from API link
            organise_data[dt]['params'][params]['API_info']['response'] = req.json() #convert to json format for easy processing

    return organise_data

def clean_env_API_response(location_id,organise_data):
    """
    target_location (str): location id
    """
    for dt, attr in organise_data.items():
        for params in attr['params'].keys():
            response = organise_data[dt]['params'][params]['API_info']['response']
            metadata = response['metadata']
            location_values = response['items'] #list of dict with keys: timestamp, readings
            #readings is a dict with keys station_id,value
            obs_key = params+'_'+ metadata['reading_unit']#create column name (env param) + units
            df_dict = {'date':[],'time':[],obs_key:[]} #initialise dictionary to store all the date,time and values
            for lv in location_values: #lv is a dict and represents a single observation
                ts = lv['timestamp']
                dt_obs = datetime.strptime(ts,'%Y-%m-%dT%H:%M:%S+08:00')
                if dt_obs > attr['time_range']['start'] and dt_obs < attr['time_range']['end']: #filter by date
                    df_dict['date'].append(dt) # append date
                    t = datetime.strftime(dt_obs,"%H:%M:%S") #time
                    df_dict['time'].append(t) #append time
                    if location_id in [r['station_id'] for r in lv['readings']]: #check whether target location is in the list of readings
                        v = [r['value'] for r in lv['readings'] if r['station_id'] == location_id][0] #only obtain the value if target location is in the list of readings
                        df_dict[obs_key].append(v)
                    else:
                        df_dict[obs_key].append(np.nan) #if target location is not in the list of readings then assign NA
            df = pd.DataFrame.from_dict(df_dict).set_index(['date','time'])
            organise_data[dt]['params'][params]['df'] = df
    return organise_data

def create_env_df(organise_data,location_id,fp_store):
    fp_store_env = join(fp_store,"retrieved_env_variables")
    if not exists(fp_store_env):
        mkdir(fp_store_env)
    df_list = []
    for dt, attr in organise_data.items():
        param_dfs = []
        for params in attr['params'].keys():
            param_dfs.append(organise_data[dt]['params'][params]['df'])
        df_concat = pd.concat(param_dfs, axis=1).reset_index()
        df_list.append(df_concat)
        #save file to csv, create a folder called "retrieved_env_variables" in your directory
        df_concat.to_csv(join(fp_store_env,"env_variables_{}_{}.csv".format(dt,location_id)),index=False)
    return df_list



def plot_env_df(fp):
    """
    filename (str): selected file from the list of fp_list
    """
    # fp = join(fp_store_env,filename)
    try:
        df = pd.read_csv(fp)
    except:
        df = None 

    if df is None:
        return None #if return None, then sg.popup(f"{E}",title="Error")
    
    else:
        params = list(df.columns)
        params = params[2:]
        nrows = 2
        ncols = ceil(len(params)/nrows)
        date = df.iloc[0,0]
        fig, axes = plt.subplots(nrows,ncols)
        for ax, p in zip(axes.flatten(),params):
            ax.plot(df['time'],df[p])
            plt_title, units = p.split('_')
            ax.set_title(f'{plt_title}')
            ax.set_ylabel(f'{plt_title} ({units})')
            ax.set_xlabel("Time")
            ax.tick_params(axis='x', labelrotation = 90)
        plt.tight_layout()
        plt.show(block=False)
        return



#--------------------drawing sunglint bbox------------------------
class LineBuilder:
    # lock = "glint"  # only one can be animated at a time
    def __init__(self,xs_glint,ys_glint,xs_nonglint,ys_nonglint,\
            line_glint,line_nonglint,r_glint,r_nonglint,\
            img_line_glint,img_line_nonglint,\
            img_bbox_glint,img_bbox_nonglint,\
            canvas,lock,current_fp):

        self.line_glint = line_glint
        self.r_glint = r_glint
        self.line_nonglint = line_nonglint
        self.r_nonglint = r_nonglint
        self.xs_glint = xs_glint#list(line.get_xdata())
        self.ys_glint = ys_glint#list(line.get_ydata())
        self.xs_nonglint = xs_nonglint#list(line.get_xdata())
        self.ys_nonglint = ys_nonglint#list(line.get_ydata())
        # self.cid = fig.canvas.mpl_connect('button_press_event', self)
        self.canvas = canvas
        self.cid = canvas.mpl_connect('button_press_event', self)
        #to save into json file, keep track of the last drawn bbox drawn on the img_line
        self.img_line_glint = img_line_glint
        self.img_line_nonglint = img_line_nonglint
        self.img_bbox_glint = img_bbox_glint
        self.img_bbox_nonglint = img_bbox_nonglint
        self.lock = lock
        self.current_fp = current_fp


    def __call__(self, event):
        if event.inaxes!=self.line_glint.axes or event.inaxes!=self.line_nonglint.axes:
            return
        # if LineBuilder.lock == "glint":
        if self.lock == "glint":
            self.xs_glint.append(event.xdata)
            self.ys_glint.append(event.ydata)
            self.line_glint.set_data(self.xs_glint[-2:], self.ys_glint[-2:])
            self.line_glint.figure.canvas.draw_idle()
            print(self.xs_glint[-2:])
            print(self.ys_glint[-2:])
            self.draw_rect(event)
        else:
            self.xs_nonglint.append(event.xdata)
            self.ys_nonglint.append(event.ydata)
            self.line_nonglint.set_data(self.xs_nonglint[-2:], self.ys_nonglint[-2:])
            self.line_nonglint.figure.canvas.draw_idle()
            print(self.xs_nonglint[-2:])
            print(self.ys_nonglint[-2:])
            self.draw_rect(event)

    def draw_rect(self ,_event):
        print(self.lock)

        if self.lock == "glint":
            x1,x2 = self.xs_glint[-2:]
            y1,y2 = self.ys_glint[-2:]
            self.img_bbox_glint = ((int(x1),int(y1)),(int(x2),int(y2)))
            h = y2 - y1
            w = x2 - x1
            self.r_glint.set_xy((x1,y1))
            self.r_glint.set_height(h)
            self.r_glint.set_width(w)
            self.img_line_glint = self.current_fp #update img_line based don the latest rect patch drawn
            print(self.img_bbox_glint)
        else:
            x1,x2 = self.xs_nonglint[-2:]
            y1,y2 = self.ys_nonglint[-2:]
            self.img_bbox_nonglint = ((int(x1),int(y1)),(int(x2),int(y2)))
            h = y2 - y1
            w = x2 - x1
            self.r_nonglint.set_xy((x1,y1))
            self.r_nonglint.set_height(h)
            self.r_nonglint.set_width(w)
            self.img_line_nonglint = self.current_fp #update img_line based don the latest rect patch drawn
            print(self.img_bbox_nonglint)

def reset_sgc(linebuilder):
    linebuilder.xs_glint = []
    linebuilder.ys_glint = []
    linebuilder.xs_nonglint = []
    linebuilder.ys_nonglint = []
    x1 = y1 = h = w = 0
    linebuilder.line_glint.set_data(linebuilder.xs_glint, linebuilder.ys_glint)
    linebuilder.r_glint.set_xy((x1,y1))
    linebuilder.r_glint.set_height(h)
    linebuilder.r_glint.set_width(w)

    linebuilder.line_nonglint.set_data(linebuilder.xs_nonglint, linebuilder.ys_nonglint)
    linebuilder.r_nonglint.set_xy((x1,y1))
    linebuilder.r_nonglint.set_height(h)
    linebuilder.r_nonglint.set_width(w)
    linebuilder.canvas.draw()
    return

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    
    # fp = r"C:\Users\PAKHUIYING\Documents\image_processing\F3_processed_surveys\2021_11_10\11_34_17\2021_10_11_11-34-17_rgb_image_line_08_15713_17196.tif"
    
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    #add toolbar
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)
    
    return figure_canvas_agg#linebuilder

class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)

#--------------------sunglint correction------------------------
class SunglintCorrection:
    def __init__(self,stitch_hyperspectral_object,bbox_fp,reflectance_glint,mode="least_sq",NIR_ref="mean",NIR_threshold=0.8,NIR_band=37):
        """
        stitch_hyperspectral_object (StitchHyperspectral): that already contains the attribute information of line_number,index etc
        reflectance_glint (dict): radiometrically corrected reflectance images from get_stitched_reflectance that contains glint area
        bbox_fp (str): filepath of a json file. Nested dict with keys (glint (line,bbox),non_glint (line,bbox))
            line (str): filepath of the image_line that contains the bbox of the glinted area
        mode (str): sunglint corection mode (pearson_cor, least_sq,covariance)
        NIR_ref (str): mean, min
        NIR_threshold (float): 0-1
        NIR_band (int): which NIR band index to use for sunglint correction
        """
        self.mode = mode
        self.NIR_ref = NIR_ref
        self.NIR_threshold = NIR_threshold
        self.NIR_band = NIR_band
        self.stitch_hyperspectral_object = stitch_hyperspectral_object
        self.reflectance_glint = reflectance_glint#radio_cor_reflectance
        self.bbox_fp = bbox_fp
        self.img_fp = join(self.stitch_hyperspectral_object.fp_store,self.stitch_hyperspectral_object.rgb_fp)
        self.bands = bands_wavelengths()
        try:
            self.img = np.asarray(Image.open(self.img_fp))
        except Exception as E:
            self.img = None
        try:
            with open(bbox_fp,"r") as cf:
                bbox = json.load(cf)
            ((x1,y1),(x2,y2)) = bbox['glint']['bbox'] #a deep-water area containing sun-glint patterns
            if x1 > x2: #columns
                x1,x2 = x2,x1
            if y1 > y2: #rows
                y1,y2 = y2,y1
            self.bbox = ((x1,y1),(x2,y2))
            self.glint_img = np.asarray(Image.open(bbox['glint']['fp'])) #image line where bbox of the glint area was drawn on
        except Exception as E:
            self.bbox = None
            self.glint_img = None
    
    def covariance_NIR(self,NIR,b):
        """
        NIR & b are vectors
        reflectance for band i
        """
        n = len(NIR)
        pij = np.dot(NIR,b)/n - np.sum(NIR)/n*np.sum(b)/n
        pjj = np.dot(NIR,NIR)/n - (np.sum(NIR)/n)**2
        return pij/pjj

    def least_sq_NIR(self,NIR,b):
        """
        NIR & b are vectors
        reflectance for band i
        """
        A = np.vstack([NIR,np.ones(len(NIR))]).T
        m, _ = np.linalg.lstsq(A,b, rcond=None)[0]
        return m
    
    def correlation_bands_reflectance(self):
        """
        calculate correlation between NIR and other bands for reflectance
        NIR_band is 750 nm
        """
        if self.bbox is None:
            return None
        ((x1,y1),(x2,y2)) = self.bbox
        reflectance_bands = [v[y1:y2,x1:x2].flatten() for v in self.reflectance_glint.values()] #flattened images
        NIR_reflectance = reflectance_bands[self.NIR_band] #flattened images
        if self.mode == "pearson_corr":
            return [pearsonr(NIR_reflectance,v)[0] for v in reflectance_bands]
        elif self.mode == "least_sq":
            return [self.least_sq_NIR(NIR_reflectance,v) for v in reflectance_bands]
        else:
            return [self.covariance_NIR(NIR_reflectance,v) for v in reflectance_bands]


    def correlation_bands_rgb(self):
        if self.bbox is None or self.glint_img is None:
            return
        ((x1,y1),(x2,y2)) = self.bbox

        rgb_layers = [self.glint_img[y1:y2,x1:x2,i] for i in range(3)]
        rgb_layers_flatten = [i.flatten() for i in rgb_layers]
        NIR_reflectance = rgb_layers_flatten[0]
        if self.mode == "pearson_corr":
            return [pearsonr(NIR_reflectance,v)[0] for v in rgb_layers_flatten]
        elif self.mode == "least_sq":
            return [self.least_sq_NIR(NIR_reflectance,v) for v in rgb_layers_flatten]
        else:
            return [self.covariance_NIR(NIR_reflectance,v) for v in rgb_layers_flatten]

    def get_glint_mask(self):
        """
        NIR_threshold is for red:blue band ratio
        """
        if self.img is None:
            return
        r_b_im = self.img[:,:,0]/self.img[:,:,-1]
        mask = np.where(r_b_im>self.NIR_threshold,1,0)
        return mask

    def sunglint_correction_reflectance(self):
        """
        correction is done in reflectance
        mode (str): pearson_corr,least_sq, covariance
        NIR_ref (str): mean, min

        Inputs required for Hedley's algorithm
        Note that sunglint_json_fp is only automatically generated when using GUI_platform.py

        sunglint_json_fp = "inputs/sunglint_correction_test_alignment.txt" 

        with open(sunglint_json_fp,"r") as cf:
                bbox = json.load(cf)

        print(f'bbox: {bbox}')

        try:
            line_glint = bbox['glint']['line']
            start_i,end_i = indexes_list[line_glint]
            test_stitch_class = StitchHyperspectral(fp_store,prefix,image_fp,spectrometer_fp,
                int(height),line_glint,start_i,end_i,
                gps_indices, unique_gps_df,reverse=reverse_boolean_list[line_glint])

            reflectance_glint = test_stitch_class.get_stitched_reflectance() #radiometrically corrected reflectance only for the image line where glint bbox is drawn on
            print("get reflectance glint...")
        except Exception as E:
            sunglint_json_fp = None
            reflectance_glint = None
            bbox = None

        if sunglint_checkbox is True and sunglint_json_fp is not None and reflectance_glint is not None:
            sgc = SunglintCorrection(test_stitch_class,sunglint_json_fp,reflectance_glint)
            sgc.sunglint_correction_rgb()
            sgc_reflectance = sgc.sunglint_correction_reflectance()
            reflectance = None
            print("Performing sunglint correction on hyperspectral reflectances...")
        """
        if self.bbox is None:
            return None

        corr = self.correlation_bands_reflectance()
        mask = self.get_glint_mask()
        radio_cor_reflectance = self.stitch_hyperspectral_object.get_stitched_reflectance()
        NIR_reflectance = radio_cor_reflectance[self.NIR_band]

        if self.NIR_ref == 'mean':
            ref = np.mean(NIR_reflectance)
        elif self.NIR_ref == 'min':
            ref = np.min(NIR_reflectance)

        corrected_reflectance = {i:None for i in range(len(radio_cor_reflectance.keys()))}
        for i,(r,c) in enumerate(zip(radio_cor_reflectance.values(),corr)): #iterate across bands
            r_corrected_reflectance =  r - c*(NIR_reflectance - ref)
            r_glint = r_corrected_reflectance.copy()
            r_glint[mask == 0] = 0 #keep corrected glinted areas, remove unglinted areas
            r_nonglint = r.copy()
            r_nonglint[mask == 1] = 0 #remove glint areas, keep non-glint areas
            #strategy-2: combine corrected glint areas + non corrected non-glint areas
            combined_glint_unglint = r_nonglint + r_glint
            corrected_reflectance[i] = combined_glint_unglint

        return corrected_reflectance
    
    def sunglint_correction_rgb(self):
        """
        correction is done in DN
        mode (str): pearson_corr,least_sq, covariance
        NIR_ref (str): mean, min
        TODO: should not be used anymore
        """

        if self.bbox is None:
            return None

        corr = self.correlation_bands_rgb()
        mask = self.get_glint_mask()
        rgb_layers = [self.img[:,:,i] for i in range(3)]
        if self.NIR_ref == 'mean':
            ref = np.mean(self.img[:,:,0])
        elif self.NIR_ref == 'min':
            ref = np.min(self.img[:,:,0])

        corrected_rgb = []
        for r,c in zip(rgb_layers,corr): #iterate across bands
            r_corrected =  r - c*(rgb_layers[0] - ref) #for red band, r= rgb_layers[0], c=1, thus r_corrected = ref only
            corrected_rgb.append(r_corrected)
        corrected_rgb = np.dstack(corrected_rgb).astype(np.uint8)
        
        corrected_rgb_glint = corrected_rgb.copy()
        corrected_rgb_glint[mask == 0] = 0 #remove nonglint areas and keep the corrected glint areas
        corrected_rgb_nonglint = self.img.copy()
        corrected_rgb_nonglint[mask == 1] = 0 #remove glint areas and keep the nonglint areas
        combined_img = corrected_rgb_glint + corrected_rgb_nonglint

        #save
        # glint_directory = "Glint_corrected"
        fp_store_glint = join(self.stitch_hyperspectral_object.fp_store,self.glint_directory)
        if not exists(fp_store_glint): #if glint folder d.n.e, create one
            mkdir(fp_store_glint)
        img = Image.fromarray(combined_img, 'RGB')
        fp = join(fp_store_glint,self.stitch_hyperspectral_object.rgb_fp)
        img.save(fp,save_all=True) #creates a tif file with 3 bands
        return combined_img

#-----------------------------------stitching algo----------------------------------------
def import_gps(image_file_path):

    gps_file_path = image_file_path + '/UAV'
    gps_files = listdir(gps_file_path) 
    gps_file = list(filter(lambda f: f.endswith('.csv'), gps_files))
    # print(gps_file)
    for f in gps_file:
        if '._' not in f:
            gps_file_path = gps_file_path + '/' + f
    
    print('gps file path:{}'.format(gps_file_path))

    df = pd.read_csv(gps_file_path,skiprows=5,index_col=0)
 
    return df

def get_unique_df(gps_df):
    n = len(gps_df.index)
    unique_coords = []
    unique_coords.append(gps_df.iloc[0,:]) #initialise with the first 0
    #remove duplicated timestamp and gps
    prev_timestamp = gps_df.iloc[0,0]
    for i in range(1,n):
        if gps_df.iloc[i,0] == prev_timestamp:
            continue
        else:
            prev_timestamp = gps_df.iloc[i,0]
            unique_coords.append(gps_df.iloc[i,:])

    unique_df = pd.concat(unique_coords,axis = 1).transpose()
    return unique_df

def rev_boolean_list(unique_gps_df,test_gps_index):
    #image lines may not be consecutive tho because they can be selected manually via the GUI
    test_gps_index.sort()
    reverse_boolean_list = []
    for i in range(0,len(test_gps_index),2):
        lat_start = unique_gps_df.iloc[test_gps_index[i],1]
        lat_end = unique_gps_df.iloc[test_gps_index[i+1],1]
        if lat_start > lat_end:
            flip_images_config = True
        else:
            flip_images_config = False
        reverse_boolean_list.append(flip_images_config)
    return reverse_boolean_list

def gps_to_image_indices(unique_gps_df,image_file_path,test_gps_index,millisecond_delay):
    """
    this function converts gps indices to image indices using binary tree search
    test_gps_index (list of int): indices of unique_gps_df that indicate the start and stop point
    millisecond_delay (int): ranges from 0 to 999
    *do not use rawfiles_attr's image index as that is not the same as the list's index. 
    *using rawfile_attr's image index therein as the list's index will retrieve the wrong matched time 
    >>> gps_to_image_indices(unique_gps_df,image_file_path,test_test_gps_index)
    """
    def format_img_dt(x):
        ms = x.split('-')[3].zfill(3)
        formatted_dt = '-'.join(x.split('-')[:3]) + '-' + ms
        return datetime.strptime(formatted_dt,'%H-%M-%S-%f')
    fp = image_file_path + '/' + 'RawImages'
    
    rawfiles = [f for f in sorted(listdir(fp)) if isfile(join(fp, f))] #list of rawfile names
    rawfiles_attr = [f.split('_') for f in rawfiles] #index (str), _, datetime, _
    s,ms = divmod(millisecond_delay,1000)
    time_delta = timedelta(seconds=s,milliseconds=ms)
    test_gps_index.sort()
    df_indexes = test_gps_index
    unique_gps_df['datetime'] = pd.to_datetime(unique_gps_df.iloc[:,0],format='%H:%M:%S:%f')
    unique_gps_df['datetime'] = unique_gps_df['datetime'] - time_delta

    matched_timestamp_image = []
    image_index = []
    n_images = len(rawfiles_attr)-1

    for t in unique_gps_df['datetime']:
        l = 0 #a tuple of index,_,dt,_
        h = n_images #a tuple of index,_,dt,_
        mid = n_images//2 #a tuple of index (0),_,dt (2),_
        while l < h:
            guess = rawfiles_attr[mid][2]
            guess = format_img_dt(guess)
            if guess > t: #if guess is too high
                h = mid -1 
                mid = (l + h)//2
            elif guess < t: #if guess is too low
                l = mid+1
                mid = (l + h)//2
            else: #if guess = t
                break
            # print(l,h,mid)
        closest_match = rawfiles_attr[mid]
        matched_timestamp_image.append(closest_match[2])
        image_index.append(mid)

    unique_gps_df['matched_timestamp_image'] = matched_timestamp_image
    unique_gps_df['image_index'] = image_index
    # print(unique_gps_df)
    filtered_unique_gps_df = unique_gps_df.iloc[df_indexes,:]
    img_indices = filtered_unique_gps_df['image_index'].values
    return [(img_indices[i],img_indices[i+1]) for i in range(0,len(img_indices),2)]



def plot_flight_camera_attributes(unique_gps_df,height,test_gps_index):
    """ 
    unique_gps_df is the net unique_gps_df after running gps_to_image_indices
    """
    pixel_size_at_sensor = 5.3 #um
    total_pixel_of_sensor_x = int(1280/61) #because hyperspectral images are line images, so actual pixel_x is 20
    focal_length = 16 #mm
    actual_size_of_sensor_x = pixel_size_at_sensor*total_pixel_of_sensor_x/1000 #mm
    fov_x = 2*math.atan(actual_size_of_sensor_x/2/focal_length)*180/math.pi #deg
    total_gnd_coverage_x = 2*height*math.tan(math.pi*fov_x/2/180) #metres #angle is converted to radians first
    OR_x_list = []
    frame_rate_list = []
    speed_list = []
    altitude_list = []
    # unique_gps_df = unique_gps_df.iloc[:,[1,2,6,11,3]].dropna() #latitude, longitude, datetime,image_index (float), altitude
    unique_gps_df = unique_gps_df.loc[:,["latitude","longitude","datetime","image_index","altitude"]].dropna()
    for i in range(len(unique_gps_df.index)-1):
        lat_prev, lon_prev = unique_gps_df.iloc[i,[0,1]].values
        lat_next, lon_next = unique_gps_df.iloc[i+1,[0,1]].values
        image_index_prev = int(unique_gps_df.iloc[i,3]) #cast float to int
        image_index_next = int(unique_gps_df.iloc[i+1,3]) #cast float to int
        # print(lat_prev,lon_prev)
        time_diff = unique_gps_df.iloc[i+1,2] - unique_gps_df.iloc[i,2]
        time_diff = time_diff.total_seconds()

        frame_rate = int((image_index_next - image_index_prev)/time_diff)
        # print("image_prev: {},image_next: {}, time_diff:{},frame_rate: {}".format(image_index_prev,image_index_next,time_diff,frame_rate))
        dist = Haversine((lon_prev,lat_prev),(lon_next,lat_next)).meters
        speed = dist/time_diff
        if frame_rate == 0:
            overlap_x_ratio = 1
        else:
            d = (1/frame_rate)*speed #distance covered by drone in 1 fps
            overlap_x = total_gnd_coverage_x - d
            overlap_x_ratio = overlap_x/total_gnd_coverage_x #in ratio instead of percentage
        OR_x_list.append(overlap_x_ratio)
        frame_rate_list.append(frame_rate)
        speed_list.append(speed)
        altitude_list.append(unique_gps_df.iloc[i+1,4])
        # from the time diff, and knowing the frame_rate, we know how many images are taken
    
    test_gps_index.sort()
    filtered_unique_gps = unique_gps_df.iloc[test_gps_index,:]

    #--------plot----------
    timestamp_x = unique_gps_df['datetime'][:-1]
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(timestamp_x,OR_x_list,'g--',alpha=0.25,label='Overlap ratio')
    ax.plot(timestamp_x,frame_rate_list,'b--',alpha=0.25,label='Frame rate')
    ax.plot(timestamp_x,speed_list,'k--',alpha=0.25,label='Drone speed')
    ax.plot(timestamp_x,altitude_list,'r--',alpha=0.25,label='Drone altitude')
    for j in range(0,len(filtered_unique_gps.index),2):
        lat_start, lon_start = filtered_unique_gps.iloc[j,[0,1]].values
        lat_end, lon_end = filtered_unique_gps.iloc[j+1,[0,1]].values
        image_index_start = int(filtered_unique_gps.iloc[j,3]) #cast float to int
        image_index_end = int(filtered_unique_gps.iloc[j+1,3]) #cast float to int
        time_diff = filtered_unique_gps.iloc[j+1,2] - filtered_unique_gps.iloc[j,2]
        time_diff = time_diff.total_seconds()
        frame_rate = int((image_index_end - image_index_start)/time_diff)
        dist = Haversine((lon_start,lat_start),(lon_end,lat_end)).meters
        speed = dist/time_diff
        # print("Avg speed of drone: {:.2f}, Avg frame rate: {}".format(speed,frame_rate))
        d = (1/frame_rate)*speed #distance covered by drone in 1 fps
        overlap_x = total_gnd_coverage_x - d
        overlap_x_ratio = overlap_x/total_gnd_coverage_x #in ratio instead of percentage
        filtered_timestamp_x = [filtered_unique_gps.iloc[j,2],filtered_unique_gps.iloc[j+1,2]]
        filtered_OR_x_list = [overlap_x_ratio,overlap_x_ratio]
        filtered_frame_rate_list = [frame_rate,frame_rate]
        filtered_speed_list = [speed,speed]
        filtered_altitude_list = [filtered_unique_gps.iloc[j,4],filtered_unique_gps.iloc[j+1,4]]
        avg_altitude = sum(filtered_altitude_list)/2
        ax.plot(filtered_timestamp_x,filtered_OR_x_list,'go-')
        ax.plot(filtered_timestamp_x,filtered_frame_rate_list,'bo-')
        ax.plot(filtered_timestamp_x,filtered_speed_list,'ko-')
        ax.plot(filtered_timestamp_x,filtered_altitude_list,'ro-')
        ax.text(filtered_unique_gps.iloc[j,2], overlap_x_ratio+0.5, "{:.2f}".format(overlap_x_ratio),bbox={'facecolor': 'white', 'alpha': 0.5},fontsize="medium")
        ax.text(filtered_unique_gps.iloc[j,2], frame_rate+20, "{:.2f}".format(frame_rate),bbox={'facecolor': 'white', 'alpha': 0.5},fontsize="medium")
        ax.text(filtered_unique_gps.iloc[j,2], speed+3, "{:.2f}".format(speed),bbox={'facecolor': 'white', 'alpha': 0.5},fontsize="medium")
        ax.text(filtered_unique_gps.iloc[j,2], avg_altitude+20, "{:.2f}".format(avg_altitude),bbox={'facecolor': 'white', 'alpha': 0.5},fontsize="medium")

    ax.set_yscale('log')
    ax.set_title('Flight attributes')
    ax.legend()
    plt.xticks(rotation=90)
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))   #to get a tick every 15 minutes
    # plt.xticks(np.arange(0,max(locs),step=30), label= [timestamp_x[s] for s in range(0,len(timestamp_x),30)],rotation=90)
    plt.show(block=False)

    return



def compute_ground_resolution(camera_height):
    #framerate in fps
    #camera_height in metres
    pixel_size_at_sensor = 5.3 #um
    total_pixel_of_sensor_x = int(1280/61) #because hyperspectral images are line images, so actual pixel_x is 20
    total_pixel_of_sensor_y = 1024
    focal_length = 16 #mm
    actual_size_of_sensor_x = pixel_size_at_sensor*total_pixel_of_sensor_x/1000 #mm
    actual_size_of_sensor_y = pixel_size_at_sensor*total_pixel_of_sensor_y/1000 #mm
    fov_x = 2*math.atan(actual_size_of_sensor_x/2/focal_length)*180/math.pi #deg
    fov_y = 2*math.atan(actual_size_of_sensor_y/2/focal_length)*180/math.pi #deg
    total_gnd_coverage_x = 2*camera_height*math.tan(math.pi*fov_x/2/180) #metres #angle is converted to radians first
    total_gnd_coverage_y = 2*camera_height*math.tan(math.pi*fov_y/2/180) #metres
    gnd_resolution = total_gnd_coverage_x/total_pixel_of_sensor_x #metres
    
    return gnd_resolution, total_gnd_coverage_x, total_gnd_coverage_y

class Haversine:
    '''
    use the haversine class to calculate the distance between
    two lon/lat coordnate pairs.
    output distance available in kilometers, meters, miles, and feet.
    example usage: Haversine([lon1,lat1],[lon2,lat2]).feet
    
    '''
    def __init__(self,coord1,coord2):
        lon1,lat1=coord1
        lon2,lat2=coord2
        
        R=6371000                               # radius of Earth in meters
        phi_1=math.radians(lat1)
        phi_2=math.radians(lat2)

        delta_phi=math.radians(lat2-lat1)
        delta_lambda=math.radians(lon2-lon1)

        a=math.sin(delta_phi/2.0)**2+\
           math.cos(phi_1)*math.cos(phi_2)*\
           math.sin(delta_lambda/2.0)**2
        c=2*math.atan2(math.sqrt(a),math.sqrt(1-a))
        
        self.meters=R*c                         # output distance in meters
        self.km=self.meters/1000.0              # output distance in kilometers
        self.miles=self.meters*0.000621371      # output distance in miles
        self.feet=self.miles*5280               # output distance in feet



def calculate_overlap_images(unique_gps_df,test_gps_index,height, frame_rate):


    gps_start_end_points = np.transpose(unique_gps_df.iloc[test_gps_index,1:3].values)

    coord_time_df = unique_gps_df.iloc[test_gps_index,[1,2,3,6]] #1 = latitude column, 2=longitude column, 3=altitude, 6 = datetime column

    speed_list = []
    for i in range(0,len(test_gps_index),2):
        # print(coord_time_df.iloc[i+1,2])
        time_diff = coord_time_df.iloc[i+1,3] - coord_time_df.iloc[i,3]
        time_diff = time_diff.total_seconds()
        dist = Haversine((coord_time_df.iloc[i+1,1],coord_time_df.iloc[i+1,0]),(coord_time_df.iloc[i,1],coord_time_df.iloc[i,0])).meters
        speed = dist/time_diff
        speed_list.append(speed)


    avg_speed = sum(speed_list[:-1])/len(speed_list[:-1]) #remove the last speed as speed is reduced when drone returns to home point

    #calculate avg height or specify height as an argument <- most accurate
    avg_height = coord_time_df.iloc[:,2].mean()
    height_above_MSL = 15
    avg_height = avg_height - height_above_MSL
    
    gnd_resolution, total_gnd_coverage_x, total_gnd_coverage_y = compute_ground_resolution(height)

    d = (1/frame_rate)*avg_speed
    print('avg speed:{}, d_x: {}, gnd_coverage_x:{}'.format(round(avg_speed,1),round(d,1),round(total_gnd_coverage_x,1)))
    overlap_x = total_gnd_coverage_x - d
    overlap_x_perc = overlap_x/total_gnd_coverage_x*100

    d_list = []
    for i in range(0,len(test_gps_index)-3,2):
        # print(shortened_test_gps_index[i])
        start_lon_prev, start_lat_prev = coord_time_df.iloc[i,1], coord_time_df.iloc[i,0] #start_prev
        end_lon_prev, end_lat_prev = coord_time_df.iloc[i+1,1], coord_time_df.iloc[i+1,0] #end_prev

        start_lon_next, start_lat_next = coord_time_df.iloc[i+2,1], coord_time_df.iloc[i+2,0] #start_next
        end_lon_next, end_lat_next = coord_time_df.iloc[i+3,1], coord_time_df.iloc[i+3,0] #end_next

        d1 = Haversine((start_lon_prev, start_lat_prev),(end_lon_next, end_lat_next)).meters
        d2 = Haversine((end_lon_prev, end_lat_prev),(start_lon_next, start_lat_next)).meters
        d_list.append(d1)
        d_list.append(d2)
    
    d = np.median(d_list)
    print('d_y: {}, gnd_coverage_y:{}'.format(round(d,1),round(total_gnd_coverage_y,1)))
    overlap_y = total_gnd_coverage_y - d
    if overlap_y<0:
        print("Line images doesn't overlap! Fly drone with higher overlap")
    overlap_y_perc = overlap_y/total_gnd_coverage_y*100
    print('overlap_x_perc: {:.2f}, overlap_y_perc: {:.2f}'.format(overlap_x_perc,overlap_y_perc))

    return overlap_x_perc,overlap_y_perc

def add_padding(og_gray_image,upscaled_image):
    """ 
    input images must be grayscale i.e. one channel only
    og_gray_image may be smaller or bigger than the upscaled_image
    """
    og_row, og_col = og_gray_image.shape
    upscaled_row, upscaled_col = upscaled_image.shape
    # print("OG shape: {}, upscaled shape: {}".format(og_gray_image.shape,upscaled_image.shape))
    padded_image = np.zeros(og_gray_image.shape)
    # compute center offset (// = floor division)
    x_center = (og_col - upscaled_col) // 2 #if positive, og is biggger than the upscaled
    y_center = (og_row - upscaled_row) // 2 #if positive, og is biggger than the upscaled
    # print("x_center: {}, y_center: {}".format(x_center,y_center))
    # copy img image into center of result image
    if (x_center >= 0 and y_center >= 0): #if positive, og is biggger than the upscaled
        padded_image[y_center:y_center+upscaled_row, x_center:x_center+upscaled_col] = upscaled_image
    elif (x_center < 0 and y_center >= 0): #upscale row is smaller, upscale col is greater
        # print(y_center,y_center+upscaled_row)
        padded_image[y_center:y_center+upscaled_row,:] = upscaled_image[:,:og_col]
        
    elif (x_center >= 0 and y_center <= 0): #upscale row is greater, upscale col is smaller 
        padded_image[:,x_center:x_center+upscaled_col] = upscaled_image[:og_row,:]
    else: #x_center < 0 and y_center < 0 #if negative, upscaled is biggger than the og
        padded_image = upscaled_image[:og_row,:og_col]
    # print("Does padded image have the same shape as OG? {}".format(padded_image.shape == og_gray_image.shape))
    return padded_image


class GeotransformImage:
    def __init__(self,stitch_class):
        """ 
        :param img_fp (str): filepath of the image
        :param stitch_class (custom StitchHyperspectral Class)
        geotransforms a stitched line image
        """
        self.stitch_class = stitch_class
        # attributes
        self.prefix = self.stitch_class.prefix
        self.line_number = self.stitch_class.line_number
        self.start_index = self.stitch_class.start_index
        self.end_index = self.stitch_class.end_index
        self.test_gps_index = self.stitch_class.test_gps_index
        self.unique_gps_df = self.stitch_class.unique_gps_df
        self.reverse = self.stitch_class.reverse
        # file names
        self.rgb_fp = self.stitch_class.rgb_fp
        self.rgb_reflectance_fp = self.stitch_class.rgb_reflectance_fp
        self.rgb_deglinted_fp = self.stitch_class.rgb_deglinted_fp
        self.predicted_fp = self.stitch_class.predicted_fp
        self.mask_fp = self.stitch_class.mask_fp
        # directories
        self.fp_store = self.stitch_class.fp_store
        self.rgb_reflectance_directory = self.stitch_class.rgb_reflectance_directory
        self.glint_directory = self.stitch_class.glint_directory
        self.prediction_directory = self.stitch_class.prediction_directory
        self.mask_directory = self.stitch_class.mask_directory

    def load_image(self, fp):
        """ 
        :param fp (str): filepath of the image
        """
        try:
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED) #to read a >8bit image
        except OSError as err:
            print("Could not open image file, image d.n.e? OS error: {0}".format(err))
        return np.flipud(np.array(img)) #flip vertically
    
    def get_angle(self, lat, lon):
        """ returns angle in deg """
        lat_start, lat_end = lat
        lon_start, lon_end = lon

        direction_vector = np.array([lon_end,lat_end]) - np.array([lon_start,lat_start])
        direction_vector = direction_vector/np.linalg.norm(direction_vector) #convert to unit vector
        east_vector = np.array([1,0]) #measured from the horizontal as a reference
        angle = np.arccos(np.dot(direction_vector,east_vector))/(2*np.pi)*360 #direction vector already converted to a unit vector
        
        #---cross pdt---
        #if vector is always on the left side of east_vector, cross pdt will be pointing in (i.e. -ve), otherwise it will be pointing outwards (i.e. +ve)
        if np.cross(direction_vector,east_vector) > 0: #point outwards aka to the right of the east vector
            angle = 180 - angle 
        
        return angle
    
    def affine_transformation(self,img,angle_to_rotate):
        """ 
        :param img (np.ndarray)
        :param angle_to_rotate (float)
        """
        rows, cols = img.shape[0],img.shape[1] #rgb image
        center = (cols//2,rows//2)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center,angle_to_rotate,1) #center, angle, scale
        # rotate the image using cv2.warpAffine
        # rotated_image = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(width, height))
        cosofRotationMatrix = np.abs(rotation_matrix[0][0]) #scale*cos(angle)
        sinofRotationMatrix = np.abs(rotation_matrix[0][1]) #scale*sin(angle)
        # Now will compute new height & width of
        # an image so that we can use it in
        # warpAffine function to prevent cropping of image sides
        newImageHeight = int((cols * sinofRotationMatrix) +
                            (rows * cosofRotationMatrix))
        newImageWidth = int((cols * cosofRotationMatrix) +
                            (rows * sinofRotationMatrix))
        # After computing the new height & width of an image
        # we also need to update the values of rotation matrix
        rotation_matrix[0][2] += (newImageWidth/2) - center[0]
        rotation_matrix[1][2] += (newImageHeight/2) - center[1]

        # Now, we will perform actual image rotation WITHOUT MASK
        rotatedImage = cv2.warpAffine(img, rotation_matrix, (newImageWidth, newImageHeight))
        if (rotatedImage.shape[-1] == 3):
            # for rgb image
            rotatedImage = cv2.cvtColor(rotatedImage,cv2.COLOR_BGR2RGB)
        return rotatedImage
    
    def get_geotransform_params(self,angle,lat,lon,img):
        """ 
        :param img (np.ndarray): original image that has not been geotransformed
        returns a 6-tuple of geotransformation parameters e.g. (left_extent, lon_res_per_pixel, 0, top_extent, 0, lat_res_per_pixel)
        """
        
        #get original img shape to calculate transformed img shape
        nrow, ncol = img.shape[0], img.shape[1]

        if angle > 90 and angle < 180:
            lat_res_per_pixel = (np.max(lat) - np.min(lat))/(ncol*np.sin(angle/360*2*np.pi)) #170 deg it's a +ve value
            lon_res_per_pixel = (np.max(lon) - np.min(lon))/(ncol*np.cos(angle/360*2*np.pi)) #170 deg it's a -ve value
            # top_extent = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #y
            top_extent = np.max(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve. eq to max(lat) + nrow/2*cos\theta
            left_extent = np.min(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel + ncol*np.cos(angle/360*2*np.pi)*lon_res_per_pixel #correct
           
            print("Upper left corner coord: lat:{}, lon:{}".format(top_extent,left_extent))
            #---------EXTRAPOLATE GPS (end)---------------

            #---------GEOTRANSFORM (start)---------------
            #after flipping it will be the upper left corner
            #The 3rd and 5th parameter are used (together with the 2nd and 4th) to define the rotation if your image doesn't have 'north up'.
            # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
            # GT(1) w-e pixel resolution / pixel width.
            # GT(2) row rotation (typically zero).
            # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
            # GT(4) column rotation (typically zero).
            # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
            geotransform = (left_extent, lon_res_per_pixel, 0, top_extent, 0, -lat_res_per_pixel) #xmin=longitude, ymax=lat
            #sequences of 6 numbers in which the first and fourth are the x and y offsets and the second and sixth are the x and y pixel sizes.
        elif angle > 0 and angle < 90:
            lat_res_per_pixel = (np.max(lat) - np.min(lat))/(ncol*np.sin(angle/360*2*np.pi)) #deg it's a +ve value
            lon_res_per_pixel = (np.max(lon) - np.min(lon))/(ncol*np.cos(angle/360*2*np.pi)) #deg it's a +ve value
            # top_extent = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #y
            top_extent = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve. eq to max(lat) + nrow/2*cos\theta
            left_extent = np.min(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel + ncol*np.cos(angle/360*2*np.pi)*lon_res_per_pixel #correct
            #eq to min(lon) + nrow/2*sin\theta + ncol*cos\theta --> upper right corner
            print("Upper left corner coord: lat:{}, lon:{}".format(top_extent,left_extent))
            geotransform = (left_extent, -lon_res_per_pixel, 0, top_extent, 0, -lat_res_per_pixel) #xmin=longitude, ymax=lat

        return geotransform
    
    def gdal_save(self, rotatedImage, geotransform, save_fp):
        """
        :param rotatedImage (np.ndarray)
        :param geotransform (6 tuple): e.g. (left_extent, lon_res_per_pixel, 0, top_extent, 0, lat_res_per_pixel)
        :param save_fp (str): filepath to save as
        :param image_bit (int: 8 or 16): image bit in either np.uint8 or np.uint16
        """
        flipped_transformed_img = np.fliplr(rotatedImage) #flip images horizontally
        nrow_transformed, ncol_transformed = rotatedImage.shape[0], rotatedImage.shape[1]
        n_bands = rotatedImage.shape[2] if (len(rotatedImage.shape) == 3) else 1
        if (rotatedImage.dtype == 'uint8'):
            dst_ds = gdal.GetDriverByName('GTiff').Create(save_fp,\
                ncol_transformed, nrow_transformed, n_bands, gdal.GDT_Byte)
        elif (rotatedImage.dtype == 'float32' or rotatedImage.dtype == 'uint16'):
            dst_ds = gdal.GetDriverByName('GTiff').Create(save_fp,\
                ncol_transformed, nrow_transformed, n_bands, gdal.GDT_UInt16)
        else:
            raise ValueError("image dtype is not uint8 or float32")
        
        dst_ds.SetGeoTransform(geotransform)    # specify coords
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(4326)                # WGS84 lat/long
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file

        if (n_bands > 1):
            for i in range(n_bands):
                dst_ds.GetRasterBand(i+1).WriteArray(flipped_transformed_img[:,:,i])
        else:
            dst_ds.GetRasterBand(1).WriteArray(flipped_transformed_img)

        dst_ds.FlushCache()                     # write to disk
        dst_ds = None

        return
    
    def get_geotransform_fp(self, fn):
        """
        :param fn (str): original filename
        returns geotransformed filename in .tif
        """
        geotransform_fp = fn.replace(self.prefix, f'{self.prefix}_Geotransformed')
        fn = splitext(geotransform_fp)[0]
        return f'{fn}.tif'

    def geotransform_image(self):
        """
        perform geotransformation
        """
        sub_directories = [self.rgb_reflectance_directory, self.prediction_directory, self.mask_directory, self.glint_directory]
        sub_directories = [join(self.fp_store,d) for d in sub_directories]
        fp_names = [self.rgb_fp,self.rgb_reflectance_fp,self.predicted_fp,self.mask_fp, self.rgb_deglinted_fp]
        directories = [self.fp_store] + sub_directories
        assert len(fp_names) == len(directories), "number of fp names and directories must match"
        
        gps_indexes = [(self.test_gps_index[i],self.test_gps_index[i+1]) for i in range(0,len(self.test_gps_index)-1,2)]
        # print(gps_indexes)
        gps_start_index, gps_end_index = gps_indexes[self.line_number] #use line number
        #inclusive of gps start index and gps end index

        coords = np.transpose(self.unique_gps_df.iloc[[gps_start_index, gps_end_index],[1,2]].values) #1=latitude, 2=longitude
        lat = coords[0,:] #list of latitude, lat[0] = lat_start
        lon = coords[1,:] #list of lon
        # angle and geotransform just needs to be calculated once
        angle = self.get_angle(lat,lon)

        for fn, directory in zip(fp_names,directories):
            # check if file exists
            fp = join(directory,fn)
            if exists(fp):
                img = self.load_image(fp)
                break
        geotransform = self.get_geotransform_params(angle,lat,lon,img)

        # for each directory that has been automatically created, geotransform each line image
        for fn, directory in zip(fp_names,directories):
            # check if file exists
            fp = join(directory,fn)
            if exists(fp):
                # means images exist in the directory unless manually deleted
                img = self.load_image(fp)
                rotatedImage = self.affine_transformation(img,angle)
                save_fn = self.get_geotransform_fp(fn)
                save_fp = join(directory,save_fn)
                self.gdal_save(rotatedImage, geotransform, save_fp)
                
        return
                
    



#CREATE CLASS FOR GEOTRANSFORM
class GeotransformImageDeprecated:
    def __init__(self,stitch_class,mask=None,classify=False,transform_predicted_image=False,sunglint_correction=True,acw=True):
        self.stitch_class = stitch_class
        self.fp_store = self.stitch_class.fp_store
        self.prefix = self.stitch_class.prefix
        self.line_number = self.stitch_class.line_number
        self.start_index = self.stitch_class.start_index
        self.end_index = self.stitch_class.end_index
        self.test_gps_index = self.stitch_class.test_gps_index
        self.unique_gps_df = self.stitch_class.unique_gps_df
        self.reverse = self.stitch_class.reverse
        self.rgb_fp = self.stitch_class.rgb_fp
        self.rgb_reflectance_fp = self.stitch_class.rgb_reflectance_fp
        self.mask = mask
        self.classify = classify
        self.transform_predicted_image = transform_predicted_image
        self.sunglint_correction = sunglint_correction
        self.acw = acw
        # self.glint_directory = self.stitch_class.glint_directory
        # self.prediction_directory = self.stitch_class.prediction_directory
        # self.rgb_reflectance_directory = self.stitch_class.rgb_reflectance_directory

    

    def affine_transformation(self,angle_to_rotate):
        """
        this function only rotates the image and makes sure the bounding box are optimal
        inputs are filepath of rgb image, 
        angle_to_rotate (can be manually specified), but it's calculated from flight path angle. 
        Rotation is about the horizontal. +ve means rotate acw, -ve means rotate cw about horizontal
        mask (None or np.array): if none, no masking is conducted. if np.array of mask is provided, masking is conducted
        transform_predicted_image (boolean): whether to geotransform a predicted_image.
        classify (boolean): indicates whether different objects should be identified with a different colour
        """
        try:
            if self.sunglint_correction is False:
                img_og = cv2.imread(join(self.fp_store,self.rgb_fp))
            else:
                fp_store_glint = join(self.fp_store,self.glint_directory,self.rgb_fp)
                img_og = cv2.imread(fp_store_glint)
        except OSError as err:
            print("Could not open image file, image d.n.e? OS error: {0}".format(err))

        img = np.flipud(np.array(img_og)) #flip vertically

        if self.transform_predicted_image == True:
            fp_predicted_img = join(self.fp_store,self.prediction_directory,self.rgb_fp.replace('rgb','predicted'))
            try:
                predicted_img_og = cv2.imread(fp_predicted_img, cv2.IMREAD_UNCHANGED) #to read a >8bit image
            except OSError as err:
                print("Could not open image file, image d.n.e? OS error: {0}".format(err))
            predicted_img = np.flipud(np.array(predicted_img_og)) #flip vertically

        #predicted_img and img are separate entities

        #-------PERFORM MASKING-----------
        if self.mask is not None:
            mask = np.flipud(self.mask) #flip vertically
            if self.classify is False:
                img_mask = img.copy()
                img_mask[mask!=0] = 0 #objects other than water turns to black
            else:
                img_mask = img.copy()
                img_mask[mask == 2] = 128 #caisson labelled as gray
                img_mask[mask==1] = 0 #vessels labelled as black
            img_mask = img_mask.astype(np.uint8)
            if self.transform_predicted_image is True:
                predicted_img_mask = predicted_img.copy()
                predicted_img_mask[mask!=0] = 0
                predicted_img_mask = predicted_img_mask.astype(np.uint16) #cast from float to int for image output

        #-------PERFORM AFFINE TRANSFORMATION-----------

        rows, cols = img.shape[0],img.shape[1] #rgb image
        center = (cols//2,rows//2)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center,angle_to_rotate,1) #center, angle, scale
        # rotate the image using cv2.warpAffine
        # rotated_image = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(width, height))
        cosofRotationMatrix = np.abs(rotation_matrix[0][0]) #scale*cos(angle)
        sinofRotationMatrix = np.abs(rotation_matrix[0][1]) #scale*sin(angle)
        # Now will compute new height & width of
        # an image so that we can use it in
        # warpAffine function to prevent cropping of image sides
        newImageHeight = int((cols * sinofRotationMatrix) +
                            (rows * cosofRotationMatrix))
        newImageWidth = int((cols * cosofRotationMatrix) +
                            (rows * sinofRotationMatrix))
        # After computing the new height & width of an image
        # we also need to update the values of rotation matrix
        rotation_matrix[0][2] += (newImageWidth/2) - center[0]
        rotation_matrix[1][2] += (newImageHeight/2) - center[1]

        # Now, we will perform actual image rotation WITHOUT MASK
        rotatingimage = cv2.warpAffine(img, rotation_matrix, (newImageWidth, newImageHeight))

        output = {'rgb': cv2.cvtColor(rotatingimage,cv2.COLOR_BGR2RGB)}
        if self.mask is not None:
            rotatingimage_masked = cv2.warpAffine(img_mask, rotation_matrix, (newImageWidth, newImageHeight))
            output['rgb_masked'] = cv2.cvtColor(rotatingimage_masked,cv2.COLOR_BGR2RGB)
            if self.transform_predicted_image is True:
                rotating_predictedimage = cv2.warpAffine(predicted_img, rotation_matrix, (newImageWidth, newImageHeight))
                rotating_predictedimage_masked = cv2.warpAffine(predicted_img_mask, rotation_matrix, (newImageWidth, newImageHeight))
                output['predicted'] = rotating_predictedimage
                output['predicted_masked'] = rotating_predictedimage_masked

        else:
            if self.transform_predicted_image is True:
                rotating_predictedimage = cv2.warpAffine(predicted_img, rotation_matrix, (newImageWidth, newImageHeight))
                output['predicted'] = rotating_predictedimage

        return output

    
    def geotransform_image(self):
        """
        this function takes in line_number (index of list of gps indexes),gps_indexes,unique_gps_df,image_file_path
        test_gps_index are the indices from GUI_flight_lines.py
        images are flipped vertically by default
        fp_store (str): folderpath where images are being stored
        additional arguments are reverse and acw
        mask (None or np.array): if none, no masking is conducted. if np.array of mask is provided, masking is conducted
        acw = indicates whether flight lines are acw or cw based on flight plan angle in DJI Go app (default: acw = True)
            where reference of the horizontal is the east vector
        DJI's flight angle is the angle measured acw from the horizontal
        if flight angle is < 90 then set acw = True, else, set acw = False
        mask_objects (boolean): whether to mask the objects e.g. barge/caissons in the output image
        transform_predicted_image (boolean): whether the image to be transformed is an rgb image or the predicted_image.
            Default = False (i.e. geotransform the rgb image)
        >>>geotransform_image(line_number,gps_indexes,unique_gps_df,image_file_path,prefix,reverse=False,acw=True)
        """
        # gps_indexes is a list of just 1 image line with start and stop

        gps_indexes = [(self.test_gps_index[i],self.test_gps_index[i+1]) for i in range(0,len(self.test_gps_index)-1,2)]
        # print(gps_indexes)
        gps_start_index, gps_end_index = gps_indexes[self.line_number] #use line number to replace 0

        image_start_index, image_end_index = self.start_index,self.end_index
        fp = join(self.fp_store,self.rgb_fp)
        print('image obtained from gps indexes {}: {}'.format(gps_indexes[self.line_number],fp))
        # print("settings: mask_objects: {}, acw: {}".format(str(mask_objects),str(acw)))
        try:
            img = np.array(Image.open(fp)) #can it open a 16bit image? yes
            # print("img dtype:{}".format(img.dtype))
        except OSError as err:
            print("Could not open image file, image d.n.e? OS error: {0}".format(err))
        
        #inclusive of gps start index and gps end index
        coords = np.transpose(self.unique_gps_df.iloc[[gps_start_index, gps_end_index],[1,2]].values) #1=latitude, 2=longitude
        lat = coords[0,:] #list of latitude, lat[0] = lat_start
        lon = coords[1,:] #list of lon
        lat_start, lat_end = lat
        lon_start, lon_end = lon

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
        transformed_dict = self.affine_transformation(angle)

        nrow_transformed, ncol_transformed = transformed_dict['rgb'].shape[0], transformed_dict['rgb'].shape[1]
        # nrow_transformed, ncol_transformed = transformed_img.shape[0], transformed_img.shape[1]
        #-------CONDUCT TRANSFORMATION (end)-------------
        print("Performing geotransformation...")
        #---------EXTRAPOLATE GPS (start)---------------
        # nrow,ncol,channels = img.shape #nrow = h, ncol = w
        nrow, ncol = img.shape[0], img.shape[1] #get original img shape to calculate transformed img shape
        if angle > 90 and angle < 180:
            lat_res_per_pixel = (np.max(lat) - np.min(lat))/(ncol*np.sin(angle/360*2*np.pi)) #170 deg it's a +ve value
            lon_res_per_pixel = (np.max(lon) - np.min(lon))/(ncol*np.cos(angle/360*2*np.pi)) #170 deg it's a -ve value
            # top_extent = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #y
            top_extent = np.max(lat) - nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve. eq to max(lat) + nrow/2*cos\theta
            left_extent = np.min(lon) - nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel + ncol*np.cos(angle/360*2*np.pi)*lon_res_per_pixel #correct
           
            print("Upper left corner coord: lat:{}, lon:{}".format(top_extent,left_extent))
            #---------EXTRAPOLATE GPS (end)---------------

            #---------GEOTRANSFORM (start)---------------
            #after flipping it will be the upper left corner
            #The 3rd and 5th parameter are used (together with the 2nd and 4th) to define the rotation if your image doesn't have 'north up'.
            # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
            # GT(1) w-e pixel resolution / pixel width.
            # GT(2) row rotation (typically zero).
            # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
            # GT(4) column rotation (typically zero).
            # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
            geotransform = (left_extent, lon_res_per_pixel, 0, top_extent, 0, -lat_res_per_pixel) #xmin=longitude, ymax=lat
            #sequences of 6 numbers in which the first and fourth are the x and y offsets and the second and sixth are the x and y pixel sizes.
        elif angle > 0 and angle < 90:
            lat_res_per_pixel = (np.max(lat) - np.min(lat))/(ncol*np.sin(angle/360*2*np.pi)) #deg it's a +ve value
            lon_res_per_pixel = (np.max(lon) - np.min(lon))/(ncol*np.cos(angle/360*2*np.pi)) #deg it's a +ve value
            # top_extent = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #y
            top_extent = np.max(lat) + nrow/2*np.cos(angle/360*2*np.pi)*lat_res_per_pixel #cos is -ve. eq to max(lat) + nrow/2*cos\theta
            left_extent = np.min(lon) + nrow/2*np.sin(angle/360*2*np.pi)*lon_res_per_pixel + ncol*np.cos(angle/360*2*np.pi)*lon_res_per_pixel #correct
            #eq to min(lon) + nrow/2*sin\theta + ncol*cos\theta --> upper right corner
            print("Upper left corner coord: lat:{}, lon:{}".format(top_extent,left_extent))
            geotransform = (left_extent, -lon_res_per_pixel, 0, top_extent, 0, -lat_res_per_pixel) #xmin=longitude, ymax=lat

        #-----------------CREATE RASTER FILE-----------------
        # dst_ds = gdal.GetDriverByName('GTiff').Create('test_Geotransformed_rgb_image_{}_{}.tif'.format(image_start_index, image_end_index), ncol_transformed, nrow_transformed, 3, gdal.GDT_Byte)
        if self.transform_predicted_image is True:
            # prediction_directory = 'Prediction'
            fp_store_prediction = join(self.fp_store,self.prediction_directory)
            if not exists(fp_store_prediction): #if prediction folder d.n.e, create one
                mkdir(fp_store_prediction)

        for xform_type, arr in transformed_dict.items():
            flipped_transformed_img = np.fliplr(arr) #flip images horizontally
            # flipped_transformed_img = arr
            gtiff_fp = '{}_Geotransformed_{}_image_line_{}_{}_{}.tif'.format(self.prefix,xform_type,str(self.line_number).zfill(2),image_start_index, image_end_index)
            if len(arr.shape) == 3: #then it has 3-bands
                gtiff_fp = join(self.fp_store,gtiff_fp)
                dst_ds = gdal.GetDriverByName('GTiff').Create(gtiff_fp,\
                    ncol_transformed, nrow_transformed, 3, gdal.GDT_Byte)
                dst_ds.SetGeoTransform(geotransform)    # specify coords
                srs = osr.SpatialReference()            # establish encoding
                srs.ImportFromEPSG(4326)                # WGS84 lat/long
                dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
                dst_ds.GetRasterBand(1).WriteArray(flipped_transformed_img[:,:,0])   # write r-band to the raster
                dst_ds.GetRasterBand(2).WriteArray(flipped_transformed_img[:,:,1])   # write g-band to the raster
                dst_ds.GetRasterBand(3).WriteArray(flipped_transformed_img[:,:,2])   # write b-band to the raster
                dst_ds.FlushCache()                     # write to disk
                dst_ds = None
            else: #greyscale image with 1 bands
                if 'predicted' in xform_type and self.transform_predicted_image is True:
                    gtiff_fp = join(self.fp_store,fp_store_prediction,gtiff_fp)
                else:
                    gtiff_fp = join(self.fp_store,gtiff_fp)
                dst_ds = gdal.GetDriverByName('GTiff').Create(gtiff_fp,\
                    ncol_transformed, nrow_transformed, 1, gdal.GDT_UInt16) #save as 16bit
                dst_ds.SetGeoTransform(geotransform)    # specify coords
                srs = osr.SpatialReference()            # establish encoding
                srs.ImportFromEPSG(4326)                # WGS84 lat/long
                dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
                dst_ds.GetRasterBand(1).WriteArray(flipped_transformed_img)   # write 1-band to the raster
                dst_ds.FlushCache()                     # write to disk
                dst_ds = None

        #---------GEOTRANSFORM (end)---------------

        return



def raw_to_hyperspectral_img(fp):
    """
    fp (str): filepath of the raw image
    outputs arrays of rgb images only
    """
    band_width = int(1280/61)
    try:
        scene_infile = open(fp,'rb')
        scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1280*1024)
        # print(scene_image_array.shape)
    except Exception as E:
        print(f'image file cannot be opened: {E}')
        pass
    reshaped_raw = scene_image_array.reshape(1024,1280)
    image_array = []
    for i in range(61):
        row_start = i*band_width
        row_end = i*band_width + band_width
        band_array = reshaped_raw[:,row_start:row_end]
        image_array.append(band_array)
    hyperspectral_img = np.dstack(image_array)
    return hyperspectral_img

class StitchHyperspectral:
# (image_folder_filepath,spectro_filepath,overlap_ratio,start_index,end_index,s1_bands_fp='s1_bands.csv',Reflectance_White_Ref=95):
    def __init__(self,fp_store,prefix,image_folder_filepath,spectro_filepath,\
        height,line_number,start_index,end_index,\
        test_gps_index, unique_gps_df,destriping=True,\
        Reflectance_White_Ref=95,reverse=True):
        
        # directories to store images
        self.fp_store = fp_store
        self.glint_directory = "Glint_corrected"
        self.prediction_directory = 'Prediction'
        self.rgb_reflectance_directory = 'rgb_reflectance'
        self.hyperspectral_directory = 'Hyperspectral'
        self.mask_directory = 'Mask'
        # instance variable unique to each image
        self.prefix = prefix
        self.image_folder_filepath = image_folder_filepath
        self.spectro_filepath = spectro_filepath
        self.height = height
        self.line_number = line_number
        self.start_index = start_index
        self.end_index = end_index
        self.test_gps_index = test_gps_index
        self.unique_gps_df = unique_gps_df
        self.destriping = destriping
        self.Reflectance_White_Ref = Reflectance_White_Ref
        self.reverse = reverse
        #constants
        self.band_width = int(1280/61)
        self.band_numbers = int(61)
        self.rgb_bands = [38,23,15]
        # file paths
        self.rgb_fp = '{}_rgb_image_line_{}_{}_{}.tif'.format(self.prefix,str(self.line_number).zfill(2),self.start_index,self.end_index)
        self.rgb_reflectance_fp = self.rgb_fp.replace('rgb',self.rgb_reflectance_directory)
        # just let deglinted reflectance img have the same fn as rgb_reflectance
        self.rgb_deglinted_fp = self.rgb_reflectance_fp
        self.predicted_fp = self.rgb_fp.replace('rgb','predicted')
        self.mask_fp = self.rgb_fp.replace('rgb_image','mask')#f'{self.prefix}_mask_line_{str(self.line_number).zfill(2)}_{self.start_index}_{self.end_index}.png'


    def get_camera_settings(self,filepath):
        """ 
        GET IMAGE ATTRIBUTE of calibration files (white & dark)
        input is the image folder filepath
        outputs (exposure time of sensor 1 , datetime)
        >>>get_camera_settings(filepath)
        """
        image_result = re.search('EXP(.*)_RES', filepath).group(0)
        # print(filepath)
        date_time = datetime.strptime(filepath[:19], "%Y-%m-%d_%H-%M-%S")
        #split image_result by delimiter
        image_result = image_result.split('_')
        img_exp_s1 = image_result[0]
        img_exp_s1 = float(img_exp_s1.replace('EXP',''))/10
        return img_exp_s1, date_time

    def get_spectroradiometer_data(self,date_time):
        """ 
        GET SPECTRORADIOMETER DATA
        inputs are spectrometer filepath and date_time when the image was taken
        outputs spectroradiometer df
        >>>get_spectroradiometer_data(spectro_filepath,date_time)
        """
        delta = timedelta(seconds = 1)
        string_to_match = date_time.strftime("%m_%d_%Y_AbsoluteIrradiance_%H-%M-%S")
        next_time = date_time + delta
        
        filtered_files = [f for f in sorted(listdir(self.spectro_filepath)) if re.match('^.*{}.*'.format(string_to_match), f)]
        if len(filtered_files) == 0: #if image timestamp not inside spectroradiometer folder, try next sec timestamp
            string_to_match = next_time.strftime("%m_%d_%Y_AbsoluteIrradiance_%H-%M-%S")
            filtered_files = [f for f in sorted(listdir(self.spectro_filepath)) if re.match('^.*{}.*'.format(string_to_match), f)]
            if len(filtered_files) == 0: #if still not inside folder, likely that timestamp is wrong
                raise NameError('Timestamps dont match!')
        
        try:
            filepath = self.spectro_filepath + '/' + filtered_files[0]
            #dataframe of wavelength and the corresponding irradiance
            column_name = string_to_match.replace('_AbsoluteIrradiance','')
            df = pd.read_csv(filepath, delimiter = "\t",header=None,skiprows=13,names=['wavelength',column_name])
        except OSError as err:
            print("File cant be opened - OS error: {0}".format(err))

        return df

    def calibration_attributes(self):
        """ 
        GET calibration attributes e.g. DN_max, DN_avg
        inputs image_folder_filepath
        outputs a nested dictionary of calibration dictionary where keys are 'White' and 'Dark'
        >>>calibration_attributes(image_folder_filepath)
        """
        Dark_fp = join(self.image_folder_filepath, 'Dark')
        White_fp = join(self.image_folder_filepath, 'WhiteRef')
        calibration_list = [White_fp,Dark_fp] #0 is white, 1 is black
        calibration_dictionary = {} #create empty dictionary
        for i in range(len(calibration_list)):
            calib_img = [f for f in sorted(listdir(calibration_list[i])) if isfile(join(calibration_list[i], f))] #list files in folder
            calib_fp_list = [join(calibration_list[i],f) for f in calib_img] #create absolute filepath
            scene_infile_list = [open(f,'rb') for f in calib_fp_list]
            scene_image_array_list = [np.fromfile(scene_infile,dtype=np.uint8,count=1280*1024) for scene_infile in scene_infile_list]
            scene_image_array = np.mean(scene_image_array_list,axis=0)
            
            exposure,dt = self.get_camera_settings(calib_img[0]) #use the function above
            img_dict = {'raw_file':scene_image_array,'exp_time':exposure,'datetime':dt}

            calibration_dictionary[i] = img_dict
        #rename dictionary keys
        calibration_dictionary['White'] = calibration_dictionary.pop(0)
        calibration_dictionary['Dark'] = calibration_dictionary.pop(1)
        return calibration_dictionary #returns nested dictionary
    
    def raw_attributes(self):
        """ 
        GET raw images attributes
        outputs a nested dictionary where keys are name of image file, and values are dictionary of raw image array, exposure time and datetime
        >>>raw_attributes(image_folder_filepath,date,start_index,end_index)
        """
        image_result = re.search('EXP(.*)_RES', self.image_folder_filepath).group(0)
        image_result = image_result.split('_')
        img_exp_s1 = image_result[0]
        img_exp_s1 = float(img_exp_s1.replace('EXP',''))/10
        calib_info = self.calibration_attributes()
        date = calib_info['White']['datetime']

        fp = join(self.image_folder_filepath, 'RawImages')
        rawfiles = [f for f in sorted(listdir(fp)) if isfile(join(fp, f))]
        # rawfiles_indexed = rawfiles[self.start_index:self.end_index]
        rawfiles_indexed = rawfiles[self.start_index:self.end_index+1]
        timestamps = ['-'.join(t.split('_')[2].split('-')[:3]) for t in rawfiles_indexed]
        # timestamps = [re.search('_(.*)-', t).group(1)[5:] for t in rawfiles_indexed]
        date_formatted = date.strftime('%m-%d-%Y')
        timestamps = [date_formatted + '_' + t for t in timestamps]
        dt = [datetime.strptime(t,'%m-%d-%Y_%H-%M-%S') for t in timestamps]
        img_dict = {}
        for i in range(len(rawfiles_indexed)):
            scene_infile = open(fp + '/' + rawfiles_indexed[i],'rb')
            scene_image_array = np.fromfile(scene_infile,dtype=np.uint8,count=1280*1024)
            #key is the name of the image file, value is the datetime
            img_dict[rawfiles_indexed[i]] = {'raw_file':scene_image_array,'exp_time': img_exp_s1,'datetime':dt[i]} #dt[i]
        return img_dict

    def get_spec_calib(self):
        """ 
        get spectrometer calibration info 
        outputs a df with spectrometer data and datetime for raw images
        *Note: use this function to obtain combined_df
        >>>get_spec_calib(image_folder_filepath,spectro_filepath)
        """
        print("Getting spectroradiometer data...")
        calib_info = self.calibration_attributes() #nested dictionary
        # date = calib_info['White']['datetime']
        # raw_exposure, raw_dt = get_camera_settings(image_folder_filepath)
        # print(raw_exposure,raw_dt)
        # create df of spectroradiometer data for dark and white ref
        spec_calib = []
        for calib_type, attr_dict in calib_info.items():
            calib_df = self.get_spectroradiometer_data(attr_dict['datetime'])
            calib_df.rename(columns={ calib_df.columns[1]: calib_df.columns[1]+'_'+calib_type }, inplace = True)
            calib_df.set_index('wavelength',inplace=True)
            spec_calib.append(calib_df)
        
        spec_calib_dfs = pd.concat(spec_calib,axis=1)
        
        s1_bands = bands_wavelengths()#self.get_s1_bands()
        spec_bands = spec_calib_dfs.index
        spec_bands_index = [] #istantiate empty list to store index of spec_bands
        s1_band_count = 0 #initialise as the first band
        for k,v in enumerate(spec_bands):
            # print(k,v)
            if s1_band_count==len(s1_bands):
                break
            else:
                if v>s1_bands[s1_band_count]:
                    spec_bands_index.append(k)
                    s1_band_count+=1
        
        #subset rows from spec_calib_dfs
        spec_calib_dfs = spec_calib_dfs.iloc[spec_bands_index,:]

        spec_raw = []
        # date = calib_info['White']['datetime']
        raw_info = self.raw_attributes() #nested dictionary
        for img_name, attr_dict in raw_info.items():
            raw_df = self.get_spectroradiometer_data(attr_dict['datetime'])
            #subset rows from spec_calib_dfs
            raw_df = raw_df.iloc[spec_bands_index,:]
            raw_df.rename(columns={raw_df.columns[1]: raw_df.columns[1]+'_'+img_name}, inplace = True)
            raw_df.set_index('wavelength',inplace=True)
            spec_raw.append(raw_df)
        
        # spec_raw_dfs = pd.concat([df.set_index('wavelength') for df in spec_raw],axis=1)
        spec_raw_dfs = pd.concat(spec_raw,axis=1)
        combined_df = pd.concat([spec_raw_dfs,spec_calib_dfs],axis=1)
        return combined_df
    
    # def get_calibration_curve(self,curve_fitting_correction_fp):
    def get_calibration_curve(self):
        """
        curve_fitting_correction_fp (str): fp to the csv file that contains parameters for the best fit cubic curve for each wavelength
        returns a df with columns 'wavelength', 'w_a', 'w_b', 'w_c', 'w_d', 'd_a', 'd_b', 'd_c', 'd_d', 'irr_cutoff'
        """
        try:
            # df = pd.read_csv(curve_fitting_correction_fp)
            df_dict = calibration_curve()
            df = pd.DataFrame.from_dict(df_dict)
        except:
            # print("curve_fitting_correction_fp: {} cannot be found!".format(curve_fitting_correction_fp))
            df == None
        
        return df


    def get_naive_stitched(self):
        """ 
        outputs band_list of naive_stitched in the form of a dictionary. keys are band number, values are image array
        >>>get_naive_stitched()
        """
       
        raw_info = self.raw_attributes()

        print("Performing correction...")

        band_list = {} #store all the separate bands
        # print("Stitching images...({} perc overlap ratio)".format(round(self.overlap_ratio,2)))

        for i in range(self.band_numbers): #61 bands are evenly distributed across rows
            image_array = [] #uncorrected
            
            
            for (_,raw_attr) in raw_info.items(): #iterate across images #enumerate across dictionary  
                f = raw_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                image_array.append(band_array)

            naive_stitched = np.hstack(image_array) #stitch all images together
            band_list[i] = naive_stitched

        return band_list

    def get_reflectance_naive_stitched(self,curve_fitting_correction_fp=None):
        """ 
        correct images
        outputs band_list of reflectance naive stitched img in the form of a dictionary. keys are band number, values are corrected reflectance array
        >>>get_reflectance_naive_stitched()
        """
      
        spec_df = self.get_spec_calib()
        calib_info = self.calibration_attributes() #keys: White, Dark, sub keys: raw_file,exp_time, datetime
        # date = calib_info['White']['datetime']
        raw_info = self.raw_attributes()

        exp_white = calib_info['White']['exp_time']
        exp_dark = calib_info['Dark']['exp_time']

        if curve_fitting_correction_fp is None:
            print("Performing correction...")
        else:
            calibration_curve_df = self.get_calibration_curve()
            print("Performing correction using calibration curve...")

        #calibration curve fn
        def cubic_fn(x,a,b,c,d):
            """
            where x is the radiance
            returns the corrected DN i.e. DN/exp_time
            """
            y = a*x**3 + b*x**2 + c*x + d
            return y if y >0 else 0
        
        def reflectance_eqn(corrected_raw,corrected_white,corrected_dark):
            if corrected_raw > corrected_white:
                return self.Reflectance_White_Ref
            elif corrected_raw < corrected_dark:
                return 0
            else:
                return (corrected_raw - corrected_dark)/(corrected_white - corrected_dark)*self.Reflectance_White_Ref

        band_list = {} #store all the separate bands
        #band_list_overlapped = [] #store all the separate bands
        # print("Stitching images...({} perc overlap ratio)".format(round(self.overlap_ratio,2)))

        for i in range(self.band_numbers): #61 bands are evenly distributed across rows
            image_array = [] #uncorrected
            reflectance_image_array = [] #corrected & transformed to reflectance units
            for calib_name,calib_attr in calib_info.items():
                f = calib_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                # calib_info[calib_name][i] = np.mean(band_array) #mean DN and assign it the band number
                calib_info[calib_name][i] = np.max(band_array) #for both dark and white ref

            for j,(_,raw_attr) in enumerate(raw_info.items()): #iterate across images #enumerate across dictionary  
                f = raw_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                image_array.append(band_array)
                #perform correction on the line image (band_array)
                exp_raw = raw_attr['exp_time']
                new_radiance = spec_df.iloc[i,j] #indexing band & image
                old_radiance = spec_df.iloc[i,-2] #indexing band & White column
                White = calib_info['White'][i]
                Dark = calib_info['Dark'][i]
                if curve_fitting_correction_fp is None or calibration_curve_df is None:
                    new_White = new_radiance/old_radiance * White
                    reflectance_formula = lambda DN,White_DN: (DN/exp_raw)/(White_DN/exp_white)*self.Reflectance_White_Ref if ((White_DN - Dark) < 0.5) else self.Reflectance_White_Ref*((DN/exp_raw)-(Dark/exp_dark))/((White_DN/exp_white)-(Dark/exp_dark))
                    corrected_band_array = reflectance_formula(band_array,new_White) #np.array(map(reflectance_formula, band_array))
                else:
                    w_a, w_b, w_c, w_d, d_a, d_b, d_c, d_d, w_a_max, w_b_max, w_c_max, w_d_max, d_a_max, d_b_max, d_c_max, d_d_max, irr_cutoff = calibration_curve_df.iloc[i,1:]
                    corrected_White = cubic_fn(new_radiance,w_a, w_b, w_c, w_d)
                    corrected_Dark = cubic_fn(new_radiance,d_a, d_b, d_c, d_d)
                    corrected_Raw = band_array/exp_raw
                    corrected_band_array = reflectance_eqn(corrected_Raw,corrected_White,corrected_Dark)

                reflectance_image_array.append(corrected_band_array)

            reflectance_stitched = np.hstack(reflectance_image_array) #stitch all reflectance images together
            #keys are the wavelengths, values is another dictionary--> band_list is a nested dictionary
            band_list[i] = reflectance_stitched

        return band_list

    def get_overlap_ratios_per_line(self):
        """
        calculate overlap ratios per flight line. Cannot assume that overlap ratio is constant across the flight line
        number_of_image_slices (int): len(image_array) that corresponds the number of image slices
        frame_rate is not constant throughout flight
        >>>get_overlap_ratios_per_line()
        """
        # rawfiles = [f for f in sorted(listdir(self.image_folder_filepath)) if isfile(join(self.image_folder_filepath, f))]
        
        test_gps_list = [(self.test_gps_index[i],self.test_gps_index[i+1]) for i in range(0,len(self.test_gps_index),2)]
        gps_start_index,gps_end_index = test_gps_list[self.line_number]
        intermediate_gps_pts = self.unique_gps_df.iloc[gps_start_index:gps_end_index+1,[1,2,6,8]] #latitude, longitude, datetime,image_index (float)
        # frame_rate = 50
        pixel_size_at_sensor = 5.3 #um
        total_pixel_of_sensor_x = self.band_width#int(1280/61) #because hyperspectral images are line images, so actual pixel_x is 20
        focal_length = 16 #mm
        actual_size_of_sensor_x = pixel_size_at_sensor*total_pixel_of_sensor_x/1000 #mm
        fov_x = 2*math.atan(actual_size_of_sensor_x/2/focal_length)*180/math.pi #deg
        total_gnd_coverage_x = 2*self.height*math.tan(math.pi*fov_x/2/180) #metres #angle is converted to radians first
        # band_width = int(1280/61)


        lat_start, lon_start = intermediate_gps_pts.iloc[0,[0,1]].values
        lat_end, lon_end = intermediate_gps_pts.iloc[-1,[0,1]].values
        image_index_start = int(intermediate_gps_pts.iloc[0,3]) #cast float to int
        image_index_end = int(intermediate_gps_pts.iloc[-1,3]) #cast float to int
        time_diff = intermediate_gps_pts.iloc[-1,2] - intermediate_gps_pts.iloc[0,2]
        time_diff = time_diff.total_seconds()
        frame_rate = int((image_index_end - image_index_start)/time_diff)
        dist = Haversine((lon_start,lat_start),(lon_end,lat_end)).meters
        speed = dist/time_diff
        # print("Avg speed of drone: {:.2f}, Avg frame rate: {}".format(speed,frame_rate))
        d = (1/frame_rate)*speed #distance covered by drone in 1 fps
        overlap_x = total_gnd_coverage_x - d
        overlap_x_ratio = overlap_x/total_gnd_coverage_x #in ratio instead of percentage
        if overlap_x_ratio < 0: #or overlap_x_ratio > 1:
            print("Overlap_x_ratio < 0. Overlap ratio set as 0. Stitching will not be accurate!")
            overlap_x_ratio = 0
        elif overlap_x_ratio > 1:
            print("Overlap_x_ratio > 1. Overlap ratio set as 1. Stitching will not be accurate!")
            overlap_x_ratio = 1
        
        overlapped_cols = int(overlap_x_ratio*self.band_width) 
        # OR_x_list.append([(i,i+1), overlap_x_ratio])
        # OR_x_list = {'image_frames_list':[image_index_end - image_index_start],'OR_list':[overlapped_cols]}
        return overlapped_cols#OR_x_list

      

    def trimm_image(self,image_array):
        """ 
        conduct linear blending for an image array
        inputs:
        image_array (list of np array): greyscale slices of a band
        overlap_ratios_per_line (dictionaries): contains information on image_frames_range & overlapped_cols for adaptive overlap ratio
            keys of the dictionaries are:
                image_frames_list: contains a list of image frame index, 
                OR_list: contains a list of OR that corresponds to the overlapped cols for this image_frame and the prev one
        """
        overlapped_cols = self.get_overlap_ratios_per_line()
        trimmed_initial = [image_array[0]]
        trimmed_subsequent = [img[:,overlapped_cols:self.band_width] for img in image_array[1:]]
        trimmed_image_array = trimmed_initial+trimmed_subsequent
        stitched_img = np.hstack(trimmed_image_array)
        return stitched_img if self.reverse is False else np.fliplr(stitched_img)
       

    def get_stitched_img(self,destriping_fp="destriping_array.csv"):
        """ 
        outputs band_list of stitched img in the form of a dictionary. keys are band number, values are image array
        >>>get_stitched_img()
        """
        try:
            destriping_array = np.loadtxt(destriping_fp,delimiter=',')
        except Exception as E:
            print("Destriping array cannot be imported. Destriping of rgb images aborted.")
            destriping_array = np.ones((self.band_numbers,1024))
        raw_info = self.raw_attributes()
        print("generating hyperspectral images...")

        white_fp = join(self.image_folder_filepath,"WhiteRef")
        fp_list_white = [join(white_fp,i) for i in listdir(white_fp)]
        hyperspectral_img_list_white = [raw_to_hyperspectral_img(f) for f in fp_list_white]
        hyperspectral_white = np.mean(hyperspectral_img_list_white,axis=0)
        hyperspectral_white_array = np.mean(hyperspectral_white,axis=1)

        def destriping_img(img,hyperspectral_white_array,band):
            """
            destriping array.shape = (61,1024)
            """
            adjust_DN = lambda x,max_DN: max_DN/x
            avg_DN = hyperspectral_white_array[:,band]
            max_DN = np.max(avg_DN)
            avg_DN = np.where(avg_DN <=0,1,avg_DN)
            corrected_DN = adjust_DN(avg_DN,max_DN)

            nrows,ncols = img.shape
            adjust_DN_rgb = np.transpose(corrected_DN) #1D array
            repeated_DN_rgb = np.repeat(adjust_DN_rgb[:,np.newaxis],ncols,axis=1)
            destriped_img = repeated_DN_rgb*img
            destriped_img = np.where(destriped_img>255,255,destriped_img)
            return destriped_img.astype(np.uint8)

        band_list = {} #store all the separate bands
        # print("Stitching images...({} perc overlap ratio)".format(round(self.overlap_ratio,2)))
        
        for i in range(self.band_numbers): #61 bands are evenly distributed across rows
            image_array = [] #uncorrected
            for (_,raw_attr) in raw_info.items(): #iterate across images #enumerate across dictionary  
                f = raw_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                image_array.append(band_array)

            # band_list[i] = self.trimm_image(image_array)
            stitched_img = self.trimm_image(image_array)
            # destriped_img = destriping_img(stitched_img,destriping_array,i)
            destriped_img = destriping_img(stitched_img,hyperspectral_white_array,i)
            band_list[i] = destriped_img
            # overlap_ratios_per_line = self.get_overlap_ratios_per_line(len(image_array))
            # band_list[i] = self.trimm_image(image_array,overlap_ratios_per_line)

        return band_list
    
    def get_stitched_uncorrected_reflectance(self):
        """ 
        outputs band_list in reflectance values in % (i.e. values between 0 to 100)
        """
        def destriping_img(img,hyperspectral_white_array,band):
            """
            destriping array.shape = (61,1024)
            """
            adjust_DN = lambda x,max_DN: max_DN/x
            avg_DN = hyperspectral_white_array[:,band]
            max_DN = np.max(avg_DN)
            avg_DN = np.where(avg_DN <=0,1,avg_DN)
            corrected_DN = adjust_DN(avg_DN,max_DN)

            nrows,ncols = img.shape
            adjust_DN_rgb = np.transpose(corrected_DN) #1D array
            repeated_DN_rgb = np.repeat(adjust_DN_rgb[:,np.newaxis],ncols,axis=1)
            destriped_img = repeated_DN_rgb*img
            destriped_img = np.where(destriped_img>255,255,destriped_img)
            return destriped_img.astype(np.uint8)
        # Note: raw_info is only for one image_folder
        # if (self.overlap_ratio<5):
        #     print("Enter a higher overlap ratio")
        # band_width = int(1280/61)
        calib_info = self.calibration_attributes()
        # date = calib_info['White']['datetime']
        raw_info = self.raw_attributes()
        exp_white = calib_info['White']['exp_time']
        exp_dark = calib_info['Dark']['exp_time']

        white_fp = join(self.image_folder_filepath,"WhiteRef")
        fp_list_white = [join(white_fp,i) for i in listdir(white_fp)]
        hyperspectral_img_list_white = [raw_to_hyperspectral_img(f) for f in fp_list_white]
        hyperspectral_white = np.mean(hyperspectral_img_list_white,axis=0)
        hyperspectral_white_array = np.mean(hyperspectral_white,axis=1)

        print("No correction conducted!")
        #no need for calibration curve because we are not interpolating the new irradiance, so DN_white and DN_dark stays fixed
        def reflectance_eqn(raw,white,dark):
            if raw > white:
                return self.Reflectance_White_Ref
            elif raw < dark:
                return 0
            else:
                if abs(white - dark) < 0.001:
                    return 0
                else:
                    return (raw - dark)/(white - dark)*self.Reflectance_White_Ref

        vectorised_reflectance_eqn = np.vectorize(reflectance_eqn)
        band_list = {} #store all the separate bands
        #band_list_overlapped = [] #store all the separate bands
        # print("Stitching images...({} perc overlap ratio)".format(round(self.overlap_ratio,2)))
        
        for i in range(self.band_numbers): #61 bands are evenly distributed across rows
            image_array = [] #uncorrected
            uncor_reflectance_image_array = [] #uncorrected & transformed to reflectance units
            for calib_name,calib_attr in calib_info.items():
                f = calib_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                # calib_info[calib_name][i] = np.mean(band_array) #mean DN and assign it the band number
                calib_info[calib_name][i] = np.max(band_array) #for both dark and white ref

            for (_,raw_attr) in raw_info.items(): #iterate across images #enumerate across dictionary  
                f = raw_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                if self.destriping is True:
                    band_array = destriping_img(band_array, hyperspectral_white_array,i)
                image_array.append(band_array)
                #perform correction on the line image (band_array)
                exp_raw = raw_attr['exp_time']
                White = calib_info['White'][i]
                Dark = calib_info['Dark'][i]
                # reflectance_formula = lambda DN,White_DN: (DN/exp_raw)/(White_DN/exp_white)*self.Reflectance_White_Ref if ((White_DN - Dark) < 0.5) else self.Reflectance_White_Ref*((DN/exp_raw)-(Dark/exp_dark))/((White_DN/exp_white)-(Dark/exp_dark))
                # uncorrected_band_array = reflectance_formula(band_array,White)
                uncorrected_raw = band_array/exp_raw
                uncorrected_white = White/exp_white
                uncorrected_dark = Dark/exp_dark
                uncorrected_band_array = vectorised_reflectance_eqn(uncorrected_raw,uncorrected_white,uncorrected_dark)
                # uncorrected_band_array = reflectance_eqn(uncorrected_raw,uncorrected_white,uncorrected_dark)
                uncor_reflectance_image_array.append(uncorrected_band_array)

            # overlap_ratios_per_line = self.get_overlap_ratios_per_line(len(uncor_reflectance_image_array))
            #keys are the wavelengths, values is another dictionary--> band_list is a nested dictionary
            # band_list[i] = self.trimm_image(uncor_reflectance_image_array,overlap_ratios_per_line)
            band_list[i] = self.trimm_image(uncor_reflectance_image_array)
             #nested dictionary

        # save the rgb reflectance img
        self.get_rgb_reflectance_img(band_list)
        return band_list

    # def get_stitched_reflectance(self,curve_fitting_correction_fp=None,destriping_fp="destriping_array.csv"):
    def get_stitched_reflectance(self,curve_fitting_correction=True):
        """ 
        correction wrt to downwelling irradiance images
        outputs band_list of reflectance naive stitched img in the form of a dictionary. keys are band number, values are corrected reflectance array
        Must perform destriping before converting to reflectances since units are different.
        Destriping is performed on raw DN
        Must perform radiometric correction for each individual frame before stitching them up because irradiance for image frames is different
        >>>get_stitched_reflectance()
        """
       
        spec_df = self.get_spec_calib()
        calib_info = self.calibration_attributes()
        # date = calib_info['White']['datetime']
        raw_info = self.raw_attributes()

        exp_white = calib_info['White']['exp_time']
        exp_dark = calib_info['Dark']['exp_time']

        if curve_fitting_correction is False:
            print("Performing correction...")
            print("Linear radiometric calibration...")
        else:
            calibration_curve_df = self.get_calibration_curve()
            print("Performing correction using calibration curve...")
            print("Cubic function calibration...")

     
        white_fp = join(self.image_folder_filepath,"WhiteRef")
        fp_list_white = [join(white_fp,i) for i in listdir(white_fp)]
        hyperspectral_img_list_white = [raw_to_hyperspectral_img(f) for f in fp_list_white]
        hyperspectral_white = np.mean(hyperspectral_img_list_white,axis=0)
        hyperspectral_white_array = np.mean(hyperspectral_white,axis=1)

        def destriping_img(img,hyperspectral_white_array,band):
            """
            destriping array.shape = (61,1024)
            """
            adjust_DN = lambda x,max_DN: max_DN/x
            avg_DN = hyperspectral_white_array[:,band]
            max_DN = np.max(avg_DN)
            avg_DN = np.where(avg_DN <=0,1,avg_DN)
            corrected_DN = adjust_DN(avg_DN,max_DN)

            nrows,ncols = img.shape
            adjust_DN_rgb = np.transpose(corrected_DN) #1D array
            repeated_DN_rgb = np.repeat(adjust_DN_rgb[:,np.newaxis],ncols,axis=1)
            destriped_img = repeated_DN_rgb*img
            destriped_img = np.where(destriped_img>255,255,destriped_img)
            return destriped_img.astype(np.uint8)
        #calibration curve fn
        def cubic_fn(x,a,b,c,d):
            """
            used to interpolate corrected DN given a radiance value
            where x is the radiance
            a,b,c,d = parameters of cubic curve fitting
            returns the corrected DN i.e. DN/exp_time
            """
            y = a*x**3 + b*x**2 + c*x + d
            return y if y >0 else 0
        
        def reflectance_eqn(corrected_raw,corrected_white,corrected_dark):
            """
            corrected means DN/exp_time
            """
            if corrected_raw > corrected_white:
                return float(self.Reflectance_White_Ref)
            elif corrected_raw < corrected_dark:
                return 0.0
            else:
                if abs(corrected_white - corrected_dark) < 0.001:
                    return 0.0
                else:
                    return (corrected_raw - corrected_dark)/(corrected_white - corrected_dark)*self.Reflectance_White_Ref

        vectorised_reflectance_eqn = np.vectorize(reflectance_eqn)

        band_list = {} #store all the separate bands
        
        for i in range(self.band_numbers): #61 bands are evenly distributed across rows
            reflectance_image_array = [] #corrected & transformed to reflectance units for a wavelength
            for calib_name,calib_attr in calib_info.items():
                f = calib_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                calib_info[calib_name][i] = np.mean(band_array) #mean DN and assign it the band number
                # calib_info[calib_name][i] = np.max(band_array) #for both dark and white ref
                
            for j,(_,raw_attr) in enumerate(raw_info.items()): #iterate across images #enumerate across dictionary  
                f = raw_attr['raw_file']
                reshaped_raw = f.reshape(1024,1280)
                row_start = i*self.band_width
                row_end = i*self.band_width + self.band_width
                band_array = reshaped_raw[:,row_start:row_end]
                if self.destriping is True:
                    band_array = destriping_img(band_array, hyperspectral_white_array,i)
                #perform correction on the line image (band_array)
                exp_raw = raw_attr['exp_time']
                new_radiance = spec_df.iloc[i,j] #indexing band & image
                old_radiance = spec_df.iloc[i,-2] #indexing band & White column
                White = calib_info['White'][i] #mean DN of white ref
                Dark = calib_info['Dark'][i] #mean DN of dark red
                # new_White = new_radiance/old_radiance * White
                # reflectance_formula = lambda DN,White_DN: (DN/exp_raw)/(White_DN/exp_white)*self.Reflectance_White_Ref if ((White_DN - Dark) < 0.5) else self.Reflectance_White_Ref*((DN/exp_raw)-(Dark/exp_dark))/((White_DN/exp_white)-(Dark/exp_dark))
                # corrected_band_array = reflectance_formula(band_array,new_White) #np.array(map(reflectance_formula, band_array))
                if curve_fitting_correction is False or calibration_curve_df is None:
                    # print("Linear radiometric calibration...")
                    new_White = new_radiance/old_radiance * White
                    corrected_Raw = band_array/exp_raw
                    corrected_White = new_White/exp_white
                    corrected_Dark = Dark/exp_dark
                    corrected_band_array = vectorised_reflectance_eqn(corrected_Raw,corrected_White,corrected_Dark)
                    # reflectance_formula = lambda DN,White_DN: (DN/exp_raw)/(White_DN/exp_white)*self.Reflectance_White_Ref if ((White_DN - Dark) < 0.5) else self.Reflectance_White_Ref*((DN/exp_raw)-(Dark/exp_dark))/((White_DN/exp_white)-(Dark/exp_dark))
                    # corrected_band_array = reflectance_formula(band_array,new_White) #np.array(map(reflectance_formula, band_array))
                    # corrected_band_array = reflectance_formula(band_array,new_White)
                else:
                    # print("Cubic function calibration...")
                    w_a, w_b, w_c, w_d, d_a, d_b, d_c, d_d, w_a_max, w_b_max, w_c_max, w_d_max, d_a_max, d_b_max, d_c_max, d_d_max, irr_cutoff = calibration_curve_df.iloc[i,1:] #subset at the band
                    corrected_White = cubic_fn(new_radiance,w_a, w_b, w_c, w_d)
                    corrected_Dark = cubic_fn(new_radiance,d_a, d_b, d_c, d_d)
                    # corrected_White = cubic_fn(new_radiance,w_a_max, w_b_max, w_c_max, w_d_max)
                    # corrected_Dark = cubic_fn(new_radiance,d_a_max, d_b_max, d_c_max, d_d_max)
                    # corrected_Raw = band_array/exp_raw
                    corrected_Raw = band_array/exp_raw
                    corrected_band_array = vectorised_reflectance_eqn(corrected_Raw,corrected_White,corrected_Dark)
                    # corrected_band_array = vectorised_reflectance_eqn(corrected_Raw,corrected_White,corrected_Dark,new_radiance,irr_cutoff)
                reflectance_image_array.append(corrected_band_array)

            # overlap_ratios_per_line = self.get_overlap_ratios_per_line(len(reflectance_image_array))
            # band_list[i] = self.trimm_image(reflectance_image_array,overlap_ratios_per_line)
            band_list[i] = self.trimm_image(reflectance_image_array)

        # save the rgb reflectance img
        self.get_rgb_reflectance_img(band_list)
        return band_list
    
    def sunglint_sugar_correction(self,reflectance):
        """ 
        :param reflectance (dict): stitched reflectance where keys are band index, and values are reflectance of each band
        returns dict of reflectances where keys are band index
        """
        def get_rgb_reflectance_img(reflectance):
            """
            :param reflectance (dict of reflectance values): reflectance values are in %, i.e. ranges from 0 to 100
            """
            reflectance_rgb = np.stack([reflectance[i] for i in self.rgb_bands],axis=2)/100*255
            reflectance_rgb = reflectance_rgb.astype(np.uint8)
            img = Image.fromarray(reflectance_rgb, 'RGB')

            parent_dir = join(self.fp_store,self.glint_directory)
            if (not exists(parent_dir)):
                mkdir(parent_dir)
            rgb_fp = join(parent_dir,self.rgb_deglinted_fp)
            img.save(rgb_fp,save_all=True) #creates a tif file with 3 bands
            return reflectance_rgb
        
        im_aligned = np.stack(list(reflectance.values()),axis=2)
        sgc = sugar.SUGAR(im_aligned,glint_mask_method="cdf")
        sgc_reflectance = sgc.get_corrected_bands()
        sgc_reflectance = {i: sgc_reflectance[i] for i in range(len(sgc_reflectance))}
        get_rgb_reflectance_img(sgc_reflectance)
        
        return sgc_reflectance

    # def view_pseudo_colour(self,red_band,green_band,blue_band,destriping_fp="destriping_array.csv"):
    def view_pseudo_colour(self,red_band,green_band,blue_band):
        """ 
        view pseudo-color
        red_band (int)
        green_band (int)
        blue_band (int)
        destriping_fp (str): filepath of where adjust_DN array is found to correct for the stripe noises
        >>>view_pseudo_colour(red_band,green_band,blue_band,prefix)
        """
      

        def raw_to_img(scene_image_array,bands=(red_band,green_band,blue_band),band_width=self.band_width):
            """
            fp (str): filepath of the raw image
            bands (tuple of int): 3 tuple of list of the rgb bands e.g. ()
            outputs arrays of rgb images only
            """
            reshaped_raw = scene_image_array.reshape(1024,1280)
            image_array = []
            for i in bands:
                row_start = i*band_width
                row_end = i*band_width + band_width
                band_array = reshaped_raw[:,row_start:row_end]
                image_array.append(band_array)
            rgb_img = np.dstack(image_array)
            return rgb_img

       
        def destriping_img(img,bands=[red_band,green_band,blue_band]):
            """
            destriping array.shape = (61,1024)
            """
            white_fp = join(self.image_folder_filepath,"WhiteRef")
            fp_list_white = [join(white_fp,i) for i in listdir(white_fp)]
            hyperspectral_img_list_white = [raw_to_hyperspectral_img(f) for f in fp_list_white]
            hyperspectral_white = np.mean(hyperspectral_img_list_white,axis=0)
            hyperspectral_white_array = np.mean(hyperspectral_white,axis=1)
            adjust_DN = lambda x,max_DN: max_DN/x
            rgb_array = hyperspectral_white_array[:,bands]

            adjusted_rgb_array = [adjust_DN(rgb_array[:,i],np.max(rgb_array[:,i])) for i in range(len(bands))]
            adjusted_rgb_array = np.transpose(np.vstack(adjusted_rgb_array))

            nrows,ncols,c = img.shape
            repeated_DN_rgb = np.repeat(adjusted_rgb_array[:,:,np.newaxis],ncols,axis=2)
            repeated_DN_rgb = np.swapaxes(repeated_DN_rgb,1,2)
            destriped_img = repeated_DN_rgb*img
            destriped_img = np.where(destriped_img>255,255,destriped_img)
            return destriped_img.astype(np.uint8)

        raw_info = self.raw_attributes()
        raw_image_list = [img['raw_file'] for img in raw_info.values()]
        img_arrays = [raw_to_img(f) for f in raw_image_list]
        stitched_rgb = self.trimm_image(img_arrays)
        # destriped_stitched_rgb = destriping_img(stitched_rgb,destriping_array,bands=[red_band,green_band,blue_band])
        if self.destriping is True:
            stitched_rgb = destriping_img(stitched_rgb)
        # img = Image.fromarray(stitched_rgb, 'RGB')
        img = Image.fromarray(stitched_rgb, 'RGB')
        # rgb_fp = '{}_rgb_image_line_{}_{}_{}.tif'.format(self.prefix,str(self.line_number).zfill(2),self.start_index,self.end_index)
        rgb_fp = join(self.fp_store,self.rgb_fp)
        img.save(rgb_fp,save_all=True) #creates a tif file with 3 bands
        return stitched_rgb
      
    def get_rgb_reflectance_img(self,reflectance,red_band=38,green_band=23,blue_band=15):
        """
        :param reflectance (dict of reflectance values): reflectance values are in %, i.e. ranges from 0 to 100
        """
        reflectance_rgb = np.stack([reflectance[i] for i in [red_band,green_band,blue_band]],axis=2)/100*255
        reflectance_rgb = reflectance_rgb.astype(np.uint8)
        img = Image.fromarray(reflectance_rgb, 'RGB')

        parent_dir = join(self.fp_store,self.rgb_reflectance_directory)
        if (not exists(parent_dir)):
            mkdir(parent_dir)
        rgb_fp = join(parent_dir,self.rgb_reflectance_fp)
        img.save(rgb_fp,save_all=True) #creates a tif file with 3 bands
        return reflectance_rgb

    def generate_hyperspectral_images(self):
        """
        save individual bands as a greyscale image
        """
        fp_store_hyperspectral = join(self.fp_store,self.hyperspectral_directory)
        if not exists(fp_store_hyperspectral): #if hyperspectral folder d.n.e, create one
            mkdir(fp_store_hyperspectral)

        band_list = self.get_stitched_img()
        for band_number,img in band_list.items():
            band_fp = f'{self.prefix}_band{band_number}_image_line_{str(self.line_number).zfill(2)}_{self.start_index}_{self.end_index}.tif'
            band_fp = join(fp_store_hyperspectral,band_fp)
            img = Image.fromarray(img)
            img.save(band_fp)


    def get_affine_transformation(self,img,angle_to_rotate):
        """
        this function only rotates the image and makes sure the bounding box are optimal
        inputs are filepath of rgb image, 
        angle_to_rotate (can be manually specified), but it's calculated from flight path angle. 
        Rotation is about the horizontal. +ve means rotate acw, -ve means rotate cw about horizontal
        reverse (Default = False)
        >>>get_affine_transformation(img,angle_to_rotate,reverse=False)
        """
        img = np.flipud(img) #flip vertically
        # angle_to_rotate = float(input("Enter angle of rotation. +ve means rotate acw, -ve means rotate cw about horizontal"))
        rows, cols = img.shape[0], img.shape[1]
        center = (cols//2,rows//2)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center,angle_to_rotate,1) #center, angle, scale
        # rotate the image using cv2.warpAffine
        # rotated_image = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(width, height))
        cosofRotationMatrix = np.abs(rotation_matrix[0][0]) #scale*cos(angle)
        sinofRotationMatrix = np.abs(rotation_matrix[0][1]) #scale*sin(angle)
        # Now will compute new height & width of
        # an image so that we can use it in
        # warpAffine function to prevent cropping of image sides
        newImageHeight = int((cols * sinofRotationMatrix) +
                            (rows * cosofRotationMatrix))
        newImageWidth = int((cols * cosofRotationMatrix) +
                            (rows * sinofRotationMatrix))
        # After computing the new height & width of an image
        # we also need to update the values of rotation matrix
        rotation_matrix[0][2] += (newImageWidth/2) - center[0]
        rotation_matrix[1][2] += (newImageHeight/2) - center[1]
        # Now, we will perform actual image rotation
        rotatingimage = cv2.warpAffine(img, rotation_matrix, (newImageWidth, newImageHeight))

        return rotatingimage#cv2.cvtColor(rotatingimage,cv2.COLOR_BGR2RGB)

    def get_reflectance_ROI(self,line_number,img,reflectance_list,x,y,label,width,height):
        """ 
        view reflectance (corrected)
        plot graph of reflectance and corresponding location in image
        """
      

        fig, ax = plt.subplots(1,2,figsize=(10,5))
        # ax[0].imshow(uncorrected_reflectance[int(61/2)],cmap='gray')
        ax[0].imshow(img,cmap='gray')
        #(0,0) defined at upper left corner
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none') #xy defined as the upper left corner for image arrays
        ax[0].add_patch(rect)
        ax[0].set_title("Stitched image (mid band image)")
        ax[0].text(x, y, label, style='italic',bbox={'facecolor': 'white', 'alpha': 0.5},fontsize="medium")
        wavelength_list = bands_wavelengths()#self.get_s1_bands()
        ax[1].plot(wavelength_list,reflectance_list,'g-',label="Reflectance (corrected)",)
        ax[1].legend()
        ax[1].set_xlabel('Wavelength (nm)')
        ax[1].set_ylabel('Reflectance (%)')
        ax[1].set_title('Reflectance of selected ROI\nLine: {}'.format(line_number))
        ax[1].set_ylim(0,120)
        plt.show()
        # plt.savefig('{}_line{}_reflectance_ROI.png'.format(self.prefix,line_number))
        return

    
    def get_imaging_time(self,unique_gps_df,gps_start_index,gps_end_index,P):
        """
        this function gets the imaging time from a particular point P in the image
        P is stored as lon then lat
        imaging time is obtained from GPS information since GPS information is recorded every second during imaging
        to obtain the closest imaging time, we will iterate across every GPS point between gps_start_index and gps_end_index
        saves more computation time compared to iterating across all GPS points
        """
        coords = np.transpose(unique_gps_df.iloc[gps_start_index: gps_end_index,[0,1,2]].values) #0=time,1=latitude, 2=longitude
        #first row is time, second row is latitute, third row is longitude
        n_coords = coords.shape[1] #number of GPS points between gps_start_index & gps_end_index
        closest_dist = np.linalg.norm(P-coords[[2,1],0])
        closest_imaging_time = coords[0,0]
        # print(closest_dist,closest_imaging_time)
        for i in range(n_coords):
            c = coords[[2,1],i] #subset lon & lat, in that order
            d = np.linalg.norm(P-c)
            t = coords[0,i]
            if d < closest_dist:
                closest_dist = d
                closest_imaging_time = t

        return closest_imaging_time
    
    def get_nechad_predicted_image(self,reflectance,reflectance_band=27,A_tau=87.75,C=0.2324):
        """ 
        :param reflectance (dict): reflectance dictionary
        :param reflectance_band (int): band index that corresponds to band 660nm
        """
        nechad = lambda x,a,c: a*x/(1-x/c)
        band_red = reflectance[reflectance_band]/100 #convert reflectance from 0 to 1
        predicted = nechad(band_red, A_tau,  C)
        predicted_image = predicted.astype(np.uint16)
        #-------------save predicted img------------------
        # prediction_directory = 'Prediction'
        fp_store_prediction = join(self.fp_store,self.prediction_directory)
        if not exists(fp_store_prediction): #if prediction folder d.n.e, create one
            mkdir(fp_store_prediction)

        img = Image.fromarray(predicted_image)
        # img_fp = self.rgb_fp.replace('rgb','predicted')
        img_fp = join(self.fp_store,self.prediction_directory,self.predicted_fp)
        img.save(img_fp)
        return 

    def get_predicted_image(self,model,covariates_index_list,\
        model_type="XGBoost",scaling_factor=40,\
        reflectance=None,\
        reflectance_to_csv=False,transform_y=False):
        """
        this function gets the reflectance of the *entire image* rather from fixed points
        image is first downsampled to reduce processing time then upsampled again
        fp_store (str): folderpath of where images are stored
        model (str): name of a xgboost trained model that takes in an input of nx61, where 61 columns are the reflectance of 61 bands
        scaling_factor (int): size of a block for averaging nearby pixels (default = 40)
        glint_corrected_reflectance (dict): sunglint corrected reflectance
        reflectance (dict): non-sunlgint corrected reflectance from .get_stitched_reflectance()
        covariates_index_list (list of int): to indicate the covariates that are used for prediction
            if covariates_index_list == None, then all variables are used for stitching
        reflectance_to_csv (boolean): whether to export all the reflectance pixels to csv. Default = False.
            *Not recommended to export to csv because files are very big and will slow down processing a lot!
        transform_y (bool): conduct log transformation or not to y variable
        transform_x (bool): conduct log transformation or not to all x covariates
        >>> get_predicted_image(model,scaling_factor=40)
        outputs a grayscale image (.tif) in the user's directory
        #     # import machine learning model, can be an XGBoost model or a Deep Learning model
        #     model_fp = r"Models\turbidity_prediction\XGB_corrected_trimmed100_1.json" #input ur XGBoost model
        #     # model_fp = r"Models\turbidity_prediction\saved_models\DNN_corrected_trimmed100_1" #DNN model
        #     model_type = "XGBoost" #If XGBoost model is used, change to "XGBoost", else if DNN model is used, change to "DNN"
        #     covariates_index_list = "4:15,33:38,43:46,53:54" #indices of covariates
        #     covariates_index_list = covariates_str_to_list(covariates_index_list)
        #     downsample_slider = 40

        #     #4:54, #4:15,33:38,43:46,53:54
        #     # test_stitch_class.get_predicted_image(model=config_file['-MODEL_FILEPATH-'],\
        #     #         covariates_index_list = config_file['-MODEL_PREDICTORS-'],\
        #     #         model_type=config_file['model_type'],\
        #     #         # radiometric_correction=config_file['-RADIOMETRIC_CHECKBOX-'],\
        #     #         scaling_factor=int(config_file['-DOWNSAMPLE_SLIDER-']),\
        #     #         glint_corrected_reflectance=sgc_reflectance,reflectance=reflectance)
        """
        
        # if glint_corrected_reflectance is not None:
        #     corrected_reflectance = glint_corrected_reflectance
        # else:
        #     corrected_reflectance = reflectance

        # if glint_corrected_reflectance is None and reflectance is None:
        #     return None
        corrected_reflectance = reflectance
        
        mid_band_img = corrected_reflectance[self.band_numbers//2]
        nrow,ncol = mid_band_img.shape
        # masked_corrected_reflectance = [np.multiply(arr,mask) for arr in corrected_reflectance.values()]
        bands = bands_wavelengths()#self.get_s1_bands().tolist()
        #PERFORM DOWNSCALING FIRST BEFORE RESHAPING AND PREDICTING
        flattened_corrected_reflectance = np.hstack([arr.reshape(-1,1) for arr in corrected_reflectance.values()]) #matrix, where rows are observations, columns are bands
        if reflectance_to_csv == True:
            
            bands = ["{:.2f}".format(float(b)) for b in bands] #cast bands to a string to ensure decimal place are consistent
            pd.DataFrame(flattened_corrected_reflectance,columns=bands).to_csv('{}_image_line_{}_{}_{}.csv'.format(self.prefix,str(self.line_number).zfill(2),self.start_index,self.end_index),index=False)
        #----load model--------
        if model_type == "XGBoost":
            print("Using XGBoost model for prediction...")
            bst = xgb.Booster({'nthread':4})
            bst.load_model(model)
            #----load model--------
            #----preprocess data--------
            if type(covariates_index_list) != list or len(covariates_index_list) > len(bands):
                print("Input must be a list or number of inputs > bands!")
            elif covariates_index_list == None:
                flattened_corrected_reflectance = flattened_corrected_reflectance
            else:
                flattened_corrected_reflectance = flattened_corrected_reflectance[:,covariates_index_list]
            #----preprocess data--------
            dtest = xgb.DMatrix(flattened_corrected_reflectance)
            y_hat = bst.predict(dtest) #outputs a float
        elif model_type == "DNN":
            print("Using DNN model for prediction...")
            # loaded_model = load_model("saved_models/DNN_survey234", custom_objects=ak.CUSTOM_OBJECTS)
            from tensorflow.keras.models import load_model
            loaded_model = load_model(model)
            if type(covariates_index_list) != list or len(covariates_index_list) > len(bands):
                print("Input must be a list or number of inputs > bands!")
            elif covariates_index_list == None:
                flattened_corrected_reflectance = flattened_corrected_reflectance
            else:
                flattened_corrected_reflectance = flattened_corrected_reflectance[:,covariates_index_list]
            #Try out the loaded model with loaded data
            y_hat = loaded_model.predict(flattened_corrected_reflectance)

        if transform_y == True:
            y_hat = np.exp(y_hat)
        y_hat[y_hat<0] = 0 #replace all -ve values with 0 to ensure that negative values dont get converted to 16bit values
        predicted_image = y_hat.astype(np.uint16).reshape(nrow,ncol) #cast into unsigned 16bit int
        # print("predicted_image (16bit):{}".format(predicted_image))
        #-------------Plotting------------------
        # fig, ax = plt.subplots(1,2,figsize=(15,5))
        # ax[0].imshow(predicted_image, cmap='jet')
        print("Before block reduce: (min:{}, max:{})".format(np.min(predicted_image),np.max(predicted_image)))
        #-------------Plotting------------------
        #-------------Block reduce------------------
        predicted_reduced = block_reduce(predicted_image,block_size=(scaling_factor,scaling_factor),func=np.mean)
        predicted_upscale = rescale(predicted_reduced, scaling_factor, anti_aliasing=False)
        predicted_image = add_padding(predicted_image,predicted_upscale)
        print("After block reduce: (min:{}, max:{})".format(np.min(predicted_image),np.max(predicted_image)))
        # im = ax[1].imshow(predicted_image,cmap='jet')
        # cax = fig.add_axes([0.91,0.3,0.03,0.4])
        # fig.colorbar(im, cax=cax)
        # plt.show()
        #-------------Block reduce------------------

        #-------------save predicted img------------------
        # prediction_directory = 'Prediction'
        fp_store_prediction = join(self.fp_store,self.prediction_directory)
        if not exists(fp_store_prediction): #if prediction folder d.n.e, create one
            mkdir(fp_store_prediction)

        img = Image.fromarray(predicted_image)
        # img_fp = self.rgb_fp.replace('rgb','predicted')
        img_fp = join(self.fp_store,self.prediction_directory,self.predicted_fp)
        img.save(img_fp)
        return #predicted_image---

    def get_reflectance_from_GPS(self,tss_lat,tss_lon,tss_measurements,\
        radius,mask = None,\
        reflectance = None,preview=False):
        """ 
        tss_lat, tss_lon, tss_mesasurements: corresponding information from water quality df (stored in a list)
        radius (int): extent of ROI (in integer)
        preview (boolean): plot graph of reflectance and corresponding location in image (Default = True)
        """
        gps_indexes = [(self.test_gps_index[i],self.test_gps_index[i+1]) for i in range(0,len(self.test_gps_index)-1,2)]
        # print(gps_indexes)
        gps_start_index, gps_end_index = gps_indexes[self.line_number] #use line number to replace 0
        # print("Performing correction...")
        # if glint_corrected_reflectance is None:
        #     corrected_reflectance = reflectance#self.get_stitched_reflectance() #stored as dictionary. keys are bands and values are img arrays
        # else:
        #     corrected_reflectance = glint_corrected_reflectance

        # if glint_corrected_reflectance is None and reflectance is None:
        #     return None
        corrected_reflectance = reflectance
        corrected_reflectance_stack = np.dstack([arr for arr in corrected_reflectance.values()]) #stack along depth
        # corrected_reflectance = self.get_stitched_reflectance()
        #-----------PERFORM MASKING--------------------
        if mask is not None:
            corrected_reflectance_stack[mask!=0] = 0

        #-----------PERFORM MASKING--------------------

        #apply get_affine_transformation to each corrected_reflectance band
        #inclusive of gps start index and gps end index
        wavelength_list = bands_wavelengths()#self.get_s1_bands().tolist()
        coords = np.transpose(self.unique_gps_df.iloc[[gps_start_index, gps_end_index],[1,2]].values) #1=latitude, 2=longitude
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
        transformed_imges = self.get_affine_transformation(corrected_reflectance_stack,angle)
        nrow_transformed, ncol_transformed = transformed_imges.shape[0],transformed_imges.shape[1]
        # transformed_imges = [self.get_affine_transformation(arr,angle) for arr in masked_corrected_reflectance] #apply transformation to all bands
        # nrow_transformed, ncol_transformed = transformed_imges[0].shape[0], transformed_imges[0].shape[1]

        #-------CONDUCT TRANSFORMATION (end)-------------
        

        #---------EXTRAPOLATE GPS (start)---------------
        nrow,ncol = corrected_reflectance[0].shape #nrow = h, ncol = w
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
        TSS_info = {} #initialise empty dict for averaged band reflectance
        TSS_df_dict = {} #save info as a pd df for averaged band reflectance
        TSS_df_dict_list = [] #an empty list
        #keys are the indexes of tss measurements
        for i in range(len(tss_lat)): #where i = number of tss observations
            # if tss_lat[i]<ul[0] and tss_lat[i]>lr[0] and tss_lon[i]<lr[1] and tss_lon[i]>ul[1]:
            P = np.array([tss_lon[i],tss_lat[i]])
            if check_within_bounding_box(vertices,P) == True:
                vert_pixel = int((ul[0] - tss_lat[i])/np.abs(lat_res_per_pixel)) #reference from the uppermost extent-->row no.
                hor_pixel = ncol_transformed - int((lr[1]-tss_lon[i])/np.abs(lon_res_per_pixel)) #reference from the most left extent-->col no.
                #extract patch from image array
                #need to check if patch contains black area due to affine transformation
                imaging_time = self.get_imaging_time(self.unique_gps_df,gps_start_index,gps_end_index,P) #get imaging time of a particular point (tss measurement)
                ROI = transformed_imges[vert_pixel-radius:vert_pixel+radius , hor_pixel-radius:hor_pixel+radius , :]
                # print('transformed_img.shape:{}, ROI.shape: {}'.format(transformed_imges.shape,ROI.shape))
                ROI_list = [ROI[:,:,i] for i in range(self.band_numbers)]
                band_reflectance = [np.mean(i[i!=0]) for i in ROI_list] #remove 0s, then calculate the mean of the ROI for each layer
                
                TSS_info[i] = {'line_number':self.line_number,'imaging_time':imaging_time,'tss_lat':tss_lat[i],'tss_lon':tss_lon[i],'tss_conc':tss_measurements[i],'band_reflectance':band_reflectance,'ROI':(hor_pixel-radius,vert_pixel-radius),'ROI_extent':int(2*radius)}
                TSS_df_dict[i] = [i,imaging_time, tss_measurements[i],tss_lat[i],tss_lon[i]] + band_reflectance #where i is the observation number of tss measurements
        
        #------save spectral info in a df---------------

        # else:
        extracted_spectral_directory = 'Extracted_Spectral_Information'
        fp_store_extracted_spectral = join(self.fp_store,extracted_spectral_directory)
        if not exists(fp_store_extracted_spectral): #if hyperspectral folder d.n.e, create one
            mkdir(fp_store_extracted_spectral)
        df_columns = ['observation_number','imaging_time','tss_conc','tss_lat','tss_lon'] + ['band_{}'.format(i) for i in range(self.band_numbers)]
        fp_csv = '{}_TSS_spectral_info_line{}.csv'.format(self.prefix,self.line_number)
        fp_csv = join(fp_store_extracted_spectral,fp_csv)
        pd.DataFrame.from_dict(TSS_df_dict,orient='index',columns=df_columns).to_csv(fp_csv,index=False)
        #------save spectral info in a df---------------

        if preview == True:
        #-------plot to see if extraction of spectral info is correct----------
            for k in TSS_info.keys():
                reflectance_list = TSS_info[k]['band_reflectance']
                x,y = TSS_info[k]['ROI']
                width = TSS_info[k]['ROI_extent']
                label = TSS_info[k]['tss_conc']
                self.get_reflectance_ROI(self.line_number,transformed_imges[int(self.band_numbers/2)],reflectance_list,x,y,label,width,width)
            # reflectance_list = [TSS_info[i]['band_reflectance'] for i in range(3)] #list of a list of band reflectance

        # self.get_reflectance_ROI(transformed_imges[int(61/2)],reflectance_list,x,y,width,height)


    def preprocess_spectral_info(self,export_to_array = False):
        """ 
        takes in prefix as an input and imports all spectral info csv and preprocesses them by removing rows with all 0s or NAs
        outputs a preprocessed df. 
        Note that there may be duplicates in values because the same TSS gps coordinates may be found in more than 1 line image
        Some reflectance values may be low because ROI may be located on the edge of the images where it covers the black background,
        and thus the averaged values is low
        """
        extracted_spectral_directory = 'Extracted_Spectral_Information'
        all_csv_files = []
        
        if (export_to_array == True):
            TSS_array_fp = join(self.fp_store,extracted_spectral_directory,"{}_TSS_spectral_array*.csv".format(self.prefix))
            # for file in glob("{}_TSS_spectral_array*.csv".format(self.prefix)):
            for file in glob(TSS_array_fp):
                all_csv_files.append(file)
        else:
            TSS_info_fp = join(self.fp_store,extracted_spectral_directory,"{}_TSS_spectral_info_line*.csv".format(self.prefix))
            # for file in glob("{}_TSS_spectral_info*.csv".format(self.prefix)):
            for file in glob(TSS_info_fp):
                all_csv_files.append(file)

        list_df = [] #contains a list of all the df from differrent line images
        for f in all_csv_files:
            if (export_to_array == True):
                line_number = f.replace(join(self.fp_store,extracted_spectral_directory,f"{self.prefix}_TSS_spectral_array_"),'').replace('.csv','').replace('line','')
                # line_number = f.replace(self.prefix+'_TSS_spectral_array_','').replace('.csv','').replace('line','')
            else:
                line_number = f.replace(join(self.fp_store,extracted_spectral_directory,f"{self.prefix}_TSS_spectral_info_line"),'').replace('.csv','')
                # line_number = f.replace(self.prefix+'_TSS_spectral_info_','').replace('.csv','').replace('line','')
            # print(line_number)
            df = pd.read_csv(f)
            df.insert(loc=0,column='line_number',value=int(line_number))
            
            list_df.append(df)

        # number_lines_df = len(list_df)

        if export_to_array == True:
            TSS_spectral_array_fp =  join(self.fp_store,extracted_spectral_directory,'{}_TSS_spectral_array.csv'.format(self.prefix))
            pd.concat(list_df,axis=0).sort_values(['line_number','Wavelength','observation_number','imaging_time','tss_lat','tss_lon'],ascending=True).dropna().to_csv(TSS_spectral_array_fp,index=False) 
        else:
            bands = bands_wavelengths()#self.get_s1_bands().tolist()
            bands = ["{:.2f}".format(float(b)) for b in bands] #cast bands to a string 

            reflectance_df = pd.concat(list_df,axis=0)
            reflectance_df.columns = ['line_number','observation_number','imaging_time', "Concentration", 'Lat','Lon'] + bands
            TSS_spectral_info_fp = join(self.fp_store,extracted_spectral_directory,'{}_TSS_spectral_info.csv'.format(self.prefix))
            reflectance_df[(reflectance_df.iloc[:,-self.band_numbers:]!=0).any(axis=1)].dropna(axis=0,thresh=self.band_numbers).sort_values(['line_number','imaging_time','Lat','Lon'],ascending=True).to_csv(TSS_spectral_info_fp,index=False) 
        return 

    def get_mask(self,model,type="XGBoost"):
        """
        cut_images (dict): with keys image, padded, ncol (if padded is True)
        reconstruct mask image from cut_images and save the mask
        model (loaded from checkpoints)
        type (str): unet or XGBoost
        """
        rgb_fp = '{}_rgb_image_line_{}_{}_{}.tif'.format(self.prefix,str(self.line_number).zfill(2),self.start_index,self.end_index)
        rgb_fp = join(self.fp_store,rgb_fp)
        print(f'rgb_fp loaded for masking: {rgb_fp}')
        if type == "unet":
            img = cv2.imread(rgb_fp,1)
            test_tif = cv2.imread(rgb_fp) #has to be loaded using cv2 BGR format as the input to the model
            rgb_tif = cv2.cvtColor(test_tif, cv2.COLOR_BGR2RGB) 
            cut_images = cut_into_512(img) #dictionary with with keys: image, padded, ncol (if padded is True)
            cut_images = predict_mask(model,cut_images) #create new keys: 'mask'
            #----reconstruct the mask into a complete image---
            cut_img_list = [v['mask'] for k,v in cut_images.items()]
            ncol = len(cut_images.keys())//2
            row1 = np.hstack(cut_img_list[:ncol])
            row2 = np.hstack(cut_img_list[ncol:])
            mask = np.vstack([row1,row2])
            if cut_images[1,ncol-1]['padded'] is True:
                end_col = cut_images[1,ncol-1]['ncol']
                mask = mask[:,:end_col]

        elif type == "XGBoost":
            mask = XGBoost_segmentation(rgb_fp,model)
        #save mask to a directory
        # mask_directory = 'Mask'
        fp_store_mask = join(self.fp_store,self.mask_directory)
        if not exists(fp_store_mask): #if mask folder d.n.e, create one
            mkdir(fp_store_mask)

        # mask_fp = f'{self.prefix}_mask_line_{str(self.line_number).zfill(2)}_{self.start_index}_{self.end_index}.png'
        mask_fp = join(fp_store_mask,self.mask_fp)


        img_mask = Image.fromarray(mask.astype(np.uint8))
        img_mask.save(mask_fp)


        return mask#,masked_img


def plot_TSS_conc(df,band_numbers=int(61)):
    """ 
    input is a df with last 61 columns as wavelengths and a column called 'Concentration' that corresponds to TSS concentration
    unit (str): label for the colour map e.g. FNU or mg/l
    outputs individual reflectance curve mapped to TSS concentration
    """
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

    
    df = df.fillna(0) #fill nas with 0
    #df.iloc[:,[3,6:67]].groupby('Conc').mean().transpose()
    ncols = len(df.columns)
    col_names = df.columns.tolist()
    # columns = [5] + list(range(ncols-61,ncols))
    columns = ['Concentration'] + col_names[-band_numbers:]
    df_plot = df.loc[:,columns].set_index('Concentration')
    # df_plot = df.iloc[:,columns].set_index('Conc')#.transpose()
    wavelength = df_plot.columns.tolist()

    concentration = df_plot.index.tolist()
    n_lines = len(concentration)
    y_array = df_plot.reset_index().iloc[:,-band_numbers:].values #reflectance
    x_array = np.array([wavelength for i in range(n_lines)])

    fig, ax = plt.subplots(figsize=(8,6))
    lc = multiline(x_array, y_array, concentration, cmap='Spectral_r', lw=1)
    # ax.set_yscale("log")
    ax.set_ylim(0,int(np.max(y_array)*1.10))
    ax.set_xlim(450,950)
    
    axcb = fig.colorbar(lc)
    axcb.set_label('Concentration')
    ax.set_title('Reflectance of water quality variable\nN = {}'.format(len(df.index)))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (%)')
    plt.show(block=False)
    #plt.savefig('{}_reflectance_ROI.png'.format(prefix))
    return #df_plot.to_csv('{}_spectral_info.csv'.format(prefix))


def bands_wavelengths():
    bands = [411.79,
            421.04,
            430.29,
            439.54,
            448.78,
            458.02,
            470,
            476.48,
            485.71,
            494.93,
            504.15,
            513.36,
            522.57,
            531.78,
            540.98,
            550,
            559.37,
            568.56,
            577.74,
            586.92,
            596.1,
            605.27,
            614.44,
            620,
            632.77,
            641.93,
            651.08,
            660.23,
            669.37,
            678.51,
            687.65,
            696.78,
            705.91,
            715.04,
            724.16,
            733.27,
            742.39,
            751.49,
            760.6,
            769.7,
            778.8,
            787.89,
            796.98,
            806.06,
            815.14,
            824.22,
            833.29,
            842.36,
            851.42,
            860.48,
            869.54,
            878.59,
            887.64,
            896.68,
            905.72,
            914.76,
            923.79,
            932.82,
            941.84,
            950.86,
            959.88]
    return bands

def calibration_curve():
    return {'wavelength': {0: '411.709', 1: '421.175', 2: '430.252', 3: '439.666', 4: '448.692', 5: '458.054', 6: '469.896', 7: '476.337', 8: '485.618', 9: '494.871', 10: '504.096', 11: '513.293', 12: '522.462', 13: '531.954', 14: '541.065', 15: '550.148', 16: '559.201', 17: '568.573', 18: '577.567', 19: '586.877', 20: '596.155', 21: '605.401', 22: '614.273', 23: '620.059', 24: '632.606', 25: '642.060', 26: '651.143', 27: '660.193', 28: '669.209', 29: '678.524', 30: '687.803', 31: '696.716', 32: '705.923', 33: '715.093', 34: '724.226', 35: '733.322', 36: '742.380', 37: '751.401', 38: '760.704', 39: '769.647', 40: '778.869', 41: '787.734', 42: '796.874', 43: '805.973', 44: '815.029', 45: '824.354', 46: '833.324', 47: '842.252', 48: '851.442', 49: '860.585', 50: '869.682', 51: '878.733', 52: '887.736', 53: '896.692', 54: '905.600', 55: '914.755', 56: '923.859', 57: '932.910', 58: '941.910', 59: '950.858', 60: '959.753'}, 'w_a': {0: -1.5142166405558148e-06, 1: -3.4619212096470413e-06, 2: -2.643331013156784e-05, 3: -2.4441716765647918e-05, 4: -2.379761009039184e-05, 5: -3.176746832199805e-05, 6: -3.858866709580749e-05, 7: -3.769806992063878e-05, 8: -5.658785863138352e-05, 9: -4.866626982286441e-05, 10: -5.459659117149863e-05, 11: -6.0217067111630567e-05, 12: -6.540057668825938e-05, 13: -6.278466469710911e-05, 14: -7.789625556142565e-05, 15: -7.859524497763439e-05, 16: -9.1563634463271e-05, 17: -9.712140913823556e-05, 18: -0.00010356943050294344, 19: -0.00011189843567896147, 20: -0.00013661797980895658, 21: -0.00012262706577662808, 22: -0.00014786393796433392, 23: -0.0001571477465364945, 24: -0.00018717178716759806, 25: -0.00019612549976285432, 26: -0.0002295094102546121, 27: -0.00023017289760299834, 28: -0.0002142355281551476, 29: -0.00025014953279365297, 30: -0.00043143553099846897, 31: -0.0003787872586524875, 32: -0.0003434209198766996, 33: -0.00039749182599771687, 34: -0.000928958342270456, 35: -0.0014258780476257495, 36: -0.0003967114561424867, 37: -0.00044309927288764637, 38: -0.004509051588719218, 39: -0.0005675289828034089, 40: -0.00030675453579949983, 41: -0.0005967001241484593, 42: -0.0006918600983584472, 43: -0.0006234495166636982, 44: -0.0018234167372129086, 45: -0.0011923470342845824, 46: -0.0008231083400901478, 47: -0.0005733717515816093, 48: -0.0005966730137090556, 49: -0.0005427725085308022, 50: -0.0004982460011528429, 51: -0.000508057054661033, 52: -0.0005690366544908852, 53: -0.001107916048722521, 54: -0.0007390121694505615, 55: -0.0015921384259921553, 56: -0.0007437725092068227, 57: -0.005395214204283965, 58: -0.0014300458928879704, 59: -0.0003215094698875448, 60: -5.144609849333946e-06}, 'w_b': {0: 0.0006521626300319628, 1: 0.0009722774712041511, 2: 0.004428472597895539, 3: 0.005013496767263008, 4: 0.0056636194438103495, 5: 0.007669118644181956, 6: 0.00926737694323768, 7: 0.009442908604697134, 8: 0.012865581224129471, 9: 0.012197327906268944, 10: 0.013090660661155329, 11: 0.014382604115461241, 12: 0.015649299893640674, 13: 0.015058034079593866, 14: 0.017980737515131618, 15: 0.018500209822086097, 16: 0.020816915595053543, 17: 0.021594514694440133, 18: 0.022786934316177752, 19: 0.02430427101681308, 20: 0.028397235899190413, 21: 0.026703054890571475, 22: 0.03081447184359328, 23: 0.032472908460802245, 24: 0.036364260842226964, 25: 0.03846034667246718, 26: 0.042638615179634035, 27: 0.04408453876419867, 28: 0.04128118172864514, 29: 0.04667899477349791, 30: 0.06454963643342998, 31: 0.06081592763281599, 32: 0.05562711181514715, 33: 0.05972438384655969, 34: 0.10751017329693978, 35: 0.15014365131166185, 36: 0.06183631334867984, 37: 0.06705847878170773, 38: 0.2650653990822888, 39: 0.07611101421448141, 40: 0.04628824662634482, 41: 0.0760776070866623, 42: 0.08549859265275253, 43: 0.07741995997559803, 44: 0.1501791682064369, 45: 0.1192614697172342, 46: 0.0860933489807924, 47: 0.06776309389376099, 48: 0.0699039407534553, 49: 0.062227133577931464, 50: 0.056645254174146675, 51: 0.05400674055841997, 52: 0.06146008217014385, 53: 0.07608604736730841, 54: 0.05662793632280819, 55: 0.0896899473454962, 56: 0.051615563097084094, 57: 0.1797615874740785, 58: 0.07621022127268454, 59: 0.016817962747313345, 60: 0.0003245265089641588}, 'w_c': {0: -0.015408408846749985, 1: -0.02201881393890808, 2: -0.08120458910260929, 3: -0.12309004744467428, 4: -0.1582165947239665, 5: -0.2127232648480718, 6: -0.24779221042162886, 7: -0.2485671491056141, 8: -0.2959569684043132, 9: -0.31983362068906485, 10: -0.27384273232897594, 11: -0.3038954782285524, 12: -0.32246505882642384, 13: -0.2705462451087505, 14: -0.296525842276602, 15: -0.30006711697007593, 16: -0.3131139400034216, 17: -0.24443476940852224, 18: -0.2408371951889283, 19: -0.2332845682086369, 20: -0.2465327189529767, 21: -0.31035056464059896, 22: -0.25269902425471485, 23: -0.2096410336329928, 24: -0.14643925327577972, 25: -0.23355732065092955, 26: -0.14716101143453802, 27: -0.3260153198274481, 28: -0.20282178798401526, 29: -0.13799445717861109, 30: 0.18334321529256656, 31: -0.09662446388228776, 32: -0.13561047200280635, 33: 0.1330141718398243, 34: 0.3032903989790575, 35: -1.0076834184285617, 36: -0.25216015598790725, 37: 0.22982973436033124, 38: 3.2337120288054733, 39: 0.2199407339669856, 40: 0.4494741336220261, 41: 0.9598411145015954, 42: 0.7527831084845322, 43: 0.6827424067521763, 44: 1.3521009041882923, 45: 0.18300565006780128, 46: 0.1479168945824936, 47: 0.2873408558184266, 48: 0.4636169495403654, 49: 0.6692349690719437, 50: 0.6291459706642554, 51: 0.7989956268856089, 52: 0.4302589034223797, 53: 1.4337419322254172, 54: 0.8025603595188665, 55: 0.3874459831158617, 56: 0.1223400686987078, 57: 0.4675454566805294, 58: -0.18915574794961315, 59: 0.010606942209591836, 60: 0.11230554559545562}, 'w_d': {0: 0.2921850842020223, 1: 0.5149818538276124, 2: 1.509748248714712, 3: 2.7787315479713355, 4: 4.260084874350142, 5: 5.802761428738849, 6: 6.7517240955885605, 7: 7.2736155181751725, 8: 7.913062766913932, 9: 9.01163426920535, 10: 8.556545807250648, 11: 8.820194639503468, 12: 9.393360485112256, 13: 8.75173665869611, 14: 9.556082851815232, 15: 9.84819874918599, 16: 10.053688703346289, 17: 9.772349731895835, 18: 9.69848191738501, 19: 9.879948008428403, 20: 11.318635513789838, 21: 9.592066637839329, 22: 10.51253627802026, 23: 10.670886876510444, 24: 10.520214720307491, 25: 10.820389311562767, 26: 11.737964410303102, 27: 11.712555349975279, 28: 10.046656262943662, 29: 10.526821008033393, 30: 9.464581139557684, 31: 11.205248140272076, 32: 9.754903693704806, 33: 7.5975180319550635, 34: 12.743157357501127, 35: 11.257462869438477, 36: 6.246618319710746, 37: 5.049322480320283, 38: 5.543255429172686, 39: 3.9888486152270985, 40: 2.5500445236715112, 41: 3.7757189424763453, 42: 5.321760452758488, 43: 2.9471909702861496, 44: 7.728016198874661, 45: 6.063395764472612, 46: 3.6552360122411818, 47: 1.3169656263746647, 48: 1.7048921806849362, 49: 0.038168951307452785, 50: -0.2428682707490119, 51: -0.7611872414464658, 52: 2.5563185540256432, 53: 0.5756831460894039, 54: -0.0028283815091591433, 55: 1.3166611101781054, 56: 0.5054008565821823, 57: 1.4552659470629161, 58: 2.176197341871184, 59: -0.17464702663830506, 60: -1.6886811942655882}, 'd_a': {0: 4.3066820136802223e-07, 1: -9.996073054043204e-08, 2: -3.2076167285451214e-07, 3: -6.878155305092384e-08, 4: -1.6089959372676795e-07, 5: -1.90132396423241e-07, 6: -1.5328559021169927e-07, 7: -1.3488730074064115e-07, 8: -2.0120872354093984e-07, 9: -1.232347178524909e-07, 10: -1.514205309546725e-07, 11: -2.0156477153540945e-07, 12: -2.0624703060485827e-07, 13: -2.1689300373772125e-07, 14: -2.2983307599531784e-07, 15: -2.3711413924656977e-07, 16: -2.8214775664931003e-07, 17: -3.102057083613334e-07, 18: -3.163125831044023e-07, 19: -3.263265006593517e-07, 20: -3.400980657520238e-07, 21: -3.858374907422985e-07, 22: -4.999082409746404e-07, 23: -4.856922692111328e-07, 24: -5.026377354534607e-07, 25: -5.664404513944444e-07, 26: -6.544151701761442e-07, 27: -6.28909051866656e-07, 28: -7.323862761669924e-07, 29: -7.546656235847859e-07, 30: -1.2260414029382703e-06, 31: -9.360607115418902e-07, 32: -9.351927867391295e-07, 33: -1.2210796929033425e-06, 34: -2.17722958024578e-06, 35: -3.6813322806044724e-06, 36: -1.483886843948064e-06, 37: -1.4690369807129575e-06, 38: -1.4920461951750305e-05, 39: -1.756621178447704e-06, 40: -2.011145070949864e-06, 41: -2.269603052423913e-06, 42: -2.665993046748981e-06, 43: -2.478017339654909e-06, 44: -6.74228740002253e-06, 45: -4.037608675333547e-06, 46: -3.914619462825724e-06, 47: -3.1856360984069247e-06, 48: -3.761709283386862e-06, 49: -3.606229114057444e-06, 50: -3.977378211137961e-06, 51: -4.215593369281809e-06, 52: -5.688363399539785e-06, 53: -1.0157818944725655e-05, 54: -6.370914110661938e-06, 55: -2.2228520351891212e-05, 56: -9.917752643918996e-06, 57: -0.00012905260191222983, 58: 2.9449370614138373e-05, 59: -2.79146934734625e-05, 60: -2.7070810695786195e-07}, 'd_b': {0: 0.0002494653963796221, 1: 0.00023735995154864703, 2: 0.0003258067319138232, 3: 0.00016096298211284608, 4: 0.00012706977320349293, 5: 0.00011931941766521967, 6: 0.00010536151646755196, 7: 9.635705074707524e-05, 8: 0.00011264457601890677, 9: 8.564644311024865e-05, 10: 9.398524373520804e-05, 11: 9.957664957872784e-05, 12: 0.00010014552757826071, 13: 9.677483816483176e-05, 14: 0.00010158787498047372, 15: 0.00010295664888124555, 16: 0.00011365156022123058, 17: 0.00011967765181154917, 18: 0.00011902316382075312, 19: 0.00012005879034196621, 20: 0.00012929010978009252, 21: 0.00013265574359063752, 22: 0.00015833471332983416, 23: 0.00015441938968565472, 24: 0.00015396334863258658, 25: 0.0001673694779332275, 26: 0.00018688458094611102, 27: 0.00018062244827238077, 28: 0.00019899980806050783, 29: 0.0001988291154550109, 30: 0.00027031969354804195, 31: 0.00023490664293795147, 32: 0.0002307840348690816, 33: 0.0002626018301561098, 34: 0.00042879492233442035, 35: 0.0006028733007267957, 36: 0.0003077694848288001, 37: 0.0002945963879322462, 38: 0.001288675101396717, 39: 0.0003236274869643125, 40: 0.00036265056392081554, 41: 0.0003837424650042814, 42: 0.00044161157374818943, 43: 0.00040595545345282254, 44: 0.0008088038118640673, 45: 0.0005998474880293843, 46: 0.0005648727032498108, 47: 0.000476564246492029, 48: 0.0005492839972386428, 49: 0.0005185428187918479, 50: 0.0005575706167831429, 51: 0.0005691511849451724, 52: 0.0007542649715877603, 53: 0.0009965137330879737, 54: 0.0006807492210053213, 55: 0.0016510743408438796, 56: 0.0009678141455640984, 57: 0.004864425206322312, 58: 0.0007701422741003867, 59: 0.0015776503363357974, 60: 1.4670227954150327e-05}, 'd_c': {0: -0.002593023693546423, 1: -0.002806026812008755, 2: -0.0030590951878325737, 3: -0.0012545160056922735, 4: -0.0014345310253195927, 5: -0.0011533334673690685, 6: -0.0003940363850269029, 7: -3.882299443823327e-05, 8: 2.950456293343752e-05, 9: 0.0005689534476638753, 10: 0.0010184344475405323, 11: 0.0009414648159485324, 12: 0.0013364701778187551, 13: 0.0014528844486630032, 14: 0.001871663548107612, 15: 0.001994534581085897, 16: 0.002233351256901666, 17: 0.002643411968567609, 18: 0.002891207769183806, 19: 0.003015881754451321, 20: 0.0031491456019474297, 21: 0.002783766604464082, 22: 0.002653478752737574, 23: 0.00296118483944329, 24: 0.003442142070357618, 25: 0.0032272744217601637, 26: 0.0035532156301418907, 27: 0.0029724786300511018, 28: 0.0033114772296294176, 29: 0.00311524642534598, 30: 0.005615878588275462, 31: 0.004673572512018132, 32: 0.004410236405518359, 33: 0.005594648948810107, 34: 0.007468859245975297, 35: 0.004332633087416201, 36: 0.0042842380535161135, 37: 0.004682345265177804, 38: 0.0202566963406128, 39: 0.005832922235157883, 40: 0.004794284369206734, 41: 0.006485749779736605, 42: 0.0058826780744692065, 43: 0.006495428447663958, 44: 0.010597486273371159, 45: 0.007629471742810926, 46: 0.00733611822667892, 47: 0.0073035873936791685, 48: 0.005432099619974557, 49: 0.00605865247031356, 50: 0.0056840253182746025, 51: 0.007710720125188393, 52: 0.002294274494725385, 53: 0.015362217124017991, 54: 0.017194887179404986, 55: 0.023017936130550733, 56: 0.017605361228763782, 57: 0.046695928822796606, 58: 0.026024883622891726, 59: 0.028874731382305525, 60: 0.02593937246087781}, 'd_d': {0: 0.1629040799151015, 1: 0.1556406933287901, 2: 0.14700018554652186, 3: 0.1278666172045354, 4: 0.125121990774422, 5: 0.11617779622433252, 6: 0.10044530791431562, 7: 0.09767483136369104, 8: 0.08740634529245855, 9: 0.07957079404692519, 10: 0.07705126175402564, 11: 0.07027277587842112, 12: 0.06493927238736495, 13: 0.059099407027930775, 14: 0.05695235546833932, 15: 0.05601064129606553, 16: 0.05481468741499723, 17: 0.05829870151699864, 18: 0.053011509479575336, 19: 0.05531561864148502, 20: 0.0676094510772126, 21: 0.058054498116585376, 22: 0.06457832547296137, 23: 0.0606785017134543, 24: 0.055170484237387366, 25: 0.05997174398449484, 26: 0.07202503261481998, 27: 0.06627633167512179, 28: 0.06497153965862823, 29: 0.06161037154857533, 30: 0.060851631397807526, 31: 0.06974184357596243, 32: 0.07097309654300679, 33: 0.06861754849594122, 34: 0.10808184033004566, 35: 0.1003459973461345, 36: 0.06989361275126063, 37: 0.053427803530784086, 38: 0.07065322461164479, 39: 0.04436370765776963, 40: 0.05571535691089389, 41: 0.05876051257242825, 42: 0.07396647186457257, 43: 0.0588582985322149, 44: 0.10275818102649163, 45: 0.09213888926649176, 46: 0.08255150801586755, 47: 0.0559210125100541, 48: 0.06873625885529866, 49: 0.05387196878226497, 50: 0.05088838473298486, 51: 0.04690764101834495, 52: 0.09987734344958668, 53: 0.07424956886910081, 54: 0.05372335299932848, 55: 0.09874744663140572, 56: 0.07813368089390058, 57: 0.111201377724557, 58: 0.20002049822731024, 59: 0.058354934006199226, 60: -0.17521138714859835}, 'w_a_max': {0: -6.17115903388018e-05, 1: -3.212900593868333e-05, 2: -9.410235326431419e-05, 3: -4.327741680633551e-05, 4: -4.52747686542797e-05, 5: -5.197536163230481e-05, 6: -6.286916186205777e-05, 7: -5.698794244959978e-05, 8: -8.839247572558225e-05, 9: -6.0419204944720874e-05, 10: -8.061531707218425e-05, 11: -8.385641361653382e-05, 12: -9.713746622998064e-05, 13: -8.641256316931188e-05, 14: -0.0001135189633705156, 15: -0.00010782638335199908, 16: -0.0001230389880716218, 17: -0.00013097980805703113, 18: -0.00014055879222768533, 19: -0.0001457097887756971, 20: -0.00018546072289453197, 21: -0.00016592536294758362, 22: -0.0002078702881775136, 23: -0.0002113737047510797, 24: -0.00025522986594173384, 25: -0.00025541737617207485, 26: -0.0003068331957142043, 27: -0.00030731725282907813, 28: -0.0002934022394611265, 29: -0.0003286803230886456, 30: -0.0005800902133989799, 31: -0.00048366302168831017, 32: -0.0004717050974673418, 33: -0.0005190715123951031, 34: -0.0011418354316139608, 35: -0.001834697946690729, 36: -0.0005547993629231861, 37: -0.0005566142821733657, 38: -0.0059102708845953575, 39: -0.0006936150951679849, 40: -0.0006402690491811573, 41: -0.0007744993746656715, 42: -0.0008822534504810877, 43: -0.0007877069340380887, 44: -0.002321783729962557, 45: -0.001398296687284449, 46: -0.0010270064340616811, 47: -0.0007276520842670764, 48: -0.0007781792615959442, 49: -0.0007150139012477677, 50: -0.0006739021051330972, 51: -0.0006781513071616905, 52: -0.0007979142001015733, 53: -0.0014832943729294649, 54: -0.0008640054401356463, 55: -0.002852447932057545, 56: -0.0010376817989660835, 57: -0.008066538094708028, 58: -0.0018218235143745497, 59: -0.0015650769692971862, 60: -1.4978591370128374e-05}, 'w_b_max': {0: 0.010107929245523846, 1: 0.005955289946006466, 2: 0.014115761008761088, 3: 0.009042155413972942, 4: 0.010503223147070974, 5: 0.012403220764454388, 6: 0.014795622062170182, 7: 0.014144152738715115, 8: 0.019778485595036026, 9: 0.0150991564176479, 10: 0.019165710696853064, 11: 0.019916115593869254, 12: 0.02283156409014759, 13: 0.020705266255258512, 14: 0.02570049109110107, 15: 0.02506993250549619, 16: 0.027656009914363282, 17: 0.028702089203411236, 18: 0.030636653091567063, 19: 0.03114394386935512, 20: 0.03775220433349253, 21: 0.035981409649487935, 22: 0.04262608004639478, 23: 0.04320538663516691, 24: 0.04899722421486108, 25: 0.04962440795258246, 26: 0.05625264505101865, 27: 0.05804229248909683, 28: 0.05597904087938059, 29: 0.06040608869297037, 30: 0.08529182544414053, 31: 0.07672541277148474, 32: 0.07499061904494607, 33: 0.07737130875152992, 34: 0.12859026816067104, 35: 0.18868051569000746, 36: 0.08622633627541272, 37: 0.08265965355644751, 38: 0.3391818257767915, 39: 0.0934363476345814, 40: 0.0887558531485522, 41: 0.09677683860899657, 42: 0.10771949923679633, 43: 0.09533193533988607, 44: 0.1846245521349104, 45: 0.1368572536230156, 46: 0.10576184726377083, 47: 0.08479327510606703, 48: 0.0891540261770254, 49: 0.08084592683014784, 50: 0.07457679863603724, 51: 0.06994669047500103, 52: 0.08419781321393949, 53: 0.09648312149426796, 54: 0.06304712281389714, 55: 0.14807548882394395, 56: 0.07071056455446723, 57: 0.25595229720950186, 58: 0.10987309077431703, 59: 0.07842498579921084, 60: 0.00041825882513508745}, 'w_c_max': {0: -0.2533842636291882, 1: -0.11603738583609795, 2: -0.2723794679053237, 3: -0.20927127586979236, 4: -0.28491233587588416, 5: -0.32779419338488164, 6: -0.4101750571485927, 7: -0.3771882542273137, 8: -0.4683395702968614, 9: -0.2933519183463398, 10: -0.4085366306225567, 11: -0.4252115184281802, 12: -0.5026523892813064, 13: -0.33725799900474873, 14: -0.4363147847359883, 15: -0.4133618167141795, 16: -0.40059452527468337, 17: -0.32369733438292897, 18: -0.34589087754757764, 19: -0.2452330861196565, 20: -0.31097158155322197, 21: -0.4117026896047444, 22: -0.3767861456167396, 23: -0.33385548574130897, 24: -0.2623285913636724, 25: -0.2993885427443995, 26: -0.2038576632650811, 27: -0.4338655013485662, 28: -0.2688959055783392, 29: -0.18883548476568773, 30: 0.15321053297595336, 31: 0.0039965152139992645, 32: -0.2314095161690423, 33: 0.25339697054194954, 34: 0.7111038619654173, 35: -0.5391023858190697, 36: -0.1665022414247592, 37: 0.5185450397299451, 38: 4.016906297734431, 39: 0.7072678548150597, 40: 0.6531921312370724, 41: 1.168052805174782, 42: 0.9667358951138744, 43: 0.9738287449663672, 44: 1.940992309106816, 45: 1.1224730925149278, 46: 0.5056910901076224, 47: 0.4964385012040924, 48: 0.6798354062769488, 49: 0.924542880508057, 50: 0.8528629277588927, 51: 1.1747462402438278, 52: 0.5466550157605451, 53: 2.179536490872457, 54: 1.7344940461002456, 55: 0.9057152627911318, 56: 0.5970885253825258, 57: 1.0881905779795604, 58: 0.23326675239360298, 59: 0.0069391226885442145, 60: 0.4067407878712253}, 'w_d_max': {0: 3.454395177425264, 1: 2.1584601709045783, 2: 4.116118635286943, 3: 4.975692061731031, 4: 7.359579317524633, 5: 8.823538080239922, 6: 10.365620505800567, 7: 10.439127599555162, 8: 11.599220079776654, 9: 9.545527777534987, 10: 12.000183737176853, 11: 11.916498424460574, 12: 13.415391412904711, 13: 11.245190048172423, 14: 12.895418876105115, 15: 12.764820276807024, 16: 12.680639013598995, 17: 12.297081366653035, 18: 12.43936076771899, 19: 11.753416074457695, 20: 14.252664567044778, 21: 13.143392354956234, 22: 14.421248446589608, 23: 14.047524373786516, 24: 13.748469952860392, 25: 13.424704615387633, 26: 14.543708360988349, 27: 15.28087583225592, 28: 13.688826254492072, 29: 12.713628673092801, 30: 11.819462829370918, 31: 13.331000929333134, 32: 12.78657128990149, 33: 9.40261580948592, 34: 13.86950102622981, 35: 12.193829146843667, 36: 9.854142403940417, 37: 5.681752017415709, 38: 5.86586250231198, 39: 4.362099280153871, 40: 4.829331103183033, 41: 3.773045224364689, 42: 6.511348459204365, 43: 1.8751728965048893, 44: 7.752456051607263, 45: 5.027936832979173, 46: 3.8890764763815215, 47: 1.4800715629324752, 48: 1.6486990590205457, 49: -0.7648424040562941, 50: -1.2894938864833552, 51: -2.727686255709005, 52: 2.989398835988016, 53: -0.6905742653382845, 54: -1.5351874471910976, 55: 1.928896277942363, 56: 0.04653549103816056, 57: 2.24033419170116, 58: 4.942615490102373, 59: 0.15632515602231795, 60: -4.378238937991869}, 'd_a_max': {0: -3.9500650772279135e-05, 1: -3.140441072857723e-05, 2: -5.352174379953252e-05, 3: -1.9844389592778067e-05, 4: -9.294381829842334e-06, 5: -1.0356142285791163e-05, 6: -1.0215289729701378e-05, 7: -7.396727921677823e-06, 8: -1.0439964679005563e-05, 9: -1.751555490453828e-05, 10: -8.489509033587426e-06, 11: -8.42910555808662e-06, 12: -1.0582080501258467e-05, 13: -7.153562081642372e-06, 14: -1.0165382527834077e-05, 15: -8.203098163207258e-06, 16: -1.127156393789636e-05, 17: -8.862882631300874e-06, 18: -1.0372630561330183e-05, 19: -7.846071472444478e-06, 20: -1.1168736614195197e-05, 21: -8.825715505170862e-06, 22: -9.999981313372326e-06, 23: -1.0348254343980696e-05, 24: -1.3447932927829902e-05, 25: -1.116927491425842e-05, 26: -1.4167585739996286e-05, 27: -1.6394193425144973e-05, 28: -1.5192612720280405e-05, 29: -1.7121985735168984e-05, 30: -2.516346278980173e-05, 31: -2.536071256968693e-05, 32: -2.102426921349193e-05, 33: -2.2290882337062468e-05, 34: -4.892728260705218e-05, 35: -7.717553178972847e-05, 36: -4.513626750179035e-05, 37: -3.105552977546211e-05, 38: -0.000288337821810866, 39: -4.9497001242656156e-05, 40: -2.9557833241190874e-05, 41: -7.567187033460379e-05, 42: -3.759908311389029e-05, 43: -4.314593882867498e-05, 44: -0.0001186836742150571, 45: -5.268796261743532e-05, 46: -8.255448420722305e-05, 47: -6.254405876582488e-05, 48: -8.22464648691004e-05, 49: -6.372700763555322e-05, 50: -6.698908929032559e-05, 51: -4.4628849449909366e-05, 52: -7.791415321542058e-05, 53: -0.00024058029309192104, 54: -0.00017458179176583284, 55: -0.0002400192450897105, 56: -0.00015361234947363915, 57: -0.0022878074434805953, 58: -0.00020073037617368369, 59: -0.000318458509471078, 60: -2.5912435445169768e-05}, 'd_b_max': {0: 0.007075593784190624, 1: 0.005876278550231035, 2: 0.008483746756406445, 3: 0.0041314605786671645, 4: 0.002454784339937629, 5: 0.0026558611993840056, 6: 0.002614281842247094, 7: 0.002029517695870037, 8: 0.002548296799977693, 9: 0.00462482206108422, 10: 0.0021590414578954984, 11: 0.0021962196589435565, 12: 0.002571190247982615, 13: 0.0018500107425102454, 14: 0.0025214310851455245, 15: 0.002050539929805689, 16: 0.002606631372033277, 17: 0.002051752146020683, 18: 0.002454081952079089, 19: 0.001809770751076352, 20: 0.0023545349203741807, 21: 0.0019713524195899327, 22: 0.0021886572142308207, 23: 0.002204939723652404, 24: 0.002668478472750702, 25: 0.002253822039815773, 26: 0.0025568003409807303, 27: 0.003232769155638502, 28: 0.002994974195885553, 29: 0.003297947980773647, 30: 0.003879514642219357, 31: 0.004207211557052469, 32: 0.0034250492553127964, 33: 0.00346468194511718, 34: 0.005625317604090317, 35: 0.008749258575334507, 36: 0.00682639268463701, 37: 0.004842634499929889, 38: 0.016956999195538787, 39: 0.007205336953245577, 40: 0.004176282680174671, 41: 0.010083931366994602, 42: 0.004967491036372267, 43: 0.005532670546033484, 44: 0.009165055450533685, 45: 0.005506059094498729, 46: 0.008420460250128927, 47: 0.007810564361933981, 48: 0.009445043674600095, 49: 0.007722793410701126, 50: 0.0077156215917425705, 51: 0.004939090076825157, 52: 0.008378792638093192, 53: 0.015856168216848033, 54: 0.013399130949469644, 55: 0.013778140759333923, 56: 0.010855232412479262, 57: 0.07542648983828458, 58: 0.018807324184809904, 59: 0.016325722747406728, 60: 0.0030433982319316146}, 'd_c_max': {0: -0.1533263734472244, 1: -0.13545527191163242, 2: -0.1758665630522639, 3: -0.09158756143719328, 4: -0.05392355488998393, 5: -0.06788270609078477, 6: -0.06859342058114706, 7: -0.0373899373963344, 8: -0.045437369855718335, 9: -0.1186826482628428, 10: -0.03591584300268556, 11: -0.019829226292481537, 12: -0.04306510635614904, 13: -0.0117660264160456, 14: -0.04443135842953348, 15: -0.022670296312957267, 16: -0.036550701365603455, 17: 0.0042753641326055076, 18: -0.021353503811963315, 19: 0.013799803876617259, 20: 0.001261018261465931, 21: 0.012622382113340984, 22: 0.006628409331239496, 23: 0.008047716369887836, 24: 0.006306062221883579, 25: 0.02232592406608323, 26: 0.052375205750238775, 27: -0.019272917594487547, 28: -0.008364029061251277, 29: 0.00974366964291207, 30: 0.021924164027015646, 31: -0.006014164465698003, 32: 0.029639006599769764, 33: 0.0406946425264427, 34: 0.062276247500625735, 35: 0.0018491497586125294, 36: 0.02254892872159855, 37: 0.00044505659906156033, 38: 0.18239383037726212, 39: 0.022043620872537303, 40: 0.04535155746229815, 41: 0.014087643230687985, 42: 0.048303506473361686, 43: 0.12125900120953555, 44: 0.15279031203185153, 45: 0.1112381873398972, 46: 0.04017844504524904, 47: -0.027815552355903567, 48: -0.04948514978416378, 49: -0.014898619546841003, 50: 0.0014644692976288226, 51: 0.11310529043892803, 52: 0.016898748757980815, 53: 0.1776288237632146, 54: 0.08692345654185556, 55: 0.24020660472730865, 56: 0.1472910634567181, 57: 0.20522181838357126, 58: 0.19334932119642007, 59: 0.21058725540691256, 60: 0.17310611086295669}, 'd_d_max': {0: 2.793161638131741, 1: 2.5615748533393843, 2: 2.7951803798263244, 3: 2.146719671653811, 4: 1.8051569524305044, 5: 1.9979534316190104, 6: 2.0218095167914, 7: 1.5790445079380666, 8: 1.5903718037359489, 9: 3.712950734968184, 10: 1.4917501896503649, 11: 1.3327661141738387, 12: 1.5952413196170616, 13: 1.039761231891585, 14: 1.6351743535556307, 15: 1.147106526299384, 16: 1.3625047051154164, 17: 0.8571859119210441, 18: 1.25950873956485, 19: 0.8042279480620649, 20: 0.9488071646982345, 21: 0.6467810794613935, 22: 0.8282583259365159, 23: 0.8598916269228621, 24: 0.8183479088961394, 25: 0.5528085448432776, 26: 0.5105277454204503, 27: 1.2136336910335306, 28: 0.9970697321340954, 29: 0.9227241770400356, 30: 0.7775473581794703, 31: 1.2268657140233679, 32: 0.8130559380119503, 33: 0.7388860360135193, 34: 1.0144825689989303, 35: 1.214731781976176, 36: 1.3695065603050047, 37: 1.0576809031200658, 38: 0.7215397384499359, 39: 1.2350361441552142, 40: 0.5066038061975364, 41: 1.7215657775499074, 42: 0.7610939328378828, 43: 0.4573853948224714, 44: 0.7848849132698033, 45: 0.5216364465355972, 46: 1.0229249482415814, 47: 1.3477836161686498, 48: 1.3417048076172309, 49: 1.1665649821655597, 50: 0.8832019173166773, 51: 0.18027864749175027, 52: 0.9350831557508936, 53: 0.8620254122241526, 54: 0.9168024176268588, 55: 0.7507164072241992, 56: 0.7195134656703974, 57: 1.5886874346114883, 58: 2.1062893292885674, 59: 0.6692143199904711, 60: -0.7089789062106597}, 'irr_cutoff': {0: 89.66787878787878, 1: 18.46545454545455, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.0, 30: 0.0, 31: 0.0, 32: 0.0, 33: 0.0, 34: 0.0, 35: 0.0, 36: 0.0, 37: 0.0, 38: 0.0, 39: 0.0, 40: 0.0, 41: 0.0, 42: 0.0, 43: 0.0, 44: 0.0, 45: 0.0, 46: 0.0, 47: 0.0, 48: 0.0, 49: 1.6985858585858586, 50: 1.6535353535353534, 51: 2.3509090909090906, 52: 0.0, 53: 0.5154545454545455, 54: 1.192929292929293, 55: 0.0, 56: 1.0155555555555555, 57: 0.0, 58: 0.0, 59: 5.013333333333333, 60: 19.44363636363636}}

class PlotTSS:
    def __init__(self,fp_dir):
        self.fp_dir = fp_dir
        self.fp_list = glob(join(fp_dir,'*line*.csv'))
        fp_dict = {}
        for f in self.fp_list:
            line_number = int(os.path.splitext(os.path.basename(f))[0].split('line')[1])
            fp_dict[line_number] = f
        self.fp_dict = fp_dict
        self.conc_column_idx = 2
        self.band_columns = [5,66]
        self.crop_bands = [2,-3]

    def multiline(self, xs, ys, c, ax=None, **kwargs):
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
    
    def get_df(self, fp):
        df = pd.read_csv(fp)
        columns = [self.conc_column_idx] + list(range(self.band_columns[0],self.band_columns[1]))
        df = df.iloc[:,columns]
        df.columns = ['Concentration'] + bands_wavelengths()
        df = df.set_index('Concentration')
        return df
    
    def plot_TSS(self,df,norm,ax=None):
        wavelength = df.columns.tolist()
        wavelength = wavelength[self.crop_bands[0]:self.crop_bands[1]]
        concentration = df.index.tolist()
        n_lines = len(concentration)
        y_array = df.reset_index().iloc[:,-61:].values #reflectance
        y_array = y_array[:,self.crop_bands[0]:self.crop_bands[1]]
        x_array = np.array([wavelength for i in range(n_lines)])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        ax.set_ylim(0,40)
        ax.set_xlim(450,950)
        
        # ax.set_title('Reflectance of water quality variable')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance (%)')
        plt.rcParams["font.size"] = "8"

        # norm=plt.Normalize(5,50)
        lc = self.multiline(x_array, y_array, concentration, ax, cmap='BrBG_r',norm=norm,lw=1)
        axcb = plt.colorbar(lc, ax=ax)
        axcb.set_label('Turbidity (FNU)',fontsize=12)
        return
    
    def get_stats(self,y_hat,y):
        rmse = (np.sum((y_hat - y)**2)/len(y_hat))**(1/2)
        mape = np.sum(np.abs((y - y_hat)/y))/len(y_hat)
        r2 = r2_score(y,y_hat)
        return r2, rmse, mape
    
    def nechad_2009(self, x, a, b):
        return a*x/(1-x/b)
    
    def linReg_params(self,x,a,b):
        return a*x + b
    
    def curve_fit(self,X,y, ax0 = None, ax1 = None, bounds=([0,0],[50,10])):
        """ 
        :param X (np.array) is the reflectance
        :param y (np.array) is the concentration
        """
        # fit curve
        # convert X into reflectance by dividing by 100
        X = X/100 # ranges from 0 to 1
        popt, _ = curve_fit(self.nechad_2009, X, y,bounds=bounds)
        print(f'curve fit params: {popt}')
        y_hat = self.nechad_2009(X,*popt) #y_pred
        r2, rmse, mape = self.get_stats(y_hat = y_hat, y = y)
        fontsize = 12
        if ax1 is not None:
            ax1.plot(y_hat,y,'ko',alpha=0.5)
            ax1.set_xlabel('Predicted Turbidity (FNU)', fontsize=fontsize)
            ax1.set_ylabel('Observed Turbidity (FNU)', fontsize=fontsize)
            one_to_one_line = np.linspace(y.min(), y.max(),100)
            ax1.plot(one_to_one_line,one_to_one_line, 'g--',alpha=0.5,linewidth=3)
            # add text
            bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
            
            stats = (f'$RMSE$ = {rmse:.3f}\n'
            f'$MAPE$ = {mape:.3f}\n'
            f'$R^2$ = {r2:.3f}')
            ax1.set_title(f'N = {len(X)}')
            ax1.text(0.95,0.07,stats, fontsize=12, bbox=bbox,
                transform=ax1.transAxes, horizontalalignment='right')
            
        # plot turbidity vs reflectance
        if ax0 is not None:
            ax0.plot(X*100,y,'ko',alpha=0.5)
            x = np.linspace(X.min(),X.max(),100)
            y_fitted = self.nechad_2009(x,*popt)
            ax0.plot(x*100,y_fitted, 'b--',alpha=0.5,linewidth=3)
            ax0.set_xlabel('Reflectance (%)', fontsize=fontsize)
            ax0.set_ylabel('Turbidity (FNU)', fontsize=fontsize)

        # plt.rcParams.update({'font.size': 10})
        
        return popt
    
    def lin_reg(self,X,y, ax0 = None, ax1 = None, log=False):
        if log is False:
            reg = LinearRegression().fit(X,y) #regression model
            y_pred = reg.predict(X)
        else:
            reg = LinearRegression().fit(np.log(X),np.log(y)) #regression model
            y_pred = reg.predict(np.log(X))
            y_pred = np.exp(y_pred)

        coef = reg.coef_
        intercept = reg.intercept_
        r2, rmse, mape = self.get_stats(y_hat = y_pred, y = y)

        fontsize=10
        # plot observed vs predicted
        if ax1 is not None:
            ax1.plot(y_pred,y,'ko',alpha=0.5)
            ax1.set_xlabel('Predicted Turbidity (FNU)', fontsize=fontsize)
            ax1.set_ylabel('Observed Turbidity (FNU)', fontsize=fontsize)
            one_to_one_line = np.linspace(y.min(), y.max(),100)
            ax1.plot(one_to_one_line,one_to_one_line, 'g--',alpha=0.5,linewidth=3)
            # add text
            bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
            if log is False:
                stats = (f'$RMSE$ = {rmse:.3f}\n'
                f'$MAPE$ = {mape:.3f}\n'
                f'$R^2$ = {r2:.3f}\n'
                f'$y$ = {coef[0][0]:.3f}$x$ + {intercept[0]:.3f}')
            else:
                stats = (f'$RMSE$ = {rmse:.3f}\n'
                f'$MAPE$ = {mape:.3f}\n'
                f'$R^2$ = {r2:.3f}\n'
                f'$log(y)$ = {coef[0][0]:.3f}$log(x)$')
            ax1.text(0.95,0.07,stats, fontsize=9, bbox=bbox,
                transform=ax1.transAxes, horizontalalignment='right')
            
        # plot turbidity vs reflectance
        if ax0 is not None:
            ax0.plot(X,y,'ko',alpha=0.5)
            x = np.linspace(X.min(),X.max(),100)
            y_fitted = reg.predict(x)
            ax0.plot(x,y_fitted, 'b--',alpha=0.5,linewidth=3)
            ax0.set_xlabel('Reflectance (%)', fontsize=fontsize)
            ax0.set_ylabel('Turbidity (FNU)', fontsize=fontsize)

        return coef[0][0],intercept[0]
    
    def plot_predicted(self, df,band_number,method='linear',title="",bounds=([0,0],[50,10]),save_fp=None):
        """
        :param method (str): 'linear' for linear regression, 'curve' for semi-empirical using Nechad 2009
        df must not have NAs
        """
        columnName = df.columns[band_number]
        columnData = df.iloc[:,band_number]

        fig, axes = plt.subplots(1,2,figsize=(7,4))
        ax0 = axes[0]
        ax1 = axes[1]
        
        if method == 'linear':
            params = self.lin_reg(X = columnData.values.reshape(-1, 1),y=df.index.to_numpy().reshape(-1, 1), 
                                  ax0=ax0, ax1=ax1, log = False)
        else:
            params = self.curve_fit(X = columnData.values,y=df.index.to_numpy(), 
                                    ax0=ax0, ax1 = ax1, bounds=bounds)

        handles = [Line2D([0], [0], linewidth=3,c='b',ls='--'),
                   Line2D([0], [0], linewidth=3,c='g',ls='--')]
        labels = ['model prediction','1:1 line']
        
        titleFontSize = 12
        ax0.set_title(f'{title}',fontdict={'fontsize': titleFontSize, 'fontweight': 'medium'})
        ax1.set_title(f'N = {len(df.index)} ({columnName} nm)',fontdict={'fontsize': titleFontSize, 'fontweight': 'medium'})

        for ax in axes.flatten():
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(14)

        legendFontSize = 12
        fig.legend(handles=handles,labels=labels,
                   loc='upper center', bbox_to_anchor=(0.5, 0),
                   ncol=2, prop={'size': legendFontSize})
        # plt.rcParams.update({'font.size': 22})
        plt.tight_layout()
        plt.show()

        if save_fp is not None:
            fig.savefig(f'{save_fp}.png',bbox_inches='tight')

        
        return params
    
    def plot_predicted_bands(self, df,line_number,method='linear',save_fp=None):
        """
        :param method (str): 'linear' for linear regression, 'curve' for semi-empirical using Nechad 2009
        df must not have NAs
        """
        n_bands = len(df.columns)
        ncols = 5
        nrows = ceil(n_bands/ncols)
        fig, axes = plt.subplots(nrows,ncols, figsize=(ncols*3,nrows*3))
        for i,((columnName, columnData), ax) in enumerate(zip(df.iteritems(), axes.flatten())):
            ax.set_title(f'{columnName} nm')
            if method == 'linear':
                self.lin_reg(X = columnData.values.reshape(-1, 1),y=df.index.to_numpy().reshape(-1, 1), ax=ax, log = False)
            else:
                self.curve_fit(X = columnData.values,y=df.index.to_numpy(), ax=ax)
        # PT.curve_fit()
        n_del = nrows*ncols - n_bands
        for ax in axes.flatten()[-n_del:]:
            plt.delaxes(ax=ax)

        handles = [Line2D([0], [0], linewidth=3,c='g',ls='--')]
        labels = ['1:1 line']
        fig.legend(handles=handles,labels=labels,
                   loc='upper center', bbox_to_anchor=(0.75, 0.05),
                   ncol=2, prop={'size': 12})
        plt.tight_layout()
        fig.suptitle(f'Line {line_number} (N = {len(df.index)})')
        plt.show()
        if save_fp is not None:
            fig.savefig(f'{save_fp}.png')
        return

    def plot_grid_TSS(self,norm, ncols = 3, save_fp=None):
        """ 
        :param norm (plt.Normalize): that maps the y-variable to the colormap 
        e.g. plt.Normalize(5,50) 5 - 50 FNU map to colorbar
        """
        n_lines = len(self.fp_list)
        nrows = ceil(n_lines/ncols)
        fig, axes = plt.subplots(nrows,ncols, figsize = (ncols*3,nrows*2.5),sharey=True)
        for i, ax in zip(range(n_lines),axes.flatten()):
            df = self.get_df(self.fp_dict[i])
            ax.set_title(f'Line {i}')
            if (len(df.index) == 0):
                plt.delaxes(ax = ax)
                continue
            self.plot_TSS(df,norm,ax=ax)
        
        n_del = ncols*nrows - n_lines
        for ax in axes.flatten()[-n_del:]:
            plt.delaxes(ax = ax)
        
        plt.tight_layout()
        # plt.suptitle('Reflectance of individual image lines')
        plt.show()
        if save_fp is not None:
            fig.savefig(f'{save_fp}.png')
        return
    
    def plot_compiled_TSS(self,norm,title="",ax=None,save_fp=None,font_size=12):
        n_lines = len(self.fp_list)
        df_concat = pd.concat([self.get_df(self.fp_dict[i]) for i in range(n_lines) if (i > 6 and i< 15)])
        if ax is None:
            fig, ax = plt.subplots()
        n = len(df_concat.index)
        ax.set_title(f'{title} (N = {n})')

        self.plot_TSS(df_concat,norm,ax=ax)

        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_size)

        if ax is None:
            plt.show()
        if save_fp is not None:
            parent_dir = os.path.dirname(save_fp)
            fn = os.path.basename(save_fp)
            store_dir = join(parent_dir,'exported_reflectance')
            if not os.path.exists(store_dir):
                os.mkdir(store_dir)
            df_concat.to_csv(join(store_dir,f'{fn}.csv'))
            fig.savefig(f'{save_fp}.png')
        return

def get_stitched_reflectance(StitchClass,curve_fitting_correction=True):
    """ 
    StitchClass method for a single class
    correction wrt to downwelling irradiance images
    outputs band_list of reflectance naive stitched img in the form of a dictionary. keys are band number, values are corrected reflectance array
    Must perform destriping before converting to reflectances since units are different.
    Destriping is performed on raw DN
    Must perform radiometric correction for each individual frame before stitching them up because irradiance for image frames is different
    >>>get_stitched_reflectance()
    # for line_number in range(line_start,line_stop+1):
    # indexes_list = corrected_indices # overwrite
    # start_i,end_i = indexes_list[line_number]
    # test_stitch_class = StitchHyperspectral(fp_store,config_file['-PREFIX-'],config_file['-IMAGE_FOLDER_FILEPATH-'],config_file['-SPECTRO_FILEPATH-'],\
    #                     int(config_file['-HEIGHT-']),line_number,start_i,end_i,\
    #                     test_gps_index, unique_gps_df, destriping = config_file['-NOISE_CHECKBOX-'],\
    #                     reverse=reverse_boolean_list[line_number])
    # band_list = get_stitched_reflectance(test_stitch_class,curve_fitting_correction=True)
    # band_660 = band_list[27]
    # band_660 = band_660.reshape(band_660.shape[0],band_660.shape[1],1)
    # sgc = sugar.SUGAR(band_660,glint_mask_method="cdf")
    # sgc_reflectance = sgc.get_corrected_bands()[0]

    # with open(os.path.join(fp_store,f'{prefix}_line{line_number}.pickle'), 'wb') as handle:
    #     pickle.dump(sgc_reflectance, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(os.path.join(fp_store,f'{prefix}_line{line_number}.pickle'), 'rb') as handle:
    #     reflectance = pickle.load(handle)
    # predicted = nechad(reflectance/100, 87.74878821,  0.23236878)
    # predicted_image = np.clip(predicted,8.41,41.56).astype(np.uint16) # clip to the minimum and maximum df concentration and 
    # img = Image.fromarray(predicted_image)
    # img_fp = join(test_stitch_class.fp_store,test_stitch_class.prediction_directory,test_stitch_class.predicted_fp)
    # img.save(img_fp)
    # gti = GeotransformImage(test_stitch_class)
    # gti.geotransform_image()
    """
    
    spec_df = StitchClass.get_spec_calib()
    calib_info = StitchClass.calibration_attributes()
    # date = calib_info['White']['datetime']
    raw_info = StitchClass.raw_attributes()

    exp_white = calib_info['White']['exp_time']
    exp_dark = calib_info['Dark']['exp_time']

    if curve_fitting_correction is False:
        print("Performing correction...")
        print("Linear radiometric calibration...")
    else:
        calibration_curve_df = StitchClass.get_calibration_curve()
        print("Performing correction using calibration curve...")
        print("Cubic function calibration...")

    
    white_fp = join(StitchClass.image_folder_filepath,"WhiteRef")
    fp_list_white = [join(white_fp,i) for i in listdir(white_fp)]
    hyperspectral_img_list_white = [raw_to_hyperspectral_img(f) for f in fp_list_white]
    hyperspectral_white = np.mean(hyperspectral_img_list_white,axis=0)
    hyperspectral_white_array = np.mean(hyperspectral_white,axis=1)

    def destriping_img(img,hyperspectral_white_array,band):
        """
        destriping array.shape = (61,1024)
        """
        adjust_DN = lambda x,max_DN: max_DN/x
        avg_DN = hyperspectral_white_array[:,band]
        max_DN = np.max(avg_DN)
        avg_DN = np.where(avg_DN <=0,1,avg_DN)
        corrected_DN = adjust_DN(avg_DN,max_DN)

        nrows,ncols = img.shape
        adjust_DN_rgb = np.transpose(corrected_DN) #1D array
        repeated_DN_rgb = np.repeat(adjust_DN_rgb[:,np.newaxis],ncols,axis=1)
        destriped_img = repeated_DN_rgb*img
        destriped_img = np.where(destriped_img>255,255,destriped_img)
        return destriped_img.astype(np.uint8)
    #calibration curve fn
    def cubic_fn(x,a,b,c,d):
        """
        used to interpolate corrected DN given a radiance value
        where x is the radiance
        a,b,c,d = parameters of cubic curve fitting
        returns the corrected DN i.e. DN/exp_time
        """
        y = a*x**3 + b*x**2 + c*x + d
        return y if y >0 else 0
    
    def reflectance_eqn(corrected_raw,corrected_white,corrected_dark):
        """
        corrected means DN/exp_time
        """
        if corrected_raw > corrected_white:
            return float(StitchClass.Reflectance_White_Ref)
        elif corrected_raw < corrected_dark:
            return 0.0
        else:
            if abs(corrected_white - corrected_dark) < 0.001:
                return 0.0
            else:
                return (corrected_raw - corrected_dark)/(corrected_white - corrected_dark)*StitchClass.Reflectance_White_Ref

    vectorised_reflectance_eqn = np.vectorize(reflectance_eqn)

    band_list = {} #store all the separate bands
    # stitch class for a single band
    for i in range(27,28): #61 bands are evenly distributed across rows
        reflectance_image_array = [] #corrected & transformed to reflectance units for a wavelength
        for calib_name,calib_attr in calib_info.items():
            f = calib_attr['raw_file']
            reshaped_raw = f.reshape(1024,1280)
            row_start = i*StitchClass.band_width
            row_end = i*StitchClass.band_width + StitchClass.band_width
            band_array = reshaped_raw[:,row_start:row_end]
            calib_info[calib_name][i] = np.mean(band_array) #mean DN and assign it the band number
            # calib_info[calib_name][i] = np.max(band_array) #for both dark and white ref
            
        for j,(_,raw_attr) in enumerate(raw_info.items()): #iterate across images #enumerate across dictionary  
            f = raw_attr['raw_file']
            reshaped_raw = f.reshape(1024,1280)
            row_start = i*StitchClass.band_width
            row_end = i*StitchClass.band_width + StitchClass.band_width
            band_array = reshaped_raw[:,row_start:row_end]
            if StitchClass.destriping is True:
                band_array = destriping_img(band_array, hyperspectral_white_array,i)
            #perform correction on the line image (band_array)
            exp_raw = raw_attr['exp_time']
            new_radiance = spec_df.iloc[i,j] #indexing band & image
            old_radiance = spec_df.iloc[i,-2] #indexing band & White column
            White = calib_info['White'][i] #mean DN of white ref
            Dark = calib_info['Dark'][i] #mean DN of dark red
            # new_White = new_radiance/old_radiance * White
            # reflectance_formula = lambda DN,White_DN: (DN/exp_raw)/(White_DN/exp_white)*StitchClass.Reflectance_White_Ref if ((White_DN - Dark) < 0.5) else StitchClass.Reflectance_White_Ref*((DN/exp_raw)-(Dark/exp_dark))/((White_DN/exp_white)-(Dark/exp_dark))
            # corrected_band_array = reflectance_formula(band_array,new_White) #np.array(map(reflectance_formula, band_array))
            if curve_fitting_correction is False or calibration_curve_df is None:
                # print("Linear radiometric calibration...")
                new_White = new_radiance/old_radiance * White
                corrected_Raw = band_array/exp_raw
                corrected_White = new_White/exp_white
                corrected_Dark = Dark/exp_dark
                corrected_band_array = vectorised_reflectance_eqn(corrected_Raw,corrected_White,corrected_Dark)
                # reflectance_formula = lambda DN,White_DN: (DN/exp_raw)/(White_DN/exp_white)*StitchClass.Reflectance_White_Ref if ((White_DN - Dark) < 0.5) else StitchClass.Reflectance_White_Ref*((DN/exp_raw)-(Dark/exp_dark))/((White_DN/exp_white)-(Dark/exp_dark))
                # corrected_band_array = reflectance_formula(band_array,new_White) #np.array(map(reflectance_formula, band_array))
                # corrected_band_array = reflectance_formula(band_array,new_White)
            else:
                # print("Cubic function calibration...")
                w_a, w_b, w_c, w_d, d_a, d_b, d_c, d_d, w_a_max, w_b_max, w_c_max, w_d_max, d_a_max, d_b_max, d_c_max, d_d_max, irr_cutoff = calibration_curve_df.iloc[i,1:] #subset at the band
                corrected_White = cubic_fn(new_radiance,w_a, w_b, w_c, w_d)
                corrected_Dark = cubic_fn(new_radiance,d_a, d_b, d_c, d_d)
                # corrected_White = cubic_fn(new_radiance,w_a_max, w_b_max, w_c_max, w_d_max)
                # corrected_Dark = cubic_fn(new_radiance,d_a_max, d_b_max, d_c_max, d_d_max)
                # corrected_Raw = band_array/exp_raw
                corrected_Raw = band_array/exp_raw
                corrected_band_array = vectorised_reflectance_eqn(corrected_Raw,corrected_White,corrected_Dark)
                # corrected_band_array = vectorised_reflectance_eqn(corrected_Raw,corrected_White,corrected_Dark,new_radiance,irr_cutoff)
            reflectance_image_array.append(corrected_band_array)

        # overlap_ratios_per_line = StitchClass.get_overlap_ratios_per_line(len(reflectance_image_array))
        # band_list[i] = StitchClass.trimm_image(reflectance_image_array,overlap_ratios_per_line)
        band_list[i] = StitchClass.trimm_image(reflectance_image_array)

    # save the rgb reflectance img
    # self.get_rgb_reflectance_img(band_list)
    return band_list