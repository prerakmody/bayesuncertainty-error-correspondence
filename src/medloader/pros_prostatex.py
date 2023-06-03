# Import private libraries
# 

# Import public libraries
import os
import pdb
import math
import tqdm
import random
import urllib
import zipfile
import nibabel # pip install nibabel
import pydicom # pip install pydicom
import skimage
import traceback
import subprocess
import numpy as np
import urllib.request
import skimage.transform
from pathlib import Path
import matplotlib.pyplot as plt

# Libraries for smoothing contours
import cv2
from scipy import ndimage
from scipy.interpolate import splprep, splev

# Deep Learning Libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" # to avoid large "Kernel Launch Time"
# import tensorflow as tf # v2.9.1 (cuda-11.2.2, cudnn=8.1.0.77)
# import tensorflow_addons as tfa # v0.17.1

import tensorflow as tf # v2.10.0 (cuda-11.2.2, cudnn=8.1.0.77)
import tensorflow_addons as tfa # v0.18.0
import tensorflow_probability as tfp # v0.18.0

#############################################################
#                       AUGMENTATIONS                       #
#############################################################

class MinMaxNormalizer:

    def __init__(self, min=0., max=200.):

        self.min  = 0
        self.max  = 200
    
    @tf.function
    def execute(self, x, y, meta1, meta2): 

        x = (x - self.min) / (self.max - self.min) 

        return x, y, meta1, meta2

class Translate:

    def __init__(self, label_map, translations=[40,40], prob=0.2):

        self.translations = translations
        self.prob         = prob
        self.label_ids    = label_map.values()
        self.class_count  = len(label_map)
        self.name         = 'Translate' 

    @tf.function
    def execute(self,x,y,meta1,meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,class]

        Ref
        ---
        - tfa.image.translate(image): image= (num_images, num_rows, num_columns, num_channels)
        """
        
        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            
            translate_x = tf.random.uniform([], minval=-self.translations[0], maxval=self.translations[0], dtype=tf.dtypes.int32)
            translate_y = tf.random.uniform([], minval=-self.translations[1], maxval=self.translations[1], dtype=tf.dtypes.int32)

            x = tf.expand_dims(x[:,:,:,0], axis=0) # [1,H,W,D]
            x = tf.transpose(x) # [D,W,H,1] 
            x = tfa.image.translate(x, [translate_x, translate_y], interpolation='bilinear') # [D,W,H,1]; x=(num_images, num_rows, num_columns, num_channels)
            x = tf.transpose(x) # [1,H,W,D]
            x = tf.expand_dims(x[0], axis=-1) # [H,W,D,1]
            
            y = tf.concat([
                    tf.expand_dims(
                        tf.transpose( # [1,H,W,D]
                            tfa.image.translate( # [D,W,H,1]
                                tf.transpose( # [D,W,H,1]
                                    tf.expand_dims(y[:,:,:,class_id], axis=0) # [1,H,W,D]
                                )
                                , [translate_x, translate_y], interpolation='bilinear'
                            )
                        )[0] # [H,W,D]
                        , axis=-1 # [H,W,D,1]
                    ) for class_id in range(self.class_count)
                ], axis=-1) # [H,W,D,10]

        return x,y,meta1,meta2

class Rotate3D:

    def __init__(self, label_map, prob=0.2, angle_degrees=5):
        
        self.name = 'Rotate3DSmallZ'
        
        self.label_ids = label_map.values()
        self.class_count = len(label_map)

        self.prob = prob
        self.angle_degrees = angle_degrees
    
    @tf.function
    def execute(self, x, y, meta1, meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,C]
         - Ref: https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate

        """

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:

            angle_radians = tf.random.uniform([], minval=math.radians(-self.angle_degrees), maxval=math.radians(self.angle_degrees)
                                    , dtype=tf.dtypes.float32)
                
            x = tf.expand_dims(x[:,:,:,0], axis=0) # [1,H,W,D]
            x = tf.transpose(x) # [D,W,H,1] 
            x = tfa.image.rotate(x, angle_radians, interpolation='bilinear') # [D,W,H,1]; x=(num_images, num_rows, num_columns, num_channels)
            x = tf.transpose(x) # [1,H,W,D]
            x = tf.expand_dims(x[0], axis=-1) # [H,W,D,1]
            
            y = tf.concat([
                    tf.expand_dims(
                        tf.transpose( # [1,H,W,D]
                            tfa.image.rotate( # [D,W,H,1]
                                tf.transpose( # [D,W,H,1]
                                    tf.expand_dims(y[:,:,:,class_id], axis=0) # [1,H,W,D]
                                )
                                , angle_radians, interpolation='nearest'
                            )
                        )[0] # [H,W,D]
                        , axis=-1 # [H,W,D,1]
                    ) for class_id in range(self.class_count)
                ], axis=-1) # [H,W,D,C]

            
        else:
            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
        
        return x, y, meta1, meta2

class Noise:

    def __init__(self, x_shape, mean=0.0, std=0.01, prob=0.1):
        
        self.mean    = mean
        self.std     = std 
        self.prob    = prob
        self.x_shape = x_shape
        self.name    = 'Noise'
    
    @tf.function
    def execute(self,x,y,meta1,meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,class]
        """

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            
            x = x + tf.random.normal(self.x_shape, self.mean, self.std)
            x = tf.clip_by_value(x, 0., 1.0)

        return x,y,meta1,meta2

class Mask:
    
    def __init__(self, label_map, img_size, mask_size=[40,40], prob=0.2):

        self.mask_size = mask_size
        self.img_size  = img_size # (H,W,D,1)

        self.prob         = prob
        self.label_ids    = label_map.values()
        self.class_count  = len(label_map)
        self.name         = 'Mask' 

    @tf.function
    def execute(self,x,y,meta1,meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,class]

        Ref
        ---
        - tfa.image.cutout(image): image= (num_images, num_rows, num_columns, num_channels)
        """
        
        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            
            mask_x = tf.random.uniform([], minval=5, maxval=self.mask_size[0], dtype=tf.dtypes.int32)
            mask_y = tf.random.uniform([], minval=5, maxval=self.mask_size[1], dtype=tf.dtypes.int32)
            if not mask_x % 2 == 0: mask_x = mask_x + 1
            if not mask_y % 2 == 0: mask_y = mask_y + 1
            offset_x = tf.random.uniform([], minval=0, maxval=self.img_size[0] - mask_x - 3, dtype=tf.dtypes.int32)
            offset_y = tf.random.uniform([], minval=0, maxval=self.img_size[1] - mask_x - 3, dtype=tf.dtypes.int32)

            x = tf.expand_dims(x[:,:,:,0], axis=0) # [1,H,W,D]
            x = tf.transpose(x) # [D,W,H,1] 
            x = tfa.image.cutout(x, [mask_x, mask_y], [offset_x, offset_y]) # [D,W,H,1]; x=(num_images, num_rows, num_columns, num_channels)
            x = tf.transpose(x) # [1,H,W,D]
            x = tf.expand_dims(x[0], axis=-1) # [H,W,D,1]
            
            # Note: No change needs to be made for the segmentation
            # y = tf.concat([
            #         tf.expand_dims(
            #             tf.transpose( # [1,H,W,D]
            #                 tfa.image.cutout( # [D,W,H,1]
            #                     tf.transpose( # [D,W,H,1]
            #                         tf.expand_dims(y[:,:,:,class_id], axis=0) # [1,H,W,D]
            #                     )
            #                     , [mask_x, mask_y], [offset_x, offset_y]
            #                 )
            #             )[0] # [H,W,D]
            #             , axis=-1 # [H,W,D,1]
            #         ) for class_id in range(self.class_count)
            #     ], axis=-1) # [H,W,D,10]

        return x,y,meta1,meta2

class Equalize:

    def __init__(self, x_shape=(200,200,28,1), bins_range=[20,60], prob=0.1): # not too much of an effect
        
        self.x_shape = x_shape
        self.x_len   = tf.cast(self.x_shape[0]*self.x_shape[1]*self.x_shape[2], dtype=tf.float32) # (200 * 200 * 28) = 1,120,000

        self.bins_range = bins_range
        self.prob       = prob
        
        self.name    = 'equalize'
    
    @tf.function
    def execute(self,x,y,meta1,meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,class]
        """

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            
            random_bins = tf.cast(tf.random.uniform([], minval=self.bins_range[0], maxval=self.bins_range[1], dtype=tf.dtypes.int32), dtype=tf.float32)
            
            if 0:
                x = tf.expand_dims(x[:,:,:,0], axis=0) # [1,H,W,D]
                x = tf.transpose(x) # [D,W,H,1] 
                x = tfa.image.equalize(x, bins=random_bins) # [D,W,H,1]; x=(num_images, num_rows, num_columns, num_channels)
                x = tf.transpose(x) # [1,H,W,D]
                x = tf.expand_dims(x[0], axis=-1) # [H,W,D,1]
            else:
                
                # Step 1 - Get idxs that bin an equal amount of pixels in between them (i.e. equal-mass binning) --> tests reveal that does not happen
                # Ref: https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py#L89
                x_flat = tf.reshape(x, [-1])
                x_sort = tf.sort(x_flat)
                idxs = tf.concat([[0.001], tf.range(random_bins)/random_bins + (1/2/random_bins), [0.999]], axis=0)
                idxs = tf.cast(self.x_len*idxs + 0.5, tf.int32)
                
                # Step 2 - Do histogram normalization
                xs_tf = tf.sort(tf.unique(tf.gather(x_sort, idxs))[0])
                ys_tf = tf.linspace(tf.cast(0., dtype=tf.float32), 1., len(xs_tf))
                x = tfp.math.batch_interp_rectilinear_nd_grid(x=tf.expand_dims(x_flat, axis=-1), x_grid_points=(xs_tf, ), y_ref=ys_tf, axis=0)
                x = tf.reshape(x, self.x_shape)
                x = tf.clip_by_value(x, 0., 1.)

        return x,y,meta1,meta2

class ImageQuality:
    """
     - Ref: # https://www.tensorflow.org/api_docs/python/tf/image/random_brightness
    """

    def __init__(self, prob=0.3):
        self.prob = prob

    def execute(self, x, y, meta1, meta2):

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            prob = tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32)
            if  prob <= 0.33:
                # tf.print('\n - [Pre] Brightness | min:', tf.math.reduce_min(x), ' || max: ', tf.math.reduce_max(x))
                # x = tf.image.adjust_brightness(x,1) # No effect
                # x = tf.image.adjust_brightness(x,0.25) # Some effect
                # x = tf.image.adjust_brightness(x,0.5) # More Effect
                x = tf.image.random_brightness(x, 0.5)
                # tf.print(' - [Post] Brightness | min:', tf.math.reduce_min(x), ' || max: ', tf.math.reduce_max(x))
                
            elif prob > 0.33 and prob < 0.66:
                # tf.print('\n - [Pre] Contrast | min:', tf.math.reduce_min(x), ' || max: ', tf.math.reduce_max(x))
                # x = tf.image.adjust_contrast(x,0.5)
                # x = tf.image.adjust_contrast(x,1.5)
                # x = tf.image.adjust_contrast(x,2.5)
                x = tf.image.random_contrast(x, 0.1, 2.5)
                # tf.print(' - [Post] Contrast | min:', tf.math.reduce_min(x), ' || max: ', tf.math.reduce_max(x))
            
            else:
                # For gamma greater than 1, the histogram will shift towards left and the output image will be darker than the input image. 
                # For gamma less than 1, the histogram will shift towards right and the output image will be brighter than the input image.
                # x = tf.image.adjust_gamma(x, gamma=0.5)
                # x = tf.image.adjust_gamma(x, gamma=1.5)
                x = tf.image.adjust_gamma(x, gamma=tf.random.uniform([], minval=0.5, maxval=1.5, dtype=tf.dtypes.float32))

        x = tf.clip_by_value(x, 0., 1.0)

        return x, y, meta1, meta2

class Shadow:

    def __init__(self, img_shape=(200,200,28,1), prob=0.3):
        
        self.img_shape = img_shape

        self.prob = prob
        self.name = 'Shadow'
    
    def execute(self, x, y, meta1, meta2):

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            
            shadow_height = self.img_shape[1] * 2

            if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= 0.5:
                shadow        = tf.cast(tf.linspace(1.,0.0,shadow_height), dtype=tf.float32)               # [shadow_height] # [1, ....., 0]
            else:
                shadow        = tf.cast(tf.exp(tf.linspace(0., -2., shadow_height)*1.0), dtype=tf.float32) # [shadow_height] # [1, ....., 0]

            shadow        = shadow[-self.img_shape[1]:] # [0.x, ....., 0] # thus, this removes the larger numbers
            shadow        = shadow[tf.newaxis, :]                        # [1                , shadow_height]
            shadow        = tf.repeat(shadow, self.img_shape[0], axis=0) # [self.img_shape[0], shadow_height]
            shadow        = tf.expand_dims(shadow, axis=-1)              # [self.img_shape[0], shadow_height, 1]
            shadow        = tf.repeat(shadow, self.img_shape[2], axis=2) # [self.img_shape[0], shadow_height, self.img_shape[2]]
            shadow        = tf.expand_dims(shadow, axis=-1)              # [img_shape[0], img_shape[1], img_shape[2], 1]


            x = tf.clip_by_value(x, 0., 1.)
            x = x - shadow

        x = tf.clip_by_value(x, 0., 1.)

        return x, y, meta1, meta2

#############################################################
#                           UTILS                           #
#############################################################

def read_zip(filepath_zip, filepath_output=None, leave=False, meta=''):
    
    EXT_ZIP = '.zip'

    # Step 0 - Init
    if Path(filepath_zip).exists():
        if filepath_output is None:
            filepath_zip_parts     = list(Path(filepath_zip).parts)
            filepath_zip_name      = filepath_zip_parts[-1].split(EXT_ZIP)[0]
            filepath_zip_parts[-1] = filepath_zip_name
            filepath_output        = Path(*filepath_zip_parts)

        zip_fp = zipfile.ZipFile(filepath_zip, 'r')
        zip_fp_members = zip_fp.namelist()
        with tqdm.tqdm(total=len(zip_fp_members), desc=' - [{}][Unzip] {} '.format(meta, str(filepath_zip.parts[-1])), leave=leave) as pbar_zip:
            for member in zip_fp_members:
                zip_fp.extract(member, filepath_output)
                pbar_zip.update(1)

        return filepath_output
    else:
        print (' - [{}][ERROR][utils.read_zip()] Path does not exist: {}'.format(meta, filepath_zip) )
        return None

def download_dcm2niix(DIR_TMP, meta=''):
    # Ref: https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20220720

    try:
        import sys
        path_file = Path(DIR_TMP).joinpath('dcm2niix')
        if sys.platform == 'win32':
            if not Path(path_file).exists():
                path_zip = Path(DIR_TMP).joinpath('dcm2niix_win.zip')
                urllib.request.urlretrieve(url='https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20220720/dcm2niix_win.zip', filename=str(path_zip))
                read_zip(path_zip, path_file, meta=meta)
                Path(path_zip).unlink()

            return Path(path_file).joinpath('dcm2niix.exe')    

        elif sys.platform == 'linux':
            if not Path(path_file).exists():
                path_zip = Path(DIR_TMP).joinpath('dcm2niix_lnx.zip')
                urllib.request.urlretrieve(url='https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20220720/dcm2niix_lnx.zip', filename=str(path_zip))
                read_zip(path_zip, path_file, meta=meta)
                Path(path_zip).unlink()
            return Path(path_file).joinpath('dcm2niix')  
        
    
    except:
        traceback.print_exc()
        pdb.set_trace()
        return None

def freqhist_bins(array, bins=100):
    """
    A numpy based function to split the range of pixel values into groups, such that each group has around the same number of pixels

    Ref: https://github.com/AIEMMU/MRI_Prostate/blob/master/Pre%20processing%20Promise12.ipynb
    Ref: https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py
    """
    
    imsd = np.sort(array.flatten())
    t = np.array([0.001])
    t = np.append(t, np.arange(bins)/bins+(1/2/bins))
    t = np.append(t, 0.999)
    t = (len(imsd)*t+0.5).astype(int)
    return np.unique(imsd[t])

def hist_scaled(array, brks=None, bins=100):
    """
    Intensity Tranformations: Intensity Normalization vs Histogram Equalization (this)
    Scales a tensor using `freqhist_bins` to values between 0 and 1

    Ref: https://github.com/AIEMMU/MRI_Prostate/blob/master/Pre%20processing%20Promise12.ipynb
    Ref: https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py
    """
    
    if brks is None: 
        brks = freqhist_bins(array, bins=bins)
    ys = np.linspace(0., 1., len(brks))
    if 0:
        plt.plot(brks, ys); plt.show()
        pdb.set_trace()
    x = array.flatten()
    
    x = np.interp(x, brks, ys)
    array_tmp = np.reshape(x, array.shape)
    array_tmp = np.clip(array_tmp, 0.,1.)
    return array_tmp

#############################################################
#                         DATASET                           #
#############################################################

class ProstateXDataset:
    """
    This dataset from Radmoud UMC, Netherlands of 2017 contains prostate masks on T2-weighted MR scans.
    Data is avaialble on the TCIA: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080779#61080779ccf02ddde6884e4ba1ec73d08be362c6

    """

    def __init__(self, path_data
                    , transforms=[], patient_shuffle=False
                    , parallel_calls=4, deterministic=False
                    , download=True
                    , single_patient=False):
        
        # Step 0 - Init names
        self.name = 'Pros_ProstateX'
        self.DIRNAME_TMP = '_tmp'
        self.DIRNAME_RAW = 'raw'
        self.DIRNAME_PROCESSED = 'processed'
        # self.DIRNAME_PROCESSED2 = 'processed2'
        self.DIRNAME_PROCESSED2 = 'processed3'  # HistStandardization via pytorch
        print (' - [ProstateXDataset] self.DIRNAME_PROCESSED2: ', self.DIRNAME_PROCESSED2)

        self.FOLDERNAME_MRI    = '{}_mri{}'
        self.FOLDERNAME_MASK   = '{}_mask{}' 
        
        self.EXT_ZIP = '.zip'
        self.EXT_NII = '.nii'

        # Step 0.2 - Values fixed for dataset by me
        self.LABEL_MAP    = {'Prostate': 1}
        self.LABEL_COLORS = {1: [0,255,0]}
        self.VOXELS_X_MIDPT = 100
        self.VOXELS_Y_MIDPT = 100
        self.VOXELS_Z_MAX   = 28
        self.PATIENT_COUNT = 66
        self.SPACING = (0.5, 0.5, 3.0)

        # Step 1 - Init params
        self.path_data   = path_data
        self.transforms = transforms
        self.patient_shuffle = patient_shuffle
        self.single_patient = single_patient
        
        self.parallel_calls = parallel_calls
        self.deterministic  = deterministic

        self.PATIENT_IDS = [4, 7, 9, 15, 20, 26, 46, 54, 56, 65, 66, 69, 70, 72, 76, 83, 84, 89, 90, 94, 96, 102, 111, 112, 117, 118, 121, 125, 129, 130, 134, 136, 141, 142, 144, 150, 156, 161, 168, 170, 176, 177, 182, 183, 184, 188, 193, 196, 198, 201, 209, 217, 219, 234, 241, 244, 249, 254, 265, 275, 297, 309, 311, 323, 334, 340]

        # Step 2 - Init process
        if download:
            self._get()
    
    def _preprocess_download(self):
        
        try:

            class TCIAClient:
                """
                - Ref: https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide
                - Ref: https://github.com/TCIA-Community/TCIA-API-SDK/tree/master/tcia-rest-client-python/src
                """
                GET_IMAGE = "getImage"
                GET_MANUFACTURER_VALUES = "getManufacturerValues"
                GET_MODALITY_VALUES = "getModalityValues"
                GET_COLLECTION_VALUES = "getCollectionValues"
                GET_BODY_PART_VALUES = "getBodyPartValues"
                GET_PATIENT_STUDY = "getPatientStudy"
                GET_SERIES = "getSeries"
                GET_PATIENT = "getPatient"
                GET_SERIES_SIZE = "getSeriesSize"
                CONTENTS_BY_NAME = "ContentsByName"

                def __init__(self, baseUrl, resource):
                    self.baseUrl = baseUrl + "/" + resource
                    self.STATUS_OK = 200
                    self.DECODER = 'utf-8'

                def execute(self, url, queryParameters={}, verbose=False):
                    queryParameters = dict((k, v) for k, v in queryParameters.items() if v)
                    queryString = "?%s" % urllib.parse.urlencode(queryParameters)
                    requestUrl = url + queryString
                    request = urllib.request.Request(url=requestUrl, headers={'Connection': 'keep-alive', 'User-Agent': 'PostmanRuntime/7.28.4'}) # headers = {'User-Agent': 'PostmanRuntime/7.28.4'}, headers={'Connection': 'keep-alive'}
                    if verbose:
                        print (' - [execute()] URL: ', requestUrl)
                        print (request.headers)
                    resp = urllib.request.urlopen(request)
                    return resp

                def read_response(self, resp):
                    if resp.status == self.STATUS_OK:
                        return eval(resp.read().decode(self.DECODER))
                    else:
                        return None

                def get_patient(self,collection = None , outputFormat = "json" ):
                    serviceUrl = self.baseUrl + "/query/" + self.GET_PATIENT
                    queryParameters = {"Collection" : collection , "format" : outputFormat }
                    resp = self.execute(serviceUrl , queryParameters)
                    return resp

                def get_patient_study(self,collection = None , patientId = None , studyInstanceUid = None , outputFormat = "json" ):
                    serviceUrl = self.baseUrl + "/query/" + self.GET_PATIENT_STUDY
                    queryParameters = {"Collection" : collection , "PatientID" : patientId , "StudyInstanceUID" : studyInstanceUid , "format" : outputFormat }
                    resp = self.execute(serviceUrl , queryParameters)
                    return resp

                def get_series(self, collection=None, patientId=None, modality=None, studyInstanceUid=None, seriesInstanceUid=None, outputFormat = "json" ):
                    serviceUrl = self.baseUrl + "/query/" + self.GET_SERIES
                    queryParameters = {"Collection" : collection, "patientId": patientId, "StudyInstanceUID" : studyInstanceUid, 'SeriesInstanceUID': seriesInstanceUid,"Modality" : modality, "format" : outputFormat }
                    resp = self.execute(serviceUrl , queryParameters)
                    return resp

                def get_image(self , seriesInstanceUid , downloadPath, zipFileName):
                    try:
                        serviceUrl = self.baseUrl + "/query/" + self.GET_IMAGE
                        queryParameters = { "SeriesInstanceUID" : seriesInstanceUid }   
                        resp = self.execute(serviceUrl, queryParameters)
                        filepath = Path(downloadPath).joinpath(zipFileName)
                        data = resp.read()
                        with open(filepath, 'wb') as fp:
                            fp.write(data)
                        
                        tmp = list(Path(filepath).parts)
                        tmp[-1] = tmp[-1].split('.zip')[0]
                        filepath_output = Path(*tmp)
                        read_zip(filepath, filepath_output, leave=False, meta='')

                        Path(filepath).unlink()

                    except:
                        traceback.print_exc()


            # Step 0.1 - Init TCIA client
            baseUrl = 'https://services.cancerimagingarchive.net/services/v4'
            resource = 'TCIA'
            collection = 'PROSTATEx'
            client = TCIAClient(baseUrl=baseUrl, resource=resource)

            # Step 0.2 - Init seriesids from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080779#61080779ccf02ddde6884e4ba1ec73d08be362c6
            seriesids_mask = ['1.2.276.0.7230010.3.1.3.1070885483.26120.1599221759.302','1.2.276.0.7230010.3.1.3.1070885483.24432.1599221753.549','1.2.276.0.7230010.3.1.3.1070885483.5008.1599221756.491','1.2.276.0.7230010.3.1.3.1070885483.27140.1599221767.245','1.2.276.0.7230010.3.1.3.1070885483.27624.1599221764.667','1.2.276.0.7230010.3.1.3.1070885483.16240.1599221761.962','1.2.276.0.7230010.3.1.3.1070885483.8220.1599221770.50','1.2.276.0.7230010.3.1.3.1070885483.9460.1599221773.50','1.2.276.0.7230010.3.1.3.1070885483.24532.1599221775.720','1.2.276.0.7230010.3.1.3.1070885483.24608.1599221781.105','1.2.276.0.7230010.3.1.3.1070885483.24736.1599221778.322','1.2.276.0.7230010.3.1.3.1070885483.22584.1599221783.667','1.2.276.0.7230010.3.1.3.1070885483.24012.1599221786.269','1.2.276.0.7230010.3.1.3.1070885483.21792.1599221788.891','1.2.276.0.7230010.3.1.3.1070885483.5464.1599221791.439','1.2.276.0.7230010.3.1.3.1070885483.5672.1599221794.52','1.2.276.0.7230010.3.1.3.1070885483.16948.1599221799.294','1.2.276.0.7230010.3.1.3.1070885483.16324.1599221796.636','1.2.276.0.7230010.3.1.3.1070885483.26880.1599221801.878','1.2.276.0.7230010.3.1.3.1070885483.26756.1599221804.455','1.2.276.0.7230010.3.1.3.1070885483.21116.1599221807.124','1.2.276.0.7230010.3.1.3.1070885483.23772.1599221809.995','1.2.276.0.7230010.3.1.3.1070885483.23924.1599221812.620','1.2.276.0.7230010.3.1.3.1070885483.24056.1599221815.833','1.2.276.0.7230010.3.1.3.1070885483.23100.1599221818.605','1.2.276.0.7230010.3.1.3.1070885483.4648.1599221821.338','1.2.276.0.7230010.3.1.3.1070885483.5584.1599221824.60','1.2.276.0.7230010.3.1.3.1070885483.17604.1599221827.76','1.2.276.0.7230010.3.1.3.1070885483.23788.1599221829.820','1.2.276.0.7230010.3.1.3.1070885483.26076.1599221832.601','1.2.276.0.7230010.3.1.3.1070885483.22052.1599221835.360','1.2.276.0.7230010.3.1.3.1070885483.26444.1599221837.990','1.2.276.0.7230010.3.1.3.1070885483.27528.1599221841.604','1.2.276.0.7230010.3.1.3.1070885483.3400.1599221844.204','1.2.276.0.7230010.3.1.3.1070885483.15416.1599221846.853','1.2.276.0.7230010.3.1.3.1070885483.22008.1599221849.372','1.2.276.0.7230010.3.1.3.1070885483.16016.1599221851.990','1.2.276.0.7230010.3.1.3.1070885483.8088.1599221854.809','1.2.276.0.7230010.3.1.3.1070885483.5396.1599221858.335','1.2.276.0.7230010.3.1.3.1070885483.25220.1599221863.968','1.2.276.0.7230010.3.1.3.1070885483.5464.1599221860.888','1.2.276.0.7230010.3.1.3.1070885483.7992.1599221866.575','1.2.276.0.7230010.3.1.3.1070885483.16948.1599221869.130','1.2.276.0.7230010.3.1.3.1070885483.24588.1599221871.775','1.2.276.0.7230010.3.1.3.1070885483.24928.1599221874.342','1.2.276.0.7230010.3.1.3.1070885483.14224.1599221879.641','1.2.276.0.7230010.3.1.3.1070885483.25300.1599221882.469','1.2.276.0.7230010.3.1.3.1070885483.21512.1599221877.6','1.2.276.0.7230010.3.1.3.1070885483.5808.1599221885.265','1.2.276.0.7230010.3.1.3.1070885483.22380.1599221887.327','1.2.276.0.7230010.3.1.3.1070885483.23580.1599221890.67','1.2.276.0.7230010.3.1.3.1070885483.15904.1599221892.237','1.2.276.0.7230010.3.1.3.1070885483.4220.1599221894.379','1.2.276.0.7230010.3.1.3.1070885483.25552.1599221897.694','1.2.276.0.7230010.3.1.3.1070885483.26088.1599221900.380','1.2.276.0.7230010.3.1.3.1070885483.5296.1599221903.189','1.2.276.0.7230010.3.1.3.1070885483.6284.1599221908.467','1.2.276.0.7230010.3.1.3.1070885483.17416.1599221911.141','1.2.276.0.7230010.3.1.3.1070885483.1780.1599221905.866','1.2.276.0.7230010.3.1.3.1070885483.24236.1599221913.873','1.2.276.0.7230010.3.1.3.1070885483.24632.1599221916.651','1.2.276.0.7230010.3.1.3.1070885483.25044.1599221924.876','1.2.276.0.7230010.3.1.3.1070885483.8668.1599221919.430','1.2.276.0.7230010.3.1.3.1070885483.24644.1599221922.113','1.2.276.0.7230010.3.1.3.1070885483.27560.1599221927.547','1.2.276.0.7230010.3.1.3.1070885483.25100.1599221930.643']
            seriesids_mri = ['1.3.6.1.4.1.14519.5.2.1.7310.5101.107276353018221365492863559094','1.3.6.1.4.1.14519.5.2.1.7310.5101.127076787998581344993055912043','1.3.6.1.4.1.14519.5.2.1.7310.5101.128915395083536972793214199559','1.3.6.1.4.1.14519.5.2.1.7310.5101.156355474235536930096765953749','1.3.6.1.4.1.14519.5.2.1.7310.5101.182666056353308077379356745296','1.3.6.1.4.1.14519.5.2.1.7310.5101.217705819956691748023253683197','1.3.6.1.4.1.14519.5.2.1.7310.5101.233894782007809884236855995183','1.3.6.1.4.1.14519.5.2.1.7310.5101.234424615181029312892423746307','1.3.6.1.4.1.14519.5.2.1.7310.5101.259467270859277582122403078829','1.3.6.1.4.1.14519.5.2.1.7310.5101.260386623049829078909727548239','1.3.6.1.4.1.14519.5.2.1.7310.5101.306748150836286513093400468929','1.3.6.1.4.1.14519.5.2.1.7310.5101.318856235542799349557600990165','1.3.6.1.4.1.14519.5.2.1.7310.5101.337896560664468718350275750005','1.3.6.1.4.1.14519.5.2.1.7310.5101.568951414433420775694157731047','1.3.6.1.4.1.14519.5.2.1.7310.5101.877607202381357458819664904037','1.3.6.1.4.1.14519.5.2.1.7310.5101.898676446450078900613536879876','1.3.6.1.4.1.14519.5.2.1.7311.5101.101130890931274399577478152963','1.3.6.1.4.1.14519.5.2.1.7311.5101.101130934168942593154270621032','1.3.6.1.4.1.14519.5.2.1.7311.5101.105766764724449379107432362750','1.3.6.1.4.1.14519.5.2.1.7311.5101.109832262779867794170417053903','1.3.6.1.4.1.14519.5.2.1.7311.5101.119868581724901379856626647643','1.3.6.1.4.1.14519.5.2.1.7311.5101.124933244802704100684011471210','1.3.6.1.4.1.14519.5.2.1.7311.5101.126808882287123482193592987944','1.3.6.1.4.1.14519.5.2.1.7311.5101.146422450520894659910416624671','1.3.6.1.4.1.14519.5.2.1.7311.5101.149728379277305470281540696443','1.3.6.1.4.1.14519.5.2.1.7311.5101.155875154686610653777230856177','1.3.6.1.4.1.14519.5.2.1.7311.5101.156910737340231640626298466189','1.3.6.1.4.1.14519.5.2.1.7311.5101.160225377533762960695041069832','1.3.6.1.4.1.14519.5.2.1.7311.5101.165636727522505811947315188478','1.3.6.1.4.1.14519.5.2.1.7311.5101.166265209513832781969059787909','1.3.6.1.4.1.14519.5.2.1.7311.5101.175570448971296597162147241386','1.3.6.1.4.1.14519.5.2.1.7311.5101.178750617003896154912438088527','1.3.6.1.4.1.14519.5.2.1.7311.5101.180041601751316950966041961765','1.3.6.1.4.1.14519.5.2.1.7311.5101.180650601474055581355948988643','1.3.6.1.4.1.14519.5.2.1.7311.5101.186553403053805718209125341361','1.3.6.1.4.1.14519.5.2.1.7311.5101.196511587017551386049158542619','1.3.6.1.4.1.14519.5.2.1.7311.5101.206828891270520544417996275680','1.3.6.1.4.1.14519.5.2.1.7311.5101.226939533786316417608293472819','1.3.6.1.4.1.14519.5.2.1.7311.5101.236967836312224538274213979128','1.3.6.1.4.1.14519.5.2.1.7311.5101.242424835264181527414562151046','1.3.6.1.4.1.14519.5.2.1.7311.5101.245315537897059083725251245833','1.3.6.1.4.1.14519.5.2.1.7311.5101.248203933751895346486742820088','1.3.6.1.4.1.14519.5.2.1.7311.5101.252185783543855817242399041825','1.3.6.1.4.1.14519.5.2.1.7311.5101.252385246988168654915352850239','1.3.6.1.4.1.14519.5.2.1.7311.5101.257928599770116866513356683535','1.3.6.1.4.1.14519.5.2.1.7311.5101.258067469374761412596368241889','1.3.6.1.4.1.14519.5.2.1.7311.5101.262719318163046156031519223454','1.3.6.1.4.1.14519.5.2.1.7311.5101.266720221212035669954581062639','1.3.6.1.4.1.14519.5.2.1.7311.5101.271991298059338527584681788043','1.3.6.1.4.1.14519.5.2.1.7311.5101.282008038881694967871335639325','1.3.6.1.4.1.14519.5.2.1.7311.5101.287403883614425048490255475041','1.3.6.1.4.1.14519.5.2.1.7311.5101.289711251504454079274667690291','1.3.6.1.4.1.14519.5.2.1.7311.5101.304058605325520782900293953679','1.3.6.1.4.1.14519.5.2.1.7311.5101.312442050391027773578890036831','1.3.6.1.4.1.14519.5.2.1.7311.5101.318660391556220519803669869815','1.3.6.1.4.1.14519.5.2.1.7311.5101.322192860849906453257530108507','1.3.6.1.4.1.14519.5.2.1.7311.5101.333332554059735467244825907777','1.3.6.1.4.1.14519.5.2.1.7311.5101.334326985258868181144176796864','1.3.6.1.4.1.14519.5.2.1.7311.5101.357054868848226402548612686613','1.3.6.1.4.1.14519.5.2.1.7311.5101.460453169764361555745878708935','1.3.6.1.4.1.14519.5.2.1.7311.5101.472900684997509231929411950536','1.3.6.1.4.1.14519.5.2.1.7311.5101.666342438883185464672112795625','1.3.6.1.4.1.14519.5.2.1.7311.5101.785303327607349403635620326985','1.3.6.1.4.1.14519.5.2.1.7311.5101.813054628707353396925223754592','1.3.6.1.4.1.14519.5.2.1.7311.5101.905763996607212880296069425861','1.3.6.1.4.1.14519.5.2.1.7311.5101.928012760939767746081753156593']

            # Step 1 - Loop over all seriesids_mask
            print ('')
            with tqdm.tqdm(total=len(seriesids_mask), desc=' - [{}][TCIA Download][Masks]'.format(self.name)) as pbar_mask:
                for id_, seriesid_mask in enumerate(seriesids_mask):
                    
                    try:
                        resp_series = client.get_series(collection=collection, seriesInstanceUid=seriesid_mask)
                        if resp_series.status == 200:
                            obj_series = client.read_response(resp_series)
                            if len(obj_series):
                                patient_id = obj_series[0]['PatientID']
                                downloadPath = Path(self.path_dir_raw).joinpath(patient_id)
                                Path(downloadPath).mkdir(exist_ok=True, parents=True)
                                client.get_image(seriesInstanceUid=seriesid_mask, downloadPath=str(downloadPath), zipFileName=self.FOLDERNAME_MASK.format(patient_id, self.EXT_ZIP))
                            else:
                                print (' - [{}][ERROR] No patient info for seriesid_mask: {}'.format(self.name, seriesid_mask))
                        else:
                            print (' - [{}][ERROR] Could not read seriesid_mask: {}'.format(self.name, seriesid_mask))
                    
                    except:
                        print (' - [{}][ERROR] Some error with seriesid_mask: {}'.format(self.name, seriesid_mask) )
                        traceback.print_exc()
                        pdb.set_trace()
                    
                    pbar_mask.update(1)
            
            # Step 2 - Loop over all seriesids_mri
            print ('')
            with tqdm.tqdm(total=len(seriesids_mri), desc=' - [{}][TCIA Download][MRI]'.format(self.name)) as pbar_mri:
                for id_, seriesid_mri in enumerate(seriesids_mri):

                    try:
                        resp_series = client.get_series(collection=collection, seriesInstanceUid=seriesid_mri)
                        if resp_series.status == 200:
                            obj_series = client.read_response(resp_series)
                            if len(obj_series):
                                patient_id = obj_series[0]['PatientID']
                                downloadPath = Path(self.path_dir_raw).joinpath(patient_id)
                                Path(downloadPath).mkdir(exist_ok=True, parents=True)
                                client.get_image(seriesInstanceUid=seriesid_mri, downloadPath=str(downloadPath), zipFileName=self.FOLDERNAME_MRI.format(patient_id, self.EXT_ZIP))
                            else:
                                print (' - [ERROR] No patient info for seriesid_mri: ', seriesid_mri)
                        else:
                            print (' - [ERROR] Could not read seriesid_mri: ', seriesid_mri)
                    
                    except:
                        print (' - [ERROR] Some error with seriesid_mri', seriesid_mri)
                        traceback.print_exc()
                        pdb.set_trace()

                    pbar_mri.update(1)

            return True
        
        except:
            traceback.print_exc()
            pdb.set_trace()
            return False

    def _preprocess_converttonifti(self):

        # Step 1 - Convert to NIFTI using dcm2niix
        try:
            print ('')
            path_dcm2niix = download_dcm2niix(self.path_dir_tmp)
            if path_dcm2niix is not None:
                with tqdm.tqdm(total=len(list(Path(self.path_dir_raw).glob('*'))), desc=' - [{}][Convert to volume] '.format(self.name)) as pbar_convert:
                    for path_patient in Path(self.path_dir_raw).iterdir():
                        if Path(path_patient).is_dir():

                            # Step 1.1 - Init vars
                            patient_id = Path(path_patient).parts[-1]
                            foldername_mri = self.FOLDERNAME_MRI.format(patient_id, '')
                            path_mri = Path(path_patient).joinpath(foldername_mri)
                            foldername_mask = self.FOLDERNAME_MASK.format(patient_id, '')
                            path_masks = list(Path(path_patient).joinpath(foldername_mask).glob('*')) # there should only be 1, but I am still looping over it

                            # Step 1.2 - Volumize MRI
                            if 1:
                                try:
                                    launch_mri = [str(path_dcm2niix), "-e", "n", "-b", "n", "-v", "0" , "-o", str(path_patient), "-f", foldername_mri, str(path_mri)]
                                    _ = subprocess.run(launch_mri, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                                    # parameters for conversion.
                                    # -e : export as NRRD (y) or MGH (o) instead of NIfTI (y/n/o, default n)
                                    # -o : output directory (omit to save to input folder)
                                    # -b : BIDS sidecar (y/n/o [o=only: no NIfTI], default y)
                                    # -f : filename (%a=antenna (coil) name, %b=basename, %c=comments, %d=description, %e=echo number, %f=folder name,
                                
                                except:
                                    print (' - [ERROR][_preprocess_converttonifti()][MRI] patient_id: ', patient_id)
                                    traceback.print_exc()
                                    pdb.set_trace()
                            
                            # Step 1.3 - Volumize MASK
                            if 1:
                                try:
                                    
                                    for path_dcm in path_masks:

                                        # Step 3.3.1 - Read data
                                        ds_mask  = pydicom.dcmread(str(path_dcm))
                                        # len(ds_mask.ReferencedSeriesSequence[0].ReferencedInstanceSequence) == len(vol_mri)
                                        #  - e.g. --> ds_mask.ReferencedSeriesSequence[0].ReferencedInstanceSequence[0].ReferencedSOPInstanceUID
                                        # len(ds_mask.PerFrameFunctionalGroupsSequence) == len(vol_mask)
                                        #  - e.g. --> ds_mask.PerFrameFunctionalGroupsSequence[idx].DerivationImageSequence

                                        vol_mask = ds_mask.pixel_array # [H,W,D]
                                        nii_mri  = nibabel.load(str(Path(path_patient).joinpath(self.FOLDERNAME_MRI.format(patient_id, self.EXT_NII)))) 
                                        vol_mri  = nii_mri.get_fdata()
                                        if 0:
                                            print ('\n - [_preprocess_converttonifti()][mask] pixelspacing: {} | slicespacing: {}'.format(
                                                ds_mask.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
                                                ,  ds_mask.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness))
                                            print (' - [_preprocess_converttonifti()][vol]  pixelspacing: {} | slicespacing: {}'.format(nii_mri.header['pixdim'][1:3], nii_mri.header['pixdim'][3] ))

                                        # Step 3.3.2 - Downsample segmentation
                                        if 0:
                                            end      = vol_mask.shape[0]
                                            step     = end // vol_mri.shape[2]
                                            vol_mask = vol_mask[0:end:step, ...]
                                            diff     = vol_mask.shape[0] - vol_mri.shape[2]
                                            if diff != 0:
                                                vol_mask = vol_mask[:-diff, ...]
                                            assert vol_mask.shape[0] == vol_mri.shape[2]

                                        else:
                                            idxs = [idx for idx, item in enumerate(ds_mask.PerFrameFunctionalGroupsSequence) if len(ds_mask.PerFrameFunctionalGroupsSequence[idx].DerivationImageSequence)]
                                            vol_mask = vol_mask[idxs, :, :]
                                            assert vol_mri.shape[2] == vol_mask.shape[0]

                                        vol_mask = np.rot90(vol_mask, k=-1, axes=(1, 2))
                                        vol_mask = np.swapaxes(vol_mask, 0, 1)
                                        vol_mask = np.swapaxes(vol_mask, 1, 2)

                                        if vol_mri.shape != vol_mask.shape:
                                            # print ('  - [INFO][_preprocess_converttonifti()][{}] vol_mri: {} | vol_mask: {} '.format(patient_id, vol_mri.shape, vol_mask.shape))
                                            vol_mask = skimage.transform.resize(vol_mask, vol_mri.shape, preserve_range=True, order=0) # order = 0 is nearest neighbour interpolation
                                        assert vol_mask.shape == vol_mri.shape
                                        
                                        # saving segmentations
                                        niiobj = nibabel.Nifti1Image(vol_mask, nii_mri.affine)
                                        path_mask = Path(path_patient).joinpath(self.FOLDERNAME_MASK.format(patient_id, self.EXT_NII))
                                        nibabel.save(niiobj, str(path_mask))

                                except:
                                    print (' - [ERROR][_preprocess_converttonifti()][MASK] patient_id: ', patient_id)
                                    traceback.print_exc()
                                    pdb.set_trace()
                        
                        pbar_convert.update(1)
            return True

        except:
            traceback.print_exc()
            pdb.set_trace()
            return False

    def _preprocess_resample(self):

        try:


            with tqdm.tqdm(total=len(list(Path(self.path_dir_raw).glob('*'))), desc=' - [{}][Resampling to {}] '.format(self.name, self.SPACING)) as pbar_resample:
                for path_patient in Path(self.path_dir_raw).iterdir():
                    if Path(path_patient).is_dir():
                        patient_id = Path(path_patient).parts[-1]
                        path_mask_raw  = Path(path_patient).joinpath(self.FOLDERNAME_MASK.format(patient_id, self.EXT_NII))
                        path_img_raw   = Path(path_patient).joinpath(self.FOLDERNAME_MRI.format(patient_id, self.EXT_NII))

                        if 1:

                            path_img_proc  = Path(self.path_dir_processed).joinpath(patient_id, self.FOLDERNAME_MRI.format(patient_id, self.EXT_NII))
                            path_mask_proc = Path(self.path_dir_processed).joinpath(patient_id, self.FOLDERNAME_MASK.format(patient_id, self.EXT_NII))

                            import math
                            import skimage.transform as skTrans

                            img_og   = nibabel.load(path_img_raw)
                            ratio    = [spacing_old / self.SPACING[i] for i, spacing_old in enumerate(img_og.header['pixdim'][1:4])]
                            size_new = tuple(math.ceil(size_old * ratio[i]) - 1 for i, size_old in enumerate(img_og.header['dim'][1:4]))
                            img_og.header['pixdim'][1:4] = self.SPACING

                            img_resampled  = skTrans.resize(img_og.get_fdata(), size_new, order=3, preserve_range=True)
                            img_resampled = img_resampled.astype(np.int16)
                            niiobj = nibabel.Nifti1Image(img_resampled, img_og.affine, img_og.header)
                            Path(path_img_proc).parent.mkdir(exist_ok=True, parents=True)
                            nibabel.save(niiobj, str(path_img_proc))

                            mask_og = nibabel.load(path_mask_raw)
                            mask_og.header['pixdim'][1:4] = self.SPACING
                            
                            mask_resampled = skTrans.resize(mask_og.get_fdata(), size_new, order=0, preserve_range=True) # 0 = Nearest Neighbour
                            mask_resampled = mask_resampled.astype(np.uint8)
                            # print (' - [{}] np.unique(mask_resampled): {}'.format(patient_id, np.unique(mask_resampled)))
                            niiobj = nibabel.Nifti1Image(mask_resampled, mask_og.affine, mask_og.header)
                            Path(path_mask_proc).parent.mkdir(exist_ok=True, parents=True)
                            nibabel.save(niiobj, str(path_mask_proc))
            
                    pbar_resample.update(1)

            return True
        
        except:
            traceback.print_exc()
            pdb.set_trace()
            return False

    def _preprocess_crop(self):

        try:
            
            with tqdm.tqdm(total=len(list(Path(self.path_dir_processed).glob('*'))), desc=' - [][Crop/IntensityOps] '.format(self.name)) as pbar_cropping:
                for path_patient in Path(self.path_dir_processed).iterdir():
                    if Path(path_patient).is_dir():

                        # Step 7.1 - Get paths
                        patient_id = Path(path_patient).parts[-1]
                        path_mask_proc  = Path(path_patient).joinpath(self.FOLDERNAME_MASK.format(patient_id, self.EXT_NII))
                        path_mri_proc   = Path(path_patient).joinpath(self.FOLDERNAME_MRI.format(patient_id, self.EXT_NII))

                        # Step 7.1 - Get mask (crop and pad)
                        mri  = nibabel.load(path_mri_proc)
                        mri_array = mri.get_fdata()
                        if 0:
                            import matplotlib.pyplot as plt
                            f,axarr = plt.subplots(1,2); axarr[0].imshow(mri_array[:,:,18], cmap='gray'); axarr[1].imshow(hist_scaled(mri_array, bins=100)[:,:,18], cmap='gray', vmin=0., vmax=1.); plt.suptitle('Bins=100');plt.show(block=False)
                            f,axarr = plt.subplots(1,2); axarr[0].imshow(mri_array[:,:,18], cmap='gray'); axarr[1].imshow(hist_scaled(mri_array, bins=1000)[:,:,18], cmap='gray', vmin=0., vmax=1.); plt.suptitle('Bins=1000');plt.show(block=False)
                            f,axarr = plt.subplots(1,2); axarr[0].imshow(mri_array[:,:,18], cmap='gray'); axarr[1].imshow(hist_scaled(mri_array, bins=10000)[:,:,18], cmap='gray', vmin=0., vmax=1.); plt.suptitle('Bins=10000');plt.show(block=False)
                            pdb.set_trace()

                        mri_array = hist_scaled(mri_array, bins=10000)
                        mask = nibabel.load(path_mask_proc)
                        mask_array = mask.get_fdata()
                        mask_idxs  = np.argwhere(mask_array > 0)
                        mask_midpt = np.array([np.mean(mask_idxs[:,0]), np.mean(mask_idxs[:,1]), np.mean(mask_idxs[:,2])]).astype(np.uint8).tolist()

                        # Step 7.2 - Crop
                        mask_array_new = np.array(mask_array[mask_midpt[0] - self.VOXELS_X_MIDPT:mask_midpt[0] + self.VOXELS_X_MIDPT, mask_midpt[1] - self.VOXELS_Y_MIDPT:mask_midpt[1] + self.VOXELS_Y_MIDPT, :], copy=True)
                        mri_array_new  = np.array(mri_array[mask_midpt[0] - self.VOXELS_X_MIDPT:mask_midpt[0] + self.VOXELS_X_MIDPT, mask_midpt[1] - self.VOXELS_Y_MIDPT:mask_midpt[1] + self.VOXELS_Y_MIDPT, :], copy=True)
                        
                        # Step 7.3 - Pad (in z)
                        mask_array_height = mask_array_new.shape[2]
                        mask_array_diff   = np.abs(mask_array_height - self.VOXELS_Z_MAX)
                        pad_left          = mask_array_diff//2
                        pad_right         = mask_array_diff-mask_array_diff//2
                        if mask_array_height > self.VOXELS_Z_MAX:
                            mask_array_new = mask_array_new[:,:,pad_left:-pad_right]
                            mri_array_new  = mri_array_new[:,:,pad_left:-pad_right]
                        else:
                            if mask_array_diff:
                                mask_array_new = np.pad(mask_array_new, (pad_left, pad_right), 'constant')
                                mask_array_new = mask_array_new[pad_left:-pad_right, pad_left:-pad_right, :]

                                mri_array_new = np.pad(mri_array_new, (pad_left, pad_right), 'constant')
                                mri_array_new = mri_array_new[pad_left:-pad_right, pad_left:-pad_right, :]

                        print (' - [_preprocess_crop()][{}] mask_array: {} || mask_array_new: {}'.format(patient_id, mask_array.shape, mask_array_new.shape))
                        print (' - [_preprocess_crop()][{}] mri_array : {} || mri_array_new : {}'.format(patient_id, mri_array.shape, mri_array_new.shape))

                        # Step 7.4 - Save to disk
                        path_mask_proc2  = Path(self.path_dir_processed2).joinpath(patient_id, self.FOLDERNAME_MASK.format(patient_id, self.EXT_NII))
                        Path(path_mask_proc2).parent.mkdir(exist_ok=True, parents=True)
                        niiobj = nibabel.Nifti1Image(mask_array_new, mask.affine, mask.header)
                        nibabel.save(niiobj, str(path_mask_proc2))

                        path_mri_proc2  = Path(self.path_dir_processed2).joinpath(patient_id, self.FOLDERNAME_MRI.format(patient_id, self.EXT_NII))
                        Path(path_mri_proc2).parent.mkdir(exist_ok=True, parents=True)
                        niiobj = nibabel.Nifti1Image(mri_array_new, mri.affine, mri.header)
                        nibabel.save(niiobj, str(path_mri_proc2))

            return True

        except:
            traceback.print_exc()
            pdb.set_trace()
            return False

    def _get(self):

        # Step 0 - Init paths
        self.path_dir_dataset    = Path(self.path_data).joinpath(self.name)
        self.path_dir_tmp        = Path(self.path_dir_dataset).joinpath(self.DIRNAME_TMP)
        self.path_dir_raw        = Path(self.path_dir_dataset).joinpath(self.DIRNAME_RAW)
        self.path_dir_processed  = Path(self.path_dir_dataset).joinpath(self.DIRNAME_PROCESSED)
        self.path_dir_processed2 = Path(self.path_dir_dataset).joinpath(self.DIRNAME_PROCESSED2)

        Path(self.path_dir_tmp).mkdir(exist_ok=True, parents=True)
        Path(self.path_dir_raw).mkdir(exist_ok=True, parents=True)
        Path(self.path_dir_processed).mkdir(exist_ok=True, parents=True)
        Path(self.path_dir_processed2).mkdir(exist_ok=True, parents=True)

        
        # Step 1 - Download and process dataset
        if len(list(Path(self.path_dir_processed2).glob('*'))) != self.PATIENT_COUNT:
            print ('')
            status_download = self._preprocess_download()
            # status_download = True
            if status_download:
                print ('')
                status_convert = self._preprocess_converttonifti()
                if status_convert:
                    print ('')
                    status_resample = self._preprocess_resample()
                    if status_resample:
                        print ('')
                        self.status_preprocessed = self._preprocess_crop()
        
        else:
            self.status_preprocessed = True
        
        
        # Step 2 - Get all paths
        self.paths_mri, self.paths_mask = [], []
        for path_patient in Path(self.path_dir_processed2).iterdir():
            if Path(path_patient).is_dir():
                patient_id = Path(path_patient).parts[-1]
                self.paths_mri.append(Path(path_patient).joinpath(self.FOLDERNAME_MRI.format(patient_id, self.EXT_NII)))
                self.paths_mask.append(Path(path_patient).joinpath(self.FOLDERNAME_MASK.format(patient_id, self.EXT_NII)))
    
    def __len__(self):
        return len(self.paths_mri)

    def generator(self):
        """
         - Note: 
            - In general, even when running your model on an accelerator like a GPU or TPU, the tf.data pipelines are run on the CPU
                - Ref: https://www.tensorflow.org/guide/data_performance_analysis#analysis_workflow
        """

        try:
            
            if len(self.paths_mri) and len(self.paths_mask):
                
                # Step 1 - Create basic generator
                dataset = None
                dataset = tf.data.Dataset.from_generator(self._generator3D
                    , output_types=(tf.float32, tf.uint8, tf.int32, tf.string)
                    ,args=())
                
                # Step 1 - Data augmentations
                if len(self.transforms):
                    for transform in self.transforms:
                        try:
                            dataset = dataset.map(transform.execute, num_parallel_calls=self.parallel_calls, deterministic=self.deterministic)
                        except:
                            traceback.print_exc()
                            print (' - [ERROR][{}] Issue with transform: {}'.format(self.name, transform.name))
                else:
                    print ('')
                    print (' - [INFO][{}] No transformations available!'.format(self.name))
                    print ('')

                # Step 6 - Return
                return dataset
            
            else:
                return None

        except:
            traceback.print_exc()
            pdb.set_trace()
            return None
    
    def _generator3D(self):

        # Step 1 - Get patient_idxs
        idxs = np.arange(len(self.paths_mri)).tolist()
        if self.patient_shuffle: np.random.shuffle(idxs)
        self.mri_max = []

        # Step 2 - Yield
        for idx in idxs:

            if self.single_patient:
                idx = 13 # ProstateX-0072

            path_mri  = self.paths_mri[idx]
            path_mask = self.paths_mask[idx]
            
            patient_id = Path(path_mri).parts[-2]

            img_mri  = nibabel.load(path_mri)
            array_mri = np.rot90(img_mri.get_fdata(), k=2, axes=(0,1)) # [TODO: Why am I doing this? this is rotation by 180degrees!]
            self.mri_max.append(np.max(array_mri))

            img_mask  = nibabel.load(path_mask)
            array_mask = np.rot90(img_mask.get_fdata(), k=2, axes=(0,1))
            
            meta1 = [idx, array_mri.shape[0], array_mri.shape[1], array_mri.shape[2], self.SPACING[0]*100, self.SPACING[1]*100, self.SPACING[2]*100,0,0,0,0,0] # (idx)+(dims)+(spacing)+(augmentations)
            meta2 = patient_id

            if 0:
                print (' - [DEBUG][{}] patient: {} || mri: {}, mask: {}, meta1: {}, meta2: {}'.format(self.name, patient_id, array_mri.shape, array_mask.shape, meta1, meta2))

            yield (
                tf.expand_dims(tf.cast(array_mri, dtype=tf.float32), axis=-1)
                , tf.expand_dims(tf.cast(array_mask, dtype=tf.uint8), axis=-1)
                , tf.cast(meta1, dtype=tf.int32)
                , patient_id
            ) 

    def plot(self, X, Y, meta2, slice_id, binary_mask=False):

        try:
            
            # Step 0.1 - Init some vars
            batch_size = X.shape[0]
            f,axarr = plt.subplots(3, batch_size, gridspec_kw = {'wspace':0.05, 'hspace':0.01}, dpi=200)
            X = np.array(X)
            Y = np.array(Y)

            # Step 0.2 - Init some contour plotting params
            SMOOTHING = 20 # contour smoothing condition
            CONTOUR_PERC_POINTS = 0.9
            CONTOUR_GT    = [(0,255,0), (0,0,255), (255,0,0), (241, 85, 230)]
            CONTOUR_ALPHA = 0.5
            CONTOUR_WIDTH = 0.75

            def get_smooth_contour(contour):
                # https://gist.github.com/shubhamwagh/b8148e65a8850a974efd37107ce3f2ec
                x = contour[0][:,0,0].tolist()
                y = contour[0][:,0,1].tolist()
                tck, u       = splprep([contour[0][:,0,0].tolist(), contour[0][:,0,1].tolist()], u=None, s=SMOOTHING, per=0) # higher the s value, more the smoothing
                u_new        = np.linspace(u.min(), u.max(), int(len(x) * CONTOUR_PERC_POINTS))
                x_new, y_new = splev(u_new, tck, der=0)
                contour_new  = np.array([[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)])
                return contour_new

            # Step 1 - Plot
            for batch_id in range(batch_size):
                axarr[0][batch_id].imshow(X[batch_id, :,:, slice_id-1], cmap='gray'); axarr[0][batch_id].set_title(meta2[batch_id].numpy().decode('utf-8'))
                axarr[1][batch_id].imshow(X[batch_id, :,:, slice_id], cmap='gray')
                axarr[2][batch_id].imshow(X[batch_id, :,:, slice_id+1], cmap='gray')

                if batch_id == 0:
                    axarr[0][batch_id].set_ylabel('Slice: ' + str(slice_id-1))
                    axarr[1][batch_id].set_ylabel('Slice: ' + str(slice_id))
                    axarr[2][batch_id].set_ylabel('Slice: ' + str(slice_id+1))

                for id_, slice_id_ in enumerate([slice_id-1, slice_id, slice_id+1]):
                    if not binary_mask:
                        for label_id in np.unique(Y[batch_id,:,:,slice_id_]):
                            if label_id > 0: # assuming background label is 0
                                contours_gt, _    = cv2.findContours((Y[batch_id,:,:,slice_id_] == label_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # [cv2.RETR_EXTERNAL, cv2.RETR_CCOMP]
                                for contour_gt_ in contours_gt:
                                    if len(contour_gt_) > 2:
                                        coords_gt     = get_smooth_contour([contour_gt_])
                                        coords_gt     = np.append(coords_gt, [coords_gt[0]], axis=0)
                                        coords_gt     = coords_gt[:,0,:].tolist()
                                        xs_gt, ys_gt  = zip(*coords_gt)
                                        axarr[id_][batch_id].plot(xs_gt,ys_gt    , color=np.array(CONTOUR_GT[int(label_id)-1])/255.0, linewidth=CONTOUR_WIDTH, alpha=CONTOUR_ALPHA)
                    else:
                        axarr[id_][batch_id].imshow(Y[batch_id,:,:,slice_id_], alpha=0.2)

            
            # _ = [axarr[0][ax_id].axis('off') for ax_id in range(len(axarr[0]))]
            # _ = [axarr[1][ax_id].axis('off') for ax_id in range(len(axarr[1]))]
            # _ = [axarr[2][ax_id].axis('off') for ax_id in range(len(axarr[2]))]
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.suptitle('Slices=({},{},{})'.format(slice_id-1, slice_id, slice_id+1))
            plt.show()

        except:
            traceback.print_exc()
            pdb.set_trace()

if __name__ == "__main__":

    # Step 0 - Init
    path_file = Path(__file__)
    path_data = Path(path_file).parent.absolute().parent.absolute().parent.absolute().parent.absolute().joinpath('_data')
    print (' - path_data: ', path_data)
    
    batch_size = 2
    epochs     = 1
    # transforms = []
    

    if Path(path_data).exists():

        # Step 1 - Create dataset object and generator
        dataset_prostatex = ProstateXDataset(path_data=path_data
                                    , patient_shuffle=False
                                    , transforms=[]
                                    , single_patient=False
                                    , parallel_calls=1
                                    )
        transforms = [
            # MinMaxNormalizer()
            # , Translate(label_map=dataset_prostatex.LABEL_MAP, translations=[40,40], prob=0.5)
            # , Rotate3D(label_map=dataset_prostatex.LABEL_MAP, angle_degrees=15, prob=0.5)
            # , Noise(x_shape=[dataset_prostatex.VOXELS_X_MIDPT*2, dataset_prostatex.VOXELS_Y_MIDPT*2, dataset_prostatex.VOXELS_Z_MAX,1], prob=0.5)
            # , Mask(label_map=dataset_prostatex.LABEL_MAP, img_size=(dataset_prostatex.VOXELS_X_MIDPT*2, dataset_prostatex.VOXELS_Y_MIDPT*2, dataset_prostatex.VOXELS_Z_MAX,1), prob=0.5)
            # , Equalize(prob=0.5)
            # , ImageQuality(prob=0.5)
            # , Shadow(prob=0.5)
        ]
        dataset_prostatex.transforms = transforms
        datagen_prostatex = dataset_prostatex.generator().repeat().shuffle(1).batch(batch_size).apply(tf.data.experimental.prefetch_to_device(device='/GPU:0', buffer_size=1))

        # Step 2 - Loop over dataset
        epoch_step = 0
        epoch = 0
        for (X,Y,meta1,meta2) in datagen_prostatex:
            
            # Step 2.1 - Init
            if epoch_step == 0:
                pbar = tqdm.tqdm(total=len(dataset_prostatex), desc='')

            # Step 2.2 - Core of the loop
            # print (' - X:', X.shape, 'Y:', Y.shape, meta1, meta2)

            # Step 2.3 - Visual Debug
            if 1:
                # dataset_prostatex.plot(X,Y,meta2, slice_id=random.randint(0+2, 28-2), binary_mask=False)
                dataset_prostatex.plot(X,Y,meta2, slice_id=18, binary_mask=False)
                # pdb.set_trace()
            
            # Step 2.4 - Ending of each iteration of the loop
            epoch_step += batch_size
            pbar.update(batch_size)

            # Step 2.4 - When epoch ends
            if epoch_step >= len(dataset_prostatex):
                pbar.close()
                epoch_step = 0
                epoch += 1

                if epoch >= epochs:
                    break

        if 1:
            plt.hist(dataset_prostatex.mri_max); plt.title('ProstateX'); plt.show()

        pdb.set_trace()
