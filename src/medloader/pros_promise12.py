# Import private libraries
#

# Import public libraries
import os
import pdb
import tqdm
import math
import random
import shutil
import urllib
import zipfile
import traceback
import numpy as np
import urllib.request
import skimage.transform
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Libraries for smoothing contours
import cv2
from scipy import ndimage
from scipy.interpolate import splprep, splev

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" # to avoid large "Kernel Launch Time"
import tensorflow as tf

class MinMaxNormalizer:

    def __init__(self, min=0., max=200.):

        self.min  = 0
        self.max  = 200
    
    @tf.function
    def execute(self, x, y, meta1, meta2): 

        x = (x - self.min) / (self.max - self.min) 

        return x, y, meta1, meta2

############################################################
#                    DOWNLOAD RELATED                      #
############################################################

class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_zip(url_zip, filepath_zip, filepath_output, meta=''):
    import urllib
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=' - [{}][Download] {}'.format(meta, str(url_zip.split('/')[-1])) ) as pbar:
        urllib.request.urlretrieve(url_zip, filename=filepath_zip, reporthook=pbar.update_to)
    read_zip(filepath_zip, filepath_output, meta=meta)

def read_zip(filepath_zip, filepath_output=None, leave=False, meta=''):
    
    import zipfile

    # Step 0 - Init
    if Path(filepath_zip).exists():
        if filepath_output is None:
            filepath_zip_parts     = list(Path(filepath_zip).parts)
            filepath_zip_name      = filepath_zip_parts[-1].split('.zip')[0]
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
        print (' - [{}][ERROR][utils.read_zip()] Path does not exist: {}'.format(meta,filepath_zip) )
        return None

#############################################################
#                           UTILS                           #
#############################################################
def sitk_to_array(img):
    return np.moveaxis(sitk.GetArrayFromImage(img), [0,1,2], [2,1,0]) # [H,W,D] --> [D,W,H]

def array_to_sitk(array):
    return sitk.GetImageFromArray(np.moveaxis(array, [0,1,2], [2,1,0]).copy())  # [D,W,H] --> [H,W,D] 

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
    t = (len(imsd)*t+0.5).astype(np.int)
    return np.unique(imsd[t])

def hist_scaled(array, brks=None, bins=100):
    """
    Scales a tensor using `freqhist_bins` to values between 0 and 1

    Ref: https://github.com/AIEMMU/MRI_Prostate/blob/master/Pre%20processing%20Promise12.ipynb
    Ref: https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py
    """
    
    if brks is None: 
        brks = freqhist_bins(array, bins=bins)
    ys = np.linspace(0., 1., len(brks))

    if 0:
        plt.plot(brks, ys, color='blue', label='Non-Normalized', marker='x', markersize=2)
        plt.xlim([0,np.max(array)])
        plt.xlabel('Boundaries (of intensity values) such that each bin has equivalent count of values \n i.e. Equal Mass Bins')
        plt.show()

        # hist_xs = plt.hist(array.flatten(), bins=bins)[1]
        # hist_ys = np.linspace(0., 1., len(hist_xs))
        # plt.xlabel('Boundaries (of intensity values) such that each bin has equivalent count of values \n i.e. Equal Mass Bins')
        # plt.plot(hist_xs, hist_ys, color='orange', label='Equal Width Bins')
        # plt.plot(brks   , ys     , color='blue'  , label='Equal Mass Bins' )
        # plt.legend()
        # plt.show()

        pdb.set_trace()

    x = array.flatten()
    x = np.interp(x, brks, ys)
    array_tmp = np.reshape(x, array.shape)
    array_tmp = np.clip(array_tmp, 0.,1.)
    return array_tmp


#############################################################
#                         DATASET                           #
#############################################################

class Promise12Dataset:
    """
    Some usnusual cases with rectal pipe: [Case00]
    Links: https://promise12.grand-challenge.org/Download/
    """

    def __init__(self, path_data
                    , transforms=[], patient_shuffle=False
                    , parallel_calls=4, deterministic=False
                    , download=True
                    , single_patient=False):
        
        # Step 0 - Init names
        self.name = 'Pros_Promise12'
        self.DIRNAME_TMP = '_tmp'
        self.DIRNAME_RAW = 'raw'
        self.DIRNAME_PROCESSED = 'processed'
        # self.DIRNAME_PROCESSED2 = 'processed2'
        self.DIRNAME_PROCESSED2 = 'processed3' # HistStandardization
        if download:
            print (' - [Promise12Dataset] self.DIRNAME_PROCESSED2: ', self.DIRNAME_PROCESSED2)

        self.FOLDERNAME_PATIENT      = 'Case{:02d}'
        self.FILENAME_MRI_PROMISE12  = self.FOLDERNAME_PATIENT + '{}'
        self.FILENAME_MASK_PROMISE12 = self.FOLDERNAME_PATIENT + '_segmentation{}'

        self.EXT_MHD  = '.mhd'
        self.EXT_NRRD = '.nrrd' 

        # Step 0.2 - Values fixed for dataset by me
        self.LABEL_MAP    = {'Prostate': 1}
        self.LABEL_COLORS = {1: [0,255,0]}
        self.VOXELS_X_MIDPT = 100
        self.VOXELS_Y_MIDPT = 100
        self.VOXELS_Z_MIDPT = 14
        self.PATIENT_COUNT = 50
        self.SPACING = (0.5, 0.5, 3.0)

        # Step 1 - Init params
        self.path_data   = path_data
        self.transforms = transforms
        self.patient_shuffle = patient_shuffle
        self.single_patient = single_patient
        
        self.parallel_calls = parallel_calls
        self.deterministic  = deterministic

        # Step 2 - Init process
        if download:
            self._get()

    def _preprocess_download(self):
        
        # Step 1 - Download and unzip folders to DIR_TMP
        if 1:
            urls_zip = [
                'https://www.dropbox.com/s/1d8x8dy0pfauk89/TrainingData_Part1.zip?dl=1'
                ,'https://www.dropbox.com/s/g9tnmam7cmk3khx/TrainingData_Part2.zip?dl=1'
                ,'https://www.dropbox.com/s/5mzbrk480uu7i5h/TrainingData_Part3.zip?dl=1'
            ]

            import concurrent
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for url_zip in urls_zip:
                    filepath_zip = Path(self.path_dir_tmp, url_zip.split('/')[-1].split('?')[0])
                    # path_folder_op = Path(self.path_dir_tmp, url_zip.split('/')[-1].split('?')[0].split('.zip'))
                    # path_folders_op.append()
                    
                    # Step 1.1 - Download .zip and then unzip it
                    if not Path(filepath_zip).exists():
                        executor.submit(download_zip, url_zip, filepath_zip, self.path_dir_tmp, meta=self.name)
                    else:
                        executor.submit(read_zip, filepath_zip, self.path_dir_tmp, meta=self.name)
        
        # Step 2 - Move files from unzipped folder to self.path_dir_raw
        if 1:
            try:
                path_files = [list(Path(self.path_dir_tmp).glob(ext)) for ext in ['*.mhd', '*.raw']]
                path_files = [tmp_path for tmp_paths in path_files for tmp_path in tmp_paths]
                for path_file in path_files:
                    path_file_op = list(Path(path_file).parts)
                    path_file_op[-2] = self.DIRNAME_RAW 
                    path_file_op = Path(*path_file_op)
                    shutil.move(src=str(path_file), dst=str(path_file_op))

            except:
                traceback.print_exc()
                pdb.set_trace()

        return True
    
    def _preprocess_resample(self):

        try:

            import skimage.transform as skTrans 
            
            with tqdm.tqdm(total=self.PATIENT_COUNT, desc=' - [{}][Resampling]'.format(self.name)) as pbar_resample:
                for patient_id in range(self.PATIENT_COUNT ):
                    
                    try:
                        # Step 3.1.1 - Read mri
                        path_patient_mri_raw  = Path(self.path_dir_raw).joinpath(self.FILENAME_MRI_PROMISE12.format(patient_id, self.EXT_MHD))
                        path_patient_mri_proc = Path(self.path_dir_processed).joinpath(self.FOLDERNAME_PATIENT.format(patient_id), self.FILENAME_MRI_PROMISE12.format(patient_id, self.EXT_NRRD))
                        img_mri               = sitk.ReadImage(str(path_patient_mri_raw))
                        array_mri             = np.moveaxis(sitk.GetArrayFromImage(img_mri), [0,1,2], [2,1,0]) # [D,W,H] --> [H,W,D]

                        # Step 3.1.2 - Calculate new size and resample        
                        ratio    = [spacing_old / self.SPACING[i] for i, spacing_old in enumerate(img_mri.GetSpacing())]
                        size_new = tuple(math.ceil(size_old * ratio[i]) - 1 for i, size_old in enumerate(img_mri.GetSize()))
                        array_mri_resampled = skTrans.resize(array_mri, size_new, order=3, preserve_range=True)
                        array_mri_resampled = array_mri_resampled.astype(np.int16)

                        # Step 3.1.3 - Save resampled mri
                        img_mri_resampled = sitk.GetImageFromArray(np.moveaxis(array_mri_resampled, [0,1,2], [2,1,0]).copy()) # [H,W,D] --> [D,W,H]
                        img_mri_resampled.SetOrigin(tuple(img_mri.GetOrigin()))
                        img_mri_resampled.SetSpacing(tuple(self.SPACING))
                        Path(path_patient_mri_proc).parent.mkdir(exist_ok=True, parents=True)
                        sitk.WriteImage(img_mri_resampled, str(path_patient_mri_proc), useCompression=True)

                        # Step 3.2.1 - Read mask
                        path_patient_mask_raw  = Path(self.path_dir_raw).joinpath(self.FILENAME_MASK_PROMISE12.format(patient_id, self.EXT_MHD))
                        path_patient_mask_proc = Path(self.path_dir_processed).joinpath(self.FOLDERNAME_PATIENT.format(patient_id), self.FILENAME_MASK_PROMISE12.format(patient_id, self.EXT_NRRD))
                        img_mask               = sitk.ReadImage(str(path_patient_mask_raw))
                        array_mask             = np.moveaxis(sitk.GetArrayFromImage(img_mask), [0,1,2], [2,1,0]) # [D,W,H] --> [H,W,D]
                        array_mask = array_mask.astype(np.float32) # [TODO: This is a weird step]

                        # Step 3.2.2 - Calculate new size and resample
                        array_mask_resampled = skTrans.resize(array_mask, size_new, order=0, preserve_range=True) # 0 = Nearest Neighbour
                        array_mask_resampled = array_mask_resampled.astype(np.int8)
                        # print (' - [{}][resampling] Patient:{} | sum(array_mask_resampled):{}, unique(array_mask_resampled):{}'.format(self.name, patient_id, np.sum(array_mask_resampled), np.unique(array_mask_resampled)))
                        if np.sum(array_mask_resampled) == 0:
                            print (np.sum(array_mask), array_mask.dtype, array_mask.shape)
                            pdb.set_trace()

                        # Step 3.1.3 - Save resampled mask
                        img_mask_resampled = sitk.GetImageFromArray(np.moveaxis(array_mask_resampled, [0,1,2], [2,1,0]).copy()) # [H,W,D] --> [D,W,H]
                        img_mask_resampled.SetOrigin(tuple(img_mask.GetOrigin()))
                        img_mask_resampled.SetSpacing(tuple(self.SPACING))
                        Path(path_patient_mask_proc).parent.mkdir(exist_ok=True, parents=True)
                        sitk.WriteImage(img_mask_resampled, str(path_patient_mask_proc), useCompression=True)
                    
                    except:
                        traceback.print_exc()
                        pdb.set_trace()

                    pbar_resample.update(1)

            return True

        except:
            traceback.print_exc()
            pdb.set_trace()
            return False

    def _preprocess_crop(self):

        try:
            
            mri_maxvals = []
            with tqdm.tqdm(total=self.PATIENT_COUNT, desc=' - [{}][Crop/IntensityOps]'.format(self.name)) as pbar_crop:
                for patient_id in range(self.PATIENT_COUNT):
                    
                    # Step 5.1 - Get mask paths
                    path_patient_mri_proc  = Path(self.path_dir_processed).joinpath(self.FOLDERNAME_PATIENT.format(patient_id), self.FILENAME_MRI_PROMISE12.format(patient_id, self.EXT_NRRD))
                    path_patient_mask_proc = Path(self.path_dir_processed).joinpath(self.FOLDERNAME_PATIENT.format(patient_id), self.FILENAME_MASK_PROMISE12.format(patient_id, self.EXT_NRRD))

                    # Step 5.2 - Get mask (crop and pad)
                    img_mri    = sitk.ReadImage(str(path_patient_mri_proc))
                    array_mri  = sitk_to_array(img_mri) # [D,W,H] --> [H,W,D]
                    mri_maxval = np.max(array_mri)
                    mri_maxvals.append(mri_maxval)
                    array_mri  = hist_scaled(array_mri, bins=10000)

                    img_mask   = sitk.ReadImage(str(path_patient_mask_proc))
                    array_mask = sitk_to_array(img_mask) # [D,W,H] --> [H,W,D]    
                    mask_idxs  = np.argwhere(array_mask > 0)
                    mask_midpt = np.array([np.mean(mask_idxs[:,0]), np.mean(mask_idxs[:,1]), np.mean(mask_idxs[:,2])]).astype(np.uint8).tolist()
                    
                    if 0:
                        print (array_mri.shape, np.min(array_mri), np.max(array_mri))
                        array_mri_scaled = hist_scaled(array_mri, bins=100)
                        print (array_mri_scaled.shape, np.min(array_mri_scaled), np.max(array_mri_scaled))
                        import matplotlib.pyplot as plt
                        f,axarr = plt.subplots(1,2); axarr[0].imshow(array_mri[:,:,18], cmap='gray', vmin=0, vmax=12000); axarr[1].imshow(hist_scaled(array_mri, bins=100)[:,:,18], cmap='gray', vmin=0., vmax=1.); plt.show(block=False)
                        f,axarr = plt.subplots(1,2); axarr[0].imshow(array_mri[:,:,18], cmap='gray', vmin=0, vmax=12000); axarr[1].imshow(hist_scaled(array_mri, bins=1000)[:,:,18], cmap='gray', vmin=0., vmax=1.); plt.show(block=False)
                        f,axarr = plt.subplots(1,2); axarr[0].imshow(array_mri[:,:,18], cmap='gray', vmin=0, vmax=12000); axarr[1].imshow(hist_scaled(array_mri, bins=5000)[:,:,18], cmap='gray', vmin=0., vmax=1.); plt.show(block=False)
                        pdb.set_trace()

                    # Step 5.3 - Crop
                    x_left_buffer = mask_midpt[0]
                    x_right_buffer = array_mask.shape[0] - mask_midpt[0]
                    y_left_buffer = mask_midpt[1]
                    y_right_buffer = array_mask.shape[1] - mask_midpt[1]
                    z_left_buffer = mask_midpt[2]
                    z_right_buffer = array_mask.shape[2] - mask_midpt[2]

                    if x_left_buffer < self.VOXELS_X_MIDPT:
                        x_pad_left  = x_left_buffer
                        x_pad_right = self.VOXELS_X_MIDPT + (self.VOXELS_X_MIDPT - x_left_buffer)
                    elif x_right_buffer < self.VOXELS_X_MIDPT:
                        x_pad_right = x_right_buffer
                        x_pad_left  = self.VOXELS_X_MIDPT + (self.VOXELS_X_MIDPT - x_right_buffer)
                    else:
                        x_pad_left  = self.VOXELS_X_MIDPT
                        x_pad_right = self.VOXELS_X_MIDPT

                    if y_left_buffer < self.VOXELS_Y_MIDPT:
                        y_pad_left  = y_left_buffer
                        y_pad_right = self.VOXELS_Y_MIDPT + (self.VOXELS_Y_MIDPT - y_left_buffer)
                    elif y_right_buffer < self.VOXELS_Y_MIDPT:
                        y_pad_right = y_right_buffer
                        y_pad_left  = self.VOXELS_Y_MIDPT + (self.VOXELS_Y_MIDPT - y_right_buffer)  
                    else:
                        y_pad_left  = self.VOXELS_Y_MIDPT
                        y_pad_right = self.VOXELS_Y_MIDPT
                    
                    if z_left_buffer < self.VOXELS_Z_MIDPT and z_right_buffer >= self.VOXELS_Z_MIDPT +  (self.VOXELS_Z_MIDPT - z_left_buffer):
                        z_pad_left  = z_left_buffer
                        z_pad_right = self.VOXELS_Z_MIDPT +  (self.VOXELS_Z_MIDPT - z_left_buffer)
                    elif z_right_buffer < self.VOXELS_Z_MIDPT and z_left_buffer >= self.VOXELS_Z_MIDPT +  (self.VOXELS_Z_MIDPT - z_right_buffer):
                        z_pad_right = z_right_buffer
                        z_pad_left  = self.VOXELS_Z_MIDPT +  (self.VOXELS_Z_MIDPT - z_right_buffer)  
                    elif z_left_buffer >= self.VOXELS_Z_MIDPT and z_right_buffer >= self.VOXELS_Z_MIDPT:
                        z_pad_left  = self.VOXELS_Z_MIDPT
                        z_pad_right = self.VOXELS_Z_MIDPT
                    else: # need to pad instead of remove
                        mask_array_height = array_mask.shape[2]
                        mask_array_diff   = np.abs(mask_array_height - self.VOXELS_Z_MIDPT * 2)
                        pad_top           = mask_array_diff//2
                        pad_bottom        = mask_array_diff - mask_array_diff//2
                        array_mask        = np.pad(array_mask, (pad_top, pad_bottom), 'constant')
                        array_mask        = array_mask[pad_top:-pad_bottom, pad_top:-pad_bottom, :]
                        array_mri        = np.pad(array_mri, (pad_top, pad_bottom), 'constant')
                        array_mri        = array_mri[pad_top:-pad_bottom, pad_top:-pad_bottom, :]

                        z_pad_left = mask_midpt[2]
                        z_pad_right = array_mask.shape[2] - mask_midpt[2]
                        
                    array_mask_new = np.array(array_mask[mask_midpt[0] - x_pad_left:mask_midpt[0] + x_pad_right
                                                        , mask_midpt[1] - y_pad_left:mask_midpt[1] + y_pad_right
                                                        , mask_midpt[2] - z_pad_left:mask_midpt[2] + z_pad_right
                                                    ], copy=True)
                    array_mri_new = np.array(array_mri[mask_midpt[0] - x_pad_left:mask_midpt[0] + x_pad_right
                                                        , mask_midpt[1] - y_pad_left:mask_midpt[1] + y_pad_right
                                                        , mask_midpt[2] - z_pad_left:mask_midpt[2] + z_pad_right
                                                    ], copy=True)
                    print (' - [crop/pad][{}] array_mask: {} || array_mask_new: {}'.format(patient_id, array_mask.shape, array_mask_new.shape))
                    print (' - [crop/pad][{}] array_mri : {} || array_mri_new : {} | maxval: {}'.format(patient_id, array_mri.shape, array_mri_new.shape, mri_maxval))

                    # Step 5.4 - Save mask
                    img_mask_cropped = array_to_sitk(array_mask_new)
                    img_mask_cropped.SetOrigin(tuple(img_mask.GetOrigin()))
                    img_mask_cropped.SetSpacing(tuple(img_mask.GetSpacing()))
                    path_patient_mask_proc2 = Path(self.path_dir_processed2).joinpath(self.FOLDERNAME_PATIENT.format(patient_id), self.FILENAME_MASK_PROMISE12.format(patient_id, self.EXT_NRRD))
                    Path(path_patient_mask_proc2).parent.mkdir(exist_ok=True, parents=True)
                    sitk.WriteImage(img_mask_cropped, str(path_patient_mask_proc2), useCompression=True)

                    img_mri_cropped = array_to_sitk(array_mri_new)
                    img_mri_cropped.SetOrigin(tuple(img_mri.GetOrigin()))
                    img_mri_cropped.SetSpacing(tuple(img_mri.GetSpacing()))
                    path_patient_mri_proc2 = Path(self.path_dir_processed2).joinpath(self.FOLDERNAME_PATIENT.format(patient_id), self.FILENAME_MRI_PROMISE12.format(patient_id, self.EXT_NRRD))
                    Path(path_patient_mri_proc2).parent.mkdir(exist_ok=True, parents=True)
                    sitk.WriteImage(img_mri_cropped, str(path_patient_mri_proc2), useCompression=True)
                
                    pbar_crop.update(1)

            # if len(mri_maxvals):
            #     import matplotlib.pyplot as plt
            #     plt.hist(mri_maxvals)
            #     plt.title('PROMISE12: Max Intensity Vals \n Histogram showing the need for intensity transformations in MRI')
            #     plt.show()

            return True
        
        except:
            print (' - [{}][ERROR] in _preprocess_crop() '.format(self.name))
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
            if status_download:
                print ('')
                status_resample = self._preprocess_resample()
                if status_resample:
                    print ('')
                    self.status_preprocessed = self._preprocess_crop()
        
        else:
            self.status_preprocessed = True
                
        # Step 2 - Get all paths
        self.paths_mri, self.paths_mask = [], []
        for patient_id in range(self.PATIENT_COUNT):
            path_patient = Path(self.path_dir_processed2).joinpath(self.FOLDERNAME_PATIENT.format(patient_id))
            self.paths_mri.append(Path(path_patient).joinpath(self.FILENAME_MRI_PROMISE12.format(patient_id, self.EXT_NRRD)))
            self.paths_mask.append(Path(path_patient).joinpath(self.FILENAME_MASK_PROMISE12.format(patient_id, self.EXT_NRRD)))
    
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
                idxs = 0

            path_mri  = self.paths_mri[idx]
            path_mask = self.paths_mask[idx]

            patient_id = Path(path_mri).parts[-2]

            img_mri  = sitk.ReadImage(str(path_mri))
            array_mri = sitk_to_array(img_mri) # [D,W,H] --> [H,W,D]
            self.mri_max.append(np.max(array_mri))

            img_mask  = sitk.ReadImage(str(path_mask))
            array_mask = sitk_to_array(img_mask) # [D,W,H] --> [H,W,D]
            
            meta1 = [idx, array_mri.shape[0], array_mri.shape[1], array_mri.shape[2], self.SPACING[0]*100, self.SPACING[1]*100, self.SPACING[2]*100,0,0,0,0,0] # (idx)+(dims)+(spacing)+(augmentations)
            meta2 = patient_id

            if 0:
                print (' - [DEBUG][{}] patient: {} || mri: {}, mask: {}'.format(self.name, patient_id, array_mri.shape, array_mask.shape))

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

            
            _ = [axarr[0][ax_id].axis('off') for ax_id in range(len(axarr[0]))]
            _ = [axarr[0][ax_id].axis('off') for ax_id in range(len(axarr[1]))]
            _ = [axarr[0][ax_id].axis('off') for ax_id in range(len(axarr[2]))]
            plt.suptitle('Slices=({},{},{})'.format(slice_id-1, slice_id, slice_id+1))
            plt.show()

        except:
            traceback.print_exc()
            pdb.set_trace()

if __name__ == "__main__":

    # Step 0 - Init
    path_file = Path(__file__)
    path_data = Path(path_file).parent.absolute().parent.absolute().parent.absolute().parent.absolute().joinpath('_data')
    batch_size = 2
    epochs     = 1
    # transforms = []

    if Path(path_data).exists():

        # Step 1 - Create dataset object and generator
        dataset_promise12 = Promise12Dataset(path_data=path_data
                                    , patient_shuffle=False
                                    , transforms=[]
                                    , single_patient=True
                                    )
        transforms = [
            MinMaxNormalizer()
            # Translate(label_map=dataset_prostatex.LABEL_MAP, translations=[40,40], prob=1.0)
            # , Rotate3D(label_map=dataset_prostatex.LABEL_MAP, angle_degrees=15, prob=1.0)
            # , Noise(x_shape=[dataset_prostatex.VOXELS_X_MIDPT*2, dataset_prostatex.VOXELS_Y_MIDPT*2, dataset_prostatex.VOXELS_Z_MAX,1], prob=1.0)
            # , Mask(label_map=dataset_prostatex.LABEL_MAP, img_size=(dataset_prostatex.VOXELS_X_MIDPT*2, dataset_prostatex.VOXELS_Y_MIDPT*2, dataset_prostatex.VOXELS_Z_MAX,1), prob=1.0)
            # Equalize()
        ]
        dataset_promise12.transforms = transforms
        datagen_promise12 = dataset_promise12.generator().repeat().shuffle(1).batch(batch_size).apply(tf.data.experimental.prefetch_to_device(device='/GPU:0', buffer_size=1))

        # Step 2 - Loop over dataset
        epoch_step = 0
        epoch = 0
        for (X,Y,meta1,meta2) in datagen_promise12:
            
            # Step 2.1 - Init
            if epoch_step == 0:
                pbar = tqdm.tqdm(total=len(dataset_promise12), desc='')

            # Step 2.2 - Core of the loop
            print (' - X:', X.shape, 'Y:', Y.shape, meta1, meta2)
            if 1:
                # dataset_promise12.plot(X,Y,meta2, slice_id=random.randint(0+2, 28-2), binary_mask=True)
                dataset_promise12.plot(X,Y,meta2, slice_id=18, binary_mask=False)
                # pdb.set_trace()
            
            # Step 2.3 - Ending of each iteration of the loop
            epoch_step += batch_size
            pbar.update(batch_size)

            # Step 2.4 - When epoch ends
            if epoch_step >= len(dataset_promise12):
                pbar.close()
                epoch_step = 0
                epoch += 1

                if epoch >= epochs:
                    break

        if 1:
            plt.hist(dataset_promise12.mri_max); plt.title('PROMISE12'); plt.show()
        
        pdb.set_trace()
