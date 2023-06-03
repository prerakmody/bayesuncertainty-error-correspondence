# Import private libraries
# 

# Import public libraries
import os
import pdb
import tqdm
import random
import nibabel # pip install nibabel
import traceback
import numpy as np
import SimpleITK as sitk # pip install SimpleITK
from pathlib import Path
import matplotlib.pyplot as plt

# Libraries for smoothing contours
import cv2
from scipy import ndimage
from scipy.interpolate import splprep, splev

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" # to avoid large "Kernel Launch Time"
import tensorflow as tf

#############################################################
#                       AUGMENTATIONS                       #
#############################################################

class MinMaxNormalizer:

    def __init__(self, min=0., max=200., prob=0.3):

        self.min  = 0
        self.max  = 200
        self.prob = prob
    
    @tf.function
    def execute(self, x, y, meta1, meta2): 

        x = (x - self.min) / (self.max - self.min) 

        return x, y, meta1, meta2

#############################################################
#                           UTILS                           #
#############################################################
def read_nii(path_nii):
    img = nibabel.load(path_nii)
    return img.headers, img.get_fdata(), img.affine

def read_img(path_img):
    return sitk.ReadImage(str(path_img))

def sitk_to_array(img, array_bool=False):
    if not array_bool:
        array = sitk.GetArrayFromImage(img)
    else:
        array = img
    return np.moveaxis(array, [0,1,2], [2,1,0]) # [D,W,H] --> [H,W,D] 

def array_to_sitk(array):
    return sitk.GetImageFromArray(np.moveaxis(array, [0,1,2], [2,1,0]).copy())  # [H,W,D] --> [D,W,H]  

def save_img(path_img, array_img, array_spacing, array_origin):
    img = array_to_sitk(array_img)
    img.SetOrigin(tuple(array_origin))
    img.SetSpacing(tuple(array_spacing))
    sitk.WriteImage(img, str(path_img), useCompression=True)

#############################################################
#                         DATASET                           #
#############################################################

class ProstateMedDecDataset:
    """
    Data downloader from: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
    """
    def __init__(self, path_data
                    , transforms=[], patient_shuffle=False
                    , parallel_calls=4, deterministic=False
                    , single_patient=False
                    , download=True):
        
        # Step 0 - Init names
        self.name = 'Pros_MedicalDecathlon'
        self.DIRNAME_TMP  = '_tmp'
        self.DIRNAME_ZIPS = '_zips'
        self.DIRNAME_RAW  = 'raw'
        # self.DIRNAME_PROCESSED = 'processed' # download, resampling and cropping
        self.DIRNAME_PROCESSED = 'processed3' # after separate pytorch script that does histogram normalization
        if self.DIRNAME_PROCESSED == 'processed3':
            self.download = False
        else:
            self.download = download

        print (' - [ProstateMedDecDataset] self.DIRNAME_PROCESSED: ', self.DIRNAME_PROCESSED)

        self.FOLDERNAME_ZIP      = 'Task05_Prostate' # after unzipping the downloaded file
        self.FOLDERNAME_ZIP_MRI  = 'imagesTr'
        self.FOLDERNAME_ZIP_MASK = 'labelsTr'
        self.FILENAME_ZIP_PRE    = 'prostate_'

        self.PATIENT_PRE        = 'ProsMedDec-'
        self.FOLDERNAME_PATIENT = self.PATIENT_PRE + '{:02d}'
        self.FILENAME_MRI       = self.FOLDERNAME_PATIENT + '_mri.nrrd'
        self.FILENAME_MASK      = self.FOLDERNAME_PATIENT + '_mask.nrrd'

        self.EXT_NII = '.nii.gz'

         # Step 0.2 - Values fixed for dataset by me
        self.LABEL_MAP    = {'Prostate': 1}
        self.LABEL_COLORS = {1: [0,255,0]}
        self.SPACING      = (0.5, 0.5, 3.0)
        self.VOXELS_X_MIDPT = 100
        self.VOXELS_Y_MIDPT = 100
        self.VOXELS_Z_MIDPT = 14

        # Step 1 - Init params
        self.path_data   = path_data
        self.transforms = transforms
        self.patient_shuffle = patient_shuffle
        self.single_patient = single_patient
        
        self.parallel_calls = parallel_calls
        self.deterministic  = deterministic

        self.PATIENT_IDS = [0, 1, 2, 4, 6, 7, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47]

        # Step 2 - Init process
        self._get()
    
    def _preprocess_download(self):
        
        # Step 1 - Download data from https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
        pass

        # Step 2 - Unzip data into self.path_dir_zips
        pass

        # Step 3 - Read data of each patient and copy into self.path_dir_raw
        path_folder_unzipped = Path(self.path_dir_zips).joinpath(self.FOLDERNAME_ZIP)
        if Path(path_folder_unzipped).exists():
            path_folder_unzipped_images = Path(path_folder_unzipped).joinpath(self.FOLDERNAME_ZIP_MRI)
            path_folder_unzipped_masks  = Path(path_folder_unzipped).joinpath(self.FOLDERNAME_ZIP_MASK)
            if Path(path_folder_unzipped_images).exists() and Path(path_folder_unzipped_masks).exists():
                
                if 1:
                    for path_patient_img in Path(path_folder_unzipped_images).iterdir():
                        patient_id = int(Path(path_patient_img).parts[-1].split(self.FILENAME_ZIP_PRE)[-1].split(self.EXT_NII)[0])
                        path_folder_raw_patient = Path(self.path_dir_raw).joinpath(self.FOLDERNAME_PATIENT.format(patient_id))
                        Path(path_folder_raw_patient).mkdir(exist_ok=True, parents=True)
                        
                        img_mri = read_img(path_patient_img)
                        img_mri_array = sitk.GetArrayFromImage(img_mri)[0] # [C,D,H,W] --> [D,H,W] (chose t2W from t2w and adc)
                        img_mri_array = sitk_to_array(img_mri_array, array_bool=True)
                        save_img(Path(path_folder_raw_patient).joinpath(self.FILENAME_MRI.format(patient_id))
                                , img_mri_array, img_mri.GetSpacing(), img_mri.GetOrigin())
                    
                if 1:
                    for path_patient_mask in Path(path_folder_unzipped_masks).iterdir():
                        patient_id = int(Path(path_patient_mask).parts[-1].split(self.FILENAME_ZIP_PRE)[-1].split(self.EXT_NII)[0])
                        path_folder_raw_patient = Path(self.path_dir_raw).joinpath(self.FOLDERNAME_PATIENT.format(patient_id))
                        Path(path_folder_raw_patient).mkdir(exist_ok=True, parents=True)

                        img_mask = read_img(path_patient_mask)
                        img_mask_array = sitk_to_array(img_mask)
                        img_mask_array[img_mask_array > 0] = 1 # supress PZ and TZ into 1 label
                        save_img(Path(path_folder_raw_patient).joinpath(self.FILENAME_MASK.format(patient_id))
                                , img_mask_array, img_mask.GetSpacing(), img_mask.GetOrigin())
                
                return True

            else:
                return False

        else:
            return False
    
    def _preprocess_resample(self):

        if 1:
            import math
            import skimage.transform as skTrans

            with tqdm.tqdm(total=len(list(Path(self.path_dir_raw).glob('*'))), desc=' - [{}][Resampling to {}] '.format(self.name, self.SPACING)) as pbar_resample:
                for path_patient in Path(self.path_dir_raw).iterdir():
                    if Path(path_patient).is_dir():
                        patient_id = int(Path(path_patient).parts[-1].split(self.PATIENT_PRE)[-1])
                        path_mri_raw       = Path(path_patient).joinpath(self.FILENAME_MRI.format(patient_id))
                        path_mask_raw      = Path(path_patient).joinpath(self.FILENAME_MASK.format(patient_id))
                        foldername_patient = self.FOLDERNAME_PATIENT.format(patient_id)
                        if Path(path_mri_raw).exists() and Path(path_mask_raw).exists():
                            
                            try:
                                path_img_proc  = Path(self.path_dir_processed).joinpath(foldername_patient, self.FILENAME_MRI.format(patient_id))
                                path_mask_proc = Path(self.path_dir_processed).joinpath(foldername_patient, self.FILENAME_MASK.format(patient_id))
                                Path(path_img_proc).parent.mkdir(exist_ok=True, parents=True)

                                # Step x - resample img_mri
                                if 1:
                                    img_mri      = read_img(path_mri_raw)
                                    spacing_raio = [spacing_old/self.SPACING[id_] for id_, spacing_old in enumerate(img_mri.GetSpacing())]
                                    size_new     = tuple(math.ceil(size_old * spacing_raio[i]) - 1 for i, size_old in enumerate(img_mri.GetSize()))
                                    array_mri_resampled  = skTrans.resize(sitk_to_array(img_mri), size_new, order=3, preserve_range=True) # [H,W,D]
                                    # array_mri_resampled  = array_mri_resampled.astype(np.int16)
                                    # print (' - patient_id: {} || size_new: {}'.format(patient_id, size_new))

                                # Step y - resample img_mask
                                if 1:
                                    img_mask             = read_img(path_mask_raw)
                                    array_mask_resampled = skTrans.resize(sitk_to_array(img_mask), size_new, order=0, preserve_range=True)
                                    array_mask_resampled = array_mask_resampled.astype(np.uint8) # [H,W,D]

                                # Step z - padding/cropping 
                                if 1:    
                                    mask_idxs  = np.argwhere(array_mask_resampled > 0) # [H,W,D]
                                    mask_midpt = np.array([np.mean(mask_idxs[:,0]), np.mean(mask_idxs[:,1]), np.mean(mask_idxs[:,2])]).astype(np.uint8).tolist()
                                    x_left_buffer  = mask_midpt[0]
                                    x_right_buffer = array_mask_resampled.shape[0] - mask_midpt[0]
                                    y_left_buffer  = mask_midpt[1]
                                    y_right_buffer = array_mask_resampled.shape[1] - mask_midpt[1]
                                    z_left_buffer  = mask_midpt[2]
                                    z_right_buffer = array_mask_resampled.shape[2] - mask_midpt[2]

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
                                    
                                    if z_left_buffer < self.VOXELS_Z_MIDPT and z_right_buffer >= self.VOXELS_Z_MIDPT + (self.VOXELS_Z_MIDPT - z_left_buffer):
                                        z_pad_left  = z_left_buffer
                                        z_pad_right = self.VOXELS_Z_MIDPT + (self.VOXELS_Z_MIDPT - z_left_buffer)
                                    elif z_right_buffer < self.VOXELS_Z_MIDPT and z_left_buffer >= self.VOXELS_Z_MIDPT + (self.VOXELS_Z_MIDPT - z_right_buffer):
                                        z_pad_right = z_right_buffer
                                        z_pad_left  = self.VOXELS_Z_MIDPT + (self.VOXELS_Z_MIDPT - z_right_buffer)  
                                    elif z_left_buffer >= self.VOXELS_Z_MIDPT and z_right_buffer >= self.VOXELS_Z_MIDPT:
                                        z_pad_left  = self.VOXELS_Z_MIDPT
                                        z_pad_right = self.VOXELS_Z_MIDPT
                                    else: # need to pad instead of remove
                                        mask_array_height = array_mask_resampled.shape[2]
                                        mask_array_diff   = np.abs(mask_array_height - self.VOXELS_Z_MIDPT * 2)
                                        pad_top           = mask_array_diff//2
                                        pad_bottom        = mask_array_diff - mask_array_diff//2

                                        array_mask_resampled        = np.pad(array_mask_resampled, (pad_top, pad_bottom), 'constant')
                                        array_mask_resampled        = array_mask_resampled[pad_top:-pad_bottom, pad_top:-pad_bottom, :]
                                        array_mri_resampled        = np.pad(array_mri_resampled, (pad_top, pad_bottom), 'constant')
                                        array_mri_resampled        = array_mri_resampled[pad_top:-pad_bottom, pad_top:-pad_bottom, :]

                                        z_pad_left = mask_midpt[2]
                                        z_pad_right = array_mask_resampled.shape[2] - mask_midpt[2]

                                # Step z - Saving
                                if 1:
                                    array_mask_resampled_new = np.array(array_mask_resampled[mask_midpt[0] - x_pad_left:mask_midpt[0] + x_pad_right
                                                        , mask_midpt[1] - y_pad_left:mask_midpt[1] + y_pad_right
                                                        , mask_midpt[2] - z_pad_left:mask_midpt[2] + z_pad_right
                                                    ], copy=True)
                                    array_mri_resampled_new = np.array(array_mri_resampled[mask_midpt[0] - x_pad_left:mask_midpt[0] + x_pad_right
                                                                        , mask_midpt[1] - y_pad_left:mask_midpt[1] + y_pad_right
                                                                        , mask_midpt[2] - z_pad_left:mask_midpt[2] + z_pad_right
                                                                    ], copy=True)
                                    # print (' - [crop/pad][{}] array_mask: {} || array_mask_new: {}'.format(patient_id, array_mask_resampled.shape, array_mask_resampled_new.shape))
                                    # print (' - [crop/pad][patient={}] array_mri : {} || array_mri_new : {}'.format(patient_id, array_mri_resampled.shape, array_mri_resampled_new.shape))

                                    save_img(path_img_proc, array_mri_resampled_new, self.SPACING, img_mri.GetOrigin())
                                    save_img(path_mask_proc, array_mask_resampled_new, self.SPACING, img_mask.GetOrigin())
                                    # pdb.set_trace()
                        
                            except:
                                traceback.print_exc()
                                pdb.set_trace()
                            
                            pbar_resample.update(1)
        
        return True         
        
    def _get(self):

        # Step 0 - Init paths
        self.path_dir_dataset    = Path(self.path_data).joinpath(self.name)
        self.path_dir_tmp        = Path(self.path_dir_dataset).joinpath(self.DIRNAME_TMP)
        self.path_dir_zips       = Path(self.path_dir_dataset).joinpath(self.DIRNAME_ZIPS)
        self.path_dir_raw        = Path(self.path_dir_dataset).joinpath(self.DIRNAME_RAW)
        self.path_dir_processed  = Path(self.path_dir_dataset).joinpath(self.DIRNAME_PROCESSED)

        Path(self.path_dir_tmp).mkdir(exist_ok=True, parents=True)
        Path(self.path_dir_zips).mkdir(exist_ok=True, parents=True)
        Path(self.path_dir_raw).mkdir(exist_ok=True, parents=True)
        Path(self.path_dir_processed).mkdir(exist_ok=True, parents=True)

        if self.download:
            if len(list(Path(self.path_dir_processed).glob('*'))) != len(self.PATIENT_IDS):
                status_download = self._preprocess_download()
                if status_download:
                    status_resample = self._preprocess_resample()
            else:
                status_resample = True
        else:
            status_resample = True
        
        if status_resample:
            # Step 2 - Get all paths
            self.paths_mri, self.paths_mask = [], []
            for patient_id in self.PATIENT_IDS:
                path_patient = Path(self.path_dir_processed).joinpath(self.FOLDERNAME_PATIENT.format(patient_id))
                self.paths_mri.append(Path(path_patient).joinpath(self.FILENAME_MRI.format(patient_id)))
                self.paths_mask.append(Path(path_patient).joinpath(self.FILENAME_MASK.format(patient_id)))

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
                idx = 0

            path_mri  = self.paths_mri[idx]
            path_mask = self.paths_mask[idx]

            patient_id_str = Path(path_mri).parts[-2]

            img_mri        = sitk.ReadImage(str(path_mri))
            array_mri      = sitk_to_array(img_mri) # [D,W,H] --> [H,W,D]
            array_mri      = np.rot90(array_mri, k=2, axes=(0,1))
            self.mri_max.append(np.max(array_mri))
            
            img_mask       = sitk.ReadImage(str(path_mask))
            array_mask     = sitk_to_array(img_mask) # [D,W,H] --> [H,W,D]
            array_mask     = np.rot90(array_mask, k=2, axes=(0,1))
            
            meta1          = [idx, array_mri.shape[0], array_mri.shape[1], array_mri.shape[2], self.SPACING[0]*100, self.SPACING[1]*100, self.SPACING[2]*100,0,0,0,0,0] # (idx)+(dims)+(spacing)+(augmentations)
            meta2          = patient_id_str

            if 0:
                print (' - [DEBUG][{}] patient: {} || mri: {}, mask: {}'.format(self.name, patient_id_str, array_mri.shape, array_mask.shape))

            yield (
                tf.expand_dims(tf.cast(array_mri, dtype=tf.float32), axis=-1)
                , tf.expand_dims(tf.cast(array_mask, dtype=tf.uint8), axis=-1)
                , tf.cast(meta1, dtype=tf.int32)
                , meta2
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
    print ('\n - path_data: ', path_data, '\n')
    
    batch_size = 2
    epochs     = 1
    # transforms = []

    if Path(path_data).exists():
        
        # Step 1 - Create dataset object and generator
        dataset_prostatemeddec = ProstateMedDecDataset(path_data=path_data
                                    , patient_shuffle=False
                                    , transforms=[]
                                    , single_patient=False
                                    , parallel_calls=1
                                    )
        transforms = [
            MinMaxNormalizer(prob=1.0)
        ]

        dataset_prostatemeddec.transforms = transforms
        datagen_prostatemeddec = dataset_prostatemeddec.generator().repeat().shuffle(1).batch(batch_size).apply(tf.data.experimental.prefetch_to_device(device='/GPU:0', buffer_size=1))

        # Step 2 - Loop over dataset
        epoch_step = 0
        epoch = 0
        for (X,Y,meta1,meta2) in datagen_prostatemeddec:
            
            # Step 2.1 - Init
            if epoch_step == 0:
                pbar = tqdm.tqdm(total=len(dataset_prostatemeddec), desc='')

            # Step 2.2 - Core of the loop
            print (' - X:', X.shape, 'Y:', Y.shape, meta1, meta2)
            if 1:
                # dataset_prostatemeddec.plot(X,Y,meta2, slice_id=random.randint(0+2, 28-2), binary_mask=True)
                dataset_prostatemeddec.plot(X,Y,meta2, slice_id=18, binary_mask=False)
                # pdb.set_trace()
            
            # Step 2.3 - Ending of each iteration of the loop
            epoch_step += batch_size
            pbar.update(batch_size)

            # Step 2.4 - When epoch ends
            if epoch_step >= len(dataset_prostatemeddec):
                pbar.close()
                epoch_step = 0
                epoch += 1

                if epoch >= epochs:
                    break

        if 1:
            plt.hist(dataset_prostatemeddec.mri_max); plt.title('ProstateMedDec'); plt.show()
        
        pdb.set_trace()

    else:
        print (' - ERROR: path_data not exists: ', path_data)