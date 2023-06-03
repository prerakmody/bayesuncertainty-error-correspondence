# Import private libraries
import medloader.dataloader.utils as utils
import medloader.dataloader.config as config 
from medloader.dataloader.extractors.han_structseg import DICOMVolumizerStructSegData

# Import public libraries
import os
import sys
import pdb
import time
import tqdm
import json
import itertools
import traceback
import numpy as np
from pathlib import Path
import tensorflow as tf


class HaNStructSegDataset:
    """
    Public Ref:
     - https://structseg2019.grand-challenge.org/
     - https://zenodo.org/record/3718885
     - https://drive.google.com/drive/folders/1ijZ6MOw3nedEDOjl3vlZXTBav7AZkcEd (as provided by huangrui@sensetime.com and hsli@ee.cuhk.edu.hk)
    Personal Ref: 
     - the unzipped folder is Task1_HaN_OAR.zip that is unzipped and the following data structure is to be created
        - ./raw/{1...50}
    """
    
    def __init__(self, dataset_dir, fold_id=None, fold_type=None
                    , force_extract=False
                    , spacing=(0.8, 0.8, 2.5), grid=True, crop_init=False, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False, random_grid=False, pregridnorm=False
                    , parallel_calls=None, deterministic=False
                    , patient_shuffle=False
                    , centred_dataloader_prob = 0.0
                    , debug=False, single_sample=False
            ):

        self.name = config.KEY_DATASET_NAME_STRUCTSEG

        # Params - Source 
        self.dataset_dir     = dataset_dir
        self.fold_id         = fold_id
        self.fold_type       = fold_type 
        self.force_extract   = force_extract # this variable has become moot now
        
        # Params - Spatial (x,y)
        self.spacing   = spacing
        self.grid      = grid
        self.crop_init = crop_init
        self.mask_type = mask_type

        # Params - Transforms/Filters
        self.transforms  = transforms
        self.filter_grid = filter_grid
        self.random_grid = random_grid
        self.pregridnorm = pregridnorm

        # Params - Memory related
        self.patient_shuffle = patient_shuffle
        
        # Params - Dataset related
        self.centred_dataloader_prob = centred_dataloader_prob

        # Params - TFlow Dataloader related
        self.parallel_calls = parallel_calls # [1, tf.data.experimental.AUTOTUNE]
        self.deterministic  = deterministic

        # Params - Debug
        self.debug         = debug
        self.single_sample = single_sample

        # Data items
        self.data              = {}
        self.patient_meta_info = {}
        self.paths_img         = []
        self.paths_mask        = []
        self.cache             = {}
        self.filter            = None

        # Config
        self.dataset_config = getattr(config, self.name)

        # Function calls
        self._process_niftis()
        self._init_data()

    def __len__(self):
        
        total_samples = None

        if self.grid:
            
            if self.crop_init:
                if self.voxel_shape_cropped == [240,240,80]:
                    if self.grid_size == [96,96,40]:
                        total_samples = len(self.paths_img)*(18)
                    elif self.grid_size == [140,140,40]:
                        total_samples = len(self.paths_img)*8
                    elif self.grid_size == [240,240,40]:
                        total_samples = len(self.paths_img)*2
                    elif self.grid_size == [240,240,80]:
                        total_samples = len(self.paths_img)*1
                
                elif self.voxel_shape_cropped == [240,240,100]:
                    if self.grid_size == [140,140,60]:
                        total_samples = len(self.paths_img)*8
        
        else:
            total_samples = len(self.paths_img)
        
        return total_samples

    def _sorter(self, patient_id):
        try:
            return int(str(patient_id)) # [e.g. 1, 2]
        except:
            return np.inf

    def _process_niftis(self):
        """
        Step 1 - Manually (or via code) select CT session and download ScanFormat=NIFTI from https://xnat.bmia.nl/app/action/ProjectDownloadAction/project/stwstrategyhn1
        Step 2 - Unzip the downloaded file in to <DATA_DIR>/config.KEY_DATASET_NAME_MAASTRO/config.DIRNAME_RAW
        Step 3 - Run the code below
        """

        self.dataset_dir_raw       = Path(self.dataset_dir).joinpath(config.DIRNAME_RAW)
        self.dataset_dir_processed = Path(self.dataset_dir).joinpath(config.DIRNAME_PROCESSED_SPACING.format('_'.join([str(float(each)) for each in self.spacing])))

        if self.force_extract or not Path(self.dataset_dir_processed).exists():
            print ('')
            print (' ------------------ {} Dataset ------------------'.format(self.name))

            t0 = time.time()
            total_patients = len(list(Path(self.dataset_dir_raw).glob('*')))
            with tqdm.tqdm(total=total_patients) as pbar:
                for path_patient in sorted(Path(self.dataset_dir_raw).iterdir(), key=self._sorter):

                    try:
                        patient_id = Path(path_patient).parts[-1]
                        pbar.set_description_str(' - Patient: {}'.format(patient_id))

                        DICOMVolumizerStructSegData(path_patient).volumize(self.spacing)
                    
                    except:
                        traceback.print_exc()
                        print (' ****** [ERROR] Patient: ', patient_id, '\n')

                    pbar.update(1)

            print ('')
            print (' ------------------------- * -------------------------')
            print (' -- Total time: ', round(time.time() - t0, 2), 's')
            print ('')
    
    def _init_data(self):

        try:
            # Total patients = 50
            self.ignore_patients = []

            # Step 1 - Get patient_ids
            if self.fold_id is None:
                self.patient_ids = list( [Path(each).parts[-1] for each in list(Path(self.dataset_dir_processed).glob('*'))] )
            else:
                json_filepath    = Path(self.dataset_dir_raw).joinpath(config.FILENAME_FOLDS)
                json_folds_info  = utils.read_json(json_filepath)
                if json_folds_info is not None:
                    self.patient_ids = json_folds_info[str(self.fold_id)][self.fold_type]
                else:
                    print (' - [ERROR][{}] Issue with {}: {}'.format(self.name, config.FILENAME_FOLDS, json_filepath)) 
                    sys.exit(1)

            # Step 2 - Meta for labels
            self.LABEL_MAP          = self.dataset_config[config.KEY_LABEL_MAP]
            self.LABEL_COLORS       = self.dataset_config[config.KEY_LABEL_COLORS]
            self.LABEL_WEIGHTS      = np.array(self.dataset_config[config.KEY_LABEL_WEIGHTS])
            self.LABEL_WEIGHTS      = (self.LABEL_WEIGHTS / np.sum(self.LABEL_WEIGHTS)).tolist()

            # Step 3.2 - Meta for voxel HU
            self.HU_MIN             = self.dataset_config[config.KEY_HU_MIN]
            self.HU_MAX             = self.dataset_config[config.KEY_HU_MAX]

            # Step 4 - Get paths_img, paths_mask and patient_meta_info
            for patient_id in sorted(self.patient_ids, key=self._sorter):
                try:
                    
                    if patient_id in self.ignore_patients:
                        continue

                    patient_id_path = Path(self.dataset_dir_processed).joinpath(str(patient_id))
                    patient_config  = utils.read_json(str(Path(patient_id_path).joinpath(config.FILENAME_CONFIG)))

                    mean_midpoint = patient_config[config.KEYNAME_MEAN_MIDPOINT]
                    if len(mean_midpoint):

                        self.patient_meta_info[patient_id] = {
                            config.KEYNAME_MEAN_MIDPOINT      : mean_midpoint
                            , config.KEYNAME_LABEL_MISSING    : patient_config[config.KEYNAME_LABEL_MISSING]
                            , config.KEYNAME_SHAPE_RESAMPLED  : patient_config[config.KEYNAME_SHAPE]
                        }

                        path_img  = Path(patient_id_path).joinpath(config.FILENAME_STRUCTSEG_IMG_PROC)
                        path_mask = Path(patient_id_path).joinpath(config.FILENAME_STRUCTSEG_MASK_PROC)

                        if Path(path_img).exists() and Path(path_mask).exists():
                            self.paths_img.append(str(path_img))
                            self.paths_mask.append(str(path_mask))
                        else:
                            print (' - [ERROR][{}] Patient: {} has path issues: '.format(self.name, patient_id))

                    else:
                        print (' - [ERROR][{}] Patient: {} has config issues: '.format(self.name, patient_id))

                except:
                    print ('\n - [ERROR][{}] Patient: {} has some issue. \n'.format(self.name, patient_id))
                    traceback.print_exc()
                    pdb.set_trace()
            
            # Step 5 - Meta info for grid sampling
            if self.crop_init:
                self.crop_info        = self.dataset_config[config.KEY_PREPROCESS][config.KEY_CROP][str(self.spacing)]
                self.w_lateral_crop   = self.crop_info[config.KEY_MIDPOINT_EXTENSION_W_LEFT]
                self.w_medial_crop    = self.crop_info[config.KEY_MIDPOINT_EXTENSION_W_RIGHT]
                self.h_posterior_crop = self.crop_info[config.KEY_MIDPOINT_EXTENSION_H_BACK]
                self.h_anterior_crop  = self.crop_info[config.KEY_MIDPOINT_EXTENSION_H_FRONT]
                self.d_cranial_crop   = self.crop_info[config.KEY_MIDPOINT_EXTENSION_D_TOP]
                self.d_caudal_crop    = self.crop_info[config.KEY_MIDPOINT_EXTENSION_D_BOTTOM]
                self.voxel_shape_cropped = [self.w_lateral_crop+self.w_medial_crop
                                            , self.h_posterior_crop+self.h_anterior_crop
                                            , self.d_cranial_crop+self.d_caudal_crop]

            if self.grid:
                grid_3D_params         = self.dataset_config[config.KEY_GRID_3D][str(self.spacing)]
                self.grid_size         = grid_3D_params[config.KEY_GRID_SIZE]
                self.grid_overlap      = grid_3D_params[config.KEY_GRID_OVERLAP]
                self.SAMPLER_PERC      = grid_3D_params[config.KEY_GRID_SAMPLER_PERC]
                self.RANDOM_SHIFT_MAX  = grid_3D_params[config.KEY_GRID_RANDOM_SHIFT_MAX]
                self.RANDOM_SHIFT_PERC = grid_3D_params[config.KEY_GRID_RANDOM_SHIFT_PERC]

                self.w_grid, self.h_grid, self.d_grid          = self.grid_size
                self.w_overlap, self.h_overlap, self.d_overlap = self.grid_overlap
        
        except:
            traceback.print_exc()
            pdb.set_trace()

    def get_label_map(self):
        return self.LABEL_MAP

    def get_label_weights(self):
        return self.LABEL_WEIGHTS

    def generator(self):
        """
         - Note: 
            - In general, even when running your model on an accelerator like a GPU or TPU, the tf.data pipelines are run on the CPU
                - Ref: https://www.tensorflow.org/guide/data_performance_analysis#analysis_workflow
        """

        try:
            
            if len(self.paths_img) and len(self.paths_mask):
                
                # Step 1 - Create basic generator
                dataset = None
                dataset = tf.data.Dataset.from_generator(self._generator3D
                    , output_types=(config.DATATYPE_TF_FLOAT32, config.DATATYPE_TF_UINT8, config.DATATYPE_TF_INT32, tf.string)
                    ,args=())
                
                # Step 2 - Get 3D data
                dataset = dataset.map(self._get_data_3D, num_parallel_calls=self.parallel_calls, deterministic=self.deterministic)
                        
                # Step 3 - Filter function
                if self.filter_grid:
                    dataset = dataset.filter(self.filter.execute)

                # Step 4 - Data augmentations
                if len(self.transforms):
                    for transform in self.transforms:
                        try:
                            dataset = dataset.map(transform.execute, num_parallel_calls=self.parallel_calls, deterministic=self.deterministic)
                        except:
                            traceback.print_exc()
                            print (' - [ERROR][HaNMICCAI2015Dataset] Issue with transform: ', transform.name)
                else:
                    print ('')
                    print (' - [INFO][{}] No transformations available for fold:{} and type:{} !'.format(self.name, self.fold_id, self.fold_type))
                    print ('')

                # Step 6 - Return
                return dataset
            
            else:
                return None

        except:
            traceback.print_exc()
            pdb.set_trace()
            return None

    def _get_paths(self, idx):
        patient_id = ''
        study_id   = ''
        path_img, path_mask = '', ''

        if self.debug:
            path_img = Path(self.paths_img[0]).absolute()
            path_mask = Path(self.paths_mask[0]).absolute()
            path_img, path_mask = self.path_debug_3D(path_img, path_mask, idx)
        else:
            path_img = Path(self.paths_img[idx]).absolute()
            path_mask = Path(self.paths_mask[idx]).absolute()

        if path_img.exists() and path_mask.exists():
            patient_id = Path(path_img).parts[-2]
            study_id = Path(path_img).parts[-3] # folder representing spacing
        else:
            print (' - [ERROR] Issue with path')
            print (' -- [ERROR][HaNMICCAI2015] path_img : ', path_img)
            print (' -- [ERROR][HaNMICCAI2015] path_mask: ', path_mask)
        
        return path_img, path_mask, patient_id, study_id

    def _generator3D(self):
        """
        This function loop over the self.paths_img list
        """

        # Step 0 - Init
        res = []

        # Step 1 - Get patient idxs
        idxs = np.arange(len(self.paths_img)).tolist()  #[:3]
        if self.single_sample: idxs = idxs[0:2] # [2:4]
        if self.patient_shuffle: np.random.shuffle(idxs)

        # Step 2 - Proceed on the basis of grid sampling or full-volume (self.grid=False) sampling
        if self.grid:
            
            # Step 2.1 - Get grid sampler info for each patient-idx
            sampler_info = {}
            for idx in idxs:
                path_img           = Path(self.paths_img[idx]).absolute() 
                patient_id         = path_img.parts[-3]
                
                if self.crop_init:
                    voxel_shape = self.voxel_shape_cropped
                else:
                    voxel_shape = self.patient_meta_info[patient_id][config.KEYNAME_SHAPE_RESAMPLED]

                grid_idxs_width   = utils.split_into_overlapping_grids(voxel_shape[0], len_grid=self.grid_size[0], len_overlap=self.grid_overlap[0])
                grid_idxs_height  = utils.split_into_overlapping_grids(voxel_shape[1], len_grid=self.grid_size[1], len_overlap=self.grid_overlap[1])   
                grid_idxs_depth   = utils.split_into_overlapping_grids(voxel_shape[2], len_grid=self.grid_size[2], len_overlap=self.grid_overlap[2])
                sampler_info[idx] = list(itertools.product(grid_idxs_width,grid_idxs_height,grid_idxs_depth))
                
                if 0: #patient_id in self.patients_z_prob:
                    print (' - [DEBUG] patient_id: ', patient_id, ' || voxel_shape: ', voxel_shape)
                    print (' - [DEBUG] sampler_info: ', sampler_info[idx])

            # Step 2.2 - Loop over all patients and their grids
            # Note - Grids of a patient are extracted in order
            for i, idx in enumerate(idxs):
                path_img, path_mask, patient_id, patient_folder = self._get_paths(idx)
                missing_labels                            = self.patient_meta_info[patient_id][config.KEYNAME_LABEL_MISSING]
                bgd_mask                                  = 1 # by default
                if len(missing_labels): 
                    bgd_mask = 0
                if path_img.exists() and path_mask.exists():
                    for sample_info in sampler_info[idx]:
                        grid_idxs = sample_info
                        meta1     = [idx] + [grid_idxs[0][0], grid_idxs[1][0], grid_idxs[2][0]]  # only include w_start, h_start, d_start
                        meta2     = '-'.join([self.name, patient_folder, patient_id])
                        path_img  = str(path_img)
                        path_mask = str(path_mask)
                        res.append((path_img, path_mask, meta1, meta2, bgd_mask))
        
        else:
            label_names = list(self.LABEL_MAP.keys())
            for i, idx in enumerate(idxs):
                path_img, path_mask, patient_id, patient_folder = self._get_paths(idx)
                missing_label_names = self.patient_meta_info[patient_id][config.KEYNAME_LABEL_MISSING]
                bgd_mask = 1
                
                # if len(missing_labels): bgd_mask = 0
                if len(missing_label_names):
                    if len(set(label_names) - set(missing_label_names)): 
                        bgd_mask = 0
                    
                if path_img.exists() and path_mask.exists():
                    meta1 = [idx] + [0,0,0] # dummy for w_start, h_start, d_start
                    meta2 ='-'.join([self.name, patient_folder, patient_id])
                    path_img = str(path_img)
                    path_mask = str(path_mask)
                    res.append((path_img, path_mask, meta1, meta2, bgd_mask))

        # Step 3 - Yield
        for each in res:
            path_img, path_mask, meta1, meta2, bgd_mask = each
            
            vol_img_npy, vol_mask_npy, spacing = self._get_cache_item_old(path_img, path_mask)
            if vol_img_npy is None and vol_mask_npy is None:
                vol_img_npy, vol_mask_npy, spacing = self._get_volume_from_path(path_img, path_mask)    
                self._set_cache_item_old(path_img, path_mask, vol_img_npy, vol_mask_npy, spacing)
            
            spacing           = tf.constant(spacing, dtype=tf.int32)
            vol_img_npy_shape = tf.constant(vol_img_npy.shape, dtype=tf.int32)
            meta1             = tf.concat([meta1, spacing, vol_img_npy_shape, [bgd_mask]], axis=0) # [idx,[grid_idxs],[spacing],[shape]]

            yield (vol_img_npy, vol_mask_npy, meta1, meta2)

    def _get_cache_item_old(self, path_img, path_mask):
        if 'img' in self.cache and 'mask' in self.cache:
            if path_img in self.cache['img'] and path_mask in self.cache['mask']:
                # print (' - [_get_cache_item()] ')
                return self.cache['img'][path_img], self.cache['mask'][path_mask], self.cache['spacing']
            else:
                return None, None, None
        else:
            return None, None, None
    
    def _set_cache_item_old(self, path_img, path_mask, vol_img, vol_mask, spacing):
        # print (' - [_set_cache_item() ]: ', vol_img.shape, vol_mask.shape)
        self.cache = {
            'img': {path_img: vol_img}
            , 'mask': {path_mask: vol_mask}
            , 'spacing': spacing
        }
    
    def _get_volume_from_path(self, path_img, path_mask, verbose=False):

        # Step 1 - Get volumes
        if verbose: t0 = time.time()
        vol_img_sitk   = utils.read_mha(path_img)
        vol_img_npy    = utils.sitk_to_array(vol_img_sitk)
        vol_mask_sitk  = utils.read_mha(path_mask)
        vol_mask_npy   = utils.sitk_to_array(vol_mask_sitk)     
        spacing        = np.array(vol_img_sitk.GetSpacing())

        # Step 2 - Perform init crop on volumes
        if self.crop_init:
            patient_id       = str(Path(path_img).parts[-2])
            mean_point = np.array(self.patient_meta_info[patient_id][config.KEYNAME_MEAN_MIDPOINT]).astype(np.uint16).tolist()
            vol_img_npy_shape_prev = vol_img_npy.shape

            # Step 2.1 - Perform crops in H,W region
            vol_img_npy = vol_img_npy[
                    mean_point[0] - self.w_lateral_crop    : mean_point[0] + self.w_medial_crop
                    , mean_point[1] - self.h_anterior_crop : mean_point[1] + self.h_posterior_crop
                    , :
                ]
            vol_mask_npy = vol_mask_npy[
                    mean_point[0] - self.w_lateral_crop    : mean_point[0] + self.w_medial_crop
                    , mean_point[1] - self.h_anterior_crop : mean_point[1] + self.h_posterior_crop
                    , :
                ]
            
            # Step 2.2 - Perform crops in D region
            vol_img_npy  = vol_img_npy[:,  :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
            vol_mask_npy = vol_mask_npy[:, :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
            
            # Step 3 - Pad (with=0) if volume is less in z-dimension
            (vol_img_npy_x, vol_img_npy_y, vol_img_npy_z) = vol_img_npy.shape
            if vol_img_npy_z < self.d_caudal_crop + self.d_cranial_crop:
                del_z = self.d_caudal_crop + self.d_cranial_crop - vol_img_npy_z
                vol_img_npy = np.concatenate((vol_img_npy, np.zeros((vol_img_npy_x, vol_img_npy_y, del_z))), axis=2)
                vol_mask_npy = np.concatenate((vol_mask_npy, np.zeros((vol_img_npy_x, vol_img_npy_y, del_z))), axis=2)
            
            # print (' - [cropping] z_mean: ', mean_point[2], ' || -', self.d_caudal_crop, ' || + ', self.d_cranial_crop)
            # print (' - [cropping] || shape: ', vol_img_npy.shape)
            # print (' - [DEBUG] change: ', vol_img_npy_shape_prev, vol_img_npy.shape)
        
        if verbose: print (' - [{}._get_volume_from_path()] Time: ({}):{}s'.format(self.name, Path(path_img).parts[-2],  round(time.time() - t0,2)))        
        if self.pregridnorm:
            vol_img_npy[vol_img_npy <= self.HU_MIN] = self.HU_MIN
            vol_img_npy[vol_img_npy >= self.HU_MAX] = self.HU_MAX
            vol_img_npy                             = (vol_img_npy - np.mean(vol_img_npy))/np.std(vol_img_npy) #Standardize (z-scoring)
            
        return tf.cast(vol_img_npy, dtype=config.DATATYPE_TF_FLOAT32), tf.cast(vol_mask_npy, dtype=config.DATATYPE_TF_UINT8), tf.constant(spacing*100, dtype=config.DATATYPE_TF_INT32)
        
    @tf.function
    def _get_new_grid_idx(self, start, end, max):
        
        start_prev = start
        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.RANDOM_SHIFT_PERC:
            
            delta_left   = start
            delta_right  = max - end
            shift_voxels = tf.random.uniform([], minval=0, maxval=self.RANDOM_SHIFT_MAX, dtype=tf.dtypes.int32)

            if delta_left > self.RANDOM_SHIFT_MAX and delta_right > self.RANDOM_SHIFT_MAX:
                if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.RANDOM_SHIFT_PERC:
                    start = start - shift_voxels
                    end   = end - shift_voxels
                else:
                    start = start + shift_voxels
                    end   = end + shift_voxels

            elif delta_left > self.RANDOM_SHIFT_MAX and delta_right <= self.RANDOM_SHIFT_MAX:
                start = start - shift_voxels
                end   = end - shift_voxels

            elif delta_left <= self.RANDOM_SHIFT_MAX and delta_right > self.RANDOM_SHIFT_MAX:
                start = start + shift_voxels
                end   = end + shift_voxels

        return start_prev, start, end

    @tf.function
    def _get_new_grid_idx_centred(self, grid_size_half, max_pt, mid_pt):
        
        # Step 1 - Return vars
        start, end = 0,0

        # Step 2 - Define margin on either side of mid point
        margin_left = mid_pt
        margin_right = max_pt - mid_pt

        # Ste p2 - Calculate vars
        if margin_left >= grid_size_half and margin_right >= grid_size_half:
            start = mid_pt - grid_size_half
            end = mid_pt + grid_size_half
        elif margin_right < grid_size_half:
            if margin_left >= grid_size_half + (grid_size_half - margin_right): 
                end = mid_pt + margin_right
                start = mid_pt - grid_size_half - (grid_size_half - margin_right)
            else:
                tf.print(' - [ERROR][_get_new_grid_idx_centred()] Cond 2 problem')
        elif margin_left < grid_size_half:
            if margin_right >= grid_size_half + (grid_size_half - margin_left):
                start = mid_pt - margin_left
                end = mid_pt + grid_size_half + (grid_size_half-margin_left)
            else:
                tf.print(' - [ERROR][_get_new_grid_idx_centred()] Cond 3 problem')
        
        return start, end

    @tf.function
    def _get_data_3D(self, vol_img, vol_mask, meta1, meta2):
        """
        Params
        ------
        meta1: [idx, [w_start, h_start, d_start], [spacing_x, spacing_y, spacing_z], [shape_x, shape_y, shape_z], [bgd_mask]]
        """

        vol_img_npy = None
        vol_mask_npy = None

        # Step 1 - Proceed on the basis of grid sampling or full-volume (self.grid=False) sampling
        if self.grid:
            
            if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.centred_dataloader_prob:

                # Step 1.1 - Get a label_id and its mean (PS: this assumes that the label_id is present in vol_mask)
                if 0:
                    label_id           = tf.cast(tf.random.categorical(tf.math.log([self.LABEL_WEIGHTS]), 1), dtype=config.DATATYPE_TF_UINT8)[0][0]  # the LABEL_WEGIHTS sum to 1; the log is added as tf.random.categorical expects logits
                    label_id_idxs      = tf.where(tf.math.equal(label_id, vol_mask))
                    label_id_idxs_mean = tf.math.reduce_mean(label_id_idxs, axis=0)
                    label_id_idxs_mean = tf.cast(label_id_idxs_mean, dtype=config.DATATYPE_TF_INT32)
                
                # Step 1.1 - Get a label_id and its mean (PS: this does not assume that the label_id is present in vol_mask)
                else:
                    label_ids, _  = tf.unique(tf.reshape(vol_mask, [-1]))
                    label_weights = tf.gather(self.LABEL_WEIGHTS, tf.cast(label_ids, dtype=tf.int32))
                    label_id_idx  = tf.random.categorical(tf.math.log([label_weights]), 1)
                    label_id      = tf.gather(label_ids, label_id_idx)
                    label_id      = tf.reshape(label_id, [-1])
                    label_id_idxs      = tf.where(tf.math.equal(vol_mask, label_id))
                    label_id_idxs_mean = tf.math.reduce_mean(label_id_idxs, axis=0)
                    label_id_idxs_mean = tf.cast(label_id_idxs_mean, dtype=config.DATATYPE_TF_INT32)
                    
                # Step 1.2 - Create a grid around that mid-point
                w_start_prev = meta1[1]
                h_start_prev = meta1[2]
                d_start_prev = meta1[3]
                w_max        = meta1[7]
                h_max        = meta1[8]
                d_max        = meta1[9]
                w_grid       = self.grid_size[0] 
                h_grid       = self.grid_size[1] 
                d_grid       = self.grid_size[2]
                w_mid        = label_id_idxs_mean[0]
                h_mid        = label_id_idxs_mean[1] 
                d_mid        = label_id_idxs_mean[2]

                w_start, w_end = self._get_new_grid_idx_centred(w_grid//2, w_max, w_mid)
                h_start, h_end = self._get_new_grid_idx_centred(h_grid//2, h_max, h_mid)
                d_start, d_end = self._get_new_grid_idx_centred(d_grid//2, d_max, d_mid)

                meta1_diff = tf.convert_to_tensor([0,w_start - w_start_prev, h_start - h_start_prev, d_start - d_start_prev,0,0,0,0,0,0,0])
                meta1      = meta1 + meta1_diff

            else:
                
                # tf.print(' - [INFO] regular dataloader: ', self.dir_type)
                # Step 1.1 - Get raw images/masks and extract grid
                w_start = meta1[1]
                w_end   = w_start + self.grid_size[0]
                h_start = meta1[2]
                h_end   = h_start + self.grid_size[1]
                d_start = meta1[3]
                d_end   = d_start + self.grid_size[2]

                # Step 1.2 - Randomization of grid 
                if self.random_grid:
                    w_max = meta1[7]
                    h_max = meta1[8]
                    d_max = meta1[9]

                    w_start_prev = w_start
                    d_start_prev = d_start
                    w_start_prev, w_start, w_end = self._get_new_grid_idx(w_start, w_end, w_max)
                    h_start_prev, h_start, h_end = self._get_new_grid_idx(h_start, h_end, h_max)
                    d_start_prev, d_start, d_end = self._get_new_grid_idx(d_start, d_end, d_max)

                    meta1_diff = tf.convert_to_tensor([0,w_start - w_start_prev, h_start - h_start_prev, d_start - d_start_prev,0,0,0,0,0,0,0])
                    meta1 = meta1 + meta1_diff
                
            # Step 1.3 - Extracting grid
            vol_img_npy  = tf.identity(vol_img[w_start:w_end, h_start:h_end, d_start:d_end])
            vol_mask_npy = tf.identity(vol_mask[w_start:w_end, h_start:h_end, d_start:d_end])
            
        else:
            vol_img_npy = vol_img
            vol_mask_npy = vol_mask

        # Step 2 - One-hot or not
        vol_mask_classes = []
        label_ids_mask = []
        label_ids = sorted(list(self.LABEL_MAP.values()))
        if self.mask_type == config.MASK_TYPE_ONEHOT:
            vol_mask_classes = tf.concat([tf.expand_dims(tf.math.equal(vol_mask_npy, label), axis=-1) for label in label_ids], axis=-1) # [H,W,D,L]
            for label_id in label_ids:
                label_ids_mask.append(tf.cast(tf.math.reduce_any(vol_mask_classes[:,:,:,label_id]), dtype=config.DATATYPE_TF_INT32))
            
        elif self.mask_type == config.MASK_TYPE_COMBINED:
            vol_mask_classes = vol_mask_npy
            unique_classes, _ = tf.unique(tf.reshape(vol_mask_npy,[-1]))
            unique_classes = tf.cast(unique_classes, config.DATATYPE_TF_INT32)
            for label_id in label_ids:
                label_ids_mask.append(tf.cast(tf.math.reduce_any(tf.math.equal(unique_classes, label_id)), dtype=config.DATATYPE_TF_INT32))    
        
        # Step 2.2 - Handling background mask explicitly if there is a missing label
        bgd_mask = meta1[-1]
        label_ids_mask[0] = bgd_mask
        meta1 = meta1[:-1] # removes the bgd mask index

        # Step 3 - Dtype conversion and expading dimensions            
        if self.mask_type == config.MASK_TYPE_ONEHOT:
            x = tf.cast(tf.expand_dims(vol_img_npy, axis=3), dtype=config.DATATYPE_TF_FLOAT32) # [H,W,D,1]
        else:
            x = tf.cast(vol_img_npy, dtype=config.DATATYPE_TF_FLOAT32) # [H,W,D]
        y = tf.cast(vol_mask_classes, dtype=config.DATATYPE_TF_FLOAT32) # [H,W,D,L]

        # Step 4 - Append info to meta1
        meta1 = tf.concat([meta1, label_ids_mask], axis=0)

        # Step 5 - return
        return (x, y, meta1, meta2)
    
    def path_debug_3D(self, path_img, path_mask, idx):
        
        # patient_numbers = ['25', '29'] 
        patient_numbers = ['43', '46'] 
        
        patient_idx = idx % len(patient_numbers)

        path_img_parts = list(Path(path_img).parts)
        path_img_parts[-2] = patient_numbers[patient_idx]
        path_img = Path(*path_img_parts)

        path_mask_parts = list(Path(path_mask).parts)
        path_mask_parts[-2] = patient_numbers[patient_idx]
        path_mask = Path(*path_mask_parts)

        if 1:
            utils.print_debug_header()
            print (' - path_img : ', path_img.parts[-5:])
            print (' - path_mask: ', path_mask.parts[-5:])

        return path_img, path_mask 