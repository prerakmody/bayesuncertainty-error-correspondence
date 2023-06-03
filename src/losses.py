# Import internal libraries
import src.config as config

# Import external libraries
import pdb
import copy
import time
import skimage.util
import traceback
import numpy as np
from pathlib import Path
import SimpleITK as sitk 
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp


MAX_FUNC_TIME = 300 

############################################################
#                           UTILS                          #
############################################################

@tf.function
def get_mask(mask_1D, Y):
    # mask_1D: [[1,0,0,0, ...., 1]] - [B,L] something like this
    # Y : [B,H,W,D,L] 
    if 0:
        mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(mask_1D, axis=1),axis=1),axis=1) # mask.shape=[B,1,1,1,L]
        mask = tf.tile(mask, multiples=[1,Y.shape[1],Y.shape[2],Y.shape[3],1]) # mask.shape = [B,H,W,D,L]
        mask = tf.cast(mask, tf.float32)
        return mask
    else:
        return mask_1D

def get_largest_component(y, verbose=True):
    """
    Takes as input predicted predicted probs and returns a binary mask with the largest component for each label

    Params
    ------
    y: [H,W,D,L] --> predicted probabilities of np.float32 
    - Ref: https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/filters.html
    """

    try:
        
        label_count = y.shape[-1]
        y = np.argmax(y, axis=-1)
        y = np.concatenate([np.expand_dims(y == label_id, axis=-1) for label_id in range(label_count)],axis=-1)
        ccFilter = sitk.ConnectedComponentImageFilter()
        
        for label_id in range(y.shape[-1]):

            if label_id > 0 and label_id not in []: # Note: pointless to do it for background
                if verbose: t0 = time.time()
                y_label = y[:,:,:,label_id] # [H,W,D]

                component_img = ccFilter.Execute(sitk.GetImageFromArray(y_label.astype(np.uint8)))
                component_array = sitk.GetArrayFromImage(component_img) # will contain pseduo-labels given to different components
                component_count = ccFilter.GetObjectCount()
                
                if component_count >= 2: # background and foreground
                    component_sizes = np.bincount(component_array.flatten()) # count the voxels belong to different components
                    component_sizes_sorted = np.asarray(sorted(component_sizes, reverse=True))
                    if verbose: print ('\n - [INFO][losses.get_largest_component()] label_id: ', label_id, ' || sizes: ', component_sizes_sorted)
                
                    component_largest_sortedidx = np.argwhere(component_sizes == component_sizes_sorted[1])[0][0] # Note: idx=1 as idx=0 is background # Risk: for optic nerves
                    y_label_mask = (component_array == component_largest_sortedidx).astype(np.float32)
                    y[:,:,:,label_id] = y_label_mask
                    if verbose: print (' - [INFO][losses.get_largest_component()]: label_id: ', label_id, '(',round(time.time() - t0,3),'s)')
                else:
                    if verbose: print (' - [INFO][losses.get_largest_component()] label_id: ', label_id, ' has only background!!')

                # [TODO]: set other components as background (i.e. label=0)

        y = y.astype(np.float32)
        return y

    except:
        traceback.print_exc()
        pdb.set_trace()

def remove_smaller_components(y_true, y_pred, meta='', label_ids_small = [], verbose=False):
    """
    Takes as input predicted probs and returns a binary mask by removing some of the smallest components for each label

    Params
    ------
    y_true: [H,W,D,C] # only need to pass if your doing some testing
    y_pred: [H,W,D,C] --> predicted probabilities of np.float32

    Returns
    -------
    y: [H,W,D,C]

    - Ref: https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/filters.html
    """
    t0 = time.time()

    try:
        
        # Step 0 - Init
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Step 0 - Preprocess by selecting one voxel per class
        y_pred_copy = copy.deepcopy(y_pred) # [H,W,D,L] with probs
        label_count = y_pred_copy.shape[-1]
        if label_count > 1:
            y_pred_copy = np.argmax(y_pred_copy, axis=-1) # [H,W,D]
            y_pred_copy = np.concatenate([np.expand_dims(y_pred_copy == label_id, axis=-1) for label_id in range(label_count)],axis=-1) # [H,W,D,L] as a binary mask
        elif label_count == 1:
            thresh = 0.5
            y_pred_copy[y_pred_copy > thresh] = 1
            y_pred_copy[y_pred_copy <= thresh] = 0
        
        for label_id in range(y_pred_copy.shape[-1]):

            if ((label_id > 0 and label_count > 1) or (label_count == 1)): # Note: pointless to do it for background
                if verbose: t0 = time.time()
                if label_count == 1:
                    y_label = y_pred_copy[:,:,:,label_id-1] # [H,W,D]
                else:
                    y_label = y_pred_copy[:,:,:,label_id]

                # Step 1 - Get different components
                ccFilter = sitk.ConnectedComponentImageFilter()
                component_img = ccFilter.Execute(sitk.GetImageFromArray(y_label.astype(np.uint8)))
                component_array = sitk.GetArrayFromImage(component_img) # will contain pseduo-labels given to different components
                component_count = ccFilter.GetObjectCount()
                component_sizes = np.bincount(component_array.flatten()) # count the voxels belong to different components    

                # Step 2 - Evaluate each component
                if component_count >= 1: # at least a foreground (along with background)

                    # Step 2.1 - Sort them on the basis of voxel count
                    component_sizes_sorted = np.asarray(sorted(component_sizes, reverse=True))
                    if verbose:
                        print ('\n - [INFO][losses.get_largest_component()] label_id: ', label_id, ' || sizes: ', component_sizes_sorted)
                        print (' - [INFO][losses.get_largest_component()] unique_comp_labels: ', np.unique(component_array)) 
                        
                    # Step 2.1 - Remove really small components for good Hausdorff calculation
                    component_sizes_sorted_unique = np.unique(component_sizes_sorted[::-1]) # ascending order
                    for comp_size_id, comp_size in enumerate(component_sizes_sorted_unique):
                        if label_id in label_ids_small:
                            if comp_size <= config.MIN_SIZE_COMPONENT:
                                components_labels = [each[0] for each in np.argwhere(component_sizes == comp_size)]
                                for component_label in components_labels:
                                    component_array[component_array == component_label] = 0
                        else:
                            if comp_size_id < len(component_sizes_sorted_unique) - 2: # set to 0 except background and foreground
                                components_labels = [each[0] for each in np.argwhere(component_sizes == comp_size)]
                                for component_label in components_labels:
                                    component_array[component_array == component_label] = 0
                    if verbose: print (' - [INFO][losses.get_largest_component()] unique_comp_labels: ', np.unique(component_array))
                    if label_count == 1:
                        y_pred_copy[:,:,:,label_id-1] = component_array.astype(np.bool).astype(np.float32)
                    else:
                        y_pred_copy[:,:,:,label_id] = component_array.astype(np.bool).astype(np.float32)
                    if verbose: print (' - [INFO][losses.get_largest_component()] label_id: ', label_id, '(',round(time.time() - t0,3),'s)')

                    if 0:
                        # Step 1 - Hausdorff
                        y_true_label = y_true[:,:,:,label_id]
                        y_pred_label = y_pred_copy[:,:,:,label_id]

                        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
                        hausdorff_distance_filter.Execute(sitk.GetImageFromArray(y_true_label.astype(np.uint8)), sitk.GetImageFromArray(y_pred_label.astype(np.uint8)))
                        print (' - hausdorff: ', hausdorff_distance_filter.GetHausdorffDistance())
                        
                        # Step 2 - 95% Hausdorff
                        y_true_contour = sitk.LabelContour(sitk.GetImageFromArray(y_true_label.astype(np.uint8)), False)
                        y_pred_contour = sitk.LabelContour(sitk.GetImageFromArray(y_pred_label.astype(np.uint8)), False)
                        y_true_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_true_contour, squaredDistance=False, useImageSpacing=True)) # i.e. euclidean distance
                        y_pred_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_pred_contour, squaredDistance=False, useImageSpacing=True))
                        dist_y_pred = sitk.GetArrayViewFromImage(y_pred_distance_map)[sitk.GetArrayViewFromImage(y_true_distance_map)==0] # pointless?
                        dist_y_true = sitk.GetArrayViewFromImage(y_true_distance_map)[sitk.GetArrayViewFromImage(y_pred_distance_map)==0]
                        print (' - 95 hausdorff:', np.percentile(dist_y_true,95), np.percentile(dist_y_pred,95))

                else:
                    print (' - [INFO][losses.get_largest_component()] for meta: {} || label_id: {} has only background!! ({}) '.format(meta, label_id, component_sizes))
            
            if time.time() - t0 > MAX_FUNC_TIME:
                print (' - [INFO][losses.get_largest_component()] Taking too long: ', round(time.time() - t0,2),'s')

        y = y_pred_copy.astype(np.float32)
        return y

    except:
        traceback.print_exc()
        pdb.set_trace()

def get_hausdorff(y_true, y_pred, spacing, verbose=False):
    """
    :param y_true: [H, W, D, L]
    :param y_pred: [H, W, D, L] 
    - Ref: https://simpleitk.readthedocs.io/en/master/filters.html?highlight=%20HausdorffDistanceImageFilter()#simpleitk-filters
    - Ref: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    """

    try:
        # Step 0 - Init
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

        hausdorff_labels = []
        for label_id in range(y_pred.shape[-1]):
            
            try:
                y_true_label = y_true[:,:,:,label_id] # [H,W,D]
                y_pred_label = y_pred[:,:,:,label_id] # [H,W,D]

                # Calculate loss (over all pixels)
                if np.sum(y_true_label) > 0:
                    if label_id > 0:
                        try:
                            if np.sum(y_true_label) > 0:
                                y_true_sitk    = sitk.GetImageFromArray(y_true_label.astype(np.uint8))
                                y_pred_sitk    = sitk.GetImageFromArray(y_pred_label.astype(np.uint8))
                                y_true_sitk.SetSpacing(tuple(spacing))
                                y_pred_sitk.SetSpacing(tuple(spacing))
                                hausdorff_distance_filter.Execute(y_true_sitk, y_pred_sitk)
                                hausdorff_labels.append(hausdorff_distance_filter.GetHausdorffDistance())
                                if verbose: print (' - [INFO][get_hausdorff()] label_id: {} || hausdorff: {}'.format(label_id, hausdorff_labels[-1]))
                            else:
                                hausdorff_labels.append(-1)    
                        except:
                            print (' - [ERROR][get_hausdorff()] label_id: {}'.format(label_id))
                            hausdorff_labels.append(-1)    
                    else:
                        hausdorff_labels.append(0)
                else:
                    hausdorff_labels.append(0)
            
            except:
                hausdorff_labels.append(-1)
                traceback.print_exc()
                pdb.set_trace()
        
        hausdorff_labels = np.array(hausdorff_labels)
        hausdorff = np.mean(hausdorff_labels[hausdorff_labels>0])
        
    
    except:
        traceback.print_exc()
        pdb.set_trace()

    return hausdorff, hausdorff_labels

def get_surface_distances(y_true, y_pred, spacing, meta='', verbose=False):
    
    """
    :param y_true: [H, W, D, L] --> binary mask of np.float32
    :param y_pred: [H, W, D, L] --> also, binary mask of np.float32
    - Ref: https://discourse.itk.org/t/computing-95-hausdorff-distance/3832/3
    - Ref: https://git.lumc.nl/mselbiallyelmahdy/jointregistrationsegmentation-via-crossstetch/-/blob/master/lib/label_eval.py
    """

    try:

        # Step 0 - Init
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        label_count = y_pred.shape[-1]

        hausdorff_labels   = []
        hausdorff95_labels = []
        msd_labels         = []
        for label_id in range(label_count):
            
            y_true_label = y_true[:,:,:,label_id] # [H,W,D]
            y_pred_label = y_pred[:,:,:,label_id] # [H,W,D]

            # Calculate loss (over all pixels)
            if np.sum(y_true_label) > 0:

                if (label_id > 0 and label_count > 1) or (label_count == 1):
                    if np.sum(y_pred_label) > 0:
                        y_true_sitk    = sitk.GetImageFromArray(y_true_label.astype(np.uint8))
                        y_pred_sitk    = sitk.GetImageFromArray(y_pred_label.astype(np.uint8))
                        y_true_sitk.SetSpacing(tuple(spacing))
                        y_pred_sitk.SetSpacing(tuple(spacing))
                        y_true_contour = sitk.LabelContour(y_true_sitk, False, backgroundValue=0)
                        y_pred_contour = sitk.LabelContour(y_pred_sitk, False, backgroundValue=0)
                        
                        y_true_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_true_sitk, squaredDistance=False, useImageSpacing=True))
                        y_pred_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_pred_sitk, squaredDistance=False, useImageSpacing=True))
                        dist_y_true         = sitk.GetArrayFromImage(y_true_distance_map*sitk.Cast(y_pred_contour, sitk.sitkFloat32))
                        dist_y_pred         = sitk.GetArrayFromImage(y_pred_distance_map*sitk.Cast(y_true_contour, sitk.sitkFloat32))
                        dist_y_true         = dist_y_true[dist_y_true != 0]
                        dist_y_pred         = dist_y_pred[dist_y_pred != 0]
                            
                        if len(dist_y_true):
                            msd_labels.append(np.mean(np.array(list(dist_y_true) + list(dist_y_pred))))
                            if len(dist_y_true) and len(dist_y_pred):
                                hausdorff_labels.append( np.max( [np.max(dist_y_true), np.max(dist_y_pred)] ) )
                                hausdorff95_labels.append(np.max([np.percentile(dist_y_true, 95), np.percentile(dist_y_pred, 95)]))
                            elif len(dist_y_true) and not len(dist_y_pred):
                                hausdorff_labels.append(np.max(dist_y_true))
                                hausdorff95_labels.append(np.percentile(dist_y_true, 95))
                            elif not len(dist_y_true) and not len(dist_y_pred):
                                hausdorff_labels.append(np.max(dist_y_pred))
                                hausdorff95_labels.append(np.percentile(dist_y_pred, 95))
                        else:
                            hausdorff_labels.append(-1)
                            hausdorff95_labels.append(-1)
                            msd_labels.append(-1)
                    
                    else:
                        hausdorff_labels.append(-1)
                        hausdorff95_labels.append(-1)
                        msd_labels.append(-1)
                
                else:
                    hausdorff_labels.append(0)
                    hausdorff95_labels.append(0)
                    msd_labels.append(0)
            
            else:
                hausdorff_labels.append(0)
                hausdorff95_labels.append(0)
                msd_labels.append(0)
        
        hausdorff_labels   = np.array(hausdorff_labels)
        hausdorff_mean     = np.mean(hausdorff_labels[hausdorff_labels > 0])
        hausdorff95_labels = np.array(hausdorff95_labels)
        hausdorff95_mean   = np.mean(hausdorff95_labels[hausdorff95_labels > 0])
        msd_labels         = np.array(msd_labels)
        msd_mean           = np.mean(msd_labels[msd_labels > 0])
        return hausdorff_mean, hausdorff_labels, hausdorff95_mean, hausdorff95_labels, msd_mean, msd_labels

    except:
        traceback.print_exc()
        return -1, [], -1, []

def dice_numpy_slice(y_true_slice, y_pred_slice):
    """
    Specifically designed for 2D slices
    
    Params
    ------
    y_true_slice: [H,W]
    y_pred_slice: [H,W]
    """

    sum_true = np.sum(y_true_slice)
    sum_pred = np.sum(y_pred_slice)
    if sum_true > 0 and sum_pred > 0:
        num = 2 * np.sum(y_true_slice *y_pred_slice)
        den = sum_true + sum_pred
        return num/den
    elif sum_true > 0 and sum_pred == 0:
        return 0
    elif sum_true == 0 and sum_pred > 0:
        return -0.1
    else:
        return -1

def dice_numpy(y_true, y_pred):
    """
    :param y_true: [H, W, D, L]
    :param y_pred: [H, W, D, L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """

    dice_labels = []
    for label_id in range(y_pred.shape[-1]):
        
        y_true_label = y_true[:,:,:,label_id]        # [H,W,D]
        y_pred_label = y_pred[:,:,:,label_id] + 1e-8 # [H,W,D]

        # Calculate loss (over all pixels)
        if np.sum(y_true_label) > 0:
            num = 2*np.sum(y_true_label * y_pred_label)
            den = np.sum(y_true_label + y_pred_label)
            dice_label = num/den
        else:
            dice_label = -1.0

        dice_labels.append(dice_label)
    
    dice_labels = np.array(dice_labels)
    dice = np.mean(dice_labels[dice_labels>0])
    return dice, dice_labels


############################################################
#                            AVU                           #
############################################################

#################### Generic ####################

@tf.function
def update_ypred_mask(y_pred_class_id, label_mask, batch_range):
    """
    Params
    ------
    y_pred_class_id: [B,H,W,D]
    label_mask     : [B,L]
    batch_range    : list 
    """

    print(' - [losses] Table lookup for y_pred_mask! ')
    label_mask      = tf.cast(label_mask, tf.int32)
    label_mask_vals = tf.range(label_mask.shape[-1], dtype=tf.int32) # e.g. tf.range(10)
    
    tf.print(' - label_mask: ', label_mask)
    tf.print(label_mask.shape)
    tf.print(y_pred_class_id.shape)

    y_pred_mask = []
    for batch_id in batch_range: # NB: Bad code
        y_pred_mask_batch =  tf.expand_dims(tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(label_mask_vals, label_mask[batch_id], tf.int32, tf.int32), default_value=0).lookup(y_pred_class_id[batch_id]), axis=0)
        if len(y_pred_mask) == 0: y_pred_mask = y_pred_mask_batch
        else                    : y_pred_mask = tf.concat([y_pred_mask, y_pred_mask_batch], axis=0)

    y_pred_class_id = y_pred_class_id *  y_pred_mask # [B,H,W,D] (only considers those predictions that have a GT available)

    return y_pred_class_id

@tf.function
def get_distmap(y_true_class_binary_batch):
    """
    Params
    ------
    y_true_class_binary: [H,W,D,1]
    """
    
    y_true_distmap_out = tf.expand_dims(tf.transpose(tfa.image.euclidean_dist_transform(tf.cast(tf.transpose(1 - y_true_class_binary_batch, perm=(2,0,1,3)), tf.uint8)), perm=(1,2,0,3)), axis=0)  # [H,W,D,1] -> [D,H,W,1] -> [H,W,D,1] -> [1,H,W,D,1]
    y_true_distmap_in  = tf.expand_dims(tf.transpose(tfa.image.euclidean_dist_transform(tf.cast(tf.transpose(y_true_class_binary_batch, perm=(2,0,1,3)), tf.uint8)), perm=(1,2,0,3)), axis=0)      # [H,W,D,1] -> [D,H,W,1] -> [H,W,D,1] -> [1,H,W,D,1]
    y_true_distmap     = y_true_distmap_out + y_true_distmap_in

    return y_true_distmap

@tf.function
def get_y_true_pred_dilated(y_true_class_id, y_pred_class_id, dilation_ksize):

    print (' - [losses] Mask = GT + AI dilation')
    print (' - [losses] dilation_ksize: ', dilation_ksize)

    y_true_class_binary  = tf.expand_dims(tf.where(tf.math.greater(tf.cast(y_true_class_id, dtype=tf.float32), 0.0), 1.0, 0.0), axis=-1) # [B,H,W,D] -> [B,H,W,D,1] (add channel dimenion for pooling purposes)
    y_pred_class_binary  = tf.expand_dims(tf.where(tf.math.greater(tf.cast(y_pred_class_id, dtype=tf.float32), 0.0), 1.0, 0.0), axis=-1) # [B,H,W,D] -> [B,H,W,D,1]
    y_true_class_binary_dilated = tf.nn.max_pool3d(y_true_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')                          # [B,H,W,D,1]
    y_pred_class_binary_dilated = tf.nn.max_pool3d(y_pred_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')                          # [B,H,W,D,1]
    y_true_pred_binary_dilated  = tf.cast(tf.cast(y_true_class_binary_dilated + y_pred_class_binary_dilated, dtype=tf.bool), tf.float32) # [B,H,W,D,1]

    return y_true_pred_binary_dilated[:,:,:,:,0]

@tf.function
def get_hi_lo_err_masks(y_true_class_id, y_pred_class_id, dilation_ksize, error_ksize, SPACING_XY, DIST_THRESHOLD):

    batch_size = 2

    print (' - [losses] Mask = HiErr (with surface dist) + LoErr')
    print (' - [losses] dilation_ksize: ', dilation_ksize, ' || error_ksize: ', error_ksize)

    # Step 1 - Get the dilated versions of the GT and prediction (NB: prediction has already had organs zeroed if there is no GT)
    y_true_class_binary  = tf.expand_dims(tf.where(tf.math.greater(tf.cast(y_true_class_id, dtype=tf.float32), 0.0), 1.0, 0.0), axis=-1) # [B,H,W,D] -> [B,H,W,D,1] (add channel dimenion for pooling purposes)
    y_pred_class_binary  = tf.expand_dims(tf.where(tf.math.greater(tf.cast(y_pred_class_id, dtype=tf.float32), 0.0), 1.0, 0.0), axis=-1) # [B,H,W,D] -> [B,H,W,D,1]
    y_true_class_binary_dilated = tf.nn.max_pool3d(y_true_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')                          # [B,H,W,D,1]
    y_pred_class_binary_dilated = tf.nn.max_pool3d(y_pred_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')                          # [B,H,W,D,1]
    y_true_pred_binary_dilated  = tf.cast(tf.cast(y_true_class_binary_dilated + y_pred_class_binary_dilated, dtype=tf.bool), tf.float32) # [B,H,W,D,1]

    # Step 2  - Calculate high error areas  
    y_error_areas          = tf.math.abs(y_true_class_binary - y_pred_class_binary)                                     # [B,H,W,D,1]
    y_error_areas_eroded   = -tf.nn.max_pool3d(-y_error_areas, ksize=error_ksize, strides=1, padding='SAME')            # [B,H,W,D,1]
    y_error_areas_high     = tf.nn.max_pool3d(y_error_areas_eroded, ksize=error_ksize, strides=1, padding='SAME')       # [B,H,W,D,1]

    y_true_distmap                  = tf.concat([get_distmap(y_true_class_binary[batch_id]) for batch_id in range(batch_size)], axis=0)  # tf.concat([1,H,W,D,1]) --> [B,H,W,D,1]
    y_true_distmap_errorarea        = y_true_distmap * y_error_areas * SPACING_XY # in x-y direction only                       # [B,H,W,D,1]
    y_true_distmap_errorarea_binary = tf.math.greater_equal(y_true_distmap_errorarea, DIST_THRESHOLD)
    y_error_areas_high              = tf.cast(tf.math.logical_or(tf.cast(y_error_areas_high, tf.bool), y_true_distmap_errorarea_binary), dtype=tf.float32) 

    # Step 3 - Calculate low error areas
    y_nonerror_areas           = tf.math.abs(y_true_pred_binary_dilated - y_error_areas_high)                           # [B,H,W,D,1]
    y_error_areas_low          = y_nonerror_areas

    return y_error_areas_high, y_error_areas_low

#################### AvU ####################

VERBOSITY_PROB_AVU = 0.001 # [0.001, 0.005, 0.2]
VERBOSITY_PROB = 0.001

@tf.function
def get_avu_threshold(array_ac, array_au, array_ic, array_iu, y_pred_unc, thresh_uncer, RATIO_N_AC):

    # Step 1 - Get N's for a particular unc threshold
    y_certain_mask      = tf.where(tf.math.less_equal(y_pred_unc, thresh_uncer), 1.0, 0.0)
    y_uncertain_mask    = tf.where(tf.math.greater(y_pred_unc, thresh_uncer)   , 1.0, 0.0)
    n_ac = tf.math.reduce_sum(array_ac   * y_certain_mask   , axis=(1,2,3)) # [B,H,W,D] -> [B]
    n_au = tf.math.reduce_sum(array_au   * y_uncertain_mask , axis=(1,2,3)) # [B,H,W,D] -> [B]
    n_ic = tf.math.reduce_sum(array_ic   * y_certain_mask   , axis=(1,2,3)) # [B,H,W,D] -> [B]
    n_iu = tf.math.reduce_sum(array_iu   * y_uncertain_mask , axis=(1,2,3)) # [B,H,W,D] -> [B]
    
    res = tf.math.log(1 + ((n_au + n_ic) / (n_ac*RATIO_N_AC + n_iu + config._EPSILON)) ) # [B] -> [B] # values should be in the range of [ln(1), ln(1+[0.1,0.5])] -> [0, [0.09, 0.4]]
    res = tf.expand_dims(res, axis=-1)                                         # [B] -> [B,1] this last dimension will store values for different thresholds

    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= VERBOSITY_PROB_AVU:
        tf.print('\n -------------- ** -------------- ')
        tf.print(' - [DEBUG][get_avu_threshold()] thresh_uncer: ', thresh_uncer, ' || ~n_ac*RATIO_N_AC:', n_ac*RATIO_N_AC, ' || ~n_au: ', n_au, ' || ~n_ic: ', n_ic, ' || ~n_iu: ', n_iu)
        tf.print(' - [DEBUG][get_avu_threshold()]: ln(1 + ((n_au + n_ic) / (n_ac*RATIO_N_AC + n_iu))): ', res)
        tf.print(' -------------- ** -------------- \n')
    
    return res

@tf.function
def get_pu_threshold(array_ac, array_au, array_ic, array_iu, y_pred_unc, thresh_uncer):

    """
    Returns p(u|i) and p(u|a) at a specific uncertainty threshold to eventually calculate ln(1 + (p(u|a) / p(u|i)))
    """
    
    # Step 1 - Get N's for a particular unc threshold
    y_certain_mask      = tf.where(tf.math.less_equal(y_pred_unc, thresh_uncer), 1.0, 0.0)
    y_uncertain_mask    = tf.where(tf.math.greater(y_pred_unc, thresh_uncer)   , 1.0, 0.0)
    n_ac = tf.math.reduce_sum(array_ac   * y_certain_mask   , axis=(1,2,3)) # [B,H,W,D] -> [B]
    n_au = tf.math.reduce_sum(array_au   * y_uncertain_mask , axis=(1,2,3)) # [B,H,W,D] -> [B]
    n_ic = tf.math.reduce_sum(array_ic   * y_certain_mask   , axis=(1,2,3)) # [B,H,W,D] -> [B]
    n_iu = tf.math.reduce_sum(array_iu   * y_uncertain_mask , axis=(1,2,3)) # [B,H,W,D] -> [B]    

    # Step 2 - Calculate p_u terms
    p_ua = n_au / (n_ac + n_au + config._EPSILON) # [B]
    p_ui = n_iu / (n_iu + n_ic + config._EPSILON) # [B]

    # Step 3 - Some array magic
    p_ua  = tf.expand_dims(p_ua, axis=-1)   # [B,1]
    p_ui  = tf.expand_dims(p_ui, axis=-1)   # [B,1]
    res   = tf.expand_dims(tf.concat([p_ua, p_ui],axis=-1), axis=-1) # [B,2,1] values: e.g. res[:,:,[0=p(u|a),1=p(u|i)],:] should always be in [0,1] range

    # Step 4 - Some verbosity
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= VERBOSITY_PROB:
        tf.print('')
        tf.print(' - [DEBUG][get_pu_threshold()] thresh_uncer: ', thresh_uncer, ' || n_ac:', n_ac, ' || n_au: ', n_au, ' || n_ic: ', n_ic, ' || n_iu: ', n_iu)
        tf.print(' - [DEBUG][get_pu_threshold()]: [p_ua, p_ui]: ', res)

    return res # [B,2,1] --> need to do res[:,0,:] / res[:,1,:] i.e. p(u|a) / p(u|i)

@tf.function
def loss_avu_3d_tf_func(y_true, y_pred, y_pred_unc=None, label_mask=[], weights=[]
                , thresh_uncer=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
                , dilation_ksize=(5,5,1), error_ksize=(3,3,1), verbose=False):
    """
    Ref: Improving model calibration with accuracy versus uncertainty optimization, NeurIPS 2020
    Diff: I apply losses to HiErr and LoErr regions extracted from dilated RoI GT + predictions 

    Params
    ------
    y_true    : [B,H,W,D,L]
    y_pred    : [B,H,W,D,L]
    y_pred_unc: [B,H,W,D]
    label_mask: [B,L]
    thresh_uncer: list
    """

    dataset = 'Prostate' # ['MICCAI', 'Prostate']
    print (' - [loss_avu_3d_tf_func()] dataset: ', dataset)

    if dataset == 'HeadAndNeck':
        SPACING_XY     = 0.8 # (0.8, 0.8, 2.5) for MICCAI
        DIST_THRESHOLD = 3 # (in mm)
        RATIO_N_AC     = 1.0 # [1 if MC=1 & entropy, 0.01 if MC=5 & MI]
        DRIFT          = False #[ False, True]
        DRIFT_COEFF    = 1.0 # [0, 1, 10, 100, 1000]
    elif dataset == 'Prostate':
        SPACING_XY     = 0.5 # (0.5, 0.5, 3.0) for Prostate
        DIST_THRESHOLD = 3 # (in mm)
        RATIO_N_AC     = 1.0 # [0.1, 1.0]
        DRIFT          = False #[ False, True]
        DRIFT_COEFF    = 1.0 # [0, 1, 10, 100, 1000]

    print (' - [loss_avu_3d_tf_func()] Doing AvU loss from paper')
    print (' --- dilation_ksize: ', dilation_ksize)
    print (' --- error_ksize   : ', error_ksize)
    print (' --- RATIO_N_AC    : ', RATIO_N_AC)
    print (' --- DIST_THRESHOLD: ', DIST_THRESHOLD)
    print (' --- SPACING_XY    :', SPACING_XY)
    print (' --- thresh_uncer: ', thresh_uncer)
    print (' --- DRIFT         : ', DRIFT)
    if DRIFT:
        print (' --- DRIFT_COEFF   : ', DRIFT_COEFF)
    print ('')

    # Step 1 - Get class_id and max prob for each voxel
    y_pred            = y_pred + config._EPSILON
    if dataset == 'HeadAndNeck':
        print (' - Dealing with HeadAndNeck class')
        y_true_class_id   = tf.argmax(y_true, axis=-1, output_type=tf.int32)          # [B,H,W,D,L] -> [B,H,W,D]
        y_pred_class_id   = tf.argmax(y_pred, axis=-1, output_type=tf.int32)          # [B,H,W,D,L] -> [B,H,W,D]
        y_pred_class_prob = tf.math.reduce_max(y_pred, axis=-1)                       # [B,H,W,D,L] -> [B,H,W,D]
    elif dataset == 'Prostate':
        print (' - Dealing with prostate class')
        y_true_class_id = tf.where(tf.math.greater(y_true, 0.5), 1., 0.)[:,:,:,:,0]
        y_pred_class_id = tf.where(tf.math.greater(y_pred, 0.5), 1., 0.)[:,:,:,:,0]
        y_pred_class_prob = y_pred[:,:,:,:,0]

    if y_pred_unc is None:
        print (' - [loss_avu_3d_tf_func()] Using y_pred_unc as uncertainty (i.e. entropy) calculated within the loss function')
        y_pred_unc        = -tf.math.reduce_sum(y_pred*tf.math.log(y_pred), axis=-1)  # [B,H,W,D,L] -> [B,H,W,D]
    else:
        print (' - [loss_avu_3d_tf_func()] Using y_pred_unc from parameters')
    
    # Step 2 - Incorporate mask
    batch_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(label_mask[:,0], axis=-1), axis=-1), axis=-1) # [B,L] --> [B] -> [B,1,1,1] pick the background mask, if=1, then consider both y_true and y_pred, else ignore it.
    
    if 0:
        tf.print(' - [loss_avu_3d_tf_func()] sum(y_pred_unc): ', tf.math.reduce_sum(y_pred_unc, axis=(1,2,3)), ' || unc_max: ', tf.math.reduce_max(y_pred_unc, axis=(1,2,3)))

    if tf.math.reduce_sum(batch_mask) > 0:

        # Step 3 - Get Accurate and Inaccurate masks
        if 1:
            # Hi-Lo Err masks
            y_error_areas_high, y_error_areas_low = get_hi_lo_err_masks(y_true_class_id, y_pred_class_id, dilation_ksize, error_ksize, SPACING_XY, DIST_THRESHOLD)
            y_accurate_masked   = y_error_areas_low[:,:,:,:,0]  * batch_mask   # [B,H,W,D,1] -> [B,H,W,D]
            y_inaccurate_masked = y_error_areas_high[:,:,:,:,0] * batch_mask
        else:
            # AI + GT dilated mask
            y_mask              = get_y_true_pred_dilated(y_true_class_id, y_pred_class_id, dilation_ksize) * batch_mask # [B,H,W,D]
            y_accurate_masked   = tf.where(tf.math.equal(y_true_class_id, y_pred_class_id), 1.0, 0.0) * y_mask           # [B,H,W,D]
            y_inaccurate_masked = tf.where(tf.math.equal(y_true_class_id, y_pred_class_id), 0.0, 1.0) * y_mask           # [B,H,W,D]

        # Step 4.1 - Calculate n_ac, n_au, n_ic, n_iu terms at a specific threshold(s) 
        array_ac = y_accurate_masked   * y_pred_class_prob         * (1.0 - tf.math.tanh(y_pred_unc))  # [B,H,W,D]
        array_au = y_accurate_masked   * y_pred_class_prob         * (tf.math.tanh(y_pred_unc))        # [B,H,W,D]
        array_ic = y_inaccurate_masked * (1.0 - y_pred_class_prob) * (1.0 - tf.math.tanh(y_pred_unc))  # [B,H,W,D]
        array_iu = y_inaccurate_masked * (1.0 - y_pred_class_prob) * (tf.math.tanh(y_pred_unc))        # [B,H,W,D]
        avus = tf.concat([get_avu_threshold(array_ac, array_au, array_ic, array_iu, y_pred_unc, thresh_unc, RATIO_N_AC)  for thresh_unc in thresh_uncer], axis=-1) # [B, len(thresh_uncer)]

        # Step 4.2 - Accuracy vs Uncertainty Loss
        res = tf.math.reduce_sum(tf.math.reduce_mean(avus, axis=-1))/tf.math.reduce_sum(batch_mask) # [B, len(thresh_uncer)] -> [B] -> [1]

        # Step 5 - P(u|a) -- reduce this
        if DRIFT:
            print (' - [loss_auc_3d_tf_func()] Adding term i.e. ln(1 + p_ua) to prevent upward drift of p_au')
            p_u_list = tf.concat([get_pu_threshold(array_ac, array_au, array_ic, array_iu, y_pred_unc, thresh_unc) for thresh_unc in thresh_uncer], axis=-1) # [B, 2, len(thresh_uncer)]
            p_ua = tf.math.log(1 + p_u_list[:,0,:])                                                       # ln(1 + p_ua) --> [B,1,len(thresh_uncer)]
            res2 = tf.math.reduce_sum(tf.math.reduce_mean(p_ua, axis=-1))/tf.math.reduce_sum(batch_mask)  # [B, 1, len(thresh_uncer)] -> [B,1] -> [1]
            res  = res + DRIFT_COEFF*res2

        # Step 6 - Some verbosity
        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= VERBOSITY_PROB_AVU:
            tf.print('\n\n ----------------------------- ')
            tf.print(' - [DEBUG][loss_avu_3d_tf_func()] min_unc: ', tf.math.reduce_min(y_pred_unc, axis=[1,2,3]), ' || max_unc: ', tf.math.reduce_max(y_pred_unc, axis=[1,2,3]))
            tf.print(' - [DEBUG][loss_avu_3d_tf_func()] avus       :', avus)
            tf.print(' - [DEBUG][loss_avu_3d_tf_func()]: mean(avus): ', tf.math.reduce_mean(avus, axis=-1))
            tf.print(' - [DEBUG][loss_avu_3d_tf_func()]: avu       : ', tf.math.reduce_sum(tf.math.reduce_mean(avus, axis=-1))/tf.math.reduce_sum(batch_mask))
            if DRIFT:
                tf.print('\n - [DEBUG][loss_auc_3d_tf_func()] log(1+p(u|a))       : ', p_ua)
                tf.print('\n - [DEBUG][loss_auc_3d_tf_func()] avg_t[log(1+p(u|a))]: ', tf.math.reduce_mean(p_ua, axis=-1))
                tf.print('\n - [DEBUG][loss_auc_3d_tf_func()]  res2               : ', res2)
            tf.print(' ----------------------------- \n\n')

    else:
        res = 0.0

    return res, [], [], []             

############################################################
#                           LOSSES                         #
############################################################

@tf.function
def loss_dice_3d_tf_func(y_true, y_pred, label_mask, weights=[], verbose=False):

    """
    Calculates soft-DICE loss

    :param y_true: [B, H, W, C, L]
    :param y_pred: [B, H, W, C, L] 
    :param label_mask: [B,L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """
    
    # Step 0 - Init
    dice_labels = []
    label_mask = tf.cast(label_mask, dtype=tf.float32) # [B,L]

    # Step 1 - Get DICE of each sample in each label
    y_pred      = y_pred + config._EPSILON
    dice_labels = (2*tf.math.reduce_sum(y_true * y_pred, axis=[1,2,3]))/(tf.math.reduce_sum(y_true + y_pred, axis=[1,2,3])) # [B,H,W,D,L] -> [B,L]
    dice_labels = dice_labels*label_mask # if mask of a label (e.g. background) has been explicitly set to 0, do not consider its loss
        
    # Step 2 - Mask results on the basis of ground truth availability
    label_mask             = tf.where(tf.math.greater(label_mask,0), label_mask, config._EPSILON) # to handle division by 0
    dice_for_train         = None
    dice_labels_for_train  = None
    dice_labels_for_report = tf.math.reduce_sum(dice_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    dice_for_report        = tf.math.reduce_mean(tf.math.reduce_sum(dice_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    
    # Step 3 - Weighted DICE
    if len(weights):
        label_weights = weights / tf.math.reduce_sum(weights) # nomalized
        dice_labels_w = dice_labels * label_weights
        dice_labels_for_train = tf.math.reduce_sum(dice_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        dice_for_train = tf.math.reduce_mean(tf.math.reduce_sum(dice_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    else:
        dice_labels_for_train = dice_labels_for_report
        dice_for_train = dice_for_report

    # Step 4 - Return results
    return 1.0 - dice_for_train, 1.0 - dice_labels_for_train, dice_for_report, dice_labels_for_report

@tf.function
def loss_ce_3d_tf_func(y_true, y_pred, label_mask, weights=[], verbose=False):
    """
    Calculates cross entropy loss

    Params
    ------
    y_true    : [B, H, W, C, L]
    y_pred    : [B, H, W, C, L] 
    label_mask: [B,L]
    - Ref: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    """
    
    # Step 0 - Init
    loss_labels = []
    label_mask = tf.cast(label_mask, dtype=tf.float32)

    # Step 1.1 - Foreground loss
    loss_labels_pos = -1.0 * y_true * tf.math.log(y_pred + config._EPSILON) # [B,H,W,D,L]
    loss_labels_pos = label_mask * tf.math.reduce_sum(loss_labels_pos, axis=[1,2,3]) # [B,H,W,D,L] --> [B,L]

    # Step 1.2 - Background loss
    loss_labels_neg   = -1.0 * (1 - y_true) * tf.math.log(1 - y_pred + config._EPSILON) # [B,H,W,D,L]
    loss_labels_neg   = label_mask * tf.math.reduce_sum(loss_labels_neg, axis=[1,2,3]) # [B,H,W,D,L] --> [B,L]
    loss_labels       = loss_labels_pos + loss_labels_neg # [B,L]
    
    # Step 2 - Mask results on the basis of ground truth availability
    label_mask = tf.where(tf.math.greater(label_mask,0), label_mask, config._EPSILON) # for reasons of division
    loss_for_train = None
    loss_labels_for_train = None
    loss_labels_for_report = tf.math.reduce_sum(loss_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0) # [B,L] -> [L], [B,L] -> [L], [L]/[L] = [L]
    loss_for_report = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1)) # [B,L] -> [B], [B,L] -> [B], mean([B]) -> [1]
    
    # Step 3 - Weighted DICE
    if len(weights):
        label_weights = weights / tf.math.reduce_sum(weights) # normalized    
        # [TODO] tf.print(' - [loss_ce_3d_tf_func] label_weights: ', label_weights)
        # tf.print(' - [loss_ce_3d_tf_func] loss_labels: ', loss_labels)
        loss_labels_w = loss_labels * label_weights
        # tf.print(' - [loss_ce_3d_tf_func] loss_labels: ', loss_labels_w)
        loss_labels_for_train = tf.math.reduce_sum(loss_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) # [L]
        loss_for_train = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1)) # [1]
        # tf.print(' - [loss_ce_3d_tf_func] loss_for_train: ', loss_for_train)
    else:
        loss_labels_for_train = loss_labels_for_report
        loss_for_train = loss_for_report
    
    # Step 4 - Return results
    return loss_for_train, loss_labels_for_train, loss_for_report, loss_labels_for_report

@tf.function
def loss_cebasic_3d_tf_func(y_true, y_pred, label_mask, weights=[], verbose=False):
    """
    Calculates cross entropy loss

    Params
    ------
    y_true    : [B, H, W, C, L]
    y_pred    : [B, H, W, C, L] 
    label_mask: [B,L]
    - Ref: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    """
    
    # Step 0 - Init
    loss_labels = []
    label_mask = tf.cast(label_mask, dtype=tf.float32)
    y_pred     = y_pred + config._EPSILON

    # Step 1 - Foreground loss
    loss_labels = -1.0 * y_true * tf.math.log(y_pred) # [B,H,W,D,L]
    loss_labels = label_mask * tf.math.reduce_sum(loss_labels, axis=[1,2,3]) # [B,H,W,D,L] --> [B,L]
    
    # Step 2 - Mask results on the basis of ground truth availability
    label_mask = tf.where(tf.math.greater(label_mask,0), label_mask, config._EPSILON) # for reasons of division
    loss_for_train = None
    loss_labels_for_train = None
    loss_labels_for_report = tf.math.reduce_sum(loss_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    loss_for_report = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))

    # Step 3 - Weighted DICE
    if len(weights):
        label_weights = weights / tf.math.reduce_sum(weights) # nomalized    
        loss_labels_w = loss_labels * label_weights
        loss_labels_for_train = tf.math.reduce_sum(loss_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        loss_for_train = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    else:
        loss_labels_for_train = loss_labels_for_report
        loss_for_train = loss_for_report
    
    # Step 4 - Return results
    return loss_for_train, loss_labels_for_train, loss_for_report, loss_labels_for_report