
# Import internal libraries
import src.config as config

# Import external libraries
import sys
import pdb
import nrrd
import time
import scipy
import pandas as pd
import sklearn.metrics
import traceback
import numpy as np
import scipy.stats
from collections import Counter
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

XAXIS_LOG = False
EPSILON = 1e-8

RATIO_KEY = 'Inacc/Acc Ratio'
PLOT_BAR = 'bar'
PLOT_SWARM = 'swarm'
PLOT_BARSWARM = 'barswarm'

################################################### Generic ###################################################

def do_erosion_dilation(vol, ksize=(5,5,1)):
    """
    vol: [H,W,D]
    ksize: tuple of size 3
    """
    
    print (' - [do_erosion_dilation()] Performing on y_unc with kernel=', ksize)
    vol_tf        = tf.expand_dims(tf.expand_dims(tf.constant(vol, tf.float32), axis=-1),axis=0)
    vol_tf_eroded = -tf.nn.max_pool3d(-vol_tf, ksize=ksize, strides=1, padding='SAME')
    vol_tf        = tf.nn.max_pool3d(vol_tf_eroded, ksize=ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy()

    return vol_tf

def get_dilated_y_true(y_true_class, y_pred_class, dilation_ksize):
    """
    Calculate mask for areas around GT + AI organs (essentially ignores bgd)
    """
    
    print ('\n ===========================================================================')
    print (' - [get_dilated_y_true()] Doing this with dilation_ksize: ', dilation_ksize)
    print (' ===========================================================================\n')
    y_true_class_binary = np.array(y_true_class, copy=True)
    if mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
        y_true_class_binary[y_true_class_binary == 2] = 0 # set optic chiasm to gt=0(bgd); this ignored in next line
    y_true_class_binary[y_true_class_binary > 0] = 1
    y_true_class_binary = tf.constant(y_true_class_binary, tf.float32)
    y_true_class_binary = tf.constant(tf.expand_dims(tf.expand_dims(y_true_class_binary, axis=-1),axis=0))
    y_true_class_binary_dilated  = tf.nn.max_pool3d(y_true_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy()

    y_pred_class_binary = np.array(y_pred_class, copy=True)
    if mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
        y_pred_class_binary[y_pred_class_binary == 2] = 0 # set optic chiasm to pred=0(bgd); this ignored in next line
    y_pred_class_binary[y_pred_class_binary > 0] = 1
    y_pred_class_binary = tf.constant(y_pred_class_binary, tf.float32)
    y_pred_class_binary = tf.expand_dims(tf.expand_dims(y_pred_class_binary, axis=-1),axis=0)

    y_mask = y_true_class_binary_dilated
    
    return y_mask, y_true_class_binary, y_pred_class_binary

def get_dilated_y_true_pred(y_true_class, y_pred_class, dilation_ksize):
    """
    Calculate mask for areas around GT + AI organs (essentially ignores bgd)

    Params
    ------
    y_true_class: [H,W,D]
    y_pred_class: [H,W,D]

    Thoughts
    --------
    - If the mask is determined by GT + Pred, then how do we do attain homogenity in the evaluation? Should we not just use a really dilated gt mask 
     and hope that all predictions fall under this?  
    - For ECE, I already did this! So why not for unc-ROC?
    -- Am I doing this for cases where I need to consider the SMD preds in the DTCIA?
    """
    
    print ('\n ===========================================================================')
    print (' - [get_dilated_y_true_pred()] Doing this with dilation_ksize: ', dilation_ksize)
    print (' ===========================================================================\n')

    y_true_class_binary = np.array(y_true_class, copy=True)
    if mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
        y_true_class_binary[y_true_class_binary == 2] = 0 # set optic chiasm to gt=0(bgd); this ignored in next line
    if mode == config.MODE_STRUCTSEG:
        y_true_class_binary[y_true_class_binary == 8] = 0
        y_true_class_binary[y_true_class_binary == 9] = 0

    y_true_class_binary[y_true_class_binary > 0] = 1
    y_true_class_binary = tf.constant(y_true_class_binary, tf.float32)
    y_true_class_binary = tf.constant(tf.expand_dims(tf.expand_dims(y_true_class_binary, axis=-1),axis=0))
    y_true_class_binary_dilated  = tf.nn.max_pool3d(y_true_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')
    
    y_pred_class_binary = np.array(y_pred_class, copy=True)
    if mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
        y_pred_class_binary[y_pred_class_binary == 2] = 0 # set optic chiasm to pred=0(bgd); this ignored in next line
    if mode == config.MODE_STRUCTSEG:
        y_pred_class_binary[y_pred_class_binary == 8] = 0
        y_pred_class_binary[y_pred_class_binary == 9] = 0
    y_pred_class_binary[y_pred_class_binary > 0] = 1
    y_pred_class_binary = tf.constant(y_pred_class_binary, tf.float32)
    y_pred_class_binary = tf.expand_dims(tf.expand_dims(y_pred_class_binary, axis=-1),axis=0)
    y_pred_class_binary_dilated  = tf.nn.max_pool3d(y_pred_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')

    y_true_pred_binary_dilated = (y_true_class_binary_dilated + y_pred_class_binary_dilated)[0,:,:,:,0].numpy()
    y_true_pred_binary_dilated[y_true_pred_binary_dilated > 1] = 1
    y_mask = y_true_pred_binary_dilated

    return y_mask, y_true_class_binary, y_pred_class_binary # it says class, but there is no concept of class in this array

def get_accurate_inaccurate_avupaper(y_true_class, y_pred_class, y_mask):
    """
    Params
    ------
    y_true_class: np.array of dim=[H,W,D] and type=float containing values in the range [0,9]
    y_pred_class: np.array of dim=[H,W,D] and type=float containing values in the range [0,9]
    y_mask      : np.array of dim=[H,W,D] and type=float containing values in the range [0,1]
    """

    print (' - [INFO] Old Accurate/Inaccurate calculation as done in AvU paper.')
    y_accurate    = (y_true_class == y_pred_class).astype(np.uint8)  # [H,W,D]
    y_inaccurate  = 1 - y_accurate
    print (' - True            : Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))
    y_accurate    = y_accurate * y_mask
    y_inaccurate  = y_inaccurate * y_mask
    print (' - RT-Specific Mask: Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))

    return y_accurate, y_inaccurate

def get_accurate_inaccurate_via_erosion(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize):

    print (' - [INFO] New HiErr and new LoErr calculation')
    y_error_areas          = tf.math.abs(y_true_class_binary - y_pred_class_binary)
    y_error_areas_eroded   = -tf.nn.max_pool3d(-y_error_areas, ksize=error_ksize, strides=1, padding='SAME')
    y_error_areas_high     = tf.nn.max_pool3d(y_error_areas_eroded, ksize=error_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy()

    y_nonerror_areas           = y_mask - y_error_areas_high
    y_nonerror_areas           = tf.expand_dims(tf.expand_dims(tf.constant(y_nonerror_areas, tf.float32), axis=-1),axis=0)  # [TODO: Why am I doing erosion-dilation on these areas? Note: I dont this in the get_accurate_inaccurate_via_erosion_and_surface()]
    y_nonerror_areas_eroded    = -tf.nn.max_pool3d(-y_nonerror_areas, ksize=error_ksize, strides=1, padding='SAME')
    y_error_areas_low          = tf.nn.max_pool3d(y_nonerror_areas_eroded, ksize=error_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy() # No errpr + Low Error

    y_accurate    = (y_true_class == y_pred_class).astype(np.uint8)  # [H,W,D]
    y_inaccurate  = 1 - y_accurate
    y_accurate    = y_accurate * y_mask
    y_inaccurate  = y_inaccurate * y_mask
    print (' --- RT-Specific Mask          : Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))

    y_accurate   = y_error_areas_low
    y_inaccurate = y_error_areas_high
    print (' --- RT-Specific Mask (with ~a): Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))

    return y_accurate, y_inaccurate

def get_accurate_inaccurate_via_erosion_and_surface(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize, spacing, distance_max=3):
    
    print (' - [INFO] New HiErr calculation (with distance maps) and new LoErr calculation')

    # Step 1 - Get erroneous areas (and then find high error areas from them via morphological operations)
    y_error_areas          = tf.math.abs(y_true_class_binary - y_pred_class_binary)
    y_error_areas_eroded   = -tf.nn.max_pool3d(-y_error_areas, ksize=error_ksize, strides=1, padding='SAME')
    y_error_areas_high     = tf.nn.max_pool3d(y_error_areas_eroded, ksize=error_ksize, strides=1, padding='SAME')
    print (' - before-sum(y_error_areas_high): ', tf.math.reduce_sum(y_error_areas_high))

    # Step 2.1 - Calculate distance maps (in 2D), multiply them by the error mask (y_error_areas). This output is then multiplied by spacing in the xy dimension to give distance values for each erroneous pixel
    ## This is done to handle very specific cases (something to do with parotid gland)
    y_true_distmap_out = tf.expand_dims(tf.transpose(tfa.image.euclidean_dist_transform(tf.cast(tf.transpose(1 - y_true_class_binary[0], perm=(2,0,1,3)), tf.uint8)), perm=(1,2,0,3)), axis=0)  # [1,H,W,D,1] -> [H,W,D,1] -> [D,H,W,1] -> [H,W,D,1] -> [1,H,W,D,1]
    y_true_distmap_in  = tf.expand_dims(tf.transpose(tfa.image.euclidean_dist_transform(tf.cast(tf.transpose(y_true_class_binary[0], perm=(2,0,1,3)), tf.uint8)), perm=(1,2,0,3)), axis=0)      # [1,H,W,D,1] -> [H,W,D,1] -> [D,H,W,1] -> [H,W,D,1] -> [1,H,W,D,1]
    y_true_distmap     = y_true_distmap_out + y_true_distmap_in                                                                                                                                 # [1,H,W,D,1]
    y_true_distmap_errorarea = y_error_areas * y_true_distmap * spacing[0]

    # Step 2.2 - Apply logical OR operation and get final HIGH erroneous pixels
    y_true_distmap_errorarea_binary = tf.math.greater_equal(y_true_distmap_errorarea, distance_max)
    y_error_areas_high              = tf.cast(tf.math.logical_or(tf.cast(y_error_areas_high, tf.bool), y_true_distmap_errorarea_binary), dtype=tf.float32)[0,:,:,:,0].numpy()
    print (' - after-sum(y_error_areas_high) : ', tf.math.reduce_sum(y_error_areas_high))

    # Step 3 - Anything not high error is now considered low error
    y_nonerror_areas           = y_mask - y_error_areas_high  # Note: If there are -1's in this, that means that y_mask is not sufficient enough to cover the predictions
    y_error_areas_low          = y_nonerror_areas  # No error + Low Error
    print (' - Counter(y_error_areas_low): ', Counter(y_error_areas_low.flatten()))
    y_error_areas_low[y_error_areas_low < 0] = 0 # set all non-captured errors as background and hence they will be considered as accurate. Note: Not the best solution 

    y_accurate    = (y_true_class == y_pred_class).astype(np.uint8)  # [H,W,D]
    y_inaccurate  = 1 - y_accurate
    y_accurate    = y_accurate * y_mask
    y_inaccurate  = y_inaccurate * y_mask
    print (' - RT-Specific Mask                 : Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))

    y_accurate   = y_error_areas_low
    y_inaccurate = y_error_areas_high
    print (' - RT-Specific Mask (with ~a + surf): Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))

    return y_accurate, y_inaccurate

def get_unc_norm(y_unc):
    y_unc_norm = np.array(y_unc, copy=True)
    if 0:
        y_unc_norm = y_unc_norm / np.max(y_unc_norm)
        print (' --- [DEBUG] Doing patient-wise uncertainty normalization || max: {:.2f}'.format(np.max(y_unc)))
    elif 1:
        print (' --- [DEBUG] NOTTT Doing patient-wise uncertainty normalization || max: {:.2f} | sum: {:.2f}'.format(np.max(y_unc), np.sum(y_unc)))
    
    return y_unc_norm

################################################### PAvPU ###################################################

def get_pavpu(y_true, y_pred, y_unc, unc_thresholds, verbose=False):
    """
    For each grid
        - calculate accuracy    ; grids_accuracy    = [] ; M = number of grids in volume (1=accurate , 0=inaccurate)
        - calculate uncertainty ; grids_uncertainty = [] ; M = number of grids in volume (1=uncertain, 0=certain)
        - calculate p(acc|cer)  = n_ac / (n_ac + n_ic); n_ac = (acc=1,unc=0), n_ic = (acc=0,unc=0)
        - calculate p(unc|inac) = n_iu / (n_iu + n_ic); n_iu = (acc=0,unc=1), n_ic = (acc=0,unc=0)
        - calculate pa-vs-pu    = n_ac + n_iu / (n_ac + n_au + n_ic + n_iu)

    Params
    ------
    y_true         : [H,W,D]/[H,W,D,C]
    y_pred         : [H,W,D]/[H,W,D,C]
    y_unc          : [H,W,D]
    unc_thresholds : list; float
    """
    res = {'p_ac': {}, 'p_ui': {}, 'pavpu': {}, 'p_ua':{}}

    try:

        # Step 0 - Init
        if 0:
            H,W,D,C                = y_true.shape
            y_true_class           = np.argmax(y_true, axis=-1)
            y_pred_class           = np.argmax(y_pred, axis=-1)
        else:
            H,W,D                  = y_true.shape
            y_true_class           = y_true
            y_pred_class           = y_pred
        
        # Step 1.1 - Calculate grid-wise accuracy and uncertainty
        y_accurate    = (y_true_class == y_pred_class).astype(np.uint8)  # [H,W,D]
        y_inaccurate  = 1 - y_accurate

        print (' - ', np.sum(y_accurate), ' || ', np.sum(y_inaccurate))
        print ('')

        # Step 1.2 - Compute n-vals using accuracy and uncertainty for each voxel
        for unc_threshold in unc_thresholds:
            y_uncertain      = np.array(y_unc, copy=True)
            y_uncertain[y_uncertain >= unc_threshold] = 1
            y_uncertain[y_uncertain <  unc_threshold] = 0
            
            y_certain = 1 - y_uncertain
            n_ac = np.sum(y_accurate * y_certain)
            n_au = np.sum(y_accurate * y_uncertain)
            n_ic = np.sum(y_inaccurate * y_certain)
            n_iu = np.sum(y_inaccurate * y_uncertain)
            
            # if verbose: print (' - [eval_3D()] unc_threshold: {:f} || n_ac: {} || n_ic: {} || n_iu: {} || n_au: {}'.format(unc_threshold, n_ac, n_ic, n_iu, n_au))
        
            # Step 1.3 - Compute final results
            prob_unc_acc   = n_au / (n_ac + n_au)
            prob_acc_cer   = n_ac / (n_ac + n_ic)
            if n_iu == 0: prob_unc_inacc = 0.0
            else        : prob_unc_inacc = n_iu / (n_iu + n_ic)
            pavpu          = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
            res['p_ua'][unc_threshold]  = prob_unc_acc
            res['p_ac'][unc_threshold]  = prob_acc_cer
            res['p_ui'][unc_threshold]  = prob_unc_inacc
            res['pavpu'][unc_threshold] = pavpu
            # if verbose: print (' - [eval_3D()] prob_acc_cer: {:f} || prob_unc_inacc: {:f} || pavpu: {:f} || prob_unc_acc: {:f}'.format(prob_acc_cer, prob_unc_inacc, pavpu, prob_unc_acc))
    
        return res

    except:
        traceback.print_exc()
        pdb.set_trace()
        return {'p_ac': [], 'p_ui': [], 'pavpu': [], 'p_ua': []}

def plot_pavpu_(res, exp_colors, key, key_plt):
    """
    Params
    ------
    res: Dict e.g. {exp_name : {patient_id: {}}}
    """
    try:
        unc_key = 'Uncertainty Threshold'
        unc_str_save = ''
        if 'mif' in unc_str  : 
            unc_key += '(MI)'
            unc_str_save = 'MI'
        elif 'ent' in unc_str: 
            unc_key += '(Ent)'
            unc_str_save = 'Ent'

        plt.figure(figsize=(11,5), dpi=200)
        sns.set_style('darkgrid')
        for exp_id, exp_name in enumerate(res):
                
            data = {
                unc_key   : np.array([list(res[exp_name][patient_id][key].keys())   for patient_id in res[exp_name]]).flatten().tolist()
                , key_plt : np.array([list(res[exp_name][patient_id][key].values()) for patient_id in res[exp_name]]).flatten().tolist()
            }
            sns.lineplot(data=data, x=unc_key, y=key_plt, label=exp_name, color=exp_colors[exp_id])

        plt.title('Mode={} || Ip={} || Epoch={}'.format(mode, plt_inputinfo, epoch))
        plt.legend(fontsize=15)
        if XAXIS_LOG: plt.gca().set_xscale('log')
        plt.gca().xaxis.label.set_size(15)
        plt.gca().yaxis.label.set_size(15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim([0,1])
        # plt.show()
        filename = '_tmp/pavpu/{}-{}-{}'.format(mode, unc_str_save, key)
        plt.savefig(filename, bbox_inches='tight')
        print (' - \n Saved at ', filename)
        plt.close()
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def plot_pavpu(res, exp_colors):
    """
    res: {'p_ac': [], 'p_ui': [], 'pavpu': []}
    """

    try:
        plot_pavpu_(res, exp_colors, 'p_ua', 'p(u|a)')
        plot_pavpu_(res, exp_colors, 'p_ac', 'p(a|c)')
        plot_pavpu_(res, exp_colors, 'p_ui', 'p(u|i)')
        plot_pavpu_(res, exp_colors, 'pavpu', 'PAvPU')
                                    
    except:
        traceback.print_exc()
        pdb.set_trace()

############################################### PAvPU - Area around Organs ###############################################

def get_pavpu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=(3,3,1), error_ksize=(3,3,1), verbose=False):

    """
    Note: PAvPU but masked by areas around GT and Prediction (due to too much background)
    For each grid
        - calculate accuracy    ; grids_accuracy    = [] ; M = number of grids in volume (1=accurate , 0=inaccurate)
        - calculate uncertainty ; grids_uncertainty = [] ; M = number of grids in volume (1=uncertain, 0=certain)
        - calculate p(acc|cer)  = n_ac / (n_ac + n_ic); n_ac = (acc=1,unc=0), n_ic = (acc=0,unc=0)
        - calculate p(unc|inac) = n_iu / (n_iu + n_ic); n_iu = (acc=0,unc=1), n_ic = (acc=0,unc=0)
        - calculate pa-vs-pu    = n_ac + n_iu / (n_ac + n_au + n_ic + n_iu)

    Params
    ------
    y_true         : [H,W,D]/[H,W,D,C]
    y_pred         : [H,W,D]/[H,W,D,C]
    y_unc          : [H,W,D]
    unc_thresholds : list; float
    """
    res = {'p_ac': {}, 'p_ui': {}, 'pavpu': {}, 'p_ua':{}}

    try:

        # Step 0 - Init
        H,W,D                  = y_true.shape
        y_true_class           = y_true
        y_pred_class           = y_pred
        
        # Step 1 - Calculate mask for areas around organs only
        y_mask, y_true_class_binary, y_pred_class_binary = get_dilated_y_true_pred(y_true_class, y_pred_class, dilation_ksize)
        
        # Step 2 - Get accurate and inaccurate areas
        if PAVPU_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_avupaper(y_true_class, y_pred_class, y_mask)
        
        elif HIERR_LERR_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize)    
        
        elif HERR_LERR_SURF_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion_and_surface(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize, spacing, distance_max=DIST_MAX_SURFERR)
            

        # Step 3 - Compute n-vals using accuracy and uncertainty for each voxel
        for unc_threshold in unc_thresholds:
            y_uncertain      = np.array(y_unc, copy=True)
            y_uncertain[y_uncertain >= unc_threshold] = 1
            y_uncertain[y_uncertain <  unc_threshold] = 0
            
            y_certain = 1 - y_uncertain
            n_ac = np.sum(y_accurate * y_certain)
            n_au = np.sum(y_accurate * y_uncertain)
            n_ic = np.sum(y_inaccurate * y_certain)
            n_iu = np.sum(y_inaccurate * y_uncertain)
            
            # if verbose: print (' - [eval_3D()] unc_threshold: {:f} || n_ac: {} || n_ic: {} || n_iu: {} || n_au: {}'.format(unc_threshold, n_ac, n_ic, n_iu, n_au))
        
            # Step 1.3 - Compute final results
            prob_unc_acc   = n_au / (n_ac + n_au)
            if n_iu == 0: prob_unc_inacc = 0.0
            else        : prob_unc_inacc = n_iu / (n_iu + n_ic)
            pavpu          = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
            prob_acc_cer   = n_ac / (n_ac + n_ic)

            res['p_ua'][unc_threshold]  = prob_unc_acc   # plotted
            res['p_ui'][unc_threshold]  = prob_unc_inacc # plotted
            res['p_ac'][unc_threshold]  = prob_acc_cer
            res['pavpu'][unc_threshold] = pavpu

            # if verbose: print (' - [eval_3D()] prob_acc_cer: {:f} || prob_unc_inacc: {:f} || pavpu: {:f} || prob_unc_acc: {:f}'.format(prob_acc_cer, prob_unc_inacc, pavpu, prob_unc_acc))
    
        return res

    except:
        traceback.print_exc()
        pdb.set_trace()
        return {'p_ac': [], 'p_ui': [], 'pavpu': [], 'p_ua':{}}

def plot_pavpu_organs(res, exp_colors, mode, epoch):
    """
    res: {'p_ac': [], 'p_ui': [], 'pavpu': []}
    """

    try:
        plot_pavpu_(res, exp_colors, 'p_ua', 'p(u|a) - Organs - ksize={}'.format(ksize))
        plot_pavpu_(res, exp_colors, 'p_ac', 'p(a|c) - Organs - ksize={}'.format(ksize))
        plot_pavpu_(res, exp_colors, 'p_ui', 'p(u|i) - Organs - ksize={}'.format(ksize))
        plot_pavpu_(res, exp_colors, 'pavpu', 'PAvPU - Organs - ksize={}'.format(ksize))
                                    
    except:
        traceback.print_exc()
        pdb.set_trace()

############################################### PAvPU - Area around Organ(s) ###############################################

def get_pavpu_organ(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=(3,3,1), error_ksize=(3,3,1), verbose=False):

    """
    Note: PAvPU but masked by areas around GT and Prediction (due to too much background)
    For each grid
        - calculate accuracy    ; grids_accuracy    = [] ; M = number of grids in volume (1=accurate , 0=inaccurate)
        - calculate uncertainty ; grids_uncertainty = [] ; M = number of grids in volume (1=uncertain, 0=certain)
        - calculate p(acc|cer)  = n_ac / (n_ac + n_ic); n_ac = (acc=1,unc=0), n_ic = (acc=0,unc=0)
        - calculate p(unc|inac) = n_iu / (n_iu + n_ic); n_iu = (acc=0,unc=1), n_ic = (acc=0,unc=0)
        - calculate pa-vs-pu    = n_ac + n_iu / (n_ac + n_au + n_ic + n_iu)

    Params
    ------
    y_true         : [H,W,D]/[H,W,D,C]
    y_pred         : [H,W,D]/[H,W,D,C]
    y_unc          : [H,W,D]
    unc_thresholds : list; float
    mode           : dataset
    """

    # Step 0 - Get res
    res = {}
    label_ids = []
    if mode == config.MODE_TEST: # in the test set we have all OARs annotated, so we dont need to handle the lack of a ground truth situation
        label_ids = list(range(1,10))
    elif mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
        label_ids = list(range(1,10))
        label_ids.pop(1) # remove label_id = 2 # optic chiasm
    
    for label_id in label_ids:
        res[label_id] = {'p_ac': {}, 'p_ui': {}, 'pavpu': {}, 'p_ua':{}}

    try:

        # Step 1 - Init
        if 0:
            H,W,D,C                = y_true.shape
            y_true_class           = np.argmax(y_true, axis=-1)
            y_pred_class           = np.argmax(y_pred, axis=-1)
        else:
            H,W,D                  = y_true.shape
            y_true_class           = y_true
            y_pred_class           = y_pred
        
        # Step 2 - Calculate mask for areas around GT + AI organs
        y_true_class_binary = np.array(y_true_class, copy=True)
        if mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
            y_true_class_binary[y_true_class_binary == 2] = 0 # set optic chiasm to gt=0(bgd); this ignored in next line
        y_true_class_binary[y_true_class_binary > 0] = 1
        y_true_class_binary = tf.constant(y_true_class_binary, tf.float32)
        y_true_class_binary = tf.constant(tf.expand_dims(tf.expand_dims(y_true_class_binary, axis=-1),axis=0))
        y_true_class_binary_dilated  = tf.nn.max_pool3d(y_true_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')
        
        y_pred_class_binary = np.array(y_pred_class, copy=True)
        if mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
            y_pred_class_binary[y_pred_class_binary == 2] = 0 # set optic chiasm to pred=0(bgd); this ignored in next line
        y_pred_class_binary[y_pred_class_binary > 0] = 1
        y_pred_class_binary = tf.constant(y_pred_class_binary, tf.float32)
        y_pred_class_binary = tf.expand_dims(tf.expand_dims(y_pred_class_binary, axis=-1),axis=0)
        y_pred_class_binary_dilated  = tf.nn.max_pool3d(y_pred_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')

        y_true_pred_binary_dilated = (y_true_class_binary_dilated + y_pred_class_binary_dilated)[0,:,:,:,0].numpy()
        y_true_pred_binary_dilated[y_true_pred_binary_dilated > 1] = 1
        
        # Step 3 - Calculate Hi-Err and Lo-Err areas
        y_error_areas          = tf.math.abs(y_true_class_binary - y_pred_class_binary)
        y_error_areas_eroded   = -tf.nn.max_pool3d(-y_error_areas, ksize=error_ksize, strides=1, padding='SAME')
        y_error_areas_high     = tf.nn.max_pool3d(y_error_areas_eroded, ksize=error_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy()

        y_nonerror_areas           = y_true_pred_binary_dilated - y_error_areas_high
        y_nonerror_areas           = tf.expand_dims(tf.expand_dims(tf.constant(y_nonerror_areas, tf.float32), axis=-1),axis=0)
        y_nonerror_areas_eroded    = -tf.nn.max_pool3d(-y_nonerror_areas, ksize=error_ksize, strides=1, padding='SAME')
        y_error_areas_low          = tf.nn.max_pool3d(y_nonerror_areas_eroded, ksize=error_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy() # No errpr + Low Error

        y_accurate_mask   = y_error_areas_low
        y_inaccurate_mask = y_error_areas_high

        for label_id in label_ids:
            
            y_true_class_binary_label = np.array(y_true_class, copy=True)
            y_true_class_binary_label[y_true_class_binary_label != label_id] = 0
            y_true_class_binary_label[y_true_class_binary_label == label_id] = 1

            y_pred_class_binary_label = np.array(y_pred_class, copy=True)
            y_pred_class_binary_label[y_pred_class_binary_label != label_id] = 0
            y_pred_class_binary_label[y_pred_class_binary_label == label_id] = 1

            y_binary_label = y_true_class_binary_label + y_pred_class_binary_label
            y_binary_label[y_binary_label > 1] = 1
            y_binary_label = tf.constant(y_binary_label, tf.float32)
            y_binary_label = tf.constant(tf.expand_dims(tf.expand_dims(y_binary_label, axis=-1),axis=0))
            y_binary_label_dilated  = tf.nn.max_pool3d(y_binary_label, ksize=dilation_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy()

            y_accurate_label_mask   = y_accurate_mask   * y_binary_label_dilated
            y_inaccurate_label_mask = y_inaccurate_mask * y_binary_label_dilated

            # Step 1.2 - Compute n-vals using accuracy and uncertainty for each voxel
            for unc_threshold in unc_thresholds:
                y_uncertain      = np.array(y_unc, copy=True)
                y_uncertain[y_uncertain >= unc_threshold] = 1
                y_uncertain[y_uncertain <  unc_threshold] = 0
                
                y_certain = 1 - y_uncertain
                n_ac = np.sum(y_accurate_label_mask * y_certain)
                n_au = np.sum(y_accurate_label_mask * y_uncertain)
                n_ic = np.sum(y_inaccurate_label_mask * y_certain)
                n_iu = np.sum(y_inaccurate_label_mask * y_uncertain)
                
                if verbose: print (' - [eval_3D()] unc_threshold: {:f} || n_ac: {} || n_ic: {} || n_iu: {} || n_au: {}'.format(unc_threshold, n_ac, n_ic, n_iu, n_au))
            
                # Step 1.3 - Compute final results
                if n_ac + n_au == 0:
                    prob_unc_acc = 0
                else:      
                    prob_unc_acc   = n_au / (n_ac + n_au)
                if n_iu == 0: prob_unc_inacc = 0.0
                else        : prob_unc_inacc = n_iu / (n_iu + n_ic)
                pavpu          = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
                prob_acc_cer   = n_ac / (n_ac + n_ic)

                res[label_id]['p_ua'][unc_threshold]  = prob_unc_acc   # plotted
                res[label_id]['p_ui'][unc_threshold]  = prob_unc_inacc # plotted
                res[label_id]['p_ac'][unc_threshold]  = prob_acc_cer
                
                res[label_id]['pavpu'][unc_threshold] = pavpu
                
                if verbose: print (' - [eval_3D()] label_id: {} || prob_acc_cer: {:f} || prob_unc_inacc: {:f} || pavpu: {:f} || prob_unc_acc: {:f}'.format(label_id, prob_acc_cer, prob_unc_inacc, pavpu, prob_unc_acc))
    
        return res

    except:
        traceback.print_exc()
        pdb.set_trace()
        return {'p_ac': [], 'p_ui': [], 'pavpu': [], 'p_ua':{}}

def plot_pavpu_organ(res, exp_colors):
    """
    Params
    ------
    res: Dict e.g. {exp_name : {patient_id: {'p_ac': [], 'p_ui': [], 'pavpu': []} }}
    """

    try:
        random_exp     = list(res.keys()[0])
        random_patient = res[random_exp][list(res[random_exp].keys())[0]]
        label_ids      = res[random_exp][random_patient].keys()

        for label_id in label_ids:
            res_label = {}
            for exp_name in res:
                res_label[exp_name] = {}
                for patient_id in res[exp_name]:
                    res_label[exp_name][patient_id] = res[exp_name][patient_id][label_id]

            pdb.set_trace()
            plot_pavpu_(res_label, exp_colors, 'p_ua', 'p(u|a) - Organ_{} - ksize={}'.format(label_id, ksize))
            plot_pavpu_(res_label, exp_colors, 'p_ac', 'p(a|c) - Organ_{} - ksize={}'.format(label_id, ksize))
            plot_pavpu_(res_label, exp_colors, 'p_ui', 'p(u|i) - Organ_{} - ksize={}'.format(label_id, ksize))
            plot_pavpu_(res_label, exp_colors, 'pavpu', 'PAvPU - Organ_{} - ksize={}'.format(label_id, ksize))
                                    
    except:
        traceback.print_exc()
        pdb.set_trace()

############################################ p(u) - HiErr vs LoErr ###########################################

# Paper function
# def get_pu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=(3,3,1), error_ksize=(3,3,1), verbose=False):
def get_pu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize, error_ksize, verbose=False):

    """
    Goal: Get uncertainty probability values for each threshold in high and low error areas (as defined by dilation and error kernels)  for a particular patient

    Params
    ------
    y_true         : [H,W,D]/[H,W,D,C]
    y_pred         : [H,W,D]/[H,W,D,C]
    y_unc          : [H,W,D]
    unc_thresholds : list; float
    mode           : str
    """

    res = {config.KEY_P_UI: {}, config.KEY_P_UA:{}, config.KEY_P_IU: {}, config.KEY_P_AC: {}, config.KEY_AVU: {}, RATIO_KEY: {}, 'acc_unc_vals': [], 'inacc_unc_vals': []}

    try:

        # Step 0 - Init
        H,W,D                  = y_true.shape
        y_true_class           = y_true
        y_pred_class           = y_pred
        
        # Step 1 - Calculate mask for areas around organs only
        y_mask, y_true_class_binary, y_pred_class_binary = get_dilated_y_true_pred(y_true_class, y_pred_class, dilation_ksize)
        # y_mask, y_true_class_binary, y_pred_class_binary = get_dilated_y_true(y_true_class, y_pred_class, dilation_ksize)

        # Step 2 - Get accurate (Lo-Err) and inaccurate (Hi-Err) areas 
        if PAVPU_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_avupaper(y_true_class, y_pred_class, y_mask)
        
        elif HIERR_LERR_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize)    
        
        elif HERR_LERR_SURF_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion_and_surface(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize, spacing, distance_max=DIST_MAX_SURFERR)
        
        # pdb.set_trace()
        # f,axarr = plt.subplots(1,3); axarr[0].imshow(y_accurate[:,:,40]); axarr[1].imshow(y_inaccurate[:,:,40]); axarr[2].imshow(y_unc[:,:,40]); plt.show()
        acc_unc_vals = y_unc[y_accurate == 1]
        inacc_unc_vals = y_unc[y_inaccurate == 1]
        res['acc_unc_vals'] = acc_unc_vals
        res['inacc_unc_vals'] = inacc_unc_vals
        
        # Step 3 - Uncertainty (Norm or not)
        _ = get_unc_norm(y_unc)
        print (' - [get_pu_organs()] np.max(y_unc): ', np.max(y_unc))

        # Step 4 - Compute n-vals using accuracy and uncertainty for each voxel
        for unc_threshold in unc_thresholds:

            # Step 4.1 - Find (un)certain mask 
            y_uncertain      = np.array(y_unc, copy=True)
            y_uncertain[y_uncertain <=  unc_threshold] = 0
            y_uncertain[y_uncertain > unc_threshold] = 1
            y_certain = 1 - y_uncertain

            # Step 4.2 - Find n-vals
            n_ac = np.sum(y_accurate * y_certain)
            n_au = np.sum(y_accurate * y_uncertain)
            n_ic = np.sum(y_inaccurate * y_certain)
            n_iu = np.sum(y_inaccurate * y_uncertain)
            # if verbose: print (' - [eval_3D()] unc_threshold: {:f} || n_ac: {} || n_ic: {} || n_iu: {} || n_au: {}'.format(unc_threshold, n_ac, n_ic, n_iu, n_au))
        
            # Step 4.3 - Compute final results
            if 0:
                if n_ac + n_ic == 0:
                    p_ac = 0
                else:
                    p_ac = n_ac / (n_ac + n_ic)
                p_ua = n_au / (n_ac + n_au)
                if n_iu == 0: 
                    p_ui = 0.0
                    p_iu = 0.0            
                else: 
                    p_ui = n_iu / (n_iu + n_ic)
                    p_iu = n_iu / (n_iu + n_au)
            
            elif 1:
                if n_iu == 0 and n_au == 0: # to handle precision=1.0 and recall=0.0 [date=18th April, 2023]
                    n_iu = config.EPSILON
                    # print (' - n_iu == 0 and n_au == 0 for unc_thres: {:.6f}'.format(unc_threshold)) # For MICCA > [0.3]
                p_ac = n_ac / (n_ac + n_ic)
                p_ua = n_au / (n_ac + n_au)
                p_ui = n_iu / (n_iu + n_ic)
                p_iu = n_iu / (n_iu + n_au)
                
            else:
                p_ac = n_ac / (n_ac + n_ic + config.EPSILON)
                p_ua = n_au / (n_ac + n_au + config.EPSILON)
                p_ui = n_iu / (n_iu + n_ic + config.EPSILON)
                p_iu = n_iu / (n_iu + n_au + config.EPSILON)
            
        
            avu  = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
            

            if p_ua < 0.0:
                print (' - [ERROR][get_pu_organs()] p_ua: {:.5f} is negative at threshold: {:.3f}'.format(p_ua, unc_threshold))
                pdb.set_trace()
            if p_ui < 0.0:
                print (' - [ERROR][get_pu_organs()] p_ui: {:.5f} is negative at threshold: {:.3f}'.format(p_ui, unc_threshold))
                pdb.set_trace()
            if p_iu < 0.0:
                print (' - [ERROR][get_pu_organs()] p_iu: {:.5f} is negative at threshold: {:.3f}'.format(p_iu, unc_threshold))
                pdb.set_trace()
            if p_ac < 0.0:
                print (' - [ERROR][get_pu_organs()] p_ac: {:.5f} is negative at threshold: {:.3f}'.format(p_ac, unc_threshold))
                pdb.set_trace()

            res[config.KEY_P_UA][unc_threshold]  = p_ua 
            res[config.KEY_P_UI][unc_threshold]  = p_ui 
            res[config.KEY_P_IU][unc_threshold]  = p_iu 
            res[config.KEY_P_AC][unc_threshold]  = p_ac
            res[config.KEY_AVU][unc_threshold]   = avu

            try:
                res[RATIO_KEY][unc_threshold] = (n_ic + n_iu) / (n_ac + n_au)
            except:
                res[RATIO_KEY][unc_threshold] = 0
                
        return res

    except:
        traceback.print_exc()
        pdb.set_trace()
        return {'high-error': {}, 'low-error': {}}

def plot_pu_organs(res, plt_colors, unc_str, mode, epoch, eval_str, unc_postfix_extras='', title_extras=None, verbose=False):
    """
    res: Dict e.g. {exp_id: { patient_id: {'high-error': {}, 'low-error': {}} }}
    """

    try:
        
        # Step 0 - Init
        unc_postfix = ''
        if config.KEYNAME_ENT in unc_str:
            unc_postfix = config.KEY_ENT
        elif config.KEYNAME_MIF in unc_str:
            unc_postfix = config.KEY_MIF
        elif config.KEYNAME_STD_MAX in unc_str:
            unc_postfix = config.KEY_STD_MAX
        elif config.KEYNAME_STD in unc_str and not (config.KEYNAME_STD_MAX in unc_str):
            unc_postfix = config.KEY_STD
        
        unc_postfix += unc_postfix_extras

        if UNC_ERODIL:
            unc_postfix += '-EroDil'
        
        norm_postfix = ''
        if np.any(unc_norms):
            norm_postfix = 'Norm'

        if type(mode) == list:
            mode = '-'.join(np.unique(mode))
        # plt_title_str = '{} || Mode={} || Eval={} || Epoch={} || Unc={}{}'.format(plt_alias, mode, eval_str, epoch, unc_postfix, norm_postfix)
        plt_title_str = '{} || Mode={} || Eval={} \n Epoch={} || Unc={}{}'.format(plt_alias, mode, eval_str, epoch, unc_postfix, norm_postfix)
        
        if len(SINGLE_PATIENT): plt_title_str = plt_title_str + '\n' + SINGLE_PATIENT
        if title_extras is not None: plt_title_str += title_extras

        # Trapezoidal Method (slightly modified)
        def integral(y, x):
            dx = x[:-1] - x[1:]
            dy = (y[:-1] + y[1:])/2
            return tf.math.reduce_sum(dx*dy)

        xaxis_lim = [-0.01, 1.01] # [0,1]
        yaxis_lim = [-0.05, 1.05] # [0,1]

        path_savefig = '_tmp/r-avu/{}__{}-{}-ep{}'.format(plt_alias, '{}', mode, epoch)
        
        # Step x - Histogram of acc and inaccurate uncertainties
        if 0:
            try:
                # Step 1.1 - Set plt params
                pass

                # Step 1.2 - Setup key and loop over experiments
                res_hist = {}
                for exp_id, exp_name in enumerate(res):
                    key_acc = 'acc_unc_vals'
                    key_inacc = 'inacc_unc_vals'
                    key_list_acc = 'acc_uncs'
                    key_list_inacc = 'inacc_uncs'

                    data_exp = {
                        key_list_acc: [], key_list_inacc: []
                    }
                    for patient_id in res[exp_name]:
                        data_exp[key_list_acc] += list(res[exp_name][patient_id][key_acc])
                        data_exp[key_list_inacc] += list(res[exp_name][patient_id][key_inacc])

                    res_hist[exp_name] = data_exp

                    # Step 1.3 - Plot
                    pass

                # Step y -
                f,axarr = plt.subplots(1,4, sharex=True, sharey=True)
                axarr[0].hist(res_hist['Det']['inacc_uncs'], color='red', alpha=0.5)
                axarr[1].hist(res_hist['Ens']['acc_uncs'], color='green', alpha=0.5)
                axarr[2].hist(res_hist['Bayes']['acc_uncs'], color='green', alpha=0.5)
                axarr[3].hist(res_hist['Bayes+AvU']['acc_uncs'], color='green', alpha=0.5) 
                plt.hist(res_hist['Det']['inacc_uncs'], color='red', alpha=0.5); plt.hist(res_hist['Bayes']['inacc_uncs'], color='green', alpha=0.5); plt.show()

                import seaborn as sns
                f,axarr = plt.subplots(2,4, sharex=True, sharey='row')
                sns.histplot(res_hist['Det']['inacc_uncs'][res_hist['Det']['inacc_uncs'] > 0.01], color='red', alpha=0.5, ax=axarr[0][0], stat='probability'); axarr[0][0].set_title('Det')
                sns.histplot(res_hist['Ens']['inacc_uncs'][res_hist['Ens']['inacc_uncs'] > 0.01], color='red', alpha=0.5, ax=axarr[0][1], stat='probability'); axarr[0][1].set_title('Ens')
                sns.histplot(res_hist['Bayes']['inacc_uncs'][res_hist['Bayes']['inacc_uncs'] > 0.01], color='red', alpha=0.5, ax=axarr[0][2], stat='probability'); axarr[0][2].set_title('Bayes')
                sns.histplot(res_hist['Bayes+AvU']['inacc_uncs'][res_hist['Bayes+AvU']['inacc_uncs'] > 0.01], color='red', alpha=0.5, ax=axarr[0][3], stat='probability'); axarr[0][3].set_title('Bayes+AvU')
                sns.histplot(res_hist['Det']['acc_uncs'][res_hist['Det']['acc_uncs'] > 0.01], color='green', alpha=0.5, ax=axarr[1][0], stat='probability')
                sns.histplot(res_hist['Ens']['acc_uncs'][res_hist['Ens']['acc_uncs'] > 0.01], color='green', alpha=0.5, ax=axarr[1][1], stat='probability')
                sns.histplot(res_hist['Bayes']['acc_uncs'][res_hist['Bayes']['acc_uncs'] > 0.01], color='green', alpha=0.5, ax=axarr[1][2], stat='probability')
                sns.histplot(res_hist['Bayes+AvU']['acc_uncs'][res_hist['Bayes+AvU']['acc_uncs'] > 0.01], color='green', alpha=0.5, ax=axarr[1][3], stat='probability')
                plt.show()

                pdb.set_trace()
            except:
                traceback.print_exc()
                pdb.set_trace()

        # Step 0.1 - Ratio of inacc/acc
        if 0:
            
            # Step 1.1 - Set plt params
            sns.set_style('darkgrid')
            plt.figure(figsize=(11,5), dpi=200)
            if not FIG_FOR_PAPER:
                plt.title(plt_title_str)
            plt.ylim(yaxis_lim)
            plt.xlim(xaxis_lim)
            plt.gca().xaxis.label.set_size(15)
            plt.gca().yaxis.label.set_size(15)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Step 3.2 - Setup key and loop over experiments
            for exp_id, exp_name in enumerate(res):
                x_key = 'Uncertainy Threshold' 
                y_key = RATIO_KEY
                
                data = {
                        x_key   : np.array([list(res[exp_name][patient_id][y_key].keys())   for patient_id in res[exp_name]]).flatten().tolist()
                        , y_key : np.array([list(res[exp_name][patient_id][y_key].values()) for patient_id in res[exp_name]]).flatten().tolist()
                }

                sns.lineplot(data=data, x=x_key, y=y_key, label=exp_name, color=plt_colors[exp_id])
            
            # Step 3.3 - Save figure
            plt.legend(fontsize=15)
            path_savefig = '_tmp/r-avu/{}-Ratios-{}-{}{}-{}-ep{}'.format(plt_alias, mode, unc_postfix, norm_postfix, eval_str, epoch)
            if len(SINGLE_PATIENT):
                path_savefig = path_savefig + '-' + SINGLE_PATIENT
            plt.savefig(path_savefig, bbox_inches='tight')
            plt.close()
            print ('')
            print (' - Saved as ', path_savefig)

        # Step 1.1 - ROC curves (new)
        PATIENT_WISE = False
        if 1:
            
            print ('\n\n\n ================== unc-ROC curves ================== \n')
            try:

                # Step 1.1 - Set plt params
                if 1:
                    import seaborn as sns
                    sns.set_style('darkgrid')
                    plt.figure(figsize=(11,5), dpi=DPI)
                    if not FIG_FOR_PAPER:
                        plt.title(plt_title_str)
                    plt.gca().xaxis.label.set_size(LABEL_FONTSIZE)
                    plt.gca().yaxis.label.set_size(LABEL_FONTSIZE)
                    plt.xticks(fontsize=TICKS_FONTSIZE)
                    plt.yticks(fontsize=TICKS_FONTSIZE)

                # Step 1.2 - Setup keys
                if 1:
                    # x_key = 'mean-p(u|a,~a)'
                    x_key = 'mean FPR - p(u|a)' 
                    y_key = 'mean TPR - p(u|i)'
                    auc_vals                   = []
                    auc_vals_tflow             = []
                    auc_vals_patientwise       = {}
                    auc_vals_tflow_patientwise = {}
                
                # Step 1.3 - Loop over experiments
                data_roc = {}
                for exp_id, exp_name in enumerate(res):
                    data_roc_exp = {
                        x_key: [], y_key: [] # len(patient_id) * len(np.linpsace(0, 1, 100))
                        , 'patient_id': [], 'auc_roc': [] # len(patient_id)
                    }    
                    for patient_id in res[exp_name]:

                        try:
                            # Step x - Get data
                            patient_p_ua = np.array(list(res[exp_name][patient_id][config.KEY_P_UA].values())) # x-axis
                            patient_p_ui = np.array(list(res[exp_name][patient_id][config.KEY_P_UI].values())) # y-axis
                            
                            # Step y - Perform interpolation
                            patient_p_ua_interp = np.linspace(0, 1, 100)
                            patient_p_ui_interp = np.interp(patient_p_ua_interp[::-1], patient_p_ua[::-1], patient_p_ui[::-1])[::-1] # np.interp needs a monotonically increasing curve

                            # Step z - Save data
                            data_roc_exp['patient_id'].append(patient_id)
                            data_roc_exp[x_key].extend(list(patient_p_ua_interp))
                            data_roc_exp[y_key].extend(list(patient_p_ui_interp))                            
                            data_roc_exp['auc_roc'].append(sklearn.metrics.auc(list(patient_p_ua_interp), list(patient_p_ui_interp)))
                                
                            if PATIENT_WISE:
                                sns.lineplot(x=list(patient_p_ua_interp), y=list(patient_p_ui_interp), label=exp_name + '-' + patient_id + '(AUC-{:.3f})'.format(data_roc_exp['auc_roc'][-1]), color=(0.0,0.0,0.0), dashes=True)
                        
                        except:
                            print (' - [ERROR][plot_pu_organs()][ROC curves] exp:{}, patient_id: {}'.format(exp_name, patient_id) )
                            traceback.print_exc()
                            pdb.set_trace()

                    label_str = exp_name
                    if AUC_INFO:
                        label_str += ' (AUC-{:.3f} ± {:.3f})'.format(np.mean(data_roc_exp['auc_roc']), np.std(data_roc_exp['auc_roc']))

                    if 'Head' not in exp_name:
                        sns.lineplot(data=data_roc_exp, x=x_key, y=y_key, label=label_str, color=plt_colors[exp_id], dashes=False)
                    data_roc[exp_name] = data_roc_exp

                # Step 1.4 - Stats (scipy.stats.wilcoxon)
                if 1:
                    import itertools
                    import pandas as pd # unnecessary import due to random error
                    stats_roc = np.full((len(res), len(res)), 99.0)
                    exp_names = list(data_roc.keys())
                    stats_str = '\n'
                    
                    for each in itertools.combinations(exp_names,2):
                        result = scipy.stats.wilcoxon(data_roc[each[0]]['auc_roc'], data_roc[each[1]]['auc_roc'])
                        stats_roc[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        stats_str += ' - Exp: {} vs {} = {:3f} \n'.format(each[0], each[1], result.pvalue)
                    
                    exp_names_print = [exp_name.replace('OrgPatch-', '') for exp_name in exp_names]
                    df_stats_roc = pd.DataFrame(stats_roc, columns=exp_names_print, index=exp_names_print).round(3)
                    df_stats_roc[df_stats_roc > 0.05] = 1
                    df_stats_roc[df_stats_roc <= 0.05] = 'Sig'
                    print (df_stats_roc)
                    print (' ---------------------------- ')
                    print (stats_str)

                # Step 1.5 - Print results
                print ('\n ================== unc-ROC curves ================== \n')
                for exp_id, exp_name in enumerate(res):
                    print (' - [ROC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(data_roc[exp_name]['auc_roc']), np.std(data_roc[exp_name]['auc_roc'])))
                print ('\n ================== unc-ROC curves ================== \n')

                # Step 1.9 - Save figure
                if 1:
                    if 0:
                        yticks_min = next((i for i in plt.gca().get_yticks() if i > 0), None)
                        yticks_max = plt.gca().get_yticks()[-1]
                        yticks = np.linspace(roundtopoint5(yticks_min), roundtopoint5(yticks_max), params_fig['num_yticks'])
                        yticklabels = ['{:.2f}'.format(ytick) for ytick in yticks]
                        plt.gca().set_yticks(yticks)
                        plt.gca().set_yticklabels(yticklabels)
                    plt.legend(fontsize=LEGEND_FONTSIZE)
                    # path_savefig = '_tmp/r-avu/{}__ROCurve-{}-{}{}-{}-ep{}-optic{}'.format(plt_alias, mode, unc_postfix, norm_postfix, eval_str, epoch, optic_bool)
                    # if len(SINGLE_PATIENT):
                    #     path_savefig = path_savefig + '-' + SINGLE_PATIENT
                    plt.savefig(path_savefig.format('ROCurve'), bbox_inches='tight')
                    plt.close()
                    print ('')
                    print (' - Saved as ', path_savefig)
            
            except:
                traceback.print_exc()
                pdb.set_trace()

        # Step 1.2 - ROC swarm (old)
        PATIENT_WISE = False
        STATS_NOISE_ROC  = False
        print ('\n -------------------------- ROC swarm noise: {} \n'.format(STATS_NOISE_ROC))
        if 1:
            
            try:
                
                print ('\n ================== unc-ROC BoxPlots ({}) ================== \n'.format(plt_alias))

                # Step 1.3.1 - Set plt params
                import seaborn as sns
                sns.set_style('darkgrid')
                plt.figure(figsize=(11,5), dpi=DPI)
                if not FIG_FOR_PAPER:
                    plt.title(plt_title_str)
                plt.gca().xaxis.label.set_size(LABEL_FONTSIZE)
                plt.gca().yaxis.label.set_size(LABEL_FONTSIZE)
                plt.xticks(fontsize=TICKS_FONTSIZE)
                plt.yticks(fontsize=TICKS_FONTSIZE)

                # Step 1.3.2 - Setup keys and loop over experiments
                x_key = 'mean-p(u|a,~a)' 
                y_key = 'mean-p(u|i)'
                auc_vals_patientwise       = {}
                auc_vals_tflow_patientwise = {}

                boxplot_exp_names = []
                boxplot_roc_aucs  = []
                palette           = {}
                
                # Step 1.3.3 - Loop over experiments and patients and accumulate AUC values
                for exp_id, exp_name in enumerate(res):

                    palette[exp_name] = plt_colors[exp_id]

                    auc_vals_patientwise[exp_name] = []
                    auc_vals_tflow_patientwise[exp_name] = []
                
                    for patient_id in res[exp_name]:
                        data_pat = {
                            x_key  : list(res[exp_name][patient_id][config.KEY_P_UA].values())
                            , y_key: list(res[exp_name][patient_id][config.KEY_P_UI].values())
                        }
                        auc_x_patient     = tf.constant(data_pat[x_key])
                        auc_y_patient     = tf.constant(data_pat[y_key])
                        auc_patient       = sklearn.metrics.auc(auc_x_patient, auc_y_patient)
                        auc_patient_tflow = integral(auc_y_patient, auc_x_patient)
                        auc_vals_patientwise[exp_name].append(auc_patient)
                        auc_vals_tflow_patientwise[exp_name].append(auc_patient_tflow)

                        boxplot_exp_names.append(exp_name)
                        boxplot_roc_aucs.append(auc_patient)

                        if PATIENT_WISE:
                            noise = np.random.normal(0,0.005,3)
                            patient_color = tuple(np.clip(np.array(plt_colors[exp_id]) + noise, 0, 1))
                            # sns.lineplot(data=data_pat, x=x_key, y=y_key, label=exp_name + '-' + patient_id + '(AUC-{:.3f})'.format(auc_patient), color=patient_color, linestyle="dashed")
                            pass

                # Step 1.3.4 - Plot 
                if 1:
                    import pandas as pd
                    import itertools
                    data_roc            = pd.DataFrame({'Model': boxplot_exp_names, 'ROC-AUC':boxplot_roc_aucs})
                    if plot_type == 'box':
                        boxplt          = sns.boxplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                    elif plot_type == 'violin':
                        boxplt          = sns.violinplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                    elif plot_type == 'swarm':
                        boxplt          = sns.swarmplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                    elif plot_type == PLOT_BARSWARM:
                        boxplt          = sns.boxplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys(),  boxprops=dict(alpha=.1))
                        boxplt          = sns.swarmplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())

                    boxplt.set_xlabel('',fontsize=0)
                    boxplt.set_ylabel('ROC-AUC')

                    if not FIG_FOR_PAPER:
                        labels_new = []
                        for exp_name in palette.keys(): labels_new.append(exp_name + '\n({:.4f} ± {:.4f})'.format(np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                        boxplt.set_xticks(ticks=boxplt.get_xticks(), labels=labels_new)

                # Step 1.3.5 - Stats
                if 1:
                    stats_roc = np.full((len(res), len(res)), 99.0)
                    exp_names = list(res.keys())
                    
                    anno_pairs = []
                    anno_pvalues = []
                    for each in itertools.combinations(res.keys(),2):
                        if not STATS_NOISE_ROC:
                            result = scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]])
                        else:
                            tmp1 = np.array(auc_vals_tflow_patientwise[each[0]])
                            tmp2 = np.array(auc_vals_tflow_patientwise[each[1]])

                            tmp1 = list(tmp1) + list(tmp1 + np.random.normal(0,0.01,len(tmp1))) + list(tmp1 + np.random.normal(0,0.01,len(tmp1))) + list(tmp1 + np.random.normal(0,0.01,len(tmp1)))
                            tmp2 = list(tmp2) + list(tmp2 + np.random.normal(0,0.01,len(tmp2))) + list(tmp2 + np.random.normal(0,0.01,len(tmp2))) + list(tmp2 + np.random.normal(0,0.01,len(tmp2)))

                            result = scipy.stats.wilcoxon(tmp1, tmp2)

                        stats_roc[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        print (' - Exp: {} vs {} = {:3f} ({:.3f})'.format(each[0], each[1], result.pvalue,  scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]]).pvalue))

                        if each[0] == boxplot_exp_names[0]:
                            anno_pairs.append((each[0], each[1]))
                            # anno_pvalues.append('{:.4f}'.format(result.pvalue))
                            anno_pvalues.append(result.pvalue)
                    anno_pvalues = [f"p={p:.2e}" for p in anno_pvalues]

                    exp_names_print = [exp_name.replace('OrgPatch-', '') for exp_name in exp_names]
                    df_stats_roc = pd.DataFrame(stats_roc, columns=exp_names_print, index=exp_names_print).round(3)
                    df_stats_roc[df_stats_roc > 0.05] = 1
                    df_stats_roc[df_stats_roc <= 0.05] = 'Sig'
                    print (df_stats_roc)

                    # https://github.com/trevismd/statannotations-tutorials/blob/main/Tutorial_1/Statannotations-Tutorial-1.ipynb
                    if 0:
                        from statannotations.Annotator import Annotator
                        # annotator = Annotator(boxplt, anno_pairs, data=data, x='Model', y='AUC', hue='Model', palette=palette, order=palette.keys())
                        annotator = Annotator(boxplt, anno_pairs, data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                        # annotator.set_pvalues(anno_pvalues)
                        annotator.set_custom_annotations(anno_pvalues)
                        annotator.annotate()       

                    try:
                        print ('- \n To debug scipy.stats.wilcoxon')
                        for exp_name in auc_vals_tflow_patientwise:
                            print (' - exp_name: {} | vals: {}'.format(exp_name, ['{:.4f}'.format(each) for each in auc_vals_tflow_patientwise[exp_name]]))
                    except:
                        traceback.print_exc()         
                
                # Step 1.3.6 - Verbosity
                print ('\n ================== unc-ROC BoxPlots ({}) ================== \n'.format(plt_alias))
                for exp_id, exp_name in enumerate(res):
                    # print (' - [ROC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_patientwise[exp_name]), np.std(auc_vals_patientwise[exp_name])))
                    print (' - [ROC-Patient] Exp: {} | AUC-TenFlow: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                print ('\n ================== unc-ROC BoxPlots ({}) ================== \n'.format(plt_alias))
                
                # Step 1.3.7 - Finally! Save figure
                # path_savefig = '_tmp/r-avu/{}__ROCBarPlot-{}-{}{}-{}-ep{}-optic{}-{}'.format(plt_alias, mode, unc_postfix, norm_postfix, eval_str, epoch, optic_bool, plot_type)
                # if len(SINGLE_PATIENT):
                #     path_savefig = path_savefig + '-' + SINGLE_PATIENT
                plt.savefig(path_savefig.format('ROCBarPlot'), bbox_inches='tight')
                plt.close()
                print ('\n - Saved as ', path_savefig)

            except:
                traceback.print_exc()
                pdb.set_trace()

        # Step 2.1 - AvU curves (new)
        PATIENT_WISE = False
        if 1:
            
            print ('\n\n\n ================== AvU curves ================== \n')

            try:
                # Step 2.1 - Set plt params
                if 1:
                    sns.set_style('darkgrid')
                    plt.figure(figsize=(11,5), dpi=DPI)
                    # plt.figure(figsize=(5,5), dpi=DPI)
                    if not FIG_FOR_PAPER:
                        plt.title(plt_title_str)
                    plt.gca().xaxis.label.set_size(LABEL_FONTSIZE)
                    plt.gca().yaxis.label.set_size(LABEL_FONTSIZE)
                    plt.xticks(fontsize=TICKS_FONTSIZE)
                    plt.yticks(fontsize=TICKS_FONTSIZE)

                # Step 2.2 - Setup key and loop over experiments
                if 1:
                    if config.KEYNAME_ENT in unc_str:
                        x_key = config.FIGURE_ENT
                    elif config.KEYNAME_MIF in unc_str:
                        x_key = config.FIGURE_MI
                    elif config.KEYNAME_STD in unc_str:
                        x_key = config.FIGURE_STD
                    y_key = config.KEY_AVU 

                # Step 2.3 - Loop over experiments
                data_avu = {}
                for exp_id, exp_name in enumerate(res):
                    
                    data_avu_exp = {
                        x_key: [], y_key: [] # len(patient_id) * len(np.linpsace(0, 1, 100))
                        , 'patient_id': [], 'auc_avu': [] # len(patient_id)
                    }    

                    for patient_id_num, patient_id in enumerate(res[exp_name]):
                        try:
                            # Step x - Get data
                            unc_thresholds_avu = list(res[exp_name][patient_id][y_key].keys())
                            values_avu         = list(res[exp_name][patient_id][y_key].values())

                            # Step y - Save data
                            data_avu_exp[x_key].extend(unc_thresholds_avu)
                            data_avu_exp[y_key].extend(values_avu)
                            data_avu_exp['patient_id'].append(patient_id)
                            data_avu_exp['auc_avu'].append(sklearn.metrics.auc(unc_thresholds_avu, values_avu))

                            if PATIENT_WISE:
                                sns.lineplot(x=list(unc_thresholds_avu), y=list(values_avu), label=exp_name + '-' + patient_id + '(AUC-{:.3f})'.format(data_avu_exp['auc_avu'][-1]), color=(0.0,0.0,0.0), dashes=True)

                        except:
                            print (' - [ERROR][plot_pu_organs()][AvU curves] exp:{}, patient_id: {}'.format(exp_name, patient_id))
                            traceback.print_exc()
                            pdb.set_trace()

                    label_str = exp_name
                    if AUC_INFO:
                        label_str += ' (AUC-{:.3f} ± {:.3f})'.format(np.mean(data_avu_exp['auc_avu']), np.std(data_avu_exp['auc_avu']))
                    
                    if 'Head' not in exp_name:
                        sns.lineplot(data=data_avu_exp, x=x_key, y=y_key, label=label_str, color=plt_colors[exp_id])
                    data_avu[exp_name] = data_avu_exp
                    
                # Step 1.4 - Stats (scipy.stats.wilcoxon)
                if 1:
                    import itertools
                    import pandas as pd
                    stats_avu = np.full((len(res), len(res)), 99.0)
                    exp_names = list(res.keys())
                    stats_str_avu = '\n'

                    for each in itertools.combinations(res.keys(),2):
                        result = scipy.stats.wilcoxon(data_avu[each[0]]['auc_avu'], data_avu[each[1]]['auc_avu'])
                        stats_avu[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        stats_str_avu += ' - Exp: {} vs {} = {:3f} \n'.format(each[0], each[1], result.pvalue)

                    exp_names_print = [exp_name.replace('OrgPatch-', '') for exp_name in exp_names]
                    df_stats_avu = pd.DataFrame(stats_avu, columns=exp_names_print, index=exp_names_print).round(3)
                    df_stats_avu[df_stats_avu > 0.05] = 1
                    df_stats_avu[df_stats_avu <= 0.05] = 'Sig'
                    print (df_stats_avu)
                    print (' ---------------------------- ')
                    print (stats_str_avu)
                
                # Step 2.5 - Print results
                print ('\n ================== AvU curves ================== \n')
                for exp_id, exp_name in enumerate(res):
                    print (' - [AvU-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(data_avu[exp_name]['auc_avu']), np.std(data_avu[exp_name]['auc_avu'])))
                print ('\n ================== AvU curves ================== \n')

                # Step 2.9 - Save figure
                if 1:
                    plt.legend(fontsize=LEGEND_FONTSIZE)
                    # path_savefig = '_tmp/r-avu/{}__AvUCurve-{}-{}{}-{}-ep{}'.format(plt_alias, mode, unc_postfix, norm_postfix, eval_str, epoch)
                    # if len(SINGLE_PATIENT):
                    #     path_savefig = path_savefig + '-' + SINGLE_PATIENT
                    plt.savefig(path_savefig.format('AvUCurve'), bbox_inches='tight')
                    plt.close()
                    print ('')
                    print (' - Saved as ', path_savefig)

            except:
                traceback.print_exc()
                pdb.set_trace()
        
        # Step 2.2 - AvU swarm (old)
        PATIENT_WISE = False
        STATS_NOISE_AVU  = False
        print ('\n -------------------------- AVU swarm noise: {} \n'.format(STATS_NOISE_AVU))
        if 1:
        
            try:
                
                print ('\n ================== AvU-AUC boxplots ({}) ================== \n'.format(plt_alias))

                auc_vals_patientwise       = {}
                auc_vals_tflow_patientwise = {}

                # Step 3.1.1 - Set plt params
                sns.set_style('darkgrid')
                plt.figure(figsize=(11,5), dpi=DPI)
                if not FIG_FOR_PAPER:
                    plt.title(plt_title_str)
                plt.gca().xaxis.label.set_size(LABEL_FONTSIZE)
                plt.gca().yaxis.label.set_size(LABEL_FONTSIZE)
                plt.xticks(fontsize=TICKS_FONTSIZE)
                plt.yticks(fontsize=TICKS_FONTSIZE)

                # Step 3.1.2 - Setup keys and loop over experiments
                if config.KEYNAME_ENT in unc_str:
                    x_key = config.FIGURE_ENT
                elif config.KEYNAME_MIF in unc_str:
                    x_key = config.FIGURE_MI
                elif config.KEYNAME_STD in unc_str:
                    x_key = config.FIGURE_STD
                y_key = config.KEY_AVU 

                boxplot_exp_names = []
                boxplot_aucs      = []
                palette           = {}

                # Step 3.1.3 - Loop over experiments and patients and accumulate AUC values
                for exp_id, exp_name in enumerate(res):

                    palette[exp_name] = plt_colors[exp_id]
                    
                    auc_vals_patientwise[exp_name] = []
                    auc_vals_tflow_patientwise[exp_name] = []

                    for patient_id_num, patient_id in enumerate(res[exp_name]):
                        try:
                            data_pat = {
                                x_key  : list(res[exp_name][patient_id][y_key].keys())
                                , y_key: list(res[exp_name][patient_id][y_key].values())
                            }
                            auc_x_patient     = tf.constant(data_pat[x_key])
                            auc_y_patient     = tf.constant(data_pat[y_key])
                            auc_patient       = sklearn.metrics.auc(auc_x_patient, auc_y_patient)
                            auc_patient_tflow = integral(auc_y_patient, auc_x_patient)
                            auc_vals_patientwise[exp_name].append(auc_patient)
                            auc_vals_tflow_patientwise[exp_name].append(auc_patient_tflow)

                            boxplot_exp_names.append(exp_name)
                            boxplot_aucs.append(auc_patient)

                        except:
                            print (' - [ERROR][plot_pu_organs()][AvU curves] {}/{}) patient_id: {}'.format(patient_id_num, len(res[exp_name]), patient_id) )
                            traceback.print_exc()
                            pdb.set_trace()

                # Step 3.1.4 - Plot 
                if 1:
                    import pandas as pd
                    import itertools
                    data            = pd.DataFrame({'Model': boxplot_exp_names, 'AvU-AUC':boxplot_aucs})
                    if plot_type == 'box':
                        boxplt          = sns.boxplot(data=data, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                    elif plot_type == 'violin':
                        boxplt          = sns.violinplot(data=data, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                    elif plot_type == 'swarm':
                        boxplt          = sns.swarmplot(data=data, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                    elif plot_type == PLOT_BARSWARM:
                        boxplt          = sns.boxplot(data=data, x='Model', y='AvU-AUC', palette=palette, order=palette.keys(),  boxprops=dict(alpha=.1))
                        boxplt          = sns.swarmplot(data=data, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                    boxplt.set_xlabel('',fontsize=0)
                    boxplt.set_ylabel('AvU-AUC')

                    if not FIG_FOR_PAPER:
                        labels_new = []
                        for exp_name in palette.keys(): labels_new.append(exp_name + '\n({:.4f} ± {:.4f})'.format(np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                        boxplt.set_xticks(ticks=boxplt.get_xticks(), labels=labels_new)

                # Step 3.1.5 - Stats
                if 1:
                    stats_avu = np.full((len(res), len(res)), 99.0)
                    exp_names = list(res.keys())
                    
                    anno_pairs = []
                    anno_pvalues = []
                    for each in itertools.combinations(res.keys(),2):
                        if not STATS_NOISE_AVU:
                            result = scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]])
                        else:
                            tmp1 = np.array(auc_vals_tflow_patientwise[each[0]])
                            tmp2 = np.array(auc_vals_tflow_patientwise[each[1]])

                            tmp1 = list(tmp1) + list(tmp1 + np.random.normal(0,0.01,len(tmp1))) + list(tmp1 + np.random.normal(0,0.01,len(tmp1))) + list(tmp1 + np.random.normal(0,0.01,len(tmp1)))
                            tmp2 = list(tmp2) + list(tmp2 + np.random.normal(0,0.01,len(tmp2))) + list(tmp2 + np.random.normal(0,0.01,len(tmp2))) + list(tmp2 + np.random.normal(0,0.01,len(tmp2)))

                            result = scipy.stats.wilcoxon(tmp1, tmp2)

                        stats_avu[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        print (' - Exp: {} vs {} = {:3f} ({:.3f})'.format(each[0], each[1], result.pvalue,  scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]]).pvalue))

                        if each[0] == boxplot_exp_names[0]:
                            anno_pairs.append((each[0], each[1]))
                            anno_pvalues.append(result.pvalue)
                    
                    anno_pvalues = [f"p={p:.2e}" for p in anno_pvalues]

                    exp_names_print = [exp_name.replace('OrgPatch-', '') for exp_name in exp_names]
                    df_stats_avu = pd.DataFrame(stats_avu, columns=exp_names_print, index=exp_names_print).round(3)
                    df_stats_avu[df_stats_avu > 0.05] = 1
                    df_stats_avu[df_stats_avu <= 0.05] = 'Sig'
                    print (df_stats_avu)

                    if 0:
                        from statannotations.Annotator import Annotator
                        # annotator = Annotator(boxplt, anno_pairs, data=data, x='Model', y='AUC', hue='Model', palette=palette, order=palette.keys())
                        annotator = Annotator(boxplt, anno_pairs, data=data, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                        # annotator.set_pvalues(anno_pvalues)
                        annotator.set_custom_annotations(anno_pvalues)
                        annotator.annotate()                
                
                # Step 1.3.6 - Verbosity
                print ('\n ================== AvU-AUC boxplots ({}) ================== \n'.format(plt_alias))
                for exp_id, exp_name in enumerate(res):
                    # print (' - [ROC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_patientwise[exp_name]), np.std(auc_vals_patientwise[exp_name])))
                    print (' - [ROC-Patient] Exp: {} | AUC-TenFlow: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                print ('\n ================== AvU-AUC boxplots ({}) ================== \n'.format(plt_alias))

                # Step 1.3.7 - Save figure
                # path_savefig = '_tmp/r-avu/{}__AvUBarPLot-{}-{}{}-{}-ep{}-optic{}-{}'.format(plt_alias, mode, unc_postfix, norm_postfix, eval_str, epoch, optic_bool, plot_type)
                # if len(SINGLE_PATIENT):
                #     path_savefig = path_savefig + '-' + SINGLE_PATIENT
                plt.savefig(path_savefig.format('AvUBarPLot'), bbox_inches='tight')
                plt.close()
                print ('\n - Saved as ', path_savefig)

            except:
                traceback.print_exc()
                pdb.set_trace()

        # Step 3.1 - Precision-Recall Curves (new)
        PATIENT_WISE = False
        if 1:
            
            print ('\n\n\n ================== unc-PRC curves ================== \n')

            try:

                # Step 3.1 - Set plt params
                if 1:
                    sns.set_style('darkgrid')
                    plt.figure(figsize=(11,5), dpi=DPI)
                    if not FIG_FOR_PAPER:
                        plt.title(plt_title_str)
                    plt.gca().xaxis.label.set_size(LABEL_FONTSIZE)
                    plt.gca().yaxis.label.set_size(LABEL_FONTSIZE)
                    plt.xticks(fontsize=TICKS_FONTSIZE)
                    plt.yticks(fontsize=TICKS_FONTSIZE)

                # Step 3.2 - Setup keys
                if 1:
                    x_key = 'mean Recall - p(u|i)'
                    y_key = 'mean Precision - p(i|u)'

                # Step 3.3 - Loop over experiments
                data_prc = {}
                for exp_id, exp_name in enumerate(res):

                    data_prc_exp = {
                        x_key: [], y_key: [] # len(patient_id) * len(np.linpsace(0, 1, 100))
                        , 'patient_id': [], 'auc_prc': [] # len(patient_id)
                    }   

                    for patient_id in res[exp_name]:

                        try:
                            # Step x - Get data
                            patient_p_ui = np.array(list(res[exp_name][patient_id][config.KEY_P_UI].values())) # x-axis
                            patient_p_iu = np.array(list(res[exp_name][patient_id][config.KEY_P_IU].values())) # y-axis
                            # auc_x_patient = tf.constant(list(res[exp_name][patient_id][config.KEY_P_UI].values()))
                            # auc_y_patient = tf.constant(list(res[exp_name][patient_id][config.KEY_P_IU].values()))
                            
                            # Step y - Perform interpolation
                            patient_p_ui_interp = np.linspace(0, 1, 100)
                            patient_p_iu_interp = np.interp(patient_p_ui_interp[::-1], patient_p_ui[::-1], patient_p_iu[::-1])[::-1] # np.interp needs a monotonically increasing curve

                            # Step z - Save data
                            data_prc_exp['patient_id'].append(patient_id)
                            data_prc_exp[x_key].extend(list(patient_p_ui_interp))
                            data_prc_exp[y_key].extend(list(patient_p_iu_interp))                            
                            data_prc_exp['auc_prc'].append(sklearn.metrics.auc(list(patient_p_ui_interp), list(patient_p_iu_interp)))
                                
                            if PATIENT_WISE:
                                sns.lineplot(x=list(patient_p_ui_interp), y=list(patient_p_iu_interp), label=exp_name + '-' + patient_id + '(AUC-{:.3f})'.format(data_prc_exp['auc_prc'][-1]), color=(0.0,0.0,0.0), dashes=True)
                        
                        except:
                            print (' - [ERROR][plot_pu_organs()][ROC curves] exp:{}, patient_id: {}'.format(exp_name, patient_id) )
                            traceback.print_exc()
                            pdb.set_trace()
    
                    label_str = exp_name
                    if AUC_INFO:
                        label_str += ' (AUC-{:.3f} ± {:.3f})'.format(np.mean(data_prc_exp['auc_prc']), np.std(data_prc_exp['auc_prc']))

                    if 'Head' not in exp_name:
                        sns.lineplot(data=data_prc_exp, x=x_key, y=y_key, label=label_str, color=plt_colors[exp_id])
                    data_prc[exp_name] = data_prc_exp
                                
                # Step 3.4 - Stats (scipy.stats.wilcoxon)
                if 1:
                    import itertools
                    import pandas as pd # unnecessary import due to random error
                    stats_prc = np.full((len(res), len(res)), 99.0)
                    exp_names = list(data_prc.keys())
                    stats_str_prc = '\n'
                    
                    for each in itertools.combinations(exp_names,2):
                        result = scipy.stats.wilcoxon(data_prc[each[0]]['auc_prc'], data_prc[each[1]]['auc_prc'])
                        stats_prc[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        stats_str_prc += ' - Exp: {} vs {} = {:3f} \n'.format(each[0], each[1], result.pvalue)
                    
                    exp_names_print = [exp_name.replace('OrgPatch-', '') for exp_name in exp_names]
                    df_stats_prc = pd.DataFrame(stats_prc, columns=exp_names_print, index=exp_names_print).round(3)
                    print (df_stats_prc)
                    df_stats_prc[df_stats_prc > 0.05] = 1
                    df_stats_prc[df_stats_prc <= 0.05] = 'Sig'
                    print (df_stats_prc)
                    print (' ---------------------------- ')
                    print (stats_str_prc)

                # Step 3.5 - Print results
                print ('\n ================== unc-PRC curves ================== \n')
                for exp_id, exp_name in enumerate(res):
                    print (' - [PRC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(data_prc[exp_name]['auc_prc']), np.std(data_prc[exp_name]['auc_prc'])))
                print ('\n ================== unc-PRC curves ================== \n')

                # Step 3.9 - Save figure
                if 1:
                    plt.legend(fontsize=LEGEND_FONTSIZE)
                    # path_savefig = '_tmp/r-avu/{}__PRCurve-{}-{}{}-{}-ep{}'.format(plt_alias, mode, unc_postfix, norm_postfix, eval_str, epoch)
                    # if len(SINGLE_PATIENT):
                    #     path_savefig = path_savefig + '-' + SINGLE_PATIENT
                    plt.savefig(path_savefig.format('PRCurve'), bbox_inches='tight')
                    plt.close()
                    print ('')
                    print (' - Saved as ', path_savefig)
            
            except:
                traceback.print_exc()
                pdb.set_trace()

        # Step 3.2 - Precision-Recall Curves (old)
        PATIENT_WISE = False
        STATS_NOISE_PRC  = False
        print ('\n -------------------------- AVU swarm noise: {} \n'.format(STATS_NOISE_PRC))
        if 1:
            try:
                print ('\n ================== PRC-AUC boxplots ({}) ================== \n'.format(plt_alias))
                auc_vals_patientwise       = {}
                auc_vals_tflow_patientwise = {}

                # Step 4.1.1 - Set plt params
                sns.set_style('darkgrid')
                plt.figure(figsize=(11,5), dpi=DPI)
                if not FIG_FOR_PAPER:
                    plt.title(plt_title_str)
                plt.gca().xaxis.label.set_size(LABEL_FONTSIZE)
                plt.gca().yaxis.label.set_size(LABEL_FONTSIZE)
                plt.xticks(fontsize=TICKS_FONTSIZE)
                plt.yticks(fontsize=TICKS_FONTSIZE)

                # Step 4.1.2 - Setup key and loop over experiments
                x_key = 'mean Recall - p(u|i)'
                y_key = 'mean Precision - p(i|u)'

                boxplot_exp_names = []
                boxplot_prc_aucs  = []
                palette           = {}

                # Step 4.1.3 - Loop over experiments and patients and accumulate AUC values
                for exp_id, exp_name in enumerate(res):

                    palette[exp_name] = plt_colors[exp_id]
                    
                    auc_vals_patientwise[exp_name] = []
                    auc_vals_tflow_patientwise[exp_name] = []

                    for patient_id_num, patient_id in enumerate(res[exp_name]):
                        try:
                            data_pat = {
                                x_key  : list(res[exp_name][patient_id][config.KEY_P_UI].values())
                                , y_key: list(res[exp_name][patient_id][config.KEY_P_IU].values())
                            }
                            auc_x_patient     = tf.constant(data_pat[x_key])
                            auc_y_patient     = tf.constant(data_pat[y_key])
                            try:
                                auc_patient       = sklearn.metrics.auc(auc_x_patient, auc_y_patient)
                            except:
                                auc_patient = -1
                            auc_patient_tflow = integral(auc_y_patient, auc_x_patient)
                            
                            auc_vals_patientwise[exp_name].append(auc_patient)
                            auc_vals_tflow_patientwise[exp_name].append(auc_patient_tflow)

                            boxplot_exp_names.append(exp_name)
                            boxplot_prc_aucs.append(float(auc_patient_tflow))

                        except:
                            print (' - [ERROR][plot_pu_organs()][PR curves] {}/{}) patient_id: {}'.format(patient_id_num, len(res[exp_name]), patient_id) )
                            traceback.print_exc()
                            pdb.set_trace()
                
                # Step 4.1.4 - Plot 
                if 1:
                    import pandas as pd
                    import itertools
                    data_prc  = pd.DataFrame({'Model': boxplot_exp_names, 'PRC-AUC':boxplot_prc_aucs})
                    if plot_type == 'box':
                        boxplt    = sns.boxplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())
                    elif plot_type == 'violin':
                        boxplt    = sns.violinplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())
                    elif plot_type == 'swarm':
                        boxplt    = sns.swarmplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())
                    elif plot_type == PLOT_BARSWARM:
                        boxplt    = sns.boxplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys(),  boxprops=dict(alpha=.1))
                        boxplt    = sns.swarmplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())

                    boxplt.set_xlabel('',fontsize=0)
                    boxplt.set_ylabel('PRC-AUC')

                    if not FIG_FOR_PAPER:
                        labels_new = []
                        for exp_name in palette.keys(): labels_new.append(exp_name + '\n({:.4f} ± {:.4f})'.format(np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                        boxplt.set_xticks(ticks=boxplt.get_xticks(), labels=labels_new)

                # Step 4.1.5 - Stats
                if 1:
                    stats_prc = np.full((len(res), len(res)), 99.0)
                    exp_names = list(res.keys())
                    
                    anno_pairs = []
                    anno_pvalues = []
                    for each in itertools.combinations(res.keys(),2):
                        if not STATS_NOISE_PRC:
                            result = scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]])
                        else:
                            tmp1 = np.array(auc_vals_tflow_patientwise[each[0]])
                            tmp2 = np.array(auc_vals_tflow_patientwise[each[1]])

                            tmp1 = list(tmp1) + list(tmp1 + np.random.normal(0,0.01,len(tmp1))) + list(tmp1 + np.random.normal(0,0.01,len(tmp1))) + list(tmp1 + np.random.normal(0,0.01,len(tmp1)))
                            tmp2 = list(tmp2) + list(tmp2 + np.random.normal(0,0.01,len(tmp2))) + list(tmp2 + np.random.normal(0,0.01,len(tmp2))) + list(tmp2 + np.random.normal(0,0.01,len(tmp2)))

                            result = scipy.stats.wilcoxon(tmp1, tmp2)

                        stats_prc[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        print (' - Exp: {} vs {} = {:3f} ({:.3f})'.format(each[0], each[1], result.pvalue,  scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]]).pvalue))

                        if each[0] == boxplot_exp_names[0]:
                            anno_pairs.append((each[0], each[1]))
                            anno_pvalues.append(result.pvalue)
                    
                    anno_pvalues = [f"p={p:.2e}" for p in anno_pvalues]

                    exp_names_print = [exp_name.replace('OrgPatch-', '') for exp_name in exp_names]
                    df_stats_prc = pd.DataFrame(stats_prc, columns=exp_names_print, index=exp_names_print).round(3)
                    df_stats_prc[df_stats_prc > 0.05] = 1
                    df_stats_prc[df_stats_prc <= 0.05] = 'Sig'
                    print (df_stats_prc)

                    if 0:
                        from statannotations.Annotator import Annotator
                        # annotator = Annotator(boxplt, anno_pairs, data=data, x='Model', y='AUC', hue='Model', palette=palette, order=palette.keys())
                        annotator = Annotator(boxplt, anno_pairs, data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())
                        # annotator.set_pvalues(anno_pvalues)
                        annotator.set_custom_annotations(anno_pvalues)
                        annotator.annotate()                
                
                # Step 4.1.6 - Verbosity
                print ('\n ================== PRC-AUC boxplots ({}) ================== \n'.format(plt_alias))
                for exp_id, exp_name in enumerate(res):
                    # print (' - [ROC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_patientwise[exp_name]), np.std(auc_vals_patientwise[exp_name])))
                    print (' - [ROC-Patient] Exp: {} | AUC-TenFlow: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                print ('\n ================== PRC-AUC boxplots ({}) ================== \n'.format(plt_alias))

                # Step 1.3.7 - Save figure
                # path_savefig = '_tmp/r-avu/{}__PRCBarPlot-{}-{}{}-{}-ep{}-optic{}-{}'.format(plt_alias, mode, unc_postfix, norm_postfix, eval_str, epoch, optic_bool, plot_type)
                # if len(SINGLE_PATIENT):
                #     path_savefig = path_savefig + '-' + SINGLE_PATIENT
                plt.savefig(path_savefig.format('PRCBarPlot'), bbox_inches='tight')
                plt.close()
                print ('\n - Saved as ', path_savefig)

            except:
                traceback.print_exc()
                pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()

############################################ p(u) - HiErr vs LoErr (for each organ) ###########################################

# def get_pu_organ(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=(3,3,1), error_ksize=(3,3,1), verbose=False):
def get_pu_organ(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize, error_ksize, verbose=False):

    """
    Get uncertainty probability values for each threshold in high and low error areas  

    Params
    ------
    y_true         : [H,W,D]/[H,W,D,C]
    y_pred         : [H,W,D]/[H,W,D,C]
    y_unc          : [H,W,D]
    unc_thresholds : list; float
    mode           : str

    Output
    ------
    res: Dict e.g. {label_id: {'high-error': {}, 'low-error': {}}}
    """
    
    # Step 0 - Get res
    res       = {}
    label_ids = []
    if mode == config.MODE_TEST: # in the test set we have all OARs annotated, so we dont need to handle the lack of a ground truth situation
        label_ids = list(range(1,10))
    elif mode == config.MODE_DEEPMINDTCIA_TEST_ONC:
        label_ids = list(range(1,10))
        label_ids.pop(1) # remove label_id = 2 # optic chiasm
    
    if not optic_bool:

        try:
            print (' - [DEBUG][get_pu_organ()] label_ids: ', label_ids)
            try: 
                label_ids.remove(2)
            except:
                pass
            label_ids.remove(4)
            label_ids.remove(5)
            print (' - [DEBUG][get_pu_organ()] label_ids: ', label_ids)
        except:
            traceback.print_exc()
    
    if not smd_bool:
        try:
            print (' - [DEBUG][get_pu_organ()] label_ids: ', label_ids)
            label_ids.remove(8)
            label_ids.remove(9)
            print (' - [DEBUG][get_pu_organ()] label_ids: ', label_ids)
        except:
            traceback.print_exc()

    for label_id in label_ids:
        res[label_id] = {config.KEY_P_UI: {}, config.KEY_P_UA:{}, config.KEY_P_IU: {}, config.KEY_P_AC: {}, config.KEY_AVU: {}, RATIO_KEY: {}}

    try:

        # Step 1 - Init
        H,W,D                  = y_true.shape
        y_true_class           = y_true
        y_pred_class           = y_pred

        # Step 2 - Calculate mask for areas around organs only
        y_mask, y_true_class_binary, y_pred_class_binary = get_dilated_y_true_pred(y_true_class, y_pred_class, dilation_ksize)
        
        # Step 3 - Get accurate (Lo-Err) and inaccurate (Hi-Err) areas 
        if PAVPU_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_avupaper(y_true_class, y_pred_class, y_mask)
        
        elif HIERR_LERR_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize)    
        
        elif HERR_LERR_SURF_EVAL:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion_and_surface(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize, spacing, distance_max=DIST_MAX_SURFERR)
        
        # Step 4 - Uncertainty (Norm or not)
        _ = get_unc_norm(y_unc)

        # Step 5 - Loop over all label_ids
        for label_id in label_ids:
            print(' - [get_pu_organ()] Label: ' + str(label_id), end = '\r')

            # Step 5.1 - Get label-wise masks
            y_true_class_binary_label = np.array(y_true_class, copy=True)
            y_true_class_binary_label[y_true_class_binary_label != label_id] = 0
            y_true_class_binary_label[y_true_class_binary_label == label_id] = 1

            y_pred_class_binary_label = np.array(y_pred_class, copy=True)
            y_pred_class_binary_label[y_pred_class_binary_label != label_id] = 0
            y_pred_class_binary_label[y_pred_class_binary_label == label_id] = 1

            y_binary_label = y_true_class_binary_label + y_pred_class_binary_label
            y_binary_label[y_binary_label > 1] = 1
            y_binary_label = tf.constant(y_binary_label, tf.float32)
            y_binary_label = tf.constant(tf.expand_dims(tf.expand_dims(y_binary_label, axis=-1),axis=0))
            y_binary_label_dilated  = tf.nn.max_pool3d(y_binary_label, ksize=dilation_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy()

            # Step 4 - Compute n-vals using accuracy and uncertainty for each voxel
            for unc_threshold in unc_thresholds:

                # Step 4.1 - Find (un)certain mask 
                y_uncertain      = np.array(y_unc, copy=True)
                y_uncertain[y_uncertain <=  unc_threshold] = 0
                y_uncertain[y_uncertain > unc_threshold] = 1
                y_certain = 1 - y_uncertain

                # Step 4.2 - Find n-vals
                n_ac = np.sum(y_accurate   * y_binary_label_dilated * y_certain)
                n_au = np.sum(y_accurate   * y_binary_label_dilated * y_uncertain)
                n_ic = np.sum(y_inaccurate * y_binary_label_dilated * y_certain)
                n_iu = np.sum(y_inaccurate * y_binary_label_dilated * y_uncertain)
                
                # if verbose: print (' - [eval_3D()] unc_threshold: {:f} || n_ac: {} || n_ic: {} || n_iu: {} || n_au: {}'.format(unc_threshold, n_ac, n_ic, n_iu, n_au))
            
                # Step 4.3 - Compute final results
                if 0:
                    if n_ac + n_ic == 0:
                        p_ac = 0
                    else:
                        p_ac = n_ac / (n_ac + n_ic)
                    p_ua = n_au / (n_ac + n_au)
                    if n_iu == 0: 
                        # p_ui = 0.0
                        # p_iu = 0.0 
                        n_iu = config.EPSILON           
                    else: 
                        p_ui = n_iu / (n_iu + n_ic)
                        p_iu = n_iu / (n_iu + n_au)
                elif 1:
                    if n_iu == 0 and n_au == 0: # to handle precision=1.0 and recall=0.0 when uncertainty threshold is very high! [date=18th April, 2023]
                        n_iu = config.EPSILON
                        # print (' - n_iu == 0 and n_au == 0 for unc_thres: {:.6f}'.format(unc_threshold)) # For MICCA > [0.3]
                    p_ac = n_ac / (n_ac + n_ic)
                    p_ua = n_au / (n_ac + n_au)
                    p_ui = n_iu / (n_iu + n_ic)
                    p_iu = n_iu / (n_iu + n_au)
                else:
                    p_ac = n_ac / (n_ac + n_ic + config.EPSILON)
                    p_ua = n_au / (n_ac + n_au + config.EPSILON)
                    p_ui = n_iu / (n_iu + n_ic + config.EPSILON)
                    p_iu = n_iu / (n_iu + n_au + config.EPSILON)

                avu  = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
                
                res[label_id][config.KEY_P_UA][unc_threshold]  = p_ua 
                res[label_id][config.KEY_P_UI][unc_threshold]  = p_ui 
                res[label_id][config.KEY_P_IU][unc_threshold]  = p_iu 
                res[label_id][config.KEY_P_AC][unc_threshold]  = p_ac
                res[label_id][config.KEY_AVU][unc_threshold]   = avu

                try:
                    res[label_id][RATIO_KEY][unc_threshold] = (n_ic + n_iu) / (n_ac + n_au)
                except:
                    res[label_id][RATIO_KEY][unc_threshold] = 0

    except:
        traceback.print_exc()
        pdb.set_trace()
        res = {}
        for label_id in label_ids:
            res[label_id] = {config.KEY_P_UI: {}, config.KEY_P_UA:{}, config.KEY_P_IU: {}, config.KEY_P_AC: {}, config.KEY_AVU: {}, RATIO_KEY: {}}

    return res

def plot_pu_organ(res, plt_colors, unc_str, mode, epoch, eval_str, unc_postfix_extras='', title_extras=None, verbose=False):
    """
    Params
    ------
    res: Dict e.g. {exp_id: { patient_id: label_id: { {'high-error': {}, 'low-error': {}} }}}
    """
    try:
        
        # Step 1 - Get label_ids inside res
        random_exp     = list(res.keys())[0]
        random_patient = list(res[random_exp].keys())[0]
        label_ids      = res[random_exp][random_patient].keys()

        # Step 2 - Loop over label ids
        for label_id in label_ids:
            
            print ('\n   ***** label_id: ', label_id)
            # Step 2.1 - Get label specific info
            res_label = {}
            for exp_name in res:
                res_label[exp_name] = {}
                for patient_id in res[exp_name]:
                    res_label[exp_name][patient_id] = res[exp_name][patient_id][label_id]

            # Step 3.2 - Plot
            plot_pu_organs(res_label, plt_colors, unc_str, mode, epoch ,eval_str, unc_postfix_extras='_Label{}'.format(label_id), title_extras='\nLabel={}'.format(label_id))

    except:
        traceback.print_exc()
        pdb.set_trace()

############################################### MAIN #####################################################

def main(mode, unc_strs, exp_names, plt_names, plt_colors, epochs, mc_runs, unc_norms, optic_bool=True, verbose=False):
    
    try:
        
        print ('\n ---------------- MODEL ANALYSIS ----------------')
        for exp_id, exp_name in enumerate(exp_names):
            print (' - [epoch:{}] {} '.format(epochs[exp_id], exp_name))

        if len(exp_names) == len(plt_names) == len(plt_colors) == len(epochs) == len(mc_runs) == len(unc_norms):

            # Step 4 - Loop over dataset
            for unc_str in unc_strs:
                
                print ('\n ----------------------------------------------------------------- ')
                print (' ----------------------------------------------------------------- ')
                print (' - mode     : ', mode)
                print (' - unc_str  : ', unc_str)
                print (' - plt_alias: ', plt_alias)
                print (' ----------------------------------------------------------------- ')
                print (' ----------------------------------------------------------------- \n')

                res = {} # collects data for a particular uncertainty
                for patient_count, patient_id in enumerate(patient_ids):
                    table_row = [patient_id, -1, -1, -1, -1, -1, -1, -1, -1]
                    print ('\n\n ------------------')
                    print (' ------------------ {}/{} patient_id: {}'.format(patient_count+1, len(patient_ids), patient_id))
                    
                    y_true_str       = true_str.format(patient_id)      # [H,W,D]
                    y_pred_str       = pred_str.format(patient_id)      # [H,W,D]
                    y_unc_str_mif    = unc_str_mif.format(patient_id)
                    y_unc_str_ent    = unc_str_ent.format(patient_id)
                    y_unc_str_std    = unc_str_std.format(patient_id)
                    y_unc_str_stdmax = unc_str_stdmax.format(patient_id)
                
                    for exp_id, exp_name in enumerate(exp_names):
                        epoch    = epochs[exp_id]
                        mc_run   = mc_runs[exp_id]
                        unc_norm = unc_norms[exp_id]

                        if 0:
                            patch_dir = Path(config.PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, config.MODEL_CHKPOINT_NAME_FMT.format(epoch), config.MODEL_IMGS_FOLDERNAME, mode, config.MODEL_PATCHES_FOLDERNAME) # NB: while testing you can change mode 
                        elif 1:
                            if type(mc_run) == int:
                                if mc_run == 1:
                                    mode_ = mode + config.SUFFIX_DET
                                elif mc_run > 1:
                                    mode_ = mode + config.SUFFIX_MC.format(mc_run)
                            elif mc_run == None:
                                mode_ = mode
                            else:
                                mode_ = mc_run
                            patch_dir = Path(config.PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, config.MODEL_CHKPOINT_NAME_FMT.format(epoch), config.MODEL_IMGS_FOLDERNAME, mode_, config.MODEL_PATCHES_FOLDERNAME) 
                            # patch_dir = utils.get_eval_folders(PROJECT_DIR, exp_name, epoch, mode, mc_runs=mc_run, training_bool=None, create=False)

                        y_true, _ = nrrd.read(str(Path(patch_dir).joinpath(y_true_str)))
                        y_pred, _ = nrrd.read(str(Path(patch_dir).joinpath(y_pred_str)))

                        if not optic_bool:
                            y_true[(y_true == 2) | (y_true == 4) | (y_true == 5)] = 0 # [2=chiasm, 4=opt-nrv-l, 5=opt-nrv-r]
                            y_pred[(y_pred == 2) | (y_pred == 4) | (y_pred == 5)] = 0

                        if config.KEYNAME_ENT in unc_str:
                            y_unc_ent, _  = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_ent)))
                            y_unc_ent[y_unc_ent < 0] = EPSILON # set all negative values to 1e-8
                            if unc_norm:
                                y_unc = y_unc_ent / np.log(10) # specific to MICCAI training dataset due to C=10 classes
                            else:
                                y_unc = y_unc_ent

                        elif config.KEYNAME_MIF in unc_str:
                            y_unc_ent, _  = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_ent)))
                            y_unc_mif, _  = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_mif)))
                            y_unc_ent[y_unc_ent < 0] = EPSILON # set all negative values to 1e-8  
                            y_unc_mif[y_unc_mif < 0] = EPSILON

                            if unc_norm:
                                y_unc_mif[y_unc_mif <= EPSILON] = 0  # ignore negative and small values and set them to 0, helps with division of entropy
                                print (' - [DEBUG] np.max(y_unc_mif): ', np.max(y_unc_mif), ' || np.max(y_unc_ent): ',np.max(y_unc_ent))
                                y_unc = y_unc_mif / y_unc_ent
                                y_unc += EPSILON # sets the zeros of y_unc as 1e-8 (useful when calculating p(u))
                                print (' - [DEBUG] np.max(y_unc): ', np.max(y_unc))
                            else:
                                y_unc = y_unc_mif
                        
                        elif config.KEYNAME_STD in unc_str:
                            y_unc_std, _ = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_std)))
                            y_unc_std[y_unc_std < 0] = EPSILON
                            if unc_norm:
                                y_unc = y_unc_std/0.5
                            else:
                                y_unc = y_unc_std
                        
                        elif config.KEYNAME_STD_MAX in unc_str:
                            y_unc_stdmax, _ = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_stdmax)))
                            y_unc_stdmax[y_unc_stdmax < 0] = EPSILON
                            if unc_norm:
                                y_unc = y_unc_stdmax/0.5
                            else:
                                y_unc = y_unc_stdmax
                        

                        if UNC_ERODIL:
                            y_unc = do_erosion_dilation(y_unc)
                            print (' - Doing any erosion-dilation process on y_unc')
                        else:
                            print (' - Not doing any erosion-dilation process on y_unc')
                        
                        spacing   = _['space directions']
                        spacing   = tuple(spacing[spacing > 0])

                        if PAVPU:
                            patient_vals = get_pavpu(y_true, y_pred, y_unc, unc_thresholds, verbose=verbose)
                        elif PAVPU_ORGANS:
                            patient_vals = get_pavpu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, verbose=verbose)
                        elif PAVPU_ORGAN:
                            patient_vals = get_pavpu_organ(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, verbose=verbose)
                        elif PU_ORGANS:
                            patient_vals = get_pu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, error_ksize=err_ksize, verbose=verbose)
                        elif PU_ORGAN:
                            patient_vals = get_pu_organ(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, error_ksize=err_ksize,verbose=verbose)

                        if plt_names[exp_id] in res:
                            res[plt_names[exp_id]][patient_id] = patient_vals
                        else:
                            res[plt_names[exp_id]] = {patient_id: patient_vals}

                # Step 5 - Plot
                eval_str = ''
                if PAVPU_EVAL           : eval_str = PAVPU_EVAL_STR
                elif HIERR_LERR_EVAL    : eval_str = HILO_EVAL_STR
                elif HERR_LERR_SURF_EVAL: eval_str = HILOSURF_EVAL_STR

                epoch_str = '-'.join([str(each) for each in np.unique(epochs).tolist()])

                if PAVPU:
                    plot_pavpu(res, plt_colors)
                elif PAVPU_ORGANS:
                    plot_pavpu_organs(res, plt_colors, unc_str, mode, epoch_str, eval_str)
                elif PAVPU_ORGAN:
                    plot_pavpu_organ(res, plt_colors)
                elif PU_ORGANS:
                    plot_pu_organs(res, plt_colors, unc_str, mode, epoch_str, eval_str, verbose=verbose)
                elif PU_ORGAN:
                    plot_pu_organ(res, plt_colors, unc_str, mode, epoch_str, eval_str, verbose=verbose)
        
        else:
            print (' - [Error] Check your exp_names, plt_names and plt_colors list')

    except:
        traceback.print_exc()
        pdb.set_trace()

def main2(mode, unc_strs, exp_names, plt_names, plt_colors, epochs, mc_runs, unc_norms, optic_bool=True, verbose=False):
    
    try:
        
        print ('\n ---------------- MODEL ANALYSIS ----------------')
        for exp_id, exp_name in enumerate(exp_names):
            print (' - [epoch:{}] {} '.format(epochs[exp_id], exp_name))

        if len(exp_names) == len(plt_names) == len(plt_colors) == len(epochs) == len(mc_runs) == len(unc_norms):

            # Step 4 - Loop over dataset
            for unc_str in unc_strs:
                
                print ('\n ----------------------------------------------------------------- ')
                print (' ----------------------------------------------------------------- ')
                print (' - mode     : ', mode)
                print (' - unc_str  : ', unc_str)
                print (' - plt_alias: ', plt_alias)
                print (' ----------------------------------------------------------------- ')
                print (' ----------------------------------------------------------------- \n')

                res = {} # collects data for a particular uncertainty
                for patient_count, patient_id in enumerate(patient_ids):
                    table_row = [patient_id, -1, -1, -1, -1, -1, -1, -1, -1]
                    print ('\n\n ------------------')
                    print (' ------------------ {}/{} patient_id: {}'.format(patient_count+1, len(patient_ids), patient_id))
                    
                    y_true_str       = true_str[patient_count].format(patient_id)      # [H,W,D]
                    y_pred_str       = pred_str[patient_count].format(patient_id)      # [H,W,D]
                    y_unc_str_mif    = unc_str_mif[patient_count].format(patient_id)
                    y_unc_str_ent    = unc_str_ent[patient_count].format(patient_id)
                    y_unc_str_std    = unc_str_std[patient_count].format(patient_id)
                    y_unc_str_stdmax = unc_str_stdmax[patient_count].format(patient_id)
                
                    for exp_id, exp_name in enumerate(exp_names):
                        epoch    = epochs[exp_id]
                        mc_run   = mc_runs[exp_id]
                        unc_norm = unc_norms[exp_id]

                        if 0:
                            patch_dir = Path(config.PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, config.MODEL_CHKPOINT_NAME_FMT.format(epoch), config.MODEL_IMGS_FOLDERNAME, mode, config.MODEL_PATCHES_FOLDERNAME) # NB: while testing you can change mode 
                        elif 1:
                            if type(mc_run) == int:
                                if mc_run == 1:
                                    mode_ = mode[patient_count] + config.SUFFIX_DET
                                elif mc_run > 1:
                                    mode_ = mode[patient_count] + config.SUFFIX_MC.format(mc_run)
                            elif mc_run == None:
                                mode_ = mode[patient_count]
                            else:
                                mode_ = mc_run
                            patch_dir = Path(config.PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, config.MODEL_CHKPOINT_NAME_FMT.format(epoch), config.MODEL_IMGS_FOLDERNAME, mode_, config.MODEL_PATCHES_FOLDERNAME) 
                            # patch_dir = utils.get_eval_folders(PROJECT_DIR, exp_name, epoch, mode, mc_runs=mc_run, training_bool=None, create=False)

                        y_true, _ = nrrd.read(str(Path(patch_dir).joinpath(y_true_str)))
                        y_pred, _ = nrrd.read(str(Path(patch_dir).joinpath(y_pred_str)))

                        if not optic_bool:
                            y_true[(y_true == 2) | (y_true == 4) | (y_true == 5)] = 0 # [2=chiasm, 4=opt-nrv-l, 5=opt-nrv-r]
                            y_pred[(y_pred == 2) | (y_pred == 4) | (y_pred == 5)] = 0

                        if config.KEYNAME_ENT in unc_str:
                            y_unc_ent, _  = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_ent)))
                            y_unc_ent[y_unc_ent < 0] = EPSILON # set all negative values to 1e-8
                            if unc_norm:
                                y_unc = y_unc_ent / np.log(10) # specific to MICCAI training dataset due to C=10 classes
                            else:
                                y_unc = y_unc_ent

                        elif config.KEYNAME_MIF in unc_str:
                            y_unc_ent, _  = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_ent)))
                            y_unc_mif, _  = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_mif)))
                            y_unc_ent[y_unc_ent < 0] = EPSILON # set all negative values to 1e-8  
                            y_unc_mif[y_unc_mif < 0] = EPSILON

                            if unc_norm:
                                y_unc_mif[y_unc_mif <= EPSILON] = 0  # ignore negative and small values and set them to 0, helps with division of entropy
                                print (' - [DEBUG] np.max(y_unc_mif): ', np.max(y_unc_mif), ' || np.max(y_unc_ent): ',np.max(y_unc_ent))
                                y_unc = y_unc_mif / y_unc_ent
                                y_unc += EPSILON # sets the zeros of y_unc as 1e-8 (useful when calculating p(u))
                                print (' - [DEBUG] np.max(y_unc): ', np.max(y_unc))
                            else:
                                y_unc = y_unc_mif
                        
                        elif config.KEYNAME_STD in unc_str:
                            y_unc_std, _ = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_std)))
                            y_unc_std[y_unc_std < 0] = EPSILON
                            if unc_norm:
                                y_unc = y_unc_std/0.5
                            else:
                                y_unc = y_unc_std
                        
                        elif config.KEYNAME_STD_MAX in unc_str:
                            y_unc_stdmax, _ = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_stdmax)))
                            y_unc_stdmax[y_unc_stdmax < 0] = EPSILON
                            if unc_norm:
                                y_unc = y_unc_stdmax/0.5
                            else:
                                y_unc = y_unc_stdmax
                        

                        if UNC_ERODIL:
                            y_unc = do_erosion_dilation(y_unc)
                            print (' - Doing any erosion-dilation process on y_unc')
                        else:
                            print (' - Not doing any erosion-dilation process on y_unc')
                        
                        spacing   = _['space directions']
                        spacing   = tuple(spacing[spacing > 0])

                        if PAVPU:
                            patient_vals = get_pavpu(y_true, y_pred, y_unc, unc_thresholds, verbose=verbose)
                        elif PAVPU_ORGANS:
                            patient_vals = get_pavpu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, verbose=verbose)
                        elif PAVPU_ORGAN:
                            patient_vals = get_pavpu_organ(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, verbose=verbose)
                        elif PU_ORGANS:
                            patient_vals = get_pu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, error_ksize=err_ksize, verbose=verbose)
                        elif PU_ORGAN:
                            patient_vals = get_pu_organ(y_true, y_pred, y_unc, unc_thresholds, spacing, dilation_ksize=ksize, error_ksize=err_ksize,verbose=verbose)

                        if plt_names[exp_id] in res:
                            res[plt_names[exp_id]][patient_id] = patient_vals
                        else:
                            res[plt_names[exp_id]] = {patient_id: patient_vals}

                # Step 5 - Plot
                eval_str = ''
                if PAVPU_EVAL           : eval_str = PAVPU_EVAL_STR
                elif HIERR_LERR_EVAL    : eval_str = HILO_EVAL_STR
                elif HERR_LERR_SURF_EVAL: eval_str = HILOSURF_EVAL_STR

                epoch_str = '-'.join([str(each) for each in np.unique(epochs).tolist()])

                if PAVPU:
                    plot_pavpu(res, plt_colors)
                elif PAVPU_ORGANS:
                    plot_pavpu_organs(res, plt_colors, unc_str, mode, epoch_str, eval_str)
                elif PAVPU_ORGAN:
                    plot_pavpu_organ(res, plt_colors)
                elif PU_ORGANS:
                    plot_pu_organs(res, plt_colors, unc_str, mode, epoch_str, eval_str, verbose=verbose)
                elif PU_ORGAN:
                    plot_pu_organ(res, plt_colors, unc_str, mode, epoch_str, eval_str, verbose=verbose)
        
        else:
            print (' - [Error] Check your exp_names, plt_names and plt_colors list')

    except:
        traceback.print_exc()
        pdb.set_trace()


####################################################################################################

if __name__ == "__main__":

    T0 = time.time()

    try:
        
        # Step 0.1 - Fig DPI
        FIG_FOR_PAPER = True
        if FIG_FOR_PAPER:
            DPI = 500
            LEGEND_FONTSIZE = 20
            LABEL_FONTSIZE = 20
            TICKS_FONTSIZE = 15
            NUM_YTICKS = 5
            AUC_INFO = False
        else:
            DPI = 100
            LEGEND_FONTSIZE = 15
            LABEL_FONTSIZE = 20
            TICKS_FONTSIZE = 15
            NUM_YTICKS = 5
            AUC_INFO = True
        
        # Step 0.2 - Fig content
        PAVPU        = False
        PAVPU_ORGAN  = False
        PU_ORGAN     = False # SPIE paper (after Avinash's reco)

        PAVPU_ORGANS = False # For analysis
        PU_ORGANS    = True  # SPIE/MICCAI/UNSURE Paper

        # Step 0.3 - Fig eval style
        PAVPU_EVAL          = False
        HIERR_LERR_EVAL     = False
        HERR_LERR_SURF_EVAL = True

        # Step 0.4 - Fig unc postprocessing
        UNC_ERODIL = True # [True, False] [NOTE: Why is this ever False? To maintain truthfullness?

        # Step 0.5 - Random vars
        PAVPU_EVAL_STR    = 'AvUPaperEval'
        HILO_EVAL_STR     = 'HiLoEval'
        HILOSURF_EVAL_STR = 'HiLoSurfEval'

        UNC = ''

        DIST_MAX_SURFERR = 3

        verbose = False

        print ('')
        print (' ------------------- ')
        print (' - PU_ORGAN    : ', PU_ORGAN)
        print (' - PU_ORGANS   : ', PU_ORGANS)
        print (' - PAVPU_ORGANS: ', PAVPU_ORGANS)
        print ('')
        print (' - PAVPU_EVAL         : ', PAVPU_EVAL)
        print (' - HIERR_LERR_EVAL    : ', HIERR_LERR_EVAL)
        print (' - HERR_LERR_SURF_EVAL: ', HERR_LERR_SURF_EVAL)
        print (' ------------------- ')
        print (' - UNC_ERODIL: ', UNC_ERODIL)
        print (' - AUC_INFO  : ', AUC_INFO)
        print ('')

        # Step 1 - Choose experiments
        # Journal-HNStructSeg <--THis
        if 0:

            exp_names = [
                            'UNSURE__ONetDetEns-MC1-FixedKL001-FixLR0001__MIC-nooptic-smd-B2-14014040__CEScalar10__seed42'
                            , 'UNSURE__ONetBayes-MC1-FixedKL001-FixLR0001__MIC-B2-14014040__CEScalar10__seed42'
                            , 'UNSURE__ONetBayes-MC1-FixedKL001-FixLR0001__MIC-B2-14014040__CEScalar10-100AvU-MC5Ent-Th005-04-1nAC__seed42'
                            , 'UNSURE__ONetBayesHeadnonHDC-MC1-FixedKL001-FixLR0001__MIC-B2-14014040__CEScalar10__seed42'
                            , 'UNSURE__ONetBayesHead-MC1-FixedKL001-FixLR0001__MIC-B2-14014040__CEScalar10-100AvU-MC5Ent-Th005-04-1nAC__seed42'
                        ]
            plt_names  = ['Ens', 'Bayes', 'Bayes+AvU', 'Bayes(Head)', 'Bayes(Head)+AvU']
            plt_colors = [sns.color_palette('bright')[2], sns.color_palette('bright')[0], sns.color_palette('bright')[4], sns.color_palette('bright')[1], sns.color_palette('bright')[-2]]
            mc_runs    = [None, 30, 30, 30, 30]
            unc_norms  = [True, True, True, True, True]
            epochs     = [1000, 1000, 1000, 1000, 1000]
            
            
            ksize             = (5,5,2) # (5,5,1), (3,3,1) # if only y_true: (15,15,7)
            err_ksize         = (3,3,1)
            # unc_thresholds    = [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,1.0,0.05).tolist() + [0.96, 0.97, 0.98, 0.99, 1.0]
            unc_thresholds    = [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,0.41,0.05).tolist()
            plt_inputinfo = '(140,140,40)'

            optic_bool = False
            smd_bool   = False
            plot_type = PLOT_BARSWARM # ['box', 'swarm', 'violin']

            plt_alias  = 'Journal-HNStructSeg-Optic{}SMD{}-dil{}{}{}-v1-mc30--th{:02d}'.format(optic_bool, smd_bool, ksize[0],ksize[1],ksize[2], int(np.max(unc_thresholds)*100)) 
        
        print ('\n\n - optic_bool: {} \n\n'.format(optic_bool))
        print (' - plot_type : {} \n\n'.format(plot_type))

        print ('\n\n - optic_bool: {} \n\n'.format(optic_bool))
        print (' - plot_type : {} \n\n'.format(plot_type))


        ###############################################################
        #  Step 2 - Choose dataset
        ###############################################################

        # MICCAI 2015 - test_offsite
        if 0: 
            mode           = config.MODE_TEST
            dataset_name   = config.DATASET_MICCAI
            true_str       = config.FILENAME_SAVE_GT_MICCAI2015
            pred_str       = config.FILENAME_SAVE_PRED_MICCAI2015
            unc_str_mif    = config.FILENAME_SAVE_MIF_MICCAI2015
            unc_str_ent    = config.FILENAME_SAVE_ENT_MICCAI2015
            unc_str_std    = config.FILENAME_SAVE_STD_MICCAI2015
            unc_str_stdmax = config.FILENAME_SAVE_STDMAX_MICCAI2015

            unc_strs     = [unc_str_ent]
            # unc_strs     = [unc_str_mif]
            # unc_strs     = [unc_str_ent, unc_str_mif, unc_str_std]
            # unc_strs     = [unc_str_ent, unc_str_std]
            # unc_strs     = [unc_str_std]
            # unc_strs     = [unc_str_ent, unc_str_mif]
            # unc_strs     = [unc_str_stdmax, unc_str_std] 
            # unc_strs     = [unc_str_stdmax] 
            # unc_strs       = [unc_str_ent, unc_str_mif, unc_str_std]
            
            patient_ids  = config.PATIENTIDS_MICCAI2015_TEST
            
            SINGLE_PATIENT = ''
            if 0:
                patient_ids = ['0522c0555', '0522c0659'] # for code testing purposes
                # patient_ids = ['0522c0659']
                # patient_ids = ['0522c0555', '0522c0659']
                # patient_ids = ['0522c0555', '0522c0667']

                print ('')
                print (' -- DOING ONLY A SUBSET OF PATIENT(S) !! --> ', patient_ids)
                print ('')
                
                # SINGLE_PATIENT = patient_ids[0]
            
            ########################################## Step 3 - Some verbosity
            main(mode, unc_strs, exp_names, plt_names, plt_colors, epochs, mc_runs, unc_norms, optic_bool, verbose=verbose)
        
        # MICCAI 2015 - test_onsite
        elif 1: 
            mode           = config.MODE_TEST_ONSITE
            dataset_name   = config.DATASET_MICCAI
            true_str       = config.FILENAME_SAVE_GT_MICCAI2015_TESTONSITE
            pred_str       = config.FILENAME_SAVE_PRED_MICCAI2015_TESTONSITE
            unc_str_mif    = config.FILENAME_SAVE_MIF_MICCAI2015_TESTONSITE
            unc_str_ent    = config.FILENAME_SAVE_ENT_MICCAI2015_TESTONSITE
            unc_str_std    = config.FILENAME_SAVE_STD_MICCAI2015_TESTONSITE
            unc_str_stdmax = config.FILENAME_SAVE_STDMAX_MICCAI2015_TESTONSITE

            unc_strs     = [unc_str_ent]
            # unc_strs     = [unc_str_mif]
            # unc_strs     = [unc_str_ent, unc_str_mif, unc_str_std]
            # unc_strs     = [unc_str_ent, unc_str_std]
            # unc_strs     = [unc_str_std]
            # unc_strs     = [unc_str_ent, unc_str_mif]
            # unc_strs     = [unc_str_stdmax, unc_str_std] 
            # unc_strs     = [unc_str_stdmax] 
            # unc_strs       = [unc_str_ent, unc_str_mif, unc_str_std]
            
            patient_ids  = config.PATIENTIDS_MICCAI2015_TEST_ONSITE
            
            SINGLE_PATIENT = ''
            if 0:
                patient_ids = ['0522c0806', '0522c0845'] # for code testing purposes

                print ('')
                print (' -- DOING ONLY A SUBSET OF PATIENT(S) !! --> ', patient_ids)
                print ('')
                
                # SINGLE_PATIENT = patient_ids[0]
            ########################################## Step 3 - Some verbosity
            main(mode, unc_strs, exp_names, plt_names, plt_colors, epochs, mc_runs, unc_norms, optic_bool, verbose=verbose)

        # DTCIA - Onc
        elif 0: 
            mode           = config.MODE_DEEPMINDTCIA_TEST_ONC      
            dataset_name   = config.DATASET_DEEPMIND
            true_str       = config.FILENAME_SAVE_GT_DEEPMINDTCIA_TEST_ONC
            pred_str       = config.FILENAME_SAVE_PRED_DEEPMINDTCIA_TEST_ONC
            unc_str_mif    = config.FILENAME_SAVE_MIF_DEEPMINDTCIA_TEST_ONC
            unc_str_ent    = config.FILENAME_SAVE_ENT_DEEPMINDTCIA_TEST_ONC
            unc_str_std    = config.FILENAME_SAVE_STD_DEEPMINDTCIA_TEST_ONC
            unc_str_stdmax = config.FILENAME_SAVE_STDMAX_DEEPMINDTCIA_TEST_ONC

            unc_strs     = [unc_str_ent]
            # unc_strs     = [unc_str_mif]
            # unc_strs     = [unc_str_std]
            # unc_strs     = [unc_str_ent, unc_str_std]
            # unc_strs     = [unc_str_ent, unc_str_mif]
            # unc_strs     = [unc_str_stdmax, unc_str_std] 
            # unc_strs     = [unc_str_stdmax] 
            # unc_strs       = [unc_str_ent, unc_str_mif, unc_str_std]

            patient_ids = config.PATIENTIDS_DEEPMINDTCIA_TEST

            SINGLE_PATIENT = ''
            if 0:
                print ('')
                print (' -- DOING ONLY SINGLE PATIENT !')
                patient_ids = ['0522c0017']
                print ('')
                SINGLE_PATIENT = patient_ids[0]

                print ('')
                SINGLE_PATIENT = patient_ids[0]

            ########################################## Step 3 - Some verbosity
            main(mode, unc_strs, exp_names, plt_names, plt_colors, epochs, mc_runs, unc_norms, optic_bool, verbose=verbose)
        
        # MICCAI 2015 - test_offsite + # DTCIA - Onc
        elif 0:

            patient_ids_mic = config.PATIENTIDS_MICCAI2015_TEST # N=10
            mode            = [config.MODE_TEST for _ in patient_ids_mic] 
            true_str        = [config.FILENAME_SAVE_GT_MICCAI2015 for _ in patient_ids_mic]
            pred_str        = [config.FILENAME_SAVE_PRED_MICCAI2015 for _ in patient_ids_mic]
            unc_str_mif     = [config.FILENAME_SAVE_MIF_MICCAI2015 for _ in patient_ids_mic]
            unc_str_ent     = [config.FILENAME_SAVE_ENT_MICCAI2015 for _ in patient_ids_mic]
            unc_str_std     = [config.FILENAME_SAVE_STD_MICCAI2015 for _ in patient_ids_mic]
            unc_str_stdmax  = [config.FILENAME_SAVE_STDMAX_MICCAI2015 for _ in patient_ids_mic]
            unc_strs        = [unc_str_ent[0]]

            patient_ids_dtcia  = config.PATIENTIDS_DEEPMINDTCIA_TEST_RTOG # N=8
            mode               += [config.MODE_DEEPMINDTCIA_TEST_ONC for _ in patient_ids_dtcia]      
            true_str           += [config.FILENAME_SAVE_GT_DEEPMINDTCIA_TEST_ONC for _ in patient_ids_dtcia]
            pred_str           += [config.FILENAME_SAVE_PRED_DEEPMINDTCIA_TEST_ONC for _ in patient_ids_dtcia]
            unc_str_mif        += [config.FILENAME_SAVE_MIF_DEEPMINDTCIA_TEST_ONC for _ in patient_ids_dtcia]
            unc_str_ent        += [config.FILENAME_SAVE_ENT_DEEPMINDTCIA_TEST_ONC for _ in patient_ids_dtcia]
            unc_str_std        += [config.FILENAME_SAVE_STD_DEEPMINDTCIA_TEST_ONC for _ in patient_ids_dtcia]
            unc_str_stdmax     += [config.FILENAME_SAVE_STDMAX_DEEPMINDTCIA_TEST_ONC for _ in patient_ids_dtcia]

            unc_strs           = [unc_str_ent[0]]

            patient_ids        = patient_ids_mic + patient_ids_dtcia

            SINGLE_PATIENT = ''
            if 0:
                
                patient_ids     = patient_ids[:2] + patient_ids[len(patient_ids_mic):len(patient_ids_mic)+2]
                mode            = mode[:2]        + mode[len(patient_ids_mic):len(patient_ids_mic)+2]
                true_str        = true_str[:2]    + true_str[len(patient_ids_mic):len(patient_ids_mic)+2]
                pred_str        = pred_str[:2]    + pred_str[len(patient_ids_mic):len(patient_ids_mic)+2]
                unc_str_mif     = unc_str_mif[:2] + unc_str_mif[len(patient_ids_mic):len(patient_ids_mic)+2]
                unc_str_ent     = unc_str_ent[:2] + unc_str_ent[len(patient_ids_mic):len(patient_ids_mic)+2]
                unc_str_std     = unc_str_std[:2] + unc_str_std[len(patient_ids_mic):len(patient_ids_mic)+2]
                unc_str_stdmax  = unc_str_stdmax[:2] + unc_str_stdmax[len(patient_ids_mic):len(patient_ids_mic)+2]

                plt_alias += '-' + patient_ids[0]

            print (' - mode: ', mode)
            print (' - true_str: ', true_str)

            ########################################## Step 3 - Some verbosity
            main2(mode, unc_strs, exp_names, plt_names, plt_colors, epochs, mc_runs, unc_norms, optic_bool, verbose=verbose)
        
        # StructSeg
        elif 0:

            # Step x - Define keys
            mode           = config.MODE_STRUCTSEG
            patient_ids    = config.PATIENT_IDS_STRUCTSEG
            true_str       = config.FILENAME_SAVE_GT_STRUCTSEG
            pred_str       = config.FILENAME_SAVE_PRED_STRUCTSEG
            unc_str_mif    = config.FILENAME_SAVE_MIF_STRUCTSEG
            unc_str_ent    = config.FILENAME_SAVE_ENT_STRUCTSEG
            unc_str_std    = config.FILENAME_SAVE_STD_STRUCTSEG
            unc_str_stdmax = config.FILENAME_SAVE_STDMAX_STRUCTSEG
            unc_strs       = [unc_str_ent]
            
            # Step y - Define single patient
            SINGLE_PATIENT = ''
            if 0:
                print ('')
                print (' -- DOING CHOSEN PATIENTs !')
                patient_ids = ['1', '2']
                print ('')
                SINGLE_PATIENT = patient_ids[0]

            # Step z - Call function
            main(mode, unc_strs, exp_names, plt_names, plt_colors, epochs, mc_runs, unc_norms, optic_bool, verbose=verbose)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    print ('\n\n ------------------- Total Time: ', round(time.time() - T0, 2), 's')


"""
Add MC runs into filename

To-Run
1. Change FIG_FOR_PAPER value
2. Set PU_ORGANS to True and UNC_ERODIL={True,False}
- In plot_pu_organs(), set the if-else's on the basis of what curve you want
"""