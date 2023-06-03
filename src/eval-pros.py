
# Import internal libraries
import src.config as config
import src.utils as utils

# Import external libraries
import pdb
import nrrd
import time
import scipy
import traceback
import numpy as np
import scipy.stats
import sklearn.metrics
from pathlib import Path
from collections import Counter

import tensorflow as tf
import tensorflow_addons as tfa

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Global Keys
RATIO_KEY         = 'Inacc/Acc Ratio'
PAVPU_EVAL_STR    = 'AvUPaperEval'
HILO_EVAL_STR     = 'HiLoEval'
HILOSURF_EVAL_STR = 'HiLoSurfEval'

# Global Vals
DIST_MAX_SURFERR = 3
EPSILON          = 1e-8

# Plotting Vals
DIR_SAVEFIG = Path('_tmp', 'r-avu')

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
    y_true_class_binary[y_true_class_binary > 0] = 1
    y_true_class_binary = tf.constant(y_true_class_binary, tf.float32)
    y_true_class_binary = tf.constant(tf.expand_dims(tf.expand_dims(y_true_class_binary, axis=-1),axis=0))
    y_true_class_binary_dilated  = tf.nn.max_pool3d(y_true_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')[0,:,:,:,0].numpy()

    y_pred_class_binary = np.array(y_pred_class, copy=True)
    y_pred_class_binary[y_pred_class_binary > 0] = 1
    y_pred_class_binary = tf.constant(y_pred_class_binary, tf.float32)
    y_pred_class_binary = tf.expand_dims(tf.expand_dims(y_pred_class_binary, axis=-1),axis=0)

    y_mask = y_true_class_binary_dilated
    
    return y_mask, y_true_class_binary, y_pred_class_binary

def get_dilated_y_true_pred(y_true_class, y_pred_class, dilation_ksize):
    """
    Calculate mask for areas around GT + AI organs (essentially ignores bgd)

    Thoughts
    --------
    - If the mask is determined by GT + Pred, then how do we do attain homogenity in the evaluation? Should we not just use a really dilated mask 
     and hope that all predictions fall under this?  
    - For ECE, I already did this! So why not for unc-ROC?
    -- Am I doing this for cases where I need to consider the SMD preds in the DTCIA?
    """
    
    # print ('\n ===========================================================================')
    # print (' - [get_dilated_y_true_pred()] Doing this with dilation_ksize: ', dilation_ksize)
    # print (' ===========================================================================\n')

    y_true_class_binary = np.array(y_true_class, copy=True)
    y_true_class_binary[y_true_class_binary > 0] = 1
    y_true_class_binary = tf.constant(y_true_class_binary, tf.float32)
    y_true_class_binary = tf.constant(tf.expand_dims(tf.expand_dims(y_true_class_binary, axis=-1),axis=0))
    y_true_class_binary_dilated  = tf.nn.max_pool3d(y_true_class_binary, ksize=dilation_ksize, strides=1, padding='SAME')
    
    y_pred_class_binary = np.array(y_pred_class, copy=True)
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
    # print (' - before-sum(y_error_areas_high): ', tf.math.reduce_sum(y_error_areas_high))

    # Step 2.1 - Calculate distance maps (in 2D), multiply them by the error mask (y_error_areas). This output is then multiplied by spacing in the xy dimension to give distance values for each erroneous pixel
    ## This is done to handle very specific cases (something to do with parotid gland)
    y_true_distmap_out = tf.expand_dims(tf.transpose(tfa.image.euclidean_dist_transform(tf.cast(tf.transpose(1 - y_true_class_binary[0], perm=(2,0,1,3)), tf.uint8)), perm=(1,2,0,3)), axis=0)  # [1,H,W,D,1] -> [H,W,D,1] -> [D,H,W,1] -> [H,W,D,1] -> [1,H,W,D,1]
    y_true_distmap_in  = tf.expand_dims(tf.transpose(tfa.image.euclidean_dist_transform(tf.cast(tf.transpose(y_true_class_binary[0], perm=(2,0,1,3)), tf.uint8)), perm=(1,2,0,3)), axis=0)      # [1,H,W,D,1] -> [H,W,D,1] -> [D,H,W,1] -> [H,W,D,1] -> [1,H,W,D,1]
    y_true_distmap     = y_true_distmap_out + y_true_distmap_in                                                                                                                                 # [1,H,W,D,1]
    y_true_distmap_errorarea = y_error_areas * y_true_distmap * spacing[0]

    # Step 2.2 - Apply logical OR operation and get final HIGH erroneous pixels
    y_true_distmap_errorarea_binary = tf.math.greater_equal(y_true_distmap_errorarea, distance_max)
    y_error_areas_high              = tf.cast(tf.math.logical_or(tf.cast(y_error_areas_high, tf.bool), y_true_distmap_errorarea_binary), dtype=tf.float32)[0,:,:,:,0].numpy()
    # print (' - after-sum(y_error_areas_high) : ', tf.math.reduce_sum(y_error_areas_high))

    # Step 3 - Anything not high error is now considered low error
    y_nonerror_areas           = y_mask - y_error_areas_high  # Note: If there are -1's in this, that means that y_mask is not sufficient enough to cover the predictions
    y_error_areas_low          = y_nonerror_areas  # No error + Low Error
    # print (' - Counter(y_error_areas_low): ', Counter(y_error_areas_low.flatten()))
    y_error_areas_low[y_error_areas_low < 0] = 0 # set all non-captured errors as background and hence they will be considered as accurate. Note: Not the best solution 

    y_accurate    = (y_true_class == y_pred_class).astype(np.uint8)  # [H,W,D]
    y_inaccurate  = 1 - y_accurate
    y_accurate    = y_accurate * y_mask
    y_inaccurate  = y_inaccurate * y_mask
    # print (' - RT-Specific Mask                 : Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))

    y_accurate   = y_error_areas_low
    y_inaccurate = y_error_areas_high
    # print (' - RT-Specific Mask (with ~a + surf): Acc: ', np.sum(y_accurate), ' || Inacc: ', np.sum(y_inaccurate))

    return y_accurate, y_inaccurate

############################################ p(u) - HiErr vs LoErr ###########################################

# Paper function
def get_pu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, eval_str, dilation_ksize=(3,3,1), error_ksize=(3,3,1), verbose=False):

    """
    Goal: Get uncertainty probability values for each threshold in high and low error areas (as defined by dilation and error kernels) for a particular patient

    Params
    ------
    y_true         : [H,W,D]
    y_pred         : [H,W,D]
    y_unc          : [H,W,D]
    unc_thresholds : list; float
    mode           : str
    """

    res = {config.KEY_P_UI: {}, config.KEY_P_UA:{}, config.KEY_P_IU: {}, config.KEY_P_AC: {}, config.KEY_AVU: {}, RATIO_KEY: {}}

    try:

        # Step 0 - Init
        H,W,D                  = y_true.shape
        y_true_class           = y_true
        y_pred_class           = y_pred
        
        # Step 1 - Calculate mask for areas around organs only
        y_mask, y_true_class_binary, y_pred_class_binary = get_dilated_y_true_pred(y_true_class, y_pred_class, dilation_ksize)

        # Step 2 - Get accurate (Lo-Err) and inaccurate (Hi-Err) areas 
        if eval_str == PAVPU_EVAL_STR:
            y_accurate, y_inaccurate = get_accurate_inaccurate_avupaper(y_true_class, y_pred_class, y_mask)
        
        elif eval_str == HILO_EVAL_STR:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize)    
        
        elif eval_str == HILOSURF_EVAL_STR:
            y_accurate, y_inaccurate = get_accurate_inaccurate_via_erosion_and_surface(y_mask, y_true_class, y_pred_class, y_true_class_binary, y_pred_class_binary, error_ksize, spacing, distance_max=DIST_MAX_SURFERR)
        

        # Step 2.1 - Debug the above step
        if 0:
            slice_id = 20;
            f,axarr = plt.subplots(2,2);
            plt.suptitle('Eval: {} | ksize_dil: {} | ksize_err: {}'.format(eval_str, dilation_ksize, error_ksize))
            axarr[0][0].imshow(y_true_class[:,:,slice_id], cmap='gray', interpolation=None);axarr[0][0].set_title('GT');
            axarr[0][0].imshow(y_mask[:,:,slice_id], cmap='gray', interpolation=None, alpha=0.3);
            axarr[0][1].imshow(y_pred_class[:,:,slice_id], cmap='gray', interpolation=None);axarr[0][1].set_title('Pred');
            axarr[0][1].imshow(y_mask[:,:,slice_id], cmap='gray', interpolation=None, alpha=0.3);
            axarr[1][0].imshow(y_accurate[:,:,slice_id], cmap='gray', interpolation=None);axarr[1][0].set_title('Accurate');
            axarr[1][1].imshow(y_inaccurate[:,:,slice_id], cmap='gray', interpolation=None);axarr[1][1].set_title('InAccurate');
            plt.savefig('pros.png')
            pdb.set_trace()

        # Step 3 - Compute n-vals using accuracy and uncertainty for each voxel
        print (' - [get_pu_organs()] np.max(y_unc): ', np.max(y_unc))
        for unc_threshold in unc_thresholds:

            # Step 3.1 - Find (un)certain mask 
            y_uncertain      = np.array(y_unc, copy=True)
            y_uncertain[y_uncertain <=  unc_threshold] = 0
            y_uncertain[y_uncertain > unc_threshold] = 1
            y_certain = 1 - y_uncertain

            # Step 3.2 - Find n-vals
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
            
            # Step 3.3.2 - Handling error cases
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
            
            # Step 3.3.3 - Adding to final results object
            res[config.KEY_P_UA][unc_threshold]  = p_ua 
            res[config.KEY_P_UI][unc_threshold]  = p_ui 
            res[config.KEY_P_IU][unc_threshold]  = p_iu 
            res[config.KEY_P_AC][unc_threshold]  = p_ac
            res[config.KEY_AVU][unc_threshold]   = avu
            if 0:
                print (' - [get_pu_organs()][thres={:.5f}] p(u|a): {:.5f}, p(u|i): {:.5f}, p(i|u): {:.5f}, p(a|c): {:.5f}, '.format(unc_threshold, p_ua, p_ui, p_iu, p_ac))
            
            # Step 3.3.4 - Some debugging vals
            try:
                res[RATIO_KEY][unc_threshold] = (n_ic + n_iu) / (n_ac + n_au)
            except:
                res[RATIO_KEY][unc_threshold] = 0
                
        return res

    except:
        traceback.print_exc()
        pdb.set_trace()
        return {'high-error': {}, 'low-error': {}}

def plot_pu_organs(res, plt_colors, plt_title_str, savefig_str, params_fig, plt_alias, plot_type, verbose=True):
    """
    res: Dict e.g. {exp_id: { patient_id: {'high-error': {}, 'low-error': {}} }}
    """

    try:
        
        # Step 0 - Init
        pass

        # Trapezoidal Method (slightly modified)
        def integral(y, x):
            dx = x[:-1] - x[1:]
            dy = (y[:-1] + y[1:])/2
            return tf.math.reduce_sum(dx*dy)

        xaxis_lim = [-0.01, 1.01] # [0,1]
        yaxis_lim = [-0.05, 1.05] # [0,1]
        
        # Step 0.1 - Ratio of inacc/acc
        if 0:
            
            # Step 1.1 - Set plt params
            sns.set_style('darkgrid')
            plt.figure(figsize=(11,5), dpi=200)
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
            plt.savefig(path_savefig, bbox_inches='tight')
            plt.close()
            print ('')
            print (' - Saved as ', path_savefig)

        roundtopoint5 = lambda x: round(x * 20) / 20

        # Step 1.1 - ROC curves (new)
        PATIENT_WISE = False
        if 1:
            
            print ('\n\n\n ================== unc-ROC curves ================== \n')
            try:

                # Step 1.1 - Set plt params
                if 1:
                    sns.set_style('darkgrid')
                    plt.figure(figsize=params_fig['figsize'], dpi=params_fig['dpi'])
                    plt.title(plt_title_str)
                    plt.gca().xaxis.label.set_size(params_fig['label_fontsize'])
                    plt.gca().yaxis.label.set_size(params_fig['label_fontsize'])
                    plt.xticks(fontsize=params_fig['ticks_fontsize'])
                    plt.yticks(fontsize=params_fig['ticks_fontsize'])

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
                    if params_fig['auc_info']:
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
                        print (' - ROC: ', plt.gca().get_yticks())
                        yticks_min = next((i for i in plt.gca().get_yticks() if i > 0), None)
                        yticks_max = plt.gca().get_yticks()[-1] if plt.gca().get_yticks()[-1] < 1.0 else 1.0
                        yticks = np.linspace(roundtopoint5(yticks_min), roundtopoint5(yticks_max), params_fig['num_yticks'])
                        yticklabels = ['{:.2f}'.format(ytick) for ytick in yticks]
                        plt.gca().set_yticks(yticks)
                        plt.gca().set_yticklabels(yticklabels)
                    plt.legend(fontsize=params_fig['legend_fontsize'])
                    path_savefig = savefig_str.format('ROCurve')
                    plt.savefig(path_savefig, bbox_inches='tight')
                    plt.close()
                    print ('\n - Saved as ', path_savefig)
            
            except:
                traceback.print_exc()
                pdb.set_trace()
        
        ## Step 1.2 - ROC swarm (old)
        PATIENT_WISE = False
        if 1:
            
            try:

                print ('\n ================== unc-ROC BoxPlots ({}) ================== \n'.format(plt_alias))

                # Step 1.3.1 - Set plt params
                sns.set_style('darkgrid')
                plt.figure(figsize=params_fig['figsize'], dpi=params_fig['dpi'])
                plt.title(plt_title_str)
                plt.gca().xaxis.label.set_size(params_fig['label_fontsize'])
                plt.gca().yaxis.label.set_size(params_fig['label_fontsize'])
                plt.xticks(fontsize=params_fig['ticks_fontsize'])
                plt.yticks(fontsize=params_fig['ticks_fontsize'])

                # Step 1.3.2 - Setup keys and loop over experiments
                x_key = 'mean-p(u|a,~a)' 
                y_key = 'mean-p(u|i)'
                auc_vals_patientwise       = {}
                auc_vals_tflow_patientwise = {}

                PATIENT_WISE = True

                boxplot_exp_names = []
                boxplot_roc_aucs  = []
                palette           = {}
                data_hmap_roc     = {'Model': []}
                

                # Step 1.3.3 - Loop over experiments and patients and accumulate AUC values
                for exp_id, exp_name in enumerate(res):

                    palette[exp_name] = plt_colors[exp_id]

                    auc_vals_patientwise[exp_name] = []
                    auc_vals_tflow_patientwise[exp_name] = []

                    data_hmap_roc['Model'].append(exp_name)
                    if exp_id == 0:
                        for patient_id in res[exp_name]:
                            data_hmap_roc[patient_id] = []

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
                        data_hmap_roc[patient_id].append(auc_patient)

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
                    
                    if plot_type == PLOT_BARSWARM:
                        sns.boxplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys(),  boxprops=dict(alpha=.1))
                        boxplt  = sns.swarmplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                    else:
                        if plot_type == PLOT_BAR    : boxplt  = sns.boxplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                        elif plot_type == PLOT_SWARM: boxplt  = sns.swarmplot(data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                        

                    boxplt.set_xlabel('',fontsize=0)
                    boxplt.set_ylabel('ROC-AUC')

                    if params_fig['auc_info']:
                        labels_new = []
                        for exp_name in palette.keys(): 
                            labels_new.append(exp_name + '\n({:.4f} ± {:.4f})'.format(np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                        boxplt.set_xticks(ticks=boxplt.get_xticks(), labels=labels_new)

                # Step 1.3.5 - Stats
                if 1:
                    stats_roc = np.full((len(res), len(res)), 99.0)
                    exp_names = list(res.keys())
                    
                    anno_pairs = []
                    anno_pvalues = []
                    for each in itertools.combinations(res.keys(),2):
                        result = scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]])
                        stats_roc[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        if verbose: print (' - Exp: {} vs {} = {}'.format(each[0], each[1], result))

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

                    if 0:
                        # https://github.com/trevismd/statannotations-tutorials/blob/main/Tutorial_1/Statannotations-Tutorial-1.ipynb
                        from statannotations.Annotator import Annotator
                        annotator = Annotator(boxplt, anno_pairs, data=data_roc, x='Model', y='ROC-AUC', palette=palette, order=palette.keys())
                        annotator.set_custom_annotations(anno_pvalues)
                        annotator.annotate()                
                
                # Step 1.3.6 - Verbosity
                print ('\n ================== unc-ROC BoxPlots ({}) ================== \n'.format(plt_alias))
                for exp_id, exp_name in enumerate(res):
                    # print (' - [ROC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_patientwise[exp_name]), np.std(auc_vals_patientwise[exp_name])))
                    print (' - [ROC-Patient] Exp: {} | AUC-TenFlow: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                print ('\n ================== unc-ROC BoxPlots ({}) ================== \n'.format(plt_alias))
                
                # Step 1.3.7 - Finally! Save figure
                if 0:
                    yticks_min = next((i for i in plt.gca().get_yticks() if i > 0), None)
                    yticks_max = plt.gca().get_yticks()[-1] if plt.gca().get_yticks()[-1] < 1.0 else 1.0
                    yticks = np.linspace(roundtopoint5(yticks_min), roundtopoint5(yticks_max), params_fig['num_yticks'])
                    yticklabels = ['{:.2f}'.format(ytick) for ytick in yticks]
                    plt.gca().set_yticks(yticks)
                    plt.gca().set_yticklabels(yticklabels)
                path_savefig = savefig_str.format('ROCBarPlot')
                plt.savefig(path_savefig, bbox_inches='tight')
                plt.close()
                print ('\n - Saved as ', path_savefig)
                
                # Step 1.3.6 - HMaps
                if 1:
                    try:
                        df_hmap_roc = pd.DataFrame(data_hmap_roc)
                        df_hmap_roc = df_hmap_roc.set_index('Model')

                        plt.figure(figsize=(20,10), dpi=500)
                        sns.heatmap(df_hmap_roc, annot=True, fmt=".1f", cmap='Oranges')
                        plt.title(plt_title_str)

                        path_savefig = savefig_str.format('ROCHMap')
                        plt.savefig(path_savefig, bbox_inches='tight')
                        plt.close()
                    except:
                        traceback.print_exc()

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
                    plt.figure(figsize=params_fig['figsize'], dpi=params_fig['dpi'])
                    plt.title(plt_title_str)
                    plt.gca().xaxis.label.set_size(params_fig['label_fontsize'])
                    plt.gca().yaxis.label.set_size(params_fig['label_fontsize'])
                    plt.xticks(fontsize=params_fig['ticks_fontsize'])
                    plt.yticks(fontsize=params_fig['ticks_fontsize'])

                # Step 2.2 - Setup key and loop over experiments
                if 1:
                    x_key = config.FIGURE_ENT
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
                    if params_fig['auc_info']:
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
                    if 0:
                        yticks_min = next((i for i in plt.gca().get_yticks() if i > 0), None)
                        yticks_max = plt.gca().get_yticks()[-1] if plt.gca().get_yticks()[-1] < 1.0 else 1.0
                        yticks = np.linspace(roundtopoint5(yticks_min), roundtopoint5(yticks_max), params_fig['num_yticks'])
                        yticklabels = ['{:.2f}'.format(ytick) for ytick in yticks]
                        plt.gca().set_yticks(yticks)
                        plt.gca().set_yticklabels(yticklabels)
                    plt.legend(fontsize=params_fig['legend_fontsize'])
                    path_savefig = savefig_str.format('AvUCurve')
                    plt.savefig(path_savefig, bbox_inches='tight')
                    plt.close()
                    print ('\n - Saved as ', path_savefig)

            except:
                traceback.print_exc()
                pdb.set_trace()
        
        # Step 2.2 - AvU swarm (old)
        PATIENT_WISE = False
        if 1:
        
            try:

                auc_vals_patientwise       = {}
                auc_vals_tflow_patientwise = {}
                print ('\n ================== AvU boxplots ({}) ================== \n'.format(plt_alias))

                # Step 3.1.1 - Set plt params
                sns.set_style('darkgrid')
                plt.figure(figsize=params_fig['figsize'], dpi=params_fig['dpi'])
                plt.title(plt_title_str)
                plt.gca().xaxis.label.set_size(params_fig['label_fontsize'])
                plt.gca().yaxis.label.set_size(params_fig['label_fontsize'])
                plt.xticks(fontsize=params_fig['ticks_fontsize'])
                plt.yticks(fontsize=params_fig['ticks_fontsize'])
                # plt.ylim([0.7, 1.0])
                

                # Step 3.1.2 - Setup keys and loop over experiments
                x_key = config.FIGURE_ENT
                # if config.KEYNAME_ENT in unc_str:
                #     x_key = config.FIGURE_ENT
                # elif config.KEYNAME_MIF in unc_str:
                #     x_key = config.FIGURE_MI
                # elif config.KEYNAME_STD in unc_str:
                #     x_key = config.FIGURE_STD
                
                y_key = config.KEY_AVU 

                boxplot_exp_names = []
                boxplot_aucs      = []
                palette           = {}

                data_hmap_avu     = {'Model': []}

                # Step 3.1.3 - Loop over experiments and patients and accumulate AUC values
                for exp_id, exp_name in enumerate(res):

                    palette[exp_name] = plt_colors[exp_id]
                    
                    auc_vals_patientwise[exp_name] = []
                    auc_vals_tflow_patientwise[exp_name] = []

                    data_hmap_avu['Model'].append(exp_name)
                    if exp_id == 0:
                        for patient_id in res[exp_name]:
                            data_hmap_avu[patient_id] = []

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

                            data_hmap_avu[patient_id].append(auc_patient)

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
                    data_avu            = pd.DataFrame({'Model': boxplot_exp_names, 'AvU-AUC':boxplot_aucs})
                    
                    if plot_type == PLOT_BARSWARM:
                        sns.boxplot(data=data_avu, x='Model', y='AvU-AUC', palette=palette, order=palette.keys(),  boxprops=dict(alpha=.1))
                        boxplt  = sns.swarmplot(data=data_avu, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                    else:
                        if plot_type == PLOT_BAR    : boxplt = sns.boxplot(data=data_avu, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                        elif plot_type == PLOT_SWARM: boxplt = sns.swarmplot(data=data_avu, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                        
                    boxplt.set_xlabel('',fontsize=0)
                    boxplt.set_ylabel('AvU-AUC')

                    if params_fig['auc_info']:
                        labels_new = []
                        for exp_name in palette.keys(): labels_new.append(exp_name + '\n({:.4f} ± {:.4f})'.format(np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                        boxplt.set_xticks(ticks=boxplt.get_xticks(), labels=labels_new)

                # Step 3.1.5 - Stats
                if 1:
                    try:
                        stats_avu = np.full((len(res), len(res)), 99.0)
                        exp_names = list(res.keys())
                        
                        anno_pairs = []
                        anno_pvalues = []
                        for each in itertools.combinations(res.keys(),2):
                            result = scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]])
                            stats_avu[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                            print (' - Exp: {} vs {} = {}'.format(each[0], each[1], result))

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
                            annotator = Annotator(boxplt, anno_pairs, data=data_avu, x='Model', y='AvU-AUC', palette=palette, order=palette.keys())
                            annotator.set_custom_annotations(anno_pvalues)
                            annotator.annotate() 

                    except:
                        traceback.print_exc()            
                
                # Step 1.3.6 - Verbosity
                print ('\n ================== AvU boxplots ({}) ================== \n'.format(plt_alias))
                for exp_id, exp_name in enumerate(res):
                    # print (' - [ROC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_patientwise[exp_name]), np.std(auc_vals_patientwise[exp_name])))
                    print (' - [ROC-Patient] Exp: {} | AUC-TenFlow: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                print ('\n ================== AvU boxplots ({}) ================== \n'.format(plt_alias))

                # Step 1.3.7 - Save figure
                if 0:
                    yticks_min = next((i for i in plt.gca().get_yticks() if i > 0), None)
                    yticks_max = plt.gca().get_yticks()[-1] if plt.gca().get_yticks()[-1] < 1.0 else 1.0
                    yticks = np.linspace(roundtopoint5(yticks_min), roundtopoint5(yticks_max), params_fig['num_yticks'])
                    yticklabels = ['{:.2f}'.format(ytick) for ytick in yticks]
                    plt.gca().set_yticks(yticks)
                    plt.gca().set_yticklabels(yticklabels)
                path_savefig = savefig_str.format('AvUBarPlot')
                plt.savefig(path_savefig, bbox_inches='tight')
                plt.close()
                print ('\n - Saved as ', path_savefig)

                # Step 1.3.6 - HMaps
                if 1:
                    try:
                        df_hmap_avu = pd.DataFrame(data_hmap_avu)
                        df_hmap_avu = df_hmap_avu.set_index('Model')
                        
                        plt.figure(figsize=(20,10), dpi=500)
                        sns.heatmap(df_hmap_avu, annot=True, fmt=".1f", cmap='Oranges')
                        plt.title(plt_title_str)

                        path_savefig = savefig_str.format('AvUHMap')
                        plt.savefig(path_savefig, bbox_inches='tight')
                        plt.close()
                    except:
                        traceback.print_exc()

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
                    plt.figure(figsize=params_fig['figsize'], dpi=params_fig['dpi'])
                    plt.title(plt_title_str)
                    plt.gca().xaxis.label.set_size(params_fig['label_fontsize'])
                    plt.gca().yaxis.label.set_size(params_fig['label_fontsize'])
                    plt.xticks(fontsize=params_fig['ticks_fontsize'])
                    plt.yticks(fontsize=params_fig['ticks_fontsize'])

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
                    if params_fig['auc_info']:
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
                    if 0:
                        yticks_min = next((i for i in plt.gca().get_yticks() if i > 0), None)
                        yticks_max = plt.gca().get_yticks()[-1] if plt.gca().get_yticks()[-1] < 1.0 else 1.0
                        yticks = np.linspace(roundtopoint5(yticks_min), roundtopoint5(yticks_max), params_fig['num_yticks'])
                        yticklabels = ['{:.2f}'.format(ytick) for ytick in yticks]
                        plt.gca().set_yticks(yticks)
                        plt.gca().set_yticklabels(yticklabels)
                    plt.legend(fontsize=params_fig['legend_fontsize'])
                    path_savefig = savefig_str.format('PRCurve')
                    plt.savefig(path_savefig, bbox_inches='tight')
                    plt.close()
                    print ('\n - Saved as ', path_savefig)
            
            except:
                traceback.print_exc()
                pdb.set_trace()

        # Step 3.2 -  Precision-Recall swarm (old)
        PATIENT_WISE = False
        if 1:
            try:
                
                print ('\n ================== PRC boxplots ({}) ================== \n'.format(plt_alias))
                auc_vals_patientwise       = {}
                auc_vals_tflow_patientwise = {}

                # Step 4.1.1 - Set plt params
                sns.set_style('darkgrid')
                plt.figure(figsize=params_fig['figsize'], dpi=params_fig['dpi'])
                plt.title(plt_title_str)
                plt.gca().xaxis.label.set_size(params_fig['label_fontsize'])
                plt.gca().yaxis.label.set_size(params_fig['label_fontsize'])
                plt.xticks(fontsize=params_fig['ticks_fontsize'])
                plt.yticks(fontsize=params_fig['ticks_fontsize'])
                # plt.ylim([0.0, 1.0])
                
                # Step 4.1.2 - Setup key and loop over experiments
                x_key = 'mean Recall - p(u|i)'
                y_key = 'mean Precision - p(i|u)'

                boxplot_exp_names = []
                boxplot_prc_aucs  = []
                palette           = {}

                data_hmap_prc     = {'Model': []}

                # Step 4.1.3 - Loop over experiments and patients and accumulate AUC values
                for exp_id, exp_name in enumerate(res):

                    palette[exp_name] = plt_colors[exp_id]
                    
                    auc_vals_patientwise[exp_name] = []
                    auc_vals_tflow_patientwise[exp_name] = []

                    data_hmap_prc['Model'].append(exp_name)
                    if exp_id == 0:
                        for patient_id in res[exp_name]:
                            data_hmap_prc[patient_id] = []

                    for patient_id_num, patient_id in enumerate(res[exp_name]):
                        try:
                            data_pat = {
                                x_key  : list(res[exp_name][patient_id][config.KEY_P_UI].values())
                                , y_key: list(res[exp_name][patient_id][config.KEY_P_IU].values())
                            }
                            auc_x_patient     = tf.constant(data_pat[x_key])
                            auc_y_patient     = tf.constant(data_pat[y_key])
                            # auc_patient       = sklearn.metrics.auc(auc_x_patient, auc_y_patient)
                            auc_patient_tflow = integral(auc_y_patient, auc_x_patient)
                            
                            # auc_vals_patientwise[exp_name].append(auc_patient)
                            auc_vals_tflow_patientwise[exp_name].append(auc_patient_tflow)

                            data_hmap_prc[patient_id].append(float(auc_patient_tflow))

                            boxplot_exp_names.append(exp_name)
                            boxplot_prc_aucs.append(float(auc_patient_tflow))

                        except:
                            print (' - [ERROR][plot_pu_organs()][PR curves] {}/{}) patient_id: {}'.format(patient_id_num, len(res[exp_name]), patient_id) )
                            traceback.print_exc()
                            pdb.set_trace()

                # import pprint
                # pprint.pprint(auc_vals_tflow_patientwise)
                
                # Step 4.1.4 - Plot 
                if 1:
                    import pandas as pd
                    import itertools
                    data_prc  = pd.DataFrame({'Model': boxplot_exp_names, 'PRC-AUC':boxplot_prc_aucs})
                    if plot_type == PLOT_BARSWARM:
                        boxplt = sns.boxplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys(),  boxprops=dict(alpha=.1))
                        boxplt = sns.swarmplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())
                    else:
                        if plot_type == PLOT_BAR    : boxplt = sns.boxplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())
                        elif plot_type == PLOT_SWARM: boxplt = sns.swarmplot(data=data_prc, x='Model', y='PRC-AUC', palette=palette, order=palette.keys())
                        

                    boxplt.set_xlabel('',fontsize=0)
                    boxplt.set_ylabel('PRC-AUC')

                    if params_fig['auc_info']:
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
                        result = scipy.stats.wilcoxon(auc_vals_tflow_patientwise[each[0]], auc_vals_tflow_patientwise[each[1]])
                        stats_prc[exp_names.index(each[0]), exp_names.index(each[1])] = result.pvalue
                        print (' - Exp: {} vs {} = {}'.format(each[0], each[1], result))

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
                print ('\n ================== PRC boxplots ({}) ================== \n'.format(plt_alias))
                for exp_id, exp_name in enumerate(res):
                    # print (' - [ROC-Patient] Exp: {} | AUC-Sklearn: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_patientwise[exp_name]), np.std(auc_vals_patientwise[exp_name])))
                    print (' - [ROC-Patient] Exp: {} | AUC-TenFlow: {:.4f} ± {:.4f}'.format(exp_name, np.mean(auc_vals_tflow_patientwise[exp_name]), np.std(auc_vals_tflow_patientwise[exp_name])))
                print ('\n ================== PRC boxplots ({}) ================== \n'.format(plt_alias))
                # import pprint
                # pprint.pprint(auc_vals_tflow_patientwise)

                # Step 1.3.7 - Save figure
                if 1:
                    if 0:
                        yticks_min = next((i for i in plt.gca().get_yticks() if i > 0), None)
                        yticks_max = plt.gca().get_yticks()[-1] if plt.gca().get_yticks()[-1] < 1.0 else 1.0
                        yticks = np.linspace(roundtopoint5(yticks_min), roundtopoint5(yticks_max), params_fig['num_yticks'])
                        yticklabels = ['{:.2f}'.format(ytick) for ytick in yticks]
                        plt.gca().set_yticks(yticks)
                        plt.gca().set_yticklabels(yticklabels)
                    path_savefig = savefig_str.format('PRCBarPlot')
                    plt.savefig(path_savefig, bbox_inches='tight')
                    plt.close()
                    print ('\n - Saved as ', path_savefig)

                # Step 1.3.6 - HMaps
                if 1:
                    try:
                        df_hmap_prc = pd.DataFrame(data_hmap_prc)
                        df_hmap_prc = df_hmap_prc.set_index('Model')
                        
                        plt.figure(figsize=(25,15), dpi=500)
                        sns.heatmap(df_hmap_prc, annot=True, fmt=".2f", cmap='Oranges')
                        plt.title(plt_title_str)

                        path_savefig = savefig_str.format('PRCHMap')
                        plt.savefig(path_savefig, bbox_inches='tight')
                        plt.close()
                    except:
                        traceback.print_exc()
                
                print (sorted(data_hmap_prc))

            except:
                traceback.print_exc()
                pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()

####################################################################################################

def main(params_global, params_exp, params_dataset):
    
    try:
        
        # Step 0.1 - Params - Global
        verbose = params_global['verbose']
        eval_str = params_global['eval_str']

        for_paper = params_global['figure']['for_paper']

        # Step 0.2 - Params - Experiment
        print ('\n ---------------- PARAMS - EXPERIMENT ----------------')
        plt_alias  = params_exp['plt_alias']
        plt_names  = params_exp['plt_names']
        plt_colors = params_exp['plt_colors']

        exp_names  = params_exp['exp_names']
        exp_epochs = params_exp['exp_epochs']
        exp_mcruns = params_exp['exp_mcruns']
        exp_trainingbool = params_exp['exp_trainingbool']
        
        unc_norms      = params_exp['unc_norms']
        unc_thresholds = params_exp['unc_thresholds']
        unc_erodedil   = params_exp['unc_erodedil']

        ksize_dil = params_exp['ksize_dil']
        ksize_err = params_exp['ksize_err']

        for exp_id, exp_name in enumerate(exp_names):
            print (' - [{:15s}][epoch:{:03d}, mc={}] {} '.format(plt_names[exp_id], exp_epochs[exp_id], exp_mcruns[exp_id], exp_names[exp_id]))

        # Step 0.3 - Params - Dataset
        print ('\n ---------------- PARAMS - DATASET ----------------')
        mode        = params_dataset['mode']
        patient_ids    = params_dataset['patient_ids']
        patient_single = params_dataset['patient_single']

        unc_strs    = params_dataset['unc_strs']
        true_str    = params_dataset['true_str']
        pred_str    = params_dataset['pred_str']
        unc_str_ent = params_dataset['unc_str_ent']

        plot_type   = params_exp['plot_type'] 

        if len(patient_single):
            print ('')
            print (' -- DOING ONLY A SUBSET OF PATIENT(S) !! --> ', params_dataset['patient_ids'])
            print ('')
    
        if len(exp_names) == len(plt_names) == len(plt_colors) == len(exp_epochs) == len(exp_mcruns) == len(unc_norms):

            # Step 1 - Loop over unc types
            for unc_str in unc_strs:
                
                print ('\n ----------------------------------------------------------------- ')
                print (' ----------------------------------------------------------------- ')
                print (' - mode     : ', mode)
                print (' - ksize_dil: ', ksize_dil)
                print (' - ksize_err: ', ksize_err)
                print (' - unc_str  : ', unc_str)
                print (' - plt_alias: ', plt_alias)
                print (' ----------------------------------------------------------------- ')
                print (' ----------------------------------------------------------------- \n')
                plt_unc_str = config.KEY_ENT
                # if config.KEY_ENT in unc_strs:
                #     plt_unc_str = config.KEY_ENT
                # if config.KEY_MI in unc_strs:
                #     plt_unc_str = config.KEY_MIF
                # if config.KEY_STD in unc_strs:
                #     plt_unc_str = config.KEY_STD
                
                # Step 2 - Loop over patients 
                res_unc = {} # collects data for a particular uncertainty
                for patient_count, patient_id in enumerate(patient_ids):
                    print ('\n\n ------------------')
                    print (' ------------------ {}/{} patient_id: {}'.format(patient_count+1, len(patient_ids), patient_id))
                    y_true_str       = true_str.format(patient_id)      # [H,W,D]
                    y_pred_str       = pred_str.format(patient_id)      # [H,W,D]
                    y_unc_str_ent    = unc_str_ent.format(patient_id)   # [H,W,D]
                    
                    # Step 3 - Loop over experiments
                    for exp_id, exp_name in enumerate(exp_names):
                        epoch    = exp_epochs[exp_id]
                        mc_runs  = exp_mcruns[exp_id]
                        unc_norm = unc_norms[exp_id]
                        training_bool = exp_trainingbool[exp_id]

                        # Step 3.1 - Get directory containing inferenced volumes
                        patch_dir, _ = utils.get_eval_folders(config.PROJECT_DIR, exp_name, exp_epochs[exp_id], mode, mc_runs=exp_mcruns[exp_id], training_bool=exp_trainingbool[exp_id])
                        # print (' - patch_dir: ', patch_dir)

                        # Step 3.2 - Read ground-truth and pred
                        y_true, _ = nrrd.read(str(Path(patch_dir).joinpath(y_true_str)))
                        y_pred, _ = nrrd.read(str(Path(patch_dir).joinpath(y_pred_str)))

                        # Step 3.3.1 - Read entropy
                        if config.KEYNAME_ENT in unc_str:
                            y_unc_ent, _  = nrrd.read(str(Path(patch_dir).joinpath(y_unc_str_ent)))
                            y_unc_ent[y_unc_ent < 0] = EPSILON # set all negative values to 1e-8
                            if unc_norms[exp_id]:
                                y_unc = y_unc_ent / np.log(2) # specific to Prostate MR training dataset due to C=1 (or 2) classes
                            else:
                                y_unc = y_unc_ent
                        
                            if unc_erodedil[exp_id]:
                                y_unc = do_erosion_dilation(y_unc)
                        
                        # Step 3.4 - Read spacing from the header
                        spacing   = _['space directions']
                        spacing   = tuple(spacing[spacing > 0])

                        # Step 3.5 - Get probability values pertaining to accuracy and uncertainty
                        patient_vals = get_pu_organs(y_true, y_pred, y_unc, unc_thresholds, spacing, eval_str, dilation_ksize=ksize_dil, error_ksize=ksize_err, verbose=verbose)
                        # print ('\n - [{}][{}] patient_vals: {}'.format(plt_names[exp_id], patient_id, patient_vals))
                        
                        if plt_names[exp_id] in res_unc:
                            res_unc[plt_names[exp_id]][patient_id] = patient_vals
                        else:
                            res_unc[plt_names[exp_id]] = {patient_id: patient_vals}

                # Step 4 - Plot
                if len(np.unique(exp_epochs)) > 1:
                    epoch_str = '-'.join([str(epoch_) for epoch_ in exp_epochs])
                else:
                    epoch_str = exp_epochs[0]
                if not for_paper:
                    title_str = '{} || Mode={} || Eval={} || Epoch={} || Unc={} || KSize={}'.format(plt_alias, mode, eval_str, epoch_str, plt_unc_str, ksize_dil)
                else:
                    title_str = ''

                savefig_str = str(Path(DIR_SAVEFIG).joinpath(plt_alias + '__{}-' + '-'.join([mode, plt_unc_str, eval_str, 'ep{}'.format(epoch_str)
                                                                                                , 'dil{}'.format(''.join([str(each) for each in ksize_dil])) 
                                                                                                , plot_type
                                                                                                , 'maxth{:02d}'.format(int(unc_thresholds[-1]*100))
                                                                                             ]
                                                                                            ))
                                                                                        )
                
                if len(patient_single):
                    title_str += '\n' + patient_single

                params_fig = {
                    'figsize': (11,5)
                    , 'dpi': params_global['figure']['dpi']
                    , 'legend_fontsize': params_global['figure']['legend_fontsize']
                    , 'label_fontsize' : params_global['figure']['label_fontsize']
                    , 'ticks_fontsize' : params_global['figure']['ticks_fontsize']
                    , 'auc_info'       : params_global['figure']['AUC_INFO']
                    , 'for_paper'      : params_global['figure']['for_paper']
                    , 'num_yticks'     : params_global['figure']['num_yticks'] 
                }
                
                plot_pu_organs(res_unc, plt_colors, title_str, savefig_str, params_fig, plt_alias, plot_type, verbose=verbose)
                
        else:
            print (' - [Error] Check your exp_names, plt_names and plt_colors list')

    except:
        traceback.print_exc()
        pdb.set_trace()

####################################################################################################

if __name__ == "__main__":

    T0 = time.time()

    try:
        
        ###############################################################
        #  Step 0 - Choose content
        ###############################################################
        params_global = {
            'figure': {
                'for_paper': True
                , 'AUC_INFO': False
            }
            , 'eval_str': HILO_EVAL_STR # [PAVPU_EVAL_STR, HILO_EVAL_STR, HILOSURF_EVAL_STR]
            , 'postprocessing': {
                'DIST_MAX_SURFERR': 3
            }
            , 'verbose': False    
        }

        # Step 0.1 - Fig DPI
        if params_global['figure']['for_paper']:
            params_global['figure']['dpi'] = 1000
            params_global['figure']['legend_fontsize'] = 20
            params_global['figure']['label_fontsize']  = 20
            params_global['figure']['ticks_fontsize']  = 15
            params_global['figure']['num_yticks']      = 6
        else:
            params_global['figure']['dpi'] = 100
            params_global['figure']['legend_fontsize'] = 15
            params_global['figure']['label_fontsize']  = 25
            params_global['figure']['ticks_fontsize']  = 12
        
        ###############################################################
        #  Step 1 - Choose experiment
        ###############################################################
        
        if 1:
            if 0:
                params_exp = {
                    'plt_alias': 'Pros-TrueDouble-1nAC-KL05-v2'
                    , 'plt_names' : ['Ens', 'Bayes', 'Bayes+1KAvU']
                    , 'plt_colors': [sns.color_palette('bright')[2], sns.color_palette('bright')[0], sns.color_palette('muted')[0]]
                    , 'exp_names': [
                                'Pros__ONetPool2Det-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
                                , 'Pros__ONetPool2BayesMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
                                , 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-1000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
                            ]
                    # , 'exp_epochs': [200, 200]
                    , 'exp_epochs': [500, 500, 500]
                    , 'exp_mcruns' : [None, 32, 32] 
                    , 'exp_trainingbool': [None, True, True]
                    , 'unc_norms'     : [True, True, True]
                    , 'unc_erodedil'  : [True, True, True]
                    , 'unc_thresholds': [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,1.0,0.05).tolist() + [0.96, 0.97, 0.98, 0.99, 1.0]
                    , 'ksize_dil' : (5,5,2) # (5,5,2), (3,3,1) # if only y_true: (15,15,7)
                    , 'ksize_err' : (3,3,1)
                    , 'plot_type' : 'swarm'
                }
            elif 0:
                params_exp = {
                    'plt_alias': 'Journal-Pros-DoublePool'
                    , 'plt_names' : ['Ens', 'Bayes', 'Bayes+AvU', 'Bayes(Head)', 'Bayes(Head)+AvU']
                    , 'plt_colors': [sns.color_palette('bright')[2], sns.color_palette('bright')[0], sns.color_palette('bright')[4], sns.color_palette('bright')[1], sns.color_palette('bright')[-2]] 
                    , 'exp_names': [
                                'Pros__ONetPool2Det-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
                                , 'Pros__ONetPool2BayesMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
                                , 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-1000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
                                , 'Pros__ONetPool2BayesHeadMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
                                , 'Pros__ONetPool2BayesHeadMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-1000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
                            ]
                    , 'exp_epochs': [500, 500, 500, 500, 500]
                    , 'exp_mcruns' : [None, 32, 32, 32, 32] 
                    , 'exp_trainingbool': [None, True, True, True, True]
                    , 'unc_norms'     : [True, True, True, True, True]
                    , 'unc_erodedil'  : [True, True, True, True, True]
                    # , 'unc_thresholds': [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,1.0,0.05).tolist() + [0.96, 0.97, 0.98, 0.99, 1.0]
                    # , 'unc_thresholds': [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,0.8,0.05).tolist()
                    , 'unc_thresholds': [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,0.6,0.05).tolist()
                    , 'ksize_dil' : (5,5,2) # (5,5,2), (3,3,1) # if only y_true: (15,15,7)
                    , 'ksize_err' : (3,3,1)
                    , 'plot_type' : PLOT_BARSWARM # ['swarm', 'boxswarm']
                }
            
            elif 1:
                params_exp = {
                    'plt_alias': 'Journal-Pros-DoublePoolVal'
                    , 'plt_names' : ['Ens', 'Bayes', 'Bayes+100AvU', 'Bayes+1KAvU', 'Bayes+10KAvU']
                    , 'plt_colors': [sns.color_palette('bright')[2], sns.color_palette('bright')[0], sns.color_palette('bright')[4], sns.color_palette('bright')[1], sns.color_palette('bright')[-2]] 
                    , 'exp_names': [
                                'Pros__ONetPool2Det-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
                                , 'Pros__ONetPool2BayesMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
                                , 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-100AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
                                , 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-1000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
                                , 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-10000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
                            ]
                    , 'exp_epochs': [500, 500, 500, 500, 500]
                    , 'exp_mcruns' : [None, 32, 32, 32, 32] 
                    , 'exp_trainingbool': [None, True, True, True, True]
                    , 'unc_norms'     : [True, True, True, True, True]
                    , 'unc_erodedil'  : [True, True, True, True, True]
                    # , 'unc_thresholds': [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,1.0,0.05).tolist() + [0.96, 0.97, 0.98, 0.99, 1.0]
                    # , 'unc_thresholds': [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,0.8,0.05).tolist()
                    , 'unc_thresholds': [0.0] + np.arange(1e-6, 1e-5, 1e-6).tolist() + np.arange(1e-5, 1e-4, 1e-5).tolist() + np.arange(1e-4, 1e-3, 1e-4).tolist() + np.arange(1e-3, 1e-2, 1e-3).tolist() + np.arange(0.01, 0.1, 0.005).tolist() + np.arange(0.1,0.6,0.05).tolist()
                    , 'ksize_dil' : (5,5,2) # (5,5,2), (3,3,1) # if only y_true: (15,15,7)
                    , 'ksize_err' : (3,3,1)
                    , 'plot_type' : PLOT_BARSWARM # ['swarm', 'boxswarm']
                }

        ###############################################################
        #  Step 2 - Choose dataset
        ###############################################################

        # PROMISE12
        if 0: 
            params_dataset = {
                'mode': config.DATASET_PROMISE12
                , 'patient_ids'   : config.PATIENTIDS_PROMISE12
                , 'patient_single': False
                , 'true_str'    : config.FILENAME_SAVE_GT
                , 'pred_str'    : config.FILENAME_SAVE_PRED
                , 'unc_str_ent' : config.FILENAME_SAVE_ENT 
            }
            params_dataset['unc_strs'] = [params_dataset['unc_str_ent']]
            
            if params_dataset['patient_single']:
                params_dataset['patient_ids'] = ['Case00', 'Case01']
                params_dataset['patient_single'] = params_dataset['patient_ids'][0]
            else:
                params_dataset['patient_single'] = ''

        # Pros-MedDec
        elif 1: 
            params_dataset = {
                'mode': config.DATASET_PROSMEDDEC
                , 'patient_ids'   : config.PATIENTIDS_PROMEDDEC
                , 'patient_single': True
                , 'true_str'    : config.FILENAME_SAVE_GT
                , 'pred_str'    : config.FILENAME_SAVE_PRED
                , 'unc_str_ent' : config.FILENAME_SAVE_ENT 
            }
            params_dataset['unc_strs'] = [params_dataset['unc_str_ent']]
            
            if params_dataset['patient_single']:
                params_dataset['patient_ids'] = ['ProsMedDec-00', 'ProsMedDec-01', 'ProsMedDec-02', 'ProsMedDec-04', 'ProsMedDec-06']
                params_dataset['patient_single'] = params_dataset['patient_ids'][0]
            else:
                params_dataset['patient_single'] = ''

        # prostateX
        elif 0:
            params_dataset = {
                'mode': config.DATASET_PROSTATEX
                , 'patient_ids'   : config.PATIENTIDS_PROSTATEX
                , 'patient_single': False
                , 'true_str'    : config.FILENAME_SAVE_GT
                , 'pred_str'    : config.FILENAME_SAVE_PRED
                , 'unc_str_ent' : config.FILENAME_SAVE_ENT 
            }
            params_dataset['unc_strs'] = [params_dataset['unc_str_ent']]
            
            if params_dataset['patient_single']:
                params_dataset['patient_ids'] = ['ProstateX-0004', 'ProstateX-0007']
                params_dataset['patient_single'] = params_dataset['patient_ids'][0]
            else:
                params_dataset['patient_single'] = ''

        ########################################## Step 3 - Some verbosity
        main(params_global, params_exp, params_dataset)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    print ('\n\n ------------------- Total Time: ', round(time.time() - T0, 2), 's')