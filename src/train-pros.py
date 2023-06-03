# Import internal libraries
import src.config as config
import src.models2 as models
import src.losses2 as losses
import src.utils as utils

from medloader.dataloader.tensorflow.pros_promise12 import Promise12Dataset
from medloader.dataloader.tensorflow.pros_prostatex import ProstateXDataset
from medloader.dataloader.tensorflow.pros_meddec import ProstateMedDecDataset
import medloader.dataloader.tensorflow.pros_prostatex as Aug

# Import external libraries
import os
import gc
import pdb
import copy
import time
import tqdm
import datetime
import traceback
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# import tensorflow as tf # v2.9.1 (cuda-11.2.2, cudnn=8.1.0.77)
# import tensorflow_probability as tfp # v0.17.0
# # import tensorflow_addons as tfa # v0.17.1

import tensorflow as tf # v2.10.0 (cuda-11.2.2, cudnn=8.1.0.77)
import tensorflow_addons as tfa # v0.18.0
import tensorflow_probability as tfp # v0.18.0

############################################################
#                      METRICS RELATED                     #
############################################################

class ModelMetrics():
    
    def __init__(self, metric_type, params):
        
        self.params = params

        self.label_map = params['internal']['label_map']
        self.label_ids = params['internal']['label_ids']
        self.logging_tboard = params['metrics']['logging_tboard']
        self.metric_type = metric_type # [Test, Train]

        self.losses_obj = self.get_losses_obj(params)
        self.metrics_layers_kl_divergence = {} # empty for now

        self.init_metrics(params)
        if self.logging_tboard:
            self.init_tboard_writers(params)

        self.reset_metrics(params)
        self.init_epoch0()
        self.reset_metrics(params)
    
    def get_losses_obj(self, params):
        losses_obj = {} 
        for loss_key in params['metrics']['metrics_loss']:
            if config.LOSS_DICE == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_dice_3d_tf_func
            if config.LOSS_FOCAL == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_focal_3d_tf_func
            if config.LOSS_CE  == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_ce_3d_tf_func
            if config.LOSS_PAVPU == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_avu_3d_tf_func
            if config.LOSS_CE_BOUNDARY == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_ce_boundary_3d_tf_func
        
        return losses_obj
    
    def init_metrics(self, params):
        """
        These are metrics derived from tensorflows library 
        """
        # Metrics for losses (during training for smaller grids)
        self.metrics_loss_obj = {}
        for metric_key in params['metrics']['metrics_loss']:
            self.metrics_loss_obj[metric_key] = {}
            self.metrics_loss_obj[metric_key]['total'] = tf.keras.metrics.Mean(name='Avg{}-{}'.format(metric_key, self.metric_type))
            if params['metrics']['metrics_loss'][metric_key] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_CE_BOUNDARY]:
                for label_id in self.label_ids:
                    self.metrics_loss_obj[metric_key][label_id] = tf.keras.metrics.Mean(name='Avg{}-Label-{}-{}'.format(metric_key, label_id, self.metric_type))
        
        # Metrics for eval (for full 3D volume)
        self.metrics_eval_obj = {}
        for metric_key in params['metrics']['metrics_eval']:
            self.metrics_eval_obj[metric_key] = {}
            self.metrics_eval_obj[metric_key]['total'] = tf.keras.metrics.Mean(name='Avg{}-{}'.format(metric_key, self.metric_type))
            if params['metrics']['metrics_eval'][metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.metrics_eval_obj[metric_key][label_id] = tf.keras.metrics.Mean(name='Avg{}-Label-{}-{}'.format(metric_key, label_id, self.metric_type))

        # Time Metrics
        self.metric_time_dataloader     = tf.keras.metrics.Mean(name='AvgTime-Dataloader-{}'.format(self.metric_type))
        self.metric_time_model_predict  = tf.keras.metrics.Mean(name='AvgTime-ModelPredict-{}'.format(self.metric_type))
        self.metric_time_model_loss     = tf.keras.metrics.Mean(name='AvgTime-ModelLoss-{}'.format(self.metric_type))
        self.metric_time_model_backprop = tf.keras.metrics.Mean(name='AvgTime-ModelBackProp-{}'.format(self.metric_type))   

        # Memory Metrics
        self.metric_memory_gpu = tf.keras.metrics.Mean(name='AvgMemory-GPU-{}'.format(self.metric_type))
        self.metric_memory_ram = tf.keras.metrics.Mean(name='AvgMemory-RAM-{}'.format(self.metric_type))

        # FlipOut Metrics
        self.metric_kl_alpha      = tf.keras.metrics.Mean(name='KL-Alpha')
        self.metric_kl_divergence = tf.keras.metrics.Mean(name='KL-Divergence')

        # Scalar Losses
        self.metric_scalarloss_data = tf.keras.metrics.Mean(name='ScalarLoss-Data')
        self.metric_scalarloss_reg  = tf.keras.metrics.Mean(name='ScalarLoss-Reg')
    
    def init_metrics_layers_kl(self, params, layers_kl):

        # Step 1 - Init 
        self.metrics_layers_kl_divergence = {}
        self.tboard_layers_kl_divergence  = {}
        self.writer_tboard_layers_std     = {}
        self.writer_tboard_layers_mean    = {}
        
        # Step 2 - Loop over all layers
        for layer_name in layers_kl:
            self.metrics_layers_kl_divergence[layer_name] = tf.keras.metrics.Mean(name='KL-Divergence-{}'.format(layer_name))
            self.tboard_layers_kl_divergence[layer_name] = utils.get_tensorboard_writer(params['exp_name'], suffix='KL-Divergence-Layer-{}'.format(layer_name))
            utils.make_summary('BayesLossExtras/FlipOut/KLDivergence-{}'.format(layer_name), epoch=0, writer1=self.tboard_layers_kl_divergence[layer_name], value1=2e5)
            self.metrics_layers_kl_divergence[layer_name].reset_states()

            if 'std' in layers_kl[layer_name]:
                keyname = layer_name + '-std'
                self.writer_tboard_layers_std[keyname] = utils.get_tensorboard_writer(params['exp_name'], suffix='Std-Layer-{}'.format(keyname))
                utils.make_summary_hist('Std/{}'.format(keyname), epoch=0, writer1=self.writer_tboard_layers_std[keyname], value1=layers_kl[layer_name]['std'])
            
            if 'mean' in layers_kl[layer_name]: # keep this between [-2,+2] for better visualization in tf.summary.histogram() 
                keyname = layer_name + '-mean'
                self.writer_tboard_layers_mean[keyname] = utils.get_tensorboard_writer(params['exp_name'], suffix='Mean-Layer-{}'.format(keyname))
                mean_vals = layers_kl[layer_name]['mean'].numpy()
                mean_vals = mean_vals[mean_vals >= -2]
                mean_vals = mean_vals[mean_vals <= 2]
                utils.make_summary_hist('Mean/{}'.format(keyname), 0, writer1=self.writer_tboard_layers_mean[keyname], value1=mean_vals)

    def reset_metrics(self, params):

        # Metrics for losses (during training for smaller grids)
        for metric_key in params['metrics']['metrics_loss']:
            self.metrics_loss_obj[metric_key]['total'].reset_states()
            if params['metrics']['metrics_loss'][metric_key] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_CE_BOUNDARY]:
                for label_id in self.label_ids:
                    self.metrics_loss_obj[metric_key][label_id].reset_states()

        # Metrics for eval (for full 3D volume)
        for metric_key in params['metrics']['metrics_eval']:
            self.metrics_eval_obj[metric_key]['total'].reset_states()
            if params['metrics']['metrics_eval'][metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.metrics_eval_obj[metric_key][label_id].reset_states()

        # Time Metrics
        self.metric_time_dataloader.reset_states()
        self.metric_time_model_predict.reset_states()
        self.metric_time_model_loss.reset_states()
        self.metric_time_model_backprop.reset_states()

        # Memory Metrics
        self.metric_memory_gpu.reset_states()
        self.metric_memory_ram.reset_states()

        # FlipOut Metrics
        self.metric_kl_alpha.reset_states()
        self.metric_kl_divergence.reset_states()

        # Scalar Losses
        self.metric_scalarloss_data.reset_states()
        self.metric_scalarloss_reg.reset_states()

        # FlipOut-Layers
        for layer_name in self.metrics_layers_kl_divergence:
            self.metrics_layers_kl_divergence[layer_name].reset_states()
    
    def init_tboard_writers(self, params):
        """
        These are tensorboard writer
        """
        # Writers for loss (during training for smaller grids)
        self.writers_loss_obj = {}
        for metric_key in params['metrics']['metrics_loss']:
            self.writers_loss_obj[metric_key] = {}
            self.writers_loss_obj[metric_key]['total'] = utils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Loss')
            if params['metrics']['metrics_loss'][metric_key] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_CE_BOUNDARY]:
                for label_id in self.label_ids:
                    self.writers_loss_obj[metric_key][label_id] = utils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Loss-' + str(label_id))
        
        # Writers for eval (for full 3D volume)
        self.writers_eval_obj = {}
        for metric_key in params['metrics']['metrics_eval']:
            self.writers_eval_obj[metric_key] = {}
            self.writers_eval_obj[metric_key]['total'] = utils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Eval')
            if params['metrics']['metrics_eval'][metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.writers_eval_obj[metric_key][label_id] = utils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Eval-' + str(label_id))

        # Time and other writers
        self.writer_lr                  = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-LR'.format(self.metric_type))
        self.writer_time_dataloader     = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-Time-Dataloader'.format(self.metric_type))
        self.writer_time_model_predict  = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-Time-Model-Predict'.format(self.metric_type))
        self.writer_time_model_loss     = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-Time-Model-Loss'.format(self.metric_type))
        self.writer_time_model_backprop = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-Time-Model-Backprop'.format(self.metric_type))

        # Memory Writers
        self.writer_memory_gpu          = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-Memory-GPU'.format(self.metric_type))
        self.writer_memory_ram          = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-Memory-RAM'.format(self.metric_type))

        # FlipOut writers
        self.writer_kl_alpha            = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-KL-Alpha'.format(self.metric_type))
        self.writer_kl_divergence       = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-KL-Divergence'.format(self.metric_type))

        # Scalar Losses
        self.writer_scalarloss_data     = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-ScalarLoss-Data'.format(self.metric_type))
        self.writer_scalarloss_reg      = utils.get_tensorboard_writer(params['exp_name'], suffix='{}-ScalarLoss-Reg'.format(self.metric_type))

    def init_epoch0(self):
        
        # Step 1.1 - Update train loss metrics
        for metric_str in self.metrics_loss_obj:
            self.update_metric_loss(metric_str, 1e-6)
            if params['metrics']['metrics_loss'][metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_CE_BOUNDARY]:
                # self.update_metric_loss_labels(metric_str, {label_id: 1e-6 for label_id in self.label_ids})
                self.update_metric_loss_labels(metric_str, [1e-6 for label_id in self.label_ids])

        # Step 1.2 - Update eval metrics
        for metric_str in self.metrics_eval_obj:
            # self.update_metric_eval_labels(metric_str, {label_id: 0 for label_id in self.label_ids})
            self.update_metric_eval_labels(metric_str, [0 for label_id in self.label_ids])
        
        # Step 1.3 - Update memory metrics
        self.update_metrics_memory(0,0)
        
        # Step 1.4 - Update time metrics
        self.update_metrics_time(time_dataloader=1e-5, time_predict=1e-5, time_loss=1e-5, time_backprop=1e-5)
        
        # Step 1.5 - Update KL divergence metrics
        self.update_metrics_kl(kl_alpha=1e-5, kl_divergence=1e-5, kl_divergence_layers={})
        self.update_metrics_scalarloss(loss_data=1e-5, loss_reg=0)
        
        # Step 2 - Write epoch stats to tensorboard
        self.write_epoch_summary(epoch=0, label_map=self.label_map, params=None, eval_condition=True)

    def update_metrics_memory(self, memory_gpu, memory_ram):
        self.metric_memory_gpu.update_state(memory_gpu)
        self.metric_memory_ram.update_state(memory_ram)

    @tf.function
    def update_metrics_kl(self, kl_alpha, kl_divergence, kl_divergence_layers):
        self.metric_kl_alpha.update_state(kl_alpha)
        self.metric_kl_divergence.update_state(kl_divergence)
        
        for layer_name in kl_divergence_layers:
            if layer_name in self.metrics_layers_kl_divergence:
                self.metrics_layers_kl_divergence[layer_name].update_state(kl_divergence_layers[layer_name]['kl'])
    
    @tf.function
    def update_metrics_scalarloss(self, loss_data, loss_reg):
        self.metric_scalarloss_data.update_state(loss_data)
        self.metric_scalarloss_reg.update_state(loss_reg)

    def update_metrics_time(self, time_dataloader, time_predict, time_loss, time_backprop):
        if time_dataloader is not None:
            self.metric_time_dataloader.update_state(time_dataloader)
        if time_predict is not None:
            self.metric_time_model_predict.update_state(time_predict)
        if time_loss is not None:
            self.metric_time_model_loss.update_state(time_loss)
        if time_backprop is not None:
            self.metric_time_model_backprop.update_state(time_backprop)  

    def update_metric_loss(self, metric_str, metric_val):
        # Metrics for losses (during training for smaller grids)
        self.metrics_loss_obj[metric_str]['total'].update_state(metric_val)
    
    @tf.function
    def update_metric_loss_labels(self, metric_str, metric_vals_labels):
        # Metrics for losses (during training for smaller grids)
        # metric_vals_labels: list

        for label_id in self.label_ids:
            if metric_vals_labels[label_id-1] > 0.0:
                self.metrics_loss_obj[metric_str][label_id].update_state(metric_vals_labels[label_id-1])
    
    def update_metric_eval(self, metric_str, metric_val):
        # Metrics for eval (for full 3D volume)
        self.metrics_eval_obj[metric_str]['total'].update_state(metric_val)
        
    def update_metric_eval_labels(self, metric_str, metric_vals_labels, do_average=False):
        # Metrics for eval (for full 3D volume)
        # metric_vals_labels: dict

        try:
            metric_avg = []
            for label_id in metric_vals_labels:
                if metric_vals_labels[label_id] > 0:
                    self.metrics_eval_obj[metric_str][label_id+1].update_state(metric_vals_labels[label_id])
                    if do_average:
                        metric_avg.append(metric_vals_labels[label_id])
            
            if do_average:
                if len(metric_avg):
                    self.metrics_eval_obj[metric_str]['total'].update_state(np.mean(metric_avg))
        
        except:
            traceback.print_exc()

    def write_epoch_summary(self, epoch, label_map, params=None, eval_condition=False):
        
        if self.logging_tboard:
            # Metrics for losses (during training for smaller grids)
            for metric_str in self.metrics_loss_obj:
                utils.make_summary('Loss/{}'.format(metric_str), epoch, writer1=self.writers_loss_obj[metric_str]['total'], value1=self.metrics_loss_obj[metric_str]['total'].result())
                if self.params['metrics']['metrics_loss'][metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_CE_BOUNDARY]:
                    if len(self.metrics_loss_obj[metric_str]) > 1: # i.e. has label ids
                        for label_id in self.label_ids:
                            label_name, _ = utils.get_info_from_label_id(label_id, label_map)
                            utils.make_summary('Loss/{}/{}'.format(metric_str, label_name), epoch, writer1=self.writers_loss_obj[metric_str][label_id], value1=self.metrics_loss_obj[metric_str][label_id].result())
                
            # Metrics for eval (for full 3D volume)
            if eval_condition:
                for metric_str in self.metrics_eval_obj:
                    utils.make_summary('Eval3D/{}'.format(metric_str), epoch, writer1=self.writers_eval_obj[metric_str]['total'], value1=self.metrics_eval_obj[metric_str]['total'].result())
                    if len(self.metrics_eval_obj[metric_str]) > 1: # i.e. has label ids
                        for label_id in self.label_ids:
                            label_name, _ = utils.get_info_from_label_id(label_id, label_map)
                            utils.make_summary('Eval3D/{}/{}'.format(metric_str, label_name), epoch, writer1=self.writers_eval_obj[metric_str][label_id], value1=self.metrics_eval_obj[metric_str][label_id].result())

            # Time Metrics
            utils.make_summary('Info/Time/Dataloader'   , epoch, writer1=self.writer_time_dataloader    , value1=self.metric_time_dataloader.result())
            utils.make_summary('Info/Time/ModelPredict' , epoch, writer1=self.writer_time_model_predict , value1=self.metric_time_model_predict.result())
            utils.make_summary('Info/Time/ModelLoss'    , epoch, writer1=self.writer_time_model_loss    , value1=self.metric_time_model_loss.result())
            utils.make_summary('Info/Time/ModelBackProp', epoch, writer1=self.writer_time_model_backprop, value1=self.metric_time_model_backprop.result())

            # Memory Metrics
            utils.make_summary('Info/Memory/GPU'        , epoch, writer1=self.writer_memory_gpu         , value1=self.metric_memory_gpu.result())
            utils.make_summary('Info/Memory/RAM'        , epoch, writer1=self.writer_memory_ram         , value1=self.metric_memory_ram.result())

            # FlipOut Metrics
            utils.make_summary('BayesLoss/FlipOut/KLAlpha'      , epoch, writer1=self.writer_kl_alpha           , value1=self.metric_kl_alpha.result())
            utils.make_summary('BayesLoss/FlipOut/KLDivergence' , epoch, writer1=self.writer_kl_divergence      , value1=self.metric_kl_divergence.result())
            for layer_name in self.metrics_layers_kl_divergence:
                utils.make_summary('BayesLossExtras/FlipOut/KLDivergence-{}'.format(layer_name), epoch, writer1=self.tboard_layers_kl_divergence[layer_name], value1=self.metrics_layers_kl_divergence[layer_name].result())

            # Scalar Loss Metrics
            utils.make_summary('BayesLoss/FlipOut/ScalarLossData'  , epoch, writer1=self.writer_scalarloss_data , value1=self.metric_scalarloss_data.result())
            utils.make_summary('BayesLoss/FlipOut/ScalarLossReg'   , epoch, writer1=self.writer_scalarloss_reg  , value1=self.metric_scalarloss_reg.result())
            
            # Learning Rate
            if params is not None:
                if 'optimizer' in params:
                    utils.make_summary('Info/LR', epoch, writer1=self.writer_lr, value1=params['optimizer'].lr)

    def write_epoch_summary_kl_hist(self, layers_kl, epoch):

        # Step 1 - Loop over all layers
        for layer_name in layers_kl:
            
            if 'std' in layers_kl[layer_name]:
                keyname = layer_name + '-std'
                utils.make_summary_hist('Std/{}'.format(keyname), epoch, writer1=self.writer_tboard_layers_std[keyname], value1=layers_kl[layer_name]['std'])
            
            if 'mean' in layers_kl[layer_name]:
                keyname = layer_name + '-mean'
                mean_vals = layers_kl[layer_name]['mean'].numpy()
                mean_vals = mean_vals[mean_vals >= -2]
                mean_vals = mean_vals[mean_vals <= 2]
                utils.make_summary_hist('Mean/{}'.format(keyname), epoch, writer1=self.writer_tboard_layers_mean[keyname], value1=mean_vals)

    def update_pbar(self, pbar):
        desc_str = ''

        # Metrics for losses (during training for smaller grids)
        # Metrics for losses (during training for smaller grids)
        if config.LOSS_PURATIO in self.metrics_loss_obj or config.LOSS_PAVPU in self.metrics_loss_obj:
            for metric_str in self.metrics_loss_obj:
                if metric_str in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_PURATIO, config.LOSS_PAVPU, config.LOSS_CE_BOUNDARY]:
                    result = self.metrics_loss_obj[metric_str]['total'].result().numpy()
                    loss_text = '{}:{:2f},'.format(metric_str, result)
                    desc_str += loss_text
        else:
            for metric_str in self.metrics_loss_obj:
                if len(desc_str): desc_str += ',' 

                if metric_str in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_CE_BOUNDARY]:
                    metric_avg = []
                    for label_id in self.label_ids:
                        if label_id > 0:
                            label_result = self.metrics_loss_obj[metric_str][label_id].result().numpy()
                            if label_result > 0:
                                metric_avg.append(label_result)
                    
                    mean_val = 0
                    if len(metric_avg):     
                        mean_val = np.mean(metric_avg)
                    loss_text = '{}Loss:{:2f}'.format(metric_str, mean_val)
                    desc_str += loss_text
        
        # GPU Memory
        if 1:
            try:
                if len(self.metrics_loss_obj) > 1:
                    desc_str = desc_str[:-1] # to remove the extra ','     
                # desc_str += ',' + str(utils.get_tf_gpu_memory())
                desc_str += ',' + str(utils.get_nvitop_gpu_memory())
            except:
                pass
        
        pbar.set_description(desc=desc_str, refresh=True)

############################################################
#                           VAL                            #
############################################################


def eval_3D_finalize(exp_name, patient_img, patient_gt, patient_pred_processed, patient_pred, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc, patient_pred_error
                        , patient_id_curr
                        , model_folder_epoch_imgs, model_folder_epoch_patches 
                        , loss_labels_val, hausdorff_labels_val, hausdorff95_labels_val, msd_labels_vals
                        , spacing, label_map, label_colors
                        , show=False, save=False):
    
    """
    Params
    -----
    patient_img           : [H,W,D]
    patient_gt            : [H,W,D,C]
    patient_pred_processed: [H,W,D,C] 
    patient_pred          : [H,W,D,C]
    patient_pred_std      : [H,W,D,C]
    patient_pred_ent      : [H,W,D]
    patient_pred_mif      : [H,W,D]
    """

    try:
        
        # pdb.set_trace()
        # medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath('nrrd_{}_maskpreds.nrrd'.format(patient_id_curr)), patient_pred, spacing)

        # Step 3.1.2 - Vizualize
        if show:
            if len(patient_pred_std):
                maskpred_std = np.max(patient_pred_std, axis=-1)
                maskpred_std = np.expand_dims(maskpred_std, axis=-1)
                maskpred_std = np.repeat(maskpred_std, repeats=10, axis=-1)

            maskpred_ent = np.expand_dims(patient_pred_ent, axis=-1)    # [H,W,D]   --> [H,W,D,1]
            maskpred_ent = np.repeat(maskpred_ent, repeats=10, axis=-1) # [H,W,D,1] --> [H,W,D,10]

            maskpred_mif = np.expand_dims(patient_pred_mif, axis=-1)    # [H,W,D]   --> [H,W,D,1]
            maskpred_mif = np.repeat(maskpred_mif, repeats=10, axis=-1) # [H,W,D,1] --> [H,W,D,10]

            if 1:
                print (' - patient_id_curr: ', patient_id_curr)
                f,axarr = plt.subplots(1,2)
                axarr[0].hist(maskpred_ent[:,:,:,0].flatten(), bins=30)
                axarr[0].set_title('Entropy')
                axarr[1].hist(maskpred_mif[:,:,:,0].flatten(), bins=30)
                axarr[1].set_title('MutInf')
                plt.suptitle('Exp: {}\nPatient:{}'.format(exp_name, patient_id_curr))
                plt.show()
                pdb.set_trace()

            utils.viz_model_output_3d(exp_name, patient_img, patient_gt, patient_pred, maskpred_std, patient_id_curr, model_folder_epoch_imgs, label_map, label_colors
                                        , vmax_unc=0.06, unc_title='Predictive Std', unc_savesufix='stdmax')
            
            utils.viz_model_output_3d(exp_name, patient_img, patient_gt, patient_pred, maskpred_ent, patient_id_curr, model_folder_epoch_imgs, label_map, label_colors
                                        , vmax_unc=1.2, unc_title='Predictive Entropy', unc_savesufix='ent')

            utils.viz_model_output_3d(exp_name, patient_img, patient_gt, patient_pred, maskpred_mif, patient_id_curr, model_folder_epoch_imgs, label_map, label_colors
                                        , vmax_unc=0.06, unc_title='Mutual Information', unc_savesufix='mif')

        # Step 3.1.3 - Save 3D grid to visualize in 3D Slicer (drag-and-drop mechanism)
        if save:
            
            class_count = patient_gt.shape[-1]

            # Step 1 - Basics (Raw/GT/Pred)
            import medloader.dataloader.utils as medutils
            medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_CT.format(patient_id_curr)), np.array(patient_img), spacing)

            if class_count > 1:
                medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_GT.format(patient_id_curr)), np.argmax(patient_gt, axis=3),spacing)
            else:
                tmp = np.array(patient_gt[:,:,:,0], copy=True)
                tmp[tmp > 0.5] = 1
                tmp[tmp <= 0.5] = 0
                medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_GT.format(patient_id_curr)), tmp, spacing)

            medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_PROB.format(patient_id_curr)), np.max(np.array(patient_pred), axis=3), spacing)
            
            if class_count > 1:
                maskpred_labels = np.argmax(patient_pred_processed, axis=3) # not "np.argmax(patient_pred, axis=3)" since it does not contain any postprocessing
                medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_PRED.format(patient_id_curr)), maskpred_labels, spacing)
            else:
                tmp_labels = np.array(patient_pred_processed[:,:,:,0], copy=True)
                tmp_labels[tmp_labels > 0.5] = 1
                tmp_labels[tmp_labels <= 0.5] = 0
                medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_PRED.format(patient_id_curr)), tmp_labels, spacing)

            # Step 2 - Uncertainties (Raw/GT/Pred)
            if np.sum(patient_pred_std):
                
                if class_count > 1:
                    maskpred_labels_std = np.take_along_axis(patient_pred_std, np.expand_dims(maskpred_labels,axis=-1), axis=-1)[:,:,:,0]
                    medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_STD.format(patient_id_curr)), maskpred_labels_std, spacing)
                else:
                    medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_STD.format(patient_id_curr)), patient_pred_std[:,:,:,0], spacing)

                if class_count > 1:
                    maskpred_std_max = np.max(patient_pred_std, axis=-1)
                    medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_STD_MAX.format(patient_id_curr)), maskpred_std_max, spacing)
                else:
                    medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_STD_MAX.format(patient_id_curr)), patient_pred_std[:,:,:,0], spacing)

            if np.sum(patient_pred_ent):
                maskpred_ent     = patient_pred_ent # [H,W,D]
                medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_ENT.format(patient_id_curr)), maskpred_ent, spacing)

            if np.sum(patient_pred_mif):
                maskpred_mif = patient_pred_mif
                medutils.write_nrrd(Path(model_folder_epoch_patches).joinpath(config.FILENAME_SAVE_MIF.format(patient_id_curr)), maskpred_mif, spacing)

            # if np.sum(patient_pred_unc):
            #     if len(patient_pred_unc.shape) == 4:
            #         maskpred_labels_unc = np.take_along_axis(patient_pred_unc, np.expand_dims(maskpred_labels,axis=-1), axis=-1)[:,:,:,0] # [H,W,D,C] --> [H,W,D]
            #     else:
            #         maskpred_labels_unc = patient_pred_unc
            #     medutils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpredunc.nrrd', maskpred_labels_unc, spacing)

            try:
                # Step 3.1.3.2 - PLot results for that patient
                f, axarr = plt.subplots(3,1, figsize=(15,10))
                boxplot_dice, boxplot_hausdorff, boxplot_hausdorff95 = {}, {}, {}
                boxplot_dice_mean_list = []
                for label_id in range(len(loss_labels_val)):
                    label_name, _ = utils.get_info_from_label_id(label_id, label_map)
                    boxplot_dice[label_name] = [loss_labels_val[label_id]]
                    boxplot_hausdorff[label_name] = [hausdorff_labels_val[label_id]]
                    boxplot_hausdorff95[label_name] = [hausdorff95_labels_val[label_id]]
                    boxplot_dice_mean_list.append(loss_labels_val[label_id])
                
                axarr[0].boxplot(boxplot_dice.values())
                axarr[0].set_xticks(range(1, len(boxplot_dice)+1))
                axarr[0].set_xticklabels(boxplot_dice.keys())
                axarr[0].set_ylim([0.0,1.1])
                axarr[0].grid()
                axarr[0].set_title('DICE - Avg: {} \n w/o chiasm: {}'.format(
                    '%.4f' % (np.mean(boxplot_dice_mean_list))
                    , '%.4f' % (np.mean(boxplot_dice_mean_list[0:1] + boxplot_dice_mean_list[2:])) # avoid label_id=2
                    )
                )

                axarr[1].boxplot(boxplot_hausdorff.values())
                axarr[1].set_xticks(range(1,len(boxplot_hausdorff)+1))
                axarr[1].set_xticklabels(boxplot_hausdorff.keys())
                axarr[1].grid()
                axarr[1].set_title('Hausdorff')
                
                axarr[2].boxplot(boxplot_hausdorff95.values())
                axarr[2].set_xticks(range(1,len(boxplot_hausdorff95)+1))
                axarr[2].set_xticklabels(boxplot_hausdorff95.keys())
                axarr[2].set_title('95% Hausdorff')
                axarr[2].grid()

                plt.savefig(str(Path(model_folder_epoch_patches).joinpath('results_' + patient_id_curr + '.png')), bbox_inches='tight') # , bbox_inches='tight'
                plt.close()
            
            except:
                traceback.print_exc()
                pdb.set_trace()
    
    except:
        print ('\n - [ERROR][eval_3D_finalize()] ')
        traceback.print_exc()
        pdb.set_trace()
        
def eval_3D_summarize(res, ece_global_obj, model, eval_type, deepsup_eval, label_map, model_folder_epoch_patches
                        , times_mcruns, MC_RUNS, ttotal, save=False, show=False, verbose=False):
    """
    Get stats for eval_3D 
    """

    try:
        
        ###############################################################################
        #                                 Summarize                                   #
        ###############################################################################    
        
        # Step 0 - Init
        loss_avg = 0.0
        loss_labels_avg = {}
        
        # Step 1 - Summarize DICE + Surface Distances
        loss_labels_avg, loss_labels_std               = [], []
        hausdorff_labels_avg, hausdorff_labels_std     = [], []
        hausdorff95_labels_avg, hausdorff95_labels_std = [], []
        msd_labels_avg, msd_labels_std                 = [], []
        if 1:
            
            loss_labels_list                               = np.array([res[patient_id][config.KEY_DICE_LABELS] for patient_id in res])
            hausdorff_labels_list                          = np.array([res[patient_id][config.KEY_HD_LABELS] for patient_id in res])
            hausdorff95_labels_list                        = np.array([res[patient_id][config.KEY_HD95_LABELS] for patient_id in res])
            msd_labels_list                                = np.array([res[patient_id][config.KEY_MSD_LABELS] for patient_id in res])

            
            for label_id in range(loss_labels_list.shape[1]): 
                tmp_vals = loss_labels_list[:,label_id]
                loss_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0]))
                loss_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

                tmp_vals = hausdorff_labels_list[:,label_id]
                hausdorff_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0])) # avoids -1 for "erroneous" HD, and 0 for "not to be calculated" HD
                hausdorff_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

                tmp_vals = hausdorff95_labels_list[:,label_id]
                hausdorff95_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0]))
                hausdorff95_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

                tmp_vals = msd_labels_list[:,label_id]
                msd_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0]))
                msd_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

            loss_avg = np.mean([res[patient_id][config.KEY_DICE_AVG] for patient_id in res])
            if config.KEY_PATIENT_GLOBAL not in res:
                res[config.KEY_PATIENT_GLOBAL] = {}

            try:
                print ('\n --------------------------- eval_type: {} || {}'.format(eval_type, utils.get_nvitop_gpu_memory()))
                print (' - dice_3D (w/o bgd): {:.4f} ± {:.4f}'.format(np.mean(loss_labels_list), np.std(loss_labels_list)))
                res[config.KEY_PATIENT_GLOBAL][config.KEY_DICE_AVG] = np.mean(loss_labels_list)
                res[config.KEY_PATIENT_GLOBAL][config.KEY_DICE_STD] = np.std(loss_labels_list)
                
                print ('')
                print (' - hausdorff_3D (w/o bgd): {:.4f} ± {:.4f}'.format(np.mean(hausdorff_labels_list), np.std(hausdorff_labels_list) ) )
                res[config.KEY_PATIENT_GLOBAL][config.KEY_HD_AVG] = np.mean(hausdorff_labels_list)
                res[config.KEY_PATIENT_GLOBAL][config.KEY_HD_STD] = np.std(hausdorff_labels_list)
                
                print ('')
                print (' - hausdorff95_3D (w/o bgd): {:.4f} ± {:.4f}'.format(np.mean(hausdorff95_labels_list), np.std(hausdorff95_labels_list) ) )
                res[config.KEY_PATIENT_GLOBAL][config.KEY_HD95_AVG] = np.mean(hausdorff95_labels_list)
                res[config.KEY_PATIENT_GLOBAL][config.KEY_HD95_STD] = np.std(hausdorff95_labels_list)
                
                print ('')
                print (' - msd_3D (w/o bgd): {:.4f} ± {:.4f}'.format(np.mean(msd_labels_list), np.std(msd_labels_list) ) )
                res[config.KEY_PATIENT_GLOBAL][config.KEY_MSD_AVG] = np.mean(msd_labels_list)
                res[config.KEY_PATIENT_GLOBAL][config.KEY_MSD_STD] = np.std(msd_labels_list)
                
            except:
                traceback.print_exc()
        
        # Step 3 - Summarize ECE
        if 1:
            print ('')
            gc.collect()

            ece_global_obj.update_state_global() 
            ece_global_obj.print()
            ece_global_obj.plot(only_global=True)
            
            for calibration_type in ece_global_obj.res:
                for patient_id in ece_global_obj.res[calibration_type]:
                    if patient_id not in res: res[patient_id] = {}
                    res[patient_id][calibration_type] = ece_global_obj.res[calibration_type][patient_id]

        # Step 4 - Plot
        if 1:
            if not deepsup_eval:
                f, axarr = plt.subplots(3,1, figsize=(15,10))
                boxplot_dice, boxplot_hausdorff, boxplot_hausdorff95, boxplot_msd = {}, {}, {}, {}
                for label_id in range(len(loss_labels_list[0])):
                    label_name, _ = utils.get_info_from_label_id(label_id, label_map)
                    boxplot_dice[label_name] = loss_labels_list[:,label_id]
                    boxplot_hausdorff[label_name] = hausdorff_labels_list[:,label_id]
                    boxplot_hausdorff95[label_name] = hausdorff95_labels_list[:,label_id]
                    boxplot_msd[label_name] = msd_labels_list[:,label_id]

                axarr[0].boxplot(boxplot_dice.values())
                axarr[0].set_xticks(range(1, len(boxplot_dice)+1))
                axarr[0].set_xticklabels(boxplot_dice.keys())
                axarr[0].set_ylim([0.0,1.1])
                axarr[0].set_title('DICE (Avg: {}) '.format( 
                    '%.4f' %  np.mean(loss_labels_avg)
                ))

                axarr[1].boxplot(boxplot_hausdorff.values())
                axarr[1].set_xticks(range(1, len(boxplot_hausdorff)+1))
                axarr[1].set_xticklabels(boxplot_hausdorff.keys())
                axarr[1].set_ylim([0.0,10.0])
                axarr[1].set_title('Hausdorff')
                
                axarr[2].boxplot(boxplot_hausdorff95.values())
                axarr[2].set_xticks(range(1, len(boxplot_hausdorff95)+1))
                axarr[2].set_xticklabels(boxplot_hausdorff95.keys())
                axarr[2].set_ylim([0.0,6.0])
                axarr[2].set_title('95% Hausdorff')
                
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_dice.values())
                axarr.set_xticks(range(1, len(boxplot_dice)+1))
                axarr.set_xticklabels(boxplot_dice.keys())
                axarr.set_ylim([0.0,1.1])
                axarr.set_yticks(np.arange(0,1.1,0.05))
                axarr.set_title('DICE')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_dice.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_hausdorff95.values())
                axarr.set_xticks(range(1, len(boxplot_hausdorff95)+1))
                axarr.set_xticklabels(boxplot_hausdorff95.keys())
                axarr.set_title('95% HD')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_hd95.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_hausdorff.values())
                axarr.set_xticks(range(1, len(boxplot_hausdorff)+1))
                axarr.set_xticklabels(boxplot_hausdorff.keys())
                axarr.set_title('Hausdorff Distance')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_hd.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_msd.values())
                axarr.set_xticks(range(1, len(boxplot_msd)+1))
                axarr.set_xticklabels(boxplot_msd.keys())
                axarr.set_title('MSD')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_msd.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

        # Step 5 - Save data as .json
        if 1:
            try:
                
                filename_json = str(Path(model_folder_epoch_patches).joinpath(config.FILENAME_EVAL3D_JSON))
                print (' - filename_json: ', filename_json)
                utils.write_json(res, filename_json)

            except:
                traceback.print_exc()
                pdb.set_trace()

        try:
            model.trainable=True # in case model is None
        except:
            pass

        print ('\n - [eval_3D()] Avg of MC={}               : {:f} +- {:f}'.format(MC_RUNS, np.mean(times_mcruns), np.std(times_mcruns)))
        print (' - [eval_3D()] Total time passed (save={})  : {}s \n'.format(save, round(time.time() - ttotal, 2)))
        if verbose: pdb.set_trace()

        return loss_avg, {i:loss_labels_avg[i] for i in range(len(loss_labels_avg))}

    except:
        model.trainable=True
        traceback.print_exc()
        return -1, {}

def eval_3D_process_outputs(res, ece_global_obj, patient_id_curr, meta1_batch, patient_img, patient_gt
                            , patient_pred_vals, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc
                            , deepsup_eval, model_folder_epoch_imgs, model_folder_epoch_patches
                            , label_map, label_colors
                            , t99, show=False, save=False
                            , verbose=False):

    """
    Params
    ------
    patient_img         : [H,W,D]
    patient_gt          : [H,W,D,C]
    patient_pred_vals   : [H,W,D,C]
    patient_pred_std    : [H,W,D,C]
    patient_pred_ent    : [H,W,D]
    patient_pred_mif    : [H,W,D]
    patient_pred_unc    : [H,W,D,C]
    """
    try:
        
        # Step 3.1.2 - Volumetric and Surface measures
        if 1:
            gc.collect() # returns number of unreachable objects collected by GC
            t0 = time.time()
            if 1:
                # print ('\n - Calling remove_smaller_components()\n')
                patient_pred_postprocessed = losses.remove_smaller_components(patient_gt, patient_pred_vals, meta=patient_id_curr)
            else:
                patient_pred_postprocessed = np.array(patient_pred_vals)
                thresh = 0.5
                patient_pred_postprocessed[patient_pred_postprocessed > thresh] = 1
                patient_pred_postprocessed[patient_pred_postprocessed <= thresh] = 0

            if verbose: print (' - [eval_3D()] Post-Process time    : ', time.time() - t0,'s')
            
            # Step 3.1.2 - Loss Calculation
            spacing = np.array([meta1_batch[4], meta1_batch[5], meta1_batch[6]])/100.0
            try:
                if verbose: t0 = time.time()
                loss_avg_val, loss_labels_val           = losses.dice_numpy(patient_gt, patient_pred_postprocessed)
                hausdorff_avg_val, hausdorff_labels_val, hausdorff95_avg_val, hausdorff95_labels_val, msd_avg_val, msd_labels_vals = losses.get_surface_distances(patient_gt, patient_pred_postprocessed, spacing, meta=patient_id_curr)
                # if verbose:
                if 0:
                    print ('\n - patient_id_curr : ', patient_id_curr)
                    print (' - [eval_3D()] DICE  : ', ['%.4f' % (each) for each in loss_labels_val])
                    print (' - [eval_3D()] HD95  : ', ['%.4f' % (each) for each in hausdorff95_labels_val])
                    print ('')
                    # pdb.set_trace()

                if loss_avg_val != -1 and len(loss_labels_val):
                    res[patient_id_curr] = {
                        config.KEY_DICE_AVG      : loss_avg_val
                        , config.KEY_DICE_LABELS : loss_labels_val
                        , config.KEY_HD_AVG      : hausdorff_avg_val
                        , config.KEY_HD_LABELS   : hausdorff_labels_val 
                        , config.KEY_HD95_AVG    : hausdorff95_avg_val
                        , config.KEY_HD95_LABELS : hausdorff95_labels_val
                        , config.KEY_MSD_AVG     : msd_avg_val
                        , config.KEY_MSD_LABELS  : msd_labels_vals
                    }
                else:
                    print (' - [ERROR][eval_3D()] patient_id: ', patient_id_curr)
                if verbose: print (' - [eval_3D()] Loss calculation time: ', time.time() - t0,'s')
            except:
                traceback.print_exc()

        # Step 3.1.3 - ECE calculation
        if 1:
            if verbose: t0 = time.time()
            if 1:
                ece_global_obj.update_state(y_true=patient_gt, y_pred=patient_pred_vals, patient_id=patient_id_curr, verbose=False)
            else:
                # print (' - [eval_3D()] Skipping CE calculation')
                pass
            if verbose: print (' - [eval_3D()] ECE time             : ', time.time() - t0,'s')

        # Step 3.1.5 - Save/Visualize
        if 1:
            if not deepsup_eval:
                if verbose: t0 = time.time()
                eval_3D_finalize(exp_name, patient_img, patient_gt, patient_pred_postprocessed, patient_pred_vals, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc, []
                    , patient_id_curr
                    , model_folder_epoch_imgs, model_folder_epoch_patches 
                    , loss_labels_val, hausdorff_labels_val, hausdorff95_labels_val, msd_labels_vals
                    , spacing, label_map, label_colors
                    , show=show, save=save)
                if verbose: print (' - [eval_3D()] Save as .nrrd time   : ', time.time() - t0,'s')

        if verbose: print (' - [eval_3D()] Total patient time   : ', time.time() - t99,'s')
            
        # Step 3.1.6
        del patient_img
        del patient_gt
        del patient_pred_vals
        del patient_pred_std
        del patient_pred_ent
        del patient_pred_postprocessed
        del patient_pred_mif
        del patient_pred_unc
        gc.collect()

        return res, ece_global_obj
    
    except:
        traceback.print_exc()
        return res, ece_global_obj

def eval_3D_get_outputs(model, X, Y, training_bool, MC_RUNS, UNC_TYPES, deepsup, deepsup_eval, save):

    # Step 0 - Init
    mc_runs_fail = False
    if MC_RUNS is None:MC_RUNS = 1

    # Step 2 - Run Monte-Carlo predictions
    try:
        tic_mcruns  = time.time()
        if deepsup:
            if deepsup_eval:
                y_predict = tf.stack([model(X, training=training_bool)[0] for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time
                X = X[:,::2,::2,::2,:]
                Y = Y[:,::2,::2,::2,:]
            else:
                y_predict = tf.stack([model(X, training=training_bool)[1] for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time    
        else:
            y_predict = tf.stack([model(X, training=training_bool) for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time
        toc_mcruns        = time.time()
        mc_runs_fail = False
    except tf.errors.ResourceExhaustedError as e:
        print ('\n - [eval_3D_get_outputs()] OOM error for MC_RUNS={} || {}'.format(MC_RUNS, utils.get_nvitop_gpu_memory()))
        mc_runs_fail = True
    
    try:
        MC_RUNS     = 5
        tic_mcruns  = time.time()
        if deepsup:
            if deepsup_eval:
                y_predict = tf.stack([model(X, training=training_bool)[0] for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time
                X = X[:,::2,::2,::2,:]
                Y = Y[:,::2,::2,::2,:]
            else:
                y_predict = tf.stack([model(X, training=training_bool)[1] for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time    
        else:
            y_predict = tf.stack([model(X, training=training_bool) for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time
        toc_mcruns        = time.time()
        mc_runs_fail = False
    except tf.errors.ResourceExhaustedError as e:
        print (' - [eval_3D_get_outputs()] OOM error for MC_RUNS=5 || ', utils.get_nvitop_gpu_memory())
        mc_runs_fail = True

        
    # Step 3 - Calculate different metrics
    if save:
        if config.KEY_MIF in UNC_TYPES:
            y_predict_mif     = y_predict * tf.math.log(y_predict + config._EPSILON)               # [MC,B,H,W,D,C]
            y_predict_mif     = tf.math.reduce_sum(y_predict_mif, axis=[0,-1])/MC_RUNS      # [MC,B,H,W,D,C] -> [B,H,W,D]
        else:
            y_predict_mif = []
        
        if config.KEY_STD in UNC_TYPES:
            y_predict_std     = tf.math.reduce_std(y_predict, axis=0)                       # [MC,B,H,W,D,C] -> [B,H,W,D,C]
            # we will later just do a max of this
        else:
            y_predict_std     = []
        
        if config.KEY_PERC in UNC_TYPES:
            y_predict_perc    = tfp.stats.percentile(y_predict, q=[30,70], axis=0, interpolation='nearest')
            y_predict_unc     = y_predict_perc[1] - y_predict_perc[0]
            del y_predict_perc
            gc.collect()
        else:
            y_predict_unc     = []
        
        y_predict         = tf.math.reduce_mean(y_predict, axis=0)    

        if config.KEY_ENT in UNC_TYPES:
            y_predict_ent     = -1*tf.math.reduce_sum(y_predict * tf.math.log(y_predict + config._EPSILON), axis=-1) # [B,H,W,D,C] -> # [B,H,W,D] ent = -p.log(p)
            if config.KEY_MIF in UNC_TYPES:
                y_predict_mif     = y_predict_ent + y_predict_mif                                             # [B,H,W,D] + [B,H,W,D] = [B,H,W,D]; MI = ent + expectation(ent)
        else:
            y_predict_ent     = []
            y_predict_mif     = []

    else:
        y_predict         = tf.math.reduce_mean(y_predict, axis=0)
        y_predict_std = []
        y_predict_ent = []
        y_predict_mif = []
        y_predict_unc = []

    return np.array(Y), np.array(y_predict), np.array(y_predict_std), np.array(y_predict_ent), np.array(y_predict_mif), np.array(y_predict_unc), toc_mcruns-tic_mcruns

def eval_3D(model, dataset_eval, dataset_eval_gen, params, show=False, save=False, verbose=False):
    
    try:

        # Step 0.0 - Variables under debugging
        pass
        
        # Step 0.1 - Extract params
        PROJECT_DIR = params['PROJECT_DIR']
        exp_name    = params['exp_name']
        eval_type   = params['eval_type']
        batch_size  = params['batch_size']
        epoch       = params['epoch']
        deepsup      = params['deepsup']
        deepsup_eval = params['deepsup_eval']
        label_map    = dict(dataset_eval.LABEL_MAP) # {'Prostate': 1}
        label_colors = dict(dataset_eval.LABEL_COLORS)
        mem_before = utils.get_memory(config.PID)

        print ('\n\n --------------------- eval_3D({}) ({}) --------------------- \n\n'.format(eval_type, utils.get_nvitop_gpu_memory()))
        print ('\n - [eval_3d] show: ', show, ' || save: ', save)
        
        # Step 0.3 - Init temp variables
        model_folder_epoch_patches = None
        model_folder_epoch_imgs    = None
        
        mc_runs       = params.get(config.KEY_MC_RUNS, None)
        training_bool = params.get(config.KEY_TRAINING_BOOL, None)
        model_folder_epoch_patches, model_folder_epoch_imgs = utils.get_eval_folders(PROJECT_DIR, exp_name, epoch, eval_type, mc_runs, training_bool, create=True)
        
        # Step 0.2 - Init results array
        res                = {}
        DILATION_STYLE = params.get('DILATION_STYLE', config.DILATION_TRUE_PRED)
        DILATION_KSIZE = params.get('DILATION_KSIZE', (3,3,2)) # (10,10,3)
        ERROR_KSIZE    = params.get('ERROR_KSIZE', (3,3,1))
        if 1:
            ece_global_obj     = utils.MedSegCalibrationError([config.CALIBRATION_ECE, config.CALIBRATION_SCE, config.CALIBRATION_ACE, config.CALIBRATION_ThACE], list(label_map.values())
                                        , dilation_ksize=DILATION_KSIZE, error_ksize=ERROR_KSIZE, path_model=model_folder_epoch_patches, num_bins=10, dilation_style=DILATION_STYLE, mode=eval_type)
        else:
            ece_global_obj = None # [TODO: Check if this is a memory hog]
        
        patient_grid_count = {}

        # Step 0.4 - Debug vars
        ttotal,t0, t99         = time.time(), None, None
        times_mcruns           = []
        
        # Step 1 - Loop over dataset_eval (which provides patients & grids in an ordered manner)
        print ('')
        model.trainable  = False
        pbar_desc_prefix = 'Eval3D_{} [batch={}]'.format(eval_type, batch_size)
        training_bool    = params.get('training_bool',True) # [True, False]
        first_run = True
        with tqdm.tqdm(total=len(dataset_eval), desc=pbar_desc_prefix, leave=False) as pbar_eval:
            for (X,Y,meta1,meta2) in dataset_eval_gen.repeat(1): # X=[B,H,W,D,1]
                
                # Step 1.0 - Warm up model
                if first_run:
                    _ = model(X, training=training_bool)
                    first_run = False

                # Step 1.1 - Get MC results
                MC_RUNS   = params.get(config.KEY_MC_RUNS,10)
                UNC_TYPES = params.get(config.KEY_UNC_TYPES,[])
                Y, y_predict, y_predict_std, y_predict_ent, y_predict_mif, y_predict_unc, mcruns_time = eval_3D_get_outputs(model, X, Y, training_bool, MC_RUNS, UNC_TYPES, deepsup, deepsup_eval, save)
                times_mcruns.append(mcruns_time)
                # pdb.set_trace()
                # dataset_eval.plot(X, y_predict, meta2, slice_id=14, binary_mask=True)

                for batch_id in range(X.shape[0]):

                    # Step 2 - Get meta info
                    meta1_batch = meta1[batch_id].numpy()
                    patient_id = meta2[batch_id].numpy().decode('utf-8')
                    
                    if save:
                        if len(y_predict_std): y_predict_std_ = y_predict_std[batch_id]
                        else                 : y_predict_std_ = []
                        if len(y_predict_mif): y_predict_mif_ = y_predict_mif[batch_id]
                        else                 : y_predict_mif_ = []
                        if len(y_predict_ent): y_predict_ent_ = y_predict_ent[batch_id]
                        else                 : y_predict_ent_ = []
                       

                        res, ece_global_obj = eval_3D_process_outputs(res, ece_global_obj, patient_id, meta1_batch, np.array(X[batch_id,:,:,:,0]), Y[batch_id]
                                                                        , y_predict[batch_id], y_predict_std_, y_predict_ent_, y_predict_mif_, y_predict_unc
                                                                        , deepsup_eval, model_folder_epoch_imgs, model_folder_epoch_patches
                                                                        , label_map, label_colors
                                                                        , t99, show=show, save=save
                                                                        , verbose=verbose)
                    else:
                        res, ece_global_obj = eval_3D_process_outputs(res, ece_global_obj, patient_id, meta1_batch, np.array(X[batch_id,:,:,:,0]), Y[batch_id]
                                                                        , y_predict[batch_id], y_predict_std, y_predict_ent, y_predict_mif, y_predict_unc
                                                                        , deepsup_eval, model_folder_epoch_imgs, model_folder_epoch_patches
                                                                        , label_map, label_colors
                                                                        , t99, show=show, save=save
                                                                        , verbose=verbose)

                    
                pbar_eval.update(batch_size)
                memory = pbar_desc_prefix + ' [' +  utils.get_memory(config.PID) + '][' + utils.get_nvitop_gpu_memory() + ']'
                pbar_eval.set_description(desc=memory, refresh=True)

        obj = eval_3D_summarize(res, ece_global_obj, model, eval_type, deepsup_eval, label_map, model_folder_epoch_patches
                                    , times_mcruns, MC_RUNS, ttotal, save=save, show=show, verbose=verbose)
        
        mem_after = utils.get_memory(config.PID)
        print ('  --- Memory Consumption: ({})  -> ({})'.format(mem_before, mem_after))
        print ('  --- res: ', utils.get_actualsize(res))
        print ('  --- ece_global_obj: ', utils.get_actualsize(ece_global_obj))

        del ece_global_obj
        mem_before = utils.get_memory(config.PID)
        gc_n = gc.collect()
        mem_after = utils.get_memory(config.PID)
        print ('  --- Unreachable objects collected by GC: {} || ({}) -> ({})'.format(gc_n, mem_before, mem_after))
        print ('  --- Mem: {} || {}'.format(utils.get_memory(config.PID), utils.get_nvitop_gpu_memory()) )

        return obj

    except:
        traceback.print_exc()
        model.trainable = True
        return -1, {} 

def val(model, dataset, params, show=False, save=False, verbose=False):

    try:
        
        # Step 1 - Load Model

        load_model_params = {'PROJECT_DIR': params['PROJECT_DIR']
                                , 'exp_name': params['exp_name']
                                , 'load_epoch': params['epoch']
                                , 'optimizer': tf.keras.optimizers.Adam()
                            }
        if 1:
            init_size = ((1,200,200,28,1))
            X_tmp = tf.random.normal(init_size) # if the final dataloader does not have the same input size, the weight initialization gets screwed up. 
            _ = model(X_tmp)

        utils.load_model(model, load_type=config.MODE_VAL, params=load_model_params)

        if params['training_bool'] is False:
            
            try:
                bool_flipout = False
                for layer in model.layers:
                    for submodule in layer.submodules:
                        if type(submodule) == tfp.layers.Convolution3DFlipout:
                            bool_flipout = True
                            break
                
                if bool_flipout:
                    random_input = tf.random.normal((1,200,200,28,1))
                    y_predict_1 = model(random_input, training=params['training_bool'])
                    y_predict_2 = model(random_input, training=params['training_bool'])
                    print ('\n ==================================================================== ')
                    print ('\n - [train.py][val()] Setting all FlipOut std as 0')
                    print (' ==================================================================== \n ')
                    # print (' - Currently y_predict_1==y_predict_2: ', np.all(y_predict_1 == y_predict_2))

                    
                    for layer in model.layers:
                        for submodule in layer.submodules:
                            if type(submodule) == tfp.layers.Convolution3DFlipout: # kernel=N(loc,scale) --> N=Normal distro
                                
                                scale_init = np.array(submodule.kernel_posterior.distribution.scale, copy=True)
                                weights = submodule.get_weights() # [scale, rho, bias] --> kernel=N(loc,scale=tfp.bijectors.Softplus(rho)) --> output=input*kernel + bias
                                weights[1] = np.full(weights[1].shape, -np.inf) 
                                submodule.set_weights(weights)
                                scale_now = np.array(submodule.kernel_posterior.distribution.scale, copy=True)
                                # print (' --- scale_init==scale_now: ', np.all(scale_init == scale_now))
                                # for each in submodule.get_weights(): print (each.shape, np.sum(each))

                    y_predict_1 = model(random_input, training=params['training_bool'])
                    y_predict_2 = model(random_input, training=params['training_bool'])
                    # print (' - Afterwards y_predict_1==y_predict_1: ', np.all(y_predict_1 == y_predict_2))
                    # pdb.set_trace()

            except:
                traceback.print_exc()
                pdb.set_trace()

        print ('')
        print (' - [pros_train.py][val()] Model({}) Loaded for {} at epoch-{} (validation purposes) !'.format(str(model), params['exp_name'], params['epoch']))
        print ('')
        print (' - [val()] dataset.transforms: ', dataset.transforms)

        # Step 3 - Calculate losses
        dataset_gen = dataset.generator().batch(params['batch_size']).prefetch(params['prefetch_batch'])
        loss_avg, loss_labels_avg = eval_3D(model, dataset, dataset_gen, params, show=show, save=save, verbose=verbose)
        
        print ('\n')
        mem_before = utils.get_memory(config.PID)
        gc_n = gc.collect()
        mem_after = utils.get_memory(config.PID)
        print(' ---- Unreachable objects collected by GC: {} || ({}) -> ({})'.format(gc_n, mem_before, mem_after))

    except:
        traceback.print_exc()
        pdb.set_trace()

############################################################
#                         TRAINER                          #
############################################################
class Trainer:

    def __init__(self, params):

        # Init
        self.params = params

        # Print
        self._train_preprint()

        # Random Seeds
        self._set_seed()

        # Set the dataloaders
        self._set_dataloaders()

        # Set the model
        self._set_model()

        # Set Metrics
        self._set_metrics()

        # Other flags
        self.write_model_done = False
    
    def _train_preprint(self):
        print ('')
        print (' -------------- {}    ({})'.format(self.params['exp_name'], str(datetime.datetime.now())))
        
        print ('')
        print (' DATALOADER ')
        print (' ---------- ')
        print (' - train: ', self.params['dataloader']['train'])
        print (' - test : ', self.params['dataloader']['test'])
        print (' -- batch_size: ', self.params['dataloader']['batch_size'])
        print ('  -- prefetch_batch : ', self.params['dataloader']['prefetch_batch'])
        print ('  -- parallel_calls : ', self.params['dataloader']['parallel_calls'])
        print ('  -- shuffle        : ', self.params['dataloader']['shuffle'])

        print (' -- single_patient: ', self.params['dataloader']['single_patient'])
        if self.params['dataloader']['single_patient']:
            print (' !!!!!!!!!!!!!!!!!!! SINGLE PATIENT !!!!!!!!!!!!!!!!!!!')
            print ('')

        print ('')
        print (' MODEL ')
        print (' ----- ')
        print (' - Model: ', str(self.params['model']['name']))
        print (' -- KL Schedule  : ', self.params['model']['kl_schedule'])
        print (' -- KL Alpha Init: ', self.params['model']['kl_alpha_init'])
        print (' -- MC Runs      : ', self.params['model']['mc_runs'])
        print (' -- Unc Metric   : ', self.params['model']['unc_metric'])

        print (' -- Model TBoard : ', self.params['model']['model_tboard'])
        print (' -- Profiler     : ', self.params['model']['profiler']['profile'])
        if self.params['model']['profiler']['profile']:
            print (' ---- Profiler Epochs: ', self.params['model']['profiler']['epochs'])
            print (' ---- Step Per Epochs: ', self.params['model']['profiler']['steps_per_epoch'])
        print (' - Optimizer: ', str(self.params['model']['optimizer']))
        print (' -- Init LR        : ', self.params['model']['init_lr'])
        print (' -- Fixed LR       : ', self.params['model']['fixed_lr'])
        print (' -- Grad Persistent: ', self.params['model']['grad_persistent'])
        if self.params['model']['grad_persistent']:
            print (' !!!!!!!!!!!!!!!!!!! GRAD PERSISTENT !!!!!!!!!!!!!!!!!!!')
            print ('')
        print (' - Epochs: ', self.params['model']['epochs'])
        print (' -- Save   : every {} epochs'.format(self.params['model']['epochs_save']))
        print (' -- Eval3D : every {} epochs '.format(self.params['model']['epochs_eval']))
        print (' -- Viz3D  : every {} epochs '.format(self.params['model']['epochs_viz']))

        print ('')
        print (' METRICS ')
        print (' ------- ')
        print (' - Logging-TBoard: ', self.params['metrics']['logging_tboard'])
        if not self.params['metrics']['logging_tboard']:
            print (' !!!!!!!!!!!!!!!!!!! NO LOGGING-TBOARD !!!!!!!!!!!!!!!!!!!')
            print ('')
        print (' - Eval: ', self.params['metrics']['metrics_eval'])
        print (' - Loss: ', self.params['metrics']['metrics_loss'])
        print (' -- Type of Loss  : ', self.params['metrics']['loss_type'])
        print (' -- Weighted Loss : ', self.params['metrics']['loss_weighted'])
        print (' -- Masked Loss   : ', self.params['metrics']['loss_mask'])
        print (' -- Combo         : ', self.params['metrics']['loss_combo'])
        print (' -- Loss Epoch    : ', self.params['metrics']['loss_epoch'])
        print (' -- Loss Rate     : ', self.params['metrics']['loss_rate'])

        print ('')
        print (' DEVOPS ')
        print (' ------ ')
        print (' - OS-PID: ', config.PID)
        print (' - Seed: ', self.params['random_seed'])

        print ('')
    
    def _set_seed(self):
        np.random.seed(self.params['random_seed'])
        tf.random.set_seed(self.params['random_seed'])
    
    def _set_dataloaders(self):

        # Params - Directories
        data_dir = self.params['dataloader']['data_dir']
        train_dataset = self.params['dataloader']['train']
        test_dataset  = self.params['dataloader']['test']
        
        # Params - Dataloader        
        batch_size     = self.params['dataloader']['batch_size']
        prefetch_batch = self.params['dataloader']['prefetch_batch']
        parallel_calls = self.params['dataloader']['parallel_calls']
        shuffle_size   = self.params['dataloader']['shuffle']
        
        # Params - Debug
        single_patient  = self.params['dataloader']['single_patient']

        # Define datasets
        self.dataset_prostatex = ProstateXDataset(path_data=data_dir
                                                , parallel_calls=parallel_calls
                                                , single_patient=single_patient
                                                )

        self.dataset_promise12 = Promise12Dataset(path_data=data_dir
                                                , parallel_calls=parallel_calls
                                                , single_patient=single_patient
                                                )

        # Define train/test datasets
        if train_dataset == config.DATASET_PROSTATEX:
            self.dataset_train = self.dataset_prostatex
            self.dataset_test  = self.dataset_promise12
        elif train_dataset == config.DATASET_PROMISE12:
            self.dataset_train = self.dataset_promise12
            self.dataset_test  = self.dataset_prostatex
        
        # Define transforms
        transforms = [
            Aug.MinMaxNormalizer()
            , Aug.Translate(label_map=self.dataset_prostatex.LABEL_MAP, translations=[40,40], prob=0.5)
            , Aug.Rotate3D(label_map=self.dataset_prostatex.LABEL_MAP, angle_degrees=15, prob=0.5)
            , Aug.Noise(x_shape=[self.dataset_prostatex.VOXELS_X_MIDPT*2, self.dataset_prostatex.VOXELS_Y_MIDPT*2, self.dataset_prostatex.VOXELS_Z_MAX,1], prob=0.1)
            , Aug.Mask(label_map=self.dataset_prostatex.LABEL_MAP, img_size=(self.dataset_prostatex.VOXELS_X_MIDPT*2, self.dataset_prostatex.VOXELS_Y_MIDPT*2, self.dataset_prostatex.VOXELS_Z_MAX,1), mask_size=(50,50), prob=0.5)
            , Aug.Equalize(prob=0.5)
            , Aug.ImageQuality(prob=0.5)
            , Aug.Shadow(prob=0.5)
        ]
        
        self.dataset_train.transforms = transforms
        self.dataset_test.transforms = [Aug.MinMaxNormalizer()]
        print (' - [Trainer][_set_dataloaders()] train: ', self.dataset_train.transforms)
        print (' - [Trainer][_set_dataloaders()] test: ', self.dataset_test.transforms)

        # Define generators
        self.dataset_train_gen = self.dataset_train.generator().repeat().shuffle(shuffle_size).batch(batch_size).apply(tf.data.experimental.prefetch_to_device(device='/GPU:0', buffer_size=prefetch_batch))
        self.dataset_train_eval_gen = self.dataset_train.generator().batch(2).prefetch(prefetch_batch)
        self.dataset_test_gen  = self.dataset_test.generator().batch(2).prefetch(prefetch_batch)

        # Define other vars
        self.label_map = dict(self.dataset_train.LABEL_MAP)
        self.label_ids = self.label_map.values()
        self.params['internal'] = {}
        self.params['internal']['label_map'] = self.label_map # for use in Metrics
        self.params['internal']['label_ids'] = self.label_ids
    
    def set_lr(self, epoch, init_lr):
        
        try:
            # if epoch == 1: # for models that are preloaded from another model
            #     print (' - [set_lr()] Setting optimizer lr to ', init_lr)
            #     optimizer.lr.assign(init_lr)

            if epoch > 1 and epoch % 20 == 0:
                tf.print(' - [Trainer][set_lr()] epoch > 1 and epoch % 20 == 0')
                self.optimizer.lr.assign(self.optimizer.lr * 0.98)
            # if epoch > 1 and epoch % 2 == 0:
            #     tf.print(' - [Trainer][set_lr()] epoch > 1 and epoch % 2 == 0')
            #     self.optimizer.lr.assign(self.optimizer.lr * 0.98)
        except:
            traceback.print_exc()
            
    def _set_model(self):

        # Step 1 - Get class ids
        class_count = len(self.label_ids)
        deepsup     = self.params['model']['deepsup']
        
        # Step 2 - Get model arch
        self.kl_schedule = self.params['model']['kl_schedule']
        if self.kl_schedule == config.KL_DIV_FIXED:
            self.kl_alpha_init = self.params['model']['kl_alpha_init']
        elif self.kl_schedule == config.KL_DIV_ANNEALING:
            pass
        
        if self.params['model']['name'] == config.MODEL_UNSURE_ORGANNET:
            print (' - [Trainer][_set_model()] OrganNet(hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=False)')
            self.model = models.OrganNet(class_count=class_count, hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=True, deepsup=deepsup)
        
        elif self.params['model']['name'] == config.MODEL_UNSURE_ORGANNET_TRIPLE:
            print (' - [Trainer][_set_model()] OrganNet(pooling=triple, hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=False)')
            self.model = models.OrganNet(class_count=class_count, pooling='triple', hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=True, deepsup=deepsup)
        
        elif self.params['model']['name'] == config.MODEL_UNSURE_ORGANNET_BAYESIAN:
            print (' - [Trainer][_set_model()] OrganNet(hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=False)')
            self.model = models.OrganNet(class_count=class_count, hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=True, deepsup=deepsup)
        
        elif self.params['model']['name'] == config.MODEL_UNSURE_ORGANNET_HEADBAYES_DOUBLE:
            print (' - [Trainer][_set_model()] OrganNet(hdc=True, bayesian=True, bayesianhead=True, dropout=None, activation=tf.nn.sigmoid, verbose=False)')
            self.model = models.OrganNet(class_count=class_count, hdc=True, bayesian=True, bayesianhead=True, dropout=None, activation=tf.nn.sigmoid, trainable=True, deepsup=deepsup)
        
        elif self.params['model']['name'] == config.MODEL_UNSURE_ORGANNET_BAYESIAN_TRIPLE:
            print (' - [Trainer][_set_model()] OrganNet(pooling=triple, hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=False)')
            self.model = models.OrganNet(class_count=class_count, pooling='triple', hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=True, deepsup=deepsup)
        
        elif self.params['model']['name'] == config.MODEL_UNSURE_ORGANNET_HEADBAYES_TRIPLE:
            print (' - [Trainer][_set_model()] OrganNet(pooling=triple, hdc=True, bayesian=True, bayesianhead=True, dropout=None, activation=tf.nn.sigmoid, verbose=False)')
            self.model = models.OrganNet(class_count=class_count, pooling='triple', hdc=True, bayesian=True, bayesianhead=True, dropout=None, activation=tf.nn.sigmoid, trainable=True, deepsup=deepsup)
        

        # Step 3 - Get optimizer
        if self.params['model']['optimizer'] == config.OPTIMIZER_ADAM:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['model']['init_lr'])

        # Step 4 - Load model if needed
        epochs = self.params['model']['epochs']
        if not self.params['model']['load_model']['load']:  
            # Step 4.1 - Set epoch range under non-loading situations
            self.epoch_range = range(1,epochs+1)
        else:

            # Step 4.2.1 - Some model-loading params
            load_epoch        = self.params['model']['load_model']['load_epoch']
            load_exp_name     = self.params['model']['load_model']['load_exp_name']
            load_optimizer_lr = self.params['model']['load_model']['load_optimizer_lr']
            load_model_params = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'load_epoch': load_epoch, 'optimizer':self.optimizer}
            
            print ('')
            print (' - [Trainer][_set_model()] Loading pretrained model')
            print (' - [Trainer][_set_model()] Model: ', self.model)
            print (' - [Trainer][_set_model()] load_exp_name: ', load_exp_name)
            
            # Step 4.2.2.1 - If loading is done from the same exp_name
            if load_exp_name is None:
                load_model_params['exp_name'] = exp_name
                self.epoch_range = range(load_epoch+1, epochs+1)
                print (' - [Trainer][_set_model()] Training from epoch:{} to {}'.format(load_epoch, epochs))
            # Step 4.2.2.1 - If loading is done from another exp_name
            else:
                self.epoch_range = range(1, epochs+1)
                load_model_params['exp_name'] = load_exp_name
                print (' - [Trainer][_set_model()] Training from epoch:{} to {}'.format(1, epochs))

            print (' - [Trainer][_set_model()] exp_name: ', load_model_params['exp_name'])

            # Step 4.3 - Finally, load model from the checkpoint
            utils.load_model(self.model, load_type=config.MODE_TRAIN, params=load_model_params)
            print (' - [Trainer][_set_model()] Model Loaded at epoch-{} !'.format(load_epoch))
            print (' -- [Trainer][_set_model()] Optimizer.lr : ', self.optimizer.lr.numpy())
            if load_optimizer_lr is not None:
                self.optimizer.lr.assign(load_optimizer_lr)
                print (' -- [Trainer][_set_model()] Optimizer.lr : ', self.optimizer.lr.numpy())
        
        # Step 5 - Creae model weights
        init_size = ((1,200,200,28,1))
        print ('\n -- [Trainer][_set_model()] Model weight creation with ', init_size, '\n')
        X_tmp = tf.random.normal(init_size) # if the final dataloader does not have the same input size, the weight initialization gets screwed up. 
        _ = self.model(X_tmp)
        self.layers_kl = self.get_layers_kl()        
        print (' -- [Trainer][_set_model()] Created model weights ')
        try:
            print (' --------------------------------------- ')
            print (self.model.summary(line_length=150))
            print (' --------------------------------------- ')
            count = 0
            for var in self.model.trainable_variables:
                print (' - var: ', var.name)
                count += 1
                if count > 20:
                    print (' ... ')
                    break 
            
            print ('\n --------------------------------------- ')
            print (' - model.losses: ', self.model.losses)
            print (' --------------------------------------- ')

        except:
            print (' - [Trainer][_set_model()] model.summary() failed')
            pass
    
    def _set_metrics(self):
        
        self.metrics = {}
        self.metrics[config.MODE_TRAIN] = ModelMetrics(metric_type=config.MODE_TRAIN, params=self.params)
        self.metrics[config.MODE_TEST]  = ModelMetrics(metric_type=config.MODE_TEST, params=self.params)

        deepsup = self.params['model']['deepsup']
        if deepsup:
            self.metrics[config.MODE_TRAIN_DEEPSUP] = ModelMetrics(metric_type=config.MODE_TRAIN_DEEPSUP, params=self.params)
            self.metrics[config.MODE_TEST_DEEPSUP] = ModelMetrics(metric_type=config.MODE_TEST_DEEPSUP, params=self.params)

    def _set_profiler(self, epoch, epoch_step):
        exp_name = self.params['exp_name']

        if self.params['model']['profiler']['profile']:
            if epoch in self.params['model']['profiler']['epochs']:
                if epoch_step == self.params['model']['profiler']['starting_step']:
                    self.logdir = Path(config.MODEL_CHKPOINT_MAINFOLDER).joinpath(exp_name, config.MODEL_LOGS_FOLDERNAME, 'profiler', str(epoch))
                    tf.profiler.experimental.start(str(self.logdir))
                    print (' - tf.profiler.experimental.start(logdir)')
                    print ('')
                elif epoch_step == self.params['model']['profiler']['starting_step'] + self.params['model']['profiler']['steps_per_epoch']:
                    print (' - tf.profiler.experimental.stop()')
                    tf.profiler.experimental.stop()
                    print ('')
    
    @tf.function
    def get_layers_kl(self):
        
        res = {}
        for layer in self.model.layers:
            if len(layer.losses):

                flipout_layers = [submodule for submodule in layer.submodules if type(submodule) == tfp.layers.Convolution3DFlipout] # kernel=N(loc,scale) --> N=Normal distro
                for loss_id, loss in enumerate(layer.losses): # only present in FlipOut layers 
                    layer_name = layer.name + '_' + str(loss_id)
                    res[layer_name] = {
                        'kl'    : loss
                        , 'mean': flipout_layers[loss_id].kernel_posterior.distribution.loc
                        , 'std' : flipout_layers[loss_id].kernel_posterior.distribution.scale
                    }

        if len(res) == 0:
            print ('\n - [INFO][Trainer][get_layers_kl()] No bayesian layers found \n')

        return res
    
    @tf.function
    def _train_loss(self, Y, y_predict, y_unc, meta1, epoch, mode):

        # Step 1 - Init params
        trainMetrics  = self.metrics[mode]
        metrics_loss  = self.params['metrics']['metrics_loss']
        loss_weighted = self.params['metrics']['loss_weighted']
        loss_combo    = self.params['metrics']['loss_combo']
        loss_epoch    = self.params['metrics']['loss_epoch']
        loss_rate     = self.params['metrics']['loss_rate']

        label_weights = tf.cast([], dtype=tf.float32)

        loss_vals = tf.cast(0.0, dtype=tf.float32)
        mask      = tf.cast(tf.ones((2,1)), dtype=tf.float32)

        inf_flag = False
        nan_flag = False
        
        # Step 2 - Loop over losses
        for metric_str in metrics_loss:
            
            weights = []
            if loss_weighted[metric_str]:
                weights = label_weights
            
            loss_epoch_metric = float(loss_epoch[metric_str])
            
            # Step 2.1 - Only calculate loss if conditions allow it.
            if epoch > loss_epoch_metric:

                if metrics_loss[metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL, config.LOSS_PAVPU, config.LOSS_CE_BOUNDARY]:   
                    
                    # Step 3 - Calculate loss
                    if metrics_loss[metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_FOCAL]:
                        loss_val_train, loss_labellist_train, metric_val_report, metric_labellist_report = trainMetrics.losses_obj[metric_str](Y, y_predict, label_mask=mask, weights=weights)
                    elif metrics_loss[metric_str] in [config.LOSS_PAVPU]:
                        loss_val_train, loss_labellist_train, metric_val_report, metric_labellist_report = trainMetrics.losses_obj[metric_str](Y, y_predict, y_pred_unc=y_unc, label_mask=mask, weights=weights)
                    else:
                        tf.print(' - [ERROR][Trainer][_train_loss()] Unknown loss: ', metrics_loss[metric_str])
                        loss_val_train = tf.constant(0.0, dtype=tf.float32)
                        loss_labellist_train = []
                        metric_labellist_report = []

                    
                    # Step 3.1 - Check for nan
                    nan_list = tf.math.is_nan(loss_labellist_train)
                    nan_val  = tf.math.is_nan(loss_val_train)
                    inf_list = tf.math.is_inf(loss_labellist_train)
                    inf_val  = tf.math.is_inf(loss_val_train)
                    if nan_val or tf.math.reduce_any(nan_list):
                        nan_flag = True
                        tf.print ('\n - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || nan_list: ', nan_list, ' || nan_val: ', nan_val, summarize=-1)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || mask: ', mask, summarize=-1)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || loss_vals: ', loss_vals, summarize=-1)
                    elif inf_val or tf.math.reduce_any(inf_list):
                        inf_flag = True
                        tf.print ('\n - [ERROR][Trainer][_train_loss()] Loss Inf spotted: ', metric_str, ' || loss_val_train: ', loss_val_train, summarize=-1)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss Inf spotted: ', metric_str, ' || inf_list: ', inf_list, ' || inf_val: ', inf_val, summarize=-1)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss Inf spotted: ', metric_str, ' || mask: ', mask, summarize=-1)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || loss_vals: ', loss_vals, summarize=-1)
                    else:
                        
                        # Step 4 - Update in tensorboard
                        if len(metric_labellist_report):
                            trainMetrics.update_metric_loss_labels(metric_str, metric_labellist_report) # in sub-3D settings, this value is only indicative of performance
                        trainMetrics.update_metric_loss(metric_str, loss_val_train)

                        # Step 5.1 - Calculate loss factor
                        if loss_epoch_metric > 0.0:
                            loss_factor = tf.math.minimum(1.0, (epoch - loss_epoch_metric)/loss_rate[metric_str])
                        else:
                            loss_factor = 1.0
                        
                        # Step 5.2 - Update raw loss
                        loss_val_train = loss_val_train*loss_combo[metric_str]*loss_factor
                        # tf.print(' - loss: ', metric_str, ' || loss_val_train: ', loss_val_train)

                        # Step 6 - Add loss to final term
                        loss_vals = tf.math.add(loss_vals, loss_val_train)
                        

        # Step 99 - If any error issue, simply ignore            
        if nan_flag or inf_flag:
            tf.print (' - [ERROR][Trainer][_train_loss()] loss_vals:', loss_vals)
            loss_vals = 0.0 # no backprop when something was wrong

        return loss_vals
    
    @tf.function
    def _train_step(self, X, Y, meta1, meta2, kl_alpha, epoch):

        try:

            if 1:
                model           = self.model
                deepsup         = self.params['model']['deepsup']
                optimizer       = self.optimizer
                grad_persistent = self.params['model']['grad_persistent']
                trainMetrics    = self.metrics[config.MODE_TRAIN]
                kl_scale_fac    = self.params['model']['kl_scale_factor']
                mc_runs         = self.params['model']['mc_runs']  #[1,5,8]
                unc_metric      = self.params['model']['unc_metric'] # [config.KEY_ENT, config.KEY_MIF, config.KEY_STD]
                
                y_predict = None
                loss_vals = None
                gradients = None 

            # Step 1 - Calculate loss
            with tf.GradientTape(persistent=grad_persistent) as tape:
                
                loss_vals = 0

                
                print ('\n')

                # Step 1.2 - Deepsup or not?
                if deepsup: 
                    (y_predict_deepsup, y_predict) = model(X, training=True)
                else:
                    
                    print (' - [Trainer][_train_step()] mc_runs: ', mc_runs)
                    t2 = tf.timestamp()
                    y_predict     = tf.stack([model(X, training=True) for _ in range(mc_runs)]) # [MC,B,H,W,D,C=10]    
                    t2_ = tf.timestamp()
                    
                    if unc_metric is not None:
                        
                        # pdb.set_trace()
                        if unc_metric == config.KEY_ENT:
                            print (' - [Trainer][_train_step()] Calculating entropy with MC='+str(mc_runs))
                            y_predict     = tf.math.reduce_mean(y_predict, axis=0)                                                    # [MC,B,H,W,D,C] --> [B,H,W,D,C]
                            y_predict_ent = -1*tf.math.reduce_sum(y_predict * tf.math.log(y_predict + config._EPSILON), axis=-1)      # [B,H,W,D,C] -> # [B,H,W,D] ent = -p.log(p)
                            loss_vals = self._train_loss(Y, y_predict, y_predict_ent, meta1, epoch, mode=config.MODE_TRAIN)

                        elif unc_metric == config.KEY_MIF:
                            print (' - [Trainer][_train_step()] Calculating MI with MC='+str(mc_runs))
                            y_predict_mif_ = tf.math.reduce_sum(y_predict * tf.math.log(y_predict + config._EPSILON), axis=[0,-1])/mc_runs  # [MC,B,H,W,D,C] --> [B,H,W,D] # Note: aleatoric uncertainty
                            y_predict     = tf.math.reduce_mean(y_predict, axis=0)                                                          # [MC,B,H,W,D,C] --> [B,H,W,D,C]
                            y_predict_ent = -1*tf.math.reduce_sum(y_predict * tf.math.log(y_predict + config._EPSILON), axis=-1)            # [B,H,W,D,C] -> # [B,H,W,D] ent = -p.log(p)
                            y_predict_mif = y_predict_ent + y_predict_mif_
                            loss_vals = self._train_loss(Y, y_predict, y_predict_mif, meta1, epoch, mode=config.MODE_TRAIN)

                        elif unc_metric == config.KEY_STD:
                            print (' - [Trainer][_train_step()] Calculating std with MC='+str(mc_runs))
                            y_predict_std = tf.math.reduce_max(tf.math.reduce_std(y_predict, axis=0), axis=-1)                     # [MC,B,H,W,D,C] --> [B,H,W,D,C] --> [B,H,W,D]
                            y_predict     = tf.math.reduce_mean(y_predict, axis=0)                                                 # [MC,B,H,W,D,C] --> [B,H,W,D,C]
                            loss_vals = self._train_loss(Y, y_predict, y_predict_std, meta1, epoch, mode=config.MODE_TRAIN)
                    else:
                        print (' - [Trainer][_train_step()] Not doing uncertainty calculation while training')
                        y_predict = tf.math.reduce_mean(y_predict, axis=0) # [MC,B,H,W,D,C] --> [B,H,W,D,C]
                        loss_vals = self._train_loss(Y, y_predict, None, meta1, epoch, mode=config.MODE_TRAIN)                                      

                if loss_vals > 0:

                    if deepsup:
                        print (' - [Trainer][_train_step()] deepsup training')
                        Y_deepsup         = Y[:,::2,::2,::2,:]
                        loss_vals_deepsup = self._train_loss(Y_deepsup, y_predict_deepsup, None, meta1, epoch, mode=config.MODE_TRAIN_DEEPSUP)
                        loss_vals         += loss_vals_deepsup
                    
                    
                    if len(model.losses):
                        print (' - [Trainer][_train_step()] Model FlipOut (doping KL loss)')
                        kl           = tf.math.add_n(model.losses)
                        kl_loss      = kl*kl_alpha/kl_scale_fac
                        
                        kl_layers = {}
                        for layer in model.layers:
                            for loss_id, loss in enumerate(layer.losses):
                                layer_name = layer.name + '_' + str(loss_id)
                                kl_layers[layer_name] = {'kl': loss}
                        trainMetrics.update_metrics_kl(kl_alpha, kl, kl_layers)
                        trainMetrics.update_metrics_scalarloss(loss_vals, kl_loss)
                        if tf.random.uniform([]) < 0.01:
                            tf.print('\n - loss_vals: ', loss_vals, ' || kl_loss: ', kl_loss)
                        loss_vals    = loss_vals + kl_loss
                    else:
                        print (' - [Trainer][_train_step()] model.losses: ', model.losses)
                
            # Step 2 - Calculate gradients
            t3 = tf.timestamp()
            if not tf.math.reduce_any(tf.math.is_nan(loss_vals)) and loss_vals > 0:
                all_vars = model.trainable_variables

                gradients = tape.gradient(loss_vals, all_vars) # dL/dW
                
                # Step 3 - Apply gradients
                optimizer.apply_gradients(zip(gradients, all_vars))
            else:
                tf.print('\n ====================== [NaN Error] ====================== ')
                tf.print(' - [ERROR][Trainer][_train_step()] Loss NaN spotted || loss_vals: ', loss_vals, summarize=-1)
                tf.print(' - [ERROR][Trainer][_train_step()] meta2: ', meta2, ' || meta1: ', meta1, summarize=-1)

            t3_ = tf.timestamp()

            return t2_-t2, t3-t2_, t3_-t3 # time_predict, time_loss, time_backprop

        except tf.errors.ResourceExhaustedError as e:
            print (' - [ERROR][Trainer][_train_step()] OOM error')
            return None, None, None

        except:
            tf.print('\n ====================== [Some Error] ====================== ')
            tf.print(' - [ERROR][Trainer][_train_step()]  meta2: ', meta2, ' || meta1: ', meta1)
            traceback.print_exc()
            return None, None, None
    
    def train(self):

        # Global params
        exp_name = self.params['exp_name']

        # Dataloader params
        batch_size = self.params['dataloader']['batch_size']

        # Model/Training params
        fixed_lr          = self.params['model']['fixed_lr']
        init_lr           = self.params['model']['init_lr']
        max_epoch         = self.params['model']['epochs']
        epoch_range       = iter(self.epoch_range)
        epoch_length      = len(self.dataset_train)
        deepsup           = self.params['model']['deepsup']
        params_save_model = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'exp_name': exp_name, 'optimizer':self.optimizer}

        # Metrics params
        metrics_eval = self.params['metrics']['metrics_eval']
        trainMetrics = self.metrics[config.MODE_TRAIN]
        trainMetrics.init_metrics_layers_kl(self.params, self.layers_kl)
        trainMetricsDeepSup = None
        if deepsup: trainMetricsDeepSup = self.metrics[config.MODE_TRAIN_DEEPSUP]

        # Eval Params
        params_eval = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'exp_name': exp_name, 'pid': config.PID
                            , 'eval_type': config.MODE_TRAIN, 'batch_size': batch_size}
        
        # Viz params
        epochs_save = self.params['model']['epochs_save'] # for model saving 
        epochs_viz  = self.params['model']['epochs_viz']  # for model results viz
        epochs_eval = self.params['model']['epochs_eval'] # for model evaluation
        epochs_info = self.params['model']['epochs_info'] # for tboard info

        # KL Divergence Params
        kl_alpha = self.kl_alpha_init # [0.0, self.kl_alpha_init]
        self.params['model']['kl_scale_factor'] = epoch_length / batch_size
        print ('')
        print (' DATALOADER ')
        print (' ---------- ')
        print (' - epoch_length   : ', epoch_length)
        print (' - kl_scale_factor: ', self.params['model']['kl_scale_factor'])
        
        # Random vars
        t_start_time = time.time()

        # Tmp
        data_counter_obj = {}

        epoch = None
        try:
            
            epoch_step = 0
            pbar       = None
            t1         = time.time()
            for (X,Y,meta1,meta2) in self.dataset_train_gen:
                t1_ = time.time()
                data_counter_obj = utils.data_counter(data_counter_obj, meta2)
                
                try:
                    # Epoch starter code                        
                    if epoch_step == 0:

                        # Get Epoch
                        epoch = next(epoch_range)

                        # Metrics
                        trainMetrics.reset_metrics(self.params)

                        # LR
                        if not fixed_lr:
                            self.set_lr(epoch, init_lr)
                        self.model.trainable = True
                        
                        # Calculate kl_alpha (commented if alpha is fixed)
                        if self.kl_schedule == config.KL_DIV_ANNEALING:
                            if epoch > self.initial_epoch:
                                if epoch % self.kl_epochs_change == 0:
                                    kl_alpha = tf.math.minimum(self.kl_alpha_max, self.kl_alpha_init + (epoch - self.initial_epoch)/float(self.kl_epochs_change) * self.kl_alpha_increase_per_epoch)   

                        # Pretty print
                        print ('')
                        print (' ===== [{}] EPOCH:{}/{} (LR={:3f}, {}) =================='.format(exp_name, epoch, max_epoch, self.optimizer.lr.numpy(), utils.get_nvitop_gpu_memory()))
                        print ('  - kl_alpha: ', kl_alpha)

                        # Start a fresh pbar
                        pbar = tqdm.tqdm(total=epoch_length, desc='')

                    # Model Writing to tensorboard
                    if self.params['model']['model_tboard'] and self.write_model_done is False :
                        self.write_model_done = True 
                        utils.write_model_tboard(self.model, X, self.params)
                    
                    # Start/Stop Profiling (after dataloader is kicked off)
                    self._set_profiler(epoch, epoch_step)

                    # Calculate loss and gradients from them
                    time_predict, time_loss, time_backprop = self._train_step(X, Y, meta1, meta2, tf.constant(kl_alpha, dtype=tf.float32), tf.constant(epoch, dtype=tf.float32))

                    # Update metrics (time + eval + plots)
                    time_dataloader = t1_ - t1
                    trainMetrics.update_metrics_time(time_dataloader, time_predict, time_loss, time_backprop)    
                            
                    # Update looping stuff
                    epoch_step += batch_size
                    pbar.update(batch_size)
                    trainMetrics.update_pbar(pbar)
                    
                except:
                    utils.print_exp_name(exp_name + '-' + config.MODE_TRAIN, epoch)
                    params_save_model['epoch'] = epoch
                    utils.save_model(self.model, params_save_model)
                    traceback.print_exc()

                if epoch_step >= epoch_length:
                    
                    # Reset epoch-loop params
                    pbar.close()
                    epoch_step = 0

                    try:
                        # Model save
                        if epoch % epochs_save == 0:
                            params_save_model['epoch'] = epoch
                            utils.save_model(self.model, params_save_model)
                        
                        # Eval on full 3D
                        try:
                            if epoch % epochs_eval == 0:
                                self.params['epoch'] = epoch
                                save=False
                                if epoch > 0 and epoch % epochs_viz == 0:
                                    save=True

                                self.model.trainable = False
                                for metric_str in metrics_eval:
                                    if metrics_eval[metric_str] in [config.LOSS_DICE]:
                                        params_eval['epoch']        = epoch
                                        params_eval['deepsup']      = deepsup
                                        params_eval['deepsup_eval'] = False
                                        params_eval['eval_type']    = self.params['dataloader']['train'] # config.MODE_TRAIN
                                        eval_avg, eval_labels_avg = eval_3D(self.model, self.dataset_train, self.dataset_train_eval_gen, params_eval, save=save)
                                        print (' - [Trainer][train()] eval_labels_avg: ', eval_labels_avg)
                                        trainMetrics.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)

                                        if deepsup:
                                            params_eval['deepsup']      = deepsup
                                            params_eval['deepsup_eval'] = True
                                            params_eval['eval_type']    = config.MODE_TRAIN_DEEPSUP
                                            eval_avg, eval_labels_avg = eval_3D(self.model, self.dataset_train, self.dataset_train_eval_gen, params_eval, save=save)
                                            trainMetricsDeepSup.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)
                        except:
                            traceback.print_exc()

                        # Test
                        try:
                            if epoch % epochs_eval == 0: # special case for first 50 epochs
                                if self.dataset_test is not None:
                                    self._test()
                                    self.model.trainable = True
                        except:
                            traceback.print_exc()

                        # Epochs summary/wrapup
                        eval_condition = epoch % epochs_eval == 0
                        if epoch > 0 and epoch % epochs_info == 0:
                            trainMetrics.update_metrics_memory(memory_gpu=utils.get_nvitop_gpu_memory(res_float=True), memory_ram=utils.get_memory(config.PID, res_float=True))

                            trainMetrics.write_epoch_summary(epoch, self.label_map, {'optimizer':self.optimizer}, eval_condition)
                            if deepsup:
                                trainMetricsDeepSup.write_epoch_summary(epoch, self.label_map, {'optimizer':self.optimizer}, eval_condition)\
                            
                            layers_kl = self.get_layers_kl()
                            trainMetrics.write_epoch_summary_kl_hist(layers_kl, epoch=epoch)

                        if epoch > 0 and epoch % self.params['others']['epochs_timer'] == 0:
                            elapsed_seconds =  time.time() - t_start_time
                            print ('\n - Total time elapsed : {}'.format( str(datetime.timedelta(seconds=elapsed_seconds)) ))
                            # print ('\n - Data Counter: ', data_counter_obj)
                            data_counter_obj = {}
                        if epoch % self.params['others']['epochs_memory'] == 0:
                            mem_before = utils.get_memory(config.PID)
                            gc_n = gc.collect()
                            mem_after = utils.get_memory(config.PID)
                            print(' - Unreachable objects collected by GC: {} || ({}) -> ({})'.format(gc_n, mem_before, mem_after))
                        
                        # Break out of loop at end of all epochs
                        if epoch == max_epoch:
                            print ('\n\n - [Trainer][train()] All epochs finished')
                            break
                    
                    except:
                        utils.print_exp_name(exp_name + '-' + config.MODE_TRAIN, epoch)
                        params_save_model['epoch'] = epoch
                        utils.save_model(self.model, params_save_model)
                        traceback.print_exc()
                        pdb.set_trace()
                 
                t1 = time.time() # reset dataloader time calculator

        except:
            utils.print_exp_name(exp_name + '-' + config.MODE_TRAIN, epoch)
            traceback.print_exc()
        
    def _test(self):
        
        exp_name = None
        epoch    = None
        try:

            # Step 1.1 - Params
            exp_name = self.params['exp_name']
            epoch    = self.params['epoch']
            deepsup  = self.params['model']['deepsup']

            metrics_eval = self.params['metrics']['metrics_eval']
            epochs_viz   = self.params['model']['epochs_viz']
            batch_size   = self.params['dataloader']['batch_size']
            
            # vars
            testMetrics = self.metrics[config.MODE_TEST]
            testMetrics.reset_metrics(self.params)
            testMetricsDeepSup = None
            if deepsup:
                testMetricsDeepSup = self.metrics[config.MODE_TEST_DEEPSUP]
                testMetricsDeepSup.reset_metrics(self.params)
            params_eval = {'PROJECT_DIR': self.params['PROJECT_DIR'], 'exp_name': exp_name, 'pid': config.PID
                            , 'eval_type': config.MODE_TEST, 'batch_size': batch_size
                            , 'epoch':epoch}
                
            # Step 2 - Eval on full 3D
            save=False
            if epoch > 0 and epoch % epochs_viz == 0:
                save=True
            for metric_str in metrics_eval:
                if metrics_eval[metric_str] in [config.LOSS_DICE]:
                    params_eval['deepsup']      = deepsup
                    params_eval['deepsup_eval'] = False
                    params_eval['eval_type']    = self.params['dataloader']['test'] # config.MODE_TEST 
                    eval_avg, eval_labels_avg = eval_3D(self.model, self.dataset_test, self.dataset_test_gen, params_eval, save=save)
                    print (' - [Trainer][_test()] eval_labels_avg: ', eval_labels_avg)
                    testMetrics.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)

                    if deepsup:
                        params_eval['deepsup']      = deepsup
                        params_eval['deepsup_eval'] = True
                        params_eval['eval_type']    = config.MODE_TEST_DEEPSUP
                        eval_avg, eval_labels_avg = eval_3D(self.model, self.dataset_test, self.dataset_test_gen, params_eval, save=save)
                        testMetricsDeepSup.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)

            testMetrics.write_epoch_summary(epoch, self.label_map, {}, True)
            if deepsup:            
                testMetricsDeepSup.write_epoch_summary(epoch, self.label_map, {}, True)

        except:
            utils.print_exp_name(exp_name + '-' + config.MODE_TEST, epoch)
            traceback.print_exc()
            pdb.set_trace()

if __name__ == "__main__":

    ##########################################################
    #                       EXP NAME                         #
    ##########################################################
    # Deterministic Models
    if 0:
        pass
        # exp_name = 'Pros__ONetPool2Detv1-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
        # exp_name = 'Pros__ONetPool2Detv2-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
        # exp_name = 'Pros__ONetPool2Detv3-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
        # exp_name = 'Pros__ONetPool2Detv4-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
        # exp_name = 'Pros__ONetPool2Detv5-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
    
    # Bayesian Models
    elif 0:
        pass
        # exp_name = 'Pros__ONetPool2BayesMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'
        # exp_name = 'Pros__ONetPool2BayesHeadMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42'

    # Bayesian-AvU Models
    elif 1:
        pass
        # exp_name = 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-100AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
        # exp_name = 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-1000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
        # exp_name = 'Pros__ONetPool2BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-10000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'
        exp_name = 'Pros__ONetPool2BayesHeadMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-1000AvU-MC5Ent-Th005-04-1nAC-ep10__seed42__load-ep50'


    ##########################################################
    #                        PARAMS                          #
    ##########################################################
    if 0:
        params = {
            'PROJECT_DIR': config.PROJECT_DIR
                , 'random_seed':42
                , 'exp_name': exp_name
                , 'dataloader':{
                    'data_dir': Path(config.MAIN_DIR).joinpath('medical_dataloader', '_data')
                    , 'train': config.DATASET_PROSTATEX
                    , 'test' : config.DATASET_PROMISE12
                    , 'batch_size'    : 2 # [2,4,8]
                    , 'shuffle'       : 6 # [5,5,12]
                    , 'prefetch_batch': 5 # [4,3,2]
                    , 'parallel_calls': 8 # [4,4,4]
                    , 'single_patient': False # [ !!!!!!!! WATCH OUT !!!!!!!!! ]
                }
                , 'model': {
                    'name': config.MODEL_UNSURE_ORGANNET_BAYESIAN # [MODEL_UNSURE_ORGANNET, MODEL_UNSURE_ORGANNET_TRIPLE, MODEL_UNSURE_ORGANNET_BAYESIAN, MODEL_UNSURE_ORGANNET_HEADBAYES_DOUBLE, MODEL_UNSURE_ORGANNET_BAYESIAN_TRIPLE, MODEL_UNSURE_ORGANNET_HEADBAYES_TRIPLE]
                    , 'kl_alpha_init'   : 0.5 #[1.0, 0.5]
                    , 'kl_schedule'     : config.KL_DIV_FIXED # [config.KL_DIV_FIXED, config.KL_DIV_ANNEALING]
                    , 'mc_runs'   : 5 # [1,5,10]
                    , 'unc_metric': config.KEY_ENT # [None, config.KEY_ENT, config.KEY_MIF]
                    , 'deepsup'   : False
                    , 'optimizer' : config.OPTIMIZER_ADAM
                    , 'grad_persistent': False
                    , 'init_lr'    : 0.01 # [0.01, 0.005, 0.001]
                    , 'fixed_lr'   : False
                    , 'epochs'     : 1000  # [1000, 1500]
                    , 'epochs_save': 50   # [10,50,100]
                    , 'epochs_eval': 100   # 50
                    , 'epochs_info': 50
                    , 'epochs_viz' : np.inf # 100
                    , 'load_model':{
                        # 'load':False, 'load_exp_name': None,  'load_epoch':-1, 'load_optimizer_lr':None # DEFAULT OPTION
                        # 'load': True, 'load_exp_name': None, 'load_epoch':30, 'load_optimizer_lr':0.0098
                        # 'load': True, 'load_exp_name': None, 'load_epoch':650, 'load_optimizer_lr':None # Pros__ONetPool3BayesMC5-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10-1000AvU-MC5Ent-Th005-04-01nAC-ep10__seed42__load-ep50
                        # 'load': True, 'load_exp_name': 'Pros__ONetPool3BayesMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42__v4', 'load_epoch':50, 'load_optimizer_lr':None # DICE for ProtateX = {ep50:0.82, ep100:0.86, ep150:0.89, ep200:0.92} thus, start at ep=82
                        # 'load': True, 'load_exp_name': 'Pros__ONetPool3BayesMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42__v4', 'load_epoch':200, 'load_optimizer_lr':None
                        # 'load': True, 'load_exp_name': 'Pros__ONetPool3BayesHeadMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42__v2', 'load_epoch':200, 'load_optimizer_lr':None
                        'load': True, 'load_exp_name': 'Pros__ONetPool2BayesMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42', 'load_epoch':50, 'load_optimizer_lr':None
                        # 'load': True, 'load_exp_name': 'Pros__ONetPool2BayesHeadMC1-KL05-DecLR001ep20__ProsXB2-20020028Augv2-HistStand__CEScalar10__seed42', 'load_epoch':50, 'load_optimizer_lr':None
                        
                    } 
                    , 'profiler': {
                        'profile': False
                        , 'epochs': [252,253]
                        , 'steps_per_epoch': 30
                        , 'starting_step': 2
                    }
                    , 'model_tboard': False
                }
                , 'metrics' : {
                    'logging_tboard': True
                    # for full 3D volume
                    , 'metrics_eval': {'Dice': config.LOSS_DICE}

                    ## for smaller grid
                    # , 'metrics_loss' : {'CE': config.LOSS_CE} 
                    # , 'loss_type'    : {'CE': config.LOSS_SCALAR}
                    # , 'loss_weighted': {'CE': True}
                    # , 'loss_mask'    : {'CE': True}
                    # , 'loss_combo'   : {'CE': 1.0}
                    # , 'loss_epoch'   : {'CE': 0}
                    # , 'loss_rate'    : {'CE': 1}
                    
                    , 'metrics_loss'  : {'CE': config.LOSS_CE, 'pavpu': config.LOSS_PAVPU}
                    , 'loss_weighted' : {'CE': True, 'pavpu': False}
                    , 'loss_mask'     : {'CE': True, 'pavpu': False}
                    , 'loss_type'     : {'CE': config.LOSS_SCALAR, 'pavpu': config.LOSS_SCALAR}
                    , 'loss_combo'    : {'CE': 1.0, 'pavpu': 100.0} 
                    , 'loss_epoch'    : {'CE': 0  , 'pavpu': 10} 
                    , 'loss_rate'     : {'CE': 1  , 'pavpu': 10} # [10,30]  
                }
                , 'others': {
                    'epochs_timer': 5
                    , 'epochs_memory':5
                }
        }

        # Call the trainer
        trainer = Trainer(params)
        trainer.train()
    
    else:
        
        ########################### Datasets ###########################
        path_data = Path(config.MAIN_DIR).joinpath('medical_dataloader', '_data')
        if 0:
            dataset_eval = ProstateXDataset(path_data=path_data)
            eval_type = config.DATASET_PROSTATEX
        elif 0:
            dataset_eval = Promise12Dataset(path_data=path_data)
            eval_type = config.DATASET_PROMISE12
        elif 1:
             dataset_eval = ProstateMedDecDataset(path_data=path_data)
             eval_type = config.DATASET_PROSMEDDEC
        # dataset_eval.transforms = [Aug.MinMaxNormalizer()]

        ########################### Models ###########################
        if 1:
            class_count = len(dataset_eval.LABEL_MAP)
            deepsup     = False

            # Deterministic Models
            if 0:
                model = models.OrganNet(class_count=class_count, hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=True, deepsup=deepsup)
                # model = models.OrganNet(class_count=class_count, pooling='triple', hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=False, deepsup=deepsup)
                pass
            
            # Bayesian Models
            if 1:
                # Bayes-Double
                model = models.OrganNet(class_count=class_count, hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=False, deepsup=deepsup)
                
                # BayesHead-Double
                # model = models.OrganNet(class_count=class_count, hdc=True, bayesian=True, bayesianhead=True, dropout=None, activation=tf.nn.sigmoid, trainable=False, deepsup=deepsup)

                # Bayes-Triple
                # model = models.OrganNet(class_count=class_count, pooling='triple', hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, trainable=False, deepsup=deepsup)
                
                # BayesHead-Triple
                # model = models.OrganNet(class_count=class_count, pooling='triple', hdc=True, bayesian=True, bayesianhead=True, dropout=None, activation=tf.nn.sigmoid, trainable=False, deepsup=deepsup)
                pass

        
        ########################### Params ###########################
        params = {
            'PROJECT_DIR'     : config.PROJECT_DIR
                , 'exp_name'      : exp_name
                , 'batch_size'    : 2
                , 'prefetch_batch': 1
                , 'eval_type'     : eval_type
                , 'epoch'         : 500 # [300, 500, 650, 900, 950, 1000, 1500, 2500]
                , 'MC_RUNS'       : 32 # [1, 10, 30, None=deterministic]
                , 'deepsup'       : deepsup
                , 'deepsup_eval'  : deepsup
                , 'training_bool' : True  # [True=gaussian-sampling-at-test-time, False=no-gaussian-sampling-at-test-time]
                , 'UNC_TYPES'     : [config.KEY_ENT] # [config.KEY_ENT]
                , 'DILATION_STYLE': config.DILATION_TRUE_PRED #config.DILATION_TRUE # if config.DILATION_TRUE_PRED, then take small dilation_ksize, if config.DILATION_TRUE then take large dilation_ksize
                , 'DILATION_KSIZE': (3,3,2) # (10,10,3)
                , 'ERROR_KSIZE'   : (3,3,1)
        }

        ########################### val() ###########################
        print ('')
        print (' - MC_RUNS  : ', params['MC_RUNS'])
        print (' - epoch    : ', params['epoch'])
        print (' - UNC_TYPES: ', params['UNC_TYPES'])
        print ('')
        val(model, dataset_eval, params, show=False, save=True, verbose=False)