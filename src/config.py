from pathlib import Path
PROJECT_DIR = Path(__file__).parent.absolute().parent.absolute()
MAIN_DIR = Path(PROJECT_DIR).parent.absolute()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" # to avoid large "Kernel Launch Time"

import tensorflow as tf
try:
    if len(tf.config.list_physical_devices('GPU')):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        # tf.config.optimizer.set_jit(True) # XLA # https://www.tensorflow.org/xla/tutorials/autoclustering_xla
        sys_details = tf.sysconfig.get_build_info()
        # print (' - [TFlow Build Info] ver: ', tf.__version__, 'CUDA(major.minor):',  sys_details["cuda_version"], ' || cuDNN(major): ', sys_details["cudnn_version"])

    else:
        print (' \n\n ================================================================== ')
        print (' - No GPU present!! Exiting ...')
        print (' ================================================================== \n\n')
        import sys
        sys.exit(1)
except:
    pass


try:
    import nvitop
    GPU_DEVICE = nvitop.Device(0)
except:
    GPU_DEVICE = None

PID = os.getpid()

############################################################
#                    MODEL RELATED                         #
############################################################
MODEL_CHKPOINT_MAINFOLDER  = '_models'
MODEL_CHKPOINT_NAME_FMT    = 'ckpt_epoch{:03d}' 
MODEL_IMGCHKPOINT_NAME_FMT = 'img_ckpt_epoch{:03d}'
MODEL_LOGS_FOLDERNAME      = 'logs' 
MODEL_IMGS_FOLDERNAME      = 'images'
MODEL_PATCHES_FOLDERNAME   = 'patches'

EXT_NRRD = '.nrrd'

MODE_TRAIN     = 'Train'
MODE_TRAIN_VAL = 'Train_val'
MODE_VAL       = 'Val'
MODE_VAL_NEW   = 'Val_New'
MODE_TEST      = 'Test'
MODE_TEST_ONSITE   = 'TestOnsite'
MODE_TRAIN_DEEPSUP = 'TrainDeepSup'
MODE_TEST_DEEPSUP  = 'TestDeepSup'
MODE_DEEPMINDTCIA_TEST_ONC = 'DeepMindTCIATestOnc'
MODE_DEEPMINDTCIA_TEST_ONC_RTOG = 'DeepMindTCIATestOncRTOG'
MODE_DEEPMINDTCIA_TEST_RAD = 'DeepMindTCIATestRad'
MODE_DEEPMINDTCIA_VAL_ONC  = 'DeepMindTCIAValOnc'
MODE_DEEPMINDTCIA_VAL_RAD  = 'DeepMindTCIAValRad'
MODE_MAASTRO_TRAINFULL     = 'MaastroFull'
MODE_STRUCTSEG             = 'StructSeg'

DATASET_MICCAI = 'MICCAI'
DATASET_DEEPMIND = 'DeepMind'
DATASET_STRUCTSEG = 'StructSeg'

ACT_SIGMOID = 'sigmoid'
ACT_SOFTMAX = 'softmax'

MODEL_UNET3D = 'ModelUNet3D'
MODEL_UNET3DASPP = 'ModelUNet3DASPP'
MODEL_UNET3DJOINTSEGANDREG = 'ModelUnet3D_JointSegAndReg'
MODEL_TRANSFORMER = 'Model_Dense3DSpatialTransformer'
MODEL_FOCUSNET = 'ModelFocusNet'
MODEL_FOCUSNET_GNORM = 'ModelFocusNetGNorm'
MODEL_FOCUSNET_ZDIL1         = 'ModelFocusNetZDil1'
MODEL_FOCUSNET_ZDIL1V2       = 'ModelFocusNetZDil1V2'
MODEL_FOCUSNET_ZDIL1V3       = 'ModelFocusNetZDil1V3'
MODEL_FOCUSNET_ZDIL1V4       = 'ModelFocusNetZDil1V4'
MODEL_FOCUSNET_ZDIL1_F16     = 'ModelFocusNetZDil1F16'
MODEL_FOCUSNET_ZDIL1_F16V2   = 'ModelFocusNetZDil1F16V2'

MODEL_UNSURE_FOCUSNET           = 'FocusNet'
MODEL_UNSURE_FOCUSNETHDC        = 'FocusNetHDC'
MODEL_UNSURE_ORGANNET           = 'OrganNet'
MODEL_UNSURE_ORGANNETNOHDC      = 'OrganNetnonHDC'
MODEL_UNSURE_ORGANNET_HEADBAYES       = 'OrganNetHeadBayes'
MODEL_UNSURE_ORGANNET_HEADBAYES_NOHDC = 'OrganNetHeadBayesNoHDC'
MODEL_UNSURE_FOCUSNET_BAYESIAN  = 'FocusNetBayesian'
MODEL_UNSURE_ORGANNET_BAYESIAN  = 'OrganNetBayesian'
MODEL_UNSURE_ORGANNET_HEADBAYES_DOUBLE = 'OrganNetHeadBayesDoublePool'
MODEL_UNSURE_ORGANNET_DROPOUT    = 'OrganNetDropOut'

MODEL_UNSURE_ORGANNET_TRIPLE          = 'OrganNetTriplePool'
MODEL_UNSURE_ORGANNET_BAYESIAN_TRIPLE = 'OrganNetBayesianTriplePool'
MODEL_UNSURE_ORGANNET_HEADBAYES_TRIPLE = 'OrganNetHeadBayesTriplePool'

MODEL_UNSURE_ORGANET_2D          = 'OrganNet2DQuadPool'
MODEL_UNSURE_ORGANET_BAYESIAN_2D = 'OrganNet2DBayesianQuadPool'

MODEL_FOCUSNETRES            = 'FocusNetRes'
MODEL_FOCUSNETRESV2          = 'FocusNetResV2'
MODEL_FOCUSNET_2D3D          = 'ModelFocusNet2D3D'

MODEL_FOCUSNET_ALUNC         = 'ModelFocusNetAlUnc'
MODEL_FOCUSNET_FLIPOUT_ALUNC = 'ModelFocusNetFlipOutAlUnc'

MODEL_FOCUSNET_FLIPOUT       = 'ModelFocusNetFlipOut'
MODEL_FOCUSNET_FLIPOUT_V2    = 'ModelFocusNetFlipOutV2'
MODEL_FOCUSNET_FLIPOUT_POOL2 = 'ModelFocusNetFlipOutPool2'

MODEL_FOCUSNETSTOCH          = 'ModelFocusNetStochastic'
MODEL_FOCUSNET_INORM = 'ModelFocusNetINorm'
MODEL_FOCUSNET_POOL2 = 'ModelFocusNetPool2'
MODEL_ASPP_ATTENTION = 'ModelUNet3DASPPAttention'
MODEL_ASPP_ATTENTION_PART3D = 'ASPPAttentionPartial3D'
MODEL_ASPP_ATTENTION_PART3DFULLRP = 'ASPPAttentionPartial3DFullRP'
MODEL_ASPP_CHANNELATTENTION = 'ModelUNet3DASPPChannelAttention'
MODEL_SPATIALATTNV3 = 'SpatialAttentionModelv3'

MODEL_ONET = 'ModelONet'
MODEL_ONETGNORM = 'ModelONetGNorm'
MODEL_ONET_FLIPOUT = 'ModelONetFlipOut'
MODEL_ONET_FLIPOUTTT = 'ModelONetFlipOuttt'
MODEL_ONET_FLIPOUTv2 = 'ModelONetFlipOutv2'

MODEL_TEMPSCALE       = 'ModelTempScaling'
MODEL_TEMPSCALE_CLASS = 'ModelTempScalingClass'
MODEL_PLATT_1X1       = 'ModelPlatt1x1'
MODEL_PLATT_3X3       = 'ModelPlatt3x3'

OPTIMIZER_ADAM = 'Adam'

THRESHOLD_SIGMA_IGNORE = 0.3
MIN_SIZE_COMPONENT = 15 # [10,15]
 
KL_DIV_FIXED     = 'fixed'
KL_DIV_ANNEALING = 'annealing'

############################################################
#                      EVAL RELATED                        #
############################################################
KEY_DICE_AVG        = 'dice_avg'
KEY_DICE_STD        = 'dice_std'
KEY_DICE_LABELS     = 'dice_labels'
KEY_DICE_LABELS_STD = 'dice_labels_std'

KEY_HD_AVG          = 'hd_avg'
KEY_HD_STD          = 'hd_std'
KEY_HD_LABELS       = 'hd_labels'
KEY_HD_LABELS_STD   = 'hd_labels_std'

KEY_HD95_AVG        = 'hd95_avg'
KEY_HD95_STD        = 'hd95_std'
KEY_HD95_LABELS     = 'hd95_labels'
KEY_HD95_LABELS_STD = 'hd95_labels_std'

KEY_MSD_AVG         = 'msd_avg'
KEY_MSD_STD         = 'msd_std'
KEY_MSD_LABELS      = 'msd_labels'
KEY_MSD_LABELS_STD  = 'msd_labels_std'

KEY_ECE_AVG     = 'ece_avg'
KEY_ECE_LABELS  = 'ece_labels'
KEY_AVU_ENT       = 'avu_ent'
KEY_AVU_PAC_ENT   = 'avu_pac_ent'
KEY_AVU_PUI_ENT   = 'avu_pui_ent'
KEY_THRESH_ENT    = 'avu_thresh_ent'
KEY_AVU_MIF       = 'avu_mif'
KEY_AVU_PAC_MIF   = 'avu_pac_mif'
KEY_AVU_PUI_MIF   = 'avu_pui_mif'
KEY_THRESH_MIF    = 'avu_thresh_mif'
KEY_AVU_UNC       = 'avu_unc'
KEY_AVU_PAC_UNC   = 'avu_pac_unc'
KEY_AVU_PUI_UNC   = 'avu_pui_unc'
KEY_THRESH_UNC    = 'avu_thresh_unc'

DILATION_TRUE_PRED = 'dilation_true_pred'
DILATION_TRUE      = 'dilation_true'

PAVPU_UNC_THRESHOLD = 'adaptive-median' # [0.3, 'adaptive', 'adaptive-median']
PAVPU_ENT_THRESHOLD  = 0.5
PAVPU_MIF_THRESHOLD  = 0.1
PAVPU_GRID_SIZE     = (4,4,2) 
PAVPU_RATIO_NEG     = 0.9

KEY_ENT  = 'Ent'
KEY_MIF  = 'MI'
KEY_STD  = 'Std'
KEY_STD_MAX = 'Std (Max)'
KEY_PERC = 'perc'

KEY_SUM  = 'sum'
KEY_AVG  = 'avg'
 
CMAP_MAGMA = 'magma'
CMAP_ORANGES = 'Oranges'
CMAP_GRAY  = 'gray'

FILENAME_EVAL3D_JSON     = 'res.json'

FOLDERNAME_TMP = '_tmp'
FOLDERNAME_TMP_BOKEH = 'bokeh-plots'
FOLDERNAME_TMP_ENTMIF = 'entmif'

VAL_ECE_NAN = -0.1
VAL_DICE_NAN = -1.0
VAL_MC_RUNS_DEFAULT = 20

KEY_PATIENT_GLOBAL = 'global'

SUFFIX_DET = '-Det'
# SUFFIX_DET = '-Detv2'
SUFFIX_MC  = '-MC{}'

KEY_MC_RUNS = 'MC_RUNS'
KEY_TRAINING_BOOL = 'training_bool'

KEY_UNC_TYPES = 'UNC_TYPES'

KEYNAME_ENT     = 'maskpredent'
KEYNAME_MIF     = 'maskpredmif'
KEYNAME_STD     = 'maskpredstd'
KEYNAME_STD_MAX = 'maskpredstdmax'

FIGURE_ENT      = 'Entropy'
FIGURE_MI       = 'Mutual Information'
FIGURE_STD      = 'Standard Deviation'

FILENAME_SAVE_CT      = 'nrrd_{}_img.nrrd'
FILENAME_SAVE_GT      = 'nrrd_{}_mask.nrrd'
FILENAME_SAVE_PROB_ALL= 'nrrd_{}_maskpredallprob.nrrd'
FILENAME_SAVE_PROB    = 'nrrd_{}_maskpredmaxprob.nrrd'
FILENAME_SAVE_PRED    = 'nrrd_{}_maskpred.nrrd'
FILENAME_SAVE_ENT     = 'nrrd_{}_{}.nrrd'.format('{}',KEYNAME_ENT)
FILENAME_SAVE_MIF     = 'nrrd_{}_{}.nrrd'.format('{}',KEYNAME_MIF)
FILENAME_SAVE_STD     = 'nrrd_{}_{}.nrrd'.format('{}',KEYNAME_STD)
FILENAME_SAVE_STD_MAX = 'nrrd_{}_{}.nrrd'.format('{}',KEYNAME_STD_MAX) 

KEY_P_UA  = 'p_ua'
KEY_P_UI  = 'p_ui'
KEY_P_IU  = 'p_iu'
KEY_P_AC  = 'p_ac'
KEY_PAVPU = 'PAvPU'
KEY_AVU   = 'AvU'

CALIBRATION       = 'calibration'
CALIBRATION_ECE   = 'ECE'
CALIBRATION_SCE   = 'SCE'
CALIBRATION_ThSCE = 'ThSCE'
CALIBRATION_ACE   = 'ACE'
CALIBRATION_ThACE = 'ThACE'

KEY_CALIBRATION_THRESHOLD = 'threshold_calib'
KEY_CALIBTRATION_BINS = 'num_bins'
NORM_L1 = 'l1'
NORM_L2 = 'l2'

CALIBRATION_RES                = 'res'
CALIBTRATION_AVG_PROBS         = 'avg_probs'
CALIBRATION_AVG_ACC            = 'avg_acc'
CALIBRATION_BINS_UPPER_BOUNDS  = 'bin_upper_bounds'
CALIBRATION_SAMPLES            = 'N'
CALIBRATION_BIN_COUNT          = 'bin_count'

KEY_Y_TRUE_LIST = 'y_true_list'
KEY_Y_PRED_LIST = 'y_pred_list'

import numpy as np
EPSILON = np.finfo(np.float32).eps
_EPSILON = tf.keras.backend.epsilon()

############################################################
#                      LOSSES RELATED                      #
############################################################
LOSS_DICE  = 'Dice'
LOSS_FOCAL = 'Focal'
LOSS_CE    = 'CE'
LOSS_CE_BOUNDARY = 'CEBoundary'
LOSS_NCC   = 'NCC'
LOSS_SCALAR = 'scalar'
LOSS_VECTOR = 'vector'
LOSS_PURATIO  = 'p_u'
LOSS_PAVPU    = 'pavpu'
LOSS_AUC      = 'auc'
LOSS_AUC_TRAP = 'auc_trap' # trapezoidal auc calculation

############################################################
#                   DATALOADER RELATED                     #
############################################################

DATASET_PROMISE12 = 'Promise12'
DATASET_PROSTATEX = 'ProstateX'
DATASET_PROSMEDDEC = 'ProsMedDec'

DATALOADER_MICCAI2015_MAASTRO_STRUCTSEG_TRAIN = 'miccai_maastro_structseg_train'
DATALOADER_MICCAI2015_STRUCTSEG_TRAIN = 'miccai_structseg_train'
DATALOADER_MAASTRO_STRUCTSEG_TRAIN = 'maastro_structseg_train'
DATALOADER_MICCAI2015_MAASTRO_TRAIN = 'miccai_maastro_train'

DATALOADER_MICCAI2015_TRAIN      = 'train'
DATALOADER_MICCAI2015_TRAIN_ADD  = 'train_additional'
DATALOADER_MICCAI2015_TEST       = 'test_offsite'
DATALOADER_MICCAI2015_TESTONSITE = 'test_onsite'

DATALOADER_DEEPMINDTCIA_TEST = 'test'
DATALOADER_DEEPMINDTCIA_VAL  = 'validation'
DATALOADER_DEEPMINDTCIA_ONC  = 'oncologist'
DATALOADER_DEEPMINDTCIA_RAD  = 'radiographer'

PREFETCH_BUFFER = 5

DATASET_MICCAI2015   = 'miccai2015'
DATASET_DEEPMINDTCIA = 'deepmindtcia'

PATIENT_MICCAI2015_TESTOFFSITE = 'HaN_MICCAI2015-test_offsite-{}_resample_True'
FILENAME_SAVE_CT_MICCAI2015    = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_img.nrrd'
FILENAME_SAVE_GT_MICCAI2015    = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_mask.nrrd'
FILENAME_SAVE_PRED_MICCAI2015  = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpred.nrrd'
FILENAME_SAVE_PRED_PROB_MAX_MICCAI2015 = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredmaxprob.nrrd'
FILENAME_SAVE_PRED_PROB_ALL_MICCAI2015 = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredallprob.nrrd' 
FILENAME_SAVE_MIF_MICCAI2015   = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredmif.nrrd'
FILENAME_SAVE_ENT_MICCAI2015   = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredent.nrrd'
FILENAME_SAVE_STD_MICCAI2015   = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredstd.nrrd'
FILENAME_SAVE_STDMAX_MICCAI2015 = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredstdmax.nrrd'

PATIENT_MICCAI2015_TESTONSITE                     = 'HaN_MICCAI2015-test_onsite-{}_resample_True'
FILENAME_SAVE_CT_MICCAI2015_TESTONSITE            = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_img.nrrd'
FILENAME_SAVE_GT_MICCAI2015_TESTONSITE            = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_mask.nrrd'
FILENAME_SAVE_PRED_MICCAI2015_TESTONSITE          = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_maskpred.nrrd'
FILENAME_SAVE_PRED_PROB_MAX_MICCAI2015_TESTONSITE = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_maskpredmaxprob.nrrd'
FILENAME_SAVE_PRED_PROB_ALL_MICCAI2015_TESTONSITE = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_maskpredallprob.nrrd' 
FILENAME_SAVE_MIF_MICCAI2015_TESTONSITE           = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_maskpredmif.nrrd'
FILENAME_SAVE_ENT_MICCAI2015_TESTONSITE           = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_maskpredent.nrrd'
FILENAME_SAVE_STD_MICCAI2015_TESTONSITE           = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_maskpredstd.nrrd'
FILENAME_SAVE_STDMAX_MICCAI2015_TESTONSITE        = 'nrrd_HaN_MICCAI2015-test_onsite-{}_resample_True_maskpredstdmax.nrrd'

PATIENTIDS_MICCAI2015_TEST_ONSITE   = ['0522c0788', '0522c0806', '0522c0845', '0522c0857', '0522c0878']
PATIENTIDS_MICCAI2015_TEST   = ['0522c0555', '0522c0576', '0522c0598', '0522c0659', '0522c0661', '0522c0667', '0522c0669', '0522c0708', '0522c0727', '0522c0746']
# PATIENTIDS_DEEPMINDTCIA_TEST = ['0522c0017', '0522c0057', '0522c0161', '0522c0226', '0522c0248', '0522c0251', '0522c0331', '0522c0416', '0522c0419', '0522c0427', '0522c0457', '0522c0479', '0522c0629', '0522c0659', '0522c0667', '0522c0669', '0522c0708', '0522c0768', '0522c0770', '0522c0773', '0522c0845', 'TCGA-CV-7236', 'TCGA-CV-7243', 'TCGA-CV-7245', 'TCGA-CV-A6JO', 'TCGA-CV-A6JY', 'TCGA-CV-A6K0', 'TCGA-CV-A6K1']
PATIENTIDS_DEEPMINDTCIA_TEST = ['0522c0331', '0522c0416', '0522c0419', '0522c0629', '0522c0768', '0522c0770', '0522c0773', '0522c0845', 'TCGA-CV-7236', 'TCGA-CV-7243', 'TCGA-CV-7245', 'TCGA-CV-A6JO', 'TCGA-CV-A6JY', 'TCGA-CV-A6K0', 'TCGA-CV-A6K1']
PATIENTIDS_DEEPMINDTCIA_TEST_RTOG = ['0522c0331', '0522c0416', '0522c0419', '0522c0629', '0522c0768', '0522c0770', '0522c0773', '0522c0845']

PATIENT_DEEPMINDTCIA_TEST_ONC              = 'HaN_DeepMindTCIA-test-oncologist-{}_resample_True'
FILENAME_SAVE_CT_DEEPMINDTCIA_TEST_ONC     = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_img.nrrd'
FILENAME_SAVE_GT_DEEPMINDTCIA_TEST_ONC     = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_mask.nrrd'
FILENAME_SAVE_PRED_DEEPMINDTCIA_TEST_ONC   = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpred.nrrd'
FILENAME_SAVE_PRED_PROB_MAX_DEEPMINDTCIA_TEST_ONC      = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredmaxprob.nrrd'
FILENAME_SAVE_PRED_PROB_ALL_DEEPMINDTCIA_TEST_ONC      = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredallprob.nrrd'
FILENAME_SAVE_MIF_DEEPMINDTCIA_TEST_ONC    = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredmif.nrrd'
FILENAME_SAVE_ENT_DEEPMINDTCIA_TEST_ONC    = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredent.nrrd'
FILENAME_SAVE_STD_DEEPMINDTCIA_TEST_ONC    = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredstd.nrrd'
FILENAME_SAVE_STDMAX_DEEPMINDTCIA_TEST_ONC = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredstdmax.nrrd'

PATIENT_DEEPMINDTCIA_TEST_RAD              = 'HaN_DeepMindTCIA-test-radiographer-{}_resample_True'
FILENAME_SAVE_CT_DEEPMINDTCIA_TEST_RAD     = 'nrrd_HaN_DeepMindTCIA-test-radiographer-{}_resample_True_img.nrrd'
FILENAME_SAVE_GT_DEEPMINDTCIA_TEST_RAD     = 'nrrd_HaN_DeepMindTCIA-test-radiographer-{}_resample_True_mask.nrrd'
FILENAME_SAVE_PRED_DEEPMINDTCIA_TEST_RAD   = 'nrrd_HaN_DeepMindTCIA-test-radiographer-{}_resample_True_maskpred.nrrd'
FILENAME_SAVE_MIF_DEEPMINDTCIA_TEST_RAD    = 'nrrd_HaN_DeepMindTCIA-test-radiographer-{}_resample_True_maskpredmif.nrrd'
FILENAME_SAVE_ENT_DEEPMINDTCIA_TEST_RAD    = 'nrrd_HaN_DeepMindTCIA-test-radiographer-{}_resample_True_maskpredent.nrrd'
FILENAME_SAVE_STD_DEEPMINDTCIA_TEST_RAD    = 'nrrd_HaN_DeepMindTCIA-test-radiographer-{}_resample_True_maskpredstd.nrrd'
FILENAME_SAVE_STDMAX_DEEPMINDTCIA_TEST_RAD = 'nrrd_HaN_DeepMindTCIA-test-radiographer-{}_resample_True_maskpredstdmax.nrrd'

PATIENTIDS_PROMISE12 = ['Case00', 'Case01', 'Case02', 'Case03', 'Case04', 'Case05', 'Case06', 'Case07', 'Case08', 'Case09', 'Case10', 'Case11', 'Case12', 'Case13', 'Case14', 'Case15', 'Case16', 'Case17', 'Case18', 'Case19', 'Case20', 'Case21', 'Case22', 'Case23', 'Case24', 'Case25', 'Case26', 'Case27', 'Case28', 'Case29', 'Case30', 'Case31', 'Case32', 'Case33', 'Case34', 'Case35', 'Case36', 'Case37', 'Case38', 'Case39', 'Case40', 'Case41', 'Case42', 'Case43', 'Case44', 'Case45', 'Case46', 'Case47', 'Case48', 'Case49']
PATIENTIDS_PROSTATEX = ['ProstateX-0004', 'ProstateX-0007', 'ProstateX-0009', 'ProstateX-0015', 'ProstateX-0020', 'ProstateX-0026', 'ProstateX-0046', 'ProstateX-0054', 'ProstateX-0056', 'ProstateX-0065', 'ProstateX-0066', 'ProstateX-0069', 'ProstateX-0070', 'ProstateX-0072', 'ProstateX-0076', 'ProstateX-0083', 'ProstateX-0084', 'ProstateX-0089', 'ProstateX-0090', 'ProstateX-0094', 'ProstateX-0096', 'ProstateX-0102', 'ProstateX-0111', 'ProstateX-0112', 'ProstateX-0117', 'ProstateX-0118', 'ProstateX-0121', 'ProstateX-0125', 'ProstateX-0129', 'ProstateX-0130', 'ProstateX-0134', 'ProstateX-0136', 'ProstateX-0141', 'ProstateX-0142', 'ProstateX-0144', 'ProstateX-0150', 'ProstateX-0156', 'ProstateX-0161', 'ProstateX-0168', 'ProstateX-0170', 'ProstateX-0176', 'ProstateX-0177', 'ProstateX-0182', 'ProstateX-0183', 'ProstateX-0184', 'ProstateX-0188', 'ProstateX-0193', 'ProstateX-0196', 'ProstateX-0198', 'ProstateX-0201', 'ProstateX-0209', 'ProstateX-0217', 'ProstateX-0219', 'ProstateX-0234', 'ProstateX-0241', 'ProstateX-0244', 'ProstateX-0249', 'ProstateX-0254', 'ProstateX-0265', 'ProstateX-0275', 'ProstateX-0297', 'ProstateX-0309', 'ProstateX-0311', 'ProstateX-0323', 'ProstateX-0334', 'ProstateX-0340']
PATIENTIDS_PROMEDDEC = ['ProsMedDec-00', 'ProsMedDec-01', 'ProsMedDec-02', 'ProsMedDec-04', 'ProsMedDec-06', 'ProsMedDec-07', 'ProsMedDec-10', 'ProsMedDec-13', 'ProsMedDec-14', 'ProsMedDec-16', 'ProsMedDec-17', 'ProsMedDec-18', 'ProsMedDec-20', 'ProsMedDec-21', 'ProsMedDec-24', 'ProsMedDec-25', 'ProsMedDec-28', 'ProsMedDec-29', 'ProsMedDec-31', 'ProsMedDec-32', 'ProsMedDec-34', 'ProsMedDec-35', 'ProsMedDec-37', 'ProsMedDec-38', 'ProsMedDec-39', 'ProsMedDec-40', 'ProsMedDec-41', 'ProsMedDec-42', 'ProsMedDec-43', 'ProsMedDec-44', 'ProsMedDec-46', 'ProsMedDec-47']

PATIENT_IDS_STRUCTSEG = list(range(1,51))
PATIENT_STRUCTSEG_PRE                 = 'nrrd_HaN_StructSeg-processed_0.8_0.8_2.5-'
FILENAME_SAVE_CT_STRUCTSEG            = PATIENT_STRUCTSEG_PRE + '{}_img.nrrd'
FILENAME_SAVE_GT_STRUCTSEG            = PATIENT_STRUCTSEG_PRE + '{}_mask.nrrd'
FILENAME_SAVE_PRED_STRUCTSEG          = PATIENT_STRUCTSEG_PRE + '{}_maskpred.nrrd'
FILENAME_SAVE_PRED_PROB_MAX_STRUCTSEG = PATIENT_STRUCTSEG_PRE + '{}_maskpredmaxprob.nrrd'
FILENAME_SAVE_PRED_PROB_ALL_STRUCTSEG = PATIENT_STRUCTSEG_PRE + '{}_maskpredallprob.nrrd' 
FILENAME_SAVE_MIF_STRUCTSEG           = PATIENT_STRUCTSEG_PRE + '{}_maskpredmif.nrrd'
FILENAME_SAVE_ENT_STRUCTSEG           = PATIENT_STRUCTSEG_PRE + '{}_maskpredent.nrrd'
FILENAME_SAVE_STD_STRUCTSEG           = PATIENT_STRUCTSEG_PRE + '{}_maskpredstd.nrrd'
FILENAME_SAVE_STDMAX_STRUCTSEG        = PATIENT_STRUCTSEG_PRE + '{}_maskpredstdmax.nrrd'

############################################################
#                    VISUALIZATION                         #
############################################################
FIGSIZE=(15,15)
IGNORE_LABELS = []
PREDICT_THRESHOLD_MASK = 0.6

ENT_MIN, ENT_MAX = 0.0, 0.5
MIF_MIN, MIF_MAX = 0.0, 0.1

############################################################
#            STOCHASTIC SEGMENTATION NETWORK               #
############################################################
NORMAL_SAMPLES_DIR = 'normal_samples'
RAW_IMG        = 'nrrd_{}_img.nrrd'
RAW_MASK       = 'nrrd_{}_mask.nrrd'
NORMAL_MEAN    = 'nrrd_{}_maskprednormalmean.nrrd'
NORMAL_COVFAC  = 'nrrd_{}_maskprednormalcovfactor.nrrd'
NORMAL_COVDIAG = 'nrrd_{}_maskprednormalcovdiag.nrrd'
NORMAL_SAMPLE  = 'nrrd_{}_maskpred{}.nrrd'
