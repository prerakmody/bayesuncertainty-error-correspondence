# Import internal libraries
import src.config as config

# Import external libraries
import pdb
import time
import traceback
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

print ('')
print (' - tf : ', tf.__version__)  # 2.9.1
print (' - tfa: ', tfa.__version__) # 0.17.1
print (' - tfp: ', tfp.__version__) # 0.17.0
print ('')

############################################################
#                     3D MODEL BLOCKS                      #
############################################################

class ConvBlock3DTrial(tf.keras.layers.Layer):
    """
    Performs a series of 3D convolutions which are residual in nature
    """

    def __init__(self, filters, dilation_rates, kernel_size=(3,3,3), strides=(1,1,1), padding='same'
                    , activation=tf.nn.relu
                    , group_factor=4
                    , trainable=False
                    , dropout=None
                    , residual=True
                    , init_filters=False
                    , bayesian=False
                    , spectral=False
                    , pool=None
                    , name=''):
        super(ConvBlock3D, self).__init__(name='{}_ConvBlock3DTrial'.format(name))

        # Step 0 - Init
        self.init_filters = init_filters
        self.pool = pool
        self.residual = residual
        if type(filters) == int:
            filters = [filters]
        
        if spectral:
            f = lambda x: tfa.layers.SpectralNormalization(x)
        else:
            f = lambda x: x

        # Step 1 - Set the filters right so that the residual can be done
        if self.init_filters:
            self.init_layer = tf.keras.Sequential(name='{}_Conv1x1Seq'.format(self.name))
            if bayesian:
                self.init_layer.add(
                        f(
                            tfp.layers.Convolution3DFlipout(filters=filters[0], kernel_size=(1,1,1), strides=strides, padding=padding
                            , dilation_rate=1
                            , activation=None
                            , name='{}_Conv1x1Flip'.format(self.name))
                        )
                )
            else:
                self.init_layer.add(
                    f(
                        tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=strides, padding=padding
                            , dilation_rate=1
                            , activation=None
                            , name='{}_Conv1x1'.format(self.name))
                        )
                    )
            self.init_layer.add(tfa.layers.GroupNormalization(groups=filters[0]//group_factor, trainable=trainable))
            self.init_layer.add(tf.keras.layers.Activation(activation))

        # Step 2 - Create residual block
        self.conv_layer = tf.keras.Sequential(name='{}_ConvSeq'.format(self.name))
        for filter_id, filter_count in enumerate(filters):
            
            # Step 2.0 - DropOut or not
            # print (' - name: {} || dropout: {}'.format(self.name, dropout) )
            if dropout is not None:
                self.conv_layer.add(tf.keras.layers.Dropout(rate=dropout, name='{}_DropOut_{}'.format(self.name, filter_id))) # before every conv layer (could also be after every layer?)

            spectral = True
            if spectral:
                f = lambda x: tfa.layers.SpectralNormalization(x)
            else:
                f = lambda x: x

            # Step 2.1 - Bayesian or not
            if bayesian:
                self.conv_layer.add(
                    f(
                        tfp.layers.Convolution3DFlipout(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                                , dilation_rate=dilation_rates[filter_id]
                                , activation=None
                                , name='{}_Conv3DFlip_{}'.format(self.name, filter_id))
                    )
                )
            else:
                self.conv_layer.add(
                    f(
                        tf.keras.layers.Conv3D(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                                , dilation_rate=dilation_rates[filter_id]
                                , activation=None
                                , name='{}_Conv3D_{}'.format(self.name, filter_id))
                    )
                )
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//group_factor, trainable=trainable)) 

            # Step 2.2 - Residual or not
            if self.residual:
                if filter_id != len(filters) -1: # dont add activation in the last conv of the block
                    self.conv_layer.add(tf.keras.layers.Activation(activation))
            else:
                self.conv_layer.add(tf.keras.layers.Activation(activation))
        
        # Step 2.1 - Finish residual block
        if self.residual:
            self.activation_layer = tf.keras.layers.Activation(activation)
        
        # Step 3 - Learnt Pooling
        if self.pool is not None:
            self.pool_layer = f(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                                , dilation_rate=(1,1,1)
                                , activation=None
                                , name='{}_Conv3DPooling'.format(self.name)))
    
    def call(self, x):

        # Step 1         
        if self.init_filters:
            x = self.init_layer(x)

        # Step 2 
        if self.residual:
            x_ = self.conv_layer(x) # Conv-GN-ReLU -- Conv-GN
            x = self.activation_layer(x_ + x) # RelU
        else:
            x = self.conv_layer(x)
        
        # Step 3
        if self.pool:
            x_pool = self.pool_layer(x)
            return x, x_pool
        else:
            return x

class ConvBlock3D(tf.keras.layers.Layer):
    """
    Performs a series of 3D convolutions which are residual in nature
    """

    def __init__(self, filters, dilation_rates, kernel_size=(3,3,3), strides=(1,1,1), padding='same'
                    , activation=tf.nn.relu
                    , group_factor=4
                    , trainable=False
                    , dropout=None
                    , residual=True
                    , init_filters=False
                    , bayesian=False
                    , spectral=False
                    , pool=None
                    , name=''):
        super(ConvBlock3D, self).__init__(name='{}_ConvBlock3D'.format(name))

        # Step 0 - Init
        self.init_filters = init_filters
        self.pool         = pool
        self.residual     = residual
        self.kernel_size  = kernel_size # for debugging
        self.filters      = filters     # for debugging
        self.group_factor = group_factor
        if type(filters) == int:
            filters = [filters]

        # Step 1 - Set the filters right so that the residual can be done
        if self.init_filters:
            self.init_layer = tf.keras.Sequential(name='{}_Conv1x1Seq'.format(self.name))
            if bayesian:
                self.init_layer.add(
                        # 
                        tfp.layers.Convolution3DFlipout(filters=filters[0], kernel_size=(1,1,1), strides=strides, padding=padding
                            , dilation_rate=1
                            , activation=None
                            , name='{}_Conv1x1Flip'.format(self.name))
                )
            else:
                self.init_layer.add(
                        tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=strides, padding=padding
                            , dilation_rate=1
                            , activation=None
                            , name='{}_Conv1x1'.format(self.name))
                    )
            self.init_layer.add(tfa.layers.GroupNormalization(groups=filters[0]//group_factor, trainable=trainable))
            self.init_layer.add(tf.keras.layers.Activation(activation))

        # Step 2 - Create residual block
        self.conv_layer = tf.keras.Sequential(name='{}_ConvSeq'.format(self.name))
        for filter_id, filter_count in enumerate(filters):
            
            # Step 2.0 - DropOut or not
            # print (' - name: {} || dropout: {}'.format(self.name, dropout) )
            if dropout is not None:
                self.conv_layer.add(tf.keras.layers.Dropout(rate=dropout, name='{}_DropOut_{}'.format(self.name, filter_id))) # before every conv layer (could also be after every layer?)

            # Step 2.1 - Bayesian or not
            if bayesian:
                self.conv_layer.add(
                        tfp.layers.Convolution3DFlipout(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                                , dilation_rate=dilation_rates[filter_id]
                                , activation=None
                                , name='{}_Conv3DFlip_{}'.format(self.name, filter_id))
                )
            else:
                self.conv_layer.add(
                        tf.keras.layers.Conv3D(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                                , dilation_rate=dilation_rates[filter_id]
                                , activation=None
                                , name='{}_Conv3D_{}'.format(self.name, filter_id))
                )
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//group_factor, trainable=trainable)) 

            # Step 2.2 - Residual or not
            if self.residual:
                if filter_id != len(filters) -1: # dont add activation in the last conv of the block
                    self.conv_layer.add(tf.keras.layers.Activation(activation))
            else:
                self.conv_layer.add(tf.keras.layers.Activation(activation))
        
        # Step 2.1 - Finish residual block
        if self.residual:
            self.activation_layer = tf.keras.layers.Activation(activation)
        
        # Step 3 - Learnt Pooling
        if self.pool is not None:
            self.pool_layer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                                , dilation_rate=(1,1,1)
                                , activation=None
                                , name='{}_Conv3DPooling'.format(self.name))
    
    def call(self, x):

        # Step 1         
        if self.init_filters:
            x = self.init_layer(x)

        # Step 2 
        if self.residual:
            x_ = self.conv_layer(x) # Conv-GN-ReLU -- Conv-GN
            x = self.activation_layer(x_ + x) # RelU
        else:
            x = self.conv_layer(x)
        
        # Step 3
        if self.pool:
            x_pool = self.pool_layer(x)
            return x, x_pool
        else:
            return x
    
    def get_config(self):

        config = {}
        
        if self.init_filters:
            config['init_filters'] = self.init_layer
        
        config['conv_layer'] = self.conv_layer

        if self.residual:
            config['residual'] = self.activation_layer
        
        if self.pool:
            config['pool'] = self.pool_layer
        
        return config

class UpConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(2,2,2), strides=(2, 2, 2), padding='same', spectral=False, trainable=False, name=''):
        super(UpConvBlock, self).__init__(name='{}_UpConvBlock'.format(name))
        
        if spectral:
            f = lambda x: tfa.layers.SpectralNormalization(x)
        else:
            f = lambda x: x

        self.upconv_layer = tf.keras.Sequential()
        self.upconv_layer.add(
                f(
                    tf.keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=kernel_size, padding=padding
                        , activation=None
                        , name='{}__ConvTranspose'.format(name))
                )
            )
    
    def call(self, x):
        return self.upconv_layer(x)
    
    def get_config(self):

        return {'upconv_layer': self.upconv_layer}

class ConvBlockSERes(tf.keras.layers.Layer):
    
    def __init__(self, filters, dilation_rates, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , activation=tf.nn.relu
                    , group_factor=4
                    , trainable=False
                    , dropout=None
                    , init_filters=False
                    , bayesian=False
                    , spectral=False
                    , pool=None
                    , squeeze_ratio=2
                    , name=''):

        super(ConvBlockSERes, self).__init__(name='{}_ConvSERes'.format(name))

        # Step 0 - Init
        self.pool = pool
        self.filters = filters # for debugging
        if spectral:
            f = lambda x: tfa.layers.SpectralNormalization(x)
        else:
            f = lambda x: x

        # Step 1 - ConvBlock    
        assert len(filters) == len(dilation_rates) # eg. len([32,32,32]) = len([(1,1,1), (3,3,1), (5,5,1)])
        self.convblock_res = ConvBlock3D(filters=filters, dilation_rates=dilation_rates, kernel_size=kernel_size, strides=strides, padding=padding
                            , activation=activation
                            , group_factor=group_factor
                            , trainable=trainable
                            , dropout=dropout
                            , init_filters=init_filters
                            , bayesian=bayesian
                            , pool=None
                            , name='{}'.format(self.name)
                            )

        # Step 2 - Squeeze and Excitation 
        ## Ref: https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
        self.seblock = tf.keras.Sequential(name='{}_SERes'.format(name))
        self.seblock.add(tf.keras.layers.GlobalAveragePooling3D())
        self.seblock.add(tf.keras.layers.Reshape(target_shape=(1,1,1,filters[0])))
        self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0]//squeeze_ratio, kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                , activation=tf.nn.relu))
        self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                , activation=tf.nn.sigmoid))

        # Step 3 - Pooling
        # Step 3 - Learnt Pooling
        if self.pool is not None:
            self.pool_layer = f(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=self.pool, strides=self.pool, padding=padding
                                , dilation_rate=(1,1,1)
                                , activation=None
                                , name='{}_Conv3DPooling'.format(self.name)))

    def call(self, x):
        
        # Step 1 - Conv Block
        x_res = self.convblock_res(x)
        
        # Step 2.1 - Squeeze and Excitation 
        x_se  = self.seblock(x_res) # squeeze and then get excitation factor
        
        # Step 2.2
        y = x_res + tf.math.multiply(x_res, x_se) # excited block
        
        # Step 3 - Pooling
        if self.pool is not None:
            return y, self.pool_layer(y)
        else:
            return y
        
    def get_config(self):

        config = {'convblock_res': self.convblock_res, 'seblock': self.seblock}
        if self.pool:
            config['pool_layer'] = self.pool_layer

        return config

############################################################
#                        3D BACKENDS                       #
############################################################

class FocusNetBackend(tf.keras.layers.Layer):
    """
    Folows "FocusNet: Imbalanced Large and Small Organ Segmentation with an End-to-End Deep Neural Network for Head and Neck CT Images"
    """

    def __init__(self, filters, dil_rates, trainable=False, verbose=False):
        super(FocusNetBackend, self).__init__(name='FocusNetBackend')
        
        self.convblock1 = ConvBlock3D(filters=filters[0]   , dilation_rates=dil_rates[0], init_filters=True , bayesian=False, trainable=trainable, pool=(2,2,2), name='Block1')
        self.convblock2 = ConvBlockSERes(filters=filters[1], dilation_rates=dil_rates[1], init_filters=True , bayesian=False, trainable=trainable, pool=None   , name='Block2')
        self.convblock3 = ConvBlockSERes(filters=filters[1], dilation_rates=dil_rates[1], init_filters=False, bayesian=False, trainable=trainable, pool=None   , name='Block3')

    def call(self, x):
        
        conv1, pool1  = self.convblock1(x)
        conv2 = pool2 = self.convblock2(pool1)
        conv3         = self.convblock3(pool2)

        return conv1, pool1, conv2, pool2, conv3

class OrganNetBackend(tf.keras.layers.Layer):
    """
    Folows "A Novel Hybrid Convolutional Neural Network for Accurate Organ Segmentation in 3D Head and Neck CT Images"
    """

    def __init__(self, filters, kernel_size, dil_rates, pooling='double', trainable=False, verbose=True):
        super(OrganNetBackend, self).__init__(name='OrganNetBackend')

        self.verbose = verbose
        self.pooling = pooling

        self.convblock1 = ConvBlock3D(filters=filters[0]   , kernel_size=kernel_size, dilation_rates=dil_rates[0], init_filters=True  , bayesian=False, trainable=trainable, pool=(2,2,1), name='Block1')
        self.convblock2 = ConvBlockSERes(filters=filters[1], kernel_size=kernel_size, dilation_rates=dil_rates[1], init_filters=True  , bayesian=False, trainable=trainable, pool=(2,2,2), name='Block2')
        if pooling == 'double':
            self.convblock3 = ConvBlockSERes(filters=filters[1], kernel_size=kernel_size, dilation_rates=dil_rates[1], init_filters=False , bayesian=False, trainable=trainable, pool=None   , name='Block3NoPool')
        elif pooling == 'triple':
            self.convblock3 = ConvBlockSERes(filters=filters[1], kernel_size=kernel_size, dilation_rates=dil_rates[1], init_filters=False , bayesian=False, trainable=trainable, pool=(2,2,2)   , name='Block3YesPool')
        elif pooling == 'quadruple':
            self.convblock3 = ConvBlockSERes(filters=filters[1], kernel_size=kernel_size, dilation_rates=dil_rates[1], init_filters=False , bayesian=False, trainable=trainable, pool=(2,2,2)   , name='Block3YesPool')
            self.convblock3plus = ConvBlockSERes(filters=filters[1], kernel_size=kernel_size, dilation_rates=dil_rates[1], init_filters=False , bayesian=False, trainable=trainable, pool=(2,2,2)   , name='Block3PlusYesPool')


    def call(self, x):
        
        conv1, pool1  = self.convblock1(x)
        conv2, pool2  = self.convblock2(pool1)
        if self.pooling == 'double':
            conv3         = self.convblock3(pool2)
            pool3         = None
            conv3plus, pool3plus  = None, None
        elif self.pooling == 'triple':
            conv3, pool3  = self.convblock3(pool2)
            conv3plus, pool3plus  = None, None
        elif self.pooling == 'quadruple':
            conv3, pool3  = self.convblock3(pool2)
            conv3plus, pool3plus  = self.convblock3plus(pool3)

        if self.verbose:
            print (' - [OrganNetBackend] x    : ', x.shape)
            print (' - [OrganNetBackend] conv1: {} | pool1: {}'.format(conv1.shape, pool1.shape))
            print (' - [OrganNetBackend] conv2: {} | pool2: {}'.format(conv2.shape, pool2.shape))
            if self.pooling == 'double':
                print (' - [OrganNetBackend] conv3: {}   | pool3: {}'.format(conv3.shape, pool3)) 
            elif self.pooling == 'triple':
                print (' - [OrganNetBackend] conv3: {}   | pool3: {}'.format(conv3.shape, pool3.shape)) 
            elif self.pooling == 'quadruple':
                print (' - [OrganNetBackend] conv3    : {}   | pool3    : {}'.format(conv3.shape    , pool3.shape))
                print (' - [OrganNetBackend] conv3plus: {}   | pool3plus: {}'.format(conv3plus.shape, pool3plus.shape)) 


        return conv1, pool1, conv2, pool2, conv3, pool3, conv3plus, pool3plus
    
    def get_config(self):

        config = {'convblock1': self.convblock1, 'convblock2': self.convblock2, 'convblock3': self.convblock3}
        if self.pooling == 'quadruple':
            config['convblock4'] = self.convblock4

        return config
    
    def build_graph(self, dim):
        x = tf.keras.Input(shape=(None,), name='{}-Input'.format(self.name))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class HDC(tf.keras.layers.Layer):
    """
    Ref: Understanding Convolutions for Semantic Segmentation (https://arxiv.org/abs/1702.08502)
       : https://gist.github.com/prerakmody/ac04e3ee4ee67cf66a4e6251d673993c
    """

    def __init__(self, filters, kernel_size, dil_rates, dropout=None, bayesian=False, trainable=False, verbose=False):
        super(HDC, self).__init__(name='HDC')

        self.verbose = verbose

        self.convblock4 = ConvBlockSERes(filters=filters[0], init_filters=False, kernel_size=kernel_size, dilation_rates=dil_rates[0], dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block4')
        self.convblock5 = ConvBlockSERes(filters=filters[1], init_filters=False, kernel_size=kernel_size, dilation_rates=dil_rates[1], dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block5')
        self.convblock6 = ConvBlockSERes(filters=filters[2], init_filters=False, kernel_size=kernel_size, dilation_rates=dil_rates[2], dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block6')
        self.convblock7 = lambda x: x

    def call(self, x):
        
        conv4 = self.convblock4(x)
        conv5 = self.convblock5(conv4)
        conv6 = self.convblock6(conv5)
        conv7 = self.convblock7(conv6)

        if self.verbose:
            print (' - [HDC] conv4: ', conv4.shape)
            print (' - [HDC] conv5: ', conv5.shape)
            print (' - [HDC] conv6: ', conv6.shape)
            print (' - [HDC] conv7: ', conv7.shape)

        return conv4, conv5, conv6, conv7
    
    def get_config(self):

        config = {'convblock4': self.convblock4, 'convblock5': self.convblock5, 'convblock6': self.convblock6, 'convblock7': self.convblock7}

        return config

class nonHDC(tf.keras.layers.Layer):
    """
    Ref: Understanding Convolutions for Semantic Segmentation (https://arxiv.org/abs/1702.08502)
       : https://gist.github.com/prerakmody/ac04e3ee4ee67cf66a4e6251d673993c
    """

    def __init__(self, filters, dil_rates, dropout=None, bayesian=False, trainable=False, verbose=False):
        super(nonHDC, self).__init__(name='nonHDC')

        self.convblock4 = ConvBlockSERes(filters=filters[0], dilation_rates=dil_rates[0], init_filters=False, dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block4')
        self.convblock5 = ConvBlockSERes(filters=filters[1], dilation_rates=dil_rates[1], init_filters=False, dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block5')
        self.convblock6 = ConvBlockSERes(filters=filters[2], dilation_rates=dil_rates[2], init_filters=False, dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block6')
        self.convblock7 = ConvBlockSERes(filters=filters[3], dilation_rates=dil_rates[3], init_filters=False, dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block7')

    def call(self, x):
        
        conv4 = self.convblock4(x)
        conv5 = self.convblock5(conv4)
        conv6 = self.convblock6(conv5)
        conv7 = self.convblock7(conv6)

        return conv4, conv5, conv6, conv7

class commonHead(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, pooling, dil_rates, filters_upsample, ksize_upsample, class_count, deepsup, dropout=None, bayesian=False, activation=tf.nn.softmax, trainable=False, verbose=False):
        super(commonHead, self).__init__(name='commonHead')

        # Step 0 - Init
        self.deepsup = deepsup
        self.pooling = pooling
        self.kernel_size = kernel_size # for debuggin
        self.filters = filters

        # Step 1 - Handle 'triple', 'quadruple' sampling
        if self.pooling == 'quadruple':
            self.upconvblock8minus = UpConvBlock(filters=filters_upsample[0], kernel_size=ksize_upsample[0], name='UpBlock8Minus')
            self.convblock8minus   = ConvBlockSERes(filters=filters[0], kernel_size=kernel_size, dilation_rates=dil_rates[0], init_filters=True, trainable=trainable, pool=None, name='Block8Minus')
            self.upconvblock8      = UpConvBlock(filters=filters_upsample[0], kernel_size=ksize_upsample[0], name='UpBlock8')
        if self.pooling == 'triple':
            self.upconvblock8  = UpConvBlock(filters=filters_upsample[0], kernel_size=ksize_upsample[0], name='UpBlock8')
        self.convblock8    = ConvBlockSERes(filters=filters[0], kernel_size=kernel_size, dilation_rates=dil_rates[0], init_filters=True, trainable=trainable, pool=None, name='Block8')
        
        # Step 2 - Go on ...
        if filters_upsample[0] is not None:
            self.upconvblock9  = UpConvBlock(filters=filters_upsample[0], kernel_size=ksize_upsample[0], name='UpBlock9')
        else:
            self.upconvblock9 = lambda x: x
        
        if filters_upsample[1] is not None:
            self.upconvblock10 = UpConvBlock(filters=filters_upsample[1], kernel_size=ksize_upsample[1], name='UpBlock10')
        else:
            self.upconvblock10 = lambda x: x
        
        if not bayesian and dropout is None:
            print (' - [models2.py][commonHead] bayesian: ', bayesian, ' || dropout: ', dropout)
            print (' - [models2.py][commonHead] filters: ', filters)
            print ('')
            self.convblock9    = ConvBlockSERes(filters=filters[1], kernel_size=kernel_size, dilation_rates=dil_rates[1], init_filters=True, trainable=trainable, pool=None, name='Block9')
            self.convblock10   = ConvBlockSERes(filters=filters[2], kernel_size=kernel_size, dilation_rates=dil_rates[2], init_filters=True, trainable=trainable, pool=None, name='Block10')
            self.convblock11   = tf.keras.layers.Conv3D(filters=class_count, strides=(1,1,1), kernel_size=(1,1,1), padding='same'
                                    , dilation_rate=(1,1,1)
                                    , activation=activation
                                    , name='Block11'
                                    )

        else:
            self.convblock9    = ConvBlockSERes(filters=filters[1], kernel_size=kernel_size, dilation_rates=dil_rates[1], init_filters=True, dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block9-Flip')
            self.convblock10   = ConvBlockSERes(filters=filters[2], kernel_size=kernel_size, dilation_rates=dil_rates[2], init_filters=True, dropout=dropout, bayesian=bayesian, trainable=trainable, pool=None, name='Block10-Flip')
            self.convblock11   = tfp.layers.Convolution3DFlipout(filters=class_count, strides=(1,1,1), kernel_size=(1,1,1), padding='same'
                                    , dilation_rate=(1,1,1)
                                    , activation=activation
                                    , name='Block11-Flip'
                                    )
        
    def call(self, conv7, conv3plus, conv3, conv2, conv1):
        
        # pdb.set_trace()
        conv8minus = None
        if self.pooling == 'quadruple':
            upconv8minus = self.upconvblock8minus(conv7)
            conv8minus   = self.convblock8minus(tf.concat([conv3plus, upconv8minus], axis=-1))
            upconv8      = self.upconvblock8(conv8minus)
            conv8        = self.convblock8(tf.concat([conv3, upconv8], axis=-1))

        elif self.pooling == 'triple':
            upconv8 = self.upconvblock8(conv7)
            conv8 = self.convblock8(tf.concat([conv3, upconv8], axis=-1))
        else:      
            conv8 = self.convblock8(tf.concat([conv3, conv7], axis=-1))
        
        upconv9 = self.upconvblock9(conv8)
        conv9   = self.convblock9(tf.concat([conv2, upconv9], axis=-1))
        
        upconv10 = self.upconvblock10(conv9)
        conv10   = self.convblock10(tf.concat([conv1, upconv10], axis=-1))
        
        conv11   = self.convblock11(conv10)

        return conv8minus, conv8, conv9, conv10, conv11
    
    def get_config(self):
        
        config = {}
        if self.pooling == 'quadruple':
            config['upconvblock8minus'] = self.upconvblock8minus
            config['convblock8minus']   = self.convblock8minus
            config['upconvblock8']      = self.upconvblock8
        if self.pooling == 'triple':
            config['upconvblock8'] = self.upconvblock8
        config['convblock8']    = self.convblock8

        config['upconvblock9']  = self.upconvblock9
        config['convblock9']    = self.convblock9
        
        config['upconvblock10'] = self.upconvblock10
        config['convblock10']   = self.convblock10
        
        config['convblock11'] = self.convblock11

        return config

############################################################
#                         3D MODELS                        #
############################################################

class OrganNet(tf.keras.Model):

    def __init__(self, class_count, kernel='3d', pooling='double', model_weights='light', hdc=True, dropout=None, bayesian=True, deepsup=False, bayesianhead=False, activation=tf.nn.softmax, trainable=False, verbose=False):
        super(OrganNet, self).__init__(name='OrganNet')

        self.pooling = pooling
    
        # model_weights = 'heavy'
        if model_weights == 'lighter':
            backend_filters   = [[8,8], [16,16]]
            backend_dil_rates = [[(1,1,1),(1,1,1)], [(1,1,1),(1,1,1)]]

            hdc_core_filters      = [[16,16,16,16], [16,16,16,16], [16,16,16,16]]
            hdc_core_dil_rates    = [[(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)]]

            nonhdc_core_filters      = [[16,16,16,16], [16,16,16,16], [16,16,16,16], [16,16,16,16]]
            nonhdc_core_dil_rates = [[(2,2,1), (2,2,1), (2,2,1)], [(3,3,1), (3,3,1), (3,3,1)], [(6,6,1), (6,6,1), (6,6,1)], [(12,12,1), (12,12,1), (12,12,1)]]

            if class_count not in [1, 2, 3]:
                head_filters      = [[16,16], [8,8], [class_count, class_count]]
            else:
                head_filters      = [[16,16], [8,8], [4, 4]]
            head_dil_rates    = [[(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)]]
            head_filters_upsample = [16,8]

        elif model_weights == 'light':
            backend_filters   = [[16,16], [32,32]]
            backend_dil_rates = [[(1,1,1),(1,1,1)], [(1,1,1),(1,1,1)]]

            hdc_core_filters      = [[32,32,32,32], [32,32,32,32], [32,32,32,32]]
            hdc_core_dil_rates    = [[(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)]]

            nonhdc_core_filters      = [[32,32,32], [32,32,32], [32,32,32], [32,32,32]]
            nonhdc_core_dil_rates = [[(2,2,1), (2,2,1), (2,2,1)], [(3,3,1), (3,3,1), (3,3,1)], [(6,6,1), (6,6,1), (6,6,1)], [(12,12,1), (12,12,1), (12,12,1)]]

            if class_count not in [1, 2, 3]:
                head_filters      = [[32,32], [16,16], [class_count, class_count]]
            else:
                head_filters      = [[32,32], [16,16], [8, 8]]
            head_dil_rates    = [[(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)]]
            head_filters_upsample = [32,16]
            
        elif model_weights == 'heavy':
        
            backend_filters   = [[32,32], [48,48]]
            backend_dil_rates = [[(1,1,1),(1,1,1)], [(1,1,1),(1,1,1)]]

            hdc_core_filters      = [[48,48,48,48], [48,48,48,48], [48,48,48,48]]
            hdc_core_dil_rates    = [[(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)]]

            nonhdc_core_filters      = [[48,48,48], [48,48,48], [48,48,48], [48,48,48]]
            nonhdc_core_dil_rates = [[(2,2,1), (2,2,1), (2,2,1)], [(3,3,1), (3,3,1), (3,3,1)], [(6,6,1), (6,6,1), (6,6,1)], [(12,12,1), (12,12,1), (12,12,1)]]

            if class_count not in [1, 2, 3]:
                head_filters      = [[48,48], [32,32], [class_count, class_count]]
            else:
                head_filters      = [[48,48], [32,32], [8, 8]]
            head_dil_rates    = [[(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)]]
            head_filters_upsample = [48,48]

        if kernel == '2d':
            kernel_size        = (3,3,1)
            head_filters_ksize = [(2,2,1), (2,2,1)]
        elif kernel == '3d':
            kernel_size        = (3,3,3)
            head_filters_ksize = [(2,2,2), (2,2,1)]

        print ('')
        print (' - [models2.py][OrganNet] weights         : ', model_weights)
        print (' - [models2.py][OrganNet] kernel_size     : ', kernel_size)
        print (' - [models2.py][OrganNet] kernel_size(up) : ', head_filters_ksize)
        print (' - [models2.py][OrganNet] pooling         : ', pooling)
        print (' - [models2.py][OrganNet] bayesian        : ', bayesian)
        print (' - [models2.py][OrganNet] dropout         : ', dropout)
        print (' - [models2.py][OrganNet] bayesianhead    : ', bayesianhead)
        print ('')

        self.backend = OrganNetBackend(filters=backend_filters, kernel_size=kernel_size, dil_rates=backend_dil_rates, pooling=pooling, trainable=trainable, verbose=verbose)

        if not bayesianhead:
            if hdc:
                self.core    = HDC(filters=hdc_core_filters, kernel_size=kernel_size, dil_rates=hdc_core_dil_rates, dropout=dropout, bayesian=bayesian, trainable=trainable, verbose=verbose)
            else:
                self.core    = nonHDC(filters=nonhdc_core_filters, dil_rates=nonhdc_core_dil_rates, dropout=dropout, bayesian=bayesian, trainable=trainable, verbose=verbose)    

            self.head    = commonHead(filters=head_filters, kernel_size=kernel_size, pooling=pooling, dil_rates=head_dil_rates, filters_upsample=head_filters_upsample, ksize_upsample=head_filters_ksize, class_count=class_count, deepsup=deepsup
                        , activation=activation
                        , trainable=trainable, verbose=verbose)

        else:
            if hdc:
                self.core    = HDC(filters=hdc_core_filters, kernel_size=kernel_size, dil_rates=hdc_core_dil_rates, dropout=None, bayesian=False, trainable=trainable, verbose=verbose)
            else:
                self.core    = nonHDC(filters=nonhdc_core_filters, dil_rates=nonhdc_core_dil_rates, dropout=None, bayesian=False, trainable=trainable, verbose=verbose)    

            self.head    = commonHead(filters=head_filters, kernel_size=kernel_size, pooling=pooling, dil_rates=head_dil_rates, filters_upsample=head_filters_upsample, ksize_upsample=head_filters_ksize, class_count=class_count, deepsup=deepsup
                        , dropout=dropout, bayesian=bayesian
                        , activation=activation
                        , trainable=trainable, verbose=verbose)

    def call(self, x):
        
        # print (' - [OrganNet] x: ', x.shape)

        if 1:
            # Step 1 - Backend
            conv1, pool1, conv2, pool2, conv3, pool3, conv3plus, pool3plus = self.backend(x)

            # Step 2 - Core
            if self.pooling == 'double':
                conv4, conv5, conv6, conv7 = self.core(conv3)
            elif self.pooling == 'triple':
                conv4, conv5, conv6, conv7 = self.core(pool3)
            elif self.pooling == 'quadruple':
                conv4, conv5, conv6, conv7 = self.core(pool3plus)

            # Step 3 - Head
            conv8minus, conv8, conv9, conv10, conv11 = self.head(conv7, conv3plus, conv3, conv2, conv1)

            return conv11
        
        else:
            # Step 1 - Backend
            conv1, pool1  = self.backend.convblock1(x)
            conv2, pool2  = self.backend.convblock2(pool1)
            if self.backend.pooling == 'double':
                conv3         = self.backend.convblock3(pool2)
                pool3         = None
            elif self.backend.pooling == 'triple':
                conv3, pool3  = self.backend.convblock3(pool2)
            
            # Step 2 - Core
            if self.pooling == 'double':
                conv4 = self.core.convblock4(conv3)
                conv5 = self.core.convblock5(conv4)
                conv6 = self.core.convblock6(conv5)
                conv7 = self.core.convblock7(conv6)
                # conv4, conv5, conv6, conv7 = self.core(conv3)

            elif self.pooling == 'triple':
                conv4 = self.core.convblock4(pool3)
                conv5 = self.core.convblock5(conv4)
                conv6 = self.core.convblock6(conv5)
                conv7 = self.core.convblock7(conv6)
                # conv4, conv5, conv6, conv7 = self.core(pool3)

            # Step 3 - Head
            if self.head.pooling == 'triple':
                upconv8 = self.head.upconvblock8(conv7)
                conv8 = self.head.convblock8(tf.concat([conv3, upconv8], axis=-1))
            else:      
                conv8 = self.head.convblock8(tf.concat([conv3, conv7], axis=-1))
            
            upconv9 = self.head.upconvblock9(conv8)
            
            conv9   = self.head.convblock9(tf.concat([conv2, upconv9], axis=-1))
            
            upconv10 = self.head.upconvblock10(conv9)
            conv10   = self.head.convblock10(tf.concat([conv1, upconv10], axis=-1))
            
            conv11   = self.head.convblock11(conv10)
            # conv8, conv9, conv10, conv11 = self.head(conv7, conv3, conv2, conv1)

            return conv11

    def get_config(self):

        config = {'backend': self.backend, 'core': self.core, 'head': self.head}

        return config
    
    def build_graph(self, dim):
        x = tf.keras.Input(shape=(dim), name='{}-Input'.format(self.name))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class FocusNet(tf.keras.Model):

    def __init__(self, class_count, hdc=False, dropout=None, bayesian=True, deepsup=False, trainable=False, verbose=False):
        super(FocusNet, self).__init__(name='FocusNet')

        if 1:
            backend_filters   = [[16,16], [32,32]]
            backend_dil_rates = [[(1,1,1),(1,1,1)], [(1,1,1),(1,1,1)]]

            hdc_core_filters      = [[32,32,32,32], [32,32,32,32], [32,32,32,32]]
            hdc_core_dil_rates    = [[(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)], [(1,1,1),(3,3,1),(5,5,1),(9,9,1)]]

            nonhdc_core_filters      = [[32,32,32], [32,32,32], [32,32,32], [32,32,32]]
            nonhdc_core_dil_rates = [[(2,2,1), (2,2,1), (2,2,1)], [(3,3,1), (3,3,1), (3,3,1)], [(6,6,1), (6,6,1), (6,6,1)], [(12,12,1), (12,12,1), (12,12,1)]]

            head_filters      = [[32,32], [16,16], [class_count, class_count]]
            head_dil_rates    = [[(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)], [(1,1,1), (1,1,1)]]
            head_filters_upsample = [None,16]
            head_filters_ksize    = [None, (2,2,2)]

            print (' - [models2.py][FocusNet] bayesian: ', bayesian)

        self.backend = FocusNetBackend(filters=backend_filters, dil_rates=backend_dil_rates, trainable=trainable, verbose=verbose)
        if hdc:
            self.core    = HDC(filters=hdc_core_filters, dil_rates=hdc_core_dil_rates, dropout=dropout, bayesian=bayesian, trainable=trainable, verbose=verbose)
        else:
            self.core    = nonHDC(filters=nonhdc_core_filters, dil_rates=nonhdc_core_dil_rates, dropout=dropout, bayesian=bayesian, trainable=trainable, verbose=verbose)
        self.head    = commonHead(filters=head_filters, dil_rates=head_dil_rates, filters_upsample=head_filters_upsample, ksize_upsample=head_filters_ksize, class_count=class_count, deepsup=deepsup, trainable=trainable, verbose=verbose)

    def call(self, x):

        # Step - Backend
        conv1, pool1, conv2, pool2, conv3 = self.backend(x)
        
        # Step 2 - Core
        conv4, conv5, conv6, conv7 = self.core(conv3)
        
        # Step 3 - Head
        conv8, conv9, conv10, conv11 = self.head(conv7, conv3, conv2, conv1)

        return conv11
    
    def get_config(self):

        config = {'backend': self.backend, 'core': self.core, 'head': self.head}
        return config
    
    def build_graph(self, dim):
        x = tf.keras.Input(shape=(dim), name='{}-Input'.format(self.name))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

############################################################
#                           UTILS                          #
############################################################
@tf.function
def write_model_trace(model, X):
    return model(X)
    
############################################################
#                           MAIN                           #
############################################################

if __name__ == "__main__":

    X = tf.random.normal((2,140,140,40,1))

    if 0:
        print ('\n ------------------- FocusNet ------------------- ')
        model = FocusNet(class_count=10)
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)
        model.summary() # ~ nonBayes(550K), Bayes(900K) params
        print (model.losses)
    
    elif 0:
        print ('\n ------------------- OrganNet ------------------- ')
        model = OrganNet(class_count=10)
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)
        model.summary() # ~ nonBayes(550K), Bayes(900K) params
        print (model.losses)
    
    elif 0:
        print ('\n ------------------- OrganNet (bayesian=False, dropout=0.3) ------------------- ')
        model = OrganNet(class_count=10, bayesian=False, dropout=0.3)
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)
        model.summary() # ~ nonBayes(550K), Bayes(900K) params
        print (model.losses)
    
    # OrganNet (for prostate)
    elif 0:
        X = tf.random.normal((2,200,200,28,1))
        print ('\n ------------------- OrganNet (class_count=1, bayesian=False, dropout=None, activation=tf.nn.sigmoid) ------------------- ')
        model = OrganNet(class_count=1, hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=True)
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)
        model.summary() # ~ nonBayes(550K), Bayes(900K) params
        print (model.losses)
    
    # OrganNet(pooling=triple) (for prostate)
    elif 0:
        X = tf.random.normal((2,200,200,28,1))
        print ('\n ------------------- OrganNet (class_count=1, bayesian=True, dropout=None, activation=tf.nn.sigmoid) ------------------- ')
        model = OrganNet(class_count=1, pooling='triple', hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=True)
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)
        model.summary() # ~ nonBayes(550K), Bayes(900K) params
        print (model.losses)

    # OrganNet(pooling=triple) (for prostate) and for visualization in netron.app
    elif 0:
        raw_shape = (200,200,28,1)
        print ('\n ------------------- OrganNet (class_count=1, pooling=triple, hdc=True, bayesian=False, dropout=None, activation=tf.nn.sigmoid) ------------------- ')
        model = OrganNet(class_count=1, pooling='triple', hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=True)
        model = model.build_graph(raw_shape)
        model.summary(line_length=150)

		# Using the save() function
        _ = model(tf.ones((1,*raw_shape)))
        model.save('OrganNetPool3.h5', save_format='h5')  # Loads in www.netron.app for viz purposes, shows the high level blocks
        # model.save('OrganNetPool3', save_format='tf')

        # Using the .to_json() function
        # import json
        # with open('OrganNetPool3-Functional.json', 'w') as fp:
        #     json.dump(json.loads(model.to_json()), fp, indent=4)

        # Using the .get_config() function
        # import json
        # with open('OrganNetPool3.json', 'w') as fp:
        #     json.dump(json.loads(model.get_config()), fp, indent=4)

        tf.keras.utils.plot_model(model, 'OrganNetPool3.png', show_shapes=True, expand_nested=True) # pip install pydot graphviz
        # tf.keras.utils.model_to_dot(model, show_shapes=True, expand_nested=True, subgraph=True)

        # # Using tf2onnx
        # import onnx
        # import tf2onnx
        # model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model=model, input_signature=[tf.TensorSpec((1,*raw_shape), tf.float32, name='ModelInput')]) # tensorspec ensures that the batch size shows up properly in tf2onnx
        # model_onnx = onnx.shape_inference.infer_shapes(model_proto)
        # tf2onnx.utils.save_protobuf('OrganNetPool3.onnx', model_onnx)

        # # Using tensorboard
        # tf.summary.trace_on(graph=True, profiler=False)
        # _ = write_model_trace(model, tf.ones(shape=(1,*raw_shape), dtype=tf.float32))
        # writer = tf.summary.create_file_writer(str('OrganNetPool3'))
        # with writer.as_default():
        #     tf.summary.trace_export(name=model.name, step=0, profiler_outdir=None)
        #     writer.flush()
        #     print (' - Run command --> tensorboard --logdir=OrganNetPool3 --port=6100')
        
        pass
    
    # OrganNet(pooling=triple) (for prostate) and for visualization in netron.app
    elif 0:
        raw_shape = (200,200,28,1)
        # print ('\n ------------------- OrganNet (class_count=1, pooling=triple, hdc=True, bayesian=True, dropout=None, activation=tf.nn.sigmoid) ------------------- ')
        # model = OrganNet(class_count=1, pooling='triple', hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.sigmoid, verbose=True)
        model = model.build_graph(raw_shape)
        model.summary(line_length=150)

		# Using the save() function
        _ = model(tf.ones((1,*raw_shape)))
        model.save('OrganNetBayesPool3.h5', save_format='h5')  # Loads in www.netron.app for viz purposes, shows the high level blocks
        # model.save('OrganNetBayesPool3.h5', save_format='tf')

        # Using tf2onnx
        # import onnx
        # import tf2onnx
        # model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model=model, input_signature=[tf.TensorSpec((1,*raw_shape), tf.float32, name='ModelInput')]) # tensorspec ensures that the batch size shows up properly in tf2onnx
        # model_onnx = onnx.shape_inference.infer_shapes(model_proto)
        # tf2onnx.utils.save_protobuf('OrganNetBayesPool3.onnx', model_onnx)

        # Using tensorboard
        # tf.summary.trace_on(graph=True, profiler=False)
        # _ = write_model_trace(model, tf.ones(shape=(1,*raw_shape), dtype=tf.float32))
        # writer = tf.summary.create_file_writer(str('OrganNetBayesPool3'))
        # with writer.as_default():
        #     tf.summary.trace_export(name=model.name, step=0, profiler_outdir=None)
        #     writer.flush()
        #     print (' - Run command --> tensorboard --logdir=OrganNetBayesPool3 --port=6100')
        pass
    
    # OrgaNetBayesianHead
    elif 0:
        print ('\n ------------------- OrganNet (bayesian=True, dropout=None, bayesianhead=True) ------------------- ')
        model = OrganNet(class_count=10, bayesian=True, dropout=None, bayesianhead=True)
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)
        model.summary() #  ~ nonBayes(550K), and this is 580K params
        print (model.losses)
    
    # 2D OrganNet(pooling=quadruple)
    elif 0:
        raw_shape = (512,512,1,3)
        
        print ('\n ------------------- OrganNet (class_count=3, kernel=2d, bayesian=False, dropout=None, activation=tf.nn.softmax) ------------------- ')
        model = OrganNet(class_count=3, kernel='2d', pooling='quadruple', model_weights='heavy', hdc=True, bayesian=False, bayesianhead=False, dropout=None, activation=tf.nn.softmax, verbose=True) # pooling=quadruple gives receptive field of (278,278)
        model = model.build_graph(raw_shape)
        model.summary(line_length=150)

        X = tf.random.normal((2,*raw_shape))
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)

    # OrganNet(pooling=triple, bayesianHead=True, model_weights=[light, heavy])
    elif 0:

        raw_shape = (200,200,28,1)
        
        print ('\n ------------------- OrganNet (class_count=1, pooling=triple, hdc=True, bayesian=True, dropout=None, activation=tf.nn.sigmoid) ------------------- ')
        model = OrganNet(class_count=1, pooling='triple', model_weights='heavy', hdc=True, bayesian=True, bayesianhead=True, dropout=None, activation=tf.nn.sigmoid, verbose=True)

        model = model.build_graph(raw_shape)
        model.summary(line_length=150)

        X = tf.random.normal((2,*raw_shape))
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)

    # OrganNet(class_count=2, pooling=double, bayesianHead=False, model_weights=lighter)
    elif 1:
        X = tf.random.normal((2,200,200,28,1))
        print ('\n ------------------- OrganNet (class_count=2, model_weights=lighter, bayesian=True, dropout=None, activation=tf.nn.softmax) ------------------- ')
        model = OrganNet(class_count=2, model_weights='lighter', hdc=True, bayesian=True, bayesianhead=False, dropout=None, activation=tf.nn.softmax, verbose=True)
        y_predict = model(X, training=True)
        print (' - y_predict: ', y_predict.shape)
        model.summary() # ~ nonBayes(550K), Bayes(900K) params
        print (model.losses)

    pdb.set_trace()