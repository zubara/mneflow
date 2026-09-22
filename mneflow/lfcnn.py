# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:11:39 2025

@author: ipzub
"""

import tensorflow as tf
import keras

import numpy as np

from mne import channels, evoked, create_info, Info
from mne.filter import filter_data


from scipy.stats import spearmanr, pearsonr
from scipy.signal import welch

from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
        
from mneflow.layers import LFTConv, VARConv, DeMixing, FullyConnected, TempPooling, LFTConvTranspose
from keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D, LSTM
from keras.layers import Flatten, Dropout, BatchNormalization
from keras.initializers import Constant
from mneflow.data import Dataset
from mneflow.models import BaseModel
from collections import defaultdict

class LFCNN(BaseModel):
    """LF-CNN. Includes basic parameter interpretation options.

    For details see [1].
    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """

        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object; ``meta.model_specs`` is populated with
            this model's default hyperparameters (see below) where
            not already set.

        dataset : mneflow.Dataset, optional
            Dataset object. Defaults to None (built from ``meta``).

        specs_prefix : bool, optional
            See :meth:`mneflow.models.BaseModel.__init__`. Defaults
            to False.

        specs : dict, optional
                If provided, merged into ``meta.model_specs`` before
                applying the defaults below. Dictionary of model
                hyperparameters {

        n_latent : int
            Number of latent components.
            Defaults to 32.

        nonlin : callable
            Activation function of the temporal convolution layer.
            Defaults to tf.nn.relu

        filter_length : int
            Length of spatio-temporal kernels in the temporal
            convolution layer. Defaults to 7.

        pooling : int
            Pooling factor of the max pooling layer. Defaults to 2

        pool_type : str {'avg', 'max'}
            Type of pooling operation. Defaults to 'max'.

        padding : str {'SAME', 'FULL', 'VALID'}
            Convolution padding. Defaults to 'SAME'.}

        stride : int
        Stride of the max pooling layer. Defaults to 2.

        """
        self.scope = 'lfcnn'
        self.nfft = 128
        if specs:
            meta.update(model_specs=specs)
        #specs = meta.model_specs
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', ['fc', 'dmx', 'tconv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        #specs.setdefault('model_path',  self.dataset.h_params['save_path'])
        super(LFCNN, self).__init__(meta, dataset, specs_prefix)
        #super().__init__(meta)
        print('Init end')

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """

        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
                            axis=3, specs=self.specs)
        self.dmx_out = self.dmx(self.inputs)

        self.tconv = LFTConv(size=self.specs['n_latent'],
                             nonlin=self.specs['nonlin'],
                             filter_length=self.specs['filter_length'],
                             padding=self.specs['padding'],
                             specs=self.specs
                             )
        self.tconv_out = self.tconv(self.dmx_out)

        self.pool = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding='SAME'
                                  )
        self.pooled = self.pool(self.tconv_out)

        self.dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(self.pooled)

        # self.fin_fc0 = FullyConnected(size=self.specs['n_latent'], nonlin=keras.activations.linear,
        #                     specs=self.specs)
        # fc0_out = self.fin_fc0(self.dropout)

        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=keras.activations.linear,
                            specs=self.specs)
        #y_pred = self.fin_fc(fc0_out)
        self.y_pred = self.fin_fc(self.dropout)
        print('Init end')
        return self.y_pred

    def build_encoder(self, encoder_specs
                      #inputs='y_pred', conv='full',
                      #reconstruction_loss=keras.losses.MAE
                      ):
        """Build computational graph for an interpretable Generator
        (decoder) that reconstructs the input from either the model's
        predictions or its latent activations.

        Parameters
        ----------
        encoder_specs : dict
            Dictionary of encoder hyperparameters. Expected keys
            include:

            - ``inputs`` : str {'y_pred', 'activations'} -- source of
              the signal fed into the encoder
            - ``conv`` : str {'full', 'depthwise'} -- type of
              transposed convolution used to upsample the temporal
              dimension
            - ``nonlin`` : callable
            - ``filter_length`` : int
            - ``stride`` : int
            - ``n_latent`` : int
            - ``l2_lambda`` : float
            - ``loss`` : callable or str
            - ``learn_rate`` : float

        Returns
        -------
        None
            This method does not return a value. It builds and
            compiles the encoder graph in place, setting
            ``self.enc_fc``, ``self.enc_tconv_activations_r``,
            ``self.enc_tconv_trans``, ``self.de_dmx``,
            ``self.X_pred`` and the compiled Keras model
            ``self.km_enc``.
        """
        self.specs['dropout'] = 0.05
        self.encoder_specs = encoder_specs

        print("Freezing the decoder")

        self.enc_fc = FullyConnected(scope='def', size=self.fin_fc.w.shape[0],
                                     nonlin=self.encoder_specs['nonlin'],
                                     specs=self.encoder_specs)
        #Get single fully-connected layer
        if self.encoder_specs['inputs'] == 'y_pred':
            enc_tconv_activations =  self.enc_fc(self.y_pred)
        elif self.encoder_specs['inputs'] == 'activations':
            enc_tconv_activations =  self.enc_fc(self.pooled)

        print("enc_tconv_activations: ", enc_tconv_activations.shape, self.pooled.shape)
        self.enc_tconv_activations_r = keras.layers.Reshape(self.pooled.shape[1:])

        enc_tconv_activations_r =  self.enc_tconv_activations_r(enc_tconv_activations)
        default_output_t = enc_tconv_activations_r.shape[2] * self.encoder_specs['stride']

        #Conv2DTranspose with restoring original shape
        diff_padding = default_output_t - self.dataset.h_params['n_t']
        if self.dataset.h_params['n_t']%self.specs['stride'] == 0:
            n_pads=None
        elif self.dataset.h_params['n_t']%self.specs['stride'] == 1:
            n_pads = max(0, self.specs['stride'] - diff_padding)
            padding = (n_pads, 1)
            print(padding)
        else :
            n_pads = max(0, self.specs['stride'] - diff_padding)
            padding = (n_pads, 1)
            print(padding)

        print("before upsampling: ", enc_tconv_activations_r.shape)
        enc_dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(enc_tconv_activations_r)

        if self.encoder_specs['conv'] == 'depthwise':
            enc_dropout_split = keras.ops.split(enc_dropout,
                                                   indices_or_sections=self.encoder_specs['n_latent'],
                                                   axis=-1)
            self.enc_tconv_trans = [keras.layers.Conv1DTranspose(
                                        filters=1,
                                        kernel_size=self.encoder_specs['filter_length'],
                                        strides=self.encoder_specs['stride'],
                                        padding='same',
                                        output_padding=(n_pads),
                                        data_format='channels_last',
                                        dilation_rate=1,
                                        activation=self.encoder_specs['nonlin'], #keras.activations.linear,#
                                        use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer='glorot_uniform',
                                        kernel_regularizer=keras.regularizers.l2(self.encoder_specs['l2_lambda']),
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        #kernel_constraint=keras.constraints.UnitNorm(axis=[0, 1]),
                                        bias_constraint=None,
                                        ) for i in range(self.specs['n_latent'])]
            enc_tconv_tans_out = [tconv(enc_dropout_split[i][:, 0, :, :]) for i, tconv in enumerate(self.enc_tconv_trans)]
            print('Build {} separate ConvTranspose layers each returning {}'.format(len(self.enc_tconv_trans),
                                                                                    enc_tconv_tans_out[0].shape))
            enc_deconv = keras.ops.expand_dims(keras.ops.concatenate(
                                                  enc_tconv_tans_out,
                                                  axis=-1), 1)
        else:
            self.enc_tconv_trans=keras.layers.Conv2DTranspose(
                                    filters=1,
                                    kernel_size=(self.encoder_specs['filter_length'],
                                                 self.encoder_specs['n_latent']),
                                    strides=(self.encoder_specs['stride'], 1),
                                    padding='same',
                                    output_padding=padding,
                                    data_format='channels_first',
                                    dilation_rate=(1, 1),
                                    activation=tf.encoder_specs['nonlin'],
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='glorot_uniform',
                                    kernel_regularizer=keras.regularizers.l2(self.encoder_specs['l2_lambda']),
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    #kernel_constraint=keras.constraints.UnitNorm(axis=[0, 1]),
                                    bias_constraint=None,
                                    )

            enc_deconv = self.enc_tconv_trans(enc_dropout)

        print("Enc_deconv:", enc_deconv.shape)
        print("TCONV OUT:", self.tconv_out.shape)
        assert enc_deconv.shape == self.tconv_out.shape

        self.de_dmx = DeMixing(scope='dede',
                               size=self.dataset.h_params['n_ch'],
                               nonlin=self.encoder_specs['nonlin'],
                               axis=3,
                               specs=self.encoder_specs)

        self.X_pred = self.de_dmx(enc_deconv)

        self.km_enc = keras.Model(inputs=self.inputs, outputs=self.X_pred)

        self.meta.train_params['enc_loss'] = [#CosMSE
                                              self.encoder_specs['loss'],
                                              #keras.losses.CosineSimilarity(axis=[3]),
                                              #keras.losses.MSE
                                              ]
        self.km_enc.compile(optimizer=keras.optimizers.Adam(self.encoder_specs['learn_rate']),
                        loss=self.meta.train_params['enc_loss'],
                        metrics=[keras.metrics.RootMeanSquaredError(name="RMSE")],
                        #loss_weights=[alpha, 1.-alpha]
                        )


    def enc_reconstruct(self, method='weight'):
        """Compute and return the mean reconstructed input for each
        class, obtained by propagating the class-conditional mean
        activations through the trained encoder (decoder) graph.

        Parameters
        ----------
        method : str, optional
            Which patterns to use as the encoder's input; one of
            {'full', 'weight', 'compwise_loss', 'output_corr',
            'combined', 'shap'}. Defaults to 'weight'.

        Returns
        -------
        reconstructed : np.array
            Mean reconstructed input for each class.
        """


        patterns_struct = self.compute_patterns(shapley_order=0, methods=['weight'])


        #Get spatial encoder weights
  
        #Get mean activations of enc_fc for each class.
        if self.encoder_specs['inputs'] == 'activations':
            ccms = patterns_struct['ccms']['pooled'] # (n_t, n_comp, n_classes, n_folds)
            f_enc = self.enc_tconv_activations_r(self.enc_fc(ccms.transpose([3, 1, 2, 0])))
        elif self.encoder_specs['inputs'] == 'y_pred':
            ccms = patterns_struct['ccms']['fc'] # (n_t, n_comp, n_classes, n_folds)
            f_enc = self.enc_tconv_activations_r(self.enc_fc(ccms.transpose()))
        #From now on class-means are treated as "samples"


        print(f_enc.shape) # n_classes, 1, n_t, n_components
        if self.encoder_specs['conv'] == 'depthwise':
            f_enc_split = keras.ops.split(f_enc,
                                               indices_or_sections=self.specs['n_latent'],
                                               axis=-1)

            comp_ts = [tconv_trans(fes[:, 0, :, :]) for tconv_trans, fes  in zip(self.enc_tconv_trans,
                                                                 f_enc_split)]

            enc_deconv = keras.ops.expand_dims(keras.ops.concatenate(
                                              comp_ts,
                                              axis=-1), 1)
        elif self.encoder_specs['conv'] == 'full':
            enc_deconv = self.enc_tconv_trans(f_enc)

        reconstructed = self.de_dmx(enc_deconv)
        print("Returning rconstructed input for {} variables".format(reconstructed.shape[0]))
        return reconstructed


    def compute_enc_patterns(self, inputs=None):
        """Compute spatial patterns of the encoder (decoder) by
        propagating inputs through the fitted encoder graph.

        Parameters
        ----------
        inputs : np.array, optional
            Inputs to the encoder's fully-connected layer. If not
            provided, an identity matrix of shape
            ``(self.out_dim, self.out_dim)`` is used, so that each
            "sample" isolates the pattern associated with one output
            unit.

        Returns
        -------
        patterns : np.array
            Encoder-derived spatial patterns, with singleton
            dimensions removed.
        """
        if not np.any(inputs):
            print('Using fake inputs')
            inputs = np.identity(self.out_dim)
        enc_fc_out = self.enc_fc(inputs)
        pooled = self.enc_tconv_activations_r(enc_fc_out)
        unpooled_wfs = self.enc_tconv_trans(pooled)
        patterns = self.de_dmx(unpooled_wfs)

        return np.squeeze(patterns) #, unpooled_wfs

    def train_encoder(self, n_epochs, eval_step=None, min_delta=1e-6,
                      mode='single_fold', early_stopping=3,
                      collect_patterns=False):
        """Train the encoder (decoder) graph built by
        :meth:`build_encoder` to reconstruct the input, while keeping
        the (already-trained) classifier/regressor weights frozen.

        Parameters
        ----------
        n_epochs : int
            Maximum number of training epochs.

        eval_step : int, optional
            Number of training steps (batches) per epoch. Defaults to
            None (one epoch equals one full pass over the training
            data).

        min_delta : float, optional
            Minimum change in the monitored validation loss to
            qualify as an improvement for early stopping. Defaults to
            1e-6.

        mode : str {'single_fold', 'cv', 'loso'}, optional
            Training regime. 'single_fold' trains on the current
            fold only; 'cv' loops over all folds in
            ``self.dataset.h_params['folds']``; 'loso' is not
            implemented. Defaults to 'single_fold'.

        early_stopping : int, optional
            Patience (in epochs) for early stopping on the validation
            loss. Defaults to 3.

        collect_patterns : bool, optional
            Currently unused placeholder for collecting patterns
            during encoder training. Defaults to False.

        Returns
        -------
        None
            Populates ``self.cv_enc_losses`` and
            ``self.cv_enc_metrics`` with the per-fold validation loss
            and metric values.
        """
        self.km.trainable = False
        if mode == 'single_fold':
            n_folds = 1
            fold = self.current_fold
        elif mode == 'cv':
            n_folds = len(self.dataset.h_params['folds'][0])
            fold = 0
            print("Running cross-validation with {} folds".format(n_folds))
        elif mode == "loso":
            print("LOSO Encoder training is not implementerd")

        self.cv_enc_losses = []
        self.cv_enc_metrics = []

        for jj in range(n_folds):
            train, val = self.dataset._build_dataset(self.dataset.h_params['train_paths'],
                                               train_batch=self.dataset.training_batch,
                                               test_batch=self.dataset.validation_batch,
                                               split=True, val_fold_ind=fold)


            dataset_train = train.map(lambda x, y : (x, x))
            dataset_val = val.map(lambda x, y : (x, x))
            stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=min_delta,
                                                          patience=early_stopping,
                                                          restore_best_weights=True)
            self.t_hist = self.km_enc.fit(dataset_train,
                                       validation_data=dataset_val,
                                       epochs=n_epochs, steps_per_epoch=eval_step,
                                       shuffle=True,
                                       validation_steps=self.dataset.validation_steps,
                                       callbacks=[stop_early], verbose=2,
                                       )
            losses, metrics = self.km_enc.evaluate(dataset_val,
                                               steps=self.dataset.validation_steps,
                                               verbose=0)
            self.cv_enc_losses.append(losses)
            self.cv_enc_metrics.append(metrics)
            fold += 1


        print("""{} with {} fold(s) completed. \n
              Validation Performance:
              Loss: {:.4f}.
              Metric: {:.4f}"""
              .format("Encoder training with", n_folds,
                      np.mean(self.cv_enc_losses), np.mean(self.cv_enc_metrics)))
        self.km.trainable = True


    def get_config(self):
            """Return a minimal config dict referencing this model's
            layer objects.

            Returns
            -------
            config : dict
                Dictionary with keys ``'dmx'``, ``'dmx_out'``,
                ``'tocnv'`` (the temporal convolution layer, note the
                key name), ``'tconv_out'``, ``'pool'``, ``'pooled'``,
                ``'dropout'`` and ``'fin_fc'``, mapping to the
                corresponding layer objects/tensors.
            """
            # Do not call super.get_config!
            # This gave an error for me.
            config = {
                "dmx": self.dmx,
                "dmx_out": self.dmx_out,
                "tocnv": self.tconv,
                "tconv_out": self.tconv_out,
                "pool": self.pool,
                "pooled": self.pooled,
                "dropout": self.dropout,
                "fin_fc": self.fin_fc
            }
            return config

    def _get_class_conditional_spatial_covariance(self, X, y):
        """Compute the spatial (channel x channel) covariance matrices
        of the input, separately for each class and for its
        complement ("anti-class").

        Parameters
        ----------
        X : tf.Tensor
            Batch of input data, shape (n_epochs, 1, n_t, n_ch).

        y : tf.Tensor
            One-hot encoded batch of class labels, shape
            (n_epochs, n_classes).

        Returns
        -------
        dcovs : np.array
            Class-conditional spatial covariance matrices, shape
            (n_ch, n_ch, n_classes).

        dcovs_n : np.array
            Spatial covariance matrices computed over all samples
            *not* belonging to each class ("anti-class"), shape
            (n_ch, n_ch, n_classes).
        """
        #TODO: Fix regression case
        dcovs = []
        dcovs_n = []
        for class_y in range(self.out_dim):
            # `tf.reshape(..., [-1])` (rather than `tf.squeeze`) keeps the
            # index array 1-D even when exactly one sample matches -- a
            # plain `tf.squeeze` collapses a (1, 1) match down to a scalar,
            # which then drops the batch dimension entirely on indexing and
            # breaks the einsum below for that (fold, class) combination.
            class_ind = tf.reshape(tf.where(tf.argmax(y, 1)==class_y), [-1])
            # axis=1 targets only the model's singleton channel-group axis
            # (see the `X` shape in the docstring above); squeezing with no
            # axis would additionally collapse a batch of exactly 1 sample.
            xs = np.squeeze(X.numpy()[class_ind, ...], axis=1)
            #xs -= np.mean(xs, axis=-2, keepdims=True)
            ddof_s = xs.shape[0]*self.dataset.h_params['n_t'] - 1
            cov_s = np.einsum('ijk, ijl -> kl', xs, xs) / ddof_s

            anti_class_ind = tf.reshape(tf.where(tf.argmax(y, 1)!=class_y), [-1])
            xn = np.squeeze(X.numpy()[anti_class_ind, ...], axis=1)
            ddof_n = xn.shape[0]*self.dataset.h_params['n_t'] - 1
            cov_n = np.einsum('ijk, ijl -> kl', xn, xn) / ddof_n

            dcovs.append(cov_s) #  - cov_n
            dcovs_n.append(cov_n)
        return np.stack(dcovs, -1), np.stack(dcovs_n, -1)


    def patterns_cov_xx(self, y, weights, activations, dcov):
        """Compute spatial patterns from the covariance of the input
        and predicted output, using precomputed class-conditional
        spatial covariance matrices.

        Parameters
        ----------
        y : np.array
            One-hot encoded target variable, shape
            (n_epochs, n_classes). Referred to below as ``y``, shape
            ``[i, ..., j]``.

        weights : dict
            Dictionary of extracted model weights, as returned by
            :meth:`extract_weights`. Uses ``weights['out_weights']``
            (referred to below as ``w``, shape ``[k, ..., j]``) and
            ``weights['dmx']``.

        activations : dict
            Dictionary of layer activations, as computed in
            :meth:`compute_patterns`. Uses
            ``activations['pooled']`` (referred to below as ``X``,
            shape ``[i, ..., m]``).

        dcov : dict
            Dictionary of covariance matrices, as returned by
            :meth:`_get_class_conditional_spatial_covariance`. Uses
            ``dcov['class_conditional']``.

        Returns
        -------
        patterns : np.array
            Spatial patterns, shape (n_ch, n_classes), computed as
            ``Sx = [k, ..., mj]``.
        """

        x_shape = list(activations['pooled'].shape)
        y_shape = list(y.shape)

        ddof = activations['pooled'].shape[0] - 1
        X = np.reshape(activations['pooled'], [activations['pooled'].shape[0], -1])

        y = np.reshape(y, [y.shape[0], -1])

        w = np.reshape(weights['out_weights'], [-1, weights['out_weights'].shape[-1]])
        assert(X.shape[-1] == w.shape[0]), 'Shape mismatch X:{} w:{}'.format(X.shape, w.shape)
        assert(y.shape[-1] == w.shape[1]), 'Shape mismatch y:{} w:{}'.format(y.shape, w.shape)
        X = X - X.mean(0, keepdims=True)
        cov_xx = np.einsum('ik ,ij -> kj ', X, X) / ddof

        #compute inverse covariance of the output
        cov_yy = np.einsum('ij, ik -> jk', y, y) / ddof
        prec_yy = tf.linalg.pinv(cov_yy)

        #compute directions of Sx: a = cov_xy*(cov_yy)**-1
        a_out = np.einsum('ii, ij, jj -> ij', cov_xx, w, prec_yy)

        Sx = np.einsum('il, il, ji -> jil', a_out, w, X)
        Sx = np.reshape(Sx, [-1, x_shape[-1], y_shape[-1]])
        Sx = Sx - Sx.mean(1, keepdims=True)
        ddof = Sx.shape[0] - 1
        cov_sx = np.einsum('ijk, ilk -> jlk', Sx, Sx) / ddof

        patterns = []

        for i in range(y.shape[-1]):
            prec_sx = np.linalg.pinv(cov_sx[...,i])
            dc = dcov['class_conditional'][..., i]
            patterns.append(np.einsum('hi, ij, jk -> h',
                                      dc, weights['dmx'], prec_sx))
        patterns = np.stack(patterns, -1)
        return patterns



    def patterns_cov_xy_hat(self, X, y, activations, weights):
        """Back-propagate the covariance between the input/latent
        activations and the model's predictions through the temporal
        convolution and demixing layers.

        Parameters
        ----------
        X : tf.Tensor
            Batch of input data.

        y : tf.Tensor
            One-hot encoded batch of class labels.

        activations : dict
            Dictionary of layer activations, as computed in
            :meth:`compute_patterns`.

        weights : dict
            Dictionary of extracted model weights, as returned by
            :meth:`extract_weights`.

        Returns
        -------
        Sx_tconv : np.array
            Back-propagated pattern at the temporal convolution
            (pooled) layer, computed by :meth:`backprop_fc`.

        Sx_dmx : np.array
            Back-propagated pattern at the spatial demixing layer,
            computed by :meth:`backprop_covxy`.
        """
        Sx_tconv = self.backprop_fc(activations['pooled'],
                                    activations['fc'],
                                    y,
                                    weights['out_weights'])
        Sx_dmx = self.backprop_covxy(X,
                                    activations['dmx'],
                                    Sx_tconv,
                                    weights['dmx'])
        return Sx_tconv, Sx_dmx


    def backprop_fc(self, X, y_hat, y, w):
        """Back-propagate the covariance between an intermediate
        activation ``X`` and the model output ``y`` through a linear
        (fully-connected) layer with weights ``w``, to obtain a
        pattern in the space of ``X``.

        Parameters
        ----------
        X : np.array
            Intermediate activation, shape ``[i, ..., m]``.

        y_hat : np.array
            Model prediction associated with ``X`` (e.g. the fully
            connected layer's output), shape ``[i, ..., j]``.

        y : np.array
            True one-hot encoded target variable, shape
            ``[i, ..., j]``.

        w : np.array
            Weights of the linear layer mapping ``X`` to ``y_hat``,
            shape ``[k, ..., j]``.

        Returns
        -------
        Sx : np.array
            Back-propagated pattern, shape ``[k, ..., mj]`` (squeezed).
        """
        x_shape = list(X.shape)
        y_shape = list(y_hat.shape)

        ddof = X.shape[0] - 1
        X = np.reshape(X, [X.shape[0], -1])
        y = np.reshape(y, [y.shape[0], -1])
        y_hat = np.reshape(y_hat, [y_hat.shape[0], -1])
        w = np.reshape(w, [-1, w.shape[-1]])

        assert(X.shape[-1] == w.shape[0]), 'Shape mismatch X:{} w:{}'.format(X.shaep, w.shape)
        assert(y_hat.shape[-1] == w.shape[1]), 'Shape mismatch y:{} w:{}'.format(y_hat.shape, w.shape)
        X = X - X.mean(0, keepdims=True)
        y_hat = y_hat - y_hat.mean(0, keepdims=True)
        y = y - y.mean(0, keepdims=True)
        cov_xy = np.einsum('ik ,ij -> kj ', X, y_hat) / ddof

        cov_yy = np.einsum('ij, ik -> jk', y, y) / ddof
        prec_yy = tf.linalg.inv(cov_yy)

        #compute directions of Sx: a = cov_xy*(cov_yy)**-1
        a_out = np.einsum('jk, kl -> jl', cov_xy, prec_yy) #shape = [n_t_pooled, n_latent, n_classes]

        # A.*w
        #aw = a_out * w #shape = [...]
        # activation of tconv by each signal component of each sample
        #Sx = np.einsum('il, kl -> ik', y_hat, aw) #shape = [n_batch, ...]
        Sx = np.einsum('il, il, ji -> jil', a_out, w, X)
        Sx = np.squeeze(np.reshape(Sx, x_shape + y_shape[1:]))
        return Sx

    def backprop_covxy(self, X, Hx, Sx, w):
        """Back-propagate a pattern ``Sx`` computed at an intermediate
        layer ``Hx`` further back to the input space ``X``, using the
        covariance between ``X`` and ``Hx`` and the spatial weights
        ``w``.

        Parameters
        ----------
        X : tf.Tensor or np.array
            Input data.

        Hx : tf.Tensor or np.array
            Activation of the intermediate (demixing) layer computed
            from ``X``.

        Sx : np.array
            Pattern already computed at the level of ``Hx`` (e.g. the
            output of :meth:`backprop_fc`).

        w : np.array
            Spatial (demixing) weights, ``weights['dmx']``.

        Returns
        -------
        a : np.array
            Back-propagated pattern in the input (channel) space.
        """
        xdmx = np.reshape(Hx, [-1, Hx.shape[-1]])
        xdmx = xdmx - xdmx.mean(0, keepdims=True)
        xinp = np.reshape(X, [-1, X.shape[-1]])
        xinp = xinp - xinp.mean(0, keepdims=True)
        cov_xy = np.dot(xinp.T, xdmx)
        print("aw", cov_xy.shape)
        aw = cov_xy
        sx = np.reshape(Sx, [-1, Sx.shape[-2], Sx.shape[-1]])
        sx = sx - sx.mean(0, keepdims=True)
        ddof = sx.shape[0] - 1
        cov_sx = np.einsum('ijk, ilk -> kjl', sx,sx) / ddof
        print("cov_sx:", cov_sx.shape)
        prec_yy_hat = np.stack([np.linalg.pinv(s) for s in cov_sx])
        print(w.shape, prec_yy_hat.shape)
        ww = np.einsum('ij, jlk -> ikl', w, prec_yy_hat)
        print(cov_xy.shape, ww.shape)
        a = np.einsum('ij, ijk -> ik', cov_xy, ww)

        return a

    def patterns_pinv_w(self, y, weights, activations, dcov):
        """Compute spatial patterns via the pseudo-inverse of the
        output, demixing and temporal-convolution weight matrices,
        combined with class-conditional spatial covariances.

        Parameters
        ----------
        y : tf.Tensor
            One-hot encoded batch of class labels.

        weights : dict
            Dictionary of extracted model weights, as returned by
            :meth:`extract_weights`. Uses ``weights['dmx']``,
            ``weights['out_w_flat']`` and ``weights['tconv']``.

        activations : dict
            Dictionary of layer activations, as computed in
            :meth:`compute_patterns`. Uses ``activations['fc']`` and
            ``activations['pooled']``.

        dcov : dict
            Dictionary of covariance matrices, as returned by
            :meth:`_get_class_conditional_spatial_covariance`. Uses
            ``dcov['class_conditional']``.

        Returns
        -------
        topos : np.array
            Spatial patterns for each class, shape (n_ch, n_classes).
        """
        combined_topos = []
        pinv_dmx = np.linalg.pinv(weights['dmx']).T#np.dot(spatial_filters, np.linalg.inv(np.dot(spatial_filters.T, spatial_filters)))
        pinv_wfc = np.linalg.pinv(weights['out_w_flat']).T#np.dot(out_w_flat, np.linalg.inv(np.dot(out_w_flat.T, out_w_flat)))
        pinv_tck = np.linalg.pinv(weights['tconv']).T #np.dot(tconv_kernels, np.linalg.inv(np.dot(tconv_kernels.T, tconv_kernels)))

        #Least square singal estimate in tconv given wfc and fc_activations
        Sx_tconv = np.einsum('jk, ik ->ij', pinv_wfc, activations['fc'])
        Sx_tconv = np.reshape(Sx_tconv, activations['pooled'].shape)

        #Reverse pooling and depthwise convolution for each class
        Sx_dmx = []
        for class_y in range(self.out_dim):
            class_ind = tf.reshape(tf.where(tf.argmax(y, 1)==class_y), [-1])  # 1-D regardless of match count (see _get_class_conditional_spatial_covariance)
            Sxm = np.squeeze(Sx_tconv[class_ind, :].mean(0, keepdims=True))
            Sxm = np.atleast_2d(Sxm)
            dc = dcov['class_conditional'][..., class_y]
            combined_topos.append(np.einsum('hi,ij,tj->ht',
                                            dc,
                                            weights['dmx'],
                                            Sxm))
        topos = np.stack(combined_topos, 1)
        return topos


    def patterns_wfc_mean(self, y, weights, activations, dcov):
        """Compute spatial patterns from the class-conditional mean
        activation of the final (fully-connected) layer, combined
        with class-conditional spatial covariances. Uses the true
        labels ``y`` directly rather than the covariance between
        input and output, so it is accurate but less directly tied to
        the model's computations than the other ``patterns_*``
        methods.

        Parameters
        ----------
        y : tf.Tensor
            One-hot encoded batch of class labels.

        weights : dict
            Dictionary of extracted model weights, as returned by
            :meth:`extract_weights`. Uses ``weights['out_w_flat']``
            and ``weights['dmx']``.

        activations : dict
            Dictionary of layer activations, as computed in
            :meth:`compute_patterns`. Uses ``activations['fc']`` and
            ``activations['pooled']``.

        dcov : dict
            Dictionary of covariance matrices, as returned by
            :meth:`_get_class_conditional_spatial_covariance`. Uses
            ``dcov['class_conditional']``.

        Returns
        -------
        topos : np.array
            Spatial patterns for each class, shape (n_ch, n_classes).
        """
        combined_topos = []
        #uses y explicitely instead of cov[x,y]
        #accurate but has little to do with the model
        for class_y in range(self.out_dim):
            #compute mean activation of final layer for each class
            #TODO: -> to self.activations
            class_ind = tf.reshape(tf.where(tf.argmax(y, 1)==class_y), [-1])  # 1-D regardless of match count (see _get_class_conditional_spatial_covariance)
            fc_bp_out = (np.dot(activations['fc'].numpy()[class_ind, :],
                               weights['out_w_flat'].T)).mean(0)

            fc_bp_out = fc_bp_out.reshape([activations['pooled'].shape[2],
                                           activations['pooled'].shape[3]],
                                           order='C')
            dc = dcov['class_conditional'][..., class_y]
            class_patterns = np.dot(dc,
                                    weights['dmx'])
            cp = np.einsum('ck, ik -> c', class_patterns, fc_bp_out)

            combined_topos.append(cp) # + spatial_biases[class_y]

        topos = np.stack(combined_topos, 1)
        return topos


    def compute_patterns(self, data_path=None, verbose=False, shapley_order=1,
                         methods=['weight']):
        """Compute spatial and temporal patterns, weights, spectra and
        feature-relevance metrics for the model's latent components.
        Required for visualization and interpretation.

        Parameters
        ----------
        data_path : str, list of str, mneflow.Dataset, tf.data.Dataset, or None, optional
            Data on which the patterns are estimated. If None
            (default), the model's validation dataset
            (``self.dataset.val``) is used. A string or list/tuple of
            strings is interpreted as (a) path(s) to TFRecord file(s).
            An ``mneflow.Dataset`` instance uses its ``test`` set if
            present, otherwise its ``val`` set. A ``tf.data.Dataset``
            is used directly.

        verbose : bool, optional
            If True, print the shapes of the extracted layer
            activations. Defaults to False.

        shapley_order : int, optional
            Highest order of component-interaction (Shapley-like)
            relevances to compute via
            :meth:`compute_componentwise_loss`, when
            ``'compwise_loss'`` is included in ``methods``. 0 skips
            this computation; 1 computes single-component
            relevances; 2 and 3 additionally compute pairwise and
            triple-wise interaction relevances. Defaults to 1.

        methods : list of str, optional
            Which additional pattern/relevance metrics to compute, in
            addition to the always-computed weights, spectra and
            covariance-based patterns. Supported values are
            ``'weight'``, ``'compwise_loss'`` (requires
            ``shapley_order`` > 0) and ``'output_corr'``. Defaults to
            ``['weight']``.

        Returns
        -------
        patterns_struct : dict
            Dictionary collecting all computed patterns and
            statistics, with (among others) the keys 
            ``'weights'`` (see :meth:`extract_weights`), 
            ``'ccms'`` (class-conditional mean activations of each layer), 
            ``'dcov'`` (spatial covariance matrices, see
            :meth:`_get_class_conditional_spatial_covariance`),
            ``'spectra'`` (see :meth:`compute_spectra`), 
            ``'freqs'``,
            ``'cov_xx'``, 
            ``'pinv_w'`` and 
            ``'wfc_mean'`` (combined patterns, see :meth:`_compute_combined_patterns`), and,
            depending on ``methods``/``shapley_order``, 
            ``'compwise_loss'``, ``'shap_o2'``, ``'shap_o3'``,
            ``'ind_top_o2'``, ``'ind_top_o3'`` and
            ``'corr_to_output'`` (see
            :meth:`get_output_correlations`).

        Raises
        ------
        AttributeError
            If ``data_path`` is not None and is not a string, list,
            tuple, ``mneflow.Dataset`` or ``tf.data.Dataset``.
        """
        patterns_struct = {'weights' : {'dmx':[], 'tconv':[], 'fc':[],
                                        'tconv_freq_resposes':{}},
                           'ccms' : {'dmx':[], 'tconv':[], 'fc':[], 'input':[], 'dmx_psd':[]},
                           'dcov' : {'input_spatial':[], 'class_conditional':[],
                                     'k-1':[]},
                           'patterns' : {},
                           'spectra': {},
                           'freqs': None
                           }

        if not data_path:
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
            #TODO: rebalnce?
            ds = self.dataset._build_dataset(data_path,
                                             split=False,
                                             test_batch=None,
                                             repeat=True)
        elif isinstance(data_path, Dataset):
            if hasattr(data_path, 'test'):
                ds = data_path.test
            else:
                ds = data_path.val
        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')

        start = time()
        X, y = [row for row in ds.take(1)][0]
        ndof = X.shape[0] * self.dataset.h_params['n_t'] - 1

        #get layer activations
        activations = {}
        # Extract activations
        activations['dmx'] = self.dmx(X)
        activations['tconv'] = self.tconv(activations['dmx'])
        activations['pooled'] = self.pool(activations['tconv'])
        activations['fc']  = self.fin_fc(activations['pooled'])
        if verbose:
            print(""""Activations: \n
                  DMX: {}
                  TCONV: {}
                  POOLED: {}
                  FC_DENSE: {}""".format(
                  activations['dmx'].shape,
                  activations['tconv'].shape,
                  activations['pooled'].shape,
                  activations['fc'].shape))
        
        pooled_flat = tf.reshape(activations['pooled'], [-1, activations['pooled'].shape[-1]])
        cov_components = np.cov(tf.transpose(pooled_flat, perm=[1, 0]))
        stop = time() - start
        print("ACTIVATIONS:  {:.2f}s".format(stop))
        
        
        
        start = time()
        weights = self.extract_weights()
        spectra = self.compute_spectra(activations=activations)

        if not patterns_struct['freqs']:
            patterns_struct['freqs'] = spectra['freqs']

        stop = time() - start
        print("Weights and spectra: {:.2f}s".format(stop))
        #CCMs are mean activations of each layer for each class
        start = time()
        dcov = {}
        dcov['input_spatial'] = np.einsum('hijk, hijl -> kl', X, X) / ndof
        dcov['class_conditional'], dcov['k-1']  = self._get_class_conditional_spatial_covariance(X, y)
        stop = time() - start
        print("DCOVS:  {:.2f}s".format(stop))

        start = time()
        ##True evoked
        if self.dataset.h_params['target_type'] == 'float':
            self.true_evoked_data = X.numpy().mean(0)

            ccm_dmx = activations['dmx'].numpy().mean(0)[..., np.newaxis]

            ccm_tconv = activations['tconv'].numpy().mean(0)[..., np.newaxis]

            ccm_pooled = activations['pooled'].numpy().mean(0)[..., np.newaxis]

            ccm_fc = activations['fc'].numpy().mean(0)[..., np.newaxis]

            cov_y_hat = np.cov(tf.transpose(activations['fc'], perm=[1, 0]))
            cov_y = np.cov(tf.transpose(y, perm=[1, 0]))

        elif self.dataset.h_params['target_type'] == 'int':
            y_int = np.argmax(y, 1)
            # Iterate over *all* declared classes (self.out_dim, e.g. from
            # class_subset), not just np.unique(y_int) -- the batch used
            # here (self.dataset.val by default) is not guaranteed to
            # contain every class, and collect_patterns() always assigns
            # into a pre-allocated array sized for self.out_dim classes.
            # A class missing from this particular batch previously shrank
            # y_unique and every ccm_*/evokeds array along with it, which
            # broke that assignment with a shape mismatch (e.g. (..., 6)
            # into (..., 7)). Missing classes are filled with zeros instead.
            n_classes = self.out_dim

            def _class_means(arr, axis=-1):
                arr = arr.numpy() if hasattr(arr, 'numpy') else arr
                means = [arr[y_int == i, ...].mean(0) if np.any(y_int == i)
                         else np.zeros(arr.shape[1:], dtype=arr.dtype)
                         for i in range(n_classes)]
                return np.stack(means, axis)

            # evokeds/true_evoked_data keep classes on axis 0 (as before,
            # and as plot_evoked_peaks' docstring expects: shape
            # (n_classes, n_t, n_ch)); the ccm_* arrays keep classes on the
            # last axis, matching collect_patterns()'s pre-allocated shapes.
            evokeds = _class_means(X, axis=0)
            self.true_evoked_data = np.squeeze(evokeds)
            ccm_dmx = _class_means(activations['dmx'])
            ccm_tconv = _class_means(activations['tconv'])
            ccm_pooled = _class_means(activations['pooled'])
            ccm_fc = _class_means(activations['fc'])
            cov_y_hat = np.cov(tf.transpose(activations['fc'], perm=[1, 0]))
            cov_y = np.cov(tf.transpose(y, perm=[1, 0]))

        stop = time() - start
        print("CCMs:  {:.2f}s".format(stop))
        # compute the effect of removing each latent component on the cost function

        patterns_struct['weights'] = weights
        patterns_struct['spectra'] = spectra
        patterns_struct['dcov'] = dcov
        patterns_struct['ccms'] = {'dmx': ccm_dmx, # n_t, n_latent, n_classes
                                   'tconv':ccm_tconv, #n_t, n_latent, n_classes
                                   'pooled':ccm_pooled, #n_t_pooled, n_latent, n_classes
                                   'fc':ccm_fc, #n_y, n_y
                                   'cov_y_hat':cov_y_hat,
                                   'cov_y': cov_y,
                                   'cov_components': cov_components,
                                   'psds': spectra['psds']} #n_y, n_y


        start = time()
        combined_patterns = self._compute_combined_patterns(y, weights, activations, dcov)
        stop = time() - start
        print("Combined patterns: {:.2}s".format(stop))
        patterns_struct.update(combined_patterns)

        #compute the effect of removing each latent component on the cost function
        start = time()
        if 'compwise_loss' in methods and shapley_order > 0:
            _, shap = self.compute_componentwise_loss(X, y, order=shapley_order)
            patterns_struct['compwise_loss'] = np.repeat(np.stack([b['o1_relevances'] for b in shap],
                                                                  axis=1)[np.newaxis, ...],
                                                         self.pooled.shape[2], axis=0) # n_t, n_comp, n_y
            if shapley_order > 1:
                patterns_struct['shap_o2'] = np.repeat(np.stack([b['o2_relevances'] for b in shap],
                                                                      axis=1)[np.newaxis, ...],
                                                             self.pooled.shape[2], axis=0) # n_t, n_comp, n_y
                patterns_struct['ind_top_o2'] = np.stack([b['o2_inds'] for b in shap],
                                                                      axis=1) # 2, n_y
            if shapley_order > 2:
                patterns_struct['shap_o3'] = np.repeat(np.stack([b['o3_relevances'] for b in shap],
                                                                      axis=1)[np.newaxis, ...],
                                                             self.pooled.shape[2], axis=0) # n_t, n_comp, n_y
                patterns_struct['ind_top_o3'] = np.stack([b['o3_inds'] for b in shap],
                                                                      axis=1) # 3, n_y

        stop = time() - start
        print("Compwise Loss: {:.2f}s".format(stop))
        #correlation of fc activations to y
        start = time()
        if 'output_corr' in methods:
            patterns_struct['corr_to_output'] = self.get_output_correlations(activations, y)
            stop = time() - start
            print("Output corr: {:.2}s".format(stop))
        del X, activations

        return patterns_struct


    def _compute_combined_patterns(self, y, weights, activations, dcov):
        """Compute the combined spatial patterns using the
        ``'cov_xx'``, ``'pinv_w'`` and ``'wfc_mean'`` methods.

        Parameters
        ----------
        y : tf.Tensor
            One-hot encoded batch of class labels.

        weights : dict
            Dictionary of extracted model weights, as returned by
            :meth:`extract_weights`.

        activations : dict
            Dictionary of layer activations, as computed in
            :meth:`compute_patterns`.

        dcov : dict
            Dictionary of covariance matrices, as returned by
            :meth:`_get_class_conditional_spatial_covariance`.

        Returns
        -------
        patterns : dict
            Dictionary with keys ``'cov_xx'``, ``'pinv_w'`` and
            ``'wfc_mean'``, each mapping to a dict with key
            ``'spatial'`` holding the corresponding spatial pattern
            array (see :meth:`patterns_cov_xx`,
            :meth:`patterns_pinv_w` and :meth:`patterns_wfc_mean`).
        """
        patterns = {'cov_xx':{}, 'pinv_w':{}, 'wfc_mean':{}}
        patterns['cov_xx']['spatial'] = self.patterns_cov_xx(y, weights, activations, dcov)

        patterns['pinv_w']['spatial'] = self.patterns_pinv_w(y, weights, activations, dcov).mean(-1)

        patterns['wfc_mean']['spatial'] = self.patterns_wfc_mean(y, weights, activations, dcov)
        return patterns

    def init_pattern_struct(self, n_folds, freqs, methods='all'):
        """Pre-allocate ``self.cv_patterns``, a nested dictionary of
        zero-filled arrays used to accumulate patterns, weights and
        feature-relevance metrics across cross-validation folds.

        Parameters
        ----------
        n_folds : int
            Number of cross-validation folds to allocate storage for.

        freqs : np.array
            Frequency bins of the spectral estimates; stored as
            ``self.cv_patterns['freqs']``.

        methods : str or list of str, optional
            Which feature-relevance methods to allocate storage for.
            If 'all' (default), allocates storage for ``['weight',
            'compwise_loss', 'weight_norm', 'output_corr',
            'shap_o2', 'shap_o3']``.

        Returns
        -------
        None
            Populates ``self.cv_patterns`` in place.
        """
        if methods == 'all':
            methods = ['weight', 'compwise_loss', 'weight_norm', 'output_corr',
                       'shap_o2','shap_o3']
        self.cv_patterns = defaultdict(dict)
        n_ch = self.meta.data['n_ch']
        n_t_pooled = self.pooled.shape[2] #patterns_struct['weights']['out_weights'].shape[0]
        n_t = self.meta.data['n_t']
        #n_fft = len(patterns_struct['freqs'])

        self.cv_patterns['freqs'] = freqs

        self.cv_patterns['dcov']['input_spatial'] = np.zeros([n_ch, n_ch,
                                                              n_folds])
        self.cv_patterns['dcov']['class_conditional'] = np.zeros([n_ch, n_ch,
                                                                  self.y_shape[0],
                                                                  n_folds])
        self.cv_patterns['dcov']['k-1'] = np.zeros([n_ch, n_ch,
                                                    self.y_shape[0],
                                                    n_folds])
        self.cv_patterns['ccms']['dmx'] = np.zeros([n_t,
                                                    self.specs['n_latent'],
                                                    self.y_shape[0],
                                                    n_folds])

        self.cv_patterns['ccms']['tconv'] = np.zeros([n_t,
                                                    self.specs['n_latent'],
                                                    self.y_shape[0],
                                                    n_folds])

        self.cv_patterns['ccms']['pooled'] = np.zeros([self.pooled.shape[2],
                                                    self.specs['n_latent'],
                                                    self.y_shape[0],
                                                    n_folds])
        self.cv_patterns['ccms']['fc'] = np.zeros([self.y_shape[0],
                                                   self.y_shape[0],
                                                   n_folds])
        self.cv_patterns['ccms']['cov_y_hat'] = np.zeros([self.y_shape[0],
                                                          self.y_shape[0],
                                                          n_folds])
        self.cv_patterns['ccms']['cov_y'] = np.zeros([self.y_shape[0],
                                                          self.y_shape[0],
                                                          n_folds])
        self.cv_patterns['ccms']['cov_components'] = np.zeros([self.specs['n_latent'],
                                                               self.specs['n_latent'],  
                                                               n_folds])
        self.cv_patterns['ccms']['cov_dmx'] = np.zeros([self.y_shape[0],
                                                          self.y_shape[0],
                                                          n_folds])
        self.cv_patterns['ccms']['cov_tconv'] = np.zeros([self.y_shape[0],
                                                          self.y_shape[0],
                                                          n_folds])

        self.cv_patterns['ccms']['psds'] = np.zeros([self.nfft,
                                                     self.specs['n_latent'],
                                                          n_folds])
        self.cv_patterns['ind_top_o3'] = np.zeros([self.specs['n_latent'],
                                                  self.y_shape[0],
                                                  n_folds])
        self.cv_patterns['ind_top_o2'] = np.zeros([self.specs['n_latent'],
                                                  self.y_shape[0],
                                                  n_folds])



        for method in methods:
            #print(method)
            self.cv_patterns[method]['feature_relevance'] = np.zeros([n_t_pooled,
                                                         self.specs['n_latent'],
                                                         self.y_shape[0],
                                                         n_folds])
    def collect_patterns(self, fold=0, n_folds=1, shapley_order=0,
                         methods=['weight',
                                  'weight_norm',
                                  'output_corr']):
        """Compute patterns for the current fold via
        :meth:`compute_patterns` and store them into the
        pre-allocated ``self.cv_patterns`` and ``self.cv_weights``
        containers (see :meth:`init_pattern_struct`).

        Parameters
        ----------
        fold : int, optional
            Index of the current cross-validation fold, used to
            index into ``self.cv_patterns``. Defaults to 0.

        n_folds : int, optional
            Total number of cross-validation folds. Currently unused
            in this method. Defaults to 1.

        shapley_order : int, optional
            Highest order of component-interaction relevances to
            compute and store; see :meth:`compute_patterns`.
            Defaults to 0.

        methods : list of str, optional
            Base set of pattern/relevance methods to compute; see
            :meth:`compute_patterns`. Defaults to
            ``['weight', 'weight_norm', 'output_corr']``.

        Returns
        -------
        None
            Updates ``self.cv_patterns`` and ``self.cv_weights`` in
            place.
        """
        print("Collecting patterns from fold {}".format(fold))
        methods = methods.copy()
        if shapley_order > 0:
            methods.append('compwise_loss')
        if shapley_order > 1:
            methods.append('shap_o2')
        if shapley_order > 2:
            methods.append('shap_o3')
        #print(methods)
        patterns_struct = self.compute_patterns(shapley_order=shapley_order, methods=methods)
        if not np.any(self.cv_patterns['freqs']):
            self.cv_patterns['freqs'] = patterns_struct['freqs']

        self.cv_patterns['dcov']['input_spatial'][:, :, fold] = patterns_struct['dcov']['input_spatial']
        self.cv_patterns['dcov']['class_conditional'][:, :, :, fold] = patterns_struct['dcov']['class_conditional']
        if shapley_order > 0:
            self.cv_patterns['compwise_loss']['feature_relevance'][:, :, :, fold] = patterns_struct['compwise_loss']

        if shapley_order > 1:
            self.cv_patterns['shap_o2']['feature_relevance'][:, :, :, fold] = patterns_struct['shap_o2']
            self.cv_patterns['ind_top_o2'][:, :, fold] = patterns_struct['ind_top_o2']
        if shapley_order > 2:
            self.cv_patterns['shap_o3']['feature_relevance'][:, :, :, fold] = patterns_struct['shap_o3']
            self.cv_patterns['ind_top_o3'][:, :, fold] = patterns_struct['ind_top_o3']


        self.cv_patterns['output_corr']['feature_relevance'][:, :, :, fold] = patterns_struct['corr_to_output']
        self.cv_patterns['weight']['feature_relevance'][:, :, :, fold] = patterns_struct['weights']['out_weights']
        [self.cv_weights[k].append(patterns_struct['weights'][k])
         for k in patterns_struct['weights'].keys()]

        self.cv_patterns['ccms']['dmx'][:, :, :, fold] = patterns_struct['ccms']['dmx'] #n_t, n_latent, n_classes, n_folds
        self.cv_patterns['ccms']['tconv'][:, :, :, fold] = patterns_struct['ccms']['tconv'] #n_t, n_latent, n_classes, n_folds
        self.cv_patterns['ccms']['pooled'][:, :, :, fold] = patterns_struct['ccms']['pooled'] #n_pooled, n_latent, n_classes, n_folds

        self.cv_patterns['ccms']['fc'][:, :, fold] = patterns_struct['ccms']['fc'] # n_logits, n_classes, n_folds
        self.cv_patterns['ccms']['cov_y_hat'][:, :, fold] = patterns_struct['ccms']['cov_y_hat'] # n_classes, n_classes, n_folds
        self.cv_patterns['ccms']['cov_y'][:, :, fold] = patterns_struct['ccms']['cov_y'] # n_classes, n_classes, n_folds
        self.cv_patterns['ccms']['cov_components'][:, :, fold] = patterns_struct['ccms']['cov_components'] # n_components, n_components, n_folds
        self.cv_patterns['ccms']['psds'][:, :, fold] = patterns_struct['ccms']['psds']

    def compute_spectra(self, activations, nfft=128):
        ##Psds
        """Compute power spectral densities (PSDs) of the latent
        (demixed) components using Welch's method.

        Parameters
        ----------
        activations : dict
            Dictionary of layer activations, as computed in
            :meth:`compute_patterns`. Uses ``activations['dmx']``.

        nfft : int, optional
            Length of the FFT used, passed to
            :func:`scipy.signal.welch` (as ``nperseg``, with
            ``nfft * 2`` used for ``nfft``). Defaults to 128.

        Returns
        -------
        spectra : dict
            Dictionary with keys ``'psds'`` (array of shape
            (n_freqs, n_latent)), ``'freqs'`` (frequency bins) and
            ``'nfft'`` (the possibly-reduced FFT length actually
            used).
        """
        psds = []
        for i in range(self.specs['n_latent']):

            ltc = activations['dmx'][:, 0, :, i] - np.mean(activations['dmx'][:, 0, :, i], axis=1, keepdims=True)
            fr, psd = welch(ltc,
                            fs=self.dataset.h_params['fs'],
                            nfft=nfft * 2,
                            nperseg=nfft)
            if len(fr[:-1]) < nfft:
                nfft = len(fr[:-1])
            psds.append(psd[:, 1:].mean(0))


        spectra = {}
        spectra['psds'] = np.array(psds).T
        spectra['freqs'] = fr[1:]
        spectra['nfft'] = nfft
        print(spectra['psds'].shape)
        return spectra



    def extract_weights(self, verbose=False):
        """Extract the trained weights of the spatial demixing,
        temporal convolution and output (fully-connected) layers.

        Parameters
        ----------
        verbose : bool, optional
            If True, print the shapes of the extracted weight
            arrays. Defaults to False.

        Returns
        -------
        weights : dict
            Dictionary with keys ``'dmx'`` (spatial demixing
            weights), ``'dmx_b'`` (demixing biases), ``'tconv'``
            (temporal convolution kernels), ``'tconv_b'`` (temporal
            convolution biases), ``'out_w_flat'`` (flattened output
            layer weights), ``'out_weights'`` (output layer weights
            reshaped to (n_t_pooled, n_latent, n_classes)) and
            ``'fc_b'`` (output layer biases).
        """
        weights = {}

        # Extract weights

        # Spatial extraction fiters
        weights['dmx'] = np.squeeze(self.dmx.w.numpy())
        weights['dmx_b'] = self.dmx.b_in.numpy()
        # Temporal kernels
        weights['tconv'] = np.squeeze(self.tconv.filters.numpy())
        weights['tconv_b'] = np.squeeze(self.tconv.b.numpy())
        # Final layer
        weights['out_w_flat'] = self.fin_fc.w.numpy()
        weights['out_weights'] = np.reshape(self.fin_fc.w.numpy(),
                                 [self.pooled.shape[2],
                                  self.dmx.size,
                                  self.out_dim],
                                 order='C')

        weights['fc_b'] = self.fin_fc.b.numpy()

        if verbose:
            print(""""Weights: \n
                  DMX: {}
                  TCONV: {}
                  FC_DENSE: {}""".format(weights['dmx'].shape,
                  weights['tconv'].shape,
                  weights['out_weights'].shape))

        return weights

    def compute_componentwise_loss(self, X, y, order=1, verbose=False):

        """Estimate the relevance of each latent component (and,
        optionally, of interactions between components) to the
        model's loss, by recursively zeroing out the corresponding
        output-layer weights and measuring the resulting change in
        loss (a Shapley-like sensitivity analysis).

        Parameters
        ----------
        X : tf.Tensor or np.array
            Input data on which the model is evaluated.

        y : tf.Tensor or np.array
            True target values corresponding to ``X``.

        order : int, optional
            Highest interaction order to compute. 1 computes
            single-component relevances only; 2 additionally
            computes pairwise interactions among the top half of
            components (by first-order relevance); 3 additionally
            computes triple-wise interactions among the top half of
            the order-2 candidates. Defaults to 1.

        verbose : bool, optional
            If True, print progress information for each evaluated
            combination. Defaults to False.

        Returns
        -------
        feature_relevance_loss : dict
            Mapping from a hyphen-joined key of component indices
            (and, for single components, the suffix ``'self'``) to
            the resulting change in loss when the corresponding
            weights are zeroed, for the last class processed.

        best : list of dict
            Per-class (length ``n_y``) list of dictionaries
            summarizing the best-found relevances and combinations at
            each computed order, with keys such as ``'o1'``,
            ``'o1_ind'``, ``'o1_sorting'``, ``'o1_relevances'`` and,
            when ``order`` > 1 or > 2, the corresponding ``'o2_*'``
            and ``'o3_*'`` entries.
        """
        #Copy of the original weights
        original_weights = self.km.get_weights()
        #Basic pefroamnce with full original weights
        base_loss, base_performance = self.km.evaluate(X, y, verbose=0)

        #This is mutable array used to zero indexed weights and set km.weights
        model_weights = original_weights.copy()
        n_t = self.pooled.shape[2]
        n_components = self.specs['n_latent']
        n_y = self.out_dim
        best = [{} for _ in range(n_y)]

        #output containers

        losses = np.zeros([self.specs['n_latent'], n_y])
        #Pre-generate flat indices for each class and component
        print(n_t, n_components, n_y, n_t*n_components*n_y)
        flat_inds = np.array([[np.ravel_multi_index((np.arange(n_t), c_i, i_y), (n_t, n_components, n_y))
                     for c_i in range(n_components)]
                              for i_y in range(n_y)]) #n_y, n_com, n_t
        print(flat_inds.shape, np.max(flat_inds))

        for jj in range(n_y):
            feature_relevance_loss = defaultdict(int)
            best[jj]['o1'] = -np.inf
            best[jj]['-'.join(['key', 'o1'])] = ''
            candidate_inds = np.arange(self.specs["n_latent"])
            print("Searching for best combo among {} components for class {}".format(len(candidate_inds), jj + 1))

            mutable_weights = model_weights[-2].copy()

            #for each class
            for i in candidate_inds:
                old_weights = mutable_weights.flat[flat_inds[jj, i, :]].copy()
                mutable_weights.flat[flat_inds[jj, i, :]] = 0.
                model_weights[-2] = mutable_weights

                self.km.set_weights(model_weights)
                new_loss = self.km.evaluate(X, y, verbose=0)[0]

                losses[i, jj] = new_loss - base_loss
                basic_key = '-'.join([str(i), 'self'])
                if new_loss - base_loss > best[jj]['o1']:
                    best[jj]['o1'] = new_loss - base_loss
                    best[jj]['-'.join(['key', 'o1'])] = basic_key

                feature_relevance_loss[basic_key] = new_loss - base_loss # larger difference is better
                mutable_weights.flat[flat_inds[jj, i, :]] = old_weights
                if verbose:
                    print("SHAP order 1 {}/{} : {:.4f}".format(i, len(candidate_inds), losses[i, jj]))

            if order > 1:
                best[jj]['o2'] = -np.inf
                best[jj]['-'.join(['key', 'o2'])] = ''
                #Drop a half of candidate fuatures in the next round
                top_inds = np.where(losses[:, jj] >= np.median(losses[:, jj]))[0]
                candidate_inds2 = candidate_inds[top_inds]
                if verbose:
                    print("Searching for the best combo among: ", candidate_inds2)
                for i1, c in enumerate(candidate_inds2):
                    #zero out component 1
                    old_weights = mutable_weights.flat[flat_inds[jj, c, :]].copy()
                    mutable_weights.flat[flat_inds[jj, c, :]] = 0.
                    for i2 in range(i1 + 1, len(candidate_inds2)):
                        #only explore combinations from i+1 onwards
                        old_weights2 = mutable_weights.flat[flat_inds[jj, candidate_inds2[i2], :]].copy()
                        mutable_weights.flat[flat_inds[jj, candidate_inds2[i2], :]] = 0.
                        model_weights[-2] = mutable_weights
                        self.km.set_weights(model_weights)
                        new_loss = self.km.evaluate(X, y, verbose=0)[0]
                        interaction_key = '-'.join([str(c), str(candidate_inds2[i2])])
                        if new_loss - base_loss > best[jj]['o2']:
                            best[jj]['o2'] = new_loss - base_loss
                            best[jj]['-'.join(['key', 'o2'])] = interaction_key
                        feature_relevance_loss[interaction_key] = new_loss - base_loss
                        #for i3 in range(i2, len(candidate_loss3)):
                        mutable_weights.flat[flat_inds[jj, candidate_inds2[i2], :]] = old_weights2
                    if verbose:
                        print("SHAP order 2 {}/{}".format(i1, len(candidate_inds2)))
                    mutable_weights.flat[flat_inds[jj, c, :]] = old_weights

            if order > 2:
                out = defaultdict(list)
                best[jj]['o3'] = -np.inf
                best[jj]['-'.join(['key', 'o3'])] = ''
                for k in feature_relevance_loss.keys():
                    k1, k2 = k.split('-')
                    out[k1].append(feature_relevance_loss[k])
                    out[k2].append(feature_relevance_loss[k])

                shap2 = np.array([np.mean(out[str(k)]) for k in candidate_inds2])

                top_inds2 = np.where(shap2 >= np.median(shap2))[0]
                candidate_inds3 = candidate_inds2[top_inds2]
                if verbose:
                    print("Searching for best combo among", candidate_inds3)
                for i1, c in enumerate(candidate_inds3):
                    #zero out component 1
                    old_weights = mutable_weights.flat[flat_inds[jj, c, :]].copy()
                    mutable_weights.flat[flat_inds[jj, c, :]] = 0.
                    for i2 in range(i1 + 1, len(candidate_inds3)):
                        #only explore combinations from i+1 onwards
                        old_weights2 = mutable_weights.flat[flat_inds[jj, candidate_inds2[i2], :]].copy()
                        mutable_weights.flat[flat_inds[jj, candidate_inds2[i2], :]] = 0.
                        for i3 in range(i2 + 1, len(candidate_inds3)):
                            old_weights3 = mutable_weights.flat[flat_inds[jj, candidate_inds3[i3], :]].copy()
                            mutable_weights.flat[flat_inds[jj, candidate_inds3[i3], :]] = 0.
                            model_weights[-2] = mutable_weights
                            self.km.set_weights(model_weights)
                            new_loss = self.km.evaluate(X, y, verbose=0)[0]

                            interaction_key = '-'.join([str(c), str(candidate_inds3[i2]), str(candidate_inds3[i3])])

                            if new_loss - base_loss > best[jj]['o3']:
                                best[jj]['o3'] = new_loss - base_loss
                                best[jj]['-'.join(['key', 'o3'])] = interaction_key
                            feature_relevance_loss[interaction_key] = new_loss - base_loss
                            mutable_weights.flat[flat_inds[jj, candidate_inds3[i3], :]] = old_weights3

                        mutable_weights.flat[flat_inds[jj, candidate_inds2[i2], :]] = old_weights2
                    if verbose:
                        print("SHAP order 3 {}/{}".format(i1, len(candidate_inds3)))
                    mutable_weights.flat[flat_inds[jj, c, :]] = old_weights

                for k in feature_relevance_loss.keys():
                    feat_keys = k.split('-')
                    if len(feat_keys) == 3:
                         k1, k2, k3 = feat_keys
                         out[k1].append(feature_relevance_loss[k])
                         out[k2].append(feature_relevance_loss[k])
                         out[k3].append(feature_relevance_loss[k])

            relevances = np.zeros(self.specs['n_latent'])

            best[jj]['o1_ind'] = int(best[jj]['key-o1'].split('-')[0])
            best[jj]['o1_sorting'] = np.argsort(losses[:, jj])
            best[jj]['o1_relevances'] = losses[:, jj]

            if order > 1:
                best[jj]['o2_sorting'] = candidate_inds2[np.argsort(shap2)]
                best[jj]['o2_relevances'] = relevances.copy()
                best[jj]['o2_relevances'][candidate_inds2] = shap2
                o2_inds = np.array([int(ind) for ind in best[jj]['key-o2'].split('-')])
                best[jj]['o2_inds'] = relevances.copy()
                best[jj]['o2_inds'][o2_inds] = 1.

            if order > 2:
                shap3 = np.array([np.mean(out[str(k)]) for k in candidate_inds3])
                best[jj]['o3_relevances'] = relevances.copy()
                best[jj]['o3_relevances'][candidate_inds3] = shap3
                o3_inds = np.array([int(ind) for ind in best[jj]['key-o3'].split('-')])
                best[jj]['o3_inds'] = relevances.copy()
                best[jj]['o3_inds'][o3_inds] = 1.
                best[jj]['o3_sorting'] = candidate_inds3[np.argsort(shap3)]


        self.km.set_weights(original_weights)
        return feature_relevance_loss, best

    def get_output_correlations(self, activations, y_true):
        """Computes a similarity metric between each of the extracted
        (pooled) features and the target variable.

        The metric is the Spearman correlation for continuous
        (``'float'``/``'signal'``) targets, and the Pearson
        correlation for discrete (``'int'``) targets.

        Parameters
        ----------
        activations : dict
            Dictionary of layer activations, as computed in
            :meth:`compute_patterns`. Uses ``activations['pooled']``.

        y_true : tf.Tensor
            True target values.

        Returns
        -------
        corr_to_output : np.array
            Correlation of each pooled feature with each column of
            ``y_true``, reshaped to
            (n_t_pooled, n_latent, n_targets). NaNs (e.g. from
            constant inputs) are replaced with 0.
        """
        corr_to_output = []
        y_true = y_true.numpy()
        flat_feats = activations['pooled'].numpy().reshape(y_true.shape[0], -1)


        for y_ in y_true.T:
            if self.dataset.h_params['target_type'] in ['float', 'signal']:
                rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])
                corr_to_output.append(rfocs.reshape(activations['pooled'].shape[1:]))


            elif self.dataset.h_params['target_type'] == 'int':
                rfocs = np.array([pearsonr(y_, f)[0] for f in flat_feats.T])

                corr_to_output.append(rfocs.reshape(activations['pooled'].shape[1:]))


        corr_to_output = np.concatenate(corr_to_output, 0).transpose([1, 2, 0])
        if np.any(np.isnan(corr_to_output)):
            corr_to_output[np.isnan(corr_to_output)] = 0
        return corr_to_output

    # --- LFCNN plot functions ---

    def plot_evoked_peaks(self, data=None, t=None, class_subset=None,
                          sensor_layout='Vectorview-mag', title=None, savefig=None):
        """Plot one spatial topography of the class-conditional
        average of the input (or of model-derived data). If a
        timepoint is not specified, it is picked as the one
        maximizing the RMS averaged over channels and classes.

        Parameters
        ----------
        data : np.array, optional
            Data to plot, shape (n_classes, n_t, n_ch). If None
            (default), ``self.true_evoked_data`` (the class-
            conditional average input, set by :meth:`compute_patterns`)
            is used.

        t : int, optional
            Timepoint index to plot. If None (default), it is picked
            automatically as the timepoint with maximum mean squared
            amplitude.

        class_subset : np.array, optional
            Subset of classes to plot. Defaults to None (all
            classes).

        sensor_layout : str, optional
            Name of the MNE sensor layout used to plot the
            topography. Defaults to 'Vectorview-mag'.

        title : str, optional
            Plot title. Defaults to None (a title is generated
            automatically based on ``data``).

        savefig : bool, optional
            If truthy, save the resulting figure to an SVG file.
            Defaults to None.

        Returns
        -------
        topoplot : matplotlib.figure.Figure
            The resulting topography figure, as returned by
            :meth:`plot_topos`.
        """
        n = self.out_dim

        if data is None:
            data = self.true_evoked_data
            if not title:
                title = 'True Patterns'
        else:
            title = 'Model-derived patterns'

        if t is None:
            t = np.argmax(np.mean(data**2, axis=0).mean(-1))
            print(t)
        title = title +  't={}'.format(t)
        ed = np.stack([data[i, t, :] for i in range(n)], axis=-1)
        assert ed.ndim==2
        topoplot = self.plot_topos(ed, sensor_layout=sensor_layout,
                                   class_subset=class_subset, title=title)

        #topoplot.figure.suptitle(title)
        topoplot.show()
        if savefig:
            figname = '-'.join([self.meta.data['path'] + self.scope, self.meta.data['data_id'], title, "topos.svg"])
            topoplot.savefig(figname, format='svg', transparent=True)
        return topoplot

    def plot_topos(self, topos, sensor_layout='Vectorview-mag', class_subset=None,
                   title="Class %g"):
        """Plot any spatial distribution in sensor space as a set of
        topographic maps.

        Parameters
        ----------
        topos : np.array
            Spatial distribution(s) to plot, shape
            (n_ch, n_classes) or (n_ch, n_classes, ...) (in which
            case it is averaged over the trailing dimension(s)
            before plotting).

        sensor_layout : str, optional
            Name of the MNE sensor layout used to plot the
            topography. Defaults to 'Vectorview-mag'.

        class_subset : np.array, optional
            Subset of classes (time-slots in the fake evoked object)
            to plot. Defaults to None (all classes).

        title : str, optional
            Format string used as the per-map title (passed as
            ``time_format`` to
            :meth:`mne.Evoked.plot_topomap`). Defaults to
            "Class %g".

        Returns
        -------
        ft : matplotlib.figure.Figure
            The resulting topography figure.
        """

        if topos.ndim > 2:
            topos = topos.mean(-1)
        topos_new = topos / topos.std(0, keepdims=True)

        n = topos.shape[1]

        if class_subset is None:
            class_subset = np.arange(0,  n, 1.)

        fake_evoked = self.make_fake_evoked(topos_new, sensor_layout)

        ft = fake_evoked.plot_topomap(times=class_subset,
                                    colorbar=True,
                                    scalings=1,
                                    time_format=title,
                                    outlines='head',
                                    #vlim= np.percentile(topos, [5, 95])
                                    )
        #ft.show()
        return ft

    def make_fake_evoked(self, topos, sensor_layout):
        """Create an ``mne.evoked.Evoked`` object for plotting and
        source-localizing model activation patterns.

        Parameters
        ----------
        topos : np.array
            Spatial activation patterns, shape
            (n_channels, n_patterns).

        sensor_layout : str or mne.channels.Layout
            Sensor layout used to build channel positions, if
            ``'info'`` is not already present in
            ``self.meta.data``.

        Returns
        -------
        fake_evoked : mne.evoked.EvokedArray
            Evoked object wrapping ``topos``, suitable for plotting
            with ``plot_topomap``.
        """
        if 'info' not in self.meta.data.keys():
            lo = channels.read_layout(sensor_layout)
            info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
            orig_xy = np.mean(lo.pos[:, :2], 0)
            for i, ch in enumerate(lo.names):
                if info['chs'][i]['ch_name'] == ch:
                    info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/4.5
                    #info['chs'][i]['loc'][4:] = 0
                else:
                    print("Channel name mismatch. info: {} vs lo: {}".format(
                        info['chs'][i]['ch_name'], ch))
        fake_evoked = evoked.EvokedArray(topos, info)
        return fake_evoked


    def explore_components(self, patterns_struct, sorting='output_corr',
                         integrate='max', info=None, sensor_layout='Vectorview-grad',
                         class_names=None):
        """Delegate to :meth:`mneflow.meta.MetaData.explore_components`
        to visualize/explore the latent components.

        Parameters
        ----------
        patterns_struct : dict
            Dictionary of computed patterns, as returned by
            :meth:`compute_patterns`. Currently not forwarded to
            ``self.meta.explore_components``, which is called with no
            arguments.

        sorting : str, optional
            Heuristic for sorting/selecting relevant components.
            Currently not forwarded to
            ``self.meta.explore_components``. Defaults to
            'output_corr'.

        integrate : str, optional
            How to integrate relevances over time. Currently not
            forwarded to ``self.meta.explore_components``. Defaults
            to 'max'.

        info : mne.Info, optional
            Currently not forwarded to
            ``self.meta.explore_components``. Defaults to None.

        sensor_layout : str, optional
            Currently not forwarded to
            ``self.meta.explore_components``. Defaults to
            'Vectorview-grad'.

        class_names : list of str, optional
            Currently not forwarded to
            ``self.meta.explore_components``. Defaults to None.

        Returns
        -------
        None
        """
        self.meta.explore_components()




    def plot_waveforms(self, patterns_struct, sorting='weight', tmin=0, class_names=None,
                       bp_filter=False, tlim=None, apply_kernels=False):
        """Plot timecourses, temporal-convolution output and relative
        power spectra of the latent components, highlighting the
        components selected for each class.

        Parameters
        ----------
        patterns_struct : dict
            Dictionary of computed patterns, as returned by
            :meth:`compute_patterns`. Uses
            ``patterns_struct['ccms']['tconv']`` as the per-class
            waveforms.

        sorting : str, optional
            Heuristic for selecting relevant components, passed to
            ``self._sorting``. Defaults to 'weight'.

        tmin : float, optional
            Beginning of the MEG epoch with regard to the reference
            event, in seconds. Defaults to 0.

        class_names : list of str, optional
            Names of the classes, used for the legend. Defaults to
            None (auto-generated as "Class {i}").

        bp_filter : tuple of float, or False, optional
            If a ``(l_freq, h_freq)`` tuple, band-pass filter the
            waveforms before plotting. Defaults to False (no
            filtering).

        tlim : tuple of float, optional
            x-axis (time) limits applied to the waveform and
            temporal-convolution-output subplots. Defaults to None
            (no limit).

        apply_kernels : bool, optional
            If True, convolve each waveform with its corresponding
            temporal filter kernel before plotting, instead of just
            scaling it. Defaults to False.

        Returns
        -------
        None
            Displays the resulting figure with
            ``matplotlib.pyplot.show``.
        """

        order, _ = self._sorting(patterns_struct, sorting)
        self.uorder = order.ravel()
        waveforms = patterns_struct['ccms']['tconv']

        if not class_names:
            class_names = ["Class {}".format(i) for i in range(self.y_shape[-1])]

        f, ax = plt.subplots(2, 2)
        f.set_size_inches([16, 16])
        if np.any(self.uorder):
            #for jj, uo in enumerate(self.uorder):
            nt = self.dataset.h_params['n_t']

            tstep = 1/float(self.dataset.h_params['fs'])
            times = tmin + tstep*np.arange(nt)
            if apply_kernels:
                scaled_waveforms = np.array([np.convolve(kern, wf, 'same')
                            for kern, wf in zip(self.filters, self.waveforms)])
            else:
                scaled_waveforms = (waveforms - waveforms.mean(-1, keepdims=True))  / (2*waveforms.std(-1, keepdims=True))
            if bp_filter:
                scaled_waveforms = scaled_waveforms.astype(np.float64)
                scaled_waveforms = filter_data(scaled_waveforms,
                                                  self.dataset.h_params['fs'],
                                                  l_freq=bp_filter[0],
                                                  h_freq=bp_filter[1],
                                                  method='iir',
                                                  verbose=False)
            [ax[0, 0].plot(times, wf, color='tab:grey', alpha=.25)
             for i, wf in enumerate(scaled_waveforms) if i not in self.uorder]

            [ax[0, 0].plot(times,
                          scaled_waveforms[uo],
                          linewidth=2., label=class_names[i], alpha=.75)
             for i, uo in enumerate(self.uorder)]
            ax[0, 0].set_title('Latent component waveforms')
            if tlim:
                ax[0, 0].set_xlim(tlim)

            tstep = float(self.specs['stride'])/self.dataset.h_params['fs']
            strides1 = np.arange(times[0], times[-1] + tstep/2, tstep)
            ax[1, 0].pcolor(strides1, np.arange(self.specs['n_latent']),
                           np.mean(self.tc_out, 0).T, #shading='auto'
                           )

            ax[1, 0].set_title("Avg. Temporal Convolution Output")
            ax[1, 0].set_ylabel("Component index")
            ax[1, 0].set_xlabel("Time, s")
            if tlim:
                ax[1, 0].set_xlim(tlim)
            if not hasattr(self, 'pattern_weights'):
                pattern_weights = np.einsum('ijk, jkl ->ikl', self.tc_out, self.out_weights)
                self.pattern_weights = np.maximum(pattern_weights + self.out_biases[None, :], 0.).mean(0)

            a = ax[0, 1].pcolor(self.pattern_weights, cmap='bone_r')
            divider = make_axes_locatable(ax[0,1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(a, cax=cax, orientation='vertical')
            r = [ptch.Rectangle((i, uo), width=1,
                                height=1, angle=0.0) for i, uo in enumerate(self.uorder)]
            pc = collections.PatchCollection(r, facecolor=None, alpha=.5,
                                             linewidth=2.,
                                             edgecolor='tab:orange')
            ax[0, 1].add_collection(pc)

            ax[0, 1].set_title("Pattern weights")
            ax[0, 1].set_ylabel("Component index")
            ax[0, 1].set_xticks(np.arange(0.5, 0.5+len(class_names), 1))
            ax[0, 1].set_xticklabels(class_names)
            rpss = []
            for i, flt in enumerate(self.filters.T):

                flt -= flt.mean()
                h = self.freq_responses[i, :]
                psd = self.psds[i, :]
                rpss.append((psd*h)) #%)/np.sum(psd*h)

            [ax[1, 1].plot(self.freqs, rpss[uo], linewidth=2.5, label=class_names[i])
                             for i, uo in enumerate(self.uorder)]
            ax[1, 1].set_xlim(0, 90.)
            ax[1, 1].set_title("Relative power, %")
            ax[1, 1].set_xlabel("Frequency, Hz")
            ax[1, 1].legend()
            plt.show()
            return

    def plot_combined_pattern(self, method='weight', sensor_layout=None,
                              names=None, n_comp=1, plot_true_evoked=False,
                              savefig=None):
        """Plot the mean (cross-validation-averaged) spatial pattern
        for each class as a topographic map.

        Parameters
        ----------
        method : str, optional
            Which pattern to plot. If patterns have been collected
            across folds (``self.cv_patterns``), the corresponding
            entry is averaged over folds; otherwise, for
            ``'weight'`` or ``'compwise_loss'``,
            ``self.single_pattern`` is used. Defaults to 'weight'.

        sensor_layout : str
            Name of the MNE sensor layout used to plot the
            topography.

        names : list of str, optional
            Class names used as topomap labels. Defaults to None
            (auto-generated as "Class {i}").

        n_comp : int, optional
            Number of components used when falling back to
            ``self.single_pattern``. Defaults to 1.

        plot_true_evoked : bool, optional
            If True, additionally plot the true (data-derived) evoked
            pattern via :meth:`plot_evoked_peaks`. Defaults to False.

        savefig : bool, optional
            If truthy, save the resulting figure(s) to SVG file(s).
            Defaults to None.

        Returns
        -------
        None
        """
        if not names:
            names = ['Class {}'.format(i) for i in range(self.y_shape[-1])]

        if len(self.cv_patterns.items()) > 0:
            print("Restoring from:", method )
            topos = np.mean(self.cv_patterns[method]['spatial'],
                                     -1)
            filters = np.mean(self.cv_patterns[method]['temporal'],
                                       -1)
            psds = np.mean(self.cv_patterns[method]['psds'],
                                    -1)


        elif method in ['weight', 'compwise_loss']:
            topos, filters, psds = self.single_pattern(sorting=method,
                                                       n_comp=n_comp)


        freqs = self.cv_patterns['freqs']

        topos /= np.maximum(topos.std(axis=0, keepdims=True),
                                     1e-3)
        n = self.y_shape[0]
        ncols = n
        lo = channels.read_layout(sensor_layout)
        #lo = channels.generate_2d_layout(lo.pos)
        info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
        orig_xy = np.mean(lo.pos[:, :2], 0)
        for i, ch in enumerate(lo.names):
            if info['chs'][i]['ch_name'] == ch:
                info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/4.5
            else:
                print("Channel name mismatch. info: {} vs lo: {}".format(
                    info['chs'][i]['ch_name'], ch))

        self.fake_evoked = evoked.EvokedArray(topos, info)
        self.fake_evoked.data[:, :n] = topos

        fake_times = np.arange(0,  n, 1.)
        ft = self.fake_evoked.plot_topomap(times=fake_times,
                                          #axes=ax[0, 0],
                                          colorbar=True,
                                          #vmax=vmax,
                                          scalings=1,
                                          time_format=method,
                                          #title='',
                                          #size=1,
                                          outlines='head',
                                          )
        if savefig:
            figname = '-'.join([self.meta.data['path'] + method, "topos.svg"])
            ft.savefig(figname, format='svg', transparent=True)
        if plot_true_evoked:
            t = self.plot_evoked_peaks(None, sensor_layout=sensor_layout,
                                       title='True evoked')
            figname = '-'.join([self.meta.data['path'] + self.scope, self.meta.data['data_id'], 'true', "topos.svg"])
            t.savefig(figname, format='svg', transparent=True)


           
class EnvelopNet(LFCNN):
    """
        Petrosyan, A., Sinkin, M., Lebedev, M. A., & Ossadtchi, A.  Decoding and interpreting cortical signals with
        a compact convolutional neural network, 2021, Journal of Neural Engineering, 2021,
        https://doi.org/10.1088/1741-2552/abe20e
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """Initialize an EnvelopNet model (Petrosyan et al., 2021), a
        two-stage variant of LF-CNN that separately convolves and
        pools the temporal and envelope information.

        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object; ``meta.model_specs`` is populated with
            this model's default hyperparameters (the same defaults
            as :meth:`LFCNN.__init__`) where not already set.

        dataset : mneflow.Dataset, optional
            Dataset object. Defaults to None (built from ``meta``).

        specs : dict, optional
            If provided, merged into ``meta.model_specs`` before
            applying the defaults. Dictionary of model
            hyperparameters; see :meth:`LFCNN.__init__` for the
            supported keys and their defaults.

        specs_prefix : bool, optional
            See :meth:`mneflow.models.BaseModel.__init__`. Defaults
            to False.
        """
        if specs:
            meta.update(model_specs=specs)
        #specs = meta.model_specs
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', ['fc', 'dmx', 'tconv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        super().__init__(meta, dataset, specs_prefix)
        self.scope = 'envelopnet_lv'
        meta.model_specs['scope'] = self.scope
        self.specs = meta.model_specs


    def build_graph(self):
        """Build the computational graph using the defined
        placeholder ``self.X`` as input: spatial demixing, followed
        by a temporal convolution/pooling stage on the raw signal
        (``tconv``/``tpool``) and a second temporal
        convolution/pooling stage on its envelope (``envconv``/
        ``envpool``), then dropout and a final fully-connected layer.

        Returns
        -------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.dmx = DeMixing(size=self.specs['n_latent'],
                            nonlin=keras.activations.linear,
                            axis=3, specs=self.specs)
        self.dmx_out = self.dmx(self.inputs)

        self.tconv = VARConv(
            size=self.specs['n_latent'],
            nonlin=self.specs['nonlin'],
            filter_length=self.specs['filter_length'],
            padding=self.specs['padding'],
            specs=self.specs)

        self.tconv_out = self.tconv(self.dmx_out)
        self.tpool = TempPooling(pooling=1,#self.specs['filter_length']//2,
                                 pool_type=self.specs['pool_type'],
                                 stride=1,#self.specs['filter_length']//2,
                                 padding='SAME'
                                 )

        self.tpooled = self.tpool(self.tconv_out)

        self.envconv = VARConv(
            size=self.specs['n_latent'],
            nonlin=self.specs['nonlin'],
            filter_length=self.specs['filter_length'],
            padding=self.specs['padding'],
            specs=self.specs
        )

        self.envconv_out = self.envconv(self.tpooled)
        self.envpool = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding='SAME'
                                  )
        self.pooled = self.envpool(self.envconv_out)

        self.dropout = Dropout(self.specs['dropout'], noise_shape=None)(self.pooled)
        
        self.fin_fc = FullyConnected(size=self.out_dim, 
                                     nonlin=keras.activations.linear,
                                     specs=self.specs)

        self.y_pred = self.fin_fc(self.dropout)

        return self.y_pred

    def compute_patterns(self, data_path=None, *, output='patterns'):
        """Compute spatial patterns, temporal filters and feature
        relevances for an EnvelopNet model. Overrides
        :meth:`LFCNN.compute_patterns` with a different signature and
        different computation, storing its results as instance
        attributes rather than returning a dict.

        Parameters
        ----------
        data_path : str, list of str, mneflow.Dataset, tf.data.Dataset, or None, optional
            Data on which the patterns are estimated. If None
            (default), the model's validation dataset
            (``self.dataset.val``) is used. See
            :meth:`LFCNN.compute_patterns` for the other accepted
            types.

        output : str, optional
            If it contains ``'patterns'``, spatial patterns are
            computed by convolving the input with each component's
            temporal filter and left-multiplying by the spatial
            (demixing) weights; if it additionally contains
            ``'old'``, patterns are instead computed as the dot
            product of the data covariance and the demixing weights.
            If it does not contain ``'patterns'``, the raw demixing
            weights are used as patterns. Defaults to 'patterns'.

        Returns
        -------
        None
            Sets ``self.out_w_flat``, ``self.out_weights``,
            ``self.out_biases``, ``self.feature_relevances``,
            ``self.branch_relevance_loss`` (via
            :meth:`branchwise_loss`), ``self.dcov``,
            ``self.patterns``, ``self.lat_tcs``, ``self.filters``,
            ``self.tc_out`` and ``self.corr_to_output``.

        Raises
        ------
        AttributeError
            If ``data_path`` is not None and is not a string, list,
            tuple, ``mneflow.Dataset`` or ``tf.data.Dataset``.
        """

        if not data_path:
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
            ds = self.dataset._build_dataset(
                data_path,
                split=False,
                test_batch=None,
                repeat=True
            )
        elif isinstance(data_path, Dataset):
            if hasattr(data_path, 'test'):
                ds = data_path.test
            else:
                ds = data_path.val
        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')

        X, y = [row for row in ds.take(1)][0]

        self.out_w_flat = self.fin_fc.w.numpy()
        self.out_weights = np.reshape(
            self.out_w_flat,
            [-1, self.dmx.size, self.out_dim]
        )
        self.out_biases = self.fin_fc.b.numpy()
        self.feature_relevances = self.componentwise_loss(X, y)
        self.branchwise_loss(X, y)

        # compute temporal convolution layer outputs for vis_dics
        tc_out = self.pool(self.tconv(self.dmx(X)).numpy())

        # compute data covariance
        X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
        X = tf.transpose(X, [3, 0, 1, 2])
        X = tf.reshape(X, [X.shape[0], -1])
        self.dcov = tf.matmul(X, tf.transpose(X))

        # get spatial extraction fiter weights
        demx = self.dmx.w.numpy()

        kern = np.squeeze(self.tconv.filters.numpy()).T

        X = X.numpy().T

        patterns = []
        X_filt = np.zeros_like(X)
        for i_comp in range(kern.shape[0]):
            for i_ch in range(X.shape[1]):
                x = X[:, i_ch]
                X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
            patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
        self.patterns = np.array(patterns).T


        if 'patterns' in output:
            if 'old' in output:
                self.patterns = np.dot(self.dcov, demx)
            else:
                patterns = []
                X_filt = np.zeros_like(X)
                for i_comp in range(kern.shape[0]):
                    for i_ch in range(X.shape[1]):
                        x = X[:, i_ch]
                        X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
                    patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
                self.patterns = np.array(patterns).T
        else:
            self.patterns = demx

        self.lat_tcs = np.dot(demx.T, X.T)

        del X

        #  Temporal conv stuff
        self.filters = kern.T
        self.tc_out = np.squeeze(tc_out)
        self.corr_to_output = self.get_output_correlations(y)


    def branchwise_loss(self, X, y):
        """Estimate the relevance of each latent component (branch)
        to the model's loss, by zeroing out its spatial and temporal
        weights/biases and measuring the resulting change in loss.

        Parameters
        ----------
        X : tf.Tensor or np.array
            Input data on which the model is evaluated.

        y : tf.Tensor or np.array
            True target values corresponding to ``X``.

        Returns
        -------
        None
            Sets ``self.branch_relevance_loss``, the per-component
            decrease in loss (baseline loss minus loss with that
            component's weights zeroed).
        """
        model_weights_original = self.km.get_weights().copy()
        base_loss, _ = self.km.evaluate(X, y, verbose=0)

        losses = []
        for i in range(self.specs["n_latent"]):
            model_weights = model_weights_original.copy()
            spatial_weights = model_weights[0].copy()
            spatial_biases = model_weights[1].copy()
            temporal_biases = model_weights[3].copy()
            env_biases = model_weights[5].copy()
            spatial_weights[:, i] = 0
            spatial_biases[i] = 0
            temporal_biases[i] = 0
            env_biases[i] = 0
            model_weights[0] = spatial_weights
            model_weights[1] = spatial_biases
            model_weights[3] = temporal_biases
            model_weights[5] = env_biases
            self.km.set_weights(model_weights)
            losses.append(self.km.evaluate(X, y, verbose=0)[0])
        self.km.set_weights(model_weights_original)
        self.branch_relevance_loss = base_loss - np.array(losses)


class SourceNet(BaseModel):
    """Source-space variant of LF-CNN: applies the temporal
    convolution before spatial demixing (LFTConv -> DeMixing ->
    TempPooling), followed by a second temporal-convolution/pooling
    stage on the envelope (LFTConv -> TempPooling), dropout and a
    final fully-connected layer.
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """Initialize a (source-space) SourceNet model.

        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object; ``meta.model_specs`` is populated with
            this model's default hyperparameters (the same defaults
            as :meth:`LFCNN.__init__`, but with
            ``l1_scope`` defaulting to
            ``['fc', 'demix', 'lf_conv']``) where not already set.

        dataset : mneflow.Dataset, optional
            Dataset object. Defaults to None (built from ``meta``).

        specs : dict, optional
            If provided, merged into ``meta.model_specs`` before
            applying the defaults.

        specs_prefix : bool, optional
            See :meth:`mneflow.models.BaseModel.__init__`. Defaults
            to False.
        """
        self.nfft = 128
        if specs:
            meta.update(model_specs=specs)
        #specs = meta.model_specs
        self.scope = 'sourcenet'
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        super(SourceNet, self).__init__(meta, dataset, specs_prefix)




    def build_graph(self):
        """Build the computational graph using the defined
        placeholder ``self.X`` as input: a temporal convolution
        (``tconv``) followed by spatial demixing (``dmx``), a
        temporal-convolution/pooling stage on the result
        (``tpool``), a second temporal convolution/pooling stage
        (``envconv``/``envpool``), dropout and a final
        fully-connected layer.

        Returns
        -------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """

        self.tconv = LFTConv(
            size=self.specs['n_latent'],
            nonlin=self.specs['nonlin'],
            filter_length=self.specs['filter_length'],
            padding=self.specs['padding'],
            specs=self.specs)

        self.tconv_out = self.tconv(self.inputs)


        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
                            axis=3, specs=self.specs)
        self.dmx_out = self.dmx(self.tconv_out)

        self.tpool = TempPooling(pooling=self.specs['filter_length']//2,
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['filter_length']//2,
                                  padding='SAME'
                                  )

        self.tpooled = self.tpool(self.dmx_out)

        self.envconv = LFTConv(
            size=self.specs['n_latent'],
            nonlin=self.specs['nonlin'],
            filter_length=self.specs['filter_length'],
            padding=self.specs['padding'],
            specs=self.specs
        )

        self.envconv_out = self.envconv(self.tpooled)

        self.envpool = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding='SAME'
                                  )
        self.pooled = self.envpool(self.envconv_out)

        self.dropout = Dropout(self.specs['dropout'], noise_shape=None)(self.pooled)

        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        self.y_pred = self.fin_fc(self.dropout)

        return self.y_pred

    # def plot_branch(
    #     self,
    #     branch_num: int,
    #     info: Info,
    #     params: 'input' #Optional[list[str]] = ['input', 'output', 'response']
    #     ):
    #     info.__setstate__(dict(_unlocked=True))
    #     info['sfreq'] = 1.
    #     sorting = np.argsort(self.branch_relevance_loss)[::-1]
    #     data = self.patterns[:, sorting]
    #     filters = self.filters[:, sorting]
    #     relevances = self.branch_relevance_loss - self.branch_relevance_loss.min()
    #     relevance = sorted([np.round(rel/relevances.sum(), 2) for rel in relevances], reverse=True)[branch_num]
    #     self.fake_evoked = evoked.EvokedArray(data, info, tmin=0)
    #     fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
    #     fig.tight_layout()

    #     self.fs = self.dataset.h_params['fs']

    #     out_filter = filters[:, branch_num]
    #     _, psd = welch(self.lat_tcs[branch_num], fs=self.fs, nperseg=self.fs * 2)
    #     w, h = (lambda w, h: (w, h))(*freqz(out_filter, 1, worN=self.fs))
    #     frange = w / np.pi * self.fs / 2
    #     z = lambda x: (x - x.mean())/x.std()

    #     for param in params:
    #         if param == 'input':
    #             finput = psd[:-1]
    #             finput = z(finput)
    #             ax2.plot(frange, finput - finput.min(), color='tab:blue')
    #         elif param == 'output':
    #             foutput = np.real(finput * h * np.conj(h))
    #             foutput = z(foutput)
    #             ax2.plot(frange, foutput - foutput.min(), color='tab:orange')
    #         elif param == 'response':
    #             fresponce = np.abs(h)
    #             fresponce = z(fresponce)
    #             ax2.plot(frange, fresponce - fresponce.min(), color='tab:green')
    #         elif param == 'pattern':
    #             fpattern = finput * np.abs(h)
    #             fpattern = z(fpattern)
    #             ax2.plot(frange, fpattern - fpattern.min(), color='tab:pink')

    #     ax2.legend([param.capitalize() for param in params])
    #     ax2.set_xlim(0, 100)

    #     fig.suptitle(f'Branch {branch_num}', y=0.95, x=0.2, fontsize=30)
    #     fig.set_size_inches(10, 5)
    #     self.fake_evoked.plot_topomap(
    #         times=branch_num,
    #         axes=ax1,
    #         colorbar=False,
    #         scalings=1,
    #         time_format="",
    #         outlines='head',
    #     )

    #     return fig

class WFNet(LFCNN):
    """Temporal-convolution + LSTM model. Applies a temporal
    convolution (LFTConv) to the input, then feeds the result to an
    LSTM layer followed by a fully-connected output layer. Unlike
    :class:`LFCNN`, it does not perform spatial demixing and does
    not include the pattern-interpretation methods of the LF-CNN
    family.
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """Initialize a WFNet model.

        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object; ``meta.model_specs`` is populated with
            this model's default hyperparameters (see below) where
            not already set.

        dataset : mneflow.Dataset, optional
            Dataset object. Defaults to None (built from ``meta``).

        specs_prefix : bool, optional
            See :meth:`mneflow.models.BaseModel.__init__`. Defaults
            to False.

        specs : dict, optional
                If provided, merged into ``meta.model_specs`` before
                applying the defaults below. Dictionary of model
                hyperparameters {

        n_latent : int
            Number of latent components.
            Defaults to 32.

        nonlin : callable
            Activation function of the temporal convolution layer.
            Defaults to tf.nn.relu

        filter_length : int
            Length of spatio-temporal kernels in the temporal
            convolution layer. Defaults to 16.

        pooling : int
            Pooling factor of the max pooling layer. Defaults to 2

        pool_type : str {'avg', 'max'}
            Type of pooling operation. Defaults to 'max'.

        padding : str {'SAME', 'FULL', 'VALID'}
            Convolution padding. Defaults to 'SAME'.}

        stride : int
        Stride of the max pooling layer. Defaults to 2.

        """
        self.scope = 'lfcnnr'
        self.nfft = 128
        if specs:
            meta.update(model_specs=specs)
        #specs = meta.model_specs
        meta.model_specs.setdefault('filter_length', 16)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', ['fc', 'dmx', 'tconv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        #specs.setdefault('model_path',  self.dataset.h_params['save_path'])
        super(WFNet, self).__init__(meta, dataset, specs_prefix)
        #super().__init__(meta)


    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input: a temporal convolution (LFTConv) followed by an
        LSTM layer and a final fully-connected output layer.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        #Apply n_latent temporal convolution kernels
        self.scope = 'wfnet'
        
        #inputs = keras.ops.transpose(self.inputs,[0,3,2,1])
        
        # self.dmx = DeMixing(size=self.specs['n_latent'], 
        #                     nonlin=keras.activations.linear,
        #                     axis=3, specs=self.specs)
        
        # self.dmx_out = self.dmx(self.inputs)
        self.tconv = LFTConv(
            size=self.specs['n_latent'],
            nonlin=self.specs['nonlin'],
            filter_length=self.specs['filter_length'],
            padding=self.specs['padding'],
            specs=self.specs)
        
        self.tconv_out = self.tconv(self.inputs)
        reshaped = keras.layers.Reshape(self.tconv_out.shape[2:])(self.tconv_out)
        print("LSTM input: {}".format(reshaped.shape))
        self.lstm = LSTM(units=self.specs['n_latent'], activation='tanh', 
                            input_shape=reshaped.shape[1:],
                            recurrent_activation='sigmoid',
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal',
                            bias_initializer='zeros',
                            unit_forget_bias=True,
                            kernel_regularizer=None,
                            recurrent_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            recurrent_constraint=None,
                            bias_constraint=None,
                            dropout=self.specs['dropout'],
                            recurrent_dropout=self.specs['dropout'],
                            seed=None,
                            return_sequences=False,
                            return_state=False,
                            go_backwards=False,
                            stateful=False,
                            unroll=False,
                            use_cudnn='auto',
                            )

        lstm_state = self.lstm(reshaped)        
        print("LSTM output: {}".format(lstm_state.shape))
        fc_out = FullyConnected(size=self.out_dim, nonlin=keras.activations.linear,
                            specs=self.specs)
        
        self.y_pred = fc_out(lstm_state)
        
        return self.y_pred