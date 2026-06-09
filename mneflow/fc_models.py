#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:33:35 2024

@author: eero.saarro@aalto.fi
"""

from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Flatten, Dropout, BatchNormalization,
                                     Conv2D, SpatialDropout2D, Conv3D,
                                     SpatialDropout3D, AveragePooling2D,
                                     MaxPooling2D, Lambda, DepthwiseConv2D)
from tensorflow.keras import initializers

working_directory = Path("/m/nbe/scratch/restmeg/eero/code/mneflow/")
os.chdir(working_directory)
import mneflow
from mneflow.layers import SquareSum3d, WeightSum3d
from mneflow.layers import se_block, channel_attention, spatial_attention


class Conv3DModel(mneflow.models.BaseModel):
    """3D convolutional model designed to decode connectomes (Model 3).
    Extracts features simultaneously along the row fingerprints for each ROI and
    across the frequency channels. Channels should be the last dimension of the
    input data: Shape: ( number of subjects x rows x columns x channels ).

    The model architecture is as follows:
        - Gaussian noise layer (optional, based on stddev hyperparameter)
        - 3D convolutional layer with kernel size (1, n_nodes, n_freqs) to
            convolve across connectivity fingerprints and frequencies.
        - Batch normalization and spatial dropout (optional)
        - Linear convolution (Conv2D with kernel size (n_nodes, 1)) to convolve
            across the remaining spatial dimension).
        - Batch normalization and spatial dropout (optional)
        - Flattening layer
        - Fully connected layer with ReLU activation and L1 regularization
        - Batch normalization and dropout
        - Output layer with linear activation and L1 regularization
    """

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """Initialize the 3D convolutional model.

        Parameters
        ----------
        meta : mneflow meta file
        Dataset : mneflow.Dataset
        specs : dict
            Dictionary of model hyperparameters.

            n_latent_dense : int, optional
                Number of latent components in the first dense layer.
                Defaults to 32.
            n_latent : int, optional
                Number of filters in the 3D convolutional layer. Defaults to 32.
            nonlin : callable, optional
                Activation function for the convolutional layers. Defaults to
                `tf.nn.relu`.
            spatial_dropout : float, optional
                Dropout rate for spatial dropout layers after the convolutional
                layers. If set to 0, spatial dropout is not applied after the
                convolutional layers. Defaults to 0.1.
            dropout : float, optional
                Dropout rate  for the fully connected layers. Defaults to 0.4.
            stddev : int, optional
                Standard deviation used for Gaussian layer applied to the inputs.
                If set to 0, Gaussian noise is not applied to the inputs.
                Defaults to 0.1.
        """
        self.scope = 'conv3d'
        meta.model_specs.setdefault('n_latent_dense', 32)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('spatial_dropout', 0.1)
        meta.model_specs.setdefault('dropout', 0.4)
        meta.model_specs.setdefault('stddev', 0.1)
        meta.model_specs.setdefault('stride', 1)
        meta.model_specs.setdefault('nonlin', 'relu')
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope

        super(Conv3DModel, self).__init__(meta=meta, dataset=dataset,
                                          specs_prefix=specs_prefix)

    def build_graph(self):
        """Build the computational graph for the 3D convolutional model using
        defined placeholder `self.X` as input.

        Returns
        -------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.specs = self.meta.model_specs

        # Gaussian Noise layer
        if self.specs['stddev'] > 0:
            self.gaussian_noise = tf.keras.layers.GaussianNoise(
                stddev=self.specs['stddev'])
            inputs1 = self.gaussian_noise(self.inputs)
            print(f"Gaussian noise with stddev: {self.specs['stddev']} "
                  f"applied to the inputs")
        else:
            inputs1 = self.inputs

        # Expand dimensions to add a channel dimension for Conv3D
        inputs1 = Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs1)
        print("Shape before 3D convolution layer:", inputs1.shape)

        # 3D convolutional layer with kernel size (1, n_nodes, n_freqs) to
        # convolve across connectivity fingerprints and frequencies.
        n_nodes = inputs1.shape[-3]
        n_freqs = inputs1.shape[-2]
        self.col_conv3d = Conv3D(filters=self.specs['n_latent'],
                                 kernel_size=(1, n_nodes, n_freqs),
                                 padding="valid",
                                 data_format=None,
                                 kernel_initializer=initializers.HeUniform(),
                                 bias_initializer=initializers.Constant(0.1),
                                 activation=self.specs['nonlin'])
        rsum = self.col_conv3d(inputs1)
        print(f"Built: 3D layer with {self.specs['n_latent']} filters. "
              f"Input shape before: {inputs1.shape}, shape after: {rsum.shape}")

        # Batch normalization and spatial dropout (optional)
        if self.specs['spatial_dropout'] > 0:
            rsum = BatchNormalization()(rsum)
            rsum = SpatialDropout3D(rate=self.specs['spatial_dropout'],
                                     data_format='channels_last')(rsum)

        # Linear convolution (Conv2D with kernel size (n_nodes, 1)) to convolve
        # across the remaining spatial dimension).
        rsum = Lambda(lambda x: tf.squeeze(x, axis=[2, 3]))(rsum)
        rsum = Lambda(lambda x: tf.expand_dims(x, axis=-1))(rsum)
        self.conv2d = Conv2D(filters=self.specs['n_latent'] // 2,
                             kernel_size=(n_nodes, 1),
                             padding='valid',
                             kernel_initializer=initializers.HeUniform(),
                             bias_initializer=initializers.Constant(0.1),
                             activation='relu')
        out = self.conv2d(rsum)
        print(f"Built: 2D layer with {self.specs['n_latent'] // 2} filters. "
              f"Input shape before: {rsum.shape}, shape after: {out.shape}")

        # Batch normalization and spatial dropout (optional)
        if self.specs['spatial_dropout'] > 0:
            out = BatchNormalization()(out)
            out = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                   data_format='channels_last')(out)

        # Flattening layer
        flat = Flatten()(out)
        print("Shape after flattening: ", flat.shape)

        # Fully connected layer with ReLU activation and L1 regularization
        self.fc1 = Dense(units=self.specs['n_latent_dense'],
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l1(
                             self.specs['l1_lambda']))
        fc1 = self.fc1(flat)
        print(f"Built: Dense layer with {self.specs['n_latent_dense']} units. "
              f"Input shape before: {flat.shape}, shape after: {fc1.shape}")

        # Batch normalization and dropout
        self.b2n = BatchNormalization(scale=False)
        b_last = self.b2n(fc1)
        dropout = Dropout(self.specs['dropout'], noise_shape=None)(b_last)

        # Output layer with linear activation and L1 regularization
        self.fc2 = Dense(
            units=1,
            activation=tf.identity,
            kernel_regularizer=tf.keras.regularizers.l1(
                self.specs['l1_lambda']))
        y_pred = self.fc2(dropout)

        return y_pred


class WeightedSum3dModel(mneflow.models.BaseModel):
    """Weighted sum 3D model designed to decode connectomes (Model 2).

    The model architecture is as follows:
        - Gaussian noise layer (optional, based on stddev hyperparameter)
        - First WeightSum3d layer (depthwise convolution using global row
            filters)
        - Batch normalization and spatial dropout
        - Second WeightSum3d layer (depthwise convolution using global column
            filters, optional based on n_latent2 hyperparameter)
        - Batch normalization and spatial dropout (optional)
        - Optional channel attention block
        - Pointwise convolution (Conv2D with 1x1 kernel)
        - Flattening layer
        - Fully connected layer with ReLU activation and L1 regularization
        - Batch normalization and dropout
        - Output layer with linear activation and L1 regularization"""

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """
        Parameters
        ----------
        meta : mneflow meta file
        Dataset : mneflow.Dataset
        specs : dict
           Dictionary of model hyperparameters.

           n_latent_dense : int, optional
               Number of latent components in the first dense layer. Defaults to 32.
           n_latent1 : int, optional
               Number of filters in the first depthwise convolutional layer. Defaults to 8.
           n_latent2 : int, optional
               Number of filters in the second depthwise convolutional layer.
               If set to 0,only the first depthwise layer is applied.
               Defaults to 8.
           n_latent_cross : int, optional
               Number of filters in the pointwise convolutional layer. Defaults to 1.
           nonlin : callable, optional
               Activation function for the convolutional layers. Defaults to `tf.nn.relu`.
           spatial_dropout : float, optional
               Dropout rate for spatial dropout layers after the convoultional layers.
               If set to 0, spatial dropout is not applied after the convolutional layers.
               Defaults to 0.1.
           dropout : float, optional
               Dropout rate  for the fully connected layers. Defaults to 0.4.
           stddev : int, optional
               Standard deviation used for Gaussian layer applied to the inputs.
               If set to 0, Gaussian noise is not applied to the inputs.
               Defaults to 0.05.
           attention_rate: int, optional
               Attention rate for the squeeze-and-excitation channel attention block.
               For details see [1].
               If set to 0, Channel attention is not applied before the pointwise operation.
               Defaults to 0.05.

        Returns
        -------
        None.

        References
        ----------
           [1] Hu, J. et al. (2018). Squeeze-and-excitation networks.
           In Proceedings of the IEEE conference on computer vision and pattern recognition. pages 7132–7141.
        """
        self.scope = 'model2_weighted_sum_3d'

        meta.model_specs.setdefault('n_latent_dense', 32)
        meta.model_specs.setdefault('n_latent_cross', 1)
        meta.model_specs.setdefault('n_latent1', 8)
        meta.model_specs.setdefault('n_latent2', 8)
        meta.model_specs.setdefault('spatial_dropout', 0.1)
        meta.model_specs.setdefault('dropout', 0.4)
        meta.model_specs.setdefault('stddev', 0.05)
        meta.model_specs.setdefault('attention_rate', 4)
        meta.model_specs.setdefault('stride', 1)
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])

        meta.model_specs['scope'] = self.scope

        super(WeightedSum3dModel, self).__init__(meta, dataset, specs_prefix)
        super().__init__(meta=meta, dataset=dataset, specs_prefix=specs_prefix)

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.scope = 'weighted_sum_3d'
        self.specs = self.meta.model_specs

        # Gaussian Noise layer
        if self.specs['stddev'] > 0:
            self.gaussian_noise = tf.keras.layers.GaussianNoise(
                stddev=self.specs['stddev'])
            inputs1 = self.gaussian_noise(self.inputs)
            print(f"Gaussian noise with stddev: {self.specs['stddev']} "
                  f"applied to the inputs")
        else:
            inputs1 = self.inputs

        # Depth-wise convolution using global row filters to compress row dim
        print("Shape before the first depthwise convolution: ", inputs1.shape)
        self.rowsum_3d = WeightSum3d(size=self.specs['n_latent1'],
                                     specs=self.specs,
                                     nonlin=tf.nn.relu)
        rsum = self.rowsum_3d(inputs1)
        print(
            f"Built: Depthwise convolutional layer with {self.specs["n_latent1"]} filters. "
            f"Input shape before: {inputs1}, shape after: {rsum}")

        # Batch normalization and spatial dropout (optional)
        if self.specs['spatial_dropout'] > 0:
            rsum = BatchNormalization()(rsum)
            rsum = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                    data_format='channels_last')(rsum)

        # Second depth-wise convolution using global column filters to compress
        # column dimension (optional based on n_latent2 hyperparameter)
        if self.specs['n_latent2'] > 0:
            self.colsum_3d = WeightSum3d(size=self.specs['n_latent2'],
                                         specs=self.specs,
                                         nonlin=tf.nn.relu)
            csum = self.colsum_3d(rsum)
            print(
                f"Built: Depthwise convolutional layer with {self.specs["n_latent2"]} filters. "
                f"Input shape before: {rsum}, shape after: {csum}")

            # Batch normalization and spatial dropout (optional)
            if self.specs['spatial_dropout'] > 0:
                print("Adding batch norm and dropout")
                csum = BatchNormalization()(csum)
                csum = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                        data_format='channels_last')(csum)
        else:
            print("Using only one depthwise convolutional layer")
            csum = rsum

        # Channel attention block (optional)
        if self.specs['attention_rate'] > 0:
            csum = se_block(csum, ratio=self.specs['attention_rate'])
            print(
                f"Built: Squeeze-and-Excitation block added with channel reduction rate: ",
                self.specs['attention_rate'])

        # Pointwise convolution (Conv2D with 1x1 kernel)
        pointwise = Conv2D(self.specs['n_latent_cross'],
                           (1, 1),
                           activation='relu',
                           data_format='channels_last')(csum)
        print(f"Built: Pointwise layer with {self.specs['n_latent_cross']} "
              f"filters. Input shape before: {csum.shape}, shape after: {pointwise.shape}")

        # Flattening layer
        flat = Flatten()(pointwise)
        print("Shape after flattening: ", flat.shape)

        # Fully connected layer with ReLU activation and L1 regularization
        self.fc1 = Dense(units=self.specs['n_latent_dense'],
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l1(
                             self.specs['l1_lambda']))
        fc1 = self.fc1(flat)
        print(f"Built: Dense layer with {self.specs['n_latent_dense']} units. "
              f"Input shape before: {flat.shape}, shape after: {fc1.shape}")

        # Batch normalization and dropout
        self.b2n = BatchNormalization(scale=False)
        b_last = self.b2n(fc1)
        dropout = Dropout(self.specs['dropout'], noise_shape=None)(b_last)

        # Output layer with linear activation and L1 regularization
        self.fc = Dense(units=1,
                        activation=tf.identity,
                        kernel_regularizer=tf.keras.regularizers.l1(
                            self.specs['l1_lambda']))
        y_pred = self.fc(dropout)

        return y_pred


class SymmetricModel(mneflow.models.BaseModel):
    """Symmetric 3D model designed to decode connectomes (Model 1).

    Extracts features first along the row fingerprints for each ROI
    and then identifies cross-frequency interactions. Channels should be the
    last dimension of the input data:
    Shape: ( number of subjects x rows x columns x channels )

    The model architecture is as follows:
        - Gaussian noise layer (optional, based on stddev hyperparameter)
        - SquareSum3d layer (symmetric operation across rows and columns
            followed by depthwise convolution)
        - Batch normalization and spatial dropout
        - Optional channel attention block
        - Pointwise convolution (Conv2D with 1x1 kernel)
        - Flattening layer
        - Fully connected layer with ReLU activation and L1 regularization
        - Batch normalization and dropout
        - Output layer with linear activation and L1 regularization
        """

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """Initialize the symmetric model.

        Parameters
        ----------
        meta : mneflow meta file
        Dataset : mneflow.Dataset
        specs : dict
           Dictionary of model hyperparameters.

           n_latent_dense : int, optional
               Number of latent components in the first dense layer. Defaults to 32.
           n_latent : int, optional
               Number of filters in the depthwise convolutional layer. Defaults to 8.
           n_latent_cross : int, optional
               Number of filters in the pointwise convolutional layer. Defaults to 1.
           nonlin : callable, optional
               Activation function for the convolutional layers. Defaults to `tf.nn.relu`.
           spatial_dropout : float, optional
               Dropout rate for spatial dropout layers after the convoultional layers.
               If set to 0, spatial dropout is not applied after the convolutional layers.
               Defaults to 0.1.
           dropout : float, optional
               Dropout rate  for the Dense layer. Defaults to 0.4.
           stddev : int, optional
               Standard deviation used for Gaussian layer applied to the inputs.
               If set to 0, Gaussian noise is not applied to the inputs.
               Defaults to 0.05.
           attention_rate: int, optional
               Attention rate for the squeeze-and-excitation channel attention block.
               For details see [1].
               If set to 0, Channel attention is not applied before the pointwise operation.
               Defaults to 0.05.

        Returns
        -------
        None.

        References
        ----------
            [1] Hu, J. et al. (2018). Squeeze-and-excitation networks.
            In Proceedings of the IEEE conference on computer vision and pattern recognition. pages 7132–7141.
        """

        self.scope = 'symmetric_tpk'

        meta.model_specs.setdefault('n_latent_dense', 32)
        meta.model_specs.setdefault('n_latent_cross', 1)
        meta.model_specs.setdefault('n_latent', 8)
        meta.model_specs.setdefault('spatial_dropout', 0.1)
        meta.model_specs.setdefault('dropout', 0.4)
        meta.model_specs.setdefault('stddev', 0.05)
        meta.model_specs.setdefault('attention_rate', 4)
        meta.model_specs.setdefault('stride', 1)
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs.setdefault('trainable_pointwise_kernel', True)

        meta.model_specs['scope'] = self.scope

        super(SymmetricModel, self).__init__(meta=meta, dataset=dataset,
                                             specs_prefix=specs_prefix)

    def build_graph(self):
        """
        Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.specs = self.meta.model_specs

        # Gaussian Noise layer
        if self.specs['stddev'] > 0:
            self.gaussian_noise = tf.keras.layers.GaussianNoise(
                stddev=self.specs['stddev'])
            inputs1 = self.gaussian_noise(self.inputs)
            print(f"Gaussian noise with stddev: {self.specs['stddev']} "
                  f"applied to the inputs")
        else:
            inputs1 = self.inputs

        # SquareSum3d layer (symmetric operation across rows and columns
        # followed by deptwise convolution)
        self.squaresum_3d = SquareSum3d(size=self.specs['n_latent'],
                                        specs=self.specs,
                                        nonlin=tf.nn.relu)
        ssum1 = self.squaresum_3d(inputs1)
        print(
            f"Built: Depthwise convolutional layer with {self.specs["n_latent"]} "
            f"filters. Input shape before: {inputs1.shape}, shape after: {ssum1.shape}")

        # Batch normalization and spatial dropout (optional)
        if self.specs['spatial_dropout'] > 0:
            ssum1 = BatchNormalization()(ssum1)
            ssum1 = SpatialDropout2D(rate=self.specs['spatial_dropout'],
                                     data_format='channels_last')(ssum1)

        # Channel attention block (optional)
        if self.specs['attention_ratio'] > 0:
            ssum1 = se_block(ssum1, ratio=self.specs['attention_ratio'])
            print(
                f"Built: Squeeze-and-Excitation block added with channel reduction rate: ",
                self.specs['attention_rate'])

        # Pointwise convolution (Conv2D with 1x1 kernel)
        if self.specs['trainable_pointwise_kernel']:
            self.pointwise = DepthwiseConv2D(
                kernel_size=(1, self.meta.data['n_ch']),
                activation='relu',
                padding="valid",
                data_format='channels_first',
                depth_multiplier=1)
        else:
            self.pointwise = Conv2D(self.specs['n_latent_cross'], (1, 1),
                                    activation='relu',
                                    data_format='channels_last')

        pointwise = self.pointwise(ssum1)
        print(f"Built: Pointwise layer with {self.specs['n_latent_cross']} "
              f"filters. Input shape before: {ssum1.shape}, shape after: {pointwise.shape}")

        # Flattening layer
        self.flat = Flatten()
        flat = self.flat(pointwise)
        print("Flattened shape: ", flat.shape)

        # Fully connected layer with ReLU activation and L1 regularization
        self.fc1 = Dense(
            units=self.specs['n_latent_dense'],
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l1(
                self.specs['l1_lambda']))
        fc1 = self.fc1(flat)

        print(f"Built: Dense layer with {self.specs['n_latent_dense']} units. "
              f"Input shape before: {flat.shape}, shape after: {fc1.shape}")

        # Batch normalization and dropout
        self.b2n = BatchNormalization(scale=False)
        batch = self.b2n(fc1)
        dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(batch)

        # Output layer with linear activation and L1 regularization
        self.fc = Dense(
            units=1,
            activation=tf.identity,
            kernel_regularizer=tf.keras.regularizers.l1(
                self.specs['l1_lambda']))
        y_pred = self.fc(dropout)

        return y_pred

    def extract_weights(self, verbose=False):
        weights = {}

        # Extract weights

        # Connectivity fingerprint extraction fiters
        weights['ssum1'] = np.squeeze(self.squaresum_3d.w.numpy())
        weights['ssum1b'] = self.squaresum_3d.b_in.numpy()

        # Pointwise Convolution kernels
        weights['pointwise'] = np.squeeze(self.pointwise.kernel.numpy())

        # First dense layer
        if hasattr(self, "fc1"):
            weights['fc1_w_flat'] = self.fc1.kernel.numpy()

        # Final layer
        weights['out_w_flat'] = self.fc.kernel.numpy()

        if verbose:
            print("""Weights: \n
                  SPATIAL: {}
                  POINTWISE: {}

                  FC_OUT: {}""".format(weights['ssum1'].shape,
                  weights['pointwise'].shape,
                  weights['fc1_w_flat'].shape,
                  weights['out_w_flat'].shape))

        return weights

    def compute_patterns(self, data_path=None, verbose=False, shapley_order=1,
                         methods=['weight', 'compwise_loss', 'output_corr']):
        """Extracts weights, activation patterns, validation data covariances,
        and feature relevances for a single training fold.
        Required for visualization.

        Parameters
        ----------
        data_path : str or list of str
            Path to TFRecord files on which the patterns are estimated.

        Returns:
        --------
        patterns_struct : dict
            keys
            'weights' - model weighs {layer : array}
            'ccms' - mean activations of each layer per class {layer:array}

        Raises:
        -------
            AttributeError: If `data_path` is not specified.
        """
        patterns_struct = {'weights' : {'squaresym':[], 'pointwise':[],
                                        'fin_fc':[], 'fc1':[]},
                           'ccms' : {'squaresym':[], 'pointwise':[], 'fin_fc':[],
                                     'fc1':[],
                                     'input':[]},
                           'dcov' : {'input_spatial':[], 'class_conditional':[],
                                     'k-1':[]},
                           'covs' : {'fc':[], 'fc1':[], 'squaresym':[]},
                           'patterns' : {},
                           'spectra': {},
                           'freqs': None
                           }

        if not data_path:
            print("Computing patterns: No path specified, using validation dataset (Default)")
            ds = self.dataset.val
        elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):

            ds = self.dataset._build_dataset(data_path,
                                             split=False,
                                             test_batch=None,
                                             repeat=True)

        elif isinstance(data_path, tf.data.Dataset):
            ds = data_path
        else:
            raise AttributeError('Specify dataset or data path.')


        X, y = [row for row in ds.take(1)][0]
        ndof = X.shape[0] - 1

        #get layer activations
        activations = {}
        activations['squaresym'] = self.squaresum_3d(X)
        activations['pointwise'] = self.pointwise(activations['squaresym'])
        if hasattr(self, "fc1"):
            activations['fc1'] = self.fc1(self.flat(activations['pointwise']))
            activations['fin_fc']  = self.fc(self.b2n(activations['fc1']))
        else:
            activations['fin_fc']  = self.fc(self.b2n(self.flat(activations['pointwise'])))

        if verbose:
            print(""""Activations: \n
                  SPATIAL: {}
                  POINTWISE: {}
                  FC1: {}
                  FC_OUT: {}""".format(
                  activations['squaresym'].shape,
                  activations['pointwise'].shape,
                  activations['fc1'].shape,
                  activations['fin_fc'].shape))

        patterns_struct['ccms']['squaresym'] = np.mean(activations['squaresym'], 0)
        patterns_struct['ccms']['pointwise'] = np.mean(activations['pointwise'], 0)
        if hasattr(self, "fc1"):
            patterns_struct['ccms']['fc1'] = np.mean(activations['fc1'], 0)
        patterns_struct['ccms']['fin_fc'] = np.mean(activations['fin_fc'], 0)
        patterns_struct['ccms']['input'] = np.mean(X, 0)

        n_samp = np.shape(X)[0]
        patterns_struct['covs']['fin_fc'] = np.cov(tf.transpose(activations['fc1']))
        fc1_flat = tf.reshape(activations['pointwise'],[n_samp, -1])
        patterns_struct['covs']['fc1'] = np.cov(tf.transpose(fc1_flat))

        weights = self.extract_weights()

        dcov = {}
        #Compute covariance across samples
        X -= tf.keras.ops.mean(X, axis=0, keepdims=True)
        dcov = np.einsum('hijk, hilk -> ilk', X, X) / ndof
        print("DCOV:", dcov.shape)


        patterns_struct['weights'] = weights
        patterns_struct['dcov'] = dcov

        if 'output_corr' in methods:
            patterns_struct['corr_to_output'] = self.get_output_correlations(activations, y)

        del X, activations

        return patterns_struct

    def get_roi_labels(self, percentile=80, patterns=None,
                       methods=['row', 'col', 'diag'], selection_method='activation'):
        """
        Computes ROIs from spatial activation patterns.


        Parameters
        ----------
        percentile : int, optional
            Inclusion threshold top (100 - percetile)% active sources across
            all folds based on selection_method. Default 80

        selection_method : str, optional
            How to integrate across folds. Possible arguments 'count' or
            'activation'. Default 'activation'

        patterns : dict, optional
            Output of self.extract_patterns, if not provided computed.
            Default None.

        Returns
        ------
        roi_labels : dict
            ROI indices identified by the interpretation pipeline
            {'extraction_method':[label indices]}
        """

        if not patterns:
            patterns = self.extract_patterns()

        roi_labels = {}

        for key in methods:
            data = (patterns['spatial_patterns'][key] - patterns['spatial_patterns'][key].mean(0)) / patterns['spatial_patterns'][key].std(0, keepdims=True)
            if selection_method == 'count':
                top_active = np.percentile(data, percentile, axis=0)
                top_label_inds = []
                for f in range(data.shape[-1]):
                    top_label_inds.append(data[:, f] > top_active[f])
                counts = np.sum(np.stack(top_label_inds, 0), 0)
                mask = np.where(counts >= 4)[0]
            elif selection_method == 'activation':
                mean_activation = np.mean(data, -1)
                top_active = np.percentile(mean_activation, percentile)
                mask = np.where(mean_activation > top_active)[0]

            roi_labels[key] = mask
        if 'row' in methods and 'col' in methods:
            roi_labels['row-col'] = np.unique(np.concatenate([roi_labels['row'],
                                                              roi_labels['col']]))


        return roi_labels

    def get_frequency_kernels(self, methods=['row', 'col', 'diag'], patterns=None):
        if not patterns:
            patterns = self.extract_patterns()

        freq_weights = {}
        activations = patterns_struct['ccms']['pointwise']

        for key in methods:
            data = (patterns['spatial_patterns'][key] - patterns['spatial_patterns'][key].mean(0)) / patterns['spatial_patterns'][key].std(0, keepdims=True)
            if selection_method == 'count':
                top_active = np.percentile(data, percentile, axis=0)
                top_label_inds = []
                for f in range(data.shape[-1]):
                    top_label_inds.append(data[:, f] > top_active[f])
                counts = np.sum(np.stack(top_label_inds, 0), 0)
                mask = np.where(counts >= 4)[0]
            elif selection_method == 'activation':
                mean_activation = np.mean(data, -1)
                top_active = np.percentile(mean_activation, percentile)
                mask = np.where(mean_activation > top_active)[0]

            roi_labels[key] = mask
        if 'row' in methods and 'col' in methods:
            roi_labels['row-col'] = np.unique(np.concatenate([roi_labels['row'],
                                                              roi_labels['col']]))


        return

    def ablation_analysis(self, name, label_inds, hyperparameters):
        """
        Parameters
        ----------
        name : 'str'
            Name of the pattern extraction method.
        label_inds : list of int
            Label Indices
        hyperparameters : dict
            Hyperparameters for the ablation model.

        Returns
        -------
        meta_abl : mneflow.MetaData
            DESCRIPTION.
        model_abl : mneflow.model.BaseModel
            DESCRIPTION.
        dataset : mneflow.dataset
            DESCRIPTION.

        """
        meta_abl = deepcopy(self.meta)
        meta_abl.data['sample_subset'] = label_inds
        meta_abl.data['data_id'] = "_".join([meta_abl.data['data_id'], 'abl_activation_', name])

        #meta_abl.data['n_seq'] = len(label_inds)
        #meta_abl.data['n_t'] = len(label_inds)

        dataset = mneflow.data.Dataset(meta_abl, train_batch=hyperparameters['batch_size'],
                                       sample_subset=label_inds)
        meta_abl.update(model_specs=self.meta.model_specs)
        meta_abl.weights = {}

        model_abl = mneflow.fc_models.SymmetricModel(meta=meta_abl,
                                                     dataset=dataset,
                                                     specs_prefix=False)

        return meta_abl, model_abl, dataset



    def get_output_correlations(self, activations, y_true):
        """Computes a similarity metric between each of the extracted
        features and the target variable.

        The metric is a Manhattan distance for dicrete targets, and
        Spearman correlation for continuous targets.
        """
        corr_to_output = []
        y_true = y_true.numpy()
        flat_feats = self.flat(activations['fc1']).numpy()#.reshape(y_true.shape[0], -1)


        for y_ in y_true.T:
            if self.dataset.h_params['target_type'] in ['float', 'signal']:
                rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])



            elif self.dataset.h_params['target_type'] == 'int':
                rfocs = np.array([pearsonr(y_, f)[0] for f in flat_feats.T])


            corr_to_output.append(rfocs)

        corr_to_output = np.concatenate(corr_to_output, -1)

        if np.any(np.isnan(corr_to_output)):
            corr_to_output[np.isnan(corr_to_output)] = 0
        return corr_to_output

    def collect_patterns(self, fold=0, n_folds=1, n_comp=1,
                          methods=['weight',
                                  #'compwise_loss',
                                  'output_corr'
                                  ]):
        """Collects patterns computed for each fold during corss-validation
        Returns:
        --------
        cv_patterns : dict
            Dictionary containing at least 'dcov', 'feature_relevance', 'weights', and 'ccms'

        """
        patterns_struct = self.compute_patterns()

        if len(self.cv_patterns.items()) == 0 or fold==0:
            n_parcels = self.meta.data['n_seq']
            n_fft = self.meta.data['n_ch']
            n_folds = self.meta.data['n_folds']

            #self.cv_patterns['freqs'] = patterns_struct['freqs']

            self.cv_patterns['dcov'] = np.zeros([n_parcels, n_parcels, n_fft, n_folds])
            #Feature relevances
            for method in methods:
                self.cv_patterns[method]['feature_relevance'] = np.zeros([
                                                            *self.flat.input.shape[1:],
                                                            n_folds])
            for k in patterns_struct['ccms'].keys():
                shape = [*patterns_struct['ccms'][k].shape, n_folds]

                self.cv_patterns['ccms'][k] = np.zeros(shape)

        max_weight = np.argmax(patterns_struct['weights']['out_w_flat'], 0)
        weight_relevance = patterns_struct['weights']['fc1_w_flat'][:, max_weight].reshape(self.flat.input.shape[1:])

        self.cv_patterns['weight']['feature_relevance'][:, :, :, fold] = weight_relevance
        self.cv_patterns['dcov'][:, :, :, fold] = patterns_struct['dcov']

        max_corr = np.argmax(patterns_struct['corr_to_output'], 0)
        corr_relevance = patterns_struct['weights']['fc1_w_flat'][:, max_corr].reshape(self.flat.input.shape[1:])
        self.cv_patterns['output_corr']['feature_relevance'][:, :, :, fold] = corr_relevance

        for k in patterns_struct['weights'].keys():
            self.cv_weights[k].append(patterns_struct['weights'][k])

        for k in patterns_struct['ccms'].keys():
            self.cv_patterns['ccms'][k][..., fold] = patterns_struct['ccms'][k]




    def extract_patterns(self):
        """
        Extracts activation patterns from a trained model.

        Returns
        -------
        out : dict
            DESCRIPTION.

        """
        out = defaultdict(dict)
        n_folds = self.meta.data['n_folds']
        max_inds_folds = [np.argmax(self.meta.weights['out_w_flat'][:, :, i], 0)
                          for i in range(n_folds)]
        pointwise_shape = self.cv_patterns['ccms']['pointwise'].shape[:-1]
        connectivity_patterns = defaultdict(list)
        freq_inds = defaultdict(list)
        comp_inds = defaultdict(list)
        for i, ind in enumerate(max_inds_folds):
            fc1 = np.dot(self.meta.weights['fc1_w_flat'][:, ind, i][..., None],
                         self.meta.weights['out_w_flat'][ind, :, i])
            fcr = np.reshape(fc1, pointwise_shape[:-1])

            row = np.unravel_index(np.argmax(np.diag(fcr)), fcr.shape)[0]

            max_freq_row = np.argmax(self.meta.weights['pointwise'][:, row, i])
            spatial_kernel_row = self.meta.weights['ssum1'][max_freq_row, :, row, i]
            spatial_pattern_row = np.dot(self.cv_patterns['dcov'][:, :, max_freq_row, i], spatial_kernel_row.T)

            col = np.unravel_index(np.argmax(fcr), fcr.shape)[1]

            max_freq_col = np.argmax(self.meta.weights['pointwise'][:, col, i])
            spatial_kernel_col = self.meta.weights['ssum1'][max_freq_col, :, col, i]
            spatial_pattern_col = np.dot(self.cv_patterns['dcov'][:, :, max_freq_col, i], spatial_kernel_col.T)


            diag = np.argmax(np.abs(np.diag(fcr)))
            max_freq_diag = np.argmax(np.abs(self.meta.weights['pointwise'][:, diag, i]))
            spatial_kernel_diag = self.meta.weights['ssum1'][max_freq_diag, :, diag, i]
            spatial_pattern_diag = np.dot(self.cv_patterns['dcov'][:, :, max_freq_diag, i], spatial_kernel_diag.T)

            comp_inds['row'].append(row)
            comp_inds['col'].append(col)
            comp_inds['diag'].append(diag)

            freq_inds['row'].append(max_freq_row)
            freq_inds['col'].append(max_freq_col)
            freq_inds['diag'].append(max_freq_diag)

            connectivity_patterns['row'].append(spatial_pattern_row)
            connectivity_patterns['col'].append(spatial_pattern_col)
            connectivity_patterns['diag'].append(spatial_pattern_diag)

        out['spatial_patterns']['row'] = np.stack(connectivity_patterns['row'], -1)
        out['spatial_patterns']['col'] = np.stack(connectivity_patterns['col'], -1)
        out['spatial_patterns']['diag'] = np.stack(connectivity_patterns['diag'], -1)
        out['freq_inds']['row'] = np.array(freq_inds['row'])
        out['freq_inds']['col'] = np.array(freq_inds['col'])
        out['freq_inds']['diag'] = np.array(freq_inds['diag'])
        out['comp_inds']['row'] = np.array(comp_inds['row'])
        out['comp_inds']['col'] = np.array(comp_inds['col'])
        out['comp_inds']['diag'] = np.array(comp_inds['diag'])

        return out



