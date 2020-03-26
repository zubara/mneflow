#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mneflow.Models implementation using keras models

@author: Gavriela Vranou
"""
import os
import warnings
import itertools

import tensorflow as tf
from tensorflow.keras.regularizers import l1_l2
# tf.enable_eager_execution()

import numpy as np

from mne import channels, evoked, create_info

from scipy.signal import freqz, welch
from scipy.stats import spearmanr
from spectrum import aryule

from sklearn.covariance import ledoit_wolf
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections

from mneflow import Dataset
from mneflow.models import uniquify

from keras_layers import DeMixing, VARConv, LFTConv, LSTMv1, Dense, DeConvLayer
from keras_utils import get_targets


class Model(tf.keras.Model):
    """Parent class similar to MNEflow v1 models."""

    def __init__(self, specs, dataset, name='basemodel', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters.

        dataset: mneflow.Dataset
            'Dataset' object
        """
        super(Model, self).__init__(name=name, **kwargs)
        self.scope = name
        self.specs = specs.copy()
        if dataset:
            self.dataset = dataset
        self.fs = dataset.h_params['fs']
        self._layers.remove(specs)  # Weirdly it's considered as a layer
        self.rate = specs.setdefault('dropout', 0.0)
        self.y_shape = specs['y_shape']
        self.out_dim = specs.setdefault('out_dim', np.prod(specs['y_shape']))
        self.target_type = dataset.h_params['target_type']

    def make_4D(self, X):
        """Forces 4D input tensors."""
        orig_dim = len(X.shape.as_list())
        print(X.shape.as_list())
        x0 = tf.cast(X, tf.float32)

        if orig_dim > 4:
            # Input shape: [batch_index, k, n_seq, n_ch, time]
            x0 = tf.squeeze(X, axis=0, name='x0-sq')

        elif orig_dim == 4:
            # TODO! check for self.dataset.h_params['n_seq']
            x0 = X

        elif orig_dim == 3:
            x0 = tf.expand_dims(X, -1)  # [batch_index, n_ch, time, 1]

        elif orig_dim == 2:
            x0 = tf.reshape(X, [1, *X.shape.as_list(), 1])  # [1, n_ch, time,1]

        print('input x0', x0.shape)

        return x0

    def check_yshape(self, y_):
        return tf.reshape(y_, [-1, *self.y_shape], name='check_y_shape')

    def plot_history(self, **args):
        from keras_utils import plot_history
        plot_history(self.history, self.name.upper(), **args)

    def plot_cm(self, dataset=None, dset='test', steps=1, class_names=None,
                normalize=False):
        """Plot a confusion matrix.

        Parameters
        ----------

        dataset : str {'training', 'validation'}
            Which dataset to use for plotting confusion matrix

        class_names : list of str, optional
            `class_names` is used as axes ticks. If not provided, the
            class labels are used.

        normalize : bool
            Whether to return percentages (if True) or counts (False).

        Raises:
        -------
            ValueError: If `dataset` has an unsupported value.

        Returns:
        --------
            f : Figure
                Figure handle.
        """
        if dataset is None:
            dataset = self.dataset
        y_true, y_pred = get_targets(self, dataset, dset, steps)

        f = plt.figure()
        cm = confusion_matrix(y_true, y_pred)
        title = 'Confusion matrix: ' + dset.upper()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        ax = f.gca()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.colorbar()

        if not class_names:
            class_names = np.arange(len(np.unique(y_true)))
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylim(-0.5, tick_marks[-1]+0.5)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        return f


class LFCNN(Model):
    """LF-CNN. Includes basic parameter interpretation options.

    For details see [1].

    Parameters
    ----------
    n_ls : int
        Number of latent components.
        Defaults to 32.

    nonlin : callable
        Activation function of the temporal Convolution layer.
        Defaults to tf.nn.relu

    filter_length : int
        Length of spatio-temporal kernels in the temporal
        convolution layer. Defaults to 7.

    pooling : int
        Pooling factor of the max pooling layer. Defaults to 2

    pool_type : str {'avg', 'max'}
        Type of pooling operation. Defaults to 'max'.

    padding : str {'SAME', 'FULL', 'VALID'}
        Convolution padding. Defaults to 'SAME'.

    stride : int
        Stride of the max pooling layer. Defaults to 1.


    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """

    def __init__(self, specs, dataset, name='lf-cnn', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters.
        """
        super(LFCNN, self).__init__(specs, dataset, name=name, **kwargs)
        l1 = specs.setdefault('l1', 0.0)
        l2 = specs.setdefault('l2', 0.0)

        self.demix = DeMixing(n_ls=specs.setdefault('n_ls', 32),
                              axis=specs.setdefault('axis', 1),
                              kernel_regularizer=l1_l2(l1=l1, l2=l2),
                              bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.tconv1 = LFTConv(scope="conv",
                              n_ls=specs.setdefault('n_ls', 32),
                              nonlin=specs.setdefault('nonlin', tf.nn.relu),
                              filter_length=specs.setdefault('filter_length', 7),
                              stride=specs.setdefault('stride', 1),
                              pooling=specs.setdefault('pooling', 2),
                              pool_type=specs.setdefault('pool_type', 'max'),
                              padding=specs.setdefault('padding', 'SAME'),
                              kernel_regularizer=l1_l2(l1=l1, l2=l2),
                              bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.fin_fc = Dense(size=self.out_dim,
                            nonlin=tf.identity,
                            dropout=self.rate,
                            kernel_regularizer=l1_l2(l1=l1, l2=l2),
                            bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.specs = specs.copy()  # specs was updated with default values


    def call(self, X):
        # Ensure the tensor is 4D
        x0 = self.make_4D(X)

        dmx = self.demix(x0)
        tconv = self.tconv1(dmx)
        y_ = self.fin_fc(tconv)

        return self.check_yshape(y_)

    def compute_patterns(self, data_path, output='patterns'):
        """Computes spatial patterns from filter weights.
        Required for visualization.

        Parameters
        ----------
        data_path : str or list of str
            Path to TFRecord files on which the patterns are estimated.

        output : str {'patterns, 'filters', 'full_patterns'}
            String specifying the output.

            'filters' - extracts weights of the spatial filters

            'patterns' - extracts activation patterns, obtained by
            left-multipying the spatial filter weights by the (spatial)
            data covariance.

            'full-patterns' - additionally multiplies activation
            patterns by the precision (inverse covariance) of the
            latent sources

        Returns
        -------
        self.patterns
            spatial filters or activation patterns, depending on the
            value of 'output' parameter.

        self.lat_tcs
            time courses of latent sourses.

        self.filters
            temporal convolutional filter coefficients.

        self.out_weights
            weights of the output layer.

        self.rfocs
            feature relevances for the output layer.
            (See self.get_output_correlations)

        Raises:
        -------
            AttributeError: If `data_path` is not specified.
        """
        vis_dict = None

        if isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
            vis_dict = self.dataset._build_dataset(data_path, n_batch=None)
        elif isinstance(data_path, Dataset):
            if hasattr(data_path, 'test'):
                vis_dict = data_path.test
            else:
                vis_dict = data_path.val
        elif isinstance(data_path, tf.data.Dataset):
            vis_dict = data_path
        else:
            raise AttributeError('Specify dataset or data path.')

        X, y = [row for row in vis_dict.take(1)][0]
        X = self.make_4D(X)
        X, y = X.numpy(), y.numpy()

        # Spatial stuff
        demx = self.demix.W.numpy()
        data = np.squeeze(X.transpose([1, 2, 3, 0]))
        data = data.reshape([data.shape[0], -1], order='F')

        self.dcov, _ = ledoit_wolf(data.T)
        self.lat_tcs = np.dot(demx.T, data)
        del data

        if 'patterns' in output:
            self.patterns = np.dot(self.dcov, demx)
            if 'full' in output:
                self.lat_cov = ledoit_wolf(self.lat_tcs)
                self.lat_prec = np.linalg.inv(self.lat_cov)
                self.patterns = np.dot(self.patterns, self.lat_prec)
        else:
            self.patterns = demx

        kern = self.tconv1.filters.numpy()
        tc_out = self.tconv1(self.demix(X)).numpy()
        out_w = self.fin_fc.w.numpy()

        print('out_w:', out_w.shape)

        #  Temporal conv stuff
        self.filters = np.squeeze(kern)
        self.tc_out = np.squeeze(tc_out)
        self.out_weights = np.reshape(out_w, [-1, self.demix.size,
                                              self.out_dim])

        print('demx:', demx.shape,
              'kern:', self.filters.shape,
              'tc_out:', self.tc_out.shape,
              'out_w:', self.out_weights.shape)

        self.get_output_correlations(y)

        self.out_biases = self.fin_fc.b.numpy()

    def get_output_correlations(self, y_true):
        """Computes a similarity metric between each of the extracted
        features and the target variable.

        The metric is a Manhattan distance for dicrete targets, and
        Spearman correlation for continuous targets.
        """
        self.rfocs = []

        flat_feats = self.tc_out.reshape(self.tc_out.shape[0], -1)

        if self.target_type == 'float':
            for y_ in y_true.T:
                rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])
                self.rfocs.append(rfocs.reshape(self.out_weights.shape[:-1]))

        elif self.target_type == 'int':
            y_true = y_true/np.linalg.norm(y_true, ord=1, axis=0)[None, :]
            flat_div = np.linalg.norm(flat_feats, 1, axis=0)[None, :]
            flat_feats = flat_feats/flat_div

            for y_ in y_true.T:
                rfocs = 2. - np.sum(np.abs(flat_feats - y_[:, None]), 0)
                self.rfocs.append(rfocs.reshape(self.out_weights.shape[:-1]))

        self.rfocs = np.dstack(self.rfocs)

        if np.any(np.isnan(self.rfocs)):
            self.rfocs[np.isnan(self.rfocs)] = 0

    # --- LFCNN plot functions ---
    def plot_out_weights(self, pat=None, t=None, tmin=-0.1, sorting='weight'):
        """Plots the weights of the output layer.

        Parameters
        ----------

        pat : int [0, self.specs['n_ls'])
            Index of the latent component to higlight

        t : int [0, self.h_params['n_t'])
            Index of timepoint to highlight

        """
        if not hasattr(self, 'out_weights'):
            self.compute_patterns(self.dataset)
        vmin = np.min(self.out_weights)
        vmax = np.max(self.out_weights)

        f, ax = plt.subplots(1, self.out_dim)
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        for ii in range(len(ax)):
            if 'weight' in sorting:
                F = self.out_weights[..., ii].T
            elif 'spear' in sorting:
                F = self.rfocs[..., ii].T
            else:
                F = self.rfocs[..., ii].T * self.out_weights[..., ii].T

            tstep = self.specs['stride']/float(self.fs)
            times = tmin+tstep*np.arange(F.shape[-1])

            im = ax[ii].pcolor(times, np.arange(self.specs['n_ls'] + 1), F,
                               cmap='bone_r', vmin=vmin, vmax=vmax)

            r = []
            if np.any(pat) and np.any(t):
                r = [ptch.Rectangle((times[tt], pp), width=tstep,
                                    height=1, angle=0.0)
                     for pp, tt in zip(pat[ii], t[ii])]

                pc = collections.PatchCollection(r, facecolor='red', alpha=.5,
                                                 edgecolor='red')
                ax[ii].add_collection(pc)

        f.colorbar(im, ax=ax[-1])
        plt.show()

    def plot_waveforms(self, tmin=0):
        """Plots timecourses of latent components.

        Parameters
        ----------
        tmin : float
            Beginning of the MEG epoch with regard to reference event.
            Defaults to 0.
        """
        if not hasattr(self, 'lat_tcs'):
            self.compute_patterns(self.dataset)

        if not hasattr(self, 'uorder'):
            order, _ = self._sorting()
            uorder = uniquify(order.ravel())
            self.uorder = uorder

        f, ax = plt.subplots(2, 2)

        nt = self.dataset.h_params['n_t']
        self.waveforms = np.squeeze(
                self.lat_tcs.reshape([self.specs['n_ls'], -1, nt]).mean(1))

        tstep = 1/float(self.fs)
        times = tmin + tstep*np.arange(nt)
        [ax[0, 0].plot(times, wf + 1e-1*i)
         for i, wf in enumerate(self.waveforms) if i not in self.uorder]

        ax[0, 0].plot(times,
                      self.waveforms[self.uorder[0]] + 1e-1*self.uorder[0],
                      'k.')
        ax[0, 0].set_title('Latent component waveforms')

        bias = self.sess.run(self.tconv1.b)[self.uorder[0]]
        ax[0, 1].stem(self.filters.T[self.uorder[0]], use_line_collection=True)
        ax[0, 1].hlines(bias, 0, len(self.filters.T[self.uorder[0]]),
                        linestyle='--', label='Bias')
        ax[0, 1].legend()
        ax[0, 1].set_title('Filter coefficients')

        conv = np.convolve(self.filters.T[self.uorder[0]],
                           self.waveforms[self.uorder[0]], mode='same')
        vmin = conv.min()
        vmax = conv.max()
        ax[1, 0].plot(times + 0.5*self.specs['filter_length']/float(self.fs),
                      conv)
        ax[1, 0].hlines(bias, times[0], times[-1], linestyle='--', color='k')

        tstep = float(self.specs['stride'])/self.fs
        strides = np.arange(times[0], times[-1] + tstep/2, tstep)[1:-1]
        pool_bins = np.arange(times[0],
                              times[-1] + tstep,
                              self.specs['pooling']/self.fs)[1:]

        ax[1, 0].vlines(strides, vmin, vmax,
                        linestyle='--', color='c', label='Strides')
        ax[1, 0].vlines(pool_bins, vmin, vmax,
                        linestyle='--', color='m', label='Pooling')
        ax[1, 0].set_xlim(times[0], times[-1])
        ax[1, 0].legend()
        ax[1, 0].set_title('Convolution output')

        if self.out_weights.shape[-1] == 1:
            ax[1, 1].pcolor(self.F)
            ax[1, 1].hlines(self.uorder[0] + .5, 0, self.F.shape[1], color='r')
        else:
            ax[1, 1].plot(self.out_weights[:, self.uorder[0], :], 'k*')

        ax[1, 1].set_title('Feature relevance map')

    def _sorting(self, sorting='best'):
        """Specify which components to plot.

        Parameters
        ----------
        sorting : str
            Sorting heuristics.

            'l2' - plots all components sorted by l2 norm of their
            spatial filters in descending order.

            'weight' - plots a single component that has a maximum
            weight for each class in the output layer.

            'spear' - plots a single component, which produces a
            feature in the output layer that has maximum correlation
            with each target variable.

            'best' - plots a single component, has maximum relevance
            value defined as output_layer_weught*correlation.

            'best_spatial' - same as 'best', but the components
            relevances are defined as the sum of all relevance scores
            over all timepoints.

        """
        order = []
        ts = []

        if sorting == 'l2':
            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=2))
            self.F = self.out_weights[..., 0].T
            ts = None

        elif sorting == 'best_spatial':
            for i in range(self.out_dim):
                self.F = self.out_weights[..., i].T * self.rfocs[..., i].T
                pat = np.argmax(self.F.sum(-1))
                order.append(np.tile(pat, self.F.shape[1]))
                ts.append(np.arange(self.F.shape[-1]))

        elif sorting == 'best':
            for i in range(self.out_dim):
                self.F = np.abs(self.out_weights[..., i].T
                                * self.rfocs[..., i].T)
                pat, t = np.where(self.F == np.max(self.F))
                print('Maximum spearman r * weight:', np.max(self.F))
                order.append(pat)
                ts.append(t)

        elif sorting == 'weight':
            for i in range(self.out_dim):
                self.F = self.out_weights[..., i].T
                pat, t = np.where(self.F == np.max(self.F))
                print('Maximum weight:', np.max(self.F))
                order.append(pat)
                ts.append(t)

        elif sorting == 'spear':
            for i in range(self.out_dim):
                self.F = self.rfocs[..., i].T
                print('Maximum r_spear:', np.max(self.F))
                pat, t = np.where(self.F == np.max(self.F))
                order.append(pat)
                ts.append(t)

        elif isinstance(sorting, int):
            for i in range(self.out_dim):
                self.F = self.out_weights[..., i].T * self.rfocs[..., i].T
                pat, t = np.where(self.F >= np.percentile(self.F, sorting))
                order.append(pat)
                ts.append(t)

        else:
            print('ELSE!')
            order = np.arange(self.specs['n_ls'])
            self.F = self.out_weights[..., 0].T

        order = np.array(order)
        ts = np.array(ts)
        return order, ts

    def plot_patterns(self, sensor_layout=None, sorting='l2', percentile=90,
                      spectra=False, scale=False, names=False):
        """Plot informative spatial activations patterns for each class
        of stimuli.

        Parameters
        ----------

        sensor_layout : str or mne.channels.Layout
            Sensor layout. See mne.channels.read_layout for details

        sorting : str, optional
            Component sorting heuristics. Defaults to 'l2'.
            See model._sorting

        spectra : bool, optional
            If True will also plot frequency responses of the associated
            temporal filters. Defaults to False.

        fs : float
            Sampling frequency.

        scale : bool, otional
            If True will min-max scale the output. Defaults to False.

        names : list of str, optional
            Class names.

        Returns
        -------

        Figure

        """
        if sensor_layout:
            lo = channels.read_layout(sensor_layout)
            info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
            self.fake_evoked = evoked.EvokedArray(self.patterns, info)

        order, ts = self._sorting(sorting)
        uorder = uniquify(order.ravel())
        self.uorder = uorder
        l_u = len(uorder)

        if sensor_layout:
            self.fake_evoked.data[:, :l_u] = self.fake_evoked.data[:, uorder]
            self.fake_evoked.crop(tmax=float(l_u))
            if scale:
                _std = self.fake_evoked.data[:, :l_u].std(0)
                self.fake_evoked.data[:, :l_u] /= _std

        nfilt = max(self.out_dim, 8)
        nrows = max(1, l_u//nfilt)
        ncols = min(nfilt, l_u)

        f, ax = plt.subplots(nrows, ncols, sharey=True)
        f.set_size_inches([16, 9])
        ax = np.atleast_2d(ax)

        for ii in range(nrows):
            fake_times = np.arange(ii * ncols,  (ii + 1) * ncols, 1.)
            vmax = np.percentile(self.fake_evoked.data[:, :l_u], 95)
            self.fake_evoked.plot_topomap(times=fake_times,
                                          axes=ax[ii],
                                          colorbar=False,
                                          vmax=vmax,
                                          scalings=1,
                                          time_format='output # %g',
                                          title='Patterns ('+str(sorting)+')')
        if np.any(ts):
            self.plot_out_weights(pat=order, t=ts, sorting=sorting)
        else:
            self.plot_out_weights()

    def plot_spectra(self, fs=None, sorting='l2', norm_spectra=None,
                     log=False):
        """Plots frequency responses of the temporal convolution filters.

        Parameters
        ----------
        fs : float
            Sampling frequency.

        sorting : str optinal
            Component sorting heuristics. Defaults to 'l2'.
            See model._sorting

        norm_sepctra : None, str {'welch', 'ar'}
            Whether to apply normalization for extracted spectra.
            Defaults to None.

        log : bool
            Apply log-transform to the spectra.

        """
        if fs is not None:
            self.fs = fs
        elif self.dataset.h_params['fs']:
            self.fs = self.dataset.h_params['fs']
        else:
            warnings.warn('Sampling frequency not specified, setting to 1.',
                          UserWarning)
            self.fs = 1.

        if norm_spectra:
            if norm_spectra == 'welch':
                fr, psd = welch(self.lat_tcs, fs=self.fs, nperseg=256)
                self.d_psds = psd[:, :-1]

            elif 'ar' in norm_spectra and not hasattr(self, 'ar'):
                ar = []
                for i, ltc in enumerate(self.lat_tcs):
                    coef, _, _ = aryule(ltc, self.specs['filter_length'])
                    ar.append(coef[None, :])
                self.ar = np.concatenate(ar)

        order, ts = self._sorting(sorting)
        uorder = uniquify(order.ravel())
        self.uorder = uorder
        out_filters = self.filters[:, uorder]
        l_u = len(uorder)

        nfilt = max(self.out_dim, 8)
        nrows = max(1, l_u//nfilt)
        ncols = min(nfilt, l_u)

        f, ax = plt.subplots(nrows, ncols, sharey=True)
        f.set_size_inches([16, 9])
        ax = np.atleast_2d(ax)

        for i in range(nrows):
            for jj, flt in enumerate(out_filters[:, i*ncols:(i+1)*ncols].T):
                if norm_spectra == 'ar':
                    # TODO! Gabi: Is this a redundant case?
                    # the plot functionality is commented out making it
                    # equivalent to the else case.
                    # Otherwise it is almost the same as plot_ar.

                    w, h = freqz(flt, 1, worN=128)
                    # w, h0 = freqz(1, self.ar[jj], worN=128)
                    # ax[i, jj].plot(w/np.pi*self.fs/2,h0.T,label='Flt input')
                    # h = h*h0

                elif norm_spectra == 'welch':
                    w, h = freqz(flt, 1, worN=128)
                    fr1 = w/np.pi*self.fs/2
                    h0 = self.d_psds[uorder[jj], :]*np.abs(h)
                    if log:
                        ax[i, jj].semilogy(fr1, self.d_psds[uorder[jj], :],
                                           label='Filter input')
                        ax[i, jj].semilogy(fr1, np.abs(h0),
                                           label='Fitler output')
                    else:
                        ax[i, jj].plot(fr1, self.d_psds[uorder[jj], :],
                                       label='Filter input')
                        ax[i, jj].plot(fr1, np.abs(h0), label='Fitler output')
                    print(np.all(np.round(fr[:-1], -4) == np.round(fr1, -4)))

                elif norm_spectra == 'plot_ar':
                    w0, h0 = freqz(flt, 1, worN=128)
                    w, h = freqz(self.ar[jj], 1, worN=128)
                    ax[i, jj].plot(w/np.pi*self.fs/2, np.abs(h0))
                    print(h0.shape, h.shape, w.shape)

                else:
                    w, h = freqz(flt, 1, worN=128)

                if log:
                    ax[i, jj].semilogy(w/np.pi*self.fs/2, np.abs(h),
                                       label='Freq response')
                else:
                    ax[i, jj].plot(w/np.pi*self.fs/2, np.abs(h),
                                   label='Freq response')
                ax[i, jj].legend()
                ax[i, jj].set_xlim(0, 125.)
        return f


class VARCNN(Model):
    """VAR-CNN

    For details see [1].

    Parameters
    ----------
    n_ls : int
        number of latent components
        Defaults to 32

    filter_length : int
        length of spatio-temporal kernels in the temporal
        convolution layer. Defaults to 7

    stride : int
        stride of the max pooling layer. Defaults to 1

    pooling : int
        pooling factor of the max pooling layer. Defaults to 2

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
        """
    def __init__(self, specs, dataset, name='var-cnn', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters.
        """
        super(VARCNN, self).__init__(specs, dataset, name=name, **kwargs)
        l1 = specs['l1']
        l2 = specs['l2']

        self.demix = DeMixing(n_ls=specs.setdefault('n_ls', 32),
                              axis=specs.setdefault('axis', 1),
                              kernel_regularizer=l1_l2(l1=l1, l2=l2),
                              bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.tconv1 = VARConv(scope="conv",
                              n_ls=specs.setdefault('n_ls', 32),
                              nonlin=tf.nn.relu,
                              filter_length=specs.setdefault('filter_length', 7),
                              stride=specs.setdefault('stride', 1),
                              pooling=specs.setdefault('pooling', 2),
                              padding=specs.setdefault('padding', 'SAME'),
                              kernel_regularizer=l1_l2(l1=l1, l2=l2),
                              bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.fin_fc = Dense(size=self.out_dim,
                            nonlin=tf.identity,
                            dropout=self.rate,
                            kernel_regularizer=l1_l2(l1=l1, l2=l2),
                            bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.specs = specs.copy()  # specs was updated with default values

    def call(self, X):
        # Ensure the tensor is 4D
        x0 = self.make_4D(X)

        dmx = self.demix(x0)
        tconv = self.tconv1(dmx)
        y_ = self.fin_fc(tconv)

        return self.check_yshape(y_)


class LFLSTM(LFCNN):
    """LF-CNN-LSTM

    Parameters
    ----------
    n_ls : int
        number of latent components
        Defaults to 32

    filter_length : int
        length of spatio-temporal kernels in the temporal
        convolution layer. Defaults to 7

    stride : int
        stride of the max pooling layer. Defaults to 1

    pooling : int
        pooling factor of the max pooling layer. Defaults to 2

    rnn_units : int
        number of nodes in the LSTM layer

    rnn_dropout : float
        dropout for the LSTM layer

    rnn_nonlin : tf.activation
        activation function for the LSTM layer

    rnn_forget_bias : bool
        whether units forget bias

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
        """
    def __init__(self, specs, dataset, name='lfcnn-lstm', **kwargs):
        """
        Parameters
        -----------
        specs : dict
                dictionary of model-specific hyperparameters.

        """
        super(LFLSTM, self).__init__(specs, dataset, name=name, **kwargs)
        l1 = specs['l1']
        l2 = specs['l2']

        self.demix = DeMixing(n_ls=specs.setdefault('n_ls', 32),
                              axis=specs.setdefault('axis', 1),
                              kernel_regularizer=l1_l2(l1=l1, l2=l2),
                              bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.tconv1 = LFTConv(scope="conv",
                              n_ls=specs.setdefault('n_ls', 32),
                              nonlin=tf.nn.relu,
                              filter_length=specs.setdefault('filter_length', 7),
                              stride=specs.setdefault('stride', 1),
                              pooling=specs.setdefault('pooling', 2),
                              padding=specs.setdefault('padding', 'SAME'),
                              kernel_regularizer=l1_l2(l1=l1, l2=l2),
                              bias_regularizer=l1_l2(l1=l1, l2=l2))

        self.lstm = LSTMv1(scope="lstm",
                           size=specs.setdefault('n_ls', 32),
                           dropout=specs.setdefault('rnn_dropout', 0.0),
                           nonlin=specs.setdefault('rnn_nonlin', 'tanh'),
                           unit_forget_bias=specs.setdefault('rnn_forget_bias', True),
                           kernel_regularizer=l1_l2(l1=l1, l2=l2),
                           bias_regularizer=l1_l2(l1=l1, l2=l2),
                           return_sequences=specs.setdefault('rnn_seq', True),
                           unroll=specs.setdefault('unroll', False))

        self.fin_fc = Dense(scope='fc',
                            size=self.out_dim,
                            nonlin=tf.identity,
                            dropout=specs.setdefault('dropout', 0.0))

        self.specs = specs.copy()  # specs was updated with default values

    def call(self, X):
        # Ensure the tensor is 4D
        x0 = self.make_4D(X)                # [k, n_ch, time, n_seq]

        _, n_ch, ntime, n_seq = x0.shape

        print('input x0', x0.shape)
        dmx = self.demix(x0)                 # [k, n_seq, time, 1, demix.size]
        print('demix dmx', dmx.shape)
        dmx = tf.reshape(dmx, [-1, self.dataset.h_params['n_t'],
                               self.specs['n_ls']], name='dmx_res')
        dmx = tf.expand_dims(dmx, -1)
        print('dmx-sqout:', dmx.shape)

        features = self.tconv1(dmx)            # [n_seq, maxpool, tconv1.size]
        print('features', features.shape)
        fshape = tf.multiply(features.shape[1], features.shape[2])
        if 'n_seq' in self.dataset.h_params.keys():
            features = tf.reshape(
                    features, [-1, self.dataset.h_params['n_seq'], fshape],
                    name='conv_res')
        else:
            features = tf.reshape(features, [-1, 1, fshape], name='conv_res')
        print('flat features:', features.shape)  # [n_seq, maxpool*tconv1.size]

        lstm_out = self.lstm(features)            # [k, n_seq, n_ls]
        print('lstm_out', lstm_out.shape)
        y_ = self.fin_fc(lstm_out)            # [k, n_seq, y_shape]
        # print('fc y_', y_.shape)

        return self.check_yshape(y_)


def test_model():
    # %%
    from mneflow import Dataset
    from presets import preset_data, get_subset, model_parameters
    from keras_utils import plot_history, report_results
    tf.compat.v1.enable_eager_execution()

    preset = 'squid'
    dropbad = True
    cont = False

    # Load data
    meta, event_names = preset_data(preset, dropbad=dropbad, cont=cont)

    # Initialise dataset
    subset = None
    subset_names = get_subset(meta, subset=subset)

    n_batch = 250  # batch size

    # batch the train dataset
    dataset = Dataset(meta, train_batch=n_batch, class_subset=subset,
                      pick_channels=None, decim=None)

    # load test dataset without batching
    test_data = dataset._build_dataset(meta['test_paths'], n_batch=None)

    # Initialise model parameters
    graph_specs, _, _ = model_parameters(preset, meta['savepath'])

    opt = tf.keras.optimizers.Adam()
    loss = 'sparse_categorical_crossentropy'
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # build model
    model = VARCNN(graph_specs)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    _ = model.fit(dataset.train, validation_data=dataset.val,
                  callbacks=[callback], epochs=1000,
                  steps_per_epoch=n_batch, validation_steps=1)

    _ = model.evaluate(test_data, steps=1, verbose=1)

    plot_history(model.history)

    for steps, dset in zip(['train', 'val', 'test'], [n_batch, 1, 1]):
        report_results(model, dataset, dset, steps, subset_names)


# %%
if __name__ == '__main__':
    print('Reloaded')
