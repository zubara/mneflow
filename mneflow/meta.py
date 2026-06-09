# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:19:08 2025

@author: ipzub
"""
import os
import mne
import numpy as np
from scipy.signal import freqz
from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne import find_events, fit_dipole
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif
import pickle

class MetaData():
    """
    Class containing all metadata required to run model training, prediction,
    and evaluation. Produced by mneflow.produce_tfrecords, saved to "path" and
    can be restored if you need to rerun training on existing tfrecords.

    See mneflow.utils.load_meta() docstring.

    Attributes
    ----------
    path : str
        A path where the output TFRecord (path + /tfrecrods/) files,
        models (path + /models/), and the corresponding metadata
        file (path + data_id + meta.pkl) will be stored.

    data_id : str
        Filename prefix for the output files and the metadata file.

    input_type : str {'trials', 'continuous', 'seq', 'fconn'}
        Type of input data.

        'trials' - treats each of n inputs as an iid sample, produces dataset
        with dimensions (n, 1, t, ch)

        'seq' - treats each of n inputs as a seqence of shorter segments,
        produces dataset with dimensions (n, seq_length, segment, ch)

        'continuous' - treats inputs as a single continuous sequence,
        produces dataset with dimensions
        (n*(t-segment)//aug_stride, 1, segment, ch)

    target_type : str {'int', 'float'}
        Type of target variable.

        'int' - for classification,
        'float' - for regression problems.
        'signal' - regression or classification a continuous (possbily multichannel)
        data. Requires "transform_targets" function to be applied to target variables.

    n_folds : int, optional
        Number of folds to split the data for training/validation/testing.
        One fold of the n_folds is used as a validation set.
        If test_set == 'holdout' generates one extra fold
        used as test set. Defaults to 5

    test_set : str {'holdout', 'loso', None}, optional
        Defines if a separate holdout test set is required.
        'holdout' saves 50% of the validation set
        'loso' produces an addtional trfrecord so that each input file can
        be used as a test test in leave-one-subject-out cross-validation.
        None does not produce a separate test set.
        Defaults to None.

    fs : float, optional
         Sampling frequency, required only if inputs are not mne.Epochs

    Notes
    -----
    See mneflow.produce_tfrecords and mneflow.utils.preprocess for more
    details and availble options.
    """

    def __init__(self):
        self.data = dict()
        self.preprocessing = dict()
        self.model_specs = dict()
        self.train_params = dict()
        self.patterns = dict()
        self.results = dict()
        self.weights = dict()


    def copy(self):

        return

    def save(self, verbose=True):
        """Saves the metadata to self.data['path'] + self.data['data_id']"""
        if 'path' in self.data.keys() and 'data_id' in self.data.keys():
            fname = self.data['data_id'] + '_meta.pkl'
            with open(os.path.join(self.data['path'], fname), 'wb') as f:
                pickle.dump(self, f)
            if verbose:
                print("""Saving MetaData as {} \n
                      to {}""".format(fname, self.data['path']))
        else:
            print("""Cannot save MetaData! \n
                  Please specify meta.data['path'] and meta.data['data_id'']""")


    def restore_model(self, load_encoder=False):
        """Restored previously saved model from metadata."""
        from mneflow import models, lfcnn
        if self.model_specs['scope'] == 'lfcnn':
            model = lfcnn.LFCNN(meta=self)
        elif self.model_specs['scope'] == 'varcnn':
            model = models.VARCNN(meta=self)
        elif self.model_specs['scope'] == 'fbcsp-ShallowNet':
            model = models.FBCSP_ShallowNet(meta=self)
        elif self.model_specs['scope'] == 'deep4':
            model = models.Deep4(meta=self)
        elif self.model_specs['scope'] == 'eegnet8':
            model = models.EEGNet(meta=self)

        model.build()
        model.model_name = "_".join([self.model_specs['scope'],
                               self.data['data_id'] + '.h5'])
        model.km.load_weights(os.path.join(self.model_specs['model_path'],
                                           model.model_name))
        if load_encoder and os.path.exists(os.path.join(self.model_specs['model_path'],
                                           model.model_name[:-3] + 'encoder_.h5')):
          print("Loading encoder")
          model.build_encoder()
          model.km_enc.load_weights(os.path.join(self.model_specs['model_path'],
                                                 model.model_name[:-3] + 'encoder_.h5'))


        #model.km = tf.keras.models.load_model(os.path.join(self.model_specs['model_path'],
        #                                      model_name))


        #model.build()
        model.cv_patterns = self.patterns
        #TODO: set weights from self.km.weights
        #TODO: set val loss, patterns, etc, specs
        return model

    def update(self, data=None, preprocessing=None, train_params=None,
               model_specs=None, patterns=None, results=None, weights=None):
        """Updates metadata file"""
        if isinstance(data, dict):
            self.data.update(data)
            print("Updating: meta.data")
        if isinstance(preprocessing, dict):
            self.preprocessing.update(preprocessing)
            print("Updating: meta.preprocessing")
        if isinstance(train_params, dict):
             self.train_params.update(train_params)
             print("Updating: meta.train_params")
        if isinstance(model_specs, dict):
             self.model_specs.update(model_specs)
             print("Updating: meta.model_specs")
        if isinstance(patterns, dict):
            self.patterns.update(patterns)
            print("Updating: meta.patterns")
        if isinstance(results, dict):
            self.results.update(results)
            print("Updating: meta.results")
        if isinstance(weights, dict):
            self.weights.update(weights)
            print("Updating: meta.weights")
        self.save(verbose=False)
        return

    def make_fake_evoked(self, topos, sensor_layout, ch_type='mag',
                         channel_subset=None):
        """
        Creates an MNE Fake Evoked object from topographies and a sensor layout
        for plotting.

            Parameters
            ----------
            topos : ndarray, shape (n_channels, n_components) or (n_channels, n_classes)
                Topography pattern to be converted to Evoked object.

            sensor_layout : str or mne.channels.layout.Layout
                Sensor layout to create the topoplot.

            ch_type : str, optional
                Channel type ('mag', 'grad', or 'eeg'). Defaults to 'mag'.

            channel_subset : array, optional
                Array of indices (int) of the channels to pick. Defaults to None.
                If None, all channels are used.

            Returns
            -------
            fake_evoked : mne.Evoked
                Evoked object with topography.
        """


        if isinstance(sensor_layout, str):
            layout = mne.channels.read_layout(sensor_layout)

        elif isinstance(sensor_layout, mne.channels.layout.Layout) and ch_type in ['mag', 'grad', 'eeg']:
            layout = sensor_layout

        else:
            raise ValueError("""Unknown sensor layout. Should be an instance of
                             mne.channels.layout.Layout or a sting""")
            #return



        if np.any(channel_subset):
            ts = topos.copy()[channel_subset, :]
        else:
            channel_subset = np.arange(0,  topos.shape[0], 1, dtype=int)

        lo = layout.copy().pick(channel_subset)
        info = mne.create_info(lo.names, 1., ch_type)

        #lo = channels.generate_2d_layout(lo.pos)
        #TODO: use mne.channels.find_layout

        orig_xy = np.mean(lo.pos[:, :2], 0)
        for i, ch in enumerate(lo.names):
            if info['chs'][i]['ch_name'] == ch:
                info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/4.5
                #info['chs'][i]['loc'][4:] = 0
            else:
                print("Channel name mismatch. info: {} vs lo: {}".format(
                    info['chs'][i]['ch_name'], ch))

        fake_evoked = mne.evoked.EvokedArray(ts, info)
        return fake_evoked

    def get_feature_relevances(self, sorting='output_corr',
                               integrate=['timepoints'],
                               fold=0, diff=True):
        """
        Returns the map of feature relevances for each interpretation method

        Returns
        -------

        F : np.array
            (n_t_pooled, n_latent, n_classes, n_folds)

        """
        # get feature relevance
        if sorting == 'combined':
            F = (self.patterns['weight']['feature_relevance']
                 * self.patterns['ccms']['pooled']
                 #* np.sign(self.patterns['output_corr']['feature_relevance'])
                 ) - self.weights['tconv_b'][np.newaxis, :, np.newaxis, :]
        else:
            F = self.patterns[sorting]['feature_relevance']

        #
        n_t, n_components, n_y, n_folds = F.shape
        #print(F.shape)

        Fd = []
        if diff == True and sorting != 'combined':

            y_cov = self.patterns['ccms']['cov_y']
            print(F.shape, y_cov.shape)
            for i in range(F.shape[3]):
                Fd.append(np.dot(F[:, :, :, i], y_cov[:, :, i]))
            F = np.stack(Fd, axis=3)
            # for i in range(F.shape[2]):
            #     F_other = np.delete(F, i, axis=2).mean(2)
            #     Fd.append(F[:, :, i, :] - F_other)
            # F = np.stack(Fd, axis=2)
            # # #F = np.matmul()
            F = np.maximum(F, 0)

        names = ['Time Points', 'Components', 'Classes', 'Folds']

        if 'folds' in integrate:
            F = F.mean(3, keepdims=True)
            names.pop(names.index('Folds'))
        if 'timepoints' in integrate:
            F = F.max(0, keepdims=True)
            names.pop(names.index('Time Points'))
        if 'vars' in integrate:
            F = F.sum(2, keepdims=True)
            names.pop(names.index('Classes'))
        if 'components' in integrate:
            F = F.sum(1, keepdims=True)
            names.pop(names.index('Components'))

        F = np.squeeze(F)
        if n_y == 1:
            #Put back the 'label' dimension
            #print(F.shape)
            F = np.expand_dims(F, 1 )#[:, np.newaxis, :]
        if F.ndim == 2:
            F = np.expand_dims(F, -1 )#[:, np.newaxis, :]



        print("Relevances: ", names, F.shape)
        return F, names

    #def explore_patterns():


    def explore_components(self, sorting='output_corr',
                           info=None,
                           sensor_layout='Vectorview-grad',
                           class_names=None, diff=True,
                           n_cols=1,
                           channel_subset=None):
        """
        Ineractive plot of feature relevances for each fold/subplot
        max-pooled over all timepoints.

        Clicking each non-zero square returns a new figure with:
            -Spatial pattern of the selected single component
            -Frequency response of the temporal convolution filter
            -Temporal convolution filter coeffs
            -Relevance of the same component for other classes

        Parameters
        ----------

        pat : int [0, self.specs['n_latent'])
            Index of the latent component to higlight

        t : int [0, self.h_params['n_t'])
            Index of timepoint to highlight

        Returns
        -------
        figure :
            Imshow [n_latent, y_shape]

        """
        #TODO: joint colorbar
        def _onclick_component(event):

            x_ind = np.maximum(np.round(event.xdata, 0).astype(int), 0)
            y_ind = np.maximum(np.round(event.ydata).astype(int), 0)
            fold_ind = int(event.inaxes.get_title().split(' ')[-1])

            f1, ax = plt.subplots(2,2, tight_layout=True)
            f1.suptitle("{}: {}  {}: {}  {}:{}"
                  .format(names[0][:-1], x_ind, names[1][:-1], y_ind,
                          "Fold", fold_ind))

            topo = topos[:, y_ind, fold_ind]
            #prec_yhat = np.linalg.inv(self.patterns['ccms']['cov_y_hat'][:, :, fold_ind])
            pattern = np.dot(dcov[:, :, y_ind], topo)


            self.fake_evoked_interactive.data[:, y_ind] = pattern[channel_subset]
            self.fake_evoked_interactive.plot_topomap(times=[y_ind],
                                                      axes=ax[0, 0],
                                                      colorbar=False,
                                                      time_format='Spatial activation pattern, [au]'
                                                      )

            flt = self.weights['tconv'][x_ind, :, fold_ind]

            flt -= flt.mean()
            w, h = freqz(flt, 1, worN=128, fs = self.data['fs'])
            freq_response = np.array(np.abs(h))
            freq_response = freq_response / np.sum(freq_response, 0, keepdims=True)

            ax[0,1].plot(self.patterns['freqs'], freq_response,
                        label='Freq_response')
            ax[0,1].set_xlim(1, 70.)
            ax[0,1].legend(frameon=False)
            ax[0,1].set_title('Relative power spectra')


            ax[1,0].stem(flt)
            ax[1,0].set_title('Temporal convolution kernel coefficients')
            ax[1,1].stem(np.arange(F.shape[1]), F[x_ind, :, fold_ind], 'k')
            ax[1,1].plot(y_ind, F[x_ind, y_ind, fold_ind], 'rs')
            ax[1,1].set_title('Contribution to each class')
            ax[1,1].set_xlabel(names[0])

        F, names = self.get_feature_relevances(sorting=sorting,
                                               integrate=['timepoints'],
                                               diff=diff) #(n_components, n_classes, n_folds)



        n_folds = len(self.data['folds'][0])
        if n_folds%n_cols != 0:
            n_rows = n_folds//n_cols + 1
        else:
            n_rows = n_folds//n_cols

        f, ax = plt.subplots(n_rows, n_cols)
        ax = ax.flatten()


        for jj in range(n_folds):
            ax[jj].set_title('Fold {}'.format(jj))

            axis = 0
            while F.ndim < 2:
                F = np.expand_dims(F, -1)
                axis = 0

            # if F.ndim != 3:
            #     print("""Integrate: {} returned shape {}. \n
            #           Can only plot 2 dimensions!""".format(integrate, F.shape))
            #     return

            # if 'timepoints' in integrate:
            #     #F = F.T
            #     trans = False
            #     #ax = 1 - ax
            #     print(names)
            # else:
            #     trans = False
            #     names = names[::-1]

            inds = np.argmax(F[:, :, jj], axis)


            topos = self.weights['dmx']

            if not np.any(channel_subset):
                channel_subset = np.arange(0,  topos.shape[0], 1, dtype=int)

            self.fake_evoked_interactive = self.make_fake_evoked(topos[:, :, 0], sensor_layout,
                                                                 channel_subset=channel_subset)

            dcov = self.patterns['dcov']['class_conditional'].mean(-1)
            #dcov = self.patterns['dcov']['input_spatial'].mean(-1)
            vmin = np.min(F)
            vmax = np.max(F)

            ax[jj].imshow(F[:, :, jj].T, cmap='bone_r', vmin=vmin, vmax=vmax)

            f.set_size_inches(16,4*n_folds)
            r = [ptch.Rectangle((ind - .5, i - .5), width=1,
                                height=1, angle=0.0, facecolor='none')
                 for i, ind in enumerate(inds)]

            pc = collections.PatchCollection(r, facecolor='red', alpha=.33,
                                              edgecolor='red')
            ax[jj].add_collection(pc)
            ax[jj].set_ylabel(names[1])
            ax[jj].set_xlabel(names[0])

        f.suptitle('Component relevance map: {} (Clickable)'.format(sorting))
        f.canvas.mpl_connect('button_press_event', _onclick_component)
        f.show()
        return f

    def explore_patterns():
        """
        Ineractive plot of combined activation patterns for each class/subplot
        averaged over all folds.

        Clicking each non-zero square returns a new figure with:
            -Spatial pattern across all folds


        """





    def plot_spatial_patterns(self, method='combined',
                              sensor_layout='Vectorview-mag',
                              class_subset=None,
                              channel_subset=None):
        """
        Plot any spatial distribution in the sensor space.
        TODO: Interpolation??


        Parameters
        ----------
        topos : np.array
            [n_ch, n_classes, ...]

        sensor_layout : TYPE, optional
            DESCRIPTION. The default is 'Vectorview-mag'.

        class_subset  : np.array, optional

        channel_subset  : np.array, optional

        diff : bool, True

        Returns
        -------
        None.

        """
        topos = self.get_spatial_patterns(method=method)

        if class_subset:
            topos = topos[:, class_subset, :]
        else:
            class_subset = np.arange(0,  topos.shape[1], 1)


        if not np.any(channel_subset):
            channel_subset = np.arange(0,  topos.shape[0], 1, dtype=int)

        _, n_y, n_folds = topos.shape

        if topos.ndim > 2:
            topos_plt = topos.mean(-1)

        topos_plt = topos_plt / np.maximum(topos_plt.std(0, keepdims=True), 1e-15)

        fake_evoked = self.make_fake_evoked(topos_plt, sensor_layout,
                                            channel_subset=channel_subset)

        ft = fake_evoked.plot_topomap(times=np.arange(n_y),
                                    colorbar=True,
                                    scalings=1,
                                    time_format="Class %g",
                                    outlines='head',
                                    #vlim= np.percentile(topos, [5, 95])
                                    )
        ft.set_size_inches(len(class_subset)*3, 3)

        def _show_folds(event):

            y_ind = int(event.inaxes.get_title().split(' ')[-1])
            print("Showing folds on class: {}".format(y_ind))
            stds = np.maximum(topos[:, y_ind, :].std(0), 1e-15)
            topos_f = topos[:, y_ind, :] / stds[None, :]
            _evoked = self.make_fake_evoked(topos_f, sensor_layout,
                                            channel_subset=channel_subset)
            _evoked.plot_topomap(times=np.arange(n_folds),
                                        #colorbar=True,
                                        #time_format='Class #{}'.format(class_subset[y_ind]),
                                        scalings=1,
                                        time_format="Fold %g",
                                        outlines='head',
                                        #vlim= np.percentile(topos, [5, 95])
                                        )



        #ft.show()
        #ft.suptitle('Component relevance map: {} (Clickable)'.format(sorting))
        ft.canvas.mpl_connect('button_press_event', _show_folds)
        ft.show()
        return ft

    # def _sorting(self, method='weight', n_comp=1, diff=False):
    #     """Return indices of signle components corresponsing to:
    #         'weight' - maximum positive weight in the output layer for each class
    #         'output_corr' - maximum correlation to the pre-softmax activation
    #                         for each class
    #         'compwise_loss' - maximum decrease of the validation loss after
    #                           setting wieghts of each compmonent to zero
    #                           for each class separately
    #         'weight_norm' - maximum norm of time resolved component activations
    #                         for each class


    #     Parameters
    #     ----------
    #     sorting : str
    #         Sorting heuristics.

    #         'weight' - maximum positive weight in the output layer for each class
    #         'output_corr' - maximum correlation to the pre-softmax activation
    #                         for each class
    #         'compwise_loss' - maximum decrease of the validation loss after
    #                           setting wieghts of each compmonent to zero
    #                           for each class separately
    #         'weight_norm' - maximum norm of time resolved component activations
    #                         for each class

    #     Returns:
    #     --------
    #     order : list of int
    #         indices of relevant components

    #     ts : list of int
    #         indices of relevant timepoints
    #     """
    #     order = []
    #     ts = []
    #     n_folds = self.data['n_folds']
    #     out_weights = self.weights['out_weights']
    #     F, names = self.get_feature_relevances(method=method,
    #                                            integrate=['timepoints'],
    #                                            diff=diff)

    #     if method in ['weight', 'compwise_loss', 'output_corr']:
    #         inds = np.argmax(F, 0)

    #     else:
    #         print("Sorting {:s} not implemented".format(method))
    #         return None


    #return inds

    def aligned_mean(self, topos):
        n_topos = topos.shape[1]
        topos_aligned = []
        for i in range(n_topos):
            t = topos[:, i, :]
            cc = np.sign(np.corrcoef(t.T)[0, :])
            if len(cc) % 2 == 1:
                majority = np.sign(np.sum(cc))
            else:
                majority = np.sign(np.sum(cc[1 :]) + 1)

            topos_aligned.append(majority * np.mean((t*cc[None, :]), -1))
        topos_aligned = np.stack(topos_aligned, 1)
        return topos_aligned

    def get_spatial_patterns(self, method='combined',
                             covariance_type='common',
                             use_y_cov=False,
                             diff=True,
                             random_weights=False):
        """


        Parameters
        ----------
        method : TYPE, optional
            Compute spatial activation patterns using a combination of spatial
            weights, spatial covariance, feature relevance and target covariance.
            The default is 'combined'. Other options are 'weight', 'output_corr',
            'compwise loss'.
        covariance_type : TYPE, optional
            DESCRIPTION. The default is 'common'. Other options are
            'class_conditional', and 'k-1'
        diff : bool, optional
            If True feature relevance is computed as
            F[class, ...] - np.mean(F[class!=class, ...]).
            The default is True.

        Returns
        -------
        patterns : np.array
                Spatial activation patterns. (n_channels, n_classes)

        """
        #TODO: implement covariance_type
        #TODO: add y_cov
        # if use_y_cov:
        #     diff = False

        F, names = self.get_feature_relevances(sorting=method,
                                        integrate=['timepoints'],
                                        diff=diff) #(n_components, n_classes, n_folds)
        n_components, n_y, n_folds = F.shape
        W = self.weights['dmx'] #(n_ch, n_components, n_folds)
        if random_weights:
            print("WARGNING: Using random spatial weights!")
            W = np.random.randn(*W.shape)#np.random.permutation(W).copy()

        if covariance_type == 'class_conditional':
            dcov = self.patterns['dcov']['class_conditional'] # (n_ch, n_ch, n_classes, folds)
        elif covariance_type == 'common':
            dcov = self.patterns['dcov']['input_spatial'] # (n_ch, n_ch, folds)
        elif  covariance_type == 'identity':
            dcov = np.identity(self.patterns['dcov']['input_spatial'].shape[0])
        else:
            raise ValueError("""Covariance type {} not implemented. Valid ooptions are
                      'common' and 'class_conditional'.format(covariance_type)""")
        # if use_y_cov:
        #     if diff:
        #         y_cov = self.patterns['ccms']['cov_y_hat']
        #     else:
        y_cov = self.patterns['ccms']['cov_y']
        y_cov_hat = self.patterns['ccms']['cov_y_hat']

        class_topos = []
        if method in ['weight', 'compwise_loss', 'output_corr', 'shap_o3']:
            ind = np.argmax(F, 0)

            for y_ind in range(n_y):
                fold_topos = []
                for fold_ind in range(n_folds):
                    if covariance_type == 'class_conditional':
                        a = np.dot(dcov[:, :, y_ind, fold_ind],
                                   W[:, ind[y_ind, fold_ind], fold_ind])
                    elif  covariance_type == 'common':
                        a = np.dot(dcov[:, :, fold_ind],
                                   W[:, ind[y_ind, fold_ind], fold_ind])
                    elif  covariance_type == 'identity':
                        a = W[:, ind[y_ind, fold_ind], fold_ind]

                    fold_topos.append(a)
                class_topos.append(np.stack(fold_topos, 1))
            patterns = np.stack(class_topos, 1)
            # if use_y_cov:

            #     patterns = np.stack([np.dot(patterns[:, :, f],
            #                                 np.dot(y_cov[:, :, f],
            #                                        np.linalg.inv(y_cov_hat[:, :, f])))
            #                          for f in range(n_folds)], 2)

            print('single:', patterns.shape)
        elif method == 'combined':

            fold_topos = []
            for fold_ind in range(n_folds):
                class_topos = []
                for y_ind in range(n_y):
                    #Computed weighted sum of spatial patterns weighted by their relevance
                    if covariance_type == 'class_conditional':
                        a = np.dot(dcov[..., y_ind, fold_ind],
                                      W[..., fold_ind]) # (n_ch, n_components)
                    elif covariance_type == 'common':
                        a = np.dot(dcov[:, :, fold_ind],
                                   W[..., fold_ind])

                    class_topos.append(a)
                class_topos = np.stack(class_topos, 2) # (n_ch, n_components, n_folds)
                #print('combined, class topos', class_topos.shape)
                if use_y_cov:
                    yc = np.dot(y_cov[:, :, fold_ind],
                                np.linalg.inv(y_cov_hat[:, :, fold_ind]))
                    w_out = np.dot(F[:, :, fold_ind], yc)
                    #print('F: ', F.shape)
                    #print('w_out: ', w_out.shape)
                    pattern = np.sum(class_topos * w_out[None, ...], axis=1)
                    #print(pattern.shape)
                else:
                    pattern = np.sum(class_topos * F[None, :, :, fold_ind], 1) #weighted sum over components
                fold_topos.append(pattern)
            patterns = np.stack(fold_topos, 2) #(n_ch, n_y, n_folds)
            print('combined:', patterns.shape)

            #patterns = self.aligned_mean(class_topos)
        else:
            raise NotImplementedError("""Method {} not implmented. Available options are
                      'weight', 'compwise_loss', 'output_corr', and 'combined'""".format(method))
            patterns = None

        return patterns

    #def plot_parcels(self, brain, method='weight'):

    def get_timecourse(self, method='weight'):

        # if average_over == 'folds':
        #     axis = 2
        # elif average_over == 'vars':
        #     axis = 0


        activations = self.patterns['ccms']['tconv']
        waveforms = self.patterns['ccms']['dmx']
        #get relevances and corresponding filter coeffificents
        relevances = self.patterns[method]['feature_relevance']
        kernels = self.weights['tconv']#.transpose([0, 2, 1])

        #get component index for each fold
        component_inds = np.argmax(np.max(relevances, axis=0), axis=0)

        n_y = np.prod(self.data['y_shape'])

        #aggregate over folds
        n_folds = len(self.data['folds'][0])
        print("kernels: ", kernels.shape, 'inds: ', component_inds.shape)
        tconv_weights = np.stack([kernels[:, component_inds[:, i], i]
                                  for i in range(n_folds)], -1) #n_classes, n_folds, filter_length
        cc_act = np.stack([np.stack([activations[:, component_inds[jj, i], jj, i]
                  for jj in range(n_y)], -1) for i in range(n_folds)], -1) # n_t_pooled, n_classes, n_folds

        cc_waveforms = np.stack([np.stack([waveforms[:, component_inds[jj, i], jj, i]
                  for jj in range(n_y)], -1) for i in range(n_folds)], -1) # n_t, n_classes, n_folds


        return cc_act, cc_waveforms, tconv_weights, component_inds

    def get_spectra(self, n_fft=128, method='weight', diff=False):

        """
        Computes the filter frequency response and reconstructed component power
        spectrum from the weights of the temporal convolution kernel.

        Returns:
        --------

        psds : np.array (n_freq, n_y, n_folds)
            Relative power spectra of each component before temporal filtering.

        out : np.array (n_freq, n_y, n_folds)
            Relative power spectra of each component after temporal filtering.

        freq_responses : np.array (n_freq, n_y, n_folds)
            Frequency responses of each convolutional kernel

        """
        #TODO: Implement 'combined'

        F, names = self.get_feature_relevances(sorting=method,
                                        integrate=['timepoints'],
                                        diff=diff) #(n_components, n_classes, n_folds)
        n_components, n_y, n_folds = F.shape
        kernels = self.weights['tconv']
        psds =  self.patterns['ccms']['psds'] #n_freq, n_components, n_folds


        fr = []
        #Compute frequency resposnes for each kernel
        for fold_ind in range(n_folds):
            realh = []
            for i, flt in enumerate(kernels[:, :, fold_ind].T):
                flt -= flt.mean()
                w, h = freqz(flt, 1, worN=n_fft, fs=self.data['fs'])
                realh.append(np.abs(h))
            fr.append(np.stack(realh, 1)) # (n_fft, n_comps)
        freq_responses = np.stack(fr, 2) # (n_fft, n_comps, n_folds)



        if method in ['weight', 'compwise_loss', 'output_corr']:
            ind = np.argmax(F, 0)
            class_hs = []
            class_psds = []
            for y_ind in range(n_y):
                fold_h = []
                fold_psd = []

                for fold_ind in range(n_folds):
                    fold_psd.append(psds[:, ind[y_ind, fold_ind], fold_ind])
                    fold_h.append(freq_responses[:, ind[y_ind, fold_ind], fold_ind])

                class_psds.append(np.stack(fold_psd, 1))
                class_hs.append(np.stack(fold_h, 1))
            class_psds = np.stack(class_psds, 1)
            class_hs = np.stack(class_hs, 1)

            out = class_psds * class_hs
        else:
            raise NotImplementedError("""Method {} is not implemented.
                                      Viable options are 'weight', 'output_corr',
                                      and 'compwise_loss'""".format(method))

        return class_psds, out, freq_responses


    def plot_spectra(self, method='weight',
                     class_subset=None,
                     log=True,
                     fs=None,
                     freqs_lim=(1, 25)):
        #TODO: class names
        #def plot_spectra(self, patterns_struct, component_ind, ax, fs=None, loag=False):
        """Relative power spectra of a given latent componende before and after
            applying the convolution.

        Parameters
        ----------

        patterns_struct :
            instance of patterns_struct produced by model.compute_patterns

        ax : axes

        fs : float
            Sampling frequency.

        log : bool
            Apply log-transform to the spectra.
        """

        psds, h, freq_responses = self.get_spectra(method=method)


        if class_subset:
            h = h[:, class_subset, :]
            freq_responses = freq_responses[:, class_subset, :]
            psds = psds[:, class_subset, :]

        else:
            class_subset = np.arange(0,  psds.shape[1], 1.)

        n_freq, n_y, n_folds = h.shape

        # psds /= np.sum(psds, 0, keepdims=True)
        # h /= np.sum(h, 0, keepdims=True)
        # freq_responses /= np.sum(freq_responses, 0, keepdims=True)

        if freqs_lim:
            #ax[i].set_xlim(freqs_lim[0], freqs_lim[1])
            vmin = .9*np.min(h[freqs_lim[0] : freqs_lim[1], :])
            vmax = 1.1*np.max(psds[freqs_lim[0] : freqs_lim[1], :])

        else:
            vmin = 0.9*np.min(h)
            vmax = 1.1*np.max(psds)
        #print(vmin, vmax)

        f, ax = plt.subplots(1, n_y, sharey=True, figsize=(3*n_y, 4))
        #f.set_size()
        if isinstance(ax, plt.matplotlib.axes._axes.Axes):
            ax = [ax]
        for i in range(n_y):
            h_std = np.std(h[:, i], -1)
            inp_std = np.std(psds[:, i], -1)
            self.plot_temporal_pattern(psds[:, i].mean(-1),
                                       h[:, i].mean(-1),
                                       freq_responses[:, i].mean(-1),
                                       log=log, freqs_lim=freqs_lim,
                                       vlim = (vmin, vmax), ax=ax[i],
                                       #h_std=h_std, inp_std=inp_std
                                       )


        if i == n_y - 1:
            ax[i].legend(frameon=False)
        # if savefig:
        #     figname = '-'.join([self.meta.data['path'] + self.model_specs['scope'],
        #                         self.meta.data['data_id'], method, "spectra.svg"])
        #    f.savefig(figname, format='svg', transparent=True)
        return f

    def plot_temporal_pattern(self, psd, h, freq_response,
                              log=False, vlim=None,
                              freqs_lim=None, ax=None,
                              h_std = None, inp_std = None):
        if not ax:
            f = plt.figure()
            ax = f.gca()

        # if vlim:
        #     vmin = vlim[0]
        #     vmax = vlim[1]
        # elif freqs_lim:
        #     #ax[i].set_xlim(freqs_lim[0], freqs_lim[1])
        #     vmin = np.min(psd[freqs_lim[0] : freqs_lim[1]])
        #     vmax = np.max(h[freqs_lim[0] : freqs_lim[1]])
        # else:
        #     vmin = np.min(psd)
        #     vmax = np.max(h)

        if log:
            ax.semilogy(self.patterns['freqs'], psd,
                           label='Filter input RPS')
            ax.semilogy(self.patterns['freqs'], h,
                                label='Fitler output RPS', color='tab:orange')
            ax.semilogy(self.patterns['freqs'], freq_response,
                            label='Freq response',
                            color='tab:green', linestyle='dotted')
            #vmin = np.log(vmin)
            #vmax = np.log(vmax)
        else:
            psd /= np.sum(psd)
            h /= np.sum(h)
            ax.plot(self.patterns['freqs'],
                       psd,
                       label='Filter input RPS')
            ax.plot(self.patterns['freqs'],
                       h,
                       label='Fitler output RPS',
                       color='tab:orange')
            if np.any(h_std):
                ax.fill_between(self.patterns['freqs'],
                                    h + h_std,
                                    h - h_std,
                                    label='fold variation', color='tab:orange',
                                    alpha=.25)
            if np.any(inp_std):
                ax.fill_between(self.patterns['freqs'],
                                    psd + inp_std,
                                    psd - inp_std,
                                    label='fold variation', color='tab:blue',
                                    alpha=.25)

            ax.plot(self.patterns['freqs'], freq_response / np.sum(freq_response),
                            label='Freq response',
                            color='tab:green', linestyle='dotted')

        # ax.set_ylim(0.75*vmin, 1.25*vmax)
        if freqs_lim:
            ax.set_xlim(freqs_lim[0], freqs_lim[1])
        return ax


    def plot_timecourses(self, method='weight', average_over='folds',
                         class_names=None, tmin=0, class_subset=None,
                         freqs_lim=(1, 70)):
        tcs = self.get_timecourse(method=method)
        cc_activations, cc_waveforms, tconv_weights, comp_inds = tcs
        print(cc_activations.shape, cc_waveforms.shape, tconv_weights.shape)

        if not class_subset:
            class_subset = np.arange(0,  np.prod(self.data['y_shape']), 1)
        else:
            cc_waveforms = cc_waveforms[:, class_subset, :]
            cc_activations = cc_activations[ :, class_subset, :]



        psds, h, freq_responses  = self.get_spectra(method=method)

        psds /= np.sum(psds, 0, keepdims=True)
        h /= np.sum(h, 0, keepdims=True)
        freq_responses /= np.sum(freq_responses, 0, keepdims=True)
        #cc_activations -= np.min(cc_activations, 0, keepdims=True)
        #cc_activations /= np.max(cc_activations, 0, keepdims=True)

        n_classes = len(class_subset)
        n_folds = len(self.data['folds'][0])





        if not class_names:
            class_names = ["Class {}".format(i) for i in range(n_classes)]

        if average_over == 'folds':
            sd_waveforms = cc_waveforms.std(-1) / np.sqrt(cc_waveforms.shape[-1])
            sd_activations = cc_activations.std(-1) / np.sqrt(cc_waveforms.shape[-1])
            cc_waveforms = cc_waveforms.mean(-1)
            cc_activations = cc_activations.mean(-1)

            h_std = h.std(-1)
            psds_std = psds.std(-1)
            psds = psds.mean(-1)
            h = h.mean(-1)
            #freq_responses = freq_responses.mean(-1)
            n_folds = 1
            n_rows = n_classes

        else:
            n_rows = n_folds
            h_std = np.zeros((1, n_rows))
            psds_std = np.zeros((1, n_rows))

        f, ax = plt.subplots(n_rows, 2)
        ax = np.atleast_2d(ax)

        f.set_size_inches([16, 16])
        n_t = self.data['n_t']

        tstep = 1/float(self.data['fs'])
        times = tmin + tstep*np.arange(n_t)

        for i in range(n_rows):

            # if apply_kernels:
            #     scaled_waveforms = np.array([np.convolve(kern, wf, 'same')
            #                 for kern, wf in zip(self.filters, self.waveforms)])
            #     #scaled_waveforms =(scaled_waveforms - scaled_waveforms.mean(-1, keepdims=True))  / (2*scaled_waveforms.std(-1, keepdims=True))
            # else:
            #     #scaling = 3*np.mean(np.std(self.waveforms, -1))

            #     scaled_waveforms = (waveforms - waveforms.mean(-1, keepdims=True))  / (2*waveforms.std(-1, keepdims=True))
            # if bp_filter:
            #     scaled_waveforms = scaled_waveforms.astype(np.float64)
            #     scaled_waveforms = filter_data(scaled_waveforms,
            #                                       self.dataset.h_params['fs'],
            #                                       l_freq=bp_filter[0],
            #                                       h_freq=bp_filter[1],
            #                                       method='iir',
            #                                       verbose=False)
            ax[i, 0].plot(times, cc_waveforms[..., i], alpha=.75, color='tab:blue')
            ax[i, 0].fill_between(times,
                                  cc_waveforms[..., i] - sd_waveforms[..., i],
                                  cc_waveforms[..., i] + sd_waveforms[..., i], alpha=.25, color='tab:blue')
            ax[i, 0].plot(times, cc_activations[..., i], alpha=.75, color='tab:orange')

            # ax[i, 0].pcolor(times[::self.model_specs['stride']], np.arange(0, 2),
            #                 np.mean(cc_activations[:, i, :], -1, keepdims=True).T,
            #                 cmap='cividis')

            #ax[i, 1].stem(np.mean(tconv_weights[i, :],-1))
            print(psds_std.shape)

            self.plot_temporal_pattern(psds[..., i], h[..., i], freq_responses[..., i],
                                       ax = ax[i, 1], freqs_lim=freqs_lim,
                                       h_std=h_std[:, i],
                                       inp_std=psds_std[:, i])



        return