# -*- coding: utf-8 -*-
"""
Specifies utility functions.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import os
import pickle
import warnings
import numpy as np
import tensorflow as tf
import scipy.io as sio
import mne
from mneflow import models
from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable   

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
        
        
    def restore_model(self):
        """Restored previously saved model from metadata."""
        
        if self.model_specs['scope'] == 'lfcnn':
            model = models.LFCNN(meta=self)
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
    
    def make_fake_evoked(self, topos, sensor_layout):
        if 'info' not in self.data.keys():
            lo = mne.channels.read_layout(sensor_layout)
            #lo = channels.generate_2d_layout(lo.pos)
            info = mne.create_info(lo.names, 1., sensor_layout.split('-')[-1])
            orig_xy = np.mean(lo.pos[:, :2], 0)
            for i, ch in enumerate(lo.names):
                if info['chs'][i]['ch_name'] == ch:
                    info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/4.5
                    #info['chs'][i]['loc'][4:] = 0
                else:
                    print("Channel name mismatch. info: {} vs lo: {}".format(
                        info['chs'][i]['ch_name'], ch))
        #info['sfreq'] = 1
        fake_evoked = mne.evoked.EvokedArray(topos, info)
        return fake_evoked
    
    def explore_components(self, sorting='output_corr',
                         integrate=['vars', 'folds'], info=None, 
                         sensor_layout='Vectorview-grad',
                         class_names=None):
        """Plots the weights of the output layer.

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

        def _onclick_component(event):
            
            x_ind = np.maximum(np.round(event.xdata, 0).astype(int), 0)
            y_ind = np.maximum(np.round(event.ydata).astype(int), 0)
            


            f1, ax = plt.subplots(2,2, tight_layout=True)
            #f1.set_layout_engine('constrained')
            f1.suptitle("{}: {}  {}: {}"
                  .format(names[0][:-1], x_ind, names[1][:-1], y_ind))

            # a = np.dot(self.patterns['dcov']['class_conditional'][x_ind].mean(-1),
            #            self.patterns['weights']['dmx'])

            self.fake_evoked_interactive.data[:, y_ind] = topos[:, y_ind]
            self.fake_evoked_interactive.plot_topomap(times=[y_ind],
                                                      axes=ax[0, 0],
                                                      colorbar=False,
                                                      time_format='Spatial activation pattern, [au]'
                                                      )
            psd = psds[:, y_ind].T
            freq_response = freq_responses[:, y_ind].T
            out_psd = psd*freq_response
            psd /= np.maximum(np.sum(psd), 1e-9)
            out_psd /= np.maximum(np.sum(out_psd), 1e-9)
            #freq /= np.maximum(np.sum(out_psd), 1e-9)
            #ax[1,].clear()
            ax[0,1].semilogy(self.patterns['freqs'], psd,
                         label='Input RPS')
            ax[0,1].semilogy(self.patterns['freqs'], out_psd,
                         label='Output RPS')
            #ax[0,1].semilogy(patterns_struct['spectra']['freqs'], freq_response,
            #             label='Freq_response')
            ax[0,1].set_xlim(1, 125.)
            ax[0,1].set_ylim(1e-6, 1.)
            ax[0,1].legend(frameon=False)
            ax[0,1].set_title('Relative power spectra')


            #ax[1,0].stem(tconv_kernels[:, y_ind])
            ax[1,0].set_title('Temporal convolution kernel coefficients')
            if trans:
                ax[1,1].plot(F[y_ind, :], 'ks')
                ax[1,1].plot(y_ind, F[y_ind, x_ind], 'rs')
            else:                    
                ax[1,1].plot(F[y_ind, :], 'ks')
                ax[1,1].plot(x_ind, F[y_ind, x_ind], 'rs')
            ax[1,1].set_title('{} per {}. Red={}'.format(sorting, names[0][:-1], x_ind))
            ax[1,1].set_xlabel(names[0])



        # get feature relevance
        F = self.patterns[sorting]['feature_relevance']
        ax = 1
        names = ['Time Points', 'Components', 'Classes', 'Folds']
        if 'folds' in integrate:
            F = F.mean(3, keepdims=True)
            names.pop(names.index('Folds'))
        if 'timepoints' in integrate:
            F = F.max(0, keepdims=True)
            ax = 0
            names.pop(names.index('Time Points'))
        if 'vars' in integrate:
            F = F.sum(2, keepdims=True)
            names.pop(names.index('Classes'))
        if 'components' in integrate:
            F = F.sum(1, keepdims=True)
            names.pop(names.index('Components'))
        
        #print(names)
        
        F = np.squeeze(F)
        while F.ndim < 2:
            F = np.expand_dims(F, -1)
            ax = 0
            
        if F.ndim > 2:
            print("""Integrate: {} returned shape {}. \n
                  Can only plot 2 dimensions!""".format(integrate, F.shape))
            return
        if 'timepoints' in integrate:
            F = F.T
            trans = False
            ax = 1 - ax
            print(names)
        else:
            trans = False
            names = names[::-1]
        
        # if F.shape[1] < F.shape[0]:
        #     F = F.T
        #     
                    
        inds = np.argmax(F, ax)
        
        print(inds)
        # new_topos = [np.dot(self.patterns['dcov']['input_spatial'][:, :, i], 
        #                     self.weights['dmx'][:, ind, i])
        #               for i, ind in enumerate(inds)]
        #new_topos = np.concatenate(new_topos, axis=0).T
        

        #print(new_topos.shape)
        

        #psds = patterns_struct['spectra']['psds']
        topos =  np.squeeze(self.patterns[sorting]['spatial'])#.mean(-1)
        #print(topos.shape)
        psds = np.squeeze(self.patterns[sorting]['psds'])#.mean(-1)
        freq_responses = np.squeeze(self.patterns[sorting]['temporal'])#.mean(-1)
        #tconv_kernels = self.patterns['weights']['tconv']
        self.fake_evoked_interactive = self.make_fake_evoked(topos, sensor_layout)

        vmin = np.min(F)
        vmax = np.max(F)

        f = plt.figure()
        ax = f.gca()
        
        im = ax.imshow(F, cmap='bone_r', vmin=vmin, vmax=vmax)
        if trans:
            r = [ptch.Rectangle((i - .5, ind - .5), width=1,
                                height=1, angle=0.0, facecolor='none') for i, ind in enumerate(inds)]
        else:
            r = [ptch.Rectangle((ind - .5, i - .5), width=1,
                                height=1, angle=0.0, facecolor='none') for i, ind in enumerate(inds)]

        pc = collections.PatchCollection(r, facecolor='none', alpha=.5,
                                          edgecolor='red')
        #ax = f.gca()
        ax.add_collection(pc)
        ax.set_ylabel(names[1])
        ax.set_xlabel(names[0])
        ax.set_title('Component relevance map: {} (Clickable)'.format(sorting))

        f.colorbar(im)
        f.canvas.mpl_connect('button_press_event', _onclick_component)
        f.show()
        return f
    
    def plot_topos(self, topos, sensor_layout='Vectorview-mag', class_subset=None):
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

        Returns
        -------
        None.

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
                                    time_format="Class %g",
                                    outlines='head',
                                    #vlim= np.percentile(topos, [5, 95])
                                    )
        #ft.show()
        return ft
    
    def plot_combined_pattern(self):
        return
    
    def plot_spectra(self, method='output_corr', log=True, savefig=False):
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

        
        #self.fs = self.data['fs']

        filters = self.patterns[method]['temporal'].mean(-1)
        p_input = self.patterns[method]['psds'].mean(-1)
        h = (self.patterns[method]['temporal']*self.patterns[method]['psds']).mean(-1)
        h_std = (self.patterns[method]['temporal']*self.patterns[method]['psds']).std(-1)
        n = p_input.shape[-1]
        #p_input = p_input/p_input.sum(0, keepdims=True).mean(-1)

        # w, h = freqz(filters, 1, worN=self.nfft)
        # fr1 = w/np.pi*self.fs/2
        # h0 = p_input*np.abs(h)
        f, ax = plt.subplots(1, n, sharey=True)
        for i in range(n):
            #h = p_input[:, i]*filters[:, i]
            if log:
                ax[i].semilogy(self.patterns['freqs'], p_input[:, i],
                               label='Filter input RPS')
                ax[i].semilogy(self.patterns['freqs'], h[:, i]/np.sum(h[:, i]),
                                    label='Fitler output RPS', color='tab:orange')
                # ax[i].semilogy(self.patterns['freqs'], filters[:, i],
                #                label='Freq response RPS')
            else:
                ax[i].plot(self.patterns['freqs'], 
                           p_input[:, i] / np.sum(p_input[:, i]),
                           label='Filter input RPS')
                ax[i].plot(self.patterns['freqs'], h[:, i]/np.sum(h[:, i]), 
                           label='Fitler output RPS', 
                           color='tab:orange')
                ax[i].fill_between(self.patterns['freqs'], 
                                   h[:, i] / np.sum(h[:, i]) + h_std[:, i], 
                                   h[:, i] / np.sum(h[:, i]) - h_std[:, i], 
                                   label='fold variation', color='tab:orange', 
                                   alpha=.25)
                # ax[i].plot(self.patterns['freqs'], filters[:, i] / np.sum(filters[:, i]),
                #                 label='Freq response')
            ax[i].set_xlim(1, 90)
        if i == n - 1:
            ax[i].legend(frameon=False)
        if savefig:
            figname = '-'.join([self.meta.data['path'] + self.model_specs['scope'], 
                                self.meta.data['data_id'], method, "spectra.svg"])
            f.savefig(figname, format='svg', transparent=True)
        return f
    
    
    def plot_waveforms(self):
        return
    
    

def _onehot(y, n_classes=False):
    """
    Transforms n-by-1 vector of class labels into n-by-n_classes array of
    one-hot encoded labels

    Parameters
    ----------
    y : array of ints
        Array of class labels

    n_classes : int
        Number of classes. If set to False (default) n_classes is set to number of
        unique labels in y


    Returns
    -------
    y_onehot : array
        array of onehot encoded labels

    """
    if not n_classes:
        """Create one-hot encoded labels."""
        n_classes = len(set(y))
    out = np.zeros((len(y), n_classes))
    for i, ii in enumerate(y):
        out[i][ii] += 1
    y_onehot = out.astype(int)
    return y_onehot


def load_meta(path, data_id=''):
    # TODO: expand functionality?
    """Load a metadata file.

    Parameters
    ----------
    path : str
        Path to TFRecord folder

    Returns
    -------
    meta : MetaData
        Metadata file

    """
    with open(path+data_id+'_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return meta

# def load_model(model_path):
    
#     return model 




def scale_to_baseline(X, baseline=None, crop_baseline=False):
    """Perform global scaling based on a specified baseline.

    Subtracts the mean of each channel and divides by the standard deviation of
    all channels during the specified baseline interval.

    Parameters
    ----------
    X : ndarray
        Data array with dimensions [n_epochs, n_channels, time].

    baseline : tuple of int, None
        Baseline definition (in samples). If baseline is set to None (default)
        the whole epoch is used for scaling.

    crop_baseline : bool
        Whether to crop the baseline after scaling is applied. Only used if
        baseline is specified.
    Returns
    -------
    X : ndarray
        Scaled data array.

    """
    #X = X_.copy()

    if baseline is None:
        print("No baseline interval specified, scaling based on the whole epoch")
        interval = np.arange(X.shape[-1])
    elif isinstance(baseline, tuple):
        print("Scaling to interval {:.1f} - {:.1f}".format(*baseline))
        interval = np.arange(baseline[0], baseline[1])
    X0m = X[..., interval].mean(axis=2, keepdims=True)
    X0sd = X[..., interval].std(axis=(1,2), keepdims=True)

    X -= X0m
    X /= X0sd
    if crop_baseline and baseline is not None:
        X = np.delete(X, interval, axis=-1)
    #print("Scaling Done")
    return X


def _make_example(X, y, n, target_type='int'):
    """Construct a serializable example proto object from data and
    target pairs."""

    feature = {}
    feature['X'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=X.flatten()))
    feature['n'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=n.flatten()))

    if target_type == 'int':
        feature['y'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=y.flatten()))
    elif target_type in ['float', 'signal']:
        y = y.astype(np.float32)
        feature['y'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=y.flatten()))
    else:
        raise ValueError('Invalid target type.')

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def _write_tfrecords(X_, y_, n_, output_file, target_type='int'):
    """Serialize and write datasets in TFRecords format.

    Parameters
    ----------
    X_ : list of ndarrays
        (Preprocessed) data matrix.
        len = `n_epochs`, shape = `(squence_length, n_timepoints, n_channels)`

    y_ : list of ndarrays
        Class labels.
        len =  `n_epochs`, shape = `y_shape`

    n_ : int
        nubmer of training examples

    output_file : str
        Name of the TFRecords file.
    """
    writer = tf.io.TFRecordWriter(output_file)

    for X, y, n in zip(X_, y_, n_):
        # print(len(X_), len(y_), X.shape, y.shape)
        X = X.astype(np.float32)
        n = n.astype(np.int64)
        # Feature contains a map of string to feature proto objects
        example = _make_example(X, y, n, target_type=target_type)
        # Serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to the disk
        writer.write(serialized)
    writer.close()


def _split_indices(X, y, n_folds=5):
    # TODO: check if indices are permuted
    """Generate indices for n-fold cross-validation"""
    n = X.shape[0]
    print('n:', n)
    #original_indices = np.arange(n)
    shuffle = np.random.permutation(n)
    subset_proportion = 1./float(n_folds)
    fold_size = int(subset_proportion*n)
    folds = [shuffle[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    return folds


def _split_sets(X, y, folds, ind=-1, sample_counter=0):
    """Split the data returning a single fold specified by ind as a holdout set
        and the rest of the data as training/validation sets.

    Parameters
    ----------
    X : ndarray
        (Preprocessed) data matrix.
        shape (n_epochs, ...)

    y : ndarray
        Class labels.
        shape (n_epochs, ...)

    folds : list of arrays
        fold indices

    ind : index of the selected fold, defaults to -1

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarray
        Pairs of data / targets split in Training and Validation sets.

    test_fold : np.array
        Array of indices of data samples in the held out fold


    """

    fold = folds.pop(ind) - sample_counter
    X_test = X[fold, ...]
    y_test = y[fold, ...]
    X_train = np.delete(X, fold, axis=0)
    y_train = np.delete(y, fold, axis=0)
    test_fold = fold + sample_counter
    # return X_train, np.squeeze(y_train), X_val, np.squeeze(y_val)
    return X_train, y_train, X_test, y_test, test_fold


def import_data(inp, array_keys={'X': 'X', 'y': 'y'}):
    """Import epoch data into `X, y` data/target pairs.

    Parameters
    ----------
    inp : list, mne.epochs.Epochs, str
        List of mne.epochs.Epochs or strings with filenames.
        If input is a single string or Epochs object, it is first converted
        into a list.

    array_keys : dict, optional
        Dictionary mapping {'X': 'data_matrix', 'y': 'labels'},
        where 'data_matrix' and 'labels' are names of the corresponding
        variables, if the input is paths to .mat or .npz files.
        Defaults to {'X': 'X', 'y': 'y'}

    Returns
    -------
    data, targets: ndarray
        data.shape =  [n_epochs, channels, times]

        targets.shape =  [n_epochs, y_shape]

    """
    if isinstance(inp, (mne.epochs.EpochsFIF, mne.epochs.BaseEpochs)):
        print('processing epochs')

        inp.load_data()
        data = inp.get_data()
        events = inp.events[:, 2]


    elif isinstance(inp, tuple) and len(inp) == 2:
        print('importing from tuple')
        data, events = inp

    elif isinstance(inp, str):
        # TODO: ADD CASE FOR RAW FILE
        fname = inp
        if fname[-3:] == 'fif':
            epochs = mne.epochs.read_epochs(fname, preload=True,
                                            verbose='CRITICAL')
            print(np.unique(epochs.events[:, 2]))
            events = epochs.events[:, 2]
            epochs.crop(tmin=-1., tmax=1.)
            data = epochs.get_data()


        else:
            if fname[-3:] == 'mat':
                datafile = sio.loadmat(fname)

            if fname[-3:] == 'npz':
                print('Importing from npz')
                datafile = np.load(fname)

            data = datafile[array_keys['X']]
            events = datafile[array_keys['y']]
            print('Extracting target variables from {}'
                  .format(array_keys['y']))
    else:
        print("Dataset not found")
        return None, None

    data = data.astype(np.float32)

    # Make sure that X is 3d here
    while data.ndim < 3:
        # (x, ) -> (1, 1, x)
        # (x, y) -> (1, x, y)
        data = np.expand_dims(data, 0)

    return data, events


def produce_tfrecords(inputs, 
                      path, 
                      data_id, 
                      fs=1.,
                      input_type='trials',
                      target_type='int',
                      array_keys={'X': 'X', 'y': 'y'},
                      n_folds=5,
                      predefined_split=None,
                      test_set=False,
                      scale=False,
                      scale_interval=None,
                      crop_baseline=False,
                      segment=False,
                      aug_stride=None,
                      seq_length=None,
                      overwrite=False,
                      transform_targets=False,
                      scale_y=False,
                      ):

    """Produce TFRecord files from input, apply (optional) preprocessing.

    Calling this function will convert the input data into TFRecords
    format that is used to effiently store and run Tensorflow models on
    the data.


    Parameters
    ----------
    inputs : mne.Epochs, list of str, tuple of ndarrays
        Input data.

    path : str
        A path where the output TFRecord and corresponding metadata
        files will be stored.

    data_id : str
        Filename prefix for the output files.

    fs : float, optional
         Sampling frequency, required only if inputs are not mne.Epochs

    input_type : str {'trials', 'continuous', 'seq', 'fconn'}
        Type of input data.

        'trials' - treats each of n inputs as an iid sample, produces dataset
        with dimensions (n, 1, t, ch)

        'seq' - treats each of n inputs as a seqence of shorter segments,
        produces dataset with dimensions (n, seq_length, segment, ch)

        'continuous' - treats inputs as a single continuous sequence,
        produces dataset with dimensions (n*(t-segment)//aug_stride, 1, segment, ch)

    target_type : str {'int', 'float'}
        Type of target variable.

        'int' - for classification,
        'float' - for regression problems.
        'signal' - regression or classification a continuous (possbily multichannel) 
        data. Requires "transform_targets" function to be applied to target 
        variables

    n_folds : int, optional
        Number of folds to split the data for training/validation/testing.
        One fold of the n_folds is used as a validation set.
        If test_set == 'holdout' generates one extra fold
        used as test set. Defaults to 5
    
    predefined_split : list or lists, optional
        Pre-defined split of the dataset into training/validation folds. 
        Should match exactly the size of the dataset, and contain n_folds

    test_set : str {'holdout', 'loso', None}, optional
        Defines if a separate holdout test set is required.
        'holdout' saves 50% of the validation set
        'loso' saves the whole dataset in original order for
        leave-one-subject-out cross-validation.
        None does not leave a separate test set. Defaults to None.


    segment : bool, int, optional
        If specified, splits the data into smaller segments of specified
        number of time points. Defaults to False

    aug_stride : int, optional
        Sliding window agumentation stride parameter.
        If specified, sets the stride (in time points) for 'segment'
        allowing to extract overalapping segments. Has to be <= segment.
        Only applied within each fold to prevent data leakeage. Only applied
        if 'segment' is not False. If None, then it is set equal to length of
        the 'segment' returning non-overlapping segments.
        Defaults to None.


    scale : bool, optional
        Whether to perform scaling to baseline. Defaults to False.

    scale_interval : NoneType, tuple of ints,  optional
        Baseline definition. If None (default) scaling is
        performed based on all timepoints of the epoch.
        If tuple, then baseline is data[tuple[0] : tuple[1]].
        Only used if scale == True.

    crop_baseline : bool, optional
        Whether to crop baseline specified by 'scale_interval'
        after scaling. Defaults to False.

    array_keys : dict, optional
        Dictionary mapping {'X':'data_matrix','y':'labels'},
        where 'data_matrix' and 'labels' are names of the
        corresponding variables if the input is paths to .mat or .npz
        files. Defaults to {'X':'X', 'y':'y'}

    transform_targets : callable, optional
        custom function used to transform target variables

    seq_length : int, optional
        Length of segment sequence.

    overwrite : bool, optional
        Whether to overwrite the metafile if it already exists at the
        specified path.

    Returns
    -------
    meta : mneflow.MetaData
        Metadata associated with the processed dataset. Contains all
        the information about the dataset required for further
        processing with mneflow.
        Whenever the function is called the copy of metadata is also
        saved to data_path/meta.pkl so it can be restored at any time.


    Notes
    -----
    Pre-processing functions are implemented mostly for for convenience
    when working with array inputs. When working with mne.epochs the
    use of the corresponding mne functions is preferred.

    Examples
    --------
    >>> meta = mneflow.produce_tfrecords(input_paths, \**import_opts)
    """
    
    assert input_type in ['trials', 'seq', 'continuous', 'fconn'], "Unknown input type."
    assert target_type in ['int', 'float', 'signal'], "Unknown target type."
    
    if not os.path.exists(path):
        os.mkdir(path)
    data_path = os.path.join(path, 'tfrecords')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    meta_fname = os.path.join(path, data_id+'_meta.pkl')
    if not os.path.exists(meta_fname) or  overwrite:
        #print("(Re)-importing data")
        train_size = 0
        test_size = 0
        val_size = 0
        folds = []
        train_paths=[]
        test_paths=[]
        
        jj = 0
        if test_set == 'holdout':
            n_folds += 1

        #meta['fs'] = fs

        if not isinstance(inputs, list):
            inputs = [inputs]
        #print("inputs:", inputs, len(inputs), type(inputs))
        if len(inputs) == 0:
            print("Cannot process Input: {} of type {}".format(inputs, type(inputs)))
            return
        for inp in inputs:
            #print("inp:", inp, len(inp), type(inp))
            

            data, events = import_data(inp, array_keys=array_keys)

            if np.any(data) == None:
                return

            else:
                if input_type == 'continuous':
                    # if input is a continuous signal ensure that target
                    # variable has shape (n_epochs, channels, time)
                    # TODO: replace with "target type?"
                    while events.ndim < 3:
                        events = np.expand_dims(events, 0)
                else:
                    # if input is trials, ensure that target variable has shape
                    # (n_trials, y_shape)
                    if events.ndim < 2:
                        events = np.expand_dims(events, -1)

                if input_type == 'trials':
                    segment_y = False
                else:
                    segment_y = True

                if input_type == 'fconn':
                    assert data.shape[1] == data.shape[2], "data.shape incompatible with fconn input type"
                    print('Input shapes: X (n, ch, ch, freq) : ', data.shape,
                          'y (n, [signal_channels], y_shape) : ', events.shape,
                          '\n',
                          'input_type : ', input_type,
                          'target_type : ', target_type,
                          'segment_y : ', segment_y)

                else:
                    print('Input shapes: X (n, ch, t) : ', data.shape,
                          'y (n, [signal_channels], y_shape) : ', events.shape,
                          '\n',
                          'input_type : ', input_type,
                          'target_type : ', target_type,
                          'segment_y : ', segment_y)

                X, Y, fold_split = preprocess(
                        data, events,
                        sample_counter=train_size,
                        input_type=input_type,
                        n_folds=n_folds,
                        scale=scale,
                        scale_interval=scale_interval,
                        crop_baseline=crop_baseline,
                        segment=segment, aug_stride=aug_stride,
                        seq_length=seq_length,
                        segment_y=segment_y)
                
                if predefined_split:
                    assert len(predefined_split) == len(fold_split) or len(predefined_split) == (len(fold_split) - 1), "Number of folds in predefined_split does not match n_folds!"
                    assert np.all([len(fpd) == len(fa) for fpd, fa in zip(predefined_split, fold_split)]), "Number of samples in predefined folds does not match the original split!"
                    fold_split = predefined_split                   

                Y = preprocess_targets(Y, scale_y=scale_y,
                                       transform_targets=transform_targets)

                if target_type == 'int':
                    Y, n_ev, class_ratio, orig_classes = produce_labels(Y)
                    Y = _onehot(Y)
                else:
                    class_ratio = dict()
                    orig_classes = dict()


                if test_set == 'holdout':
                    X, Y, x_test, y_test, test_fold = _split_sets(X, Y,
                                                                  folds=fold_split,
                                                                  sample_counter=train_size)
                    test_size += x_test.shape[0]
                else:
                    test_fold = None
                    #TODO: remove?
#                if input_type == 'fconn':
#                    _n, meta['n_ch'], meta['n_t'], meta['n_freq'] = X.shape
#                else:
                _n, n_seq, n_t, n_ch = X.shape


                if input_type == 'seq':
                    y_shape = Y[0].shape[1:]
                else:
                    y_shape = Y[-1].shape

                n = np.arange(_n) + train_size

                train_size += _n

                val_size += len(fold_split[0])

                print('Prepocessed sample shape:', X[0].shape)
                print('Target shape actual/metadata: ', Y[0].shape, y_shape)

                print('Saving TFRecord# {}'.format(jj))

                folds.append(fold_split)
                trname = ''.join([data_id, '_train_', str(jj), '.tfrecord'])
                train_filename = os.path.join(data_path, trname)
                train_paths.append(train_filename)

                _write_tfrecords(X, Y, n, train_filename, 
                                 target_type=target_type)

                if test_set == 'loso':
                    test_size = len(Y)
                    tfrname = ''.join([data_id, '_test_', str(jj), '.tfrecord'])
                    test_filename = os.path.join(data_path, tfrname)
                                        
                    _write_tfrecords(X, Y, n, test_filename,
                                     target_type=target_type)

                elif test_set == 'holdout':
                    #meta['test_fold'].append(test_fold)
                    tfrname = ''.join([data_id, '_test_', str(jj), '.tfrecord'])
                    test_filename = os.path.join(data_path, tfrname)
                    n_test = np.arange(len(test_fold))
                    
                    _write_tfrecords(x_test, y_test, n_test, test_filename,
                                     target_type=target_type)
                    test_paths.append(test_filename)
                jj += 1
                #create and save metadata file
                
            meta_data = dict(path=path,
                             data_path=data_path, 
                             target_type=target_type,
                             input_type=input_type, 
                             data_id=data_id,
                             test_set=test_set,
                             train_paths=train_paths,
                             test_paths=test_paths,
                             folds=folds,
                             n_folds=n_folds,
                             test_fold=test_fold,
                             train_size=train_size,
                             test_size=test_size,
                             val_size=val_size,
                             n_seq=n_seq,
                             n_t=n_t,
                             n_ch=n_ch,
                             y_shape=y_shape,
                             class_ratio=class_ratio,
                             orig_classees=orig_classes,
                             fs=fs)
            
            meta_preprocessing = dict(scale=scale,
                                      scale_interval=scale_interval,
                                      crop_baseline=crop_baseline,
                                      segment=segment, aug_stride=aug_stride,
                                      seq_length=seq_length,
                                      segment_y=segment_y)
            
            meta = MetaData()
            meta.update(data=meta_data, preprocessing=meta_preprocessing)       
            
            with open(path+data_id+'_meta.pkl', 'wb') as f:
                pickle.dump(meta, f)

    elif os.path.exists(meta_fname):
        print('Metadata file found, restoring')
        meta = load_meta(path, data_id=data_id)
    else:
        print(os.path.join(path, data_id+'_meta.pkl'), "Does not exit, aborting")
        return
    return meta

def produce_labels(y, return_stats=True):
    """Produce labels array from e.g. event (unordered) trigger codes.

    Parameters
    ----------
    y : ndarray, shape (n_epochs,)
        Array of trigger codes.

    return_stats : bool
        Whether to return optional outputs.

    Returns
    -------
    inv : ndarray, shape (n_epochs)
        Ordered class labels.

    total_counts : int, optional
        Total count of events.

    class_proportions : dict, optional
        {new_class: proportion of new_class in the dataset}.

    orig_classes : dict, optional
        Mapping {new_class_label: old_class_label}.
    """
    classes, inds, inv, counts = np.unique(y,
                                           return_index=True,
                                           return_inverse=True,
                                           return_counts=True)
    total_counts = np.sum(counts)
    counts = counts/float(total_counts)
    class_proportions = {clss: cnt for clss, cnt in zip(inv[inds], counts)}
    orig_classes = {new: old for new, old in zip(inv[inds], classes)}
    if return_stats:
        return inv, total_counts, class_proportions, orig_classes
    else:
        return inv


def _combine_labels(labels, new_mapping):
    """Combine event labels.

    Parameters
    ----------
    labels : ndarray
        Label vector

    combine_dict : dict
        Mapping {new_label1: [old_label1, old_label2], ...}

    Returns
    -------
    new_labels : ndarray
        Updated label vector.

    keep_ind : ndarray
        Label indices.
    """
    assert isinstance(new_mapping, dict), "Invalid label mapping."
    # Find all possible label values
    print(labels)
    tmp = []
    for k, j in new_mapping.items():
        tmp.append(k)
        if not isinstance(j, (list, tuple)):
            # for simplicity, force all old_labels to be lists
            new_mapping[k] = [j]
        tmp.extend(new_mapping[k])

    # pick the exlusion value
    inv = np.min(tmp) - 1
    new_labels = inv*np.ones(len(labels), int)

    for new_label, old_label in new_mapping.items():
        # print(old_label, new_label)
        ind = [ii for ii, v in enumerate(labels) if v in old_label]
        new_labels[ind] = int(new_label)
    keep_ind = np.where(new_labels != inv)[0]
    #print(new_labels, keep_ind)
    return new_labels, keep_ind


def _segment(data, segment_length=200,
             seq_length=None,
             stride=None,
             input_type='trials'):
    """Split the data into fixed-length segments.

    Parameters
    ----------
    data : ndarray
        Data array of shape (n_epochs, n_channels, n_times)

    labels : ndarray
        Array of labels (n_epochs, y_shape)

    seq_length: int or None
        Length of segment sequence.

    segment_length : int or False
        Length of segment into which to split the data in time samples.

    stride : int, optional
        If specified, sets the stride (in time points) for 'segment'
        allowing to extract overalapping segments. Has to be <= segment.
        Only applied within each fold to prevent data leakeage. Only applied
        if 'segment' is not False. If None, then it is set equal to length of
        the 'segment' returning non-overlapping segments.
        Defaults to None.

    Returns
    -------
    data : ndarray
        Segmented data array of shape
        (n, [seq_length,] n_channels, segment_length)
        where n = (n_epochs//seq_length)*(n_times - segment_length + 1)//stride
        """
    x_out = []
    if input_type == 'trials':
        seq_length = 1

    if not stride:
        stride = segment_length

    for jj, xx in enumerate(data):

        n_ch, n_t = xx.shape
        last_segment_start = n_t - segment_length

        starts = np.arange(0, last_segment_start+1, stride)

        segments = [xx[..., s:s+segment_length] for s in starts]

        if input_type == 'seq':
            if not seq_length:
                seq_length = len(segments)
            seq_bins = np.arange(seq_length, len(segments)+1, seq_length)
            segments = np.split(segments, seq_bins, axis=0)[:-1]
            x_new = np.array(segments)
        else:
            x_new = np.stack(segments, axis=0)
#            if not events:
#                x_new = np.expand_dims(x_new, 1)

        x_out.append(x_new)
    if len(x_out) > 1:
        X = np.concatenate(x_out)
    else:
        X = x_out[0]
    print("Segmented as: {}".format(input_type), X.shape)
    return X


def cont_split_indices(data, events, n_folds=5, segments_per_fold=10):
    """
    Parameters
    ----------
    data : ndarray
            3d data array (n, ch, t)

    n_folds : int
             number of folds

    segments_per_fold : int
                        minimum number of different (non-contiguous)
                        data segments in each fold
    Returns
    -------
    data : ndarray
           3d data array (n, ch, t)

    events : nd.array
           labels

    folds : list of ndarrays
            indices for each fold

    """
    raw_len = data.shape[-1]
    # Define minimal duration of a single, non-overlapping data segment
    ind_samples = int(raw_len//(segments_per_fold*n_folds))
    
    segments = np.arange(0, raw_len - ind_samples + 1, ind_samples)
    data = np.concatenate([data[:, :, s: s + ind_samples] for s in segments])
    # Split continous data into non-overlapping segments
    events = np.concatenate([events[:, :, s: s + ind_samples] for s in segments])


    folds = _split_indices(data, events, n_folds=n_folds)
    return data, events, folds


def preprocess_realtime(data, decimate=False, picks=None,
                        bp_filter=False, fs=None):
    """
    Implements minimal prprocessing for convenitent real-time use.

    Parameters
    ----------
    data : np.array, (n_epochs, n_channels, n_times)
           input data array

    picks : np.array
            indices of channels to pick

    decimate : int
                decimation factor for downsampling

    bp_filter : tuple of ints
                Band-pass filter cutoff frequencies

    fs : int
         sampling frequency. Only used if bp_filter is used

    Returns
    -------
    """
    if bp_filter:
        print('Filtering')
        data = data.astype(np.float64)
        data = mne.filter.filter_data(data, fs, l_freq=bp_filter[0],
                                      h_freq=bp_filter[1],
                                      method='iir', verbose=False)

    if isinstance(picks, np.ndarray):
        data = data[:, picks, :]

    if decimate:
        print("Decimating")
        data = data[..., ::decimate]
    return data


def preprocess(data, events, sample_counter,
               input_type='trials', n_folds=5,
               scale=False, scale_interval=None, crop_baseline=False,
               segment=False, aug_stride=None,
               seq_length=None,
               segment_y=False):
    """
    Preprocess input data. 
    Applies scaling, segmenting/augmentation,
    and defines the split into training/validation folds.

    Parameters
    ----------
    data : np.array, (n_epochs, n_channels, n_times)
           input data array

    events : np.array
            input array of target variables (n_epochs, ...)

    input_type : str {trials, continuous}
            See produce_tfrecords.

    n_folds : int
            Number of folds defining the train/validation/test split.

    sample_counter : int
            Number of traning examples in the dataset

    scale : bool, optional
        Whether to perform scaling to baseline. Defaults to False.

    scale_interval : NoneType, tuple of ints or floats,  optional
        Baseline definition. If None (default) scaling is
        performed based on all timepoints of the epoch.
        If tuple, than baseline is data[tuple[0] : tuple[1]].
        Only used if scale == True.

    crop_baseline : bool, optional
        Whether to crop baseline specified by \'scale_interval\'
        after scaling (defaults to False).

    segment : bool, int, optional
        If specified, splits the data into smaller segments of specified
        number of time points. Defaults to False

    aug_stride : int, optional
        If specified, sets the stride (in time points) for 'segment'
        allowing to extract overalapping segments. Has to be <= segment.
        Only applied within each fold to prevent data leakeage. Only applied
        if 'segment' is not False. If None, then it is set equal to length of
        the 'segment' returning non-overlapping segments.
        Defaults to None.

    seq_length: int or None
        Length of segment sequence.

    segment_y : bool
        whether to segment target variable in the same way as data. Only used
        if segment != False


    Returns
    -------
    X: np.array
        Data array of dimensions [n_epochs, n_seq, n_t, n_ch]

    Y : np.array
        Label arrays of dimensions [n_epochs, *(y_shape)]

    folds : list of np.arrays
    """
    print("Preprocessing:")

    # TODO: remove scale_y and transform targets?

    if scale:
        data = scale_to_baseline(data, baseline=scale_interval,
                                 crop_baseline=crop_baseline)

    #define folds
    if input_type  == 'continuous':
        data, events, folds = cont_split_indices(data, events,
                                                 n_folds=5,
                                                 segments_per_fold=10)
        shuffle = np.random.permutation(np.arange(events.shape[0]))
        data = data[shuffle]
        events = events[shuffle]
        print("Continuous events: ", events.shape)

    else:
        shuffle = np.random.permutation(np.arange(events.shape[0]))
        data = data[shuffle]
        events = events[shuffle]
        folds = _split_indices(data, events, n_folds=n_folds)

    print("Splitting into: {} folds x {}".format(len(folds), len(folds[0])))

    if segment:
        print("Segmenting")
        X = []
        Y = []
        segmented_folds = []
        jj = 0
        for fold in folds:
            #print(data[fold, ...].shape)
            x = _segment(data[fold, ...], segment_length=segment,
                         stride=aug_stride, input_type=input_type,
                         seq_length=seq_length)

            nsegments = x.shape[0]

            # if segment_y -> segment, else-> replicate
            if segment_y:
                y = _segment(events[fold, ...], segment_length=segment,
                             stride=aug_stride, input_type=input_type,
                             seq_length=seq_length)
            else:
                print("Replicating labels for segmented data")
                y = np.repeat(events[fold, ...], nsegments//len(fold), axis=0)

            if x.ndim == 3:
                x = np.expand_dims(x, 1)
            X.append(x)
            Y.append(y)
            segmented_folds.append(np.arange(jj, jj + nsegments) + sample_counter)
            jj += nsegments
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        folds = segmented_folds
    else:
        # If not segmented add a singleton "n_seq" dminesion to X
        if data.ndim == 3:
            X = np.expand_dims(data, 1)
        elif data.ndim == 4:
            X = data

        Y = events
        folds = [f + sample_counter for f in folds]
    # Finally cast X into shape [n_epochs, n_seq, n_times, n_channels]
    if input_type != 'fconn':
        X = np.swapaxes(X, -2, -1)

    print('Preprocessed:', X.shape, Y.shape,
          'folds:', len(folds), 'x', len(folds[0]))
    assert X.shape[0] == Y.shape[0], "n_epochs in X ({}) does not match n_epochs in Y ({})".format(X.shape[0], Y.shape[0])

    return X, Y, folds

def preprocess_targets(y, scale_y=False, transform_targets=None):

    if callable(transform_targets):
        y = transform_targets(y)

    if scale_y:
            y -= y.mean(axis=0, keepdims=True)
            y /= y.std(axis=0, keepdims=True)
    print('Preprocessed targets: ', y.shape)

    return y


def regression_metrics(y_true, y_pred):
    y_shape = y_true.shape[-1]

    cc = np.diag(np.corrcoef(y_true.T, y_pred.T)[:y_shape,-y_shape:])
    r2 =  r2_score(y_true, y_pred)
    cs = cosine_similarity(y_true, y_pred)
    bias = np.mean(y_true, axis=0) - np.mean(y_pred, axis=0)
    #ve = pve(y_true, y_pred)
    return dict(cc=cc, r2=r2, cs=cs, bias=bias)

def cosine_similarity(y_true, y_pred):
    # y_true -= y_true.mean()
    # y_pred -= y_pred.mean()

    return np.dot(y_pred.T, y_true) / (np.sqrt(np.sum(y_pred**2,axis=0)) * np.sqrt(np.sum(y_true**2, axis=0)))

def pve(y_true, y_pred):
    y_true -= y_true.mean(axis=0)
    y_pred -= y_pred.mean(axis=0)
    return np.dot(y_pred.T, y_true) / np.sum(y_pred**2, axis=0)

def r2_score(y_true, y_pred):
    res = np.sum((y_true - y_pred)**2, axis=0)
    tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True))**2, axis=0)
    return 1 - res/tot

def plot_confusion_matrix(cm,
                          classes=None,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues,
                          vmax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if not classes:
        classes = [' '.join(["Class", str(i)]) for i in range(cm.shape[0])]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #if not vmax:
    #    vmax = np.max(cm)
    #print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmax=vmax)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Predicted label',
           xlabel='True label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            #if i == j:
            ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #fig.show()
    return fig

