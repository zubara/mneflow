# -*- coding: utf-8 -*-
"""
Define mneflow.models.Model parent class and the implemented models as
its subclasses. Implemented models inherit basic methods from the
parent class.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import os
import warnings
import itertools
import csv

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

from .layers import ConvDSV, Dense, vgg_block, LFTConv, VARConv, DeMixing
from .keras_layers import LSTMv1
from tensorflow.keras import regularizers as k_reg, constraints


def uniquify(seq):
    un = []
    [un.append(i) for i in seq if not un.count(i)]
    return un


# ----- Base model -----
class Model(object):
    """Parent class for all MNEflow models.

    Provides fast and memory-efficient data handling and simplified API.
    Custom models can be built by overriding _build_graph and
    _set_optimizer methods.
    """

    def __init__(self, Dataset, Optimizer, specs):
        """
        Parameters
        -----------
        Dataset : mneflow.Dataset
            `Dataset` object.

        Optimizer : mneflow.Optimizer
            `Optimizer` object.

        specs : dict
            Dictionary of model-specific hyperparameters. Must include
            at least `model_path` - path for saving a trained model.
            See `Model` subclass definitions for details.
        """
        self.specs = specs
        self.model_path = specs['model_path']
        self.y_shape = Dataset.h_params['y_shape']
        self.fs = Dataset.h_params['fs']
        self.sess = tf.Session()
        self.handle = tf.placeholder(tf.string, shape=[])
        self.train_iter, self.train_handle = self._start_iterator(Dataset.train)
        self.val_iter, self.val_handle = self._start_iterator(Dataset.val)
        if hasattr(Dataset, 'test'):
            self.test_iter, self.test_handle = self._start_iterator(Dataset.test)

        self.iterator = tf.data.Iterator.from_string_handle(
                self.handle,
                tf.data.get_output_types(Dataset.train),
                tf.data.get_output_shapes(Dataset.train))

        self.X0, self.y_ = self.iterator.get_next()
        print('X0:', self.X0.shape)

        if len(self.X0.shape) == 3:
            self.X = tf.expand_dims(self.X0, -1)
        else:
            self.X = self.X0

        self.rate = tf.placeholder(tf.float32, name='rate')
        self.dataset = Dataset
        self.optimizer = Optimizer
        self.trained = False

    def _start_iterator(self, Dataset):
        """Build initializable iterator and string handle."""
        ds_iterator = tf.data.make_initializable_iterator(Dataset)
        handle = self.sess.run(ds_iterator.string_handle())
        self.sess.run(ds_iterator.initializer)

        return ds_iterator, handle

    def build(self):
        """Compile a model."""
        # Initialize computational graph
        self.y_pred = self.build_graph()
        print('X:', self.X.shape)
        print('y_pred:', self.y_pred.shape)

        # Initialize optimizer
        self.saver = tf.train.Saver(max_to_keep=1)
        opt_handles = self.optimizer._set_optimizer(self.y_pred, self.y_)
        self.train_step, self.accuracy, self.cost, self.p_classes = opt_handles

        print('Initialization complete!')

    def build_graph(self):
        """Build computational graph using defined placeholder self.X
        as input.

        Can be overriden in a sub-class for customized architecture.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        warnings.warn('Specify a model! Set to linear classifier', UserWarning)
        fc_1 = Dense(size=np.prod(self.y_shape),
                     nonlin=tf.identity,
                     dropout=self.rate)
        y_pred = fc_1(self.X)

        return y_pred

    def train(self, n_iter, eval_step=250, min_delta=1e-6, early_stopping=5):
        """
        Train a model

        Parameters
        -----------

        n_iter : int
            Maximum number of training iterations.

        eval_step : int
            How often to evaluate model performance during training.

        early_stopping : int
            Patience parameter for early stopping. Specifies the number
            of 'eval_step's during which validation cost is allowed to
            rise before training stops.

        min_delta : float
            Convergence threshold for validation cost during training.
            Defaults to 1e-6.
        """
        if not self.trained:
            self.sess.run(tf.global_variables_initializer())
            self.min_val_loss = np.inf
            self.t_hist = []
        if not early_stopping:
            early_stopping = np.inf
        patience_cnt = 0
        for i in range(n_iter+1):
            _ = self.sess.run([self.train_step],
                              feed_dict={self.handle: self.train_handle,
                                         self.rate: self.specs['dropout']})
            if i % eval_step == 0:
                self.dataset.train.shuffle(buffer_size=10000)
                t_loss, acc = self.sess.run([self.cost, self.accuracy],
                                            feed_dict={self.handle:
                                                       self.train_handle,
                                                       self.rate: 0.})
                self.v_acc, v_loss = self.sess.run([self.accuracy, self.cost],
                                                   feed_dict={self.handle:
                                                              self.val_handle,
                                                              self.rate: 0.})
                self.t_hist.append([t_loss, v_loss])

                if self.min_val_loss >= v_loss + min_delta:
                    self.min_val_loss = v_loss
                    v_acc = self.v_acc
                    self.saver.save(self.sess,
                                    ''.join([self.model_path,
                                            self.scope, '-',
                                            self.dataset.h_params['data_id']]))
                else:
                    patience_cnt += 1
                    print('* Patience count {}'.format(patience_cnt))
#        if (patience_cnt >= early_stopping//2 and self.specs['dropout'] != 0):
#            print('Setting dropout to 0.')
#            self.specs['dropout'] = 0.
#            self.optimizer.params['l1_lambda'] *= 10.

                if patience_cnt >= early_stopping:
                    print("early stopping...")
                    self.saver.restore(
                            self.sess,
                            ''.join([self.model_path, self.scope, '-',
                                     self.dataset.h_params['data_id']]))

                    self.train_params = (eval_step, early_stopping, i)
                    print('stopped at: epoch %d, val loss %g, val acc %g'
                          % (i,  self.min_val_loss, v_acc))
                    break

                print('i %d, tr_loss %g, tr_acc %g v_loss %g, v_acc %g'
                      % (i, t_loss, acc, v_loss, self.v_acc))

        self.train_params = (eval_step, early_stopping, i)
        self.trained = True

    def plot_hist(self):
        """Plot loss history during training."""
        plt.plot(np.array(self.t_hist))
        plt.legend(['t_loss', 'v_loss'])
        plt.title(self.scope.upper())
        plt.xlabel('Epochs')
        plt.show()

    def load(self):
        """Loads a pretrained model.

        To load a specific model the model object should be initialized
        using the corresponding metadata and computational graph.
        """
        self.saver.restore(self.sess,
                           ''.join([self.model_path, self.scope, '-',
                                    self.dataset.h_params['data_id']]))

        self.v_acc = self.sess.run([self.accuracy],
                                   feed_dict={self.handle: self.val_handle,
                                              self.rate: 0.})
        self.trained = True

    def _add_dataset(self, data_path):
        """Add as test dataset the one specified by `data_path`.

        Parameters
        ----------
        data_path : str, list of str
            Path to .tfrecords file(s).
        """
        self.dataset.test = self.dataset._build_dataset(data_path,
                                                        n_batch=None)
        self.test_iter, self.test_handle = self._start_iterator(self.dataset.test)

    def evaluate_performance(self, data_path=None):
        """Compute performance metric on a TFR dataset specified by
        `data_path`.

        Parameters
        ----------
        data_path : str, list of str
            path to .tfrecords file(s).

        Raises:
        -------
            AttributeError: If `data_path` is not specified.
        """
        if not data_path:
            raise AttributeError('Specify data_path!')

        # elif not hasattr(self.dataset, 'test'):
        self._add_dataset(data_path)

        acc = self.sess.run(self.accuracy,
                            feed_dict={self.handle: self.test_handle,
                                       self.rate: 0.})

        print('Finished: acc: %g +\\- %g' % (np.mean(acc), np.std(acc)))
        return np.mean(acc)

    def predict(self, data_path):
        """Predict model output on a TFR dataset specified by
        `data_path`.

        Parameters
        ----------
        data_path : str, list of str
            Path to .tfrecords file(s).

        Returns:
        --------
        pred: int or float
            The model prediction.

        true: int or float
            The true target value.

        Raises:
        -------
            AttributeError: If `data_path` is not specified.
        """
        if not data_path:
            raise AttributeError('Specify data_path!')
        else:
            self._add_dataset(data_path)
            pred, true = self.sess.run(
                    [self.y_pred, self.y_],
                    feed_dict={self.handle: self.test_handle, self.rate: 0.})
            return pred, true

    def update_log(self):
        """Logs experiment to self.model_path + self.scope + '_log.csv'.

        If the file exists, appends a line to the existing file.
        """
        appending = os.path.exists(self.model_path + self.scope + '_log.csv')

        log = dict()
        log['data_id'] = self.dataset.h_params['data_id']
        log['eval_step'], log['patience'], log['n_iter'] = self.train_params
        log['data_path'] = self.dataset.h_params['savepath']
        log['decim'] = str(self.dataset.decim)

        if self.dataset.class_subset:
            log['class_subset'] = '-'.join(
                    str(self.dataset.class_subset).split(','))
        else:
            log['class_subset'] = 'all'

# sclass = self.dataset.h_params['class_proportions']
# log['class_proportions'] = ' : '.join([str(v)[:4] for v in sclass.values()])
# if self.dataset.h_params['task'] == 'classification':
#     log['n_classes'] = self.dataset.h_params['n_classes']
# else:

        log['y_shape'] = np.prod(self.dataset.h_params['y_shape'])
        log['fs'] = str(self.dataset.h_params['fs'])
        log.update(self.optimizer.params)
        log.update(self.specs)

        v_acc, v_loss = self.sess.run(
                [self.accuracy, self.cost],
                feed_dict={self.handle: self.val_handle, self.rate: 0.})
        log['v_acc'] = v_acc
        log['v_loss'] = v_loss

        t_acc, t_loss = self.sess.run(
                [self.accuracy, self.cost],
                feed_dict={self.handle: self.train_handle, self.rate: 0.})
        log['train_acc'] = t_acc
        log['train_loss'] = t_loss
        self.log = log

        with open(self.model_path + self.scope + '_log.csv', 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.log.keys())
            if not appending:
                writer.writeheader()
            writer.writerow(self.log)

#    def evaluate_realtime(self, data_path, batch_size=None, step_size=1):
#        """Compute performance metric on a TFR dataset specified by path
#            batch by batch with updating the model after each batch."""
#        prt_batch_pred = []
#        prt_logits = []
#        n_test_points = batch_size//step_size
#        count = 0
#
#        test_dataset = tf.data.TFRecordDataset(data_path
#                                               ).map(self._parse_function)
#        test_dataset = test_dataset.batch(step_size)
#        test_iter = test_dataset.make_initializable_iterator()
#        self.sess.run(test_iter.initializer)
#        test_handle = self.sess.run(test_iter.string_handle())
#
#        while True:
#            try:
#                self.load()
#                count = count + 1
#                preds = 0
#                for jj in range(n_test_points):
#                    pred, probs = self.sess.run(
#                            [self.correct_prediction, self.p_classes],
#                            feed_dict={self.handle: test_handle,
#                                       self.rate: 0})
#                    self.sess.run(
#                            self.train_step,
#                            feed_dict={self.handle: test_handle,
#                                       self.rate: self.specs['dropout']})
#
#                    preds = preds + np.mean(pred)
#                    prt_logits.append(probs)
#                prt_batch_pred.append(preds/n_test_points)
#
#            except tf.errors.OutOfRangeError:
#                print('prt_done: count: %d, acc: %g +\\- %g'
#                      % (count, np.mean(prt_batch_pred),
#                         np.std(prt_batch_pred)))
#                break
#        return prt_batch_pred, np.concatenate(prt_logits)

    def plot_cm(self, dataset='validation', class_names=None, normalize=False):

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
        if dataset == 'validation':
            feed_dict = {self.handle: self.val_handle, self.rate: 0.}
        elif dataset == 'training':
            feed_dict = {self.handle: self.train_handle, self.rate: 0.}
        elif dataset == 'test':
            feed_dict = {self.handle: self.test_handle, self.rate: 0.}
        else:
            raise ValueError('Invalid dataset type.')

        y_true, y_pred = self.sess.run([self.y_, self.p_classes],
                                       feed_dict=feed_dict)
        y_pred = np.argmax(y_pred, 1)
        y_true = np.argmax(y_true, 1)

        f = plt.figure()
        cm = confusion_matrix(y_true, y_pred)
        title = 'Confusion matrix: '+dataset.upper()
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


# ----- Models -----
class VGG19(Model):
    """VGG-19 model.

    References
    ----------
    #[] TODO! missing
    """
    def __init__(self, Dataset, params, specs):
        super().__init__(Dataset, params, specs)
        self.specs = dict(n_ls=self.specs['n_ls'], nonlin=tf.nn.relu,
                          inch=1, padding='SAME', filter_length=(3, 3),
                          domain='2d', stride=1, pooling=1, conv_type='2d')
        self.scope = 'vgg19'

    def build_graph(self):
        X1 = self.X  # tf.expand_dims(self.X, -1)
        if X1.shape[1] == 306:
            X1 = tf.concat([X1[:, 0:306:3, :],
                            X1[:, 1:306:3, :],
                            X1[:, 2:306:3, :]], axis=3)
            self.specs['inch'] = 3

        vgg1 = vgg_block(2, ConvDSV, self.specs)
        out1 = vgg1(X1)

        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *= 2
        vgg2 = vgg_block(2, ConvDSV, self.specs)
        out2 = vgg2(out1)

        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *= 2
        vgg3 = vgg_block(4, ConvDSV, self.specs)
        out3 = vgg3(out2)

        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *= 2
        vgg4 = vgg_block(4, ConvDSV, self.specs)
        out4 = vgg4(out3)

        self.specs['inch'] = self.specs['n_ls']
        vgg5 = vgg_block(4, ConvDSV, self.specs)
        out5 = vgg5(out4)

        fc_1 = Dense(size=4096, nonlin=tf.nn.relu, dropout=self.rate)
        fc_2 = Dense(size=4096, nonlin=tf.nn.relu, dropout=self.rate)
        fc_out = Dense(size=np.prod(self.y_shape), nonlin=tf.identity,
                       dropout=self.rate)

        y_pred = fc_out(fc_2(fc_1(out5)))
        return y_pred


class EEGNet(Model):
    """EEGNet.

    Parameters
    ----------
    specs : dict

        n_ls : int
            Number of (temporal) convolution kernrels in the first layer.
            Defaults to 8

        filter_length : int
            Length of temporal filters in the first layer.
            Defaults to 32

        stride : int
            Stride of the average polling layers. Defaults to 4.

        pooling : int
            Pooling factor of the average polling layers. Defaults to 4.

        dropout : float
            Dropout coefficient.

    References
    ----------
    [1] V.J. Lawhern, et al., EEGNet: A compact convolutional neural
    network for EEG-based brain–computer interfaces 10 J. Neural Eng.,
    15 (5) (2018), p. 056013

    [2] Original EEGNet implementation by the authors can be found at
    https://github.com/vlawhern/arl-eegmodels
    """

    def build_graph(self):
        self.scope = 'eegnet'

        X1 = self.X  # tf.expand_dims(self.X, -1)
        vc1 = ConvDSV(n_ls=self.specs['n_ls'], nonlin=tf.identity, inch=1,
                      filter_length=self.specs['filter_length'], domain='time',
                      stride=1, pooling=1, conv_type='2d')
        vc1o = vc1(X1)

        bn1 = tf.layers.batch_normalization(vc1o)
        dwc1 = ConvDSV(n_ls=1, nonlin=tf.identity, inch=self.specs['n_ls'],
                       padding='VALID', filter_length=bn1.get_shape()[1].value,
                       domain='space',  stride=1, pooling=1,
                       conv_type='depthwise')
        dwc1o = dwc1(bn1)

        bn2 = tf.layers.batch_normalization(dwc1o)
        out2 = tf.nn.elu(bn2)
        out22 = tf.nn.dropout(out2, rate=self.rate)

        sc1 = ConvDSV(n_ls=self.specs['n_ls'], nonlin=tf.identity,
                      inch=self.specs['n_ls'],
                      filter_length=self.specs['filter_length']//4,
                      domain='time', stride=1, pooling=1,
                      conv_type='separable')
        sc1o = sc1(out22)

        bn3 = tf.layers.batch_normalization(sc1o)
        out3 = tf.nn.elu(bn3)

        out4 = tf.nn.avg_pool(out3, [1, 1, self.specs['pooling'], 1],
                              [1, 1, self.specs['stride'], 1], 'SAME')
        out44 = tf.nn.dropout(out4, rate=self.rate)

        sc2 = ConvDSV(n_ls=self.specs['n_ls']*2, nonlin=tf.identity,
                      inch=self.specs['n_ls'],
                      filter_length=self.specs['filter_length']//4,
                      domain='time', stride=1, pooling=1,
                      conv_type='separable')
        sc2o = sc2(out44)

        bn4 = tf.layers.batch_normalization(sc2o)
        out5 = tf.nn.elu(bn4)

        out6 = tf.nn.avg_pool(out5, [1, 1, self.specs['pooling'], 1],
                              [1, 1, self.specs['stride'], 1], 'SAME')
        out66 = tf.nn.dropout(out6, rate=self.rate)

        out7 = tf.reshape(out66, [-1, np.prod(out66.shape[1:])])
        fc_out = Dense(size=self.y_shape[0],
                       nonlin=tf.identity,
                       dropout=self.rate)
        y_pred = fc_out(out7)

        return y_pred


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

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.scope = 'lf-cnn'

        self.demix = DeMixing(n_ls=self.specs['n_ls'], axis=1)

        self.tconv1 = LFTConv(scope="conv",
                              n_ls=self.specs['n_ls'],
                              nonlin=self.specs['nonlin'],
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              pool_type=self.specs['pool_type'],
                              padding=self.specs['padding'])

        self.tconv_out = self.tconv1(self.demix(self.X))

        self.fin_fc = Dense(size=np.prod(self.y_shape),
                            nonlin=tf.identity,
                            dropout=self.rate)

        y_pred = self.fin_fc(self.tconv_out)
        # y_pred = tf.reshape(y_pred, [-1, *self.y_shape])

        return y_pred

    def compute_patterns(self, data_path=None, output='patterns'):
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

        if data_path:
            self._add_dataset(data_path)
        elif not hasattr(self.dataset, 'test') and not data_path:
            raise AttributeError('Specify data path.')

        vis_dict = {self.handle: self.test_handle, self.rate: 0}

        # Spatial stuff
        data, demx = self.sess.run([self.X, self.demix.W], feed_dict=vis_dict)
        data = np.squeeze(data.transpose([1, 2, 3, 0]))
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

        kern, tc_out, out_w = self.sess.run(
                [self.tconv1.filters, self.tconv_out, self.fin_fc.w],
                feed_dict=vis_dict)
        print('out_w:', out_w.shape)

        #  Temporal conv stuff
        self.filters = np.squeeze(kern)
        self.tc_out = np.squeeze(tc_out)
        self.out_weights = np.reshape(out_w, [-1, self.specs['n_ls'],
                                      np.prod(self.y_shape)])

        print('demx:', demx.shape,
              'kern:', self.filters.shape,
              'tc_out:', self.tc_out.shape,
              'out_w:', self.out_weights.shape)

        self.get_output_correlations()

        self.out_biases = self.sess.run(self.fin_fc.b, feed_dict=vis_dict)

    def get_output_correlations(self):
        """Computes a similarity metric between each of the extracted
        features and the target variable.

        The metric is a Manhattan distance for dicrete targets, and
        Spearman correlation for continuous targets.
        """
        self.rfocs = []
        y_true = self.sess.run(self.y_,
                               feed_dict={self.handle: self.test_handle,
                                          self.rate: 0.})
        flat_feats = self.tc_out.reshape(self.tc_out.shape[0], -1)

        if self.dataset.h_params['target_type'] == 'float':
            for y_ in y_true.T:
                rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])
                self.rfocs.append(rfocs.reshape(self.out_weights.shape[:-1]))

        elif self.dataset.h_params['target_type'] == 'int':
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
        vmin = np.min(self.out_weights)
        vmax = np.max(self.out_weights)

        f, ax = plt.subplots(1, np.prod(self.y_shape))
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        for i in range(len(ax)):
            if 'weight' in sorting:
                F = self.out_weights[..., i].T
            elif 'spear' in sorting:
                F = self.rfocs[..., i].T
            else:
                F = self.rfocs[..., i].T * self.out_weights[..., i].T

            tstep = self.specs['stride']/float(self.fs)
            times = tmin+tstep*np.arange(F.shape[-1])

            im = ax[i].pcolor(times, np.arange(self.specs['n_ls'] + 1), F,
                              cmap='bone_r', vmin=vmin, vmax=vmax)

            r = []
            if np.any(pat) and np.any(t):
                r = [ptch.Rectangle((times[t], p), width=tstep,
                                    height=1, angle=0.0)
                     for p, t in zip(pat[i], t[i])]

                pc = collections.PatchCollection(r, facecolor='red', alpha=.5,
                                                 edgecolor='red')
                ax[i].add_collection(pc)

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
        """Specify which compontens to plot.

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
            for i in range(np.prod(self.y_shape)):
                self.F = self.out_weights[..., i].T * self.rfocs[..., i].T
                pat = np.argmax(self.F.sum(-1))
                order.append(np.tile(pat, self.F.shape[1]))
                ts.append(np.arange(self.F.shape[-1]))

        elif sorting == 'best':
            for i in range(np.prod(self.y_shape)):
                self.F = np.abs(self.out_weights[..., i].T
                                * self.rfocs[..., i].T)
                pat, t = np.where(self.F == np.max(self.F))
                print('Maximum spearman r * weight:', np.max(self.F))
                order.append(pat)
                ts.append(t)

        elif sorting == 'weight':
            for i in range(np.prod(self.y_shape)):
                self.F = self.out_weights[..., i].T
                pat, t = np.where(self.F == np.max(self.F))
                print('Maximum weight:', np.max(self.F))
                order.append(pat)
                ts.append(t)

        elif sorting == 'spear':
            for i in range(np.prod(self.y_shape)):
                self.F = self.rfocs[..., i].T
                print('Maximum r_spear:', np.max(self.F))
                pat, t = np.where(self.F == np.max(self.F))
                order.append(pat)
                ts.append(t)

        elif isinstance(sorting, int):
            for i in range(np.prod(self.y_shape)):
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

        nfilt = max(np.prod(self.y_shape), 8)
        nrows = max(1, l_u//nfilt)
        ncols = min(nfilt, l_u)

        f, ax = plt.subplots(nrows, ncols, sharey=True)
        f.set_size_inches([16, 9])
        ax = np.atleast_2d(ax)

        for i in range(nrows):
            fake_times = np.arange(i * ncols,  (i + 1) * ncols, 1.)
            vmax = np.percentile(self.fake_evoked.data[:, :l_u], 95)
            self.fake_evoked.plot_topomap(times=fake_times,
                                          axes=ax[i],
                                          colorbar=False,
                                          vmax=vmax,
                                          scalings=1,
                                          time_format='output # %g',
                                          title='Patterns ('+str(sorting)+')')
        if np.any(ts):
            self.plot_out_weights(pat=order, t=ts, sorting=sorting)
        else:
            self.plot_out_weights()

        return None  # TODO! Gabi: is it supposed to return f, ax ?

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

        nfilt = max(np.prod(self.y_shape), 8)
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


class VARCNN(Model):
    """VAR-CNN.

    For details see [1].

    n_ls : int
        Number of latent components.
        Defaults to 32.

    nonlin : callable
        Activation function of the temporal Convolution layer.
        Defaults to tf.nn.relu

    filter_length : int
        Length of spatio-temporal kernels in the temporal
        convolution layer. Defaults to 7

    pooling : int
        Pooling factor of the max pooling layer. Defaults to 2

    pool_type : str {'avg', 'max'}
        Type of pooling operation. Defaults to 'max'

    padding : str {'SAME', 'FULL', 'VALID'}
        Convolution padding. Defaults to 'SAME'

    stride : int
        Stride of the max pooling layer. Defaults to 1

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def build_graph(self):
        self.scope = 'var-cnn'
        self.demix = DeMixing(n_ls=self.specs['n_ls'], axis=1)

        self.tconv1 = VARConv(scope="conv",
                              n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'],
                              pool_type=self.specs['pool_type'])

        self.tconv_out = self.tconv1(self.demix(self.X))

        self.fin_fc = Dense(size=np.prod(self.y_shape),
                            nonlin=tf.identity,
                            dropout=self.rate)

        y_pred = self.fin_fc(self.tconv_out)
        return y_pred


class LFLSTM(LFCNN):
    # TODO! Gabi: check that the description describes the model
    """LF-CNN-LSTM

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

    def build_graph(self):
        self.scope = 'lf-cnn-lstm'

        self.demix = DeMixing(n_ls=self.specs['n_ls'], axis=1)
        dmx = self.demix(self.X)
        dmx = tf.reshape(dmx, [-1, self.dataset.h_params['n_t'],
                               self.specs['n_ls']])
        dmx = tf.expand_dims(dmx, -1)
        print('dmx-sqout:', dmx.shape)

        self.tconv1 = LFTConv(scope="conv",
                              n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        features = self.tconv1(dmx)
        print('features:', features.shape)
        fshape = tf.multiply(features.shape[1], features.shape[2])
        if 'n_seq' in self.dataset.h_params.keys():
            features = tf.reshape(
                    features, [-1, self.dataset.h_params['n_seq'], fshape])
        else:
            features = tf.reshape(features, [-1, 1, fshape])
        #  features = tf.expand_dims(features, 0)
        l1_lambda = self.optimizer.params['l1_lambda']
        print('flat features:', features.shape)
        self.lstm = LSTMv1(scope="lstm",
                           size=self.specs['n_ls'],
                           kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal',
                           recurrent_regularizer=k_reg.l1(l1_lambda),
                           kernel_regularizer=k_reg.l2(l1_lambda),
                           # bias_regularizer=None,
                           # activity_regularizer= regularizers.l1(0.01),
                           # kernel_constraint= constraints.UnitNorm(axis=0),
                           # recurrent_constraint= constraints.NonNeg(),
                           # bias_constraint=None,
                           # dropout=0.1, recurrent_dropout=0.1,
                           nonlin=tf.nn.tanh,
                           unit_forget_bias=False,
                           return_sequences=False,
                           unroll=False)

        lstm_out = self.lstm(features)
        print('lstm_out:', lstm_out.shape)
        # if 'n_seq' in self.dataset.h_params.keys():
        #    lstm_out = tf.reshape(lstm_out, [-1,
        #                                     self.dataset.h_params['n_seq'],
        #                                     self.specs['n_ls']])

        self.fin_fc = Dense(size=np.prod(self.y_shape),
                            nonlin=tf.identity, dropout=0.)
#        self.fin_fc = DeMixing(n_ls=np.prod(self.y_shape),
#                               nonlin=tf.identity, axis=-1)
        y_pred = self.fin_fc(lstm_out)
        # print(y_pred)
        return y_pred
