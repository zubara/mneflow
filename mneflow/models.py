# -*- coding: utf-8 -*-
"""
Defines mneflow.models.Model parent class and the implemented models as its
subclasses. Implemented models inherit basic methods from the parent class.

"""
from .layers import ConvDSV, Dense, vgg_block, LFTConv, VARConv, DeMixing
import itertools
import tensorflow as tf
import numpy as np

from mne import channels, evoked, create_info
from scipy.signal import freqz  # , welch
from scipy.stats import spearmanr

from sklearn.covariance import ledoit_wolf
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt, patches as ptch, collections
import csv
import os
from .keras_layers import LSTMv1


class Model(object):
    """
    Parent class for all MNEflow models

    Provides fast and memory-efficient data handling and simplified API.
    Custom models can be built by overriding _build_graph and
    set_optimizer methods.

    """
    def __init__(self, Dataset, Optimizer, specs):
        """
        Parameters
        -----------
        Dataset : mneflow.Dataset
                    Dataset object.

        Optimizer : mneflow.Optimizer
                    Optimizer object.

        specs : dict
                dictionary of model-specific hyperparameters. Must include at
                least model_path - path for saving a trained model. See
                subclass definitions for details.

        """

        self.specs = specs
        self.model_path = specs['model_path']
        self.y_shape = Dataset.h_params['y_shape']
        self.fs = Dataset.h_params['fs']
        self.sess = tf.Session()
        self.handle = tf.placeholder(tf.string, shape=[])
        self.train_iter, self.train_handle = self._start_iterator(Dataset.train)
        self.val_iter, self.val_handle = self._start_iterator(Dataset.val)

        self.iterator = tf.data.Iterator.from_string_handle(self.handle, Dataset.train.output_types, Dataset.train.output_shapes)
        self.X0, self.y_ = self.iterator.get_next()
        print('X0:', self.X0.shape)
        #print(len(self.X0.shape))
        if len(self.X0.shape) == 3:
            self.X = tf.expand_dims(self.X0, -1)
#        elif len(self.X0.shape) > 4:
#            self.X = tf.squeeze(self.X0, axis=0)
        else:
            self.X = self.X0
        #print('X:', self.X.shape)
        self.rate = tf.placeholder(tf.float32, name='rate')
        self.dataset = Dataset
        self.optimizer = Optimizer
        self.trained = False

    def _start_iterator(self, Dataset):

        """
        Builds initializable iterator and string handle.
        """

        ds_iterator = Dataset.make_initializable_iterator()
        handle = self.sess.run(ds_iterator.string_handle())
        self.sess.run(ds_iterator.initializer)
        return ds_iterator, handle

    def build(self):

        """
        Compile a model

        """

        # Initialize computational graph
        self.y_pred = self.build_graph()
        print('X:', self.X.shape)
        print('y_pred:', self.y_pred.shape)
        # Initialize optimizer
        self.saver = tf.train.Saver(max_to_keep=1)
        opt_handles = self.optimizer.set_optimizer(self.y_pred, self.y_)
        self.train_step, self.accuracy, self.cost, self.p_classes = opt_handles

        print('Initialization complete!')

    def build_graph(self):

        """
        Build computational graph using defined placeholder self.X as input

        Can be overriden in a sub-class for customized architecture.

        Returns
        --------
        y_pred : tf.Tensor
                output of the forward pass of the computational graph.
                prediction of the target variable

        """
        print('Specify a model. Set to linear classifier!')
        fc_1 = Dense(size=np.prod(self.y_shape), nonlin=tf.identity,
                     dropout=self.rate)
        y_pred = fc_1(self.X)
        return y_pred

    def train(self, n_iter, eval_step=250, min_delta=1e-6, early_stopping=3):
        """
        Trains a model

        Parameters
        -----------

        n_iter : int
                maximum number of training iterations.

        eval_step : int
                How often to evaluate model performance during training.

        early_stopping : int
                Patience parameter for early stopping. Specifies the number of
                'eval_step's during which validation cost is allowed to rise
                before training stops.

        min_delta : float
                Convergence threshold for validation cost during training.
                Defaults to 0.
        """

        if not self.trained:
            self.sess.run(tf.global_variables_initializer())
            self.min_val_loss = np.inf
            self.t_hist = []

        patience_cnt = 0
        for i in range(n_iter+1):
            _ = self.sess.run([self.train_step],
                              feed_dict={self.handle: self.train_handle,
                                         self.rate: self.specs['dropout']})
            if i % eval_step == 0:
                self.dataset.train.shuffle(buffer_size=10000)
                t_loss, acc = self.sess.run([self.cost, self.accuracy],
                                           feed_dict={self.handle: self.train_handle,
                                                      self.rate: 1.})
                self.v_acc, v_loss = self.sess.run([self.accuracy, self.cost],
                                                   feed_dict={self.handle: self.val_handle,
                                                              self.rate: 1.})
                self.t_hist.append([t_loss, v_loss])

                if self.min_val_loss >= v_loss + min_delta:
                    self.min_val_loss = v_loss
                    v_acc = self.v_acc
                    self.saver.save(self.sess, ''.join([self.model_path,
                                                        self.scope, '-',
                                                        self.dataset.h_params['data_id']]))
                else:
                    patience_cnt += 1
                    print('* Patience count {}'.format(patience_cnt))
                    if  patience_cnt >= early_stopping//2 and self.specs['dropout'] != 1:
                        print('Setting dropout to 1.')
                        self.specs['dropout'] = 1.
                        #self.optimizer.params['l1_lambda'] *= 10.
                if patience_cnt >= early_stopping:
                    print("early stopping...")
                    self.saver.restore(self.sess, ''.join([self.model_path,
                                                           self.scope, '-',
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
        plt.plot(np.array(self.t_hist.T))


    def load(self):
        """
        Loads a pretrained model

        To load a specific model the model object should be initialized using
        the corresponding metadata and computational graph
        """

        self.saver.restore(self.sess, ''.join([self.model_path,
                                              self.scope, '-',
                                              self.dataset.h_params['data_id']]))
        self.v_acc = self.sess.run([self.accuracy],
                                   feed_dict={self.handle: self.val_handle,
                                              self.rate: 1.})
        self.trained = True


    def add_dataset(self, data_path):
        self.dataset.test = self.dataset._build_dataset(data_path,
                                                   n_batch=None)
        self.test_iter, self.test_handle = self._start_iterator(self.dataset.test)


    def evaluate_performance(self, data_path=None, batch_size=None):
        """
        Compute performance metric on a TFR dataset specified by path

        Parameters
        ----------
        data_path : str, list of str
                    path to .tfrecords file(s).
        """
        if not data_path:
            print('Specify data_path!')
            return

        elif not hasattr(self.dataset, 'test'):
            self.add_dataset(data_path)

        acc = self.sess.run(self.accuracy, feed_dict={self.handle: self.test_handle,
                                       self.rate: 1.})
        print('Finished: acc: %g +\\- %g' % (np.mean(acc), np.std(acc)))
        return np.mean(acc)

    def predict(self, data_path):
        """
        Compute performance metric on a TFR dataset specified by path

        Parameters
        ----------
        data_path : str, list of str
                    path to .tfrecords file(s).

        batch_size : NoneType, int
                    whether to split the dataset into batches.
        """
        if not data_path:
            print('Specify data_path!')
            return
        else:

            self.add_dataset(data_path)

            pred, true = self.sess.run([self.y_pred, self.y_],
                                   feed_dict={self.handle: self.test_handle,
                                              self.rate: 1.})
            return pred, true

    def update_log(self):
        appending = os.path.exists(self.model_path + self.scope + '_log.csv')
        log = dict()
        log['data_id'] = self.dataset.h_params['data_id']
        log['eval_step'], log['patience'], log['n_iter'] = self.train_params
        log['data_path'] = self.dataset.h_params['savepath']
        log['decim'] = str(self.dataset.decim)
        if self.dataset.class_subset:
            log['class_subset'] = '-'.join(str(self.dataset.class_subset).split(','))
        else:
            log['class_subset'] = 'all'
        #log['class_proportions'] = ' : '.join([str(v)[:4] for v in self.dataset.h_params['class_proportions'].values()])
#        if self.dataset.h_params['task'] == 'classification':
#            log['n_classes'] = self.dataset.h_params['n_classes']
        #else:
        log['y_shape'] = self.dataset.h_params['y_shape']
        log['fs'] = str(self.dataset.h_params['fs'])
        log.update(self.optimizer.params)
        log.update(self.specs)
        v_acc, v_loss = self.sess.run([self.accuracy, self.cost],
                                      feed_dict={self.handle: self.val_handle,
                                      self.rate: 1.})
        log['v_acc'] = v_acc
        log['v_loss'] = v_loss


        t_acc, t_loss = self.sess.run([self.accuracy, self.cost],
                                      feed_dict={self.handle: self.train_handle,
                                      self.rate: 1.})
        log['train_acc'] = t_acc
        log['train_loss'] = t_loss
        self.log = log
        with open(self.model_path + self.scope + '_log.csv', 'a') as csv_file:

            writer = csv.DictWriter(csv_file, fieldnames=self.log.keys())
            if not appending:
                writer.writeheader()

            writer.writerow(self.log)


#    def evaluate_realtime(self, data_path, batch_size=None, step_size=1):
#
#        """Compute performance metric on a TFR dataset specified by path
#            batch by batch with updating the model after each batch """
#
#        prt_batch_pred = []
#        prt_logits = []
#        n_test_points = batch_size//step_size
#        count = 0
#
#        test_dataset = tf.data.TFRecordDataset(data_path).map(self._parse_function)
#        test_dataset = test_dataset.batch(step_size)
#        test_iter = test_dataset.make_initializable_iterator()
#        self.sess.run(test_iter.initializer)
#        test_handle = self.sess.run(test_iter.string_handle())
#
#        while True:
#            try:
#                self.load()
#                count += 1
#                preds = 0
#                for jj in range(n_test_points):
#                    pred, probs = self.sess.run([self.correct_prediction,
#                                                self.p_classes],
#                                                feed_dict={self.handle: test_handle,
#                                                           self.rate: 1})
#                    self.sess.run(self.train_step,
#                                  feed_dict={self.handle: test_handle,
#                                             self.rate: self.specs['dropout']})
#                    preds += np.mean(pred)
#                    prt_logits.append(probs)
#                prt_batch_pred.append(preds/n_test_points)
#            except tf.errors.OutOfRangeError:
#                print('prt_done: count: %d, acc: %g +\\- %g'
#                      % (count, np.mean(prt_batch_pred), np.std(prt_batch_pred)))
#                break
#        return prt_batch_pred, np.concatenate(prt_logits)

    def plot_cm(self, dataset='validation', class_names=None, normalize=False):

        """
        Plot a confusion matrix

        Parameters
        ----------

        dataset : str {'training', 'validation'}
                which dataset to use for plotting confusion matrix

        class_names : list of str, optional
                if provided subscribes the classes, otherwise class labels
                are used

        normalize : bool
                whether to return percentages (if True) or counts (False)
        """



        if dataset == 'validation':
            feed_dict = {self.handle: self.val_handle, self.rate: 1.}
        elif dataset == 'training':
            feed_dict = {self.handle: self.train_handle, self.rate: 1.}
        elif dataset == 'test':
            feed_dict = {self.handle: self.test_handle, self.rate: 1.}

        y_true, y_pred = self.sess.run([self.y_, self.p_classes],
                                       feed_dict=feed_dict)
        y_pred = np.argmax(y_pred, 1)
        f = plt.figure()
        cm = confusion_matrix(y_true, y_pred)
        title = 'Confusion matrix'
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


class VGG19(Model):
    """
    VGG-19 model.

    References
    ----------

    """
    def __init__(self, Dataset, params, specs):
        super().__init__(Dataset, params)
        self.specs = dict(n_ls=self.params['n_ls'], nonlin=tf.nn.relu,
                          inch=1, padding='SAME', filter_length=(3, 3),
                          domain='2d', stride=1, pooling=1, conv_type='2d')
        self.scope = 'vgg19'

    def build_graph(self):
        X1 = tf.expand_dims(self.X, -1)
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
#
        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *= 2
        vgg3 = vgg_block(4, ConvDSV, self.specs)
        out3 = vgg3(out2)

        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *= 2
        vgg4 = vgg_block(4, ConvDSV, self.specs)
        out4 = vgg4(out3)
#
        self.specs['inch'] = self.specs['n_ls']
        vgg5 = vgg_block(4, ConvDSV, self.specs)
        out5 = vgg5(out4)

#
        fc_1 = Dense(size=4096, nonlin=tf.nn.relu, dropout=self.rate)
        fc_2 = Dense(size=4096, nonlin=tf.nn.relu, dropout=self.rate)
        fc_out = Dense(size=np.prod(self.y_shape), nonlin=tf.identity,
                       dropout=self.rate)
        y_pred = fc_out(fc_2(fc_1(out5)))
        return y_pred


class EEGNet(Model):
    """EEGNet

    Parameters
    ----------
    eegnet_params : dict

    n_ls : int
            number of (temporal) convolution kernrels in the first layer.
            Defaults to 8

    filter_length : int
                    length of temporal filters in the first layer.
                    Defaults to 32

    stride : int
             stride of the average polling layers. Defaults to 4.

    pooling : int
              pooling factor of the average polling layers. Defaults to 4.

    dropout : float
              dropout coefficient

    References
    ----------
    [1] V.J. Lawhern, et al., EEGNet: A compact convolutional neural network
    for EEG-based brain–computer interfaces 10 J. Neural Eng., 15 (5) (2018),
    p. 056013

    [2]  Original  EEGNet implementation by the authors can be found at
    https://github.com/vlawhern/arl-eegmodels
    """

    def build_graph(self):
        self.scope = 'eegnet'

        X1 = tf.expand_dims(self.X, -1)
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
        out22 = tf.nn.dropout(out2, self.rate)

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
        out44 = tf.nn.dropout(out4, self.rate)

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
        out66 = tf.nn.dropout(out6, self.rate)

        out7 = tf.reshape(out66, [-1, np.prod(out66.shape[1:])])
        fc_out = Dense(size=self.y_shape[0], nonlin=tf.identity,
                       dropout=self.rate)
        y_pred = fc_out(out7)
        return y_pred


class LFCNN(Model):

    """
    LF-CNN. Includes basic paramter interpretation options.

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
        """
        Build computational graph using defined placeholder self.X as input

        Returns
        --------
        y_pred : tf.Tensor
                output of the forward pass of the computational graph.
                prediction of the target variable

        """
        self.scope = 'lf-cnn'

        self.demix = DeMixing(n_ls=self.specs['n_ls'], axis=1)

        self.tconv1 = LFTConv(scope="conv", n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])
        self.tconv_out = self.tconv1(self.demix(self.X))

        self.fin_fc = Dense(size=np.prod(self.y_shape),
                            nonlin=tf.identity, dropout=self.rate)

        y_pred = self.fin_fc(self.tconv_out)

        return y_pred



    def compute_patterns(self, data_path=None, output='patterns'):
        """
        Computes spatial patterns from filter weights.

        Required for visualization.
        """

        if data_path:
            self.add_dataset(data_path)
        elif not hasattr(self.dataset, 'test') and not data_path:
            print('Specify data path')
            return

        vis_dict = {self.handle: self.test_handle, self.rate: 1}
        # Spatial stuff
        data, demx = self.sess.run([self.X,self.demix.W], feed_dict=vis_dict)
        #print('data:', data.shape)
#        d = data[0, :, :2, 0].copy()
#        d1 = data[1, :, :2, 0].copy()
#        d2 = data[-1, :, -2:, 0].copy()
#        d3 = data[-1, :, :2, 0].copy()
        data = np.squeeze(data.transpose([1, 2, 3, 0]))
        data = data.reshape([data.shape[0],-1], order='F')
#        print(np.all(d == data[:, 0:2]))
#        print(np.all(d1 == data[:, 101:103]))
#        print(np.all(d2 == data[:, -2:]))
#        print(np.all(d3 == data[:, -101:-99]))

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

        #
        kern, tc_out, out_w = self.sess.run([self.tconv1.filters, self.tconv_out,
                                       self.fin_fc.w], feed_dict=vis_dict)
        print('out_w:', out_w.shape)
        #Temporal conv stuff
        self.filters = np.squeeze(kern)
        self.tc_out = np.squeeze(tc_out)
        self.out_weights  = np.reshape(out_w, [-1, self.specs['n_ls'],
                                        np.prod(self.y_shape)])

        print('demx:', demx.shape, 'kern:', self.filters.shape,
              'tc_out:', self.tc_out.shape, 'out_w:', self.out_weights.shape)

        self.get_output_correlations()

        #order, ts = self.spearman_features(percentile=90)


        self.out_biases = self.sess.run(self.fin_fc.b, feed_dict=vis_dict)
        #self.out_weights += self.out_biases[None,None,:]

#        self.out_weights = np.reshape(self.out_weights,
#                                      [self.specs['n_ls'], -1, np.prod(self.y_shape)])


    def plot_out_weights(self, pat=None, t=None, tmin=-0.1, sorting='weights'):
        """
        Plots the weights of the output layer

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

            im = ax[i].pcolor(times, np.arange(self.specs['n_ls']+1), F, cmap='bone_r', vmin=vmin, vmax=vmax)
            r = []
            if np.any(pat) and np.any(t):
                r = [ptch.Rectangle((times[t], p), width=tstep, height=1, angle=0.0) for p, t in zip(pat[i], t[i])]
                pc = collections.PatchCollection(r, facecolor='red', alpha=.5,
                             edgecolor='red')
                ax[i].add_collection(pc)
        f.colorbar(im, ax=ax[-1])
        plt.show()

    def get_output_correlations(self):
        self.rfocs=[]
        flat_feats = self.tc_out.reshape(self.tc_out.shape[0],-1)
        y_true = self.sess.run(self.y_,
                               feed_dict={self.handle: self.test_handle, self.rate: 1.})
        for y_ in y_true.T:
            rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])
            self.rfocs.append(rfocs.reshape(self.out_weights.shape[:-1]))
        self.rfocs = np.dstack(self.rfocs)

        if np.any(np.isnan(self.rfocs)):
            self.rfocs[np.isnan(self.rfocs)] = 0




#        comp_weights = np.nansum(self.rfocs,1)
#        comp_inds = np.where(comp_weights>=np.percentile(comp_weights,percentile))[0]
#        sorting = np.argsort(comp_weights[comp_inds])[::-1]
#        print(comp_inds[sorting], comp_weights[comp_inds[sorting]])
#        comp_inds = np.where(self.rfocs>=np.percentile(self.rfocs,percentile))
#        sorting = np.argsort(self.rfocs[comp_inds])[::-1]
#        print(self.rfocs[comp_inds[0][sorting],comp_inds[1][sorting]])
#        return np.unique(comp_inds[0][sorting[::-1]])
#        return comp_inds[0][sorting], comp_inds[1][sorting]

#    def pc_features(self):
#        from sklearn.decomposition import pca
#        pc = pca.PCA(3)
#        flat_feats = self.tc_out.reshape(self.tc_out.shape[0],-1)
#        pcf = pc.fit_transform(flat_feats)
##        y_true = self.sess.run(self.y_,
##                               feed_dict={self.handle: self.test_handle, self.rate: 1.})
##
##        rfocs = [spearmanr(y_true, f)[0] for f in flat_feats.T]
##        self.rfocs = np.array(rfocs).reshape(self.tc_out.shape[1:]).T
##        if np.any(np.isnan(self.rfocs)):
##            self.rfocs[np.isnan(self.rfocs)] = 0
##
##        comp_weights = np.nansum(self.rfocs,1)
##        comp_inds = np.where(comp_weights>=np.percentile(comp_weights,percentile))[0]
##        sorting = np.argsort(comp_weights[comp_inds])[::-1]
##        print(comp_inds[sorting], comp_weights[comp_inds[sorting]])
##        comp_inds = np.where(self.rfocs>=np.percentile(self.rfocs,99))
##        sorting = np.argsort(self.rfocs[comp_inds])[::-1]
##        print(self.rfocs[comp_inds[0][sorting],comp_inds[1][sorting]])
##
##        return np.unique(comp_inds[0][sorting[::-1]])
#        return comp_inds[sorting]

    def plot_waveforms(self, tmin=-.1):
        f, ax = plt.subplots(2,2)
        nt = self.dataset.h_params['n_t']
        self.waveforms = np.squeeze(self.lat_tcs.reshape([self.specs['n_ls'],  -1, nt]).mean(1))
        tstep = 1/float(self.fs)
        times = tmin+tstep*np.arange(nt)
        [ax[0,0].plot(times,wf+1e-1*i) for i, wf in enumerate(self.waveforms) if i not in self.uorder]
        ax[0,0].plot(times,self.waveforms[self.uorder[0]]+1e-1*self.uorder[0],'k.')
        [ax[0,1].plot(ff+.1*i) for i, ff in enumerate(self.filters.T)]
        ax[0,1].plot(self.filters.T[self.uorder[0]]+1e-1*self.uorder[0],'k.')
        conv = np.convolve(self.filters.T[self.uorder[0]],self.waveforms[self.uorder[0]],mode='same')
        vmin = conv.min()
        vmax = conv.max()
        ax[1,0].plot(times, conv)
        bias = self.sess.run(self.tconv1.b)
        ax[1,0].hlines(-1*bias[self.uorder[0]],times[0], times[-1], linestyle='--')
        strides = np.arange(0,len(times) + 1,self.specs['stride'] )[1:]
        pool_bins = np.arange(0,len(times) + 1,self.specs['pooling'] )[1:]
        ax[1,0].vlines(times[strides], vmin, vmax, linestyle='--', color='c')
        ax[1,0].vlines(times[pool_bins], vmin, vmax, linestyle='--', color='m')
        if self.out_weights.shape[-1] == 1:
            ax[1,1].pcolor(self.F)
            ax[1,1].hlines(self.uorder[0]+.5,0,self.F.shape[1], color='r')
        else:
            ax[1,1].plot(self.out_weights[:,self.uorder[0],:], 'k*')
        #plt.show()


#    def plot_filters(self):
#        plt.figure()


        #plt.figure()
        #[plt.plot(np.convolve(ff,self.out_weights[...,i,0])+.1*i) for i, ff in enumerate(self.filters.T)]

#        plt.show()
#        f2 = plt.figure()




    def sorting(self, sorting='best'):
        order = []
        ts = []

        if sorting == 'l2':
            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=2))
            contribution = self.out_weights[...,i].T
            ts = None
#        elif sorting == 'l1':
#            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=1))
        #elif sorting == '95':

        elif sorting == 'best_spatial':
            #nfilt = np.prod(self.y_shape)
            for i in range(np.prod(self.y_shape)):
                contribution = self.out_weights[...,i].T * self.rfocs[...,i].T
                pat = np.argmax(contribution.sum(-1))
                order.append(np.tile(pat,contribution.shape[1]))
                ts.append(np.arange(contribution.shape[-1]))

        elif sorting == 'best':
            #nfilt = np.prod(self.y_shape)

            #_ = self.spearman_features()
            for i in range(np.prod(self.y_shape)):
                contribution = np.abs(self.out_weights[...,i].T * self.rfocs[...,i].T)
                pat, t = np.where(contribution==np.max(contribution))
                order.append(pat)
                ts.append(t)

        elif sorting == 'best_weight':
            for i in range(np.prod(self.y_shape)):
                contribution = self.out_weights[...,i].T
                pat, t = np.where(contribution==np.max(contribution))
                order.append(pat)
                ts.append(t)

        elif sorting == 'best_spear':
            for i in range(np.prod(self.y_shape)):
                contribution = self.rfocs[...,i].T
                pat, t = np.where(contribution==np.max(contribution))
                order.append(pat)
                ts.append(t)

        elif isinstance(sorting,float):
            #nfilt = np.prod(self.y_shape)

            #_ = self.spearman_features()
            for i in range(np.prod(self.y_shape)):
                contribution = self.out_weights[...,i].T * self.rfocs[...,i].T
                pat, t = np.where(contribution>=np.percentile(contribution, sorting))
                order.append(pat)
                ts.append(t)

            #self.plot_out_weights(pat=order, t=ts)
        else:
            print('ELSE!')
            order = np.arange(self.specs['n_ls'])
            contribution = self.out_weights[...,i].T

        self.F = contribution
        order = np.array(order)
        ts = np.array(ts)
        #print(order, ts)
        return order, ts

    def plot_patterns(self, sensor_layout=None, sorting='l2', percentile=90,
                      spectra=False, scale=False, names=False):
        """
        Plot informative spatial activations patterns for each class of stimuli

        Parameters
        ----------

        sensor_layout : str or mne.channels.Layout
            sensor layout. See mne.channels.read_layout for details

        sorting : str, optional

        spectra : bool, optional
            If True will also plot frequency responses of the associated
            temporal filters. Defaults to False

        fs : float
            sampling frequency

        scale : bool, otional
            If True will min-max scale the output. Defaults to False

        names : list of str, optional
            Class names

        Returns
        -------

        Figure

        """
        if sensor_layout:
            lo = channels.read_layout(sensor_layout)
            info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
            self.fake_evoked = evoked.EvokedArray(self.patterns, info)

        order, ts = self.sorting(sorting)
        #print(order, ts)
        uorder = uniquify(order.ravel())
        self.uorder = uorder
        #print(uorder)
        #uorder = np.array(uorder[ind])

        if sensor_layout:
            self.fake_evoked.data[:, :len(uorder)] = self.fake_evoked.data[:, uorder]
            self.fake_evoked.crop(tmax=float(len(uorder)))
            if scale:
                self.fake_evoked.data[:, :len(uorder)] /= self.fake_evoked.data[:, :len(uorder)].std(0)
#            self.fake_evoked.data[:, len(uorder):] *= 0

        nfilt = max(np.prod(self.y_shape), 8)
        nrows = max(1, len(uorder)//nfilt)
        ncols = min(nfilt, len(uorder))

        f, ax = plt.subplots(nrows, ncols, sharey=True)
        f.set_size_inches([16,9])
        ax = np.atleast_2d(ax)
        #zz = 0
        for i in range(nrows):
                self.fake_evoked.plot_topomap(times=np.arange(i*ncols,  (i+1)*ncols, 1.),
                                              axes=ax[i], colorbar=False,
                                              vmax=np.percentile(self.fake_evoked.data[:, :len(uorder)], 95),
                                              scalings=1, time_format='output # %g',
                                              title='Informative patterns ('+str(sorting)+')')

        if np.any(ts):
            self.plot_out_weights(pat=order, t=ts, sorting=sorting)
        else:
            self.plot_out_weights()

        return

    def plot_spectra(self, fs=None, sorting='l2',  norm_spectra=None, percentile=99):
        if norm_spectra:
            from scipy.signal import welch
            from spectrum import aryule
            if norm_spectra == 'welch' and not hasattr(self, 'd_psds'):
                    f, psd = welch(self.lat_tcs,fs=1000,nperseg=256)
                    self.d_psds = psd[:,:-1]
            elif 'ar' in norm_spectra and not hasattr(self, 'ar'):
                ar = []
                for i, ltc in enumerate(self.lat_tcs):
                    coef, _, _ = aryule(ltc, self.specs['filter_length'])
                    ar.append(coef[None,:])
                self.ar = np.concatenate(ar)
        if fs:
            self.fs = fs
        elif self.dataset.h_params['fs']:
            self.fs = self.dataset.h_params['fs']
        else:
            print('Sampling frequency not specified, setting to 1.')
            self.fs = 1.

        order, ts = self.sorting(sorting)
        print(order)
        uorder = uniquify(order.ravel())
        #self.uorder = uorder
        print(uorder)
        out_filters = self.filters[:, uorder]


        nfilt = max(np.prod(self.y_shape), 8)
        nrows = max(1, len(uorder)//nfilt)
        ncols = min(nfilt, len(uorder))

        f, ax = plt.subplots(nrows, ncols, sharey=True)
        f.set_size_inches([16,9])
        ax = np.atleast_2d(ax)
        for i in range(nrows):
                for jj, flt in enumerate(out_filters[:, i*ncols:(i+1)*ncols].T):

                    if norm_spectra == 'ar':
                        w, h = freqz(flt, self.ar[jj], worN=128)
                    elif norm_spectra == 'welch':
                        w, h = freqz(flt, 1,worN=128)
                        h = np.abs(h) * np.sqrt(self.d_psds[jj])
                        h /= h.max()
                    elif norm_spectra == 'plot_ar':
                        w0, h0 = freqz(flt, 1,worN=128)
                        w, h = freqz(self.ar[jj],1,worN=128)
                        ax[i, jj].plot(w/np.pi*self.fs/2, np.abs(h0))
                        print(h0.shape, h.shape, w.shape)
                    else:
                        w, h = freqz(flt, 1, worN=128)
                    #else:
                     #   density = np.abs(h)
                    ax[i, jj].plot(w/np.pi*self.fs/2, np.abs(h).T)






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

    def build_graph(self):
        self.scope = 'var-cnn'
        self.demix = DeMixing(n_ls=self.specs['n_ls'])

        self.tconv1 = VARConv(scope="conv", n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        self.fin_fc = Dense(size=np.prod(self.y_shape),
                            nonlin=tf.identity, dropout=self.rate)

        y_pred = self.fin_fc(self.tconv1(self.demix(self.X)))

        return y_pred

class VARCNNR(Model):

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

    def build_graph(self):
        self.scope = 'var-cnn'
        self.demix = DeMixing(n_ls=self.specs['n_ls'],axis=1)

        tconv1 = VARConv(scope="conv", n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        fin_fc = Dense(size=np.prod(self.y_shape),
                       nonlin=tf.identity, dropout=self.rate)

        y_pred = fin_fc(tconv1(self.demix(self.X)))
        return y_pred


class LFLSTM(LFCNN):

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

    def build_graph(self):
        self.scope = 'var-cnn'

        self.demix = DeMixing(n_ls=self.specs['n_ls'],axis=2)
        dmx = self.demix(self.X)
        #print('dmx-out:', dmx.shape)
        dmx = tf.reshape(dmx, [-1, self.dataset.h_params['n_t'],self.specs['n_ls']])
        dmx = tf.expand_dims(dmx,-1)
        print('dmx-sqout:', dmx.shape)

        self.tconv1 = LFTConv(scope="conv", #n_ls=self.specs['n_ls'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
                              stride=self.specs['stride'],
                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        features = self.tconv1(dmx)
        print('features:', features.shape)
        features = tf.reshape(features,[-1, self.dataset.h_params['n_seq'], tf.multiply(features.shape[1], features.shape[2])])
        #features = tf.expand_dims(features, 0)
        print('flat features:', features.shape)
        self.lstm = LSTMv1(scope="lstm-weights",
                           size=self.specs['n_ls'],
                           dropout=self.rate,
                           nonlin=tf.tanh,
                           unit_forget_bias=True,
                           return_sequences=True,
                           unroll=False)


#        self.lstm = LSTMv1(scope="lstm",
#                   size=self.specs['n_ls'],
#                   dropout=self.rate,
#                   nonlin=tf.nn.tanh,
#                   unit_forget_bias=True,
#                   return_sequences=True,
#                   #kernel_initializer='glorot_uniform',
#                   #recurrent_initializer='orthogonal',
#                   #bias_initializer='zeros',
#                   #kernel_regularizer=tf.keras.regularizers.l1(self.optimizer.params['l1_lambda']),
#                   #recurrent_regularizer=tf.keras.regularizers.l1(self.optimizer.params['l1_lambda']),
#                   #bias_regularizer=tf.keras.regularizers.l1(self.optimizer.params['l1_lambda']),
#                   #activity_regularizer=tf.keras.regularizers.l1(self.optimizer.params['l1_lambda']),
#                   #kernel_constraint=tf.keras.constraints.max_norm(1.),
#                   #recurrent_constraint=tf.keras.constraints.max_norm(1.),
#                   #bias_constraint=tf.keras.constraints.max_norm(1.),
#                   #recurrent_dropout=.5,
#                   return_state=False, go_backwards=False, stateful=False,
#                   unroll=False)


        lstm_out = self.lstm(features)
        print('lstm_out:',lstm_out.shape)
        #flat_features = tf.reshape(features, [1,None,tf.multiply(features.sh)]
        #self.lstm = tf.keras.layers.LSTMCell(units=50)
        self.fin_fc = DeMixing(n_ls=np.prod(self.y_shape),
                            nonlin=tf.tanh, axis=-1)
        y_pred = self.fin_fc(lstm_out)
        #y_pred = tf.expand_dims(y_pred,0)
        print(y_pred)
        return y_pred

def uniquify(seq):
    # order preserving
    un = []
    [un.append(i) for i in seq if not un.count(i)]
    return  un