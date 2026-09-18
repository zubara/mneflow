# -*- coding: utf-8 -*-
"""
Define mneflow.models.Model parent class and the implemented models as
its subclasses. Implemented models inherit basic methods from the
parent class.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""


import tensorflow as tf

import numpy as np

from typing import Callable
from mne import channels, evoked, create_info, Info
from mne.filter import filter_data


from scipy.stats import spearmanr, pearsonr
from scipy.signal import welch

from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .layers import LFTConv, VARConv, DeMixing, FullyConnected, TempPooling
from tf.keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D
from tf.keras.layers import Flatten, Dropout, BatchNormalization
from tf.keras.initializers import Constant
from tf.keras import regularizers as k_reg, constraints, layers

from .layers import LSTM
import csv
import os
from .data import Dataset
from .utils import regression_metrics, _onehot, r2_score
from collections import defaultdict


def uniquify(seq):
    """Return the elements of a sequence in order, with duplicates removed.

    Parameters
    ----------
    seq : sequence
        Input sequence (of hashable-comparable elements).

    Returns
    -------
    un : list
        Elements of ``seq``, keeping only the first occurrence of
        each, in their original order.

    """
    un = []
    [un.append(i) for i in seq if not un.count(i)]
    return un


# ----- Base model -----
#@tf.keras.utils.register_keras_serializable(package="mneflow")
class BaseModel():
    """Parent class for all MNEflow models.

    Provides fast and memory-efficient data handling and simplified API.
    Custom models can be built by overriding _build_graph and
    _set_optimizer methods.
    """

    def __init__(self, meta=None, dataset=None, specs_prefix=False):
        """
        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object. ``meta.model_specs`` provides the
            model-specific hyperparameters and, once this
            constructor runs, is updated with a ``'model_path'`` key
            - path for saving a trained model. See `Model` subclass
            definitions for details on the expected hyperparameters;
            unless otherwise specified, default hyperparameters are
            used for each implemented model.

        dataset : mneflow.Dataset, optional
            `Dataset` object to use. Defaults to None, in which case
            a new ``Dataset`` is built from ``meta``.

        specs_prefix : bool, optional
            Whether to derive ``self.specs_prefix`` (used in the
            saved model/weights file names) from the non-default
            entries of ``meta.model_specs``. Defaults to False (empty
            prefix).

        """
        self.specs = meta.model_specs
        meta.model_specs['model_path'] = os.path.join(meta.data['path'],
                                                      'models')
        self.current_fold = 0

        self.meta = meta
        self.model_path = meta.model_specs['model_path'] 
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


        if dataset:
            self.dataset = dataset
        elif not dataset and meta:
            #print(meta.data)
            self.dataset = Dataset(meta, **meta.data)
        else:
            print("Provide Dataset or Metadata file")
        if self.dataset.h_params['channel_subset'] is None:
            self.input_shape = (self.dataset.h_params['n_seq'],
                                self.dataset.h_params['n_t'],
                                self.dataset.h_params['n_ch'])
        else:
            self.input_shape = (self.dataset.h_params['n_seq'],
                                self.dataset.h_params['n_t'],
                                len(self.dataset.h_params['channel_subset']))
        self.y_shape = self.dataset.y_shape
        self.out_dim = np.prod(self.y_shape)
        self.inputs = layers.Input(shape=(self.input_shape))
        #self.trained = False
        self.y_pred = self.build_graph()
        self.log = dict()
        self.cm = np.zeros([self.y_shape[-1], self.y_shape[-1]])
        self.cv_patterns = defaultdict(dict)
        self.cv_weights = defaultdict(list)
        if not hasattr(self, 'scope'):
            self.scope = 'basemodel'

        if specs_prefix:
           self.specs_prefix = '_'.join([str(v).replace('.', '-') for k,v in self.specs.items() if k not in ['nonlin', 'model_path', 'l1_scope', 'l2_scope', 'unitnorm_scope', 'scope']])
        else:
            self.specs_prefix = ''
        self.model_name = "_".join([self.scope,
                                    meta.data['data_id']])

    

    def build(self, optimizer="adam",
              loss=None,
              metrics=None, mapping=None,
              learn_rate=3e-4):
        """Compile a model.

        Parameters
        ----------
        optimizer : str, tf.optimizers.Optimizer
            Deafults to "adam"

        loss : str, tf.keras.losses.Loss
            Defaults to MSE in target_type is "float" and
            "softmax_crossentropy" if "target_type" is int

        metrics : str, list of str, tf.keras.metrics.Metric
            Defaults to RMSE in target_type is "float" and
                "categorical_accuracy" if "target_type" is int

        learn_rate : float
            Learning rate, defaults to 3e-4

        mapping : str

        """
        # Initialize computational graph
        if mapping:
            map_fun = tf.keras.activations.get(mapping)
            self.y_pred = map_fun(self.y_pred)

        self.km = tf.keras.Model(inputs=self.inputs, outputs=self.y_pred)

        params = {"optimizer": tf.optimizers.get(optimizer).from_config(
                                            {"learning_rate":learn_rate})}

        if loss:
            params["loss"] = tf.keras.losses.get(loss)
            loss_name = loss

        if metrics:
            if not isinstance(metrics, list):
                metrics = [metrics]
            params["metrics"] = [tf.keras.metrics.get(metric) for metric in metrics]

       # Initialize optimizer
        if self.dataset.h_params["target_type"] in ['float', 'signal']:
            params.setdefault("loss", tf.keras.losses.MeanSquaredError(name='MSE'))

            params.setdefault("metrics", [tf.keras.metrics.R2Score(name="R2")])

        elif self.dataset.h_params["target_type"] in ['int']:
            params.setdefault("loss", tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                                   name='Cat_CE'))
            params.setdefault("metrics", [tf.keras.metrics.CategoricalAccuracy(name="Cat_Acc")])

        self.km.compile(optimizer=params["optimizer"],
                        loss=params["loss"],
                        metrics=params["metrics"])

        if not loss and self.dataset.h_params["target_type"] in ['float', 'signal']:
            loss_name = 'MSE'
        elif not loss:
            loss_name = 'Cat_CE'
        else:
            loss_name = params['loss'].name
        _ = params.pop('loss')
        metrics = params.pop('metrics')
        metric_names = ':'.join([m.name for m in metrics])

        param_names = {k: v.name for k,v in params.items()}
        param_names['metrics'] = metric_names
        param_names['loss'] = loss_name
        param_names['learn_rate'] = learn_rate
        param_names['trained'] = False
        self.meta.update(train_params=param_names)

        self.km.save_weights(os.path.join(self.model_path,
                                          ''.join([self.model_name,
                                                   self.specs_prefix,
                                                   '_init.weights.h5'])))


        print('Input shape:', self.input_shape)
        print('y_pred:', self.y_pred.shape)
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

        flat = Flatten()(self.inputs)
        self.fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                        specs=self.specs)
        y_pred = self.fc(flat)
        return y_pred


    def train(self, n_epochs=10, eval_step=None, min_delta=1e-6,
              early_stopping=3, mode='single_fold',
              collect_patterns=False, class_weights=None,
              noisy_labels=False, noise_std=.1, shapley_order=1, fold=0, 
              compute_pvalues=False, store_fold_predictions=False) :

        """
        Train a model

        Parameters
        -----------

        n_epochs : int
            Maximum number of training eopchs.

        eval_step : int, None
            iterations per epoch. If None each epoch passes the training set
            exactly once

        early_stopping : int
            Patience parameter for early stopping. Specifies the number
            of epochs's during which validation cost is allowed to
            rise before training stops.

        min_delta : float, optional
            Convergence threshold for validation cost during training.
            Defaults to 1e-6.

        mode : str, optional
            can be 'single_fold', 'cv', 'loso'. Defaults to 'single_fold'

        collect_patterns : bool
            Whether to compute and store patterns after training each fold.

        class_weights : None, dict
            Whether to apply cutom wegihts fro each class

        noisy_labels : bool, optional
            Train model with addition gaussinan noise to labels. (Experimental)
            Does not work with class_weights

        noise_std : float, optional
            Standard deviation of the noise added to labels. (Experimental)
            Does not work with class_weights

        shapley_order : int, optional
            Passed through to :meth:`collect_patterns` (only used by
            models that define it, e.g. LFCNN). Defaults to 1.

        fold : int, optional
            Fold index to start (or, with ``mode='single_fold'``,
            train) from. Defaults to 0.

        compute_pvalues : bool, optional
            Whether to compute a permutation p-value (see
            :meth:`permutation_p_value`) for each fold. Defaults to
            False.

        store_fold_predictions : bool, optional
            Whether to store each fold's validation predictions
            (``y_true``, ``y_pred``, and sample indices) in
            ``self.cv_predictions``. Defaults to False.

        Returns
        -------
        None

        """


        if not eval_step:
            train_size = self.dataset.h_params['train_size']
            eval_step = train_size // self.dataset.h_params['train_batch'] + 1

        train_params = dict(n_epochs=n_epochs,
                            eval_step=eval_step,
                            early_stopping=early_stopping,
                            mode=mode,
                            min_delta=min_delta)

        self.meta.update(train_params=train_params)


        rmss = defaultdict(list)

        self.cv_losses = []
        self.cv_metrics = []
        self.cv_test_losses = []
        self.cv_test_metrics = []
        self.cv_metric_pvalues = []
        self.cv_predictions = []
        cv_pvals = []

        if class_weights:
            multiplier = 1. / min(class_weights.values())
            class_weights = {k:v*multiplier for k,v in class_weights.items()}

        else:
            class_weights = None
            print("Class weights: ", class_weights)

        if mode == 'single_fold':
            n_folds = 1 + fold
        elif mode == 'cv':
            n_folds = len(self.dataset.h_params['folds'][0])
            print("Running cross-validation with {} folds".format(n_folds))
        elif mode == "loso":
            n_folds = len(self.dataset.h_params['train_paths'])

        if collect_patterns and self.scope=='lfcnn':
            self.init_pattern_struct(n_folds, freqs=None)

        if fold:
            self.current_fold = fold

        for jj in range(self.current_fold, min(self.current_fold + n_folds, n_folds)):
            self.current_fold = jj
            print("Running {} fold: {}".format(mode, self.current_fold))

            if mode == "loso":
                test_subj = self.dataset.h_params['train_paths'][jj]
                train_subjs = self.dataset.h_params['train_paths'].copy()
                train_subjs.pop(jj)

                train, val = self.dataset._build_dataset(train_subjs,
                                                   train_batch=self.dataset.training_batch,
                                                   test_batch=self.dataset.validation_batch,
                                                   split=True, val_fold_ind=0)

            else:

                train, val = self.dataset._build_dataset(self.dataset.h_params['train_paths'],
                                                   train_batch=self.dataset.training_batch,
                                                   test_batch=self.dataset.validation_batch,
                                                   split=True, val_fold_ind=self.current_fold)
            if not noisy_labels:
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=self.meta.train_params['min_delta'],
                                                              patience=self.meta.train_params['early_stopping'],
                                                              restore_best_weights=True)
                stop_early.best = np.inf
                self.t_hist = self.km.fit(train,
                                   validation_data=val,
                                   epochs=self.meta.train_params['n_epochs'],
                                   steps_per_epoch=self.meta.train_params['eval_step'],
                                   shuffle=True,
                                   validation_steps=self.dataset.validation_steps,
                                   callbacks=[stop_early], verbose=2,
                                   class_weight=class_weights)
            else:
                model_path = os.path.join(self.model_path,
                                          ''.join([self.model_name,
                                                   self.specs_prefix]))
                trainer = NoisyTrainer(self.km,
                                       model_path = model_path,
                                       noise_std = noise_std,  # Noise standard deviation
                                       patience = self.meta.train_params['early_stopping'],     # Stop if no improvement for 5 epochs
                                       min_delta = self.meta.train_params['min_delta'] # Minimum change to count as improvement
                                       )

                self.t_hist = trainer.train(train, val,
                                            epochs=self.meta.train_params['n_epochs'],
                                            eval_step=self.meta.train_params['eval_step']
                                            )

            self.meta.train_params.update({"trained":True})

            v_loss, v_metric = self.evaluate(val)
            self.cv_losses.append(v_loss)
            self.cv_metrics.append(v_metric)

            if compute_pvalues:
                cv_pvals.append(self.permutation_p_value(n_perm=1000))
                print("permutation_p_value : {}".format(cv_pvals[-1]))

            if mode == 'loso':
                print("Creating loso test DS")
                test = self.dataset._build_dataset(test_subj,
                                                   test_batch=None,
                                                   split=False,
                                                   repeat=False)
            elif len(self.dataset.h_params['test_paths']):
                test = self.dataset._build_dataset(self.dataset.h_params['test_paths'],
                                                   test_batch=None,
                                                   split=False,
                                                   repeat=False)
            else:
                test = None

            if test:

                t_loss, t_metric = self.evaluate(test)
                self.cv_test_losses.append(t_loss)
                self.cv_test_metrics.append(t_metric)


            y_true, y_pred = self.predict(val,
                                          n_batches=self.dataset.validation_steps)


            if self.dataset.h_params['target_type'] == 'float':
                rms = regression_metrics(y_true, y_pred)
                for k,v in rms.items():
                    rmss[k].append(v)
                print("Validation set: Corr =", rms['cc'], " R2 =", rms['r2'])

            else:
                self.cm += self._confusion_matrix(y_true, y_pred)
                rms = None

            if collect_patterns and hasattr(self, 'collect_patterns'):
                self.collect_patterns(fold=self.current_fold, n_folds=n_folds,
                                      n_comp=int(collect_patterns),
                                      shapley_order=shapley_order)
            if store_fold_predictions:
                self.cv_predictions.append({'y_true':y_true,
                                            'y_pred':y_pred,
                                            'inds':self.dataset.val_inds})
                

            if jj < n_folds - 1:
                self.km.load_weights(os.path.join(self.model_path,
                                                  ''.join([self.model_name,
                                                           self.specs_prefix,
                                                           '_init.weights.h5'])),
                                     skip_mismatch=True)
                self.shuffle_weights()


            else:
                print("Not shuffling the weights for the last fold")

            print("""Fold: {} Validation performance:\n
                  Loss: {:.4f},
                  Metric: {:.4f}""".format(jj, v_loss, v_metric))
            if test:
                print("""Test set performance:\n
                      Loss: {:.4f},
                      Metric: {:.4f}""".format(t_loss, t_metric))

        metrics = self.cv_metrics
        losses = self.cv_losses

        if self.dataset.h_params['target_type'] == 'float':
            rms = {k:np.mean(v) for k, v in rmss.items()}
            rms.update({k + '_std':np.std(v) for k, v in rmss.items()})
            rms['r2_folds'] = rmss['r2']
            rms['cc_folds'] = rmss['cc']
            print("""Validation set:
                  Corr : {:.3f} +/- {:.3f}.
                  R2: {:.3f} +/- {:.3f}""".format(
                  rms['cc'], rms['cc_std'], rms['r2'], rms['r2_std']))
            self.meta.update(results=rms)
        else:
            rms = None

        print("""{} with {} fold(s) completed. \n
              Validation Performance:
              Loss: {:.4f} +/- {:.4f}.
              Metric: {:.4f} +/- {:.4f}"""
              .format(mode, n_folds,
                      np.mean(self.cv_losses), np.std(self.cv_losses),
                      np.mean(self.cv_metrics), np.std(self.cv_metrics)))

        if len(self.dataset.h_params['test_paths']) > 0 or mode == 'loso':
            print("""\n
              Test Performance:
              Loss: {:.4f} +/- {:.4f}.
              Metric: {:.4f} +/- {:.4f}"""
              .format(np.mean(self.cv_test_losses),
                      np.std(self.cv_test_losses),
                      np.mean(self.cv_test_metrics),
                      np.std(self.cv_test_metrics)))
        if compute_pvalues:
            self.meta.update(results={'cv_pvals':cv_pvals})

        self.meta.train_params.update({"trained":True})
        self.update_log(rms=rms, prefix=mode)
        self.save()
        #return self.cv_losses, self.cv_metrics


    def prune_weights(self, increase_regularization=3.):
        """Continue training with increased L1/L2 regularization to prune weights.

        Multiplies ``self.specs['l1_lambda']`` and
        ``self.specs['l2_lambda']`` by ``increase_regularization``
        and fits the compiled model for up to 30 more epochs (with
        early stopping on validation loss), storing the resulting
        history in ``self.t_hist_p``.

        Parameters
        ----------
        increase_regularization : float, optional
            Factor by which to multiply the L1 and L2 regularization
            strengths. Defaults to 3.

        Returns
        -------
        None

        """
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=1e-6,
                                                      patience=10,
                                                      restore_best_weights=True)
        self.rate = 0
        self.specs["l1_lambda"] *= increase_regularization
        self.specs["l2_lambda"] *= increase_regularization
        print('Pruning weights')
        self.t_hist_p = self.km.fit(self.dataset.train,
                               validation_data=self.dataset.val,
                               epochs=30, steps_per_epoch=self.meta.train_params['eval_step'],
                               shuffle=True,
                               validation_steps=self.dataset.validation_steps,
                               callbacks=[stop_early], verbose=2)

    def shuffle_weights(self):
        """Randomly permute all of the model's weight tensors in place.

        Used between cross-validation folds to re-randomize the
        model before re-fitting from its initial weights.

        Returns
        -------
        None

        """
        print("Re-shuffling weights between folds")
        weights = self.km.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self.km.set_weights(weights)


    def plot_hist(self):
        """Plot loss history during training."""
        plt.plot(self.t_hist.history['loss'])
        plt.plot(self.t_hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def _confusion_matrix(self, y_true, y_pred):
        """Compute an unnormalized confusion matrix.

        Parameters
        ----------
        y_true : ndarray, shape (n_samples, n_classes)
            One-hot encoded ground-truth labels.

        y_pred : ndarray, shape (n_samples, n_classes)
            Predicted class scores/probabilities.

        Returns
        -------
        cm : ndarray, shape (n_classes, n_classes)
            Unnormalized confusion matrix, ``one_hot(argmax(y_pred)).T
            @ y_true``.

        """
        y_p = _onehot(np.argmax(y_pred,1), n_classes=self.y_shape[-1])
        cm = np.dot(y_p.T, y_true)
        return cm

    def update_results(self):
        """Add training/validation/test results to ``self.meta.results``.

        Aggregates cross-validation metrics/losses (from
        ``self.cv_metrics``/``self.cv_losses`` if this model instance
        trained them, otherwise from ``self.meta.results``), computes
        train-set loss/metric via :meth:`evaluate`, and stores
        everything (plus the confusion matrix ``self.cm``) into
        ``self.meta.results`` via ``self.meta.update``.

        Returns
        -------
        None

        """
        results = dict()
        if hasattr(self, 'cv_metrics'):
            results['v_metric'] = np.mean(self.cv_metrics)
            results['v_loss'] = np.mean(self.cv_losses)
            results['cv_metrics'] = self.cv_metrics
            results['cv_losses'] = self.cv_losses
            results['cv_metric_pvalues'] = self.cv_metric_pvalues
            if len(self.cv_test_losses) > 0:
                t_loss = np.mean(self.cv_test_losses)
                t_metric = np.mean(self.cv_test_metrics)
                if self.dataset.h_params['target_type'] == 'float':
                    y_true, y_pred = self.predict(self.dataset.h_params['test_paths'])
                    rms_test = regression_metrics(y_true, y_pred)
                    print("Test set: Corr =", rms_test['cc'], "R2 =", rms_test['r2'])
                    results.update({'test_'+k:v for k,v in rms_test.items()})
                
                results['test_metric'] = t_metric
                results['test_loss'] = t_loss
                results['test_metrics'] = self.cv_test_metrics
                results['test_losses'] = self.cv_test_losses
            else:
                results['test_metric'] = "NA"
                results['test_loss'] = "NA"
                results['test_metrics'] = "NA"
                results['test_losses'] = "NA"
                
        else:
            results['v_metric'] = np.mean(self.meta.results['cv_metrics'])
            results['v_loss'] = np.mean(self.meta.results['cv_losses'])
            
            results['cv_metrics'] = self.meta.results['cv_metrics']
            results['cv_losses'] = self.meta.results['cv_losses']
            results['cv_metric_pvalues'] = self.meta.results['cv_metric_pvalues']
           
            # results['test_metric'] = self.meta.results['t_metric']
            # results['test_loss'] = self.meta.results['test_loss']
            # results['test_metrics'] = self.meta.results['test_metrics']
            # results['test_losses'] = self.meta.results['test_losses']

        tr_loss, tr_metric = self.evaluate(self.dataset.train)
        results['tr_metric'] = tr_metric
        results['tr_loss'] = tr_loss
        results['cm'] = self.cm

        self.meta.update(results=results)

    def permutation_p_value(self, dataset=None, n_perm=10000):
        """Estimate a permutation-test p-value for the model's performance.

        Compares the observed evaluation metric to two null
        distributions built by shuffling the (mean-centered)
        true/predicted targets across samples: ``perm_metrics``
        (predictions held fixed, targets shuffled and compared to
        predictions) and ``perm_metrics2`` (predictions replaced by
        shuffled targets). Also plots histograms of both null
        distributions.

        Parameters
        ----------
        dataset : tf.data.Dataset, optional
            Dataset to evaluate on. Defaults to None, in which case
            ``self.dataset.val`` is used.

        n_perm : int, optional
            Number of permutations to draw. Defaults to 10000.

        Returns
        -------
        metric_pvalue : float
            Fraction of ``perm_metrics`` exceeding the observed
            metric.

        metric_pvalue2 : float
            Fraction of ``perm_metrics2`` exceeding the observed
            metric.

        """
        perm_metrics2 = []
        perm_metrics = []
        if self.meta.data['target_type'] == 'float':
            criterion = r2_score
        else:
            criterion = tf.keras.metrics.categorical_accuracy

        if not dataset:
            dataset = self.dataset.val
        y_true, y_pred_obs = self.predict(dataset)
        y_true -= y_true.mean(0)
        y_pred_obs -= y_pred_obs.mean(0)
        obs_loss, obs_metric = self.evaluate(dataset)
        n = y_true.shape[0]
        for i in range(n_perm):
            shuffle = np.random.permutation(n)
            y_surrogate = y_true[shuffle, :]
            perm_metrics.append(criterion(y_surrogate, y_pred_obs)[0])
            perm_metrics2.append(criterion(y_true, y_surrogate)[0])
        plt.hist(perm_metrics, 100)
        plt.hist(perm_metrics2, 100)
        print(min(perm_metrics), max(perm_metrics))
        print(min(perm_metrics2), max(perm_metrics2))
        print("criterion, corresponding to p = 0.005 : {:.4f}".format(np.percentile(perm_metrics, 99.5)))
        print("criterion, corresponding to p2 = 0.005 : {:.4f}".format(np.percentile(perm_metrics2, 99.5)))
        metric_pvalue = np.sum(np.array(perm_metrics) > obs_metric)/n_perm
        metric_pvalue2 = np.sum(np.array(perm_metrics2) > obs_metric)/n_perm
        print("Metric p-value={:.4f}".format(metric_pvalue))
        print("Metric p-value2={:.4f}".format(metric_pvalue2))
        return metric_pvalue, metric_pvalue2

    def update_log(self, rms=None, prefix=''):
        """Logs experiment to self.model_path + self.scope + '_log.csv'.

        If the file exists, appends a line to the existing file.

        Parameters
        ----------
        rms : dict, optional
            Currently unused directly (regression metrics are instead
            read from ``self.meta.results``); accepted for interface
            consistency with callers such as :meth:`train`. Defaults
            to None.

        prefix : str, optional
            Currently unused. Defaults to ''.

        Returns
        -------
        None

        """
        savepath = os.path.join(self.model_path, self.scope + '_log.csv')
        appending = os.path.exists(savepath)


        #make default header
        log_header = ['data_id',
                      'train metric',	'validation metric',	'test metric', #metrics
                      'train loss',	'validation loss',	'test loss'] #losses
        
        training_params = ['n_epochs',	'eval_step',
                           'early_stopping',	'min_delta', 'learn_rate',
                           'mode', 'optimizer', 'loss', 'metrics']
        
        model_specs = ['n_latent', #model
                       'nonlin',	'stride',
                       'filter_length',
                      #regulatization
                      'dropout', 'l1_lambda']
        
        data_info = [ 'path',	'data_path',
                      'target_type', 'input_type',
                      'train_size',	'val_size', 'test_size', 'n_folds',	 #dataset size
                      'train_batch',
                      'n_seq',	'n_t',	'n_ch',	'y_shape', # shapes
                      'fs'] # optional time domain]
        
        classif_header = ['class_ratio',	'orig_classees',
                          'rebalance_classes', 'cm']	 # optional classification
        
        regression_header = ['cc',	'r2',	'cc_std',	'r2_std'] #optional regression

        by_fold = ['cv_metrics',	'cv_losses',
                   'test_metrics',	'test_losses', #by fold
                   'cv_metric_pvalues']

        log = dict()
        log['data_id'] = self.meta.data['data_id']

        #results info
        self.update_results()
        log['train metric'] = self.meta.results['tr_metric']
        log['validation metric'] = self.meta.results['v_metric']
        log['test metric'] =  self.meta.results['test_metric']
        log['train loss'] = self.meta.results['tr_loss']
        log['validation loss'] = self.meta.results['v_loss']
        log['test loss'] =  self.meta.results['test_loss']


        #training params
        for k in training_params:
            log[k] = self.meta.train_params[k]
        log_header += training_params

        #format specs: architecture and regularization
        specs_dict = self.meta.model_specs.copy()
        #print(specs_dict)
        specs_dict['l1_scope'] = '-'.join(self.meta.model_specs['l1_scope'])
        specs_dict['l2_scope'] = '-'.join(self.meta.model_specs['l2_scope'])
        specs_dict['unitnorm_scope'] = '-'.join(self.meta.model_specs['unitnorm_scope'])
        if isinstance(specs_dict['nonlin'], Callable):
            specs_dict['nonlin'] = specs_dict['nonlin'].__name__
        
        log['model_id'] = self.model_name
        log_header += ['model_id']
        for k in model_specs:
            log[k] = specs_dict[k]
        log_header += model_specs

        for k in data_info:
            log[k] = self.meta.data[k]
        log_header += data_info
        # class specific
        if self.meta.data['target_type'] == 'int':
            log['class_ratio'] = self.meta.data['class_ratio']
            log['orig_classees'] = self.meta.data['orig_classees']
            log['rebalance_classes'] = self.meta.data['rebalance_classes']
            log['cm'] = self.meta.data['cm']
            log_header += classif_header

        elif self.meta.data['target_type'] == 'float':
            log['cc'] = self.meta.results['cc']
            log['r2'] = self.meta.results['r2']
            log['cc_std'] = self.meta.results['cc_std']
            log['r2_std'] = self.meta.results['r2_std']
            log_header += regression_header

        if self.meta.data['channel_subset']  is not None:
            log['n_ch'] = len(self.meta.data['channel_subset'])
        if self.meta.data['sample_subset'] is not None:
            log['n_t'] = len(self.meta.data['sample_subset'])
            log['n_seq'] = len(self.meta.data['sample_subset'])
        
        # by fold
        for k in by_fold:
            log[k] = self.meta.results[k]
        log_header += by_fold
        self.log = log

        with open(savepath, 'a+', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=log_header)
            if not appending:
                writer.writeheader()
            writer.writerow(log)
            print("Saving updated log to: ",  savepath)

    def save(self):
        """
        Saves the model and (optionally, patterns, confusion matrices).

        Calls :meth:`update_results`, stacks the collected
        cross-validation weights (``self.cv_weights``), updates and
        saves the metadata (``self.meta.update``), and saves the
        Keras model (and encoder, if present) to
        ``self.model_path``.

        Returns
        -------
        None

        """

        self.update_results()
        weights = {k: np.stack(self.cv_weights[k], -1) for k in self.cv_weights.keys()}

        #Update and save meta file
        self.meta.update(data=self.dataset.h_params,
                         model_specs=self.specs,
                         patterns=self.cv_patterns,
                         weights=weights)

        #save the model
        self.km.save(os.path.join(self.model_path, self.model_name + '.h5'))
        if hasattr(self, 'km_enc'):
            self.km_enc.save(os.path.join(self.model_path, self.model_name + 'encoder_.h5'))


    def predict_sample(self, x):
        """Run the model on a single (or batch of) raw input sample(s).

        Parameters
        ----------
        x : array-like
            Input array whose last two dimensions match
            ``(n_t, n_ch)``. Expanded with leading singleton
            dimensions as needed to reach 4 dimensions.

        Returns
        -------
        out : tf.Tensor
            Model output (``self.km(x, training=True)``).

        """
        n_ch = self.dataset.h_params['n_ch']
        n_t = self.dataset.h_params['n_t']
        assert x.shape[-2:] == (n_t, n_ch),  "Shape mismatch! Expected {}x{}, \
            got {}x{}".format(n_t, n_ch, x.shape[-2], x.shape[-1])

        while x.ndim < 4:
            x = np.expand_dims(x, 0)

        out = self.km(x, training=True)
        # if self.dataset.h_params['target_type'] == 'int':
        #     out = np.argmax(out, -1)

        return out

    def predict(self, dataset=None, n_batches=1):
        """
        Parameters
        ----------
        dataset : tf.data.Dataset, str, list of str, or None, optional
            Dataset to predict on. Defaults to None, in which case
            ``self.dataset.val`` is used. A string or list/tuple of
            strings is treated as (a) path(s) to TFRecords and built
            into a dataset via ``self.dataset._build_dataset``.

        n_batches : int, optional
            Number of batches to draw from ``dataset`` before
            predicting. Defaults to 1.

        Returns
        -------
        y_true : np.array
                ground truth labels taken from the dataset

        y_pred : np.array
                model predictions
        """
        if not dataset:
            print("No dataset specified using validation dataset (Default)")
            dataset = self.dataset.val
        elif isinstance(dataset, str) or isinstance(dataset, (list, tuple)):
            dataset = self.dataset._build_dataset(dataset,
                                                 split=False,
                                                 test_batch=None,
                                                 repeat=True)
        elif not isinstance(dataset, tf.data.Dataset):
            print("Specify dataset")
            return None, None

        X = []
        y = []
        for batch_idx, (x, y_) in enumerate(dataset):
            if batch_idx >= n_batches:
                break


            X.append(x)
            y.append(y_)

        y_pred = self.km.predict(np.concatenate(X))
        y_true = np.concatenate(y)

        return y_true, y_pred

    def evaluate(self, dataset=False):
        """
        Parameters
        ----------
        dataset : tf.data.Dataset, str, list of str, or False, optional
            Dataset to evaluate on. Defaults to False, in which case
            ``self.dataset.val`` is used. A string or list/tuple of
            strings is treated as (a) path(s) to TFRecords and built
            into a dataset via ``self.dataset._build_dataset``.

        Returns
        -------
        losses : list
                model loss on a specified dataset

        metrics : np.array
                metrics evaluated on a specified dataset
        """

        if not dataset:
            print("No dataset specified using validation dataset (Default)")
            dataset = self.dataset.val
        elif isinstance(dataset, str) or isinstance(dataset, (list, tuple)):
            dataset = self.dataset._build_dataset(dataset,
                                             split=False,
                                             test_batch=None,
                                             repeat=True)
        elif not isinstance(dataset, tf.data.Dataset):
            print("Specify dataset")
            return None, None

        losses, metrics = self.km.evaluate(dataset,
                                           steps=self.dataset.validation_steps,
                                           verbose=0)
        return  losses, metrics
    

class SourceNet(BaseModel):
    """SourceNet

    For details see [1].

    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None, specs_prefix=False):
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
            See :meth:`BaseModel.__init__`. Defaults to False.

        specs : dict
                dictionary of model hyperparameters {

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
            Convolution padding. Defaults to 'SAME'.}"""
        self.scope = 'varcnn'
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0)
        meta.model_specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        super(SourceNet, self).__init__(meta, dataset, specs_prefix)

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """

        self.tconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size = (1, self.specs['filter_length']),
            #strides=1,
            padding='same',
            depth_multiplier=self.specs['n_latent'],
            data_format='channels_first',
            dilation_rate=(1, 1),
            activation=self.specs['nonlin'],
            use_bias=True,
            depthwise_initializer='glorot_uniform',
            bias_initializer='zeros',
            depthwise_regularizer=tf.keras.regularizers.l1(self.specs['l1_lambda']),
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
            )(self.inputs)
        print('tconv: ', self.tconv.shape )
        self.pooled = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding=self.specs['padding'],
                                  )(self.tconv)

        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=self.specs['nonlin'],
                            axis=3, specs=self.specs)(self.pooled)

        self.dmx1 = DeMixing(size=self.specs['n_latent'], nonlin=self.specs['nonlin'],
                            axis=1, specs=self.specs)(self.dmx)

        dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(self.dmx1)


        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        y_pred = self.fin_fc(dropout)

        return y_pred



class VARCNN(BaseModel):
    """VAR-CNN.

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
            See :meth:`BaseModel.__init__`. Defaults to False.

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
            Convolution padding. Defaults to 'SAME'.}"""
        self.scope = 'varcnn'
        if specs:
            meta.update(model_specs=specs)
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 0)
        meta.model_specs.setdefault('l1_scope', ['fc', 'dmx', 'tconv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        super().__init__(meta, dataset, specs_prefix)

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
                            axis=3, specs=self.specs)(self.inputs)


        self.tconv = VARConv(size=self.specs['n_latent'],
                             nonlin=self.specs['nonlin'],
                             filter_length=self.specs['filter_length'],
                             padding=self.specs['padding'],
                             specs=self.specs
                             )(self.dmx)

        self.pooled = TempPooling(pooling=self.specs['pooling'],
                                  pool_type=self.specs['pool_type'],
                                  stride=self.specs['stride'],
                                  padding=self.specs['padding'],
                                  )(self.tconv)

        dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(self.pooled)

        #fc1 = FullyConnected(size=128, nonlin=tf.nn.elu,
        #                    specs=self.specs)(dropout)

        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        y_pred = self.fin_fc(dropout)

        return y_pred



class FBCSP_ShallowNet(BaseModel):
    """
    Shallow ConvNet model from [2a]_.
    References
    ----------
    .. [2a] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """Initialize the FBCSP Shallow ConvNet model.

        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object; ``meta.model_specs`` is populated with
            this model's default hyperparameters where not already
            set: ``filter_length`` (25), ``n_latent`` (40),
            ``pooling`` (75), ``stride`` (15), ``pool_type`` ('avg'),
            ``padding`` ('SAME'), ``nonlin`` (``tf.nn.relu``),
            ``l1_lambda`` (3e-4), ``l2_lambda`` (3e-2), and the
            regularization/constraint scopes.

        dataset : mneflow.Dataset, optional
            Dataset object. Defaults to None (built from ``meta``).

        specs : dict, optional
            If provided, merged into ``meta.model_specs`` before
            applying the defaults above. Defaults to None.

        specs_prefix : bool, optional
            See :meth:`BaseModel.__init__`. Defaults to False.

        """
        self.scope = 'fbcsp-ShallowNet'
        if specs:
            meta.update(model_specs=specs)
        meta.model_specs.setdefault('filter_length', 25)
        meta.model_specs.setdefault('n_latent', 40)
        meta.model_specs.setdefault('pooling', 75)
        meta.model_specs.setdefault('stride', 15)
        meta.model_specs.setdefault('pool_type', 'avg')
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 3e-4)
        meta.model_specs.setdefault('l2_lambda', 3e-2)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', ['conv', 'fc'])

        meta.model_specs.setdefault('unitnorm_scope', [])
        #specs.setdefault('model_path', os.path.join(self.dataset.h_params['path'], 'models'))
        super().__init__(meta, dataset, specs_prefix)

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Temporal conv_1 25 10x1 kernels, followed by a spatial
        (depth-multiplied) convolution, average pooling, and a
        log-nonlinearity before the final fully-connected layer.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        #(self.inputs)
        inputs = tf.transpose(self.inputs,[0,3,2,1])
        #print(inputs.shape)
        #df = "channels_first"
        tconv1 = DepthwiseConv2D(
                        kernel_size=(1, self.specs['filter_length']),
                        depth_multiplier = self.specs['n_latent'],
                        strides=1,
                        padding="VALID",
                        activation = tf.identity,
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda'])
                        #kernel_constraint="maxnorm"
                        )

        tconv1_out = tconv1(inputs)
        print('tconv1: ', tconv1_out.shape) #should be n_batch, sensors, times, kernels

        sconv1 = Conv2D(filters=self.specs['n_latent'],
                        kernel_size=(self.dataset.h_params['n_ch'], 1),
                        strides=1,
                        padding="VALID",
                        activation = tf.square,
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        sconv1_out = sconv1(tconv1_out)
        print('sconv1:',  sconv1_out.shape)

        pool1 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(sconv1_out)

        print('pool1: ', pool1.shape)
        fc_out = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)
        y_pred = fc_out(tf.keras.backend.log(pool1))
        return y_pred
#
#
class LFLSTM(BaseModel):
    # TODO! Gabi: check that the description describes the model
    """LF-CNN-LSTM

    For details see [1].

    Parameters
    ----------
    n_latent : int
        number of latent components
        Defaults to 32

    filter_length : int
        length of spatio-temporal kernels in the temporal
        convolution layer. Defaults to 7

    stride : int
        stride of the max pooling layer. Defaults to 1

    pooling : int
        pooling factor of the max pooling layer. Defaults to 2

    pool_type : str {'avg', 'max'}
        Type of pooling operation. Defaults to 'max'.

    padding : str {'SAME', 'FULL', 'VALID'}
        Convolution padding. Defaults to 'SAME'.

    nonlin : callable
        Activation function of the temporal convolution layer.
        Defaults to tf.nn.relu.

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
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
            See :meth:`BaseModel.__init__`. Defaults to False.

        specs : dict, optional
                If provided, merged into ``meta.model_specs`` before
                applying the defaults below. Dictionary of model
                hyperparameters {
                dictionary of model hyperparameters {

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
        Stride of the max pooling layer. Defaults to 1.
        """
        self.scope = 'lf-cnn-lstm'
        if specs:
            meta.update(model_specs=specs)
        meta.model_specs.setdefault('filter_length', 7)
        meta.model_specs.setdefault('n_latent', 32)
        meta.model_specs.setdefault('pooling', 2)
        meta.model_specs.setdefault('stride', 2)
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('nonlin', tf.nn.relu)
        meta.model_specs.setdefault('l1_lambda', 0.)
        meta.model_specs.setdefault('l2_lambda', 0.)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs['scope'] = self.scope
        meta.model_specs.setdefault('unitnorm_scope', [])
        #specs.setdefault('model_path',  self.dataset.h_params['save_path'])
        super(LFLSTM, self).__init__(meta, dataset, specs_prefix)


    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Spatial demixing, followed by a temporal convolution and
        pooling, an LSTM over the pooled temporal features, and a
        final projection to the output.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """

        self.return_sequence = True
        self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
                            axis=3, specs=self.specs)
        dmx = self.dmx(self.inputs)
        #dmx = tf.reshape(dmx, [-1, self.dataset.h_params['n_t'],
        #                       self.specs['n_latent']])
        #dmx = tf.expand_dims(dmx, -1)
        print('dmx-sqout:', dmx.shape)

        self.tconv1 = LFTConv(scope="conv",
                              size=self.specs['n_latent'],
                              nonlin=tf.nn.relu,
                              filter_length=self.specs['filter_length'],
#                              stride=self.specs['stride'],
#                              pooling=self.specs['pooling'],
                              padding=self.specs['padding'])

        features = self.tconv1(dmx)
        pool1 = TempPooling(stride=self.specs['stride'],
                            pooling=self.specs['pooling'],
                            padding='SAME',
                            pool_type='max')


        pooled = pool1(features)
        print('features:', pooled.shape)

        fshape = tf.multiply(pooled.shape[2], pooled.shape[3])

        ffeatures = tf.reshape(pooled,
                              [-1, self.dataset.h_params['n_seq'], fshape])
        #  features = tf.expand_dims(features, 0)
        #l1_lambda = self.optimizer.params['l1_lambda']
        print('flat features:', ffeatures.shape)
        self.lstm = LSTM(scope="lstm",
                           size=self.specs['n_latent'],
                           kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal',
                           recurrent_regularizer=k_reg.l1(self.specs['l1_lambda']),
                           kernel_regularizer=k_reg.l2(self.specs['l2_lambda']),
                           bias_regularizer=None,
                           # activity_regularizer= regularizers.l1(0.01),
                           # kernel_constraint= constraints.UnitNorm(axis=0),
                           # recurrent_constraint= constraints.NonNeg(),
                           # bias_constraint=None,
                           dropout=0.1, recurrent_dropout=0.1,
                           nonlin=tf.identity,
                           unit_forget_bias=False,
                           return_sequences=self.return_sequence,
                           unroll=False)

        self.lstm_out = self.lstm(ffeatures)
        print('lstm_out:', self.lstm_out.shape)

        if self.return_sequence == True:
            self.fin_fc = DeMixing(size=self.out_dim,
                                   nonlin=tf.identity, axis=2)
        else:
            self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                                specs=self.specs)

        y_pred = self.fin_fc(self.lstm_out)
        print("fin fc out:", y_pred.shape)
        return y_pred
#
#


class Deep4(BaseModel):
    """
    Deep ConvNet model from [2b]_.
    References
    ----------
    .. [2b] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """Initialize the Deep ConvNet model.

        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object; ``meta.model_specs`` is populated with
            this model's default hyperparameters where not already
            set: ``filter_length`` (10), ``n_latent`` (25),
            ``pooling`` (3), ``stride`` (3), ``pool_type`` ('max'),
            ``padding`` ('SAME'), ``nonlin`` (``tf.nn.elu``),
            ``l1_lambda`` (0), ``l2_lambda`` (0), and the
            regularization/constraint scopes.

        dataset : mneflow.Dataset, optional
            Dataset object. Defaults to None (built from ``meta``).

        specs : dict, optional
            If provided, merged into ``meta.model_specs`` before
            applying the defaults above. Defaults to None.

        specs_prefix : bool, optional
            See :meth:`BaseModel.__init__`. Defaults to False.

        """
        self.scope = 'deep4'
        if specs:
            meta.update(model_specs=specs)
        meta.model_specs.setdefault('filter_length', 10)
        meta.model_specs.setdefault('n_latent', 25)
        meta.model_specs.setdefault('pooling', 3)
        meta.model_specs.setdefault('stride', 3)
        meta.model_specs.setdefault('pool_type', 'max')
        meta.model_specs.setdefault('padding', 'SAME')
        meta.model_specs.setdefault('nonlin', tf.nn.elu)
        meta.model_specs.setdefault('l1_lambda', 0)
        meta.model_specs.setdefault('l2_lambda', 0)
        meta.model_specs.setdefault('l1_scope', [])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        #specs.setdefault('model_path', os.path.join(self.dataset.h_params['path'], 'models'))
        super(Deep4, self).__init__(meta, dataset, specs_prefix)

    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Four stages of temporal/spatial convolution followed by
        average pooling, then a final fully-connected layer.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.scope = 'deep4'

        inputs = tf.keras.ops.transpose(self.inputs,[0,3,2,1])

        tconv1 = DepthwiseConv2D(
                        kernel_size=(1, self.specs['filter_length']),
                        depth_multiplier = self.specs['n_latent'],
                        strides=1,
                        padding=self.specs['padding'],
                        activation = tf.identity,
                        depthwise_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        depthwise_regularizer=k_reg.l2(self.specs['l2_lambda'])
                        #kernel_constraint="maxnorm"
                        )
        tconv1_out = tconv1(inputs)
        print('tconv1: ', tconv1_out.shape) #should be n_batch, sensors, times, kernels

        sconv1 = Conv2D(filters=self.specs['n_latent'],
                        kernel_size=(self.dataset.h_params['n_ch'], 1),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))
        sconv1_out = sconv1(tconv1_out)
        print('sconv1:',  sconv1_out.shape)

        pool1 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(sconv1_out)

        print('pool1: ', pool1.shape)

        ############################################################

        tsconv2 = Conv2D(filters=self.specs['n_latent']*2,
                        kernel_size=(1, self.specs['filter_length']),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        tsconv2_out = tsconv2(pool1)
        print('tsconv2:',  tsconv2_out.shape)

        pool2 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(tsconv2_out)

        print('pool2: ', pool2.shape)


        ############################################################

        tsconv3 = Conv2D(filters=self.specs['n_latent']*4,
                        kernel_size=(1, self.specs['filter_length']),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        tsconv3_out = tsconv3(pool2)
        print('tsconv3:',  tsconv3_out.shape)

        pool3 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(tsconv3_out)

        print('pool3: ', pool3.shape)

        ############################################################

        tsconv4 = Conv2D(filters=self.specs['n_latent']*8,
                        kernel_size=(1, self.specs['filter_length']),
                        strides=1,
                        padding=self.specs['padding'],
                        activation=self.specs['nonlin'],
                        kernel_initializer="he_uniform",
                        bias_initializer=Constant(0.1),
                        data_format="channels_last",
                        #data_format="channels_first",
                        kernel_regularizer=k_reg.l2(self.specs['l2_lambda']))


        tsconv4_out = tsconv4(pool3)
        print('tsconv4:',  tsconv4_out.shape)

        pool4 = TempPooling(pooling=self.specs['pooling'],
                                  pool_type="avg",
                                  stride=self.specs['stride'],
                                  padding='SAME',
                                  )(tsconv4_out)

        print('pool4: ', pool4.shape)


        fc_out = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)
        y_pred = fc_out(pool4)
        return y_pred
#
#

class EEGNet(BaseModel):
    """EEGNet.

    Parameters
    ----------
    specs : dict

        n_latent : int
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
            Dropout coefficient. Defaults to 0.1.

        depth_multiplier : int
            Depth multiplier of the depthwise spatial convolution.
            Defaults to 2.

        padding : str
            Convolution padding. Defaults to 'same'.

        nonlin : str or callable
            Activation function used after the convolutional blocks.
            Defaults to 'elu'.

    References
    ----------
    [3] V.J. Lawhern, et al., EEGNet: A compact convolutional neural
    network for EEG-based brain–computer interfaces 10 J. Neural Eng.,
    15 (5) (2018), p. 056013

    [4] Original EEGNet implementation by the authors can be found at
    https://github.com/vlawhern/arl-eegmodels
    """
    def __init__(self, meta, dataset=None, specs=None, specs_prefix=False):
        """Initialize the EEGNet model.

        Parameters
        ----------
        meta : mneflow.MetaData
            Metadata object; ``meta.model_specs`` is populated with
            this model's default hyperparameters, see the class
            docstring.

        dataset : mneflow.Dataset, optional
            Dataset object. Defaults to None (built from ``meta``).

        specs : dict, optional
            If provided, merged into ``meta.model_specs`` before
            applying the defaults. Defaults to None.

        specs_prefix : bool, optional
            See :meth:`BaseModel.__init__`. Defaults to False.

        """
        self.scope = 'eegnet8'
        if specs:
            meta.update(model_specs=specs)
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs.setdefault('filter_length', 64)
        meta.model_specs.setdefault('depth_multiplier', 2)
        meta.model_specs.setdefault('n_latent', 8)
        meta.model_specs.setdefault('pooling', 4)
        meta.model_specs.setdefault('stride', 4)
        meta.model_specs.setdefault('dropout', 0.1)
        meta.model_specs.setdefault('padding', 'same')
        meta.model_specs.setdefault('nonlin', 'elu')
        meta.model_specs['scope'] = self.scope
        super(EEGNet, self).__init__(meta, dataset, specs_prefix)


    def build_graph(self):
        """Build computational graph using defined placeholder `self.X`
        as input.

        Two EEGNet convolutional blocks (temporal convolution,
        depthwise spatial convolution, and separable convolution,
        each with batch normalization, activation, average pooling,
        and dropout), followed by a final fully-connected layer.

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """

        inputs = tf.transpose(self.inputs,[0,3,2,1])

        dropoutType = Dropout

        block1       = Conv2D(self.specs['n_latent'],
                              (1, self.specs['filter_length']),
                              padding = self.specs['padding'],
                              input_shape = (1, self.dataset.h_params['n_ch'],
                                             self.dataset.h_params['n_t']),
                              use_bias = False)(inputs)
        block1       = BatchNormalization(axis = 1)(block1)
        #print("Batchnorm:", block1.shape)
        block1       = DepthwiseConv2D((self.dataset.h_params['n_ch'], 1),
                                       use_bias = False,
                                       depth_multiplier = self.specs['depth_multiplier'],
                                       depthwise_constraint = constraints.MaxNorm(1.))(block1)
        #block1       = BatchNormalization(axis = 1)(block1)
        block1       = layers.Activation(self.specs['nonlin'])(block1)
        block1       = layers.AveragePooling2D((1, self.specs['pooling']))(block1)
        print("Block 1:", block1.shape)
        block1       = dropoutType(self.specs['dropout'])(block1)

        block2       = SeparableConv2D(self.specs['n_latent']*self.specs['depth_multiplier'], (1, self.specs['filter_length']//self.specs["pooling"]),
                                       use_bias = False, padding = self.specs['padding'])(block1)
        #block2       = BatchNormalization(axis = 1)(block2)

        #print("Batchnorm 2:", block2.shape)

        block2       = layers.Activation(self.specs['nonlin'])(block2)
        block2       = layers.AveragePooling2D((1, self.specs['pooling']*2))(block2)
        block2       = dropoutType(self.specs['dropout'])(block2)
        print("Block 2:", block2.shape)

        fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)
        y_pred = fin_fc(block2)

        return y_pred


class NoisyTrainer:
    """Custom training loop that adds Gaussian noise to regression targets.

    Trains a compiled Keras model with a manual (``tf.GradientTape``)
    training loop, injecting Gaussian noise into the training labels
    each step (an experimental regularization strategy), while
    evaluating on clean validation labels and applying early stopping
    based on validation loss. Used by :meth:`BaseModel.train` when
    ``noisy_labels=True``.

    """
    def __init__(self, model, model_path, noise_std=.1, patience=5, min_delta=0.001):
        """
        Initialize the trainer with a model, noise parameters, and early stopping configuration.

        Parameters
        ----------
        model : tf.keras.Model
            Compiled Keras model to train. Its ``.loss`` is reused as
            the training/validation loss function.

        model_path : str
            Base path (without extension) used to save the best
            weights, as ``<model_path>_best.weights.h5``.

        noise_std : float, optional
            Standard deviation of Gaussian noise to add to labels.
            Defaults to 0.1.

        patience : int, optional
            Number of epochs with no improvement after which training
            will be stopped. Defaults to 5.

        min_delta : float, optional
            Minimum change in monitored metric to qualify as an
            improvement. Defaults to 0.001.

        """
        self.model = model
        self.model_path = model_path + '_best.weights.h5'
        self.noise_std = noise_std
        self.loss_fn = model.loss 
        self.metric = tf.keras.metrics.R2Score()
        self.optimizer = tf.keras.optimizers.Adam()

        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    @tf.function
    def add_noise_to_labels(self, labels):
        """Add Gaussian noise to the labels.

        Parameters
        ----------
        labels : tf.Tensor
            Target labels to perturb.

        Returns
        -------
        noisy_labels : tf.Tensor
            ``labels`` plus Gaussian noise with standard deviation
            ``self.noise_std``.

        """
        noise = tf.random.normal(shape=tf.shape(labels),
                               mean=0.0,
                               stddev=self.noise_std)
        return labels + noise

    @tf.function
    def train_step(self, x, y):
        """Single training step with noisy labels.

        Parameters
        ----------
        x : tf.Tensor
            Input batch.

        y : tf.Tensor
            Target batch (noise is added internally before computing
            the loss).

        Returns
        -------
        loss : tf.Tensor
            Training loss computed against the noisy labels.

        """
        # Add noise to labels
        noisy_y = self.add_noise_to_labels(y)

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=True)
            # Calculate loss using noisy labels
            loss = self.loss_fn(noisy_y, predictions)

        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def validation_step(self, x, y):
        """Validation step without noise and training mode.

        Parameters
        ----------
        x : tf.Tensor
            Input batch.

        y : tf.Tensor
            (Clean) target batch.

        Returns
        -------
        val_loss : tf.Tensor
            Validation loss.

        metric : tf.Tensor
            Validation metric (``self.metric``, an R2 score).

        """
        predictions = self.model(x, training=False)
        val_loss = self.loss_fn(y, predictions)
        metric = self.metric(y, predictions)
        return val_loss, metric

    def train(self, train_dataset, val_dataset=None, epochs=1000, eval_step=5,
              val_steps=1, restore_best_weights=True):
        """
        Train the model with optional validation and early stopping.

        Parameters
        ----------
        train_dataset : tf.data.Dataset
            Training dataset. Iterated ``eval_step`` batches at a
            time per epoch, and shuffled after each epoch.

        val_dataset : tf.data.Dataset, optional
            Validation dataset. Defaults to None (no validation or
            early stopping).

        epochs : int, optional
            Maximum number of training epochs. Defaults to 1000.

        eval_step : int, optional
            Number of training batches per epoch, before validation
            is run. Defaults to 5.

        val_steps : int, optional
            Number of validation batches to average over each epoch.
            Only used if ``val_dataset`` is given. Defaults to 1.

        restore_best_weights : bool, optional
            Whether to save the best-so-far weights to
            ``self.model_path`` and reload them when early stopping
            triggers. Defaults to True.

        Returns
        -------
        training_history : dict
            Dictionary with keys ``'train_loss'`` and ``'val_loss'``,
            each a list of per-epoch average losses.

        """
        training_history = {
            'train_loss': [],
            'val_loss': []
        }

        if val_dataset is not None:
            val_iter = iter(val_dataset)
        for epoch in range(epochs):
            train_iter = iter(train_dataset)

            # Train until it's time to evaluate
            train_losses = []
            for i in range(eval_step):
                x_batch, y_batch = next(train_iter)
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(float(loss))

            if val_dataset is not None:
            #Evalute on validation set
                val_losses = []
                val_metrics = []
                for i in range(val_steps):
                    x_val_batch, y_val_batch = next(val_iter)
                    val_loss, val_metric = self.validation_step(x_val_batch, y_val_batch)
                    val_losses.append(float(val_loss))
                    val_metrics.append(float(val_metric))

            #Calculate output
            avg_train_loss = np.mean(train_losses)
            training_history['train_loss'].append(avg_train_loss)
            avg_val_loss = np.mean(val_losses)
            avg_val_metric = np.mean(val_metrics)
            training_history['val_loss'].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < self.best_val_loss - self.min_delta:
                #print("Eval decr")
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                # Optionally save the best model
                if restore_best_weights:
                    self.model.save_weights(self.model_path)
            else:
                self.epochs_without_improvement += 1
                #print("Eval no decr")

            print(f"Epoch {epoch + 1}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, "
                  f"Val Metric = {avg_val_metric:.4f}")
            train_dataset.shuffle(10000)
            # Check for early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                if restore_best_weights:
                    print("Restoring best weights")
                    self.model.load_weights(self.model_path, skip_mismatch=True)
                break


        return training_history


