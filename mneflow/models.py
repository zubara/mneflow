# -*- coding: utf-8 -*-
"""
Define mneflow.models.Model parent class and the implemented models as
its subclasses. Implemented models inherit basic methods from the
parent class.

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""

#TODO: update vizualizations

import tensorflow as tf

import numpy as np
import pickle


from mne import channels, evoked, create_info, Info
from mne.filter import filter_data

from scipy.signal import freqz, welch
from scipy.stats import spearmanr

from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .layers import LFTConv, VARConv, DeMixing, FullyConnected, TempPooling
from tensorflow.keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras import regularizers as k_reg, constraints, layers

from .layers import LSTM
import csv
import os
from .data import Dataset
from .utils import regression_metrics, _onehot
from collections import defaultdict


def uniquify(seq):
    un = []
    [un.append(i) for i in seq if not un.count(i)]
    return un


# ----- Base model -----
#@tf.keras.utils.register_keras_serializable(package="mmneflow")
class BaseModel():
    """Parent class for all MNEflow models.

    Provides fast and memory-efficient data handling and simplified API.
    Custom models can be built by overriding _build_graph and
    _set_optimizer methods.
    """

    def __init__(self, meta=None, dataset=None):
        """
        Parameters
        ----------
        Dataset : mneflow.Dataset
            `Dataset` object.

        specs : dict
            Dictionary of model-specific hyperparameters. Must include
            at least `model_path` - path for saving a trained model
            See `Model` subclass definitions for details. Unless otherwise
            specified uses default hyperparameters for each implemented model.
        """
        self.specs = meta.model_specs
        meta.model_specs['model_path'] = os.path.join(meta.data['path'], 
                                                      'models')  
        
        self.meta = meta
        self.model_path = os.path.join(meta.data['path'], 'models\\')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)            
        

        if not dataset and meta:
            self.dataset = Dataset(meta,
                                 **meta.data
                                 )
        elif dataset:
            self.dataset = dataset
        else:
            print("Provide Dataset ot Metadata file")

        self.input_shape = (self.dataset.h_params['n_seq'],
                            self.dataset.h_params['n_t'],
                            self.dataset.h_params['n_ch'])
        self.y_shape = self.dataset.y_shape
        self.out_dim = np.prod(self.y_shape)


        self.inputs = layers.Input(shape=(self.input_shape))
        self.trained = False
        self.y_pred = self.build_graph()
        self.log = dict()
        self.cm = np.zeros([self.y_shape[-1], self.y_shape[-1]])
        self.cv_patterns = defaultdict(dict)
        if not hasattr(self, 'scope'):
            self.scope = 'basemodel'



    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

    # def get_config(self):
    #     config = {
    #         "fc": self.fc
    #     }
    #     return config



    def build(self, optimizer="adam", loss=None, metrics=None, mapping=None,
              learn_rate=3e-4):
        """Compile a model.

        Parameters
        ----------
        optimizer : str, tf.optimizers.Optimizer
            Deafults to "adam"

        loss : str, tf.keras.losses.Loss
            Defaults to MSE in target_type is "float" and
            "softmax_crossentropy" if "target_type" is int

        metrics : str, tf.keras.metrics.Metric
            Defaults to RMSE in target_type is "float" and
                "categorical_accuracy" if "target_type" is int

        learn_rate : float
            Learning rate, defaults to 3e-4

        mapping : str

        """
        # Initialize computational graph
        if mapping:
            map_fun = tf.keras.activations.get(mapping)
            self.y_pred= map_fun(self.y_pred)

        self.km = tf.keras.Model(inputs=self.inputs, outputs=self.y_pred)

        self.params = {"optimizer": tf.optimizers.get(optimizer).from_config(
            {"learning_rate":learn_rate})}

        if loss:
            self.params["loss"] = tf.keras.losses.get(loss)

        if metrics:
            if not isinstance(metrics, list):
                metrics = [metrics]
            self.params["metrics"] = [tf.keras.metrics.get(metric) for metric in metrics]

        #
        #self.specs.setdefault('unitnorm_scope', [])
       # Initialize optimizer
        if self.dataset.h_params["target_type"] in ['float', 'signal']:
            self.params.setdefault("loss", tf.keras.losses.MeanSquaredError(name='MSE'))
            self.params.setdefault("metrics", tf.keras.metrics.RootMeanSquaredError(name="RMSE"))

        elif self.dataset.h_params["target_type"] in ['int']:
            self.params.setdefault("loss", [tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                                   name='cce')])
            self.params.setdefault("metrics", [tf.keras.metrics.CategoricalAccuracy(name="cat_ACC")])

        self.km.compile(optimizer=self.params["optimizer"],
                        loss=self.params["loss"],
                        metrics=self.params["metrics"])


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
              early_stopping=3, mode='single_fold', prune_weights=False,
              collect_patterns = False, class_weights = None) :

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
        """

        
        if not eval_step:
            train_size = self.dataset.h_params['train_size']
            eval_step = train_size // self.dataset.h_params['train_batch'] + 1
        
        self.train_params = dict(n_epochs=n_epochs, 
                                 eval_step=eval_step, 
                                 early_stopping=early_stopping, 
                                 mode=mode)
        self.meta.update(train_params=self.train_params)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=min_delta,
                                                      patience=self.meta.train_params['early_stopping'],
                                                      restore_best_weights=True)
        rmss = defaultdict(list)
        self.cv_losses = []
        self.cv_metrics = []
        self.cv_test_losses = []
        self.cv_test_metrics = []

        if class_weights:
            multiplier = 1. / min(class_weights.values())
            class_weights = {k:v*multiplier for k,v in class_weights.items()}

        else:
            class_weights = None
        print("Class weights: ", class_weights)
        if mode == 'single_fold':
            n_folds = 1
            train, val = self.dataset._build_dataset(self.dataset.h_params['train_paths'],
                                               train_batch=self.dataset.training_batch,
                                               test_batch=self.dataset.validation_batch,
                                               split=True, val_fold_ind=0)

            self.t_hist = self.km.fit(train,
                                   validation_data=val,
                                   epochs=self.meta.train_params['n_epochs'], 
                                   steps_per_epoch=self.meta.train_params['eval_step'],
                                   shuffle=True,
                                   validation_steps=self.dataset.validation_steps,

                                   callbacks=[stop_early], verbose=2,
                                   class_weight=class_weights)

            #compute validation loss and metric
            v_loss, v_metric = self.evaluate(self.dataset.val)

            self.cv_losses.append(v_loss)
            self.cv_metrics.append(v_metric)

            if len(self.dataset.h_params['test_paths']):
                    t_loss, t_metric = self.evaluate(self.dataset.h_params['test_paths'])
                    # print("""Test performance:
                    #       Loss: {:.4f} 
                    #       Metric: {:.4f}""".format(t_loss, t_metric))
                    self.cv_test_losses.append(t_loss)
                    self.cv_test_metrics.append(t_metric)

            #compute specific metrics for classification and regresssion
            y_true, y_pred = self.predict(self.dataset.val)
            if self.dataset.h_params['target_type'] == 'float':
                rms = regression_metrics(y_true, y_pred)
                print("Validation set: Corr =", rms['cc'], " R2 =", rms['r2'])
            else:
                rms = None
                self.cm += self._confusion_matrix(y_true, y_pred)

            if collect_patterns and self.scope == 'lfcnn':
                self.collect_patterns(fold=0, n_folds=n_folds, n_comp=int(collect_patterns))

        elif mode == 'cv':

            n_folds = len(self.dataset.h_params['folds'][0])
            print("Running cross-validation with {} folds".format(n_folds))


            for jj in range(n_folds):
                print("fold:", jj)
                train, val = self.dataset._build_dataset(self.dataset.h_params['train_paths'],
                                                   train_batch=self.dataset.training_batch,
                                                   test_batch=self.dataset.validation_batch,
                                                   split=True, val_fold_ind=jj)
                self.t_hist = self.km.fit(train,
                                   validation_data=val,
                                   epochs=self.meta.train_params['n_epochs'], 
                                   steps_per_epoch=self.meta.train_params['eval_step'],
                                   shuffle=True,
                                   validation_steps=self.dataset.validation_steps,
                                   callbacks=[stop_early], verbose=2,
                                   class_weight=class_weights)

                v_loss, v_metric = self.evaluate(val)
                self.cv_losses.append(v_loss)
                self.cv_metrics.append(v_metric)
                if len(self.dataset.h_params['test_paths']):
                    t_loss, t_metric = self.evaluate(self.dataset.h_params['test_paths'])
                    print("""Test performance:
                          Loss: {:.4f} 
                          Metric: {:.4f}""".format(t_loss, t_metric))
                    self.cv_test_losses.append(t_loss)
                    self.cv_test_metrics.append(t_metric)

                y_true, y_pred = self.predict(val)

                if self.dataset.h_params['target_type'] == 'float':
                    rms = regression_metrics(y_true, y_pred)
                    for k,v in rms.items():
                        rmss[k].append(v)
                    print("Validation set: Corr =", rms['cc'], " R2 =", rms['r2'])
                else:
                    self.cm += self._confusion_matrix(y_true, y_pred)
                    rms = None

                if collect_patterns and self.scope == 'lfcnn':
                    self.collect_patterns(fold=jj, n_folds=n_folds,
                                          n_comp=int(collect_patterns))


                if jj < n_folds - 1:
                    self.shuffle_weights()
                else:
                    "Not shuffling the weights for the last fold"

                print("""Fold: {} Validation performance:\n
                      Loss: {:.4f}, 
                      Metric: {:.4f}""".format(jj, v_loss, v_metric))

            metrics = self.cv_metrics
            losses = self.cv_losses

            print("{} with {} folds completed. Loss: {:.4f} +/- {:.4f}. Metric: {:.4f} +/- {:.4f}".format(mode, n_folds, np.mean(losses), np.std(losses), np.mean(metrics), np.std(metrics)))

            if self.dataset.h_params['target_type'] == 'float':
                rms = {k:np.mean(v) for k, v in rmss.items()}
                rms.update({k + '_std':np.std(v) for k, v in rmss.items()})
                print("Validation set: Corr : {:.3f} +/- {:.3f}. \
                      R^2: {:.3f} +/- {:.3f}".format(
                      rms['cc'], rms['cc_std'], rms['r2'], rms['r2_std']))
            else:
                rms = None


        elif mode == "loso":
            n_folds = len(self.dataset.h_params['train_paths'])
            print("Running leave-one-subject-out CV with {} subject".format(n_folds))

            for jj in range(n_folds):
                print("fold:", jj)

                test_subj = self.dataset.h_params['train_paths'][jj]
                train_subjs = self.dataset.h_params['train_paths'].copy()
                train_subjs.pop(jj)

                print(train_subjs)
                print('***')
                print(test_subj)

                train, val = self.dataset._build_dataset(train_subjs,
                                                   train_batch=self.dataset.training_batch,
                                                   test_batch=self.dataset.validation_batch,
                                                   split=True, val_fold_ind=0)


                self.t_hist = self.km.fit(train,
                                   validation_data=val,
                                   epochs=self.meta.train_params['n_epochs'], 
                                   steps_per_epoch=self.meta.train_params['eval_step'],
                                   shuffle=True,
                                   validation_steps=self.dataset.validation_steps,
                                   callbacks=[stop_early], verbose=2,
                                   class_weight=class_weights)


                test = self.dataset._build_dataset(test_subj,
                                                   test_batch=None,
                                                   split=False)

                v_loss, v_metric = self.evaluate(val)
                t_loss, t_metric = self.evaluate(test)

                print("Subj: {} Loss: {:.4f}, Metric: {:.4f}".format(jj, t_loss, t_metric))
                self.cv_losses.append(v_loss)
                self.cv_metrics.append(v_metric)
                self.cv_test_losses.append(t_loss)
                self.cv_test_metrics.append(t_metric)

                y_true, y_pred = self.predict(val)
                if self.dataset.h_params['target_type'] == 'float':
                    y_true, y_pred = self.predict(val)
                    rms = regression_metrics(y_true, y_pred)
                    for k,v in rms.items():
                        rmss[k].append(v)
                    print("Validation set: Corr =", rms['cc'], " R2 =", rms['r2'])
                else:
                    self.cm += self._confusion_matrix(y_true, y_pred)
                    rms = None

                if collect_patterns and self.scope == 'lfcnn':
                    self.collect_patterns(fold=jj, n_folds=n_folds,
                                          n_comp=int(collect_patterns))

                if jj < n_folds -1:
                    self.shuffle_weights()
                else:
                    "Not shuffling the weights for the last fold"

            if self.dataset.h_params['target_type'] == 'float':
                rms = {k:np.mean(v) for k, v in rmss.items()}
                rms.update({k + '_std':np.std(v) for k, v in rmss.items()})
                print("Validation set: Corr : {:.3f} +/- {:.3f}. \
                      R^2: {:.3f} +/- {:.3f}".format(
                      rms['cc'], rms['cc_std'], rms['r2'], rms['r2_std']))
            else:
                rms = None

            self.update_log(rms, prefix='loso_')

        print("""{} with {} fold(s) completed. \n
              Validation Performance: 
              Loss: {:.4f} +/- {:.4f}.
              Metric: {:.4f} +/- {:.4f}"""
              .format(mode, n_folds,
                      np.mean(self.cv_losses), np.std(self.cv_losses),
                      np.mean(self.cv_metrics), np.std(self.cv_metrics)))
        
        if len(self.dataset.h_params['test_paths']) > 0:
            print("""\n
              Test Performance: 
              Loss: {:.4f} +/- {:.4f}.
              Metric: {:.4f} +/- {:.4f}"""
              .format(np.mean(self.cv_test_losses), 
                      np.std(self.cv_test_losses),
                      np.mean(self.cv_test_metrics), 
                      np.std(self.cv_test_metrics)))
        self.update_log(rms=rms, prefix=mode)    
        #return self.cv_losses, self.cv_metrics


    def prune_weights(self, increase_regularization=3.):
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
        print("Re-shuffling weights between folds")
        weights = self.km.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        self.km.set_weights(weights)


    # def reinitialize_weights(self):
    #     session = K.get_session()
    #     print("Re-initializing weights")
    #     for layer in self.km.layers: 
    #         if hasattr(layer, 'kernel_initializer'):
    #             layer.kernel.initializer.run(session=session)
    #         if hasattr(layer, 'bias_initializer'):
    #             layer.bias.initializer.run(session=session)
                

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
        """Compute unnormalizewd confusion matrix"""
        y_p = _onehot(np.argmax(y_pred,1), n_classes=self.y_shape[-1])
        cm = np.dot(y_p.T, y_true)
        return cm

    def update_log(self, rms=None, prefix=''):
        """Logs experiment to self.model_path + self.scope + '_log.csv'.

        If the file exists, appends a line to the existing file.
        """
        savepath = os.path.join(self.model_path, self.scope + '_log.csv')
        appending = os.path.exists(savepath)

        log = dict()
        if rms:
            log.update({prefix+k:v for k,v in rms.items()})

        #dataset info
        log['data_id'] = self.dataset.h_params['data_id']
        log['data_path'] = self.dataset.h_params['data_path']

        log['y_shape'] = np.prod(self.dataset.h_params['y_shape'])
        log['fs'] = str(self.dataset.h_params['fs'])

        #architecture and regularization
        log.update(self.specs)

        #training paramters
        log['n_epochs'] = self.train_params['n_epochs']
        log['eval_step'] = self.train_params['eval_step']
        log['early_stopping'] = self.train_params['early_stopping']
        log['mode'] = self.train_params['mode']
        
        log['v_metric'] = np.mean(self.cv_metrics)
        log['v_loss'] = np.mean(self.cv_losses)
        log['cv_metrics'] = self.cv_metrics
        log['cv_losses'] = self.cv_losses
        
        tr_loss, tr_metric = self.evaluate(self.dataset.train)
        log['tr_metric'] = tr_metric
        log['tr_loss'] = tr_loss


        if len(self.dataset.h_params['test_paths']) > 0:
            t_loss = np.mean(self.cv_test_losses)
            t_metric = np.mean(self.cv_test_metrics)
            if self.dataset.h_params['target_type'] == 'float':
                y_true, y_pred = self.predict(self.dataset.h_params['test_paths'])
                rms_test = regression_metrics(y_true, y_pred)
                print("Test set: Corr =", rms_test['cc'], "R2 =", rms_test['r2'])
                log.update({'test_'+k:v for k,v in rms_test.items()})
            log['test_metric'] = t_metric
            log['test_loss'] = t_loss
            log['test_metrics'] = self.cv_test_metrics
            log['test_losses'] = self.cv_test_losses

        else:
            log['test_metric'] = "NA"
            log['test_loss'] = "NA"
            log['test_metrics'] = "NA"
            log['test_losses'] = "NA"
        self.log.update(log)

        with open(savepath, 'a+', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.log.keys())
            if not appending:
                writer.writeheader()
            writer.writerow(self.log)
            print("Saving updated log to: ",  savepath)

    def save(self):
        """
        Saves the model and (optionally, patterns, confusion matrices)
        """
        
        
        #Update and save meta file
        self.meta.update(data=self.dataset.h_params,
                         model_specs=self.specs,
                         train_params=self.train_params,
                         patterns=self.cv_patterns)
        self.meta.save()
        
        model_name = "_".join([self.scope,
                               self.dataset.h_params['data_id']])
        #self.specs['model_name'] = model_name

        #save the model
        self.km.save(os.path.join(self.model_path, model_name + '.hf5'))
        

        # #Save results from multiple folds
        # if hasattr(self, 'cv_patterns'):
        #     #if hasattr(self, 'cv_patterns'):
        #     print("Saving patterns to: " + self.model_path + self.model_name + "\\mneflow_patterns.npz" )
        #     np.savez(self.model_path + self.model_name + "\\mneflow_patterns.npz",
        #              cv_patterns=self.cv_patterns,
        #              specs=self.specs,
        #              meta=self.dataset.h_params,
        #              log=self.log,
        #              cm=self.cm,
        #              )

        #Update metadata with specs and new self.dataset options
        #For LFCNN save patterns

    # def restore(self, meta):
    #     #TODO: take path, scope, and data_id as inputs.
    #     #TODO: build dataset from metadata
    #     #TODO: initialize from specs

    #     self.model_name = "_".join([self.scope,
    #                                 self.dataset.h_params['data_id']])
    #     self.km  = tf.keras.models.load_model(self.model_path + self.model_name)
    #     #try:

    #     if os.path.exists(self.model_path + self.model_name + "\\mneflow_patterns.npz"):
    #         print("Restoring from:" + self.model_path + self.model_name + "\\mneflow_patterns.npz" )
    #         f = np.load(self.model_path + self.model_name + "\\mneflow_patterns.npz",
    #                     allow_pickle=True)
    #         self.cv_patterns = f["cv_patterns"]#.item()
    #         self.specs = f["specs"]#.item()
    #         self.meta = f["meta"]#.item()
    #         self.log = f["log"]#.item()
    #         self.cm = f["cm"]

    def predict_sample(self, x):
        n_ch = self.dataset.h_params['n_ch']
        n_t = self.dataset.h_params['n_t']
        assert x.shape[-2:] == (n_t, n_ch),  "Shape mismatch! Expected {}x{}, \
            got {}x{}".format(n_t, n_ch, x.shape[-2], x.shape[-1])

        while x.ndim < 4:
            x = np.expand_dims(x, 0)

        out = self.km.predict(x)
        if self.dataset.h_params['target_type'] == 'int':
            out = np.argmax(out, -1)

        return out

    def predict(self, dataset=None):
        """
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
                                                 repeat=False)
        elif not isinstance(dataset, tf.data.Dataset):
            print("Specify dataset")
            return None, None

        X = []
        y = []

        for row in dataset.take(1):
            X.append(row[0])
            y.append(row[1])

        y_pred = self.km.predict(np.concatenate(X))
        y_true = np.concatenate(y)

        #y_true = y_true.numpy()
        return y_true, y_pred

    def evaluate(self, dataset=False):
        """
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

    def plot_confusion_matrix(self, cm=None,
                              classes=None,
                              normalize=True,
                              title=None,
                              cmap=plt.cm.Blues):
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
        if not np.any(cm):
            cm = self.cm
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if not classes:
            classes = [' '.join(["Class", str(i)]) for i in range(cm.shape[0])]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print(cm)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
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



class LFCNN(BaseModel):
    """LF-CNN. Includes basic parameter interpretation options.

    For details see [1].
    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None):
        """

        Parameters
        ----------
        Dataset : mneflow.Dataset

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
            Convolution padding. Defaults to 'SAME'.}

        stride : int
        Stride of the max pooling layer. Defaults to 1.
        """
        self.scope = 'lfcnn'
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
        meta.model_specs.setdefault('l1_scope', ['fc', 'demix', 'lf_conv'])
        meta.model_specs.setdefault('l2_scope', [])
        meta.model_specs.setdefault('unitnorm_scope', [])
        meta.model_specs['scope'] = self.scope
        #specs.setdefault('model_path',  self.dataset.h_params['save_path'])
        super(LFCNN, self).__init__(meta, dataset)



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
                                  padding='SAME',#self.specs['padding']
                                  )
        self.pooled = self.pool(self.tconv_out)

        self.dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(self.pooled)

        self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
                            specs=self.specs)

        y_pred = self.fin_fc(self.dropout)

        return y_pred

    def build_encoder(self):
        """Build computational graph for an interpretable Generator

        Returns
        --------
        y_pred : tf.Tensor
            Output of the forward pass of the computational graph.
            Prediction of the target variable.
        """
        self.km.trainable = False
        print("Freezing the decoder")
        #((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
        if self.dataset.h_params['n_t']%self.specs['stride'] == 0:
            padding=None
        elif self.dataset.h_params['n_t']%self.specs['stride'] - 1 >= 1:
            padding = (self.dataset.h_params['n_t']%self.specs['stride'] - 1, 0)
        else :
            padding = ((self.dataset.h_params['n_t'] - 1)%self.specs['stride'] , 0)


        print(padding)


        #start with the output of decoder
        #elf.enc_inputs = layers.Input(self.y_shape)
        self.enc_fc = FullyConnected(size=self.fin_fc.w.shape[0], nonlin=tf.identity,
                            specs=self.specs)
        enc_tconv_activations =  self.enc_fc(self.y_pred)
        enc_tconv_activations_r = tf.reshape(enc_tconv_activations,
                                             [-1, 1, self.pooled.shape[-2],
                                             self.specs['n_latent']])
        print(enc_tconv_activations_r.shape)
        enc_dropout = Dropout(self.specs['dropout'],
                          noise_shape=None)(enc_tconv_activations_r)
        #upool and apply transposed depthwise convolution
        # self.enc_tconv_trans = LFTConvTranspose1(filters=self.tconv.filters,
        #                                    kernel_size= self.specs['filter_length'],
        #                                    stride=self.specs['stride'],
        #                                    output_padding=n_padding)
        #n_padding
        self.enc_tconv_trans=tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(self.specs['filter_length'], self.specs['n_latent']),
            strides=(self.specs['stride'], 1),
            padding='same',
            output_padding=padding,
            data_format='channels_first',
            dilation_rate=(1, 1),
            activation=None,
            use_bias=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l1(self.specs['l1_lambda']),
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
            bias_constraint=None,
        )


        enc_deconv = self.enc_tconv_trans(enc_dropout)
        #enc_deconv = tf.transpose(enc_deconv, perm=[0,3,1,2])
        print("Enc_deconv:", enc_deconv.shape)

        self.de_dmx = DeMixing(size=self.dataset.h_params['n_ch'], nonlin=tf.identity,
                            axis=3, specs=self.specs)

        self.X_pred = self.de_dmx(enc_deconv)
        print(self.X_pred.shape)
        self.km_enc = tf.keras.Model(inputs=self.inputs, outputs=self.X_pred)

        self.params['enc_loss'] = [tf.keras.losses.MAE]#tf.reduce_mean((self.X_pred - self.inputs)**2)
        #self.params['enc_metrics'] = tf.keras.metrics.RootMeanSquaredError(name="RMSE")


        #print(params)
        self.km_enc.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                        loss=self.params['enc_loss'])

    def compute_enc_patterns(self, inputs=None):
        """

        """
        if not np.any(inputs):
            print('Using fake inputs')
            inputs = np.identity(self.out_dim)
        pooled = tf.reshape(self.enc_fc(inputs),
                   [self.out_dim,
                    1,
                    self.pooled.shape[-2],
                    self.specs['n_latent']])
        unpooled_wfs = self.enc_tconv_trans(pooled)
        patterns = self.de_dmx(unpooled_wfs)

        return np.squeeze(patterns), unpooled_wfs

    def train_encoder(self, n_epochs, eval_step=None, min_delta=1e-6,
              early_stopping=3, collect_patterns=False):
        dataset_train = self.dataset.train.map(lambda x, y : (x, x))
        dataset_val = self.dataset.val.map(lambda x, y : (x, x))
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=min_delta,
                                                      patience=early_stopping,
                                                      restore_best_weights=True)
        self.t_hist = self.km_enc.fit(dataset_train,
                                   validation_data=dataset_val,
                                   epochs=n_epochs, steps_per_epoch=eval_step,
                                   shuffle=True,
                                   validation_steps=self.dataset.validation_steps,
                                   callbacks=[stop_early], verbose=2,
                                   #class_weight=class_weights
                                   )

    def get_config(self):
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
        """Compute spatial class-conditional covariance matrix from the dataset

        Parameters:
        -----------
        dataset : tf.data.Dataset

        Returns : dcov [y_shape, n_ch, n_ch]

        """

        dcovs = []
        dcovs_n = []
        for class_y in range(self.out_dim):
            class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
            xs = np.squeeze(X.numpy()[class_ind, ...])
            #xs -= np.mean(xs, axis=-2, keepdims=True)
            ddof_s = xs.shape[0]*self.dataset.h_params['n_t'] - 1
            cov_s = np.einsum('ijk, ijl -> kl', xs, xs) / ddof_s

            anti_class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)!=class_y))#[0]
            xn = np.squeeze(X.numpy()[anti_class_ind, ...])
            ddof_n = xn.shape[0]*self.dataset.h_params['n_t'] - 1
            cov_n = np.einsum('ijk, ijl -> kl', xn, xn) / ddof_n

            dcovs.append(cov_s) #  - cov_n
            dcovs_n.append(cov_n)
        return np.array(dcovs), np.array(dcovs_n)


    def patterns_cov_xx(self, y, weights, activations, dcov):
        """
        X - [i,...,m]
        y - [i,...,j] - used for cov[y]
        w - [k,...,j]
        Sx - [k,...,mj]"""
        x_shape = list(activations['tconv'].shape)
        y_shape = list(y.shape)
        #out_shape = x_shape[1:] + y_shape[1:]
        #print(x_shape, y_shape, weights['out_weights'].shape)
        ddof = activations['tconv'].shape[0] - 1
        X = np.reshape(activations['tconv'], [activations['tconv'].shape[0], -1])
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

        # A.*w
        #aw = a_out * w #shape = [...]
        # activation of tconv by each signal component of each sample
        Sx = np.einsum('il, il, ji -> jil', a_out, w, X)
        Sx = np.reshape(Sx, [-1, x_shape[-1], y_shape[-1]])
        Sx = Sx - Sx.mean(1, keepdims=True)
        ddof = Sx.shape[0] - 1
        cov_sx = np.einsum('ijk, ilk -> jlk', Sx, Sx) / ddof

        patterns = []
        #dc = dcov['input_spatial']
        for i in range(y.shape[-1]):
            prec_sx = np.linalg.pinv(cov_sx[...,i])
            dc = dcov['class_conditional'][i]
            patterns.append(np.einsum('hi, ij, jk -> h',
                                      dc, weights['dmx'], prec_sx))
        patterns = np.stack(patterns, -1)
        return patterns


    def patterns_cov_xy_hat(self, X, y, activations, weights):
        Sx_tconv = self.backprop_fc(activations['tconv'],
                                    activations['fc'],
                                    y,
                                    weights['out_weights'])
        Sx_dmx = self.backprop_covxy(X,
                                    activations['dmx'],
                                    Sx_tconv,
                                    weights['dmx'])
        return Sx_tconv, Sx_dmx


    def backprop_fc(self, X, y_hat, y, w):
        """
        X - [i,...,m]
        y - [i,...,j]
        w - [k,...,j]
        Sx - [k,...,mj]"""
        x_shape = list(X.shape)
        y_shape = list(y_hat.shape)
        #out_shape = x_shape[1:] + y_shape[1:]
        #print(x_shape, y_shape, w.shape)
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
        #cov_xy = np.cov(X, y)


        #compute inverse covariance of the output
        #cov_yy = np.einsum('ij, ik -> jk', y_hat, y_hat) / ddof
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
        xdmx = np.reshape(Hx, [-1, Hx.shape[-1]])
        xdmx = xdmx - xdmx.mean(0, keepdims=True)
        xinp = np.reshape(X, [-1, X.shape[-1]])
        xinp = xinp - xinp.mean(0, keepdims=True)
        cov_xy = np.dot(xinp.T, xdmx)
        aw = cov_xy
        sx = np.reshape(Sx, [-1, Sx.shape[-2], Sx.shape[-1]])
        sx = sx - sx.mean(0, keepdims=True)
        ddof = sx.shape[0] - 1
        cov_sx = np.einsum('ijk, ilk -> kjl', sx,sx) / ddof
        prec_yy_hat = np.stack([np.linalg.pinv(s) for s in cov_sx])
        ww = np.einsum('ij, ljk -> ikl', w, prec_yy_hat)
        a = np.einsum('ij, ijk -> ik', cov_xy, ww)

        return a

        # print(cov_xy.shape)
        # patterns = []
        # for class_y in range(Sx.shape[-1]):
        #     sx = np.reshape(Sx, [-1, Sx.shape[-2], Sx.shape[-1]])
        #     sx -= sx.mean(0, keepdims=True)
        #     prec_sx = np.linalg.inv(np.dot(sx.T, sx))

        #     aw = np.dot(cov_xy, prec_sx) * w
        #     #component_activations = np.dot(aw, prec_sx)
        #     patterns.append(aw.sum(-1))

        # return np.stack(patterns, 1)

    # def backprop_dmx(self, X, y_hat, y, w):
    #     x_shape = list(X.shape)
    #     y_shape = list(y_hat.shape)
    #     k = y.shape[-1]
    #     y_hat = np.reshape(y_hat, [y_hat.shape[0], -1])
    #     #out_shape = x_shape[1:] + y_shape[1:]
    #     #print(x_shape, y_shape, w.shape)
    #     ddof_xy = np.prod(x_shape[:3]) - 1
    #     ddof_yy = np.prod(y_shape[:3]) - 1

    #     cov_xy = np.einsum('ijkl, ijmn -> lnm', X, y_hat) / ddof_xy #shape = [n_ch, n_latnet, n_bins]
    #     cov_yy = np.einsum('ijkl, ijkm -> klm', y_hat, y_hat) / ddof_yy
    #     prec_yy = np.array([np.linalg.inv(cvy) for cvy in cov_yy])
    #     a_out = np.einsum('jkm, mkl -> jlm', cov_xy, prec_yy)
    #     aw = a_out * w[..., None]
    #     # activation of tconv by each signal component of each sample
    #     Sx = np.einsum('ijkl, clk -> ick', y_hat, aw) #shape = [n_batch, ...]

    #     return Sx

    def patterns_pinv_w(self, y, weights, activations, dcov):
        combined_topos = []
        pinv_dmx = np.linalg.pinv(weights['dmx']).T#np.dot(spatial_filters, np.linalg.inv(np.dot(spatial_filters.T, spatial_filters)))
        pinv_wfc = np.linalg.pinv(weights['out_w_flat']).T#np.dot(out_w_flat, np.linalg.inv(np.dot(out_w_flat.T, out_w_flat)))
        pinv_tck = np.linalg.pinv(weights['tconv']).T #np.dot(tconv_kernels, np.linalg.inv(np.dot(tconv_kernels.T, tconv_kernels)))

        #Least square singal estimate in tconv given wfc and fc_activations
        Sx_tconv = np.einsum('jk, ik ->ij', pinv_wfc, activations['fc'])
        Sx_tconv = np.reshape(Sx_tconv, activations['tconv'].shape)

        #Reverse pooling and depthwise convolution for each class
        Sx_dmx = []
        #dc = dcov['input_spatial']
        #n_padding = self.dataset.h_params['n_t']%self.specs['stride']
        for class_y in range(self.out_dim):
            class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
            Sxm = np.squeeze(Sx_tconv[class_ind, :].mean(0, keepdims=True))
            Sxm = np.atleast_2d(Sxm)
            dc = dcov['class_conditional'][class_y]
            combined_topos.append(np.einsum('hi,ij,tj->ht',
                                            dc,
                                            weights['dmx'],
                                            Sxm))
        topos = np.stack(combined_topos, 1)
        return topos


    def patterns_wfc_mean(self, y, weights, activations, dcov):
        combined_topos = []
        #uses y explicitely instead of cov[x,y]
        #accurate but has little to do with the model
        #dc = dcov['input_spatial']
        for class_y in range(self.out_dim):
            #compute mean activation of final layer for each class
            #TODO: -> to self.activations
            class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
            fc_bp_out = (np.dot(activations['fc'].numpy()[class_ind, :],
                               weights['out_w_flat'].T)).mean(0)

            fc_bp_out = fc_bp_out.reshape([activations['tconv'].shape[2],
                                           activations['tconv'].shape[3]],
                                           order='C')
            dc = dcov['class_conditional'][class_y]
            #fc_bp_out = np.maximum(fc_bp_out, 0)
            class_patterns = np.dot(dc,
                                    weights['dmx'])
            cp = np.einsum('ck, ik -> c', class_patterns, fc_bp_out)

            combined_topos.append(cp) # + spatial_biases[class_y]

        topos = np.stack(combined_topos, 1)
        return topos



        #Experiment 4
            #
            #combined_topos = []
            #class_patterns = np.dot(self.dcov, spatial_filters)
            #for class_y in range(self.out_dim):
            #class_y = 0 # true class
                #class_not_y = np.array([i for i in range(self.out_dim) if i!=class_y])
                #class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
                #anti_class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)!=class_y))#[0]
                #Experiment 1
                # fc_bp_out = np.dot(fc_activations.numpy()[class_ind, :], out_w_flat.T).mean(0)
                # fc_anti_out = np.dot(fc_activations.numpy()[anti_class_ind, :], out_w_flat.T).mean(0)
                # fc_bp_out -= fc_anti_out
                # fc_bp_out = fc_bp_out.reshape([pooled_tconv_outputs.shape[2],
                #                                self.dmx.size],
                #                               order='C')

                #fc_bp_out = fc_bp_out.numpy().sum(-1) #- np.mean()
                #fc_bp_out_rect = np.maximum(fc_bp_out, 0)



                #

            # class_patterns = np.dot(dcovs[class_y, ...], spatial_filters)

            # #combined_topos.append(np.einsum('ck, ik -> c', class_patterns, fc_bp_out_rect))
            # combined_topos.append(np.einsum('ck, ik -> c', class_patterns, Sx))
            # topos = np.stack(combined_topos, 1)
        # topos = np.stack(combined_topos, 1)
        # if hasattr(self, 'true_evoked'):
        #     if np.ndim(topos) == 2:
        #         true_corr = np.diag(np.corrcoef(topos.T, self.true_evoked._data.T),
        #                             self.true_evoked._data.shape[-1])
        #     elif np.ndim(topos) == 3:
        #         print(topos.shape)
        #         true_corr = np.diag(np.corrcoef(topos[:,:, 1].T, self.true_evoked._data.T),
        #                             self.true_evoked._data.shape[-1])

        #     print('Mean abs corr: {:.2f}  n_bads: {} MinMax: {:.4f}-{:.4f}'.format(
        #         np.abs(true_corr).mean(),
        #         np.sum(np.abs(true_corr) < .9),
        #         np.min(true_corr), np.max(true_corr),
        #           ))

        # return topos




    def compute_patterns(self, data_path=None):
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
        #vis_dict = None
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


        X, y = [row for row in ds.take(1)][0]

        self.nfft = 128
        ndof = X.shape[0] * self.dataset.h_params['n_t'] - 1

        #combined_topos = []

        #get layer activations
        activations = {}
        # Extract activations
        activations['dmx'] = self.dmx(X)
        activations['tconv'] = self.pool(self.tconv(activations['dmx']))
        activations['fc']  = self.fin_fc(activations['tconv'])

        print(""""Activations: \n
              DMX: {}
              TCONV: {}
              FC_DENSE: {}""".format(
              activations['dmx'].shape,
              activations['tconv'].shape,
              activations['fc'].shape))

        patterns_struct = {'weights' : {'dmx':[], 'tconv':[], 'fc':[], 'tconv_freq_resposes':{}},
                           'ccms' : {'dmx':[], 'tconv':[], 'fc':[], 'input':[], 'dmx_psd':[]},
                           'dcov' : {'input_spatial':[], 'class_conditional':[],
                                     'k-1':[]},
                           'patterns' : {},
                           'spectra': {},
                           }
        weights = self.get_weights()
        spectra = self.get_spectra(weights=weights, activations=activations)

        #CCMs are mean activations of each layer for each class

        dcov = {}
        dcov['input_spatial'] = np.einsum('hijk, hijl -> kl', X, X) / ndof
        dcov['class_conditional'], dcov['k-1']  = self._get_class_conditional_spatial_covariance(X, y)

        ##True evoked
        if self.dataset.h_params['target_type'] == 'float':
            self.true_evoked_data = X.numpy().mean(0)
        elif self.dataset.h_params['target_type'] == 'int':
            y_int = np.argmax(y, 1)
            y_unique = np.unique(y_int)
            evokeds = np.array([X.numpy()[y_int == i, ...].mean(0)
                                for i in y_unique])
            self.true_evoked_data = np.squeeze(evokeds)


        # compute the effect of removing each latent component on the cost function
        self.compwise_losses = self.compute_componentwise_loss(X, y, weights)

        #TODO: class condtitional waveforms -> activations
        #self.waveforms = np.mean(activations['dmx']).T
        patterns = {}
        patterns['cov_xx'] = self.patterns_cov_xx(y, weights, activations, dcov)

        Sx_tconv, Sx_dmx = None, None#self.patterns_cov_xy_hat(X, y, activations, weights)
        # Sx_tconv_ccm = []
        # #Sx_dmx_ccm = []
        # Sx_fc_ccm = []
        # patterns_cxy = []
        # for class_y in range(self.out_dim):
        #     class_ind = tf.squeeze(tf.where(tf.argmax(y, 1)==class_y))#[0]
        #     Sx_tconv_ccm.append(Sx_tconv[class_ind, ...].mean(0))
        #     #Sx_dmx_ccm.append(Sx_dmx[class_ind, ...].mean(0, keepdims=True))
        #     #a = activations['tconv'].numpy()[class_ind, ...].mean(0)*weights['out_weights'][None, ..., class_y]
        #     Sx_fc_ccm.append(activations['fc'].numpy()[class_ind, ...].mean(0, keepdims=True))
        #     # patterns_cxy.append(np.dot(dcov['class_conditional'][class_y],
        #     #                                Sx_dmx[class_ind, ...].mean(0)))
        #     #patterns_cxy.append(np.dot(dcov['input_spatial'],
        #     #                            Sx_dmx[class_ind, ...].mean(0)))
        # ccms = {}
        # ccms['tconv'] = np.concatenate(Sx_tconv_ccm, 0)
        #ccms['dmx'] = np.concatenate(Sx_dmx_ccm, 0)
        #ccms['fc'] = np.concatenate(Sx_fc_ccm, 0)


        #patterns['cov_xy'] = Sx_dmx
        patterns['pinv_w'] = self.patterns_pinv_w(y, weights, activations, dcov).mean(-1)

        patterns['wfc_mean'] = self.patterns_wfc_mean(y, weights, activations, dcov)


        patterns_struct['weights'] = weights
        patterns_struct['spectra'] = spectra
        patterns_struct['dcov'] = dcov
        patterns_struct['ccms'] = {}#ccms
        patterns_struct['patterns'] = patterns

        #compute the effect of removing each latent component on the cost function
        patterns_struct['compwise_loss'] = self.compute_componentwise_loss(X, y, weights)
        #correlation of fc activations to y
        patterns_struct['corr_to_output'] = self.get_output_correlations(activations)
        #self.out_weights = weights['out_weights']
        del X, activations

        return patterns_struct
        # Other
        #self.y_true = np.squeeze(y.numpy())


    def collect_patterns(self, fold=0, n_folds=1, n_comp=1,
                         methods=['weight', 'compwise_loss',
                                  'l2', 'abs_weight', 'output_corr'
                                  ]):
        """
        Compute and store patterns during cross-validation.

        """

        patterns_struct = self.compute_patterns()
        combined_methods = list(patterns_struct['patterns'].keys())
        methods += combined_methods
        if len(self.cv_patterns.items()) == 0 or fold==0:
            n_ch = self.dataset.h_params['n_ch']
            self.cv_patterns = defaultdict(dict)
            for method in methods:
            #n_folds = len(self.dataset.h_params['folds'][0])
                self.cv_patterns[method]['spatial'] = np.zeros([n_ch,
                                             self.y_shape[0],
                                             n_folds])
                self.cv_patterns[method]['temporal'] = np.zeros([self.nfft,
                                                                  self.y_shape[0],
                                                                  n_folds])
                self.cv_patterns[method]['psds'] = np.zeros([self.nfft,
                                                              self.y_shape[0],
                                                              n_folds])
        for method in methods:

            if method not in combined_methods:
                #collect spatial patterns for 'weight' sorting
                topo, spectra, psds = self.single_component_pattern(patterns_struct,
                                                                    sorting=method,
                                                                    n_comp=n_comp)
                self.cv_patterns[method]['spatial'][:, :, fold] = topo
                self.cv_patterns[method]['temporal'][:, :, fold] = spectra
                self.cv_patterns[method]['psds'][:, :, fold] = psds
            else:
                self.cv_patterns[method]['spatial'][:, :, fold] = patterns_struct['patterns'][method]

    def get_spectra(self, weights, activations, nfft=128):
        ##Psds and freq responses
        #  Compute frequency responses and source spectra
        realh = []
        psds = []
        for i, flt in enumerate(weights['tconv'].T):
            flt -= flt.mean()
            #flt -= self.t_conv_biases[i]/self.specs['filter_length']
            ltc = activations['dmx'][:, 0, :, i] - np.mean(activations['dmx'][:, 0, :, i], axis=1, keepdims=True)
            fr, psd = welch(ltc,
                            fs=self.dataset.h_params['fs'],
                            nfft=nfft * 2,
                            nperseg=nfft)
            if len(fr[:-1]) < nfft:
                nfft = len(fr[:-1])

            w, h = freqz(flt, 1, worN=nfft, fs = self.dataset.h_params['fs'])

            psds.append(psd[:, :-1].mean(0))
            realh.append(np.abs(h))

        spectra = {}
        spectra['freq_responses'] = np.array(realh)
        spectra['psds'] = np.array(psds)
        spectra['freqs'] = w
        return spectra



    def get_weights(self):
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

        print(""""Weights: \n
              DMX: {}
              TCONV: {}
              FC_DENSE: {}""".format(weights['dmx'].shape,
              weights['tconv'].shape,
              weights['out_weights'].shape))
        return weights

    def compute_componentwise_loss(self, X, y, weights):



        """Compute component relevances by recursive elimination
        """
        model_weights = self.km.get_weights()
        base_loss, base_performance = self.km.evaluate(X, y, verbose=0)
        # if len(base_performance > 1):
        #    base_performance = bbase_performance[0]
        feature_relevances_loss = []
        n_out_t = weights['out_weights'].shape[0]
        n_out_y = weights['out_weights'].shape[-1]
        zeroweights = np.zeros((n_out_t,))
        losses = np.zeros([self.specs['n_latent'], n_out_y])
        for jj in range(n_out_y):
            #for each class
            for i in range(self.specs["n_latent"]):
                #for each component
                loss_per_component = []

                new_weights = weights['out_weights'].copy()
                new_bias = weights['fc_b'].copy()
                new_weights[:, i, jj] = zeroweights
                new_bias[jj] = 0
                new_weights = np.reshape(new_weights, weights['out_w_flat'].shape)
                model_weights[-2] = new_weights
                model_weights[-1] = new_bias
                self.km.set_weights(model_weights)
                loss = self.km.evaluate(X, y, verbose=0)[0]

                #loss_per_component.append(base_loss - loss)
                losses[i, jj] = base_loss - loss
            #feature_relevances_loss.append(np.array(loss_per_ccomponent))
        return losses


    def get_output_correlations(self, activations):
        """Computes a similarity metric between each of the extracted
        features and the target variable.

        The metric is a Manhattan distance for dicrete targets, and
        Spearman correlation for continuous targets.
        """
        corr_to_output = []
        y_true = activations['fc'].numpy() #y_true.numpy()
        flat_feats = activations['tconv'].numpy().reshape(y_true.shape[0], -1)

        #if self.dataset.h_params['target_type'] in ['float', 'signal']:
        for y_ in y_true.T:

            rfocs = np.array([spearmanr(y_, f)[0] for f in flat_feats.T])
            corr_to_output.append(rfocs.reshape(activations['tconv'].shape[1:]))


        # elif self.dataset.h_params['target_type'] == 'int':
        #     y_true = y_true/np.linalg.norm(y_true, ord=1, axis=0)[None, :]
        #     flat_div = np.linalg.norm(flat_feats, 1, axis=0)[None, :]
        #     flat_feats = flat_feats/flat_div
        #     #print("ff:", flat_feats.shape)
        #     #print("y_true:", y_true.shape)
        #     for y_ in y_true.T:
        #         #print('y.T:', y_.shape)
        #         rfocs = 2. - np.sum(np.abs(flat_feats - y_[:, None]), 0)
        #         corr_to_output.append(rfocs.reshape(activations['tconv'].shape[1:]))

        corr_to_output = np.nanmax(np.concatenate(corr_to_output,0), 1).T

        if np.any(np.isnan(corr_to_output)):
            corr_to_output[np.isnan(corr_to_output)] = 0
        return corr_to_output

    # --- LFCNN plot functions ---

    def plot_evoked_peaks(self, data=None, t=None, class_subset=None,
                          sensor_layout='Vectorview-mag'):
        """
        Plot one spatial topography of class-conditional average of the input.
        If timepoint is not specified it is picked as a maximum RMS for each
        class.


        Parameters
        ----------
        topos : np.array
            [n_ch, n_t, n_classes]
        sensor_layout : TYPE, optional
            DESCRIPTION. The default is 'Vectorview-mag'.

        """
        n = self.out_dim

        if data is None:
            data = self.true_evoked_data
            title = 'True Patterns'
        else:
            title = 'Model-derived patterns'

        if t is None:
            t = np.argmax(np.mean(data**2, axis=0).mean(-1))

        ed = np.stack([data[i, t, :] for i in range(n)], axis=-1)
        assert ed.ndim==2
        topoplot = self.plot_topos(ed, sensor_layout=sensor_layout,
                                   class_subset=class_subset)

        topoplot.figure.suptitle(title)
        topoplot.show()
        return topoplot


    def make_fake_evoked(self, topos, sensor_layout):
        if 'info' not in self.dataset.h_params.keys():
            lo = channels.read_layout(sensor_layout)
            #lo = channels.generate_2d_layout(lo.pos)
            info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
            orig_xy = np.mean(lo.pos[:, :2], 0)
            for i, ch in enumerate(lo.names):
                if info['chs'][i]['ch_name'] == ch:
                    info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/4.5
                    #info['chs'][i]['loc'][4:] = 0
                else:
                    print("Channel name mismatch. info: {} vs lo: {}".format(
                        info['chs'][i]['ch_name'], ch))
        #info['sfreq'] = 1
        fake_evoked = evoked.EvokedArray(topos, info)
        return fake_evoked

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


    def explore_components(self, patterns_struct, sorting='output_corr',
                         integrate='max', info=None, sensor_layout='Vectorview-mag',
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
            class_ind = np.maximum(np.round(event.xdata, 0).astype(int), 0)
            component_ind = np.maximum(np.round(event.ydata).astype(int), 0)


            f1, ax = plt.subplots(2,2)
            f1.set_layout_engine('constrained')
            f1.suptitle("Class: {}  Component: {}"
                  .format(class_ind, component_ind))

            a = np.dot(patterns_struct['dcov']['class_conditional'][class_ind],
                       patterns_struct['weights']['dmx'])

            self.fake_evoked_interactive.data[:, component_ind] = a[:, component_ind]
            self.fake_evoked_interactive.plot_topomap(times=[component_ind],
                                                      axes=ax[0, 0],
                                                      colorbar=False,
                                                      time_format='Spatial activation pattern, [au]'
                                                      )
            psd = psds[component_ind, :].T
            freq_response = freq_responses[component_ind, :].T
            out_psd = psd*freq_response
            psd /= np.maximum(np.sum(psd), 1e-9)
            out_psd /= np.maximum(np.sum(out_psd), 1e-9)
            #freq /= np.maximum(np.sum(out_psd), 1e-9)
            #ax[1,].clear()
            ax[0,1].semilogy(patterns_struct['spectra']['freqs'], psd,
                         label='Input RPS')
            ax[0,1].semilogy(patterns_struct['spectra']['freqs'], out_psd,
                         label='Output RPS')
            #ax[0,1].semilogy(patterns_struct['spectra']['freqs'], freq_response,
            #             label='Freq_response')
            ax[0,1].set_xlim(1, 125.)
            ax[0,1].set_ylim(1e-6, 1.)
            ax[0,1].legend(frameon=False)
            ax[0,1].set_title('Relative power spectra')


            ax[1,0].stem(tconv_kernels[:, component_ind])
            ax[1,0].set_title('Temporal convolution kernel coefficients')

            ax[1,1].plot(F[component_ind, :], 'ks')
            ax[1,1].plot(class_ind, F[component_ind, class_ind], 'rs')
            ax[1,1].set_title('{} per class. Red={}'.format(sorting, class_ind))
            ax[1,1].set_xlabel("Class")



        if sorting == 'weight':
            F = patterns_struct['weights']['out_weights']
        elif sorting == 'output_corr':
            F = patterns_struct['corr_to_output']
        elif sorting == 'compwise_loss':
            F = patterns_struct['compwise_loss']
        if sorting == 'weight_corr':
            F = patterns_struct['weights']['out_weights'] * patterns_struct['corr_to_output']


        if integrate in 'max':
            if F.ndim ==3:
                F = F.max(0)
            inds = np.argmax(F, 0)



        #psds = patterns_struct['spectra']['psds']
        topos = np.dot(patterns_struct['dcov']['input_spatial'],
                       patterns_struct['weights']['dmx'])
        psds = patterns_struct['spectra']['psds']
        freq_responses = patterns_struct['spectra']['freq_responses']
        tconv_kernels = patterns_struct['weights']['tconv']
        self.fake_evoked_interactive = self.make_fake_evoked(topos, sensor_layout)

        vmin = np.min(F)
        vmax = np.max(F)

        f = plt.figure()
        ax = f.gca()

        im = ax.imshow(F, cmap='bone_r', vmin=vmin, vmax=vmax)

        r = [ptch.Rectangle((i - .5, ind - .5), width=1,
                            height=1, angle=0.0, facecolor='none') for i, ind in enumerate(inds)]

        pc = collections.PatchCollection(r, facecolor='none', alpha=.5,
                                          edgecolor='red')
        #ax = f.gca()
        ax.add_collection(pc)
        ax.set_ylabel('Component')
        ax.set_xlabel('Class')
        ax.set_title('Component relevance map: [{}] (Clickable)'.format(sorting))

        f.colorbar(im)
        f.canvas.mpl_connect('button_press_event', _onclick_component)
        f.show()
        return f




    def plot_waveforms(self, patterns_struct, sorting='compwise_loss', tmin=0, class_names=None,
                       bp_filter=False, tlim=None, apply_kernels=False):
        """Plots timecourses of latent components.

        Parameters
        ----------
        tmin : float
            Beginning of the MEG epoch with regard to reference event.
            Defaults to 0.


        sorting : str
            heuristic for selecting relevant components. See LFCNN._sorting
        """
        #if not hasattr(self, 'waveforms'):
        #    self.compute_patterns(self.dataset)

        #if not hasattr(self, 'uorder'):
        order, _ = self._sorting(patterns_struct, sorting)
        self.uorder = order.ravel()
        waveforms = patterns_struct['ccms']['tconv']

            #self.uorder = np.squeeze(order)
        print(self.uorder)
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
                #scaled_waveforms =(scaled_waveforms - scaled_waveforms.mean(-1, keepdims=True))  / (2*scaled_waveforms.std(-1, keepdims=True))
            else:
                #scaling = 3*np.mean(np.std(self.waveforms, -1))

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
#            tcout = np.mean(self.tc_out_unpooled,0).T - self.t_conv_biases[:, None]
#            [ax[1, 0].plot(times, tcouti, color='tab:grey', alpha=.25)
#            for i, tcouti in enumerate(tcout) if i not in self.uorder]

#            [ax[1, 0].plot(times, tcout[uo]) for i, uo in enumerate(self.uorder)]
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

                #rpss.append(h/np.sum(h))
                rpss.append((psd*h)) #%)/np.sum(psd*h)

            [ax[1, 1].plot(self.freqs, rpss[uo], linewidth=2.5, label=class_names[i])
                             for i, uo in enumerate(self.uorder)]
            ax[1, 1].set_xlim(0,90.)
            ax[1, 1].set_title("Relative power, %")
            ax[1, 1].set_xlabel("Frequency, Hz")
            ax[1, 1].legend()
            plt.show()
            return



    def _sorting(self, patterns_struct, sorting='compwise_loss', n_comp=1):
        """Specify which components to plot.

        Parameters
        ----------
        sorting : str
            Sorting heuristics.

            'l2' - plots all components sorted by l2 norm of activations in the
            output layer in descending order.

            'commpwise_loss' - compute the effect of eliminating the latent
            component on the validation loss. Perofrmed for each class
            separately.

            'weight' - plots a single component that has a maximum
            weight for each class in the output layer.

            'output_corr' - plots a single component, which produces a
            feature in the output layer that has maximum correlation
            with each target variable.

            'weight_corr' - plots a single component, has maximum relevance
            value defined as output_layer_weught*correlation.

        Returns:
        --------
        order : list of int
            indices of relevant components

        ts : list of inttopo, freq_response, psd
            indices of relevant timepoints
        """
        order = []
        ts = []
        out_weights = patterns_struct['weights']['out_weights']
        if sorting == 'l2':
            for i in range(self.out_dim):

                self.F = out_weights[..., i].T

                norms = np.linalg.norm(self.F, axis=1, ord=2)
                pat = np.argsort(norms)[-n_comp:]
                order.append(pat)
                ts.append(np.arange(self.F.shape[-1]))
                #ts.append(None)

        elif sorting == 'compwise_loss':
            for i in range(self.out_dim):

                self.F = out_weights[..., i].T
                pat = np.argsort(patterns_struct['compwise_loss'][:, i])
                #take n smallest (largest increase in cost function)
                order.append(pat[:n_comp])
                ts.append(np.arange(self.F.shape[-1]))

        elif sorting == 'abs_weight':
            for i in range(self.out_dim):

                self.F = np.abs(out_weights[..., i].T)
                pat, t = np.where(self.F == np.max(self.F))
                #print('Maximum spearman r:', np.max(self.corr_to_output[..., i].T))
                order.append(pat)
                ts.append(t)

        elif sorting == 'weight':
            n_comp = 1
            for i in range(self.out_dim):
                self.F = out_weights[..., i].T
                pat, t = np.where(self.F == np.max(self.F))
                #print('Maximum weight:', np.max(self.F))
                order.append(pat)
                ts.append(t)

        elif sorting == 'output_corr':

            self.F = patterns_struct['corr_to_output']
            for i in range(self.out_dim):

                print('Maximum r_spear:', np.max(self.F[..., i]))
                pat = np.where(self.F[..., i] == np.max(self.F[..., i]))[0]
                order.append(pat)
                ts.append(np.arange(self.F.shape[-1]))
        else:
            print("Sorting {:s} not implemented".format(sorting))
            return None, None

        order = np.array(order)
        ts = np.array(ts)
        return order, ts


    def single_component_pattern(self, patterns_struct, sorting='compwise_loss',
                                 n_comp=1):
        order, ts = self._sorting( patterns_struct, sorting, n_comp=n_comp)
        c_topos = []
        c_psds = []
        c_frs = []
        #print(sorting, order)
        w = patterns_struct['weights']['dmx']
        #a = np.dot(patterns_struct['dcov']['input_spatial'], w)
        for i, comps in enumerate(order):
            a = np.dot(patterns_struct['dcov']['class_conditional'][i], w)

            c_topos.append(a[:, comps])
            c_psds.append(patterns_struct['spectra']['psds'][comps, :].T,)
            c_frs.append(patterns_struct['spectra']['freq_responses'][comps, :].T)
        topo = np.concatenate(c_topos, axis=-1)
        freq_response = np.concatenate(c_frs, axis=-1)
        psd = np.concatenate(c_psds, axis=-1)
        return topo, freq_response, psd

    def plot_combined_pattern(self, method='combined', sensor_layout=None,
                              names=None, n_comp=1, plot_true_evoked=False):
        if not names:
            names = ['Class {}'.format(i) for i in range(self.y_shape[-1])]


#        cc = np.array([np.corrcoef(self.cv_patterns[:, i, :].T)[i,:]
#               for i in range(self.cv_patterns.shape[1])])
        if len(self.cv_patterns.items() > 0):
            print("Restoring from:", method )
            topos = np.mean(self.cv_patterns[method]['spatial'],
                                     -1)
            filters = np.mean(self.cv_patterns[method]['temporal'],
                                       -1)
            psds = np.mean(self.cv_patterns[method]['psds'],
                                    -1)
            #freqs = self.freqs

        elif method == 'combined':
            topos, filters, psds = self.combined_pattern()

        elif method in ['weight', 'compwise_loss']:
            topos, filters, psds = self.single_pattern(sorting=method,
                                                       n_comp=n_comp)


        freqs = self.freqs

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
                #info['chs'][i]['loc'][4:] = 0
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
                                          time_format="Class %g",
                                          #title='',
                                          #size=1,
                                          outlines='head',
                                          )
        method = "paternnet_rect_cc_covdif_cc_fcactdif"
        #ft.set_size_inches([15,3.5])
        #figname = '-'.join([self.model_path + self.scope, self.dataset.h_params['data_id'], method, "topos.svg"])
        #ft.savefig(figname, format='svg', transparent=True)
        if plot_true_evoked:
            #true_times = np.argmax(np.mean(self.true_evoked_data**2, -1),1)
            #ed = np.stack([self.true_evoked_data[i, tt, :] for i, tt in enumerate(true_times)])
            self.plot_true_evoked(self.true_evoked_data, sensor_layout=sensor_layout)

#         if ncols == 1:
#             plt.figure()
#             ax = plt.gca()
#             normalized_input_rps = psds/np.maximum(np.sum(psds), 1e-6)
#             normalized_output_rps = filters*psds
#             normalized_output_rps /= np.maximum(np.sum(filters*psds), 1e-6)
#             ax.plot(freqs, normalized_input_rps, label='Input RPS')
#             ax.plot(freqs, normalized_output_rps, label='Output RPS')
# #            ax.plot(freqs, filters/np.maximum(np.sum(filters), 1e-6),
# #                    label='Freq response')
#             ax.legend()
#             ax.set_xlim(0, 90)
#             #plt.show()
#         else:
#             f, ax = plt.subplots(1, ncols, sharey=True, constrained_layout=True)
#             #f.constrained_layout()
#             f.set_size_inches([14, 3.])
#             for i in range(ncols):
#                 #ax[i].semilogy(freqs[:-1], combined_filters[:,i], label='Impulse response')
#                 normalized_output_rps = filters[:,i]*psds[:,i]
#                 normalized_output_rps /= np.maximum(np.sum(filters[:,i]*psds[:,i]), 1e-6)
#                 normalized_input_rps = psds[:,i] / np.maximum(np.sum(psds[:,i]), 1e-6)



#                 ax[i].plot(freqs, normalized_input_rps, label='Layer Input')
#                 ax[i].set_title(names[i])
#                 ax[i].plot(freqs, normalized_output_rps, label='Extracted Component')
# #                ax[i].plot(freqs, filters[:, i]/np.maximum(np.sum(filters[:, i]), 1e-6),
# #                    label='Freq response')
#                 #ax[i].plot(freqs[:-1], combined_filters[:,i], label='Output RPS')
#                 #ax[i].plot(freqs[:-1], combined_psds[:,i], label='Input RPS')
#                 ax[i].set_xlim(0, 70)

#                 ax[i].set_xticklabels(ax[i].get_xmajorticklabels(),
#                   fontdict={'fontsize':14})
#                 for spine in ['top', 'right']:
#                     ax[i].spines[spine].set_visible(False)
#                 #ax[i].set_frame_on(False)
#                 #ax[i].set_ylim(1e-5, 1.)
#                 if i == ncols-1:
#                     ax[i].legend()
#                 if i == 0:
#                     ticks = ax[i].get_yticks()
#                     if ticks[0] < 0 and ticks[1] == 0:
#                         ind = np.arange(1, len(ticks), 2, dtype=int)
#                     else:
#                         ind = np.arange(0, len(ticks), 2, dtype=int)

#                     ax[i].set_yticks(ticks[ind])
#                     ax[i].set_yticklabels(ax[i].get_ymajorticklabels(),
#                       fontdict={'fontsize':14})
#                     ax[i].set_ylabel('Relative power, %', fontsize=16)
#             f.supxlabel('Frequency, Hz', fontsize=16)
#         plt.show()
#                 #ax[i].show()





    # def plot_spectra(self, patterns_struct, component_ind, ax, fs=None, loag=False):
    #     """Relative power spectra of a given latent componende before and after
    #        applying the convolution.

    #     Parameters
    #     ----------

    #     patterns_struct :
    #         instance of patterns_struct produced by model.compute_patterns

    #     ax : axes

    #     fs : float
    #         Sampling frequency.

    #     log : bool
    #         Apply log-transform to the spectra.



    #     """

    #     if not fs:
    #        print('Sampling frequency not specified, setting to 1.')
    #        self.fs = 1.

    #     filters = patterns_struct['weights']['tconv'][:, component_ind]
    #     p_input = patterns_struct['spectra']['psds'][component_ind, :]

    #     w, h = freqz(filters, 1, worN=self.nfft)
    #     fr1 = w/np.pi*self.fs/2
    #     h0 = p_input*np.abs(h)


    #     if log:
    #         ax[i, jj].semilogy(fr1, p_input/p_input.sum(),
    #                            label='Filter input RPS')
    #         ax[i, jj].semilogy(fr1, h0/h0.sum(),
    #                            label='Fitler output RPS')
    #         ax[i, jj].semilogy(fr1, np.abs(h)/np.abs(h).sum(),
    #                            label='Freq response RPS')
    #     else:
    #         ax[i, jj].plot(fr1, p_input/p_input.sum(),
    #                        label='Filter input RPS')
    #         ax[i, jj].plot(fr1, h0/h0.sum(), label='Fitler output RPS')
    #         ax[i, jj].plot(fr1, np.abs(h)/np.abs(h).sum(),
    #                        label='Freq response')
    #     #print(np.all(np.round(fr[:-1], -4) == np.round(fr1, -4)))


    # ax[i, jj].set_xlim(0, 75.)
    # #ax[i, jj].set_xlim(0, 75.)
    # if i == 0 and jj == ncols-1:
    #     ax[i, jj].legend(frameon=False)
    # return f


class VARCNN(BaseModel):
    """VAR-CNN.

    For details see [1].

    References
    ----------
        [1] I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None):
        """
        Parameters
        ----------
        Dataset : mneflow.Dataset

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
        super(VARCNN, self).__init__(meta, dataset)

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
    def __init__(self, meta, dataset=None):
        self.scope = 'fbcsp-ShallowNet'
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
        super(FBCSP_ShallowNet, self).__init__(meta, dataset)

    def build_graph(self):

        """Temporal conv_1 25 10x1 kernels"""
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

    References
    ----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for
        decoding MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    def __init__(self, meta, dataset=None):
        """

        Parameters
        ----------
        Dataset : mneflow.Dataset

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
            Convolution padding. Defaults to 'SAME'.}

        stride : int
        Stride of the max pooling layer. Defaults to 1.
        """
        #self.scope = 'lflstm'
        self.scope = 'lf-cnn-lstm'
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
        super(LFLSTM, self).__init__(meta, dataset)


    def build_graph(self):
        
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
    def __init__(self, meta, dataset=None):
        self.scope = 'deep4'
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
        super(Deep4, self).__init__(meta, dataset)

    def build_graph(self):
        self.scope = 'deep4'

        inputs = tf.transpose(self.inputs,[0,3,2,1])

        tconv1 = DepthwiseConv2D(
                        kernel_size=(1, self.specs['filter_length']),
                        depth_multiplier = self.specs['n_latent'],
                        strides=1,
                        padding=self.specs['padding'],
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
            Dropout coefficient.

    References
    ----------
    [3] V.J. Lawhern, et al., EEGNet: A compact convolutional neural
    network for EEG-based brain–computer interfaces 10 J. Neural Eng.,
    15 (5) (2018), p. 056013

    [4] Original EEGNet implementation by the authors can be found at
    https://github.com/vlawhern/arl-eegmodels
    """
    def __init__(self, meta, dataset=None):
        self.scope = 'eegnet8'
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
        super(EEGNet, self).__init__(meta, dataset)
        

    def build_graph(self):


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

# class SimpleNet(LFCNN):
#     """
#         Petrosyan, A., Sinkin, M., Lebedev, M. A., & Ossadtchi, A.  Decoding and interpreting cortical signals with
#         a compact convolutional neural network, 2021, Journal of Neural Engineering, 2021,
#         https://doi.org/10.1088/1741-2552/abe20e
#     """
#     def __init__(self, Dataset, specs=None):
#         if specs is None:
#             specs=dict()
#         super().__init__(Dataset, specs)

#     def build_graph(self):
#         self.dmx = DeMixing(size=self.specs['n_latent'], nonlin=tf.identity,
#                             axis=3, specs=self.specs)
#         self.dmx_out = self.dmx(self.inputs)

#         self.tconv = LFTConv(
#             size=self.specs['n_latent'],
#             nonlin=self.specs['nonlin'],
#             filter_length=self.specs['filter_length'],
#             padding=self.specs['padding'],
#             specs=self.specs
#         )
#         self.tconv_out = self.tconv(self.dmx_out)

#         self.envconv = LFTConv(
#             size=self.specs['n_latent'],
#             nonlin=self.specs['nonlin'],
#             filter_length=self.specs['filter_length'],
#             padding=self.specs['padding'],
#             specs=self.specs
#         )

#         self.envconv_out = self.envconv(self.tconv_out)
#         self.pool = lambda X: X[:, :, ::self.specs['pooling'], :]

#         self.pooled = self.pool(self.envconv_out)

#         dropout = Dropout(
#             self.specs['dropout'],
#             noise_shape=None
#         )(self.pooled)

#         self.fin_fc = FullyConnected(size=self.out_dim, nonlin=tf.identity,
#                             specs=self.specs)

#         y_pred = self.fin_fc(dropout)

#         return y_pred

#     def compute_patterns(self, data_path=None, *, output='patterns'):

#         if not data_path:
#             print("Computing patterns: No path specified, using validation dataset (Default)")
#             ds = self.dataset.val
#         elif isinstance(data_path, str) or isinstance(data_path, (list, tuple)):
#             ds = self.dataset._build_dataset(
#                 data_path,
#                 split=False,
#                 test_batch=None,
#                 repeat=True
#             )
#         elif isinstance(data_path, Dataset):
#             if hasattr(data_path, 'test'):
#                 ds = data_path.test
#             else:
#                 ds = data_path.val
#         elif isinstance(data_path, tf.data.Dataset):
#             ds = data_path
#         else:
#             raise AttributeError('Specify dataset or data path.')

#         X, y = [row for row in ds.take(1)][0]

#         self.out_w_flat = self.fin_fc.w.numpy()
#         self.out_weights = np.reshape(
#             self.out_w_flat,
#             [-1, self.dmx.size, self.out_dim]
#         )
#         self.out_biases = self.fin_fc.b.numpy()
#         self.feature_relevances = self.componentwise_loss(X, y)
#         self.branchwise_loss(X, y)

#         # compute temporal convolution layer outputs for vis_dics
#         tc_out = self.pool(self.tconv(self.dmx(X)).numpy())

#         # compute data covariance
#         X = X - tf.reduce_mean(X, axis=-2, keepdims=True)
#         X = tf.transpose(X, [3, 0, 1, 2])
#         X = tf.reshape(X, [X.shape[0], -1])
#         self.dcov = tf.matmul(X, tf.transpose(X))

#         # get spatial extraction fiter weights
#         demx = self.dmx.w.numpy()

#         kern = np.squeeze(self.tconv.filters.numpy()).T

#         X = X.numpy().T

#         patterns = []
#         X_filt = np.zeros_like(X)
#         for i_comp in range(kern.shape[0]):
#             for i_ch in range(X.shape[1]):
#                 x = X[:, i_ch]
#                 X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
#             patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
#         self.patterns = np.array(patterns).T


#         if 'patterns' in output:
#             if 'old' in output:
#                 self.patterns = np.dot(self.dcov, demx)
#             else:
#                 patterns = []
#                 X_filt = np.zeros_like(X)
#                 for i_comp in range(kern.shape[0]):
#                     for i_ch in range(X.shape[1]):
#                         x = X[:, i_ch]
#                         X_filt[:, i_ch] = np.convolve(x, kern[i_comp, :], mode="same")
#                     patterns.append(np.cov(X_filt.T) @ demx[:, i_comp])
#                 self.patterns = np.array(patterns).T
#         else:
#             self.patterns = demx

#         self.lat_tcs = np.dot(demx.T, X.T)

#         del X

#         #  Temporal conv stuff
#         self.filters = kern.T
#         self.tc_out = np.squeeze(tc_out)
#         self.corr_to_output = self.get_output_correlations(y)

#     def plot_patterns(
#         self, sensor_layout=None, sorting='l2', percentile=90,
#         scale=False, class_names=None, info=None
#     ):
#         order, ts = self._sorting(sorting)
#         self.uorder = order.ravel()
#         l_u = len(self.uorder)
#         if info:
#             info.__setstate__(dict(_unlocked=True))
#             info['sfreq'] = 1.
#             self.fake_evoked = evoked.EvokedArray(self.patterns, info, tmin=0)
#             if l_u > 1:
#                 self.fake_evoked.data[:, :l_u] = self.fake_evoked.data[:, self.uorder]
#             elif l_u == 1:
#                 self.fake_evoked.data[:, l_u] = self.fake_evoked.data[:, self.uorder[0]]
#             self.fake_evoked.crop(tmax=float(l_u))
#             if scale:
#                 _std = self.fake_evoked.data[:, :l_u].std(0)
#                 self.fake_evoked.data[:, :l_u] /= _std
#         elif sensor_layout:
#             lo = channels.read_layout(sensor_layout)
#             info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
#             orig_xy = np.mean(lo.pos[:, :2], 0)
#             for i, ch in enumerate(lo.names):
#                 if info['chs'][i]['ch_name'] == ch:
#                     info['chs'][i]['loc'][:2] = (lo.pos[i, :2] - orig_xy)/3.
#                     #info['chs'][i]['loc'][4:] = 0
#                 else:
#                     print("Channel name mismatch. info: {} vs lo: {}".format(
#                         info['chs'][i]['ch_name'], ch))

#             self.fake_evoked = evoked.EvokedArray(self.patterns, info)

#             if l_u > 1:
#                 self.fake_evoked.data[:, :l_u] = self.fake_evoked.data[:, self.uorder]
#             elif l_u == 1:
#                 self.fake_evoked.data[:, l_u] = self.fake_evoked.data[:, self.uorder[0]]
#             self.fake_evoked.crop(tmax=float(l_u))
#             if scale:
#                 _std = self.fake_evoked.data[:, :l_u].std(0)
#                 self.fake_evoked.data[:, :l_u] /= _std
#         else:
#             raise ValueError("Specify sensor layout")


#         if np.any(self.uorder):
#             nfilt = max(self.out_dim, 8)
#             nrows = max(1, l_u//nfilt)
#             ncols = min(nfilt, l_u)
#             f, ax = plt.subplots(nrows, ncols, sharey=True)
#             plt.tight_layout()
#             f.set_size_inches([16, 3])
#             ax = np.atleast_2d(ax)

#             for ii in range(nrows):
#                 fake_times = np.arange(ii * ncols,  (ii + 1) * ncols, 1.)
#                 vmax = np.percentile(self.fake_evoked.data[:, :l_u], 95)
#                 self.fake_evoked.plot_topomap(
#                     times=fake_times,
#                     axes=ax[ii],
#                     colorbar=False,
#                     vmax=vmax,
#                     scalings=1,
#                     time_format="Branch #%g",
#                     title='Patterns ('+str(sorting)+')',
#                     outlines='head',
#                 )

#     def branchwise_loss(self, X, y):
#         model_weights_original = self.km.get_weights().copy()
#         base_loss, _ = self.km.evaluate(X, y, verbose=0)

#         losses = []
#         for i in range(self.specs["n_latent"]):
#             model_weights = model_weights_original.copy()
#             spatial_weights = model_weights[0].copy()
#             spatial_biases = model_weights[1].copy()
#             temporal_biases = model_weights[3].copy()
#             env_biases = model_weights[5].copy()
#             spatial_weights[:, i] = 0
#             spatial_biases[i] = 0
#             temporal_biases[i] = 0
#             env_biases[i] = 0
#             model_weights[0] = spatial_weights
#             model_weights[1] = spatial_biases
#             model_weights[3] = temporal_biases
#             model_weights[5] = env_biases
#             self.km.set_weights(model_weights)
#             losses.append(self.km.evaluate(X, y, verbose=0)[0])
#         self.km.set_weights(model_weights_original)
#         self.branch_relevance_loss = base_loss - np.array(losses)

#     def plot_branch(
#         self,
#         branch_num: int,
#         info: Info,
#         params: Optional[list[str]] = ['input', 'output', 'response']
#     ):
#         info.__setstate__(dict(_unlocked=True))
#         info['sfreq'] = 1.
#         sorting = np.argsort(self.branch_relevance_loss)[::-1]
#         data = self.patterns[:, sorting]
#         filters = self.filters[:, sorting]
#         relevances = self.branch_relevance_loss - self.branch_relevance_loss.min()
#         relevance = sorted([np.round(rel/relevances.sum(), 2) for rel in relevances], reverse=True)[branch_num]
#         self.fake_evoked = evoked.EvokedArray(data, info, tmin=0)
#         fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
#         fig.tight_layout()

#         self.fs = self.dataset.h_params['fs']

#         out_filter = filters[:, branch_num]
#         _, psd = welch(self.lat_tcs[branch_num], fs=self.fs, nperseg=self.fs * 2)
#         w, h = (lambda w, h: (w, h))(*freqz(out_filter, 1, worN=self.fs))
#         frange = w / np.pi * self.fs / 2
#         z = lambda x: (x - x.mean())/x.std()

#         for param in params:
#             if param == 'input':
#                 finput = psd[:-1]
#                 finput = z(finput)
#                 ax2.plot(frange, finput - finput.min(), color='tab:blue')
#             elif param == 'output':
#                 foutput = np.real(finput * h * np.conj(h))
#                 foutput = z(foutput)
#                 ax2.plot(frange, foutput - foutput.min(), color='tab:orange')
#             elif param == 'response':
#                 fresponce = np.abs(h)
#                 fresponce = z(fresponce)
#                 ax2.plot(frange, fresponce - fresponce.min(), color='tab:green')
#             elif param == 'pattern':
#                 fpattern = finput * np.abs(h)
#                 fpattern = z(fpattern)
#                 ax2.plot(frange, fpattern - fpattern.min(), color='tab:pink')

#         ax2.legend([param.capitalize() for param in params])
#         ax2.set_xlim(0, 100)

#         fig.suptitle(f'Branch {branch_num}', y=0.95, x=0.2, fontsize=30)
#         fig.set_size_inches(10, 5)
#         self.fake_evoked.plot_topomap(
#             times=branch_num,
#             axes=ax1,
#             colorbar=False,
#             scalings=1,
#             time_format="",
#             outlines='head',
#         )

#         return fig


