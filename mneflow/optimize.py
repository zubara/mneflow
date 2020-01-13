#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module specifies Optimizer object
"""
import tensorflow as tf


class Optimizer(object):
    """
    Optimizer object.

    """
    def __init__(self, learn_rate=3e-4, l1_lambda=0, l2_lambda=0,
                 l1_scope=['weights'], l2_scope=['weights'],
                 task='classification', class_weights=None,
                 norm_constraints=False):
        """
        Parameters
        ----------
        learn_rate : float
                    learning rate
        l1_lambda : float, optional
                    coefficient for sparse penaly on the model weights
        l2_lambda : float, optional
                    coefficient for l2 on the model weights
        l1_scope : list of str, optional
                   specifies which layers l1-penalty applies to
                   'dmx' - de-mixing
                   'fc' - fully-cnnected
                   'tconv' - temopral convolution
                   'weights' - all weights
                   deafulats to 'weights'
        l2_lambda : list of str, optional
                    specifies which layers l2-penalty applies to
                    'dmx' - de-mixing
                    'fc' - fully-cnnected
                    'tconv' - temopral convolution
                    'weights' - all weights
                    deafulats to 'weights'
        norm_constraints : None, dict *Not Implemented*
                    whether to apply norm constraints on filters
                    available keys {'dmx', 'tconv', 'fc'}
                    values {'l1', 'l2', 'nonneg'}



        task : str, {'classification', 'regression'}
        """
        self.params = dict(learn_rate=learn_rate, l1_lambda=l1_lambda,
                           l2_lambda=l2_lambda, task=task)
        self.class_weights = class_weights
        self.l1_scope = l1_scope
        self.l2_scope = l2_scope
        self.norm_constraints = norm_constraints

        # TODO : add cost function options,
        # TODO : class balance
        # TODO : regularization options,
        # TODO : performance metric options

    def _set_optimizer(self, y_pred, y_true):

        """
        Initializes the optimizer part of the computational graph

        This method can be overriden a for custom optimizer

        Parameters
        ----------
        y_pred : tf.Tensor
                        predictions of the target varible, output of the
                        computational graph

        y_true : tf.Tensor
                        target_variable, output of dataset.iterator

        Returns
        --------
        train_step : tf.Operation
                    training operation


        performance : tf.Tensor
                    performance metric

        cost : tf.Tensor
                    cost (objective) function output

        prediciton : tf.Tensor
                    model output
        """
        # Define cost, and performance metric, treat prediction if needed
        if self.params['task'] == 'classification':
            prediction = tf.nn.softmax(y_pred)
            #cost_function = tf.losses.sparse_softmax_cross_entropy
            #cost_function =
            if self.class_weights:
                print('Adjusting for imbalanced classes')
                class_weights = tf.constant(self.class_weights,
                                            dtype=tf.float32)
                #print(y_true, class_weights)
                weights = tf.gather(class_weights,y_true)
            else:
                weights = 1
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=y_pred,
                                                weights=weights))
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true,1))
            performance = tf.reduce_mean(tf.cast(correct_prediction,
                                                 tf.float32), name='accuracy')
        elif self.params['task'] == 'regression':
            #weights = tf.maximum(.5, 2*tf.abs(y_true))
            #loss = tf.losses.absolute_difference(labels=y_true, predictions=y_pred, weights=weights)  ##,, reduction='weighted_sum'
            loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)#, weights=weights
            # R^2
            total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))), axis=0)
            unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)), axis=0)
            print(unexplained_error.shape)
            performance = tf.subtract(1., tf.reduce_mean(tf.div(unexplained_error, total_error)))

            prediction = y_pred

        #  Regularization
        if self.params['l1_lambda'] == 0 and self.params['l2_lambda'] == 0:
            cost = loss
        if self.params['l1_lambda'] > 0:

            reg = [tf.reduce_sum(tf.abs(var))
                   for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                   if any(scope in var.name for scope in self.l1_scope)]
            print('L1 penalty applied to', ', '.join(self.l1_scope))
            cost = loss + self.params['l1_lambda'] * tf.add_n(reg)

        if self.params['l2_lambda'] > 0:
            reg = [tf.nn.l2_loss(var) for var in
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                   if any(scope in var.name for scope in self.l2_scope)]
            print('L2 penalty applied to', ', '.join(self.l2_scope))
            cost = loss + self.params['l2_lambda'] * tf.add_n(reg)
        train_step = tf.train.AdamOptimizer(learning_rate=self.params['learn_rate'])
#        if isinstance(self.norm_constraints, dict):
#            grads_and_vars = train_step.compute_gradients(loss, )
#            capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=123.0, axes=0), gv[1])
#                         for gv in grads_and_vars]
#            # Ask the optimizer to apply the capped gradients
#            optimizer = optimizer.apply_gradients(capped_grads_and_vars)
#            print('Not implemented')

        #  Optimizer
        #else:
        train_step = train_step.minimize(cost)
        return train_step, performance, cost, prediction
