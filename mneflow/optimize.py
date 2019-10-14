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
                 task='classification', class_weights=None):
        """
        Parameters
        ----------
        learn_rate : float
                    learning rate
        l1_lambda : float, optional
                    coefficient for sparse penaly on the model weights
        l2_lambda : float, optional
                    coefficient for l2 on the model weights
        task : str, {'classification', 'regression'}


        """
        self.params = dict(learn_rate=learn_rate, l1_lambda=l1_lambda,
                           l2_lambda=l2_lambda, task=task)
        self.class_weights = class_weights
        self.l1_scope = l1_scope
        self.l2_scope = l2_scope

        # TODO : add cost function options,
        # TODO : class balance
        # TODO : regularization options,
        # TODO : performance metric options

    def set_optimizer(self, y_pred, y_true):

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
            cost_function = tf.losses.sparse_softmax_cross_entropy
            if self.class_weights:
                print('Adjusting for imbalanced classes')
                class_weights = tf.constant(self.class_weights,
                                            dtype=tf.float32)
                #print(y_true, class_weights)
                weights = tf.gather(class_weights,y_true)
            else:
                weights = 1
            loss = tf.reduce_mean(cost_function(labels=y_true, logits=y_pred,
                                                weights=weights))
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), y_true)
            performance = tf.reduce_mean(tf.cast(correct_prediction,
                                                 tf.float32), name='accuracy')
        elif self.params['task'] == 'ae':
            loss = tf.losses.cosine_distance(y_true, y_pred, axis=1)
            #loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred, reduction='weighted_sum')
            cross_cov = tf.reduce_mean(y_true*y_pred, axis=1)
            print('cross_cov', cross_cov.shape)
            #print(y_pred.shape, y_true.shape)
            #autocov = tf.reduce_mean(y_true*y_true, axis=1)
            #loss = 1 - tf.reduce_mean(cross_cov/autocov)
            performance = tf.reduce_mean(cross_cov)
            print('PERF', performance.shape)
            #loss = tf.reduce_mean((y_true-y_pred)**2)
            #var_ = tf.reduce_mean((y_true)**2)

            #performance = 1-loss/var_
            prediction = y_pred
            print(loss.shape, performance.shape)

        elif self.params['task'] == 'regression':
            loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred, weights=tf.abs(y_true), reduction='weighted_sum')
            #mn, var_ = tf.nn.moments(y_true)
            #var_ = tf.reduce_mean(y_true**2)

            #print(var_)
            performance = tf.reduce_sum((y_true-y_pred)**2)/tf.reduce_sum(y_true**2)#loss/tf.reduce_mean(var_)
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


        #  Optimizer
        train_step = tf.train.AdamOptimizer(learning_rate=self.params['learn_rate']).minimize(cost)
        # train_step = tf.train.RMSPropOptimizer(learning_rate=self.params['learn_rate']).minimize(cost)

        return train_step, performance, cost, prediction
