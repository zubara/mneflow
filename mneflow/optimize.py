#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:00:22 2019

@author: zubarei1
"""
import tensorflow as tf


class Optimizer(object):
    def __init__(self, params):
        self.params = params
        # TODO : add cost function options,
        # TODO : regularization options,
        # TODO : performance metric options

    def _set_optimizer(self, y_pred, y_true):
        """Initializes the optimizer part of the computational graph

        This method can be overriden a for custom optimizer

        Inputs:
        -------
        y_pred : tf.Tensor
                        predictions of the target varible, output of the
                        computational graph

        y_true : tf.Tensor
                        target_variable, output of dataset.iterator

        Returns:
        --------
        train_step : tf.Operation
                    training operation


        performance_metric : tf.Tensor
                    performance metric
        cost : tf.Tensor
                    cost (objective) function

        p_classes : tf.Tensor
                    logits for classification tasks
        """

        p_classes = tf.nn.softmax(y_pred)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

#        add L1 regularization
        if self.params['l1_lambda'] > 0:
            regularizers1 = [tf.reduce_sum(tf.abs(var)) for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'weights' in var.name]
            reg1 = self.params['l1_lambda'] * tf.add_n(regularizers1)
            cost = cost + reg1

#        if self.params['l2_lambda'] > 0:
#            regularizers2 = [tf.reduce_sum(var**2) for var in
#                                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#                                               if 'weights' in var.name]# 'dense'
#            reg2 = self.params['l2_lambda'] * tf.add_n(regularizers2)
#            cost = cost + reg2

#        Optimizers, accuracy etc
        train_step = tf.train.AdamOptimizer(self.params['learn_rate']).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')
        return train_step, accuracy, cost, p_classes
