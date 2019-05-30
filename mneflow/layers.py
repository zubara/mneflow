# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:40:09 2017

@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
import tensorflow as tf
from numpy import prod, sqrt
import functools

def compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))
    
def stack_layers(*args):
    return functools.partial(functools.reduce,compose)(*args)
    
def vgg_block(n_layers, layer, kwargs):                   
    layers = []
    for i in range(n_layers):
        if i>0: 
            kwargs['inch']=kwargs['n_ls']
        layers.append(layer(**kwargs))
    layers.append(tf.layers.batch_normalization)
    layers.append(max_pool)
    return stack_layers(layers[::-1])

class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="fc", size=None, dropout=.5,
                 nonlin=tf.identity):
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlin = nonlin

    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    if len(x.shape)>2:#flatten if input is not 2d array
                        x = tf.reshape(x,[-1,self.flatsize])
                    return self.nonlin(tf.matmul(x, self.w) + self.b, name='out')
                except(AttributeError):
                    if len(x.shape)>2:
                        self.flatsize = prod(x.shape[1:]).value
                    else:
                        self.flatsize = x.shape[1].value
                    self.w = weight_variable((self.flatsize, self.size),name='sparse_')
                    self.b = bias_variable([self.size])
                    self.w = tf.nn.dropout(self.w, self.dropout)
                    print(self.scope,'_init')

class ConvLayer():
    """VAR/LF Convolutional Layer"""
    def __init__(self, scope="conv", n_ls=32,  nonlin_in=tf.identity, nonlin_out=tf.nn.relu,
                 filter_length=7, stride=1, pool=2, conv_type='var'): #,dropout=1.
        self.scope = '-'.join([conv_type,scope])
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.pool = pool
        self.nonlin_in = nonlin_in
        self.nonlin_out = nonlin_out
        #self.dropout = dropout
        self.conv_type = conv_type
        
    def __call__(self,x):
         with tf.name_scope(self.scope):
            while True:
                try: # reuse weights if already initialized
                    x_reduced = self.nonlin_in(tf.tensordot(x,self.W,axes=[[1],[0]], 
                                                            name='de-mix') + self.b_in)
                    if 'var' in self.conv_type:
                        conv_ = self.nonlin_out(conv1d(x_reduced, self.filters, 
                                                       stride=1) + self.b)
                        conv_ = tf.expand_dims(conv_, -1)
                    elif 'lf' in self.conv_type:
                        x_reduced = tf.expand_dims(x_reduced, -2)
                        conv_ = self.nonlin_out(tf.nn.depthwise_conv2d(x_reduced, 
                                                self.filters,padding='SAME', 
                                                strides=[1,1,1,1]) + self.b)
                        conv_ = tf.transpose(conv_,perm=[0,1,3,2])
                    conv_ = max_pool(conv_,ksize=[1,self.pool,1,1],
                                     strides=[1,self.stride,1,1])#
                    return tf.transpose(conv_[...,0],perm=[0,2,1], name='out')
                except(AttributeError):
                    self.W = weight_variable((x.get_shape()[1].value, self.size),
                                             name='sparse_')
                    #self.W = tf.nn.dropout(self.W, self.dropout)
                    self.b_in = bias_variable([self.size])
                    if self.conv_type == 'var':
                        self.filters = weight_variable([self.filter_length,self.size,self.size],
                                                       name='dense_')
                    elif self.conv_type == 'lf':
                        self.filters = weight_variable([self.filter_length,1,self.size,1],
                                                       name='dense_')
                    self.b = bias_variable([self.size])
                    print(self.scope,'_init')                    
                    
def spatial_dropout(x, keep_prob, seed=1234):
    # x is a convnet activation with shape BxWxHxF where F is the 
    # number of feature maps for that layer
    # keep_prob is the proportion of feature maps we want to keep
    num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]
    # get some uniform noise between keep_prob and 1 + keep_prob
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(num_feature_maps,
                                       seed=seed,
                                       dtype=x.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Reshape to multiply our feature maps by this tensor correctly
    binary_tensor = tf.reshape(binary_tensor, 
                               [-1, 1, 1, tf.shape(x)[3]])
    # Zero out feature maps where appropriate; scale up to compensate
    ret = tf.div(x, keep_prob) * binary_tensor
    return ret

class ConvDSV():
    """Standard/Depthwise/Spearable Convolutional Layer constructor"""
    def __init__(self, scope="conv", n_ls=None, nonlin_out=None, inch=None, 
                 domain = None, padding = 'SAME', filter_length=5, stride=1, 
                 pooling=2, dropout=.5, 
                 conv_type='depthwise'):
        self.scope = '-'.join([conv_type,scope,domain])
        self.padding = padding
        self.domain = domain
        self.inch = inch
        self.dropout = dropout # keep_prob
        self.size = n_ls
        self.filter_length = filter_length
        self.stride = stride
        self.pool_wind_size = pooling
        self.nonlin_out = nonlin_out
        self.conv_type = conv_type
    def __call__(self,x):
         with tf.name_scope(self.scope):
            while True:
                try:
                    if self.conv_type == 'depthwise':
                        conv_ = self.nonlin_out(tf.nn.depthwise_conv2d(x, self.filters, 
                                                strides=[1,self.stride,1,1],
                                                padding=self.padding) + self.b)
                    elif self.conv_type == 'separable':
                        conv_ = self.nonlin_out(tf.nn.separable_conv2d(x, self.filters, 
                                                self.pwf, strides=[1,self.stride,1,1],
                                                padding=self.padding) + self.b)
                    elif self.conv_type == '2d':
                        conv_ = self.nonlin_out(tf.nn.conv2d(x, self.filters, 
                                                strides=[1,self.stride,self.stride,1],
                                                padding=self.padding) + self.b)
#                    conv_ = max_pool(conv_,ksize=[1,self.pool_wind_size,1,1],
#                                     strides=[1,1,1,1])
                    return conv_
                except(AttributeError):
                    if self.domain == 'time':
                        self.filters = weight_variable([1,self.filter_length,
                                                        self.inch,self.size],
                                                        name='weights')
                                                        
                    elif self.domain == 'space':
                        self.filters = weight_variable([self.filter_length,1,
                                                        self.inch,self.size],
                                                        name='weights')
                    elif self.domain == '2d':
                        self.filters = weight_variable([self.filter_length[0],
                                                        self.filter_length[1],
                                                        self.inch,self.size],
                                                        name='weights',method='he')
                    self.b = bias_variable([self.size])
                    if self.conv_type == 'separable':
                        self.pwf = weight_variable([1,1,self.inch*self.size,
                                                    self.size],name='sep-pwf')
                    
                    print(self.scope,'_init')
                    



#        
    
# Initialize weight variable from normal distribution with SD 0.1
def weight_variable(shape,name='',method='he'):
  #initial = tf.truncated_normal(shape, stddev=.03)
  if method == 'xavier':
      xavf = 2/sum(prod(shape[:-1]))
      initial = xavf*tf.random_uniform(shape,minval=-.5,maxval=.5)
  elif method == 'he':
      hef =  sqrt(6. / prod(shape[:-1]))
      initial = hef*tf.random_uniform(shape,minval=-1.,maxval=1.)
  else:
      initial = tf.truncated_normal(shape, stddev=.1)
  #initial = tf.linspace(-.1,1.,prod(shape))
  #initial = tf.reshape()
  return tf.Variable(initial,trainable=True,name=name+'weights')

# Initialize bias variable as constant 0.1 (avoids "dead neurons" when using e.g. ReLU units)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,trainable=True,name='bias')

def conv1d(x, W,stride=1):
  return tf.nn.conv1d(x, W, stride=stride,data_format='NHWC',padding='SAME')

# Maxpooling function using 2x2 blocks
def max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1]):
  return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')
