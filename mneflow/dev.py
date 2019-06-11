#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:20:39 2019

@author: zubarei1
"""
import tensorflow as tf
from .models import Model
from .layers import DeMixing, VARConv, Dense, weight_variable, bias_variable
from .utils import scale_to_baseline
import numpy as np

class VARDAE(Model):
    """ VAR-CNN
    
    For details see [1].
    
    Paramters:
    ----------
    var_params : dict
                    {
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
                        }
    References:
    -----------
        [1]  I. Zubarev, et al., Adaptive neural network classifier for decoding 
        MEG signals. Neuroimage. (2019) May 4;197:425-434
    """
    
    def __init__(self,h_params, params, var_params):
        super().__init__(h_params, params)
        self.specs = var_params
        self.h_params['architecture'] = 'var-cnn'
    def _build_graph(self):
        self.demix = DeMixing(n_ls=self.specs['n_ls'])
        
        self.tconv1 =VARConv(scope="conv", n_ls=self.specs['n_ls'],  
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length'], 
                              stride=self.specs['stride'], 
                              pooling=self.specs['pooling'], 
                              padding=self.specs['padding'])
        self.tconv2 =VARConv(scope="conv", n_ls=self.specs['n_ls']//2,  
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length']//2, 
                              stride=self.specs['stride'], 
                              pooling=self.specs['pooling'], 
                              padding=self.specs['padding'])
        
        self.encoding_fc = Dense(size=self.specs['df'], 
                       nonlin=tf.identity, dropout=self.rate)
        
        encoder = self.encoding_fc(self.tconv2(self.tconv1(self.demix(self.X))))
        

        self.deconv = DeConvLayer(n_ls=self.specs['n_ls'],
                                  y_shape=self.h_params['y_shape'], 
                                  filter_length=self.specs['filter_length'],
                                  flat_out=False)
                    #here be z                                     
        decoder = self.deconv(encoder)
        
        return decoder
        
    def _set_optimizer(self):
        loss = tf.reduce_sum((self.y_-self.y_pred)**2)
        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in 
                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
                            if 'weights' in var.name]
            
        reg = self.params['l1_lambda'] * tf.add_n(regularizers) 
        cost = loss + reg # add regularization
        train_step = tf.train.AdamOptimizer(self.params['learn_rate']).minimize(cost)
        var = tf.reduce_sum(self.y_**2)
        pve = 1 - loss/var
        return train_step, pve, cost, var
    

class DeConvLayer():
    """DeConvolution Layer"""
    def __init__(self, n_ls, y_shape, scope="deconv", flat_out=False, 
                 filter_length=5):        
        self.scope = scope
        self.n_ch, self.n_t = y_shape
        self.size = n_ls
        self.filter_length = filter_length
        self.flat_out = flat_out
    def __call__(self,x):
         with tf.name_scope(self.scope):
            while True:
                try: # project onto latent space
                    latent_activations =  tf.nn.relu(tf.tensordot(x,self.W,axes=[[1],[0]]) + self.b_in)
                    print('x_reduced', latent_activations.shape)
                    #add "time" dimension
                    x_perm = tf.expand_dims(latent_activations,1)                    
                    #x_perm = tf.expand_dims(x_perm,-1)                    
                    print('x_perm',x_perm.shape) #(?,1,self.size)
                    
                    conv_ = tf.einsum('lij,ki->lkj',x_perm,self.filters)# +self.b
                    print('deconv:', conv_.shape) #(?,n_t,self.size)
                    out = tf.einsum('lkj,jm->lmk',conv_,self.demixing)
                    out = out#+self.b_out
                    if self.flat_out:
                        return tf.reshape(out,[-1,self.n_t*self.n_ch])
                    else: 
                        print(out.shape)
                        return out
                    
                    
                except(AttributeError):
                    self.W = weight_variable((x.get_shape()[1].value, self.size))
                    self.b_in = bias_variable([self.size])
                    self.filters = weight_variable([self.n_t,1])
                    self.b = bias_variable([self.size])
                    self.demixing = weight_variable((self.size,self.n_ch))
                    self.b_out = bias_variable([self.n_ch])
                    
                    print(self.scope,'_init')
                    
                    
                    
    #    def train_epochs(self):       
#        """Trains the model"""
#        ii=0
#        min_val_loss =  np.inf
#        patience_cnt = 0
#        self.sess.run(tf.global_variables_initializer())
#        for i in range(self.params['n_epochs']):
#            self.sess.run([self.train_iter.initializer,self.val_iter.initializer])
#            self.train_dataset.shuffle(buffer_size=10000)
#            while True:
#                
#                try:
#                    
#                    _, t_loss,acc = self.sess.run([self.train_step,self.cost,self.accuracy],feed_dict={self.handle: self.training_handle, self.rate:self.params['dropout']})
#                    ii+=1
#                except tf.errors.OutOfRangeError:
#                    break            
#            
#            if i %self.params['eval_step']==0:
#                self.v_acc, v_loss = self.sess.run([self.accuracy,self.cost],feed_dict={self.handle: self.validation_handle, self.rate:1})
#                
#                if min_val_loss >= v_loss + self.params['min_delta']:
#                    min_val_loss = v_loss
#                    v_acc = self.v_acc
#                    self.saver.save(self.sess, ''.join([self.model_path,self.h_params['architecture'],'-',self.h_params['data_id']]))
#                    print('epoch %d, train_loss %g, train acc %g val loss %g, val acc %g' % (i, t_loss,acc, v_loss, self.v_acc))
#                else:
#                    patience_cnt +=1
#                    print('* Patience count {}'.format(patience_cnt))
#                    print('epoch %d, train_loss %g, train acc %g val loss %g, val acc %g' % (i, t_loss,acc, v_loss, self.v_acc))
#                if patience_cnt >= self.params['patience']:
#                    print("early stopping...")
#                    #restore the best model
#                    self.saver.restore(self.sess, ''.join([self.model_path,self.h_params['architecture'],'-',self.h_params['data_id']]))                
#                    #self.v_acc, v_loss = self.sess.run([self.accuracy,self.cost],feed_dict={self.handle: self.validation_handle})
#                    print('stopped at: epoch %d, val loss %g, val acc %g' % (i,  min_val_loss, v_acc))
#                    break
    
       
#            print('processing epochs')


def preprocess_continuous(inputs, val_size=.1, overlap=False, segment=False, stride=1, scale=False):
    
    """
    
    Parameters:
    -----------
    inputs : list of ndarrays
            data to be preprocessed
            
    val_size : float
            proportion of data to use as validation set
    
    segment : int or False
            length of segment into which to split the data in time samples
    
    overlap : bool 
            whether to use overlapping segments, False by default
    
    stride : int
            stride in time samples for overlapping segments, defaults to 1
    
    Returns:
    --------
    segments : tuple 
            of size len(inputs)*2 traning and vaildation data split into 
            (overlapping) segments
    
    Example:
    -------
    X_train, X_val = preprocess_continuous(X, val_size=.1, overlap=False, 
                                           segment=False, stride=None)
    
    Returns two continuous data segments split into training and validation 
    sets (90% and 10%, respectively). Validation set is defined as a single 
    randomly picked segment of the data 
    with length euqal to int(X.shape[-1]*val_size)
    
    X_train, X_val, Y_train, Y_val = train_test_split_cont([X,Y],val_size=.1,
    segment=500)
    
    Returns training and validation sets for two input arrays split into 
    non-overlapping segments of 500 samples. This requires last 
    dimentions of all inputs to be equal.
    
    
    X_train, X_val = preprocess_continuous(X, val_size=.1, overlap=True, 
                                           segment=500, stride=25)
    
    Returns training and validation sets split into overlapping segments 
    of 500 samples with stride of 25 time samples.
            
    """
    if not isinstance(inputs,list):
        inputs = list(inputs)
    split_datasets = train_test_split_cont(inputs,test_size=val_size)
    if overlap:
        segments = sliding_augmentation(split_datasets, segment=segment, stride=stride)
    elif segment:
        segments = segment_raw(inputs, segment, tile_epochs=True)
    else:
        segments = split_datasets
    if scale:
        segments = (scale_to_baseline(s,baseline=None,crop_baseline=False) for s in segments)
    return segments

   
def sliding_augmentation(datas, labels=None,segment=500,stride=1, tile_epochs=True):
    """Return an image of x split in overlapping time segments"""
    output = []
    if not isinstance(datas,list):
        datas = [datas]
    #raw_len = datas[0].shape[-1]
    for x in datas:
        while x.ndim < 3:
            x = np.expand_dims(x,0)
        #assert x.shape[-1] == raw_len
        n_epochs, n_ch, n_t = x.shape
        nrows = n_t - segment + 1            
        a,b,c = x.strides
        x4D = np.lib.stride_tricks.as_strided(x,shape=(n_epochs,n_ch,nrows,segment),strides=(a,b,c,c))
        x4D = x4D[:,:,::stride,:]
        if tile_epochs:
            if labels:
                labels = np.tile(labels,x4D.shape[2])
            x4D = np.moveaxis(x4D,[2],[0])
            x4D = x4D.reshape([n_epochs*x4D.shape[0],n_ch,segment],order='C')
        #assert labels.shape[0] == x4D.shape[0]
        output.append(x4D)
#        print(x.shape,x4D.shape)
#        print(np.all(x[0,0,:segment]==x4D[0,0,...]))
#        print(np.all(x[0,1,:segment]==x4D[0,1,...]))
#        print(np.all(x[0,0,stride:segment+stride]==x4D[1,0,...]))
#        print(np.all(x[0,0,2*stride:segment+2*stride]==x4D[2,0,...]))
        #print(np.all(x[0,0,stride+segment_length:segment_length+stride*2]==x4D[2,0,...]))
    if labels: 
        output.append(labels)
    return output

def segment_raw(inputs, segment, tile_epochs=True):
    out = []
    if not isinstance(inputs, list):
        inputs = [inputs]
    raw_len = inputs[0].shape[-1]
    for x in inputs:
        assert x.shape[-1] == raw_len
        while x.ndim < 3:
            x = np.expand_dims(x,0)
        #inp_orig = x.copy()
        orig_shape = x.shape[:-1]
        leftover = raw_len%(segment)
        #print(orig_shape)
        print('dropping:', str(leftover), ':', leftover//2, '+', leftover-leftover//2)
        crop_start = leftover//2
        crop_stop = -1*(leftover-leftover//2)
        x = x[...,crop_start:crop_stop]
        x = x.reshape([*orig_shape,-1,segment])
        if tile_epochs:
            x = np.moveaxis(x,[-2],[0])
            x = x.reshape([orig_shape[0]*x.shape[0],*orig_shape[1:],segment],order='C')
        out.append(x)
        #print(x.shape)
#        print(np.all(inp_orig[0,0,crop_start:crop_start+segment]==x[0,0,...]))
#        print(np.all(inp_orig[0,1,crop_start:crop_start+segment]==x[0,1,...]))
#        print(np.all(inp_orig[0,0,crop_start+segment:crop_start+segment*2]==x[1,0,...]))
#        print(np.all(inp_orig[0,0,crop_start+segment*3:crop_start+segment*4]==x[3,0,...]))
    return out


def train_test_split_cont(inputs,test_size):
    out = []
    if not isinstance(inputs, list):
        inputs = [inputs]
    raw_len = inputs[0].shape[-1]
    test_samples = int(test_size*raw_len)
    test_start = np.random.randint(test_samples//2,int(raw_len-test_samples*1.5))    
    test_indices = np.arange(test_start,test_start+test_samples)
    for x in inputs:
        assert x.shape[-1] == raw_len
        x_test = x[...,test_indices]
        x_train = np.delete(x,test_indices,axis=-1)
        out.append(x_train)
        out.append(x_test)
    return out  