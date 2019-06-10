# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:39:39 2018


@author: Ivan Zubarev, ivan.zubarev@aalto.fi
"""
from .layers import ConvDSV, ConvLayer, Dense, vgg_block, LFTConv, VARConv, DeMixing #DeConvLayer,
import tensorflow as tf
import numpy as np
from sklearn.covariance import ledoit_wolf
#temporary measure


class Model(object):
    """Parent class for all MNEflow models
        
    Contains TFRecords dataset from TFRecords files using the metadata.
    Initializes separate iterators for training and validation datasets
    provides fast and memory efficient data handling.
    
    Custom models can be built easily by overriding _build_graph and 
    _set_optimizer methods. 
    
    """
    def __init__(self, h_params, params):
        self.h_params = h_params
        self.params = params
        self.model_path = params['model_path']
        self.rate = tf.placeholder(tf.float32,name='rate')# params['dropout']
        
        #with tf.device("/cpu:0"):
    def init_datasets(self, pick_classes=False, combine_classes=False):
        """Initialize tf.data.TFRdatasets and associated iterators"""
        self.sess = tf.Session()        
        self.train_dataset  = tf.data.TFRecordDataset(self.h_params['train_paths'])
        self.val_dataset  = tf.data.TFRecordDataset(self.h_params['val_paths'])
        # Parse the record into tensors.
        self.train_dataset = self.train_dataset.map(self._parse_function)
        self.val_dataset = self.val_dataset.map(self._parse_function)
        if pick_classes:
            self.classes = pick_classes #class labels to leave in the dataset as list, otherwise use all classes
            self.train_dataset = self.train_dataset.filter(self.select_classes)
            self.val_dataset = self.val_dataset.filter(self.select_classes)
        #if combine_classes:
            
        self.train_dataset = self.train_dataset.map(self.unpack)#.repeat()
        self.val_dataset = self.val_dataset.map(self.unpack)#.repeat()
        # Generate batches
        self.train_dataset = self.train_dataset.batch(self.params['n_batch']).repeat()
        val_size = self.get_n_samples(self.h_params['val_paths'])
        self.val_dataset = self.val_dataset.batch(val_size).repeat()
        #create iterators
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.train_iter = self.train_dataset.make_initializable_iterator()
        self.val_iter = self.val_dataset.make_initializable_iterator()
        self.training_handle = self.sess.run(self.train_iter.string_handle())
        self.validation_handle = self.sess.run(self.val_iter.string_handle())
        self.sess.run(self.train_iter.initializer)
        self.sess.run(self.val_iter.initializer)
        self.X, self.y_ = self.iterator.get_next()
        
    def get_n_samples(self,path):
        ns = 0
        for fn in path:
          for record in tf.python_io.tf_record_iterator(fn):
             ns += 1
        return ns
        
    def init_model(self):
        """Initialize computational graph"""
        self.y_pred = self._build_graph()
        print('y_pred:',self.y_pred.shape)
        #assert self.y_pred.shape == self.y_.shape
        
    def init_optimizer(self):
        """initialize optimizer"""
        #self.rate = 
        (self.train_step, self.accuracy,
        self.cost, self.p_classes) = self._set_optimizer()
        
        
    def build(self,pick_classes=None):
        """Compile a model"""
        self.init_datasets(pick_classes)
        self.init_model()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.init_optimizer()
        
        print('Initialization complete!')
        
    def _parse_function(self,example_proto):
        if self.h_params['task'] == 'classification':
            keys_to_features = {'X':tf.FixedLenFeature((self.h_params['n_ch'],self.h_params['n_t']), tf.float32),
                                  'y': tf.FixedLenFeature((), tf.int64, default_value=0)}
        else:
            keys_to_features = {'X':tf.FixedLenFeature((self.h_params['n_ch'],self.h_params['n_t']), tf.float32),
                                  'y': tf.FixedLenFeature(self.h_params['y_shape'], tf.float32)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features
    
    def select_classes(self,sample):
        if self.classes:
            return tf.reduce_any(tf.equal(sample['y'],self.classes))
        else:
            return tf.constant(True,dtype=tf.bool)
        
    def unpack(self,sample):
        return sample['X'],sample['y']#/2
        
        
            
    def _build_graph(self):
        """Build computational graph. 
        
        This method can be overriden in a sub-class for customized architecture
        """
        
        print('Specify a model. Set to linear classifier!')
        self.h_params['architecture'] = 'template_linear'
        fc_1 = Dense(size=self.h_params['n_classes'], nonlin=tf.identity, dropout=self.params['dropout'])
        y_pred = fc_1(self.X)
        return y_pred
    
    def _set_optimizer(self):
        """Initializes the optimizer part of the computational graph
    
        This method can be overriden in a sub-class for custom optimizer
        """
            
        p_classes = tf.nn.softmax(self.y_pred)
        loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, 
                                                                           logits=self.y_pred))
        
        #add L1 regularization
        regularizers = [tf.reduce_sum(tf.abs(var)) for var in
                                           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
                                           if 'weights' in var.name]# 'dense'
        reg = self.params['l1_lambda'] * tf.add_n(regularizers) 
        cost = loss + reg# add regularization
        
        #Optimizers, accuracy etc
        train_step = tf.train.AdamOptimizer(self.params['learn_rate']).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), self.y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        return train_step, accuracy, cost, p_classes
                
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run([self.train_iter.initializer,self.val_iter.initializer])
        min_val_loss =  np.inf
        
        patience_cnt = 0
        for i in range(self.params['n_iter']+1):
            _, = self.sess.run([self.train_step],feed_dict={self.handle: self.training_handle, self.rate:self.params['dropout']})
            t_loss,acc = self.sess.run([self.cost,self.accuracy],feed_dict={self.handle: self.training_handle, self.rate:1})
            if i % self.params['eval_step'] == 0:
                self.train_dataset.shuffle(buffer_size=10000)
                self.v_acc, v_loss = self.sess.run([self.accuracy,self.cost],feed_dict={self.handle: self.validation_handle, self.rate:1.})
                
                if min_val_loss >= v_loss + self.params['min_delta']:
                    min_val_loss = v_loss
                    v_acc = self.v_acc
                    self.saver.save(self.sess, ''.join([self.model_path,self.h_params['architecture'],'-',self.h_params['data_id']]))
                    #print('epoch %d, train_loss %g, train acc %g val loss %g, val acc %g' % (i, t_loss,acc, v_loss, self.v_acc))
                else:
                    patience_cnt +=1
                    print('* Patience count {}'.format(patience_cnt))
                if patience_cnt >= self.params['patience']:
                    print("early stopping...")
                    self.saver.restore(self.sess, ''.join([self.model_path,self.h_params['architecture'],'-',self.h_params['data_id']]))                
                    print('stopped at: epoch %d, val loss %g, val acc %g' % (i,  min_val_loss, v_acc))
                    break
                print('i %d, tr_loss %g, tr_acc %g v_loss %g, v_acc %g' % (i, t_loss,acc, v_loss, self.v_acc))
        
    def load(self):
        """Loads a pretrained model. 
        
        To load a specific model the model object should be initialized using 
        the corresponding metadata and computational graph"""
        
        self.saver.restore(self.sess,''.join([self.model_path,self.h_params['architecture'],'-',self.h_params['data_id']]))
        self.v_acc = self.sess.run([self.accuracy],feed_dict={self.handle: self.validation_handle})
            
                
    def evaluate_performance(self, data_path, batch_size=None):
        """Compute intial test accuracy"""
        test_dataset  = tf.data.TFRecordDataset(data_path).map(self._parse_function)  
        batch_size = self.get_n_samples([data_path])
        test_dataset = test_dataset.batch(batch_size)
        #else:
            #test_dataset = test_dataset.batch(batch_size)
            
        test_iter = test_dataset.make_initializable_iterator()
        acc = []
        self.sess.run(test_iter.initializer)
        test_handle = self.sess.run(test_iter.string_handle())
        while True:
            try:
                acc.append(self.sess.run(self.accuracy,feed_dict={self.handle: test_handle, self.rate:1.}))
            except tf.errors.OutOfRangeError:
                print('Finished: acc: %g +\- %g' % (np.mean(acc), np.std(acc)))
                break
                    
        return np.mean(acc)
             
    def evaluate_realtime(self,data_path, batch_size=None, step_size=1):
        """Compute test accuracy batch by batch with incremental updates"""
        prt_batch_pred = []
        prt_logits = []
        n_test_points = batch_size//step_size
        count=0
        
        test_dataset  = tf.data.TFRecordDataset(data_path).map(self._parse_function)  
        test_dataset = test_dataset.batch(step_size)
        test_iter = test_dataset.make_initializable_iterator()
        self.sess.run(test_iter.initializer)
        test_handle = self.sess.run(test_iter.string_handle())
        
        while True:
            try:
                self.load()
                count += 1
                preds = 0
                for jj in range(n_test_points):
                    pred, probs = self.sess.run([self.correct_prediction,
                                                 self.p_classes],
                                                 feed_dict={self.handle: test_handle,
                                                            self.rate:1})
                    self.sess.run(self.train_step,
                                  feed_dict={self.handle: test_handle,
                                             self.rate:self.params['dropuot']})
                    #for _ in range(step_size):
                     #self.sess.run(self.train.)
                    preds +=np.mean(pred)
                    prt_logits.append(probs)
                prt_batch_pred.append(preds/n_test_points)
                #print(count)
            except tf.errors.OutOfRangeError:
                print('prt_done: count: %d, acc: %g +\- %g' % (count,np.mean(prt_batch_pred), np.std(prt_batch_pred)))
                break
#                    for _ in range(self.params['test_upd_batch']):
#                        self.train_step.run(feed_dict={self.X: test_batch[0], self.y_: test_batch[1], 
#                                                      self.keep_prob: self.params['dropout']})
        return prt_batch_pred,np.concatenate(prt_logits)

#class LFCNN(Model):
#    """ LF-CNN
#    
#    This model implements several methods for interpreataion of learned 
#    parameters. For details see [1].
#    
#    Paramters:
#    ----------
#    lf_params : dict
#                    {
#                    n_ls : int
#                        number of latent components 
#                        Defaults to 32
#                    
#                    filter_length : int
#                        length of temporal filters in the temporal convolution 
#                        layer. Defaults to 7
#                        
#                    stride : int
#                        stride of the max pooling layer. Defaults to 1
#                        
#                    pooling : int
#                        pooling factor of the max pooling layer. Defaults to 2
#                        }
#    References:
#    -----------
#        [1] I. Zubarev, et al., Adaptive neural network classifier for decoding 
#        MEG signals. Neuroimage. (2019) May 4;197:425-434
#    """
#    def __init__(self,h_params, params, lf_params):
#        super().__init__(h_params, params)
#        self.specs = lf_params
#        self.h_params['architecture'] = 'lf-cnn'
#    def _build_graph(self):
#        
#        self.conv = ConvLayer(n_ls=self.specs['n_ls'], 
#                         filter_length=self.specs['filter_length'],
#                         pool=self.specs['pooling'],
#                         stride = self.specs['stride'],
#                         nonlin_in=tf.identity,
#                         nonlin_out=tf.nn.relu,
#                         conv_type='lf')
#                
#        self.fin_fc = Dense(size=self.h_params['n_classes'], 
#                       nonlin=tf.identity,
#                       dropout=self.rate)
#        y_pred = self.fin_fc(self.conv(self.X))
#        return y_pred
#    
#    def plot_out_weihts(self,):
#        from matplotlib import pyplot as plt
#        times = np.arange(self.h_params['n_t'])[::self.specs['pooling']]/float(self.h_params['fs'])
#        t_step = np.diff(times)[0]
#        f,ax = plt.subplots(1,self.h_params['n_classes'])
#        for i in range(self.h_params['n_classes']):
#            F = self.out_weights[...,i]
#            pat,t= np.where(F==np.max(F))
#            ax[i].pcolor(times,np.arange(self.specs['n_ls']),F,cmap='bone_r')#,vmin=0.0,vmax=.25
#            ax[i].plot(times[t]+.5*t_step,pat+.5,markeredgecolor='red',markerfacecolor='none',marker='s',markersize=10,markeredgewidth=2)
#        plt.show()
#    
#    def compute_patterns(self, megdata=None,output='patterns'):
#        vis_dict = {self.handle:self.training_handle, self.rate:1}
#        spatial = self.sess.run(self.conv.W,feed_dict=vis_dict)
#        self.filters = np.squeeze(self.sess.run(self.conv.filters,feed_dict=vis_dict))
#        self.patterns = spatial
#
#        if 'patterns' in output:
#            if isinstance(self.val_dataset,tf.data.Dataset):
#                data = self.sess.run(self.X,feed_dict={self.handle:self.validation_handle,self.rate:1.})           
#            data = data.transpose([0,2,1])
#            data = data.reshape([-1,data.shape[-1]])
#            self.dcov,_ = ledoit_wolf(data)
#            self.patterns = np.dot(self.dcov,self.patterns)
#        if 'full' in output:
#            #lat_cov,_ = ledoit_wolf(np.dot(data,spatial))
#            lat_cov,_ = ledoit_wolf(np.dot(data,spatial))
#            self.lat_prec = np.linalg.inv(lat_cov)
#            self.patterns = np.dot(self.patterns,self.lat_prec)
#        self.out_weights, self.out_biases = self.sess.run([self.fin_fc.w, self.fin_fc.b],feed_dict=vis_dict)
#        self.out_weights = np.reshape(self.out_weights,[self.specs['n_ls'],-1,self.h_params['n_classes']])
#    
#    
#    def plot_patterns(self,sensor_layout='Vectorview-grad',sorting='l2',spectra=True,fs=None,scale=False):
#        from mne import channels, evoked, create_info
#        import matplotlib.pyplot as plt
#        from scipy.signal import freqz
#        self.ts=[]
#        lo = channels.read_layout(sensor_layout)
#        info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
#        self.fake_evoked = evoked.EvokedArray(self.patterns,info)
#        nfilt = min(self.specs['n_ls']//self.h_params['n_classes'],8)
#        #define which components to plot       
#        if sorting == 'l2':
#            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=2))            
#        elif sorting == 'l1':
#            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=1))            
#        elif sorting == 'contribution':
#            #One col per class
#            nfilt = 3
#            order = []
#            for i in range(self.h_params['n_classes']):
#                inds = np.argsort(self.out_weights[...,i].sum(-1))[::-1]
#                order+=list(inds[:nfilt])
#            order = np.array(order)
#        elif sorting == 'abs':
#            nfilt = self.h_params['n_classes']
#            order = []
#            for i in range(self.h_params['n_classes']):
#                pat= np.argmax(np.abs(self.out_weights[...,i].sum(-1)))
#                
#                order.append(pat)
#                #print(pat.shape)
#                #self.ts.append(t)
#            order = np.array(order)
#        elif sorting == 'best':
#            #One pattern per class
#            nfilt = self.h_params['n_classes']
#            order = []
#            for i in range(self.h_params['n_classes']):
#                pat,t= np.where(self.out_weights[...,i]==np.max(self.out_weights[...,i]))
#                order.append(pat[0])
#                self.ts.append(t)
#            order = np.array(order)
#        elif sorting == 'best_neg':
#            nfilt=self.h_params['n_classes']
#            #One row per class
#            order = []
#            for i in range(self.h_params['n_classes']):
#                pat= np.argmin(self.out_weights[...,i].sum(-1))
#                order.append(pat)
#        elif sorting == 'worst':
#            nfilt=self.h_params['n_classes']
#            #One row per class
#            order = []
#            weight_sum = np.sum(np.abs(self.out_weights).sum(-1),-1)
#            pat= np.argsort(weight_sum)
#            #print(weight_sum[pat])
#            order = np.array(pat[:nfilt])
#
#        elif isinstance(sorting,list):
#            nfilt = len(sorting)
#            order = np.array(sorting)
#        
#        else:
#            order = np.arange(self.specs['n_ls'])
#        self.fake_evoked.data[:,:len(order)] = self.fake_evoked.data[:,order]
#        if scale:
#            self.fake_evoked.data[:,:len(order)] /= self.fake_evoked.data[:,:len(order)].max(0)
#        self.fake_evoked.data[:,len(order):] *=0
#        self.out_filters = self.filters[:,order]
#        order = np.array(order)
#        
#        if spectra:
#                z = 2            
#        else: 
#            z = 1
#        nrows = max(1,len(order)//nfilt)
#        ncols = min(nfilt,len(order))
#        #print('rows:',nrows, ' cols:', ncols)
#        f, ax = plt.subplots(z*nrows, ncols,sharey=True)
#        f.set_size_inches([16,9])
#        ax = np.atleast_2d(ax)
#        for i in range(nrows):
#            if spectra:
#                for jj,flt in enumerate(self.out_filters[:,i*ncols:(i+1)*ncols].T):
#                    w, h = freqz(flt,1)
#                    ax[z*i+1, jj].plot(w/np.pi*self.h_params['fs']/2, np.abs(h)) 
#                    #ax[z*i+1, jj].set_ylim(0,1.5)
#            self.fake_evoked.plot_topomap(times=np.arange(i*ncols,  (i+1)*ncols, 1.), 
#                                     axes=ax[z*i], colorbar=False, #vmin=0, 
#                                     vmax=np.percentile(self.fake_evoked.data[:,:len(order)],99), 
#                                     scalings=1,
#                                     time_format='')
#                           
#        #f.show()
#        #f.tight_layout()
#        return 


#class VARCNN(Model):
#
#    
#    def __init__(self,h_params, params, var_params):
#        super().__init__(h_params, params)
#        self.specs = var_params
#        self.h_params['architecture'] = 'var-cnn'
#    def _build_graph(self):
#        self.conv = ConvLayer(n_ls=self.specs['n_ls'], 
#                         filter_length=self.specs['filter_length'],
#                         pool=self.specs['pooling'], 
#                         stride = self.specs['stride'],
#                         nonlin_in=tf.identity,
#                         nonlin_out=tf.nn.relu,
#                         conv_type='var')          
#        self.fin_fc = Dense(size=self.h_params['n_classes'], 
#                       nonlin=tf.identity,
#                       dropout=self.rate)
#        y_pred = self.fin_fc(self.conv(self.X))
#        return y_pred
            
class VGG19(Model):
    def __init__(self,h_params, params, vgg_params):
        super().__init__(h_params, params)
        
        vgg_params = dict(n_ls=self.params['n_ls'], nonlin_out=tf.nn.relu, 
                        inch=1, padding = 'SAME', filter_length=(3,3), domain='2d', 
                       stride=1, pooling=1, conv_type='2d')        
        self.specs = vgg_params
        self.h_params['architecture'] = 'vgg19'
        
        
    def _build_graph(self):
        X1 = tf.expand_dims(self.X,-1)
        #inch = 1
        if X1.shape[1]==306:
            X1 = tf.concat([X1[:,0:306:3,:],X1[:,1:306:3,:],X1[:,2:306:3,:]],axis=3)
            self.specs['inch'] = 3
            #print(X1.shape)
        
        vgg1 = vgg_block(2,ConvDSV,self.specs)
        out1 = vgg1(X1)
        
        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *=2
        vgg2 = vgg_block(2,ConvDSV,self.specs)
        out2 = vgg2(out1)
#            
        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *=2
        vgg3 = vgg_block(4,ConvDSV,self.specs)
        out3 = vgg3(out2)
        
        self.specs['inch'] = self.specs['n_ls']
        self.specs['n_ls'] *=2
        vgg4 = vgg_block(4,ConvDSV,self.specs)
        out4 = vgg4(out3)
#            
        self.specs['inch'] = self.specs['n_ls']
        vgg5 = vgg_block(4,ConvDSV,self.specs)
        out5 = vgg5(out4)
        
#            
        fc_1 = Dense(size=4096, nonlin=tf.nn.relu,dropout=self.rate)
        fc_2 = Dense(size=4096, nonlin=tf.nn.relu,dropout=self.rate)
        fc_out = Dense(size=self.h_params['n_classes'], nonlin=tf.identity,dropout=self.rate)
        y_pred = fc_out(fc_2(fc_1(out5)))
        return y_pred
            
class EEGNet(Model):
    """EEGNet
    
    Parameters
    ----------
    eegnet_params : dict
                    {
                    n_ls : int
                        number of (temporal) convolution kernrels in the first layer. 
                        Defaults to 8
                    
                    filter_length : int
                        length of temporal filters in the first layer. 
                        Defaults to 32
                        
                    stride : int
                        stride of the average polling layers. Defaults to 4
                        
                    pooling : int
                        pooling factor of the average polling layers. Defaults to 4
                        }
        
    ---------
    [1] V.J. Lawhern, et al., EEGNet: A compact convolutional neural network 
    for EEG-based brain–computer interfaces 10 J. Neural Eng., 15 (5) (2018), 
    p. 056013 
    
    [2]  Original  EEGNet implementation by the authors can be found at 
    https://github.com/vlawhern/arl-eegmodels
    """   
    
    def __init__(self,h_params, params,  eegnet_params):
        super().__init__(h_params, params)
        self.specs = eegnet_params
        self.h_params['architecture'] = 'eegnet'
        
    def _build_graph(self):
        #Dropout = tf.keras.layers.Dropout
        #from tensorflow.keras.constraints import max_norm
        X1 = tf.expand_dims(self.X,-1)                    
        vc1 = ConvDSV(n_ls=self.specs['n_ls'], nonlin_out=tf.identity, inch=1,
                      filter_length=self.specs['filter_length'], domain='time', stride=1, 
                      pooling=1, conv_type='2d')
        vc1o = vc1(X1)
        bn1 = tf.layers.batch_normalization(vc1o)
        dwc1 = ConvDSV(n_ls=1, nonlin_out=tf.identity, inch=self.specs['n_ls'], 
                       padding = 'VALID', filter_length=self.h_params['n_ch'], 
                       domain='space',  stride=1, pooling=1,
                       conv_type='depthwise')
        dwc1o = dwc1(bn1)
        bn2 = tf.layers.batch_normalization(dwc1o)
        out2 = tf.nn.elu(bn2)
        out22 = tf.nn.dropout(out2, self.rate)#Dropout(self.rate)(out6)
        
        #out22 = spatial_dropout(out2,self.keep_prob)
        sc1 = ConvDSV(n_ls=self.specs['n_ls'], nonlin_out=tf.identity, 
                      inch=self.specs['n_ls'], filter_length=self.specs['filter_length']//4, 
                      domain='time', stride=1, pooling=1,conv_type='separable')
        
        sc1o = sc1(out22)
        bn3 = tf.layers.batch_normalization(sc1o)
        out3 = tf.nn.elu(bn3)
        out4 = tf.nn.avg_pool(out3,[1,1,self.specs['pooling'],1],[1,1,self.specs['stride'],1], 'SAME')
        out44 = tf.nn.dropout(out4, self.rate)#Dropout(self.rate)(out6)Dropout(self.rate)(out4)
        #out44 = spatial_dropout(out4,self.keep_prob)
        sc2 = ConvDSV(n_ls=self.specs['n_ls']*2, nonlin_out=tf.identity, inch=self.specs['n_ls'],
                      filter_length=self.specs['filter_length']//4, domain='time', 
                      stride=1, pooling=1,
                      conv_type='separable')
        sc2o = sc2(out44)
        bn4 = tf.layers.batch_normalization(sc2o)
        out5 = tf.nn.elu(bn4)
        out6 = tf.nn.avg_pool(out5,[1,1,self.specs['pooling'],1],[1,1,self.specs['stride'],1], 'SAME') #fix typo here out5
        out66 = tf.nn.dropout(out6, self.rate)#Dropout(self.rate)(out6)
#        flatten      = tf.keras.layers.Flatten(name = 'flatten')(out66) 
#        dense        = tf.keras.layers.Dense(self.h_params['n_classes'], name = 'dense', 
#                         kernel_constraint = max_norm(.25))(flatten)
#        y_pred      = tf.keras.layers.Activation('softmax', name = 'softmax')(dense)
        
        out7 = tf.reshape(out66,[-1,np.prod(out66.shape[1:])])#16*4
        fc_out = Dense(size=self.h_params['n_classes'], nonlin=tf.identity,dropout=self.rate)
        y_pred = fc_out(out7)
        return y_pred
    

class LFCNN(Model):
    """Refactored LFCNN"""
    def __init__(self,h_params, params, lf_params):
        super().__init__(h_params, params)
        self.specs = lf_params
        self.h_params['architecture'] = 'lf-cnn'
        
    def _build_graph(self):
        self.demix = DeMixing(n_ls=self.specs['n_ls'])
        
        self.tconv1 = LFTConv(scope="conv", n_ls=self.specs['n_ls'],  
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length'], 
                              stride=self.specs['stride'], 
                              pooling=self.specs['pooling'], 
                              padding=self.specs['padding'])
        
        self.fin_fc = Dense(size=self.h_params['n_classes'], 
                       nonlin=tf.identity, dropout=self.rate)
        
        y_pred = self.fin_fc(self.tconv1(self.demix(self.X)))        
        return y_pred
    
    def plot_out_weihts(self,):
        from matplotlib import pyplot as plt
        
        f,ax = plt.subplots(1,self.h_params['n_classes'])
        for i in range(self.h_params['n_classes']):
            F = self.out_weights[...,i]
            times = self.specs['stride']*np.arange(F.shape[-1])/float(self.h_params['fs'])
            t_step = np.diff(times)[0]
            #print(F.shape)
            pat,t= np.where(F==np.max(F))
            ax[i].pcolor(times,np.arange(self.specs['n_ls']),F,cmap='bone_r')#,vmin=0.0,vmax=.25
            ax[i].plot(times[t]+.5*t_step,pat+.5,markeredgecolor='red',markerfacecolor='none',marker='s',markersize=10,markeredgewidth=2)
        plt.show()
    
    def compute_patterns(self, megdata=None,output='patterns'):
        vis_dict = {self.handle:self.training_handle, self.rate:1}
        spatial = self.sess.run(self.demix.W,feed_dict=vis_dict)
        self.filters = np.squeeze(self.sess.run(self.tconv1.filters,feed_dict=vis_dict))
        self.patterns = spatial

        if 'patterns' in output:
            if isinstance(self.val_dataset,tf.data.Dataset):
                data = self.sess.run(self.X,feed_dict={self.handle:self.validation_handle,self.rate:1.})           
            data = data.transpose([0,2,1])
            data = data.reshape([-1,data.shape[-1]])
            self.dcov,_ = ledoit_wolf(data)
            self.patterns = np.dot(self.dcov,self.patterns)
        if 'full' in output:
            #lat_cov,_ = ledoit_wolf(np.dot(data,spatial))
            lat_cov,_ = ledoit_wolf(np.dot(data,spatial))
            self.lat_prec = np.linalg.inv(lat_cov)
            self.patterns = np.dot(self.patterns,self.lat_prec)
        self.out_weights, self.out_biases = self.sess.run([self.fin_fc.w, self.fin_fc.b],feed_dict=vis_dict)
        self.out_weights = np.reshape(self.out_weights,[self.specs['n_ls'],-1,self.h_params['n_classes']])
    
    
    def plot_patterns(self,sensor_layout='Vectorview-grad',sorting='l2',spectra=True,fs=None,scale=False):
        from mne import channels, evoked, create_info
        import matplotlib.pyplot as plt
        from scipy.signal import freqz
        self.ts=[]
        lo = channels.read_layout(sensor_layout)
        info = create_info(lo.names, 1., sensor_layout.split('-')[-1])
        self.fake_evoked = evoked.EvokedArray(self.patterns,info)
        nfilt = min(self.specs['n_ls']//self.h_params['n_classes'],8)
        #define which components to plot       
        if sorting == 'l2':
            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=2))            
        elif sorting == 'l1':
            order = np.argsort(np.linalg.norm(self.patterns, axis=0, ord=1))            
        elif sorting == 'contribution':
            #One col per class
            nfilt = 3
            order = []
            for i in range(self.h_params['n_classes']):
                inds = np.argsort(self.out_weights[...,i].sum(-1))[::-1]
                order+=list(inds[:nfilt])
            order = np.array(order)
        elif sorting == 'abs':
            nfilt = self.h_params['n_classes']
            order = []
            for i in range(self.h_params['n_classes']):
                pat= np.argmax(np.abs(self.out_weights[...,i].sum(-1)))
                
                order.append(pat)
                #print(pat.shape)
                #self.ts.append(t)
            order = np.array(order)
        elif sorting == 'best':
            #One pattern per class
            nfilt = self.h_params['n_classes']
            order = []
            for i in range(self.h_params['n_classes']):
                pat,t= np.where(self.out_weights[...,i]==np.max(self.out_weights[...,i]))
                order.append(pat[0])
                self.ts.append(t)
            order = np.array(order)
        elif sorting == 'best_neg':
            nfilt=self.h_params['n_classes']
            #One row per class
            order = []
            for i in range(self.h_params['n_classes']):
                pat= np.argmin(self.out_weights[...,i].sum(-1))
                order.append(pat)
        elif sorting == 'worst':
            nfilt=self.h_params['n_classes']
            #One row per class
            order = []
            weight_sum = np.sum(np.abs(self.out_weights).sum(-1),-1)
            pat= np.argsort(weight_sum)
            #print(weight_sum[pat])
            order = np.array(pat[:nfilt])

        elif isinstance(sorting,list):
            nfilt = len(sorting)
            order = np.array(sorting)
        
        else:
            order = np.arange(self.specs['n_ls'])
        self.fake_evoked.data[:,:len(order)] = self.fake_evoked.data[:,order]
        if scale:
            self.fake_evoked.data[:,:len(order)] /= self.fake_evoked.data[:,:len(order)].max(0)
        self.fake_evoked.data[:,len(order):] *=0
        self.out_filters = self.filters[:,order]
        order = np.array(order)
        
        if spectra:
                z = 2            
        else: 
            z = 1
        nrows = max(1,len(order)//nfilt)
        ncols = min(nfilt,len(order))
        #print('rows:',nrows, ' cols:', ncols)
        f, ax = plt.subplots(z*nrows, ncols,sharey=True)
        f.set_size_inches([16,9])
        ax = np.atleast_2d(ax)
        for i in range(nrows):
            if spectra:
                for jj,flt in enumerate(self.out_filters[:,i*ncols:(i+1)*ncols].T):
                    w, h = freqz(flt,1)
                    ax[z*i+1, jj].plot(w/np.pi*self.h_params['fs']/2, np.abs(h)) 
                    #ax[z*i+1, jj].set_ylim(0,1.5)
            self.fake_evoked.plot_topomap(times=np.arange(i*ncols,  (i+1)*ncols, 1.), 
                                     axes=ax[z*i], colorbar=False, #vmin=0, 
                                     vmax=np.percentile(self.fake_evoked.data[:,:len(order)],99), 
                                     scalings=1,
                                     time_format='')
                           
        f.show()
        #f.tight_layout()
        #return 
    
class VARCNN(Model):
    """VAR-CNN
    
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
        MEG signals. Neuroimage. (2019) May 4;197:425-434"""
        
    def __init__(self,h_params, params, lf_params):
        super().__init__(h_params, params)
        self.specs = lf_params
        self.h_params['architecture'] = 'lf-cnn'
        
    def _build_graph(self):
        self.demix = DeMixing(n_ls=self.specs['n_ls'])
        
        self.tconv1 = VARConv(scope="conv", n_ls=self.specs['n_ls'],  
                              nonlin_out=tf.nn.relu,
                              filter_length=self.specs['filter_length'], 
                              stride=self.specs['stride'], 
                              pooling=self.specs['pooling'], 
                              padding=self.specs['padding'])
        
        self.fin_fc = Dense(size=self.h_params['n_classes'], 
                       nonlin=tf.identity, dropout=self.rate)
        
        y_pred = self.fin_fc(self.tconv1(self.demix(self.X)))
        
        return y_pred
    

    