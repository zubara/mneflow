import numpy as np
import os
import tensorflow as tf
from time import time
import mneflow
import glob
import mne

# Choose either Data 1 or 2

# Data 1
data_path = '/m/nbe/scratch/braindata/hihalme/MEG_2017/syksy2017/data/'
filenames = [data_path + 'MI_epochs_' + str(j) + '.fif' for j in range(2,17)]
savepath = 'tfr1/'

# Data 2
#data_path = '/m/nbe/scratch/braindata/hihalme/MEG_2017/raw_data/epochs/'
#filenames = [data_path + 'MI_epochs_' + str(j) + '.fif' for j in range(1,19)]
#savepath = 'tfr2/'

#read epochs
epochs_list = [mne.read_epochs(f) for f in filenames]


#Specify import options
import_opt = dict(savepath=savepath, #path where TFR files will be saved
	   out_name='data1', #name of TFRecords files
	   picks={'eeg':False, 'meg':'grad'},
	   scale=True,
	   crop_baseline=False,
	   decimate = 4,
	   bp_filter=(2.,45.),
	   scale_interval=(0,200), #indices in time axis corresponding to baseline interval
	   savebatch=1, # number of input files per TFRecord file
	   save_origs=True, # whether to produce separate TFR-file for inputs in original order
	   val_size=0.1,#validations set size set to 10% of all data
           overwrite=False)

meta = mneflow.produce_tfrecords(epochs_list,**import_opt)


from mneflow.layers_modified import ConvDSV, DeMixing, Dense

class TestCNN(mneflow.models.Model):

    def build_graph(self):

        self.scope = 'time-frequency'

	#demix
        self.demix = DeMixing(n_ls=64, expand=False)
        self.X = self.demix(self.X)
        self.X = tf.reshape(self.X, [-1, self.X.shape[2],self.X.shape[1]])

        #calculate STFT
        self.X = tf.abs(tf.signal.stft(self.X, 125, 10, fft_length=128, window_fn=tf.signal.hann_window,
             pad_end=True, name=None))

        #normalize
        self.X  = (self.X - tf.math.reduce_mean(self.X, [2,3], keepdims=True)) / tf.math.reduce_std(self.X, [2,3], keepdims=True)
        self.X = tf.expand_dims(self.X, -1)

	#convolution in "3D" i.e. 2D over freqs&time
        c1 = ConvDSV(n_ls=self.specs['n_ls'], nonlin=tf.identity, inch=1,
                      filter_length=self.specs['filter_length'], domain='2d', padding='SAME',
                      stride=self.specs['stride'], pooling=self.specs['pooling'], conv_type='3d')

        self.c1o = c1(self.X)
        out = tf.reshape(self.c1o, [-1, np.prod(self.c1o.shape[1:])])

        fc_out = Dense(size=meta['n_classes'], nonlin=tf.identity,
                       dropout=self.rate)
        y_pred = fc_out(out)

        return y_pred


# define some parameters
optimizer_params = dict(l1_lambda=1e-3,
                        l2_lambda=0,
                        learn_rate=3e-3,
                        task= 'classification')

optimizer = mneflow.Optimizer(**optimizer_params)

dataset = mneflow.Dataset(meta, train_batch = 200, class_subset=None,
                                 pick_channels=None, decim=None)


params = dict(n_ls = 1,
              filter_length = [4,8],
              stride = 2,
              pooling = 2,
              dropout = .15,
              model_path = import_opt['savepath'])


# build model
model = TestCNN

# cross-validation
test_accs = mneflow.utils.leave_one_subj_out(meta, optimizer_params, params, model)

ave = 0
for j in range(len(test_accs)):
       print(test_accs[j]['test_init'])
       ave += test_accs[j]['test_init']
ave /= len(test_accs)

print("\nAVERAGE ACCURACY:", ave)



# build model
#model = TestCNN(dataset, optimizer, params)
#model.build()

# train
#model.train(n_iter=3000,eval_step=100,min_delta=1e-6,early_stopping=3)

#evaluate performance
#test_accs = model.evaluate_performance(meta['orig_paths'][0], batch_size=120)

