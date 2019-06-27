import numpy as np
import os
import tensorflow as tf
from time import time
import mneflow
import glob
import mne

#data_path = '/m/nbe/scratch/braindata/hihalme/MEG_2017/syksy2017/data/'
data_path = '/m/nbe/scratch/braindata/hihalme/MEG_2017/raw_data/epochs/'

filenames = [data_path + 'MI_epochs_' + str(j) + '.fif' for j in range(1,19)]
epochs_list = [mne.read_epochs(f) for f in filenames]


#Specify import options
import_opt = dict(savepath='tfr/', #path where TFR files will be saved
	   out_name='data1', #name of TFRecords files
	   picks={'eeg':False, 'meg':'grad'}, #used only if input_type is mne.epochs.Epochs or path to saved '*-epo.fif'
	   scale=True, #apply baseline_scaling?
	   crop_baseline=False,
	   decimate = 4,
	   bp_filter=(1.,45.),
	   scale_interval=(0,200), #indices in time axis corresponding to baseline interval
	   savebatch=1, # number of input files per TFRecord file
	   save_origs=True, # whether to produce separate TFR-file for inputs in original order
	   val_size=0.1,#validation set is 10% of all data
       overwrite=True) #if False loads existing metafile and tfrecords if they already exist, saves time!

meta = mneflow.produce_tfrecords(epochs_list,**import_opt)
dataset = mneflow.Dataset(meta, train_batch = 200, class_subset=None, pick_channels=None, decim=None)


#specify optimizer parmeters
optimizer_params = dict(l1_lambda=3e-4,learn_rate=3e-4)
#optimizer = mneflow.Optimizer(**optimizer_params)


#specify parameters specific for the model
#these are specific to LF-CNN]

lf_params = dict(n_ls=64, #number of latent factors
              filter_length=16, #convolutional filter length in time samples
              pooling = 5, #pooling factor
              stride = 2, #stride parameter for pooling layer
              padding = 'SAME',
              dropout = .5,
              model_path = import_opt['savepath']) #path for storing the saved model


# ##option 1
model = mneflow.models.VARCNN
test_accs = mneflow.utils.leave_one_subj_out(meta, optimizer_params, lf_params, model)
print(test_accs)


# ##option 2
#model = mneflow.models.LFCNN(dataset, optimizer, lf_params)
#model.build()
#start = time()
#model.train(n_iter=3000,eval_step=250,min_delta=1e-6,early_stopping=1)
#stop = time() - start
#print('Trained in {:.2f}s'.format(stop))

#evaluate performance
#test_accs = model.evaluate_performance(meta['orig_paths'][0], batch_size=120)
#model.compute_patterns(output='patterns')
#model.plot_patterns(sensor_layout='Vectorview-grad', sorting='contribution', spectra=True, scale=True)
