# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:12:48 2017
Read/ Filter 0.1-45Hz/ Epoch/ Downsample/ Get labels/ Save
mne version = 0.15.2/ python2.7
@author: zubarei1
"""

"pool: 102+101+3+4 (right), and 1+2+103+104(left)"

import mne
import numpy as np


data_path =  '/m/nbe/scratch/braindata/izbrv/opm_wrist/'#'
raw_suff = 's03_wrist_mov.fif' 
   
def rms(x,f_length=250):
    x = x**2
    cf = np.ones(f_length)/float(f_length)
    x = np.convolve(a=x,v=cf,mode='same')
    return np.sqrt(x)
    
def sliding_std(x,f_length=1000):
    x = np.concatenate([np.zeros(f_length-1),x])
    nrows = x.size - f_length + 1
    n = x.strides[0]
    x2D = np.lib.stride_tricks.as_strided(x,shape=(nrows,f_length),strides=(n,n))
    return np.std(x2D, axis=1)

    
fname = data_path+raw_suff
raw = mne.io.RawFIF(fname, preload=True, verbose=False)
events = mne.find_events(raw,stim_channel='STI101', min_duration=0.003,output='step')
event_id = {'ext_wrist':1, 'flex_wrist':2, 'flex_index':4, 'flex_thumb':8,'palm_down':16,
        'palm_up':32, 'precision_grip':64, 'grasp':128} # Joonas s01

#specify/fix channel info
opm_names = ['MEG1111','MEG1121','MEG1131','MEG1141',
             'MEG1211','MEG1221','MEG1231','MEG1241']
emg_names = ['MISC005','MISC006','MISC007','MISC008',
             'MISC011']                               

opm_picks = mne.pick_channels(raw.info['ch_names'],opm_names)
emg_picks = mne.pick_channels(raw.info['ch_names'],emg_names)

for k in emg_picks:
    raw.info['chs'][k]['kind']=302

#preprocess raw file  
fmin = 1.
fmax= 125.
raw.notch_filter(np.arange(50, 251, 50),notch_widths=5,picks=opm_picks,)
raw = raw.filter(l_freq=fmin,h_freq=fmax,picks=opm_picks,method='iir')
raw = raw.filter(l_freq=fmin,h_freq=None,picks=emg_picks,method='iir')
raw.apply_function(rms,picks=emg_picks)# = mne.io.RawFIF(fname, preload=True, verbose=False)
raw.pick_channels(opm_names+emg_names+['STI101'])



#%%extract (long) epochs
epochs = mne.epochs.Epochs(raw,events,event_id=event_id,tmin=-3.,tmax=16,decim=2.,detrend=1,reject={'mag':1e-10}, preload=True)
#del raw
epochs = epochs['ext_wrist','flex_wrist','flex_thumb','flex_index','palm_up', 'palm_down','precision_grip','grasp']# 
#epochs.equalize_event_counts(['ext_wrist','flex_wrist','flex_thumb','flex_index','palm_up', 'palm_down','precision_grip','grasp'],method='truncate')


opm_data = epochs.get_data()[...,opm_picks,2501:]

labels = epochs.events[:,2]#+1
emg = epochs.pick_types(meg=False,emg=True,stim=False)
emg_data = epochs.get_data()[...,2501:]



