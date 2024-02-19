#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:38:36 2020

@author: zubarei1
"""
import mne
#from scipy import signal
import numpy as np
from mne.viz import iter_topography
from matplotlib import pyplot as plt


data_path = "C:\\Users\\ipzub\\Desktop\\AALTO\\LST\\data\\"
fname = data_path + 'sub1_sess2.fif'
raw = mne.io.Raw(fname, preload=True)
raw = raw.set_eeg_reference(ref_channels='average')
raw.notch_filter(np.arange(50., 100, 50.), notch_widths=1.)
#raw.filter(l_freq=.1, h_freq=45)

events = mne.find_events(raw, stim_channel='trig', min_duration=0.002)
events_diff = np.concatenate([[250], np.diff(events[:, 0])])
good_events = np.where(events_diff > 240, events_diff < 260, False)
events = events[good_events, :]

epochs = mne.Epochs(raw, events, tmin=-.1, tmax=.45)

epochs, _ = epochs.equalize_event_counts(event_ids=[ '2', '3'])

evokeds = [epochs[i].average() for i in ['1', '2', '3']]

# mne.viz.plot_compare_evokeds(dict(event_2=evokeds[0], event_3=evokeds[1]),
#                               legend='upper left', show_sensors='upper right')
t = epochs.times
data = np.array([ev.get_data() for ev in evokeds])
#%
#%%
def my_callback(ax, ch_idx):
    """
    This block of code is executed once you click on one of the channel axes
    in the plot. To work with the viz internals, this function should only take
    two parameters, the axis and the channel or data index.
    """
    ax.plot(t, data[:, ch_idx, :].T)
    ax.set_xlabel('Time')
    ax.set_ylabel('uV')
    ax.set_ylim(np.min(data), np.max(data))


for ax, idx in iter_topography(raw.info,
                                fig_facecolor='white',
                                axis_facecolor='white',
                                axis_spinecolor='white',
                                on_pick=my_callback):
    ax.plot(t, data[:, idx, :].T)
    ax.set_ylim(np.min(data), np.max(data))

plt.gcf().suptitle('Power spectral densities')
plt.show()

#%%
evkd = mne.combine_evoked(evokeds, [1, -1])
evkd.info['bads'] += ['FC4']
evkd.plot_topomap(times='peaks')