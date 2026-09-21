# -*- coding: utf-8 -*-
"""
Created on Fri May 30 10:08:35 2025

@author: ipzub
"""
#%%
from time import time
from collections import defaultdict
import os
os.chdir('C:\\Users\\ipzub\\projs\\mneflow')
import mneflow
import numpy as np
from matplotlib import pyplot as plt

path = "C:\\data\\simulated\\"
log_fname = 'distractor_overlap'


comment = 'rand = random weight, diff=False, event=1'
event = 1

if log_fname == 'only_source':
    fname = 'caudalmiddlefrontal-lh_10_+-_0.5_nA_18_hz_10.0_mm'
elif log_fname == 'distractor_no_overlap':
    fname = 'caudalmiddlefrontal-lh_10_+-_0.5_nA_18_hz_10.0_mm_distractor_precentral-rh'
elif log_fname == 'distractor_overlap':
    fname = 'caudalmiddlefrontal-lh_10_+-_0.5_nA_18_hz_10.0_mm_distractor_overlapping_precentral-rh'



meta = mneflow.load_meta(path + fname)
model = meta.restore_model()
self = model
ds = self.dataset.val
X, y = [row for row in ds.take(1)][0]
 
order = 3
start = time()

feature_relevance_loss, best = self.compute_componentwise_loss(X, y, 
                                                               order=order)
stop = time() - start

print("SHAP order {} Done in {:.2f}s".format(order, stop))
#%%
#basic_shap = a[0][0]
#feature_relevance_loss = a[1]
# n_comp = 36
# out = defaultdict(list)
    
# for k in feature_relevance_loss.keys():
#     feat_keys = k.split('-')
#     if len(feat_keys) == 3:
#         k1, k2, class_ind = feat_keys
#         out[k1].append(feature_relevance_loss[k])
#         out[k2].append(feature_relevance_loss[k])
#     elif len(feat_keys) == 4:
#         k1, k2, k3, class_ind = feat_keys
#         out[k1].append(feature_relevance_loss[k])
#         out[k2].append(feature_relevance_loss[k])
#         out[k3].append(feature_relevance_loss[k])
    
    

# #for i in range(n_comp):
# #    out[str(i)].append(basic_shap[i, 0])
shap = np.stack([b['o1_relevances'] for b in best], axis=1)

# #shap = np.zeros(n_comp)
# shap = [np.mean(out[str(k)]) for k in range(36) if k != 'self']
# plt.plot(shap)
#plt.plot(shap)
#print(np.corrcoef(out['self'], shap))
for jj in range(1):
    plt.plot(best[jj]['o1_relevances'])
    plt.plot(best[jj]['o2_relevances'])
    plt.plot(best[jj]['o3_relevances'])
    plt.plot(best[jj]['o3_inds'], best[jj]['o3_relevances'][best[jj]['o3_inds']], 'g*')
    plt.plot(best[jj]['o2_inds'], best[jj]['o2_relevances'][best[jj]['o2_inds']], marker='*', 
             color='tab:orange', linestyle='none')
    plt.plot(best[jj]['o3_inds'], np.ones(3)*best[jj]['o3'], 'g*')
    #print(np.corrcoef(old_shap2, shap))

#best3 = np.array([int(b) for b in best['key-o3-c0'].split('-')[:3]])
#plt.plot(best3, np.array(shap)[best3], 'rx', label='best3')

#best2 = np.array([int(b) for b in best['key-o2-c0'].split('-')[:2]])
#plt.plot(best2, np.array(shap)[best2], 'bx', label='best3')

# best1 = np.array([int(b) for b in best['key-o1-c0'].split('-')[:1]])
# plt.plot(best1, np.array(shap)[best1], 'ro', label='best3')
    
    

    





