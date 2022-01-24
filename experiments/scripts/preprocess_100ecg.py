"""
Script for loading the data from Physionet

1. Installation of wfdb is required for this script: https://doi.org/10.13026/egpf-2788
2. Datasets to be processed can be downloaded from 
    the MIT-BIH Arrhythmia Database (https://doi.org/10.13026/C2F305).
    In the paper "Group Invariant Dictionary Learning", 100.dat was used.
"""

import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

import wfdb.processing

aa = wfdb.io.rdsamp('100')


bb = aa[0]
n1,_ = bb.shape

downsample = 1
n2 = n1//downsample

bb_resamp = np.zeros((n2,))

for i in range(n2):
    bb_resamp[i,] = np.mean(bb[i*downsample:(i+1)*downsample])


# Plot a sample of the curve

#plt.figure(figsize=(8,6))
nLim = 1000
xaxis = np.array(list(range(nLim)))/360.0
plt.ylim(-0.8,0.8)
plt.plot(xaxis,bb_resamp[:nLim],'k')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
if not os.path.exists("../output/"):
    os.makedirs("../output/")
plt.savefig('../output/OriginalECG.pdf', bbox_inches = 'tight',pad_inches = 0)


# Create the training dataset

n = 5000
d = 100
spacings = 10 # Spacing between data

Y = np.zeros((2*d+1,n))

for i in range(n):
    v = bb_resamp[i*spacings:i*spacings+2*d+1]
    Y[:,i] = v
    
# Set the data to be unit-norm
# And zero mean
for i in range(n):
    Y[:,i] -= np.sum(Y[:,i]) / (2*d+1) * np.ones(Y[:,i].shape)
    Y[:,i] = Y[:,i] / np.linalg.norm(Y[:,i])

    
# Save dataset
if not os.path.exists("../datasets/"):
    os.makedirs("../datasets/")
np.save('../datasets/100ecg.npy',Y)
