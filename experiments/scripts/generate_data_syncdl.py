"""
Script for generating synthetic data for synchronization experiment
"""

import numpy as np
import gidl.utils as gc
import os

np.random.seed(625)

q = 1  # number of generators {1,2}
unif = True  # {True, False}
sigmanoise = 0.1 if not unif else 0.0  # if unif is True, it should be 0.0; if False, we use 0.1 

d = 3  # size of data
r = 20  # number of elements to be calibrated (more, easier)
n = 1000  # number of measurements

if __name__ == "__main__":

    TrueDictionary = np.random.randn(d,r,q)
    for l in range(q):
        TrueDictionary[:,:,l] = gc.normalize(TrueDictionary[:,:,l])
    
    # Generate fake observations
    Data = np.zeros((d,r,n))
    for i in range(n):
        DataMat = np.zeros((d,r))
        for l in range(q):
            c = np.random.randn()
            G = np.random.randn(d,d)
            u,_,v = np.linalg.svd(G)
            G = u @ v
            if unif:
                DataMat += G @ TrueDictionary[:,:,l]
            else:
                DataMat += c * G @ TrueDictionary[:,:,l]
        Data[:,:,i] = DataMat
        Data[:,:,i] += np.random.randn(d,r)*sigmanoise
        
    # Save to disk
    if not os.path.exists("../datasets//"):
        os.makedirs("../datasets//")
    fname = f"q{q}"
    fname += "_unif" if unif else ""
    np.save(f"../datasets/Data_{fname}.npy", Data)
    np.save(f"../datasets/OriginalDict_{fname}.npy", TrueDictionary)