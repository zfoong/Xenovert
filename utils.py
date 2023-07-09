#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:34:17 2022

@author: foong
"""

import numpy as np
import pandas as pd
from model.xenovert import xenovert
from scipy.special import rel_entr
from scipy.stats import mstats
from math import log2, sqrt
from sklearn import preprocessing
import matplotlib.pyplot as plt

class utils:
    
    def bootstrap(xeno, df, sampling_rate = 0.8, repeat=10):
        for i in range(repeat):
            subset = df.sample(frac=sampling_rate)
            # np.random.shuffle(subset)
            x1 = df['X']
            y1 = df['Y']
            
            z = np.vstack((x1, y1))
            subset = z.T
            
            for i, x in enumerate(subset):
                xeno.input(x)
        return xeno
    
    def shift_func(x, y, nboot=200, plot=True):
        
        crit = 80.1/(min(len(x),len(y)))**2+2.73
        m = np.zeros((9,5))
        for d in range(9):
            q = (d+1)/10
            
            data = np.random.choice(x, (nboot, len(x)), replace=True)
            bvec = mstats.hdquantiles(data, prob=q, axis=1)
            sex = np.var(bvec)
            
            data = np.random.choice(y, (nboot, len(y)), replace=True)
            bvec = mstats.hdquantiles(data, prob=q, axis=1)
            sey = np.var(bvec)
            
            m[d,0] = mstats.hdquantiles(x,prob=q)
            m[d,1] = mstats.hdquantiles(y,prob=q)
            m[d,2] = m[d,0]-m[d,1]
            m[d,3] = m[d,2]-crit*np.sqrt(sex+sey)
            m[d,4] = m[d,2]+crit*np.sqrt(sex+sey)
        
        if plot is True:
            plt.figure(figsize=(4,1), dpi=300)
            plt.plot(m[:,0], m[:,2], marker=".")
            plt.errorbar(m[:,0], m[:,2], yerr=(np.abs(m[:,3]-m[:,2]), np.abs(m[:,4]-m[:,2])), fmt ='.')
            plt.xlabel('Source distribution quantiles')
            plt.ylabel('Quantile differences')
        
        return m