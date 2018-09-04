# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:00:36 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'jet'

mask=np.loadtxt("m4_v3_woNorm/misspredictions_m4v3_test.txt",usecols=0)

mask=mask.astype(int)


missp1=shuffled_X_test[0,:]
mdtchod=missp1[mask]
missp2=shuffled_X_test[1,:]
mdtrich=missp2[mask]
missp3=shuffled_X_test[2,:]
mdtktag=missp3[mask]
missp4=shuffled_X_test[3,:]
mcda=missp4[mask]
#-----------------------------------------------------------------------------------------
#for the rightly identified
missp1=shuffled_X_test[0,:]
mdtchod=np.delete(missp1,mask, axis=0)
missp2=shuffled_X_test[1,:]
mdtrich=np.delete(missp2,mask, axis=0)
missp3=shuffled_X_test[2,:]
mdtktag=np.delete(missp3,mask, axis=0)
missp4=shuffled_X_test[3,:]
mcda=np.delete(missp4,mask, axis=0)
#-----------------------------------------------------------------------------------------
hdtchod=plt.hist2d(mdtchod,mcda,bins=177,range=[[-1,1],[0,2]])
plt.xlabel('delta T(ns)')
plt.ylabel('CDA')
plt.title("mis-Identified events in CHOD")
plt.show()


hdtrich=plt.hist2d(mdtrich,mcda,bins=177,range=[[-1,1],[0,2]])
plt.xlabel('delta T(ns)')
plt.ylabel('CDA')
plt.title("Identified events in RICH")
plt.show()


hdtktag=plt.hist2d(mdtktag,mcda,bins=177,range=[[-1,1],[0,2]])
plt.xlabel('delta T(ns)')
plt.ylabel('CDA')
plt.title("Identified events in KTAG")
plt.show()

#------------------------------------------------------------------------------------------
#Same for training
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'jet'

mask=np.loadtxt("m4_v3_woNorm/misspredictions_m4v3_train.txt",usecols=0)

mask=mask.astype(int)


missp1=shuffled_X[0,:]
mdtchod=missp1[mask]
missp2=shuffled_X[1,:]
mdtrich=missp2[mask]
missp3=shuffled_X[2,:]
mdtktag=missp3[mask]
missp4=shuffled_X[3,:]
mcda=missp4[mask]
#-----------------------------------------------------------------------------------------
#for the rightly identified
missp1=shuffled_X[0,:]
mdtchod=np.delete(missp1,mask, axis=0)
missp2=shuffled_X[1,:]
mdtrich=np.delete(missp2,mask, axis=0)
missp3=shuffled_X[2,:]
mdtktag=np.delete(missp3,mask, axis=0)
missp4=shuffled_X[3,:]
mcda=np.delete(missp4,mask, axis=0)
#-----------------------------------------------------------------------------------------
hdtchod=plt.hist2d(mdtchod,mcda,bins=177,range=[[-1,1],[0,2]])
plt.xlabel('delta T(ns)')
plt.ylabel('CDA')
plt.title("mis-Identified events in CHOD")
plt.show()


hdtrich=plt.hist2d(mdtrich,mcda,bins=177,range=[[-1,1],[0,2]])
plt.xlabel('delta T(ns)')
plt.ylabel('CDA')
plt.title("mis-Identified events in RICH")
plt.show()


hdtktag=plt.hist2d(mdtktag,mcda,bins=177,range=[[-1,1],[0,2]])
plt.xlabel('delta T(ns)')
plt.ylabel('CDA')
plt.title("mis-Identified events in KTAG")
plt.show()
