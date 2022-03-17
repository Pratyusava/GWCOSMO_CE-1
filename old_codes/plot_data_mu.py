#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:17:22 2020

@author: anarya
"""

import h5py
import corner
import numpy as np
Mpc=3.086e22
ndim=4
Event_index=70
f=h5py.File('data_mu.h5','r')
A=np.array(f['Measured_values_event_no.='+str(Event_index)])
True_val=np.array(f['True_values_event_no.='+str(Event_index)])
f.close()
True_val[3]=np.log(True_val[3]/Mpc)
Mc_z=A[0]
Mu_z=A[1]
Lambdat=A[2]
deff=A[3]
data=np.vstack([Mc_z,Mu_z,Lambdat,np.log(deff/Mpc)]).T
#print(data)
fig=corner.corner(data,levels=[0.65,0.80,0.95],smooth=1.2,labels=[r'$\mathcal{M}_{z}$',r'$\mu_z$',r'$\tilde{\Lambda}$' ,r'$\log{\frac{D_{eff}}{Mpc}}$'])
axes=np.array(fig.axes).reshape((ndim,ndim))



for i in range(ndim):
    ax=axes[i,i]
    ax.axvline(x=True_val[i])
    
    
for yi in range(ndim):
    for xi in range(yi):
        ax=axes[yi,xi]
        ax.axvline(x=True_val[xi])
        ax.axhline(y=True_val[yi])
        ax.plot(True_val[xi],True_val[yi])
fig.savefig('Post_mu_n.pdf')