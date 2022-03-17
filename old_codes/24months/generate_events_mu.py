#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:53:32 2020

@author: anarya
"""

import numpy as np
from numpy import pi
from scipy import stats
from scipy import special
from scipy.interpolate import interp1d
from source_mu import Fisher,SNR
import h5py
#from source import SNR
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from matplotlib import pyplot as ppl
cosmo=FlatLambdaCDM(70.5,0.2736)
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import root
import corner
#from scipy import stats
from scipy import interpolate
import scipy.integrate as integrate
from scipy.special import erfc
#from Selection import P_det_H0
from joblib import Parallel, delayed
import multiprocessing
import math
""" Geometricised Units; c=G=1, Mass[1500 meter/Solar Mass], Distance[meter], Time[3.0e8 meter/second]"""

"""Parameters: (ln Mc_z, ln(Lambda),tc,Phi_c,ln d)"""

# standard siren (EOS: SLY)
m1_SLY = m2_SLY = 1.433684*1500. 	  	 
Lambda_SLY=2.664334e+02
PN_order=3.5
dl_dM_SLY=(3.085067e+02-1.997018e+02)/((1.433684e+00  -1.493803)*1500.*2.)
sm1=sm2=0.09*1500.
Z_true=1.5

N=100000
z_peak=2.5
z_Max=10.
#load data
m,L=np.loadtxt('Mass_Vs_TidalDeformability_SLY.txt',usecols=(0,1),unpack=True)
m*=1500.
m_l=min(m)
m_u=max(m)
mass=interp1d(L,m,kind='cubic')
Lamb=interp1d(m,L,kind='cubic')
l_l=min(L)
l_u=max(L)
ce_fs, ce_asd , et_asd, aligo_asd = np.loadtxt('Amplitude_of_Noise_Spectral_Density.txt', usecols=[0,3,2,1],unpack = True)

Rho_Th=8.0
# correct units
c=G=1
c1=3.0e8
Mpc=3.086e22

ce_fs *= c1**-1.
ce_asd *= c1**0.5
et_asd *= c1**0.5

#Draw True Values
def Draw_true(m10,m20,Lambda,dLambda,cosmo):
    beta=2.*z_Max/z_peak -2.
    z=np.random.beta(3,beta+1)*z_Max
    
#COnly draw values within interpolation range    
   
    m1_z=m10*(1.+z)
    m2_z=m20*(1.+z)
    
    Mz=m1_z+m2_z
    mu_z=m1_z*m2_z/Mz
    
    pc=np.random.uniform(0,2.*np.pi)
    tc=np.random.uniform(0,1)
    
    Mc_z=Mz**(2./5.)*mu_z**(3./5.)
    
    
    d_l=cosmo.luminosity_distance(z)
    
    d_l=d_l.value*Mpc
    Th=(np.random.beta(2,4))
    deff=d_l/Th
    
    eta=(mu_z/Mc_z)**(5./2.)
    Lambda=Lamb(m10)
    Lambdat=Lambda*(1.+7.*eta - 31.*eta**2)*16./13.
    return Th,Mc_z,mu_z,tc,pc,Lambdat,deff,m10,m20,z,d_l

#Draw Measured Values from Fisher Matrix evaluated at True Values
def Draw_Measured(Mc_z,mu_z,Lambdat,deff,Th,z_true):
     Mean=np.array([np.log(Mc_z),np.log(mu_z),np.log(abs(Lambdat)),np.log(deff),1.,1.])
     V=Fisher(Mc_z,Lambdat,mu_z,deff,1.,1.,ce_fs,ce_asd)
     #print(V)
     Cov=np.absolute(np.linalg.inv(V))
     X1=[]
     X2=[]
     X3=[]
     X4=[]
     for i in range(4000):
#Only draw values within Interpolation Range
         while True:
             Mc_z,mu_z,Lambdat,deff,tc,pc=np.random.multivariate_normal(Mean,Cov)
             M_z=2.5*Mc_z-1.5*mu_z
             if(M_z<np.log((m_u+m_u)*(1.+z_true)) and M_z>np.log((m_l+m_l)*(1.+z_true))):   
                 Mc_z=np.exp(Mc_z)
                 Lambdat=np.exp(Lambdat)
                 deff=np.exp(deff)
                 mu_z=np.exp(mu_z)
                 eta=(mu_z/Mc_z)**2.5
                 
                 Lambda=Lambdat/((1.+7.*eta - 31.*eta**2)*16./13.)
                 h=0.1*Lambda
                 
                 if(Lambda<(max(L)-h) and Lambda>(min(L)+h)):
                         break
         X1.append(Mc_z)
         X2.append(mu_z)
         X3.append(Lambdat)
         X4.append(deff)
         
     Mc_z=np.array(X1)
     Mu_z=np.array(X2)
     Lambdat=np.array(X3)
     deff=np.array(X4)
     Th=(np.random.beta(2,4))
     
     Th=np.random.beta(2,4)
     return Mc_z, Mu_z, Lambdat, deff, Th
N_events=1000
f=h5py.File('data_mu.h5','w')
for i in range(0,N_events):
#Introduce SNR Bias    
    while True:
        Th,Mc_z_true,mu_z_true,tc,pc,Lambdat_true,deff_true,m1,m2,z_true,dl_true=Draw_true(m1_SLY,m2_SLY,Lambda_SLY,dl_dM_SLY,cosmo)
        if( SNR(Mc_z_true,mu_z_true,deff_true,ce_fs,ce_asd)**0.5>Rho_Th):
            break

    Mc_z, Mu_z, Lambdat, deff, Th=Draw_Measured(Mc_z_true, mu_z_true, Lambdat_true, deff_true, Th, z_true)
    f.create_dataset('Measured_values_event_no.='+str(i),data=np.array([Mc_z,Mu_z,Lambdat,deff]))
    f.create_dataset('True_values_evebt_no.='+str(i),data=np.array([Mc_z_true,mu_z_true,Lambdat_true,deff_true]))
f.close()
