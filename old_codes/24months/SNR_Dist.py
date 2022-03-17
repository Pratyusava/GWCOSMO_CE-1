#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:23:21 2019

@author: anarya
"""

import numpy as np
from numpy import pi
from scipy import stats
from scipy import special
from source import Fisher
from source import SNR
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from matplotlib import pyplot as ppl
cosmo=FlatLambdaCDM(70.5,0.2736)
from scipy.optimize import curve_fit
""" Geometricised Units; c=G=1, Mass[1500 meter/Solar Mass], Distance[meter], Time[3.0e8 meter/second]"""

"""Parameters: (ln Mc_z, ln(Lambda),tc,Phi_c,ln d)"""

# standard siren (EOS: SLY)
m1_SLY = m2_SLY = 1.433684*1500. 	  	 
Lambda_SLY=2.664334e+02
PN_order=3.5
dl_dM_SLY=(3.085067e+02-1.997018e+02)/((1.433684e+00  -1.493803)*1500.*2.)
sm1=sm2=0.09*1500.
z_true=2.0
m_l=1.0*1500.
m_u=3.0*1500.
N=1000


#load data

ce_fs, ce_asd , et_asd, aligo_asd = np.loadtxt('Amplitude_of_Noise_Spectral_Density.txt', usecols=[0,3,2,1],unpack = True)

# correct units
c=G=1
c1=3.0e8
Mpc=3.086e22

ce_fs *= c1**-1.
ce_asd *= c1**0.5

#Compute Optimal SNR
M_SLY=m1_SLY+m2_SLY
eta_SLY=m1_SLY/m2_SLY
Mc_SLY=M_SLY*eta_SLY**0.6
SNR_opt=SNR(Mc_SLY*(1.+z_true),eta_SLY,Lambda_SLY,1.,1.,cosmo.luminosity_distance(z_true).value*Mpc,ce_fs,ce_asd)**0.5


#Draw True Values
def Draw_true(m10,m20,Lambda,dLambda,cosmo,z):
    
    
   
    
    M=m10+m20
    Mz=M*(1.+z)
    
    pc=np.random.uniform(0,2.*pi)
    tc=np.random.uniform(0,1)
    eta=m10*m20/M**2
    Mc_z=Mz*eta**0.6
    
    
    d_l=cosmo.luminosity_distance(z)
    
    d_l=d_l.value*Mpc
    
    deff=d_l/(np.random.beta(2,4))
    
    
    Lambda=Lambda+(M-m10-m20)*dLambda
    Lambdat=Lambda*(1.+7.*eta - 31.*eta**2)*16./13.
    return Mc_z,eta,tc,pc,Lambdat,deff,m10,m20

Snr_Obs=[]
Snr_True=[]
#Draw Values for N events
for i in range(0,N):
    
    while True:
        
        Mc_z,eta,tc,pc,Lambdat,deff,m1,m2=Draw_true(m1_SLY,m2_SLY,Lambda_SLY,dl_dM_SLY,cosmo,z_true)
    
        V=Fisher(Mc_z,eta,Lambdat,pc,tc,deff,ce_fs,ce_asd)
    
        Cov=np.absolute(np.linalg.inv(V))
        if(Cov.all()>0.0):
            break
    Mean=np.array([np.log(Mc_z),np.log(eta),tc,pc,np.log(abs(Lambdat)),np.log(deff)])
    
    Rho_True=SNR(Mc_z,eta,Lambdat,pc,tc,deff,ce_fs,ce_asd)
    while True:
        Mc_z,eta,tc,pc,Lambdat,deff=np.random.multivariate_normal(Mean,Cov)
        Mc_z=np.exp(Mc_z)
        eta=np.exp(eta)
        Lambdat=np.exp(Lambdat)
        deff=np.exp(deff)
        Rho_Obs=SNR(Mc_z,eta,Lambdat,pc,tc,deff,ce_fs,ce_asd)
        if(Rho_Obs<SNR_opt**2. and Rho_Obs>0.0):
            break

    
    
    
    Rho_Obs=np.sqrt(Rho_Obs)
    Snr_Obs.append(Rho_Obs)
    
    Snr_True.append(Rho_True)
    

Snr_Obs=np.array(Snr_Obs)

Snr_Obs=Snr_Obs[np.logical_not(np.isnan(Snr_Obs))]
Snr_Obs=Snr_Obs[np.logical_not(np.isinf(Snr_Obs))]
Snr_True=np.array(Snr_True)
Snr_True=Snr_Obs[np.logical_not(np.isnan(Snr_True))]
Snr_True=Snr_Obs[np.logical_not(np.isinf(Snr_True))]

def Beta1(x,A):
    
    x=x/A
    return stats.beta.pdf(x,2,4)/A


#Plot for Observed SNR
ppl.ylabel('frequency')
ppl.xlabel('Observed SNR')
Ar= ppl.hist(Snr_Obs,int(len(Snr_Obs)),density=True,label=('z='+str(z_true)))
counts=Ar[0]
bins=Ar[1]
ppl.axvline(x=SNR_opt,color='r',label='Optimal SNR')
bins=np.delete(bins,0)
ppl.plot(bins, Beta1(bins,SNR_opt),label='Beta distribution for Optimal SNR')
A=curve_fit(Beta1,bins,counts)
A=A[0]
ppl.plot(bins,Beta1(bins,A),label='Beta Distribution Fit')
ppl.show()


#Plot for Intrinsic SNR
ppl.ylabel('frequency')
ppl.xlabel('Intrinsic SNR')
Ar= ppl.hist(Snr_True,int(len(Snr_True)),density=True,label=('z='+str(z_true)))
counts=Ar[0]
bins=Ar[1]
ppl.axvline(x=SNR_opt,color='r',label='Optimal SNR')
bins=np.delete(bins,0)
ppl.plot(bins, Beta1(bins,SNR_opt),label='Beta distribution for Optimal SNR')
A=curve_fit(Beta1,bins,counts)
A=A[0]
ppl.plot(bins,Beta1(bins,A),label='Beta Distribution Fit')
ppl.show()
 