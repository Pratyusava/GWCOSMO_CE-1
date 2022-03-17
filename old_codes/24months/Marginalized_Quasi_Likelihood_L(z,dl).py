#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:40:36 2019

@author: anarya
"""

import numpy as np
from numpy import pi
from scipy import stats
from scipy import special
from scipy.interpolate import interp1d
from source import Fisher
from source import SNR
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from matplotlib import pyplot as ppl
cosmo=FlatLambdaCDM(70.5,0.2736)
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import root
import corner
from scipy import stats
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


# correct units
c=G=1
c1=3.0e8
Mpc=3.086e22

ce_fs *= c1**-1.
ce_asd *= c1**0.5
et_asd *= c1**0.5


#Draw True Values of Parameters
def Draw_true(m10,m20,Lambda,dLambda,cosmo,Z_true):
    
    
    z=Z_true
    
#Only draw masses within interpolation Range
    while True:
        m2=m1=np.exp(np.random.normal(np.log(m10),sm1/m10))
        
        if(m1<m_u and m2<m_u and m1>m_l and m2>m_l):
            break
    
    M=m1+m2
    Mz=M*(1.+z)
    
    
    eta=m1*m2/M**2
    Mc_z=Mz*eta**0.6
    
    
    d_l=cosmo.luminosity_distance(z)
    
    d_l=d_l.value*Mpc
    Th=(np.random.beta(2,4))
    deff=d_l/Th
    
    
    Lambda=Lamb(m1)
    Lambdat=Lambda*(1.+7.*eta - 31.*eta**2)*16./13.
    return Mc_z,eta,Lambdat,deff,Th,z

#Draw Measured values using Fisher matrix with Mean=True Values
def Draw_Measured(Mc_z,eta,Lambdat,deff,Th,z_true):
     Mean=np.array([np.log(Mc_z),np.log(eta),1.,1.,np.log(abs(Lambdat)),np.log(deff)])
     V=Fisher(Mc_z,eta,Lambdat,1.,1.,deff,ce_fs,ce_asd)
    
     Cov=np.absolute(np.linalg.inv(V))
     X1=[]
     X2=[]
     X3=[]
     X4=[]
     for i in range(10000):
#Only draw Lambda within Interpolation Ranfe
         while True:
             Mc_z,eta,tc,pc,Lambdat,deff=np.random.multivariate_normal(Mean,Cov)
             M_z=Mc_z-0.6*eta
             if(M_z<np.log((m_u+m_u)*(1.+z_true)) and M_z>np.log((m_l+m_l)*(1.+z_true))):   
                 Mc_z=np.exp(Mc_z)
                 Lambdat=np.exp(Lambdat)
                 deff=np.exp(deff)
                 eta=np.exp(eta)
                 eta=0.25
                 Lambda=Lambdat/((1.+7.*eta - 31.*eta**2)*16./13.)
                 h=0.1*Lambda
                 
                 if(Lambda<(max(L)-h) and Lambda>(min(L)+h)):
                         break
         X1.append(Mc_z)
         X2.append(eta)
         X3.append(Lambdat)
         X4.append(deff)
         
     Mc_z=np.array(X1)
     eta=np.array(X2)
     Lambdat=np.array(X3)
     deff=np.array(X4)
     Th=(np.random.beta(2,4))
     
     Th=np.random.beta(2,4)
     return Mc_z, eta, Lambdat, deff, Th
def z_measured(Mc_z,Lambdat,eta):
    Lambda=Lambdat/((1.+7.*eta - 31.*eta**2)*16./13.)
    m_measured=mass(Lambda)
    
    
    
    M_z=Mc_z/eta**(0.6)
    z_measured=M_z/(2.*m_measured) -1.
    return z_measured

#Draw values from Fisher Matrix to compute Marginalized Quasi Likelihood
Mc_z, eta, Lambdat, deff, Th, z_true=Draw_true(m1_SLY,m2_SLY,Lambda_SLY,dl_dM_SLY,cosmo,Z_true)
Mc_z, eta, Lambdat, deff, Th=Draw_Measured(Mc_z, eta, Lambdat, deff, Th, z_true)
Z_measured=[z_measured(Mc_z[i],Lambdat[i],eta[i]) for i in range(len(Mc_z))]
DL=np.log(deff*np.random.beta(2,4,len(deff))/Mpc)

DL_true=np.log(cosmo.luminosity_distance(z_true).value)
TH=np.linspace(0.0001,1.,100)

#Compute Marginalized Quasi Likelihood Using Gaussian KDE
data=np.vstack([Z_measured,DL]).T

f=stats.gaussian_kde(data.T)
Z1=np.linspace(min(Z_measured),max(Z_measured),100)
DL1=np.linspace(min(DL),max(DL),100)
Z2,DL2=np.meshgrid(Z1,DL1)
pos=np.vstack([Z2.ravel(),DL2.ravel()])
S=f(pos)
S=np.reshape(S.T,Z2.shape)
S=S/max([max(x) for x in S])
S=S
CS=ppl.contour(Z1,DL1,S,levels=[0.50,0.65,0.80,0.95])
ppl.clabel(CS,inline=True)
ppl.axvline(x=z_true)
ppl.axhline(y=DL_true)

ppl.plot(Z1,np.log([FlatLambdaCDM(70.5,0.2736).luminosity_distance(z).value for z in Z1]),label='DL(z,H0=70.5,Omega_m=0.2736)')
ppl.plot(Z1,np.log([FlatLambdaCDM(80.5,0.2736).luminosity_distance(z).value for z in Z1]),label='DL(z,H0=80.5,Omega_m=0.2736)')
ppl.xlabel('Redshift')
ppl.ylabel('natural log of Luminosity Distance in MPc')

ppl.ylim(DL_true-2.,DL_true+2.)
ppl.legend()

ppl.show()




#Compute Marginalized Quasi Likelihood using histogram (corner plot)
fig2=corner.corner(data,levels=[0.65,0.80,0.95],smooth=1.2,labels=['Redshift','Natural Log of Luminosity Distance in MPc'])
ndim=2
axes=np.array(fig2.axes).reshape((ndim,ndim))
Tr=np.array([z_true,DL_true])


for i in range(ndim):
    ax=axes[i,i]
    ax.axvline(x=Tr[i])
    
    
for yi in range(ndim):
    for xi in range(yi):
        ax=axes[yi,xi]
        ax.axvline(x=Tr[xi])
        ax.axhline(y=Tr[yi])
        ax.plot(Tr[xi],Tr[yi])
        

