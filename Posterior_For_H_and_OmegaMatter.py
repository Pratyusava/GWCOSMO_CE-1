#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:03:08 2020

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
#import corner
from scipy import stats
from scipy import interpolate
import scipy.integrate as integrate
from scipy.special import erfc
#from Selection import P_det_H0
from joblib import Parallel, delayed
import multiprocessing

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
m,L=np.loadtxt('SLY.txt',usecols=(0,1),unpack=True)
m*=1500.
m_l=min(m)
m_u=max(m)
mass=interp1d(L,m,kind='cubic')
Lamb=interp1d(m,L,kind='cubic')
l_l=min(L)
l_u=max(L)
ce_fs, ce_asd , et_asd, aligo_asd = np.loadtxt('CE.txt', usecols=[0,3,2,1],unpack = True)

Rho_Th=10.5
# correct units
c=G=1
c1=3.0e8
Mpc=3.086e22

ce_fs *= c1**-1.
ce_asd *= c1**0.5
et_asd *= c1**0.5
def Deff(d,th,ph,psi,io):
    F1=-0.5*(1.+np.cos(th)**2)*np.cos(2.*ph)*np.cos(2.*psi)-np.cos(th)*np.sin(2.*ph)*np.sin(2.*psi)
    F2=-0.5*(1.+np.cos(th)**2)*np.cos(2.*ph)*np.sin(2.*psi)-np.cos(th)*np.sin(2.*ph)*np.cos(2.*psi)
    return d*(F1**2*(1.+np.cos(io))**2/4.+F2**2*np.cos(io)**2)**-1

def Draw_true(m10,m20,Lambda,dLambda,cosmo,z_max,z_peak):
    beta=2.*z_Max/z_peak -2.
    z=np.random.beta(3,beta+1)*z_Max
    #z=np.random.uniform(0.001,z_max)
    #z=Z_true
    th=np.arccos(np.random.uniform(0,1))
    io=np.arccos(np.random.uniform(0,1))
    
    psi=np.random.uniform(0.,(2.*pi))
    phi=np.random.uniform(0.,(2.*pi))
    
    while True:
        m2=m1=np.exp(np.random.normal(np.log(m10),sm1/m10))
        #m2=np.exp(np.random.normal(np.log(m20),np.log(sm2)))
        if(m1<m_u and m2<m_u and m1>m_l and m2>m_l):
            break
    #m1=m2=m10
    M=m1+m2
    Mz=M*(1.+z)
    
    pc=np.random.uniform(0,2.*pi)
    tc=np.random.uniform(0,1)
    eta=m1*m2/M**2
    Mc_z=Mz*eta**0.6
    
    
    d_l=cosmo.luminosity_distance(z)
    
    d_l=d_l.value*Mpc
    Th=(np.random.beta(2,4))
    deff=d_l/Th
    
    
    Lambda=Lamb(m1)
    Lambdat=Lambda*(1.+7.*eta - 31.*eta**2)*16./13.
    return Th,Mc_z,eta,tc,pc,Lambdat,deff,m1,m2,z

def Draw_Measured2(Mc_z,eta,Lambdat,deff,Th,z_true):
     Mean=np.array([np.log(Mc_z),np.log(eta),1.,1.,np.log(abs(Lambdat)),np.log(deff)])
     V=Fisher(Mc_z,eta,Lambdat,1.,1.,deff,ce_fs,ce_asd)
    
     Cov=np.absolute(np.linalg.inv(V))
     X1=[]
     X2=[]
     X3=[]
     X4=[]
     for i in range(1000):
         
         while True:
             Mc_z,eta,tc,pc,Lambdat,deff=np.random.multivariate_normal(Mean,Cov)
             M_z=Mc_z-0.6*eta
             if(M_z<np.log((m_u+m_u)*(1.+z_true)) and M_z>np.log((m_l+m_l)*(1.+z_true))):   
                 Mc_z=np.exp(Mc_z)
                 Lambdat=np.exp(Lambdat)
                 deff=np.exp(deff)
                 eta=np.exp(eta)
                 eta=0.25
                 M_0=m_u*2.
                 eta_0=m_u*m_u/(m_u*2.)**2
                 Mc_0=M_0*eta_0**0.6*(1.+z_true)
                 Lambda=Lambdat/((1.+7.*eta - 31.*eta**2)*16./13.)
                 h=0.1*Lambda
                 
                 if(Lambda<(max(L)-h) and Lambda>(min(L)+h)):
                         break
         X1.append(Mc_z)
         X2.append(eta)
         X3.append(Lambdat)
         X4.append(deff)
         #print(i)
     Mc_z=np.array(X1)
     eta=np.array(X2)
     Lambdat=np.array(X3)
     deff=np.array(X4)
     Th=(np.random.beta(2,4))#print(A)
     #print(Mc_z)
     Th=np.random.beta(2,4)
     return Mc_z, eta, Lambdat, deff, Th
def z_measured(Mc_z,Lambdat,eta):
    Lambda=Lambdat/((1.+7.*eta - 31.*eta**2)*16./13.)
    m_measured=mass(Lambda)
    
    
    #dm=(mass(Lambda+h)-mass(Lambda-h))/(2.*h)
    #print(dm/m_measured,Cov[1,1]**0.5)
    M_z=Mc_z/eta**(0.6)
    z_measured=M_z/(2.*m_measured) -1.
    return z_measured
def func(z,DL,HO,OO):
    if(OO<0. or HO<0.):
        return DL
    return abs(DL-np.log(FlatLambdaCDM(HO,OO).luminosity_distance(z).value))
def kernel(Z,DL,i):
    #S=1.0
    #for i in range(N):
    #DL=np.exp(DL)
    #dl=np.exp(dl)
    Dat=np.vstack([Z[i],DL[i]])
    #dl=np.exp(dl)
    f=stats.gaussian_kde(Dat)
    #Z1,DL1=np.meshgrid(z,dl)
    #pos=np.vstack([Z1.ravel(),DL1.ravel()])
    
    return f
def p(z):
    u=z/10.
    return u**2*(1.-u)**9/(special.beta(3,9)*10.)
def P_det(z,DL):
    m1=m1_SLY
    eta=0.25
    Lambdat=1.
    Mc_z=2.*m1*eta**0.6
    deff=np.exp(DL)*Mpc
    y=lambda Th: erfc((Rho_Th-SNR(Mc_z,eta,Lambdat,1.,1.,deff/Th,ce_fs,ce_asd)*stats.beta.pdf(Th,2,4))/2.**0.25)
    S=integrate.quad(y,0.0001,1.)
    return S[0]
def P_det_H(H,O,N):
    func=lambda t,z: P_det(z,np.log(FlatLambdaCDM(H,O).luminosity_distance(z).value),t)*stats.beta.pdf(t,2,4)
    
    gf=lambda x: 0.0001
    hf=lambda x: 1.0
    S=integrate.dblquad(func,0.0001,10.,gf,hf)
    return N*np.log(S[0])
#print(P_det_H(70.5,0.2736,20))
def choose(S,A):
    return S[A]
def Likelihood(H,O,Z_dat,DL_dat,NN):
    #Z1=np.linspace(min(Z_measured),max(Z_measured),1000)
    n=100
    L=0.0
    Z=np.linspace(0.0001,10.0,n)
    DL=np.array([np.log(FlatLambdaCDM(H,O).luminosity_distance(z).value) for z in Z] )
    Sel=np.array([P_det(Z[k],DL[k]) for k in range(len(Z))])
    Sel=np.trapz(Sel,Z)
    
    for j in range(NN):
        
        
        Kernel=kernel(Z_dat,DL_dat,j)
        
        #Ar2=P_det(Z,DL)
        #Ar3=np.array([Ar2[k,k] for k in range(len(Z))])
        #func=kernel2(Z_dat,DL_dat,i)
        #Integrand=np.array([np.exp(X(func(z,np.log(FlatLambdaCDM(H,O).luminosity_distance(z).value))))*p(z) for z in Z])
        Integrand=np.array([choose(Kernel.pdf([Z[k],DL[k]]),0)*p(Z[k]) for k in range(len(Z))])
        s=np.trapz(Integrand,Z)
     
        L+=np.log(s/Sel)
    print(H,L)
    return(L)
    
#print(len(DL),len(Z_measured))
Z_dat=[]
DL_dat=[]
N=50
z_max=0.0
Z_tr=[]
Hmin=60.
Hmax=80.
HN1=10
HN2=100
H1=np.linspace(Hmin,Hmax,HN1)
H2=np.linspace(Hmin,Hmax,HN2)
"""Sel=np.exp([P_det_H(h,0.2736,N) for h in H2])
Sel=Sel/max(Sel)"""

for i in range(0,N):
    #Mc_z, eta, Lambdat, deff, Th, z_true=Draw_Measured()
    while True:
        Th,Mc_z,eta,tc,pc,Lambdat,deff,m1,m2,z_true=Draw_true(m1_SLY,m2_SLY,Lambda_SLY,dl_dM_SLY,cosmo,z_Max,z_peak)
        if(SNR(Mc_z,eta,Lambdat,1.,1.,deff,ce_fs,ce_asd)>Rho_Th):
            break
    
    Mc_z, eta, Lambdat, deff, Th=Draw_Measured2(Mc_z, eta, Lambdat, deff, Th, z_true)
    """Ind=[]
    for j in range(0,len(Mc_z)):
        if(SNR(Mc_z[j],eta[j],Lambdat[j],1.,1.,deff[j],ce_fs,ce_asd)<8.0):
            Ind.append(j)
    Mc_z=np.delete(Mc_z,Ind)
    eta=np.delete(eta,Ind)
    deff=np.delete(deff,Ind)
    Lambdat=np.delete(Lambdat,Ind)"""
    f0=lambda x: z_measured(Mc_z[x],Lambdat[x],eta[x])
    #Z_dat.append(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(f0)(k) for k in range(len(Mc_z))))
    Z_dat.append([f0(k) for k in range(len(Mc_z))])
    if(z_max<max(Z_dat[i])):
        z_max=max(Z_dat[i])
    DL_dat.append(np.log(deff*np.random.beta(2,4,len(deff))/Mpc))
    #DL_dat.append(np.log(deff/Mpc))
    Z_tr.append(z_true)
    """if((i+1)%20==0 and i>140):
        Lh=np.array([(Likelihood(h,0.2736,Z_dat,DL_dat,i+1)-P_det_H(h,0.2736,i+1)) for h in H1])
        Lh=np.exp(Lh-max(Lh))
        f=interpolate.interp1d(H1,Lh,kind='cubic')
        Lh=np.array([f(h) for h in H2])
        Lh=np.exp(Lh-max(Lh))
        Ind=np.where(Lh>0.5*max(Lh))
        width1=abs(max(H2[Ind])-min(H2[Ind]))
        mean=sum(Lh*H2)
        std=sum(Lh*(H2-mean)**2)**0.5
        width2=2.*std
        width_true=abs(H2[np.argmax(Lh)]-70.5)
        f=open('Data_Widthd_delta_Sel.txt','a+')
        f.write(str(i+1)+'\t'+str(width1)+'\t'+str(width2)+'\t'+str(width_true)+'\t')
        f.write('\n')
        f.close()"""
        
    #print(i)




L_h=lambda x: Likelihood(x,0.2736,Z_dat,DL_dat,N)

"""#Lh=np.array(Parallel(n_jobs=multiprocessing.cpu_count())(delayed(L_h)(H1[p]) for p in range(len(H1))))
Lh=np.array([L_h(h) for h in H1])
Lh=Lh-max(Lh)
Sel=np.array([P_det_H(h,0.2736,N) for h in H1])
Sel=Sel-max(Sel)
Lh=np.exp(Lh-Sel)

f=interpolate.interp1d(H1,Lh,kind='cubic')"""

#sel=P_det_H0(Hmin,Hmax,10)
#Sel=[sel(h) for h in H2]
#Sel=N*np.log(Sel/max(Sel))
Lh=np.array([L_h(h) for h in H1])
Lh=np.exp(Lh-max(Lh))
#Lh=np.exp(Lh-Sel)
f=interpolate.interp1d(H1,Lh,kind='cubic')
Lh=np.array([f(h) for h in H2])
#Lh=np.exp(Lh)/Sel
Lh=Lh/max(Lh)
#Lh=Lh/np.exp(Sel)


ppl.plot(H2,Lh)
ppl.axvline(x=70.5)
ppl.show()


"""O1=np.linspace(0.0,0.6,10)
A=np.zeros([10,10],dtype=float)
Amax=-1000.
for i in range(0,10):
    for j in range(0,10):
        A[i,j]=Likelihood(H1[i],O1[j],Z_dat,DL_dat,N)-P_det_H(H1[i],O1[i],N)
        if(Amax<A[i,j]):
            Amax=A[i,j]
        print(i,j)
A=A-Amax
A=np.exp(A)
f=interpolate.interp2d(H1,O1,A,kind='cubic')
#H1=np.linspace(40,120,100)
O2=np.linspace(0.,0.6,100)
A=np.zeros([100,100],dtype=float)
A_max=-1000.
for i in range(0,100):
    for j in range(0,100):
        A[i,j]=f(H2[i],O2[j])
        if(A_max<A[i,j]):
            A_max=A[i,j]
    print(i)   
print(A)
A=A/Amax
ppl.contour(H2,O2,A,levels=[0.65,0.85,0.95])
ppl.axvline(x=70.5)
ppl.axhline(y=0.2736)
ppl.show()"""