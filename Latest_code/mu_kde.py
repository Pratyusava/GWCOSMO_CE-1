# -*- coding: utf-8 -*-
"""r
Created on Fri Oct 23 19:42:06 2020

@author: anarya
"""

import h5py
import numpy as np
from scipy.stats import gaussian_kde as KDE
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as ppl
from scipy.interpolate import interp1d
from source_mu import SNR
from sklearn import mixture
from scipy.stats import beta,gaussian_kde
from functools import partial
from multiprocessing import Pool, cpu_count
ce_fs, ce_asd , et_asd, aligo_asd = np.loadtxt('Amplitude_of_Noise_Spectral_Density.txt', usecols=[0,3,2,1],unpack = True)
Rho_Th=8.0
c1=3.0e8
Mpc=3.086e22

ce_fs *= c1**-1.
ce_asd *= c1**0.5
et_asd *= c1**0.5

Mc_o=2.* 1.433684*1500.*0.25**0.6
u_o= 1.433684*1500./2.
m_o=1.433684*1500.
m,L=np.loadtxt('Mass_Vs_TidalDeformability_SLY.txt',usecols=(0,1),unpack=True)
m*=1500.
Lamb=interp1d(m,L,kind='cubic')
L_o=Lamb(m_o)
eta=0.25
Lambdat_o=L_o*(1.+7.*eta - 31.*eta**2)*16./13.

N=100

def func2(arg,DL,Th,Z,Kernel):
    return np.sum(np.exp(Kernel.score_samples(arg))*Th*(1.+Z)/DL)
"""def draw(N_s):
    
    Z=10.*np.random.uniform(0.0001,1.,N_s+1)
    TH=np.random.uniform(0.0001,1.,N_s+1)
    
    ]p_accept=np.array([beta.pdf(Z[j+1],3,9)*beta.pdf(TH[j+1],2,4)/(beta.pdf(Z[j],3,9)*beta.pdf(TH[j],2,4)) for j in range(N_s)])
    
    for i in range(N_s):
        if(np.uniform(0.,1.)<min(1.0,p_accept[i])):
            Z[i]=Z[i+1]
            TH[i]=TH[i+1]
    return np.delete(Z,N_s),np.delete(TH,N_s)"""
def Likelihood(H,Kerr):
   
    N_samples=10000
    Z=10.*np.random.beta(3,9,N_samples)
    TH=np.random.beta(2,4,N_samples)
    Z[np.where(Z<0.0001)]=min(Z[np.where(Z>0.0001)])
    TH[np.where(TH==0.)]=min(TH[np.where(TH>0.0)])
    z=np.linspace(0.0001,10.,100)
    DL=np.array([FlatLambdaCDM(H,0.2736).luminosity_distance(z[j]).value for j in range(len(z))])
    #d_DL=np.array([(FlatLambdaCDM(H,0.2736).luminosity_distance(z[j]+0.00001).value-FlatLambdaCDM(H,0.2736).luminosity_distance(z[j]-0.00001).value)*0.5/0.00001 for j in range(len(z))])
    f=interp1d(z,DL,kind='cubic')
    DL=np.array([f(Z[j]) for j in range(len(Z))])
    #f=interp1d(z,d_DL,kind='cubic')
    #d_DL=np.array([f(Z[j]) for j in range(len(Z))])
    args=np.array([[(Mc_o*(1.+Z[j])),(u_o*(1.+Z[j])),(Lambdat_o),np.log(DL[j]/TH[j])] for j in range(len(Z))])
    with Pool(cpu_count()) as pool:
        f=partial(func2,args,DL,TH,Z)
        L=pool.map(f,Kerr)
    #print(L)
    L=np.array(L)
    L[np.where(L<=0.)]=0.001*min(L[np.where(L>0.)])
    S=np.sum(np.log(L))
    print(S,H)
    return(S)


H_min=25.
H_max=140.
NH1=25
NH2=100*NH1


H=np.linspace(H_min,H_max,NH1)

f=h5py.File('Sel_simple.h5','r')
Sel_real=np.array(f['Sel'])
f.close()
Sel_real[np.where(Sel_real<=0.)]=0.001*min(Sel_real[np.where(Sel_real>0.)])
print(Sel_real)
Kerr=[]
f=h5py.File('data_real_10000_4000.h5','r')
#print(f['Data']['1'])

for i in range(N):
    dat=np.array(f['Data'+str(i)])[0]
    n=len(dat)
    #dat=dat.astype(np.float32)
    #X,Y,Z,U=np.meshgrid(dat[0],dat[1],dat[2],dat[3])
    dat=np.array([[np.exp(dat[0][j]),np.exp(dat[1][j]),np.exp(dat[2][j]),dat[3][j]-np.log(Mpc)] for j in range(4000)])
    print(dat.shape)
    gmm = mixture.GaussianMixture(n_components=30,reg_covar=1.0e6).fit(dat)
    #kde=gaussian_kde(np.vstack([dat[0],dat[1],dat[2],dat[3]]))
    #print(gmm.score_samples(dat))
    Kerr.append(gmm)
    #Kerr.append(kde)
    print(i)

f.close()
print(Sel_real)

L=np.array([Likelihood(h,Kerr) for h in H])





f=h5py.File('L_H_'+str(N)+'.h5','w')
f.create_dataset('L',data=L)
f.create_dataset('H',data=H)
f.close()

"""f=h5py.File('L_H_100.h5','r')
L=np.array(f['L'])
H=np.array(f['H'])
L=np.exp(L-max(L))
f.close()"""
f=interp1d(H,L,kind='cubic')
H=np.linspace(H_min,H_max,NH2)
L=np.array([f(h) for h in H])
#L*=1./max(L)

#f.close()
#L=np.exp(L-max(L))
#f=interp1d(H,L,kind='cubic')
#H=np.linspace(H_min,H_max,NH2)
#L=np.array([f(h) for h in H])
L[np.where(L<=0.)]=0.001*min(L[np.where(L>0.)])
print(len(H),len(L))
y=np.log(L)-8.*np.log(Sel_real)
y=np.exp(y-max(y))
print(y)
ppl.plot(H,y/np.trapz(y,H),label='with pdet')
print(L)
print(np.trapz(L,H))
ppl.savefig('w')

ppl.plot(H,L/np.trapz(L,H),label='without pdet')
ppl.axvline(x=70.5)
ppl.suptitle(r'$\Omega_{m}=0.2736,\quad \rho_{th}=8$')
ppl.xlabel(r'$H_0$')
ppl.ylabel(r'$p(H_0|\vec{d})$')
ppl.legend()
ppl.savefig('kde_H0'+str(N)+'.pdf')
ppl.close()
y=-N*np.log(Sel_real)
y=np.exp(y-max(y))
print(y)
ppl.plot(H,y/np.trapz(y,H))
ppl.savefig('1bySel_N.pdf')
ppl.close()