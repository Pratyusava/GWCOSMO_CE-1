
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Pratyusava Baral Anarya Ray
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


from MakeSamples import *
from scipy.stats import norm,beta

NEVENTS = 50000
NSAMPLES = 10000
ASD_SCALE_FACTOR = 1
FNAME = "E" + str(NEVENTS) + "S" + str(NSAMPLES) + "ASF" + str(ASD_SCALE_FACTOR)
MODE = ""
H0_inj, Om0_inj = 67.7, 0.31
SNR_CUT = 8
def get_posterior(H0,Om0_inj,m1_samples,m2_samples,logD_samples,z_samples,m1_prior, m2_prior, z_prior, ThetaPrior, th):
    arr=np.zeros(len(H0))
    for i in range(len(H0)):
        zarr=np.linspace(0.0001,10.,1000)
        dLarr=luminosity_distance(zarr,H0[i],Om0_inj)
        logdL_values=np.log(np.interp(z_samples,zarr,dLarr)*MPC_TO_METER)
        theta_Sam=np.exp(logdL_values-logD_samples)
        prior_Th=np.interp(theta_Sam,ThetaPrior[0],ThetaPrior[1])
        theta_Sam[np.where(theta_Sam == np.inf)]=0
        posterior_each_event = np.sum(z_prior[:,:]*prior_Th[:,:]*theta_Sam[:,:]*m1_prior[:,:]*m2_prior[:,:],axis=1) #*theta_Sam[:,:]*M1_prior[:,:]*M2_prior[:,:]/np.exp(logD_Sam)*MPC_TO_METER
        print (H0[i],np.where(theta_Sam==np.max(theta_Sam)))
        log_posterior_each_event=np.log(posterior_each_event+1e-300)
        arr[i]=np.sum(log_posterior_each_event)
    return (arr)

if __name__ == "__main__":
    #np.random.seed(1)
    
    #NEVENTS = 5000
    z_samples = []
    logD_samples = []
    m1_samples = []
    m2_samples = []
    th=[]
    th_true_Sam = []
    q=[]
    m,L=np.loadtxt('Mass_Vs_TidalDeformability_SLY.txt',usecols=(0,1),unpack=True)
    m*=1500.
    h=m[1]-m[0]
    f=h5py.File(FNAME+".h5",'r')
    count = 5000
    for i in range(NEVENTS):
        snr = np.array(f['snr'+str(i)])
        if count == 0:
            break
        if (snr>SNR_CUT):
            print (count,snr)
            count -=1
            z_samples.append((np.array(f[MODE+'z_samples'+str(i)])[:NSAMPLES]))
            logD_samples.append(np.array(f[MODE+'log_deff_samples'+str(i)])[:NSAMPLES])
            m1_samples.append(np.array(f[MODE+'m1_samples'+str(i)])[:NSAMPLES])
            m2_samples.append(np.array(f[MODE+'m2_samples'+str(i)])[:NSAMPLES])
            th.append(np.array(f['theta'+str(i)]))

    f.close()
    m1_samples=np.array(m1_samples)
    m2_samples=np.array(m2_samples)
    th = np.array(th)
    z_samples=np.array(z_samples)
    logD_samples=np.array(logD_samples)

    st=0.
    en=2.
    m1_prior = np.interp(m1_samples/1500,np.linspace(0,2.66,100),norm(1.33,0.09).pdf(np.linspace(0,2.66,100)))*1500
    m2_prior = np.interp(m2_samples/1500,np.linspace(0,2.66,100),norm(1.33,0.09).pdf(np.linspace(0,2.66,100)))*1500
    
    z_prior = np.interp(z_samples/10,np.linspace(0,1,100),beta(3,9).pdf(np.linspace(0,1,100)))
    H0=np.linspace(85.,95.,25)
    #Omega_M=[0.31] 
    ThetaPrior = np.array([np.linspace(0,1,100),beta(2,4).pdf(np.linspace(0,1,100))])
    posterior_values = get_posterior(H0,Om0_inj,m1_samples,m2_samples,logD_samples,z_samples,m1_prior, m2_prior, z_prior, ThetaPrior, th)
    np.savetxt(str(SNR_CUT)+'.txt',np.array([H0,posterior_values]).T)
    #posterior_values -= NEVENTS*np.log(H0)
    plt.plot(H0,np.exp(posterior_values-np.max(posterior_values)),label = str(ASD_SCALE_FACTOR))
    
    plt.xlabel('H0')
    #plt.axvline(x=H0_inj)
    plt.savefig(str(SNR_CUT)+'.pdf')
