import numpy as np
from numpy import pi
from source_mu import Fisher, SNR
from astropy.cosmology import FlatLambdaCDM
from multiprocessing import Pool, cpu_count
from functools import partial
import h5py
import time
from scipy.interpolate import interp1d
""" Geometricised Units; c=G=1, Mass[1500 meter/Solar Mass], Distance[meter], Time[3.0e8 meter/second]"""

"""Parameters: (ln Mc_z, ln(Lambda),tc,Phi_c,ln d)"""
cosmo_true=FlatLambdaCDM(70.5,0.2736)
# standard siren (EOS: SLY)
m1_SLY = m2_SLY = 1.433684*1500. 	  	 
Lambda_SLY=2.664334e+02
PN_order=3.5
dl_dM_SLY=(3.085067e+02-1.997018e+02)/((1.433684e+00  -1.493803)*1500.*2.)
sm1=sm2=0.09*1500.
Z_true=1.5

N=100000
z_Peak=2.5
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


# correct units
c=G=1
c1=3.0e8
Mpc=3.086e22

ce_fs *= c1**-1.
ce_asd *= c1**0.5/30.
et_asd *= c1**0.5

rho_th=8.0
#Draw True Values
def Draw_true(m10,m20,Lambda,cosmo,z_max,z_peak,j):
    beta=2.*z_max/z_peak -2.
    m1=m10
    m2=m20
    M=m1+m2
    tc=pc=1.0
    eta=m1*m2/M**2
    Lambda=Lamb(m1)
    Lambdat=Lambda*(1.+7.*eta - 31.*eta**2)*16./13.
    while True:
        
        z=np.random.beta(3,beta+1)*z_max
        Th=np.random.beta(2,4)

        
        Mz=M*(1.+z)
        mu_z=(m1/2.)*(1.+z)
        
        
        Mc_z=Mz*eta**0.6
    
        d_l=cosmo.luminosity_distance(z)
    
        d_l=d_l.value*Mpc
        
        deff=d_l/Th
    
    
        
        V=Fisher(Mc_z,Lambdat,mu_z,deff,1.,1.,ce_fs,ce_asd)
        try:
            Cov=np.linalg.inv(V)
        except np.linalg.LinAlgError:
            continue
        rho=SNR(Mc_z,mu_z,deff,ce_fs,ce_asd)**0.5
        if(rho>rho_th):
            
            break
    Mean=np.array([np.log(Mc_z),np.log(mu_z),np.log(abs(Lambdat)),1.,1.,np.log(deff)])
    
    return [np.array(Mean),np.array(Cov)]

#Draw Measured Values from Fisher Matrix evaluated at True Values_samp
n_dim=4



def Draw_Measured(n,Tr,j):
     mean=Tr[j][0]
     cov=Tr[j][1]
     Mc_z,mu_z,Lambdat,tc,pc,deff=np.random.multivariate_normal(mean,cov,size=n).T
    


     Mc_z=np.array(Mc_z)
     mu_z=np.array(mu_z)
     Lambdat=np.array(Lambdat)
     deff=np.array(deff)
     
     return [np.array([Mc_z, mu_z, Lambdat, deff])]
N_events=10000
n_events=10
n_samples=4000
print('no. of measured samples per event='+str(n_samples))
l=0
fle=h5py.File('data_real_'+str(N_events)+'_'+str(n_samples)+'.h5','w')
for i in range(n_events,N_events+n_events,n_events):
    with Pool(cpu_count()) as pool:
        start=time.time()
        f=partial(Draw_true,m1_SLY,m2_SLY,Lambda_SLY,cosmo_true,z_Max,z_Peak)
        Tr=pool.map(f,range(n_events))
        end=time.time()
        Tr_time='True data samples took {0:.1f} seconds'.format(end-start)

    with Pool(cpu_count()) as pool:
        start=time.time()
        f=partial(Draw_Measured,n_samples,Tr)
        Dat=pool.map(f,range(n_events))
        end=time.time()
        Meas_time='Measured data samples took {0:.1f} seconds'.format(end-start)
    #print(Tr[5][0])
    n=0
    for k in range(i-n_events,i):
        fle.create_dataset('True'+str(k),data=np.array(Tr[n][0]))
        fle.create_dataset('Data'+str(k),data=np.array(Dat[n]))
        n+=1
    print(i,Tr_time,Meas_time)
fle.close()
"""for i in range(l,l+n_events):
        fle.create_dataset('True_'+str(i),data=np.array(Tr[i-l][0]))
        fle.create_dataset('Measured_'+str(i),data=np.array(Dat)[i-l])
print(l)"""
    


