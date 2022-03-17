
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



import numpy as np
import h5py
import astropy.cosmology as cos
from scipy.interpolate import interp2d, interp1d
from numba import jit
import matplotlib.pyplot as plt

NEVENTS = 5#000
MAKE_LIKELIHOOD_SAMPLES = True
NSAMPLES = 10000
OPTIMAL = False
BNS_POPULATION_MEAN = 1.33 * 1500
BNS_POPULATION_SIGMA = 0.09 * 1500
c = 299792458
c_in_km_per_sec = c/1000#km/s
MPC_TO_METER = 3.0856775814671914e22
H0_inj, Om0_inj = 67.7, 0.31
ASD_SCALE_FACTOR = 100
INCLUDE_NOISE_IN_FISHER = False
M_LOW,M_HIGH,Z_LOW,Z_HIGH = 0.5*1500, 2*1500, 0, 10



######################################################################################
####### CALCULATE AN ARRAY OF LUMINOSITY DISTANCE GIVEN AN ARRAY OF REDSHIFT #########
#############THE REDSHIFT ARRAY SHOULD BE UNIFORMLY SPACED WITH SMALL dz############## 
######################################################################################
@jit(nopython=True)
def E(z, Om0, w_0, w_a):
    Ol0=1-Om0
    return ((Om0*(1+z)**3.)+Ol0*(1+z)**(3.0*(1+w_0+w_a*z/(1+z))))**0.5

@jit(nopython=True)
def integrate(z_l, z_h, Om0, w_0, w_a):
    z_lh=np.linspace(z_l,z_h,5)
    return np.trapz(1.0/E(z_lh,Om0,w_0,w_a),dx=z_lh[1]-z_lh[0])

@jit(nopython=True)
def luminosity_distance(z_array, H0, Om0 = 0.31, w_0 = -1, w_a = 0):
    lz=len(z_array)
    dLt=np.zeros(lz)
    dLarr=np.zeros(lz)
    t=integrate(0,z_array[0],Om0,w_0,w_a)
    dLt[0]=t
    dLarr[0]=(c*(1.0+z_array[0])*t)/H0
    for i in range(1,lz):
        dLt[i]=integrate(z_array[i-1],z_array[i],Om0,w_0,w_a)
        dLarr[i]=np.sum(dLt)*c_in_km_per_sec*(1+z_array[i])/H0
    return dLarr
######################################################################################


##################################### CALCULATE SNR ##################################
def SNR(m1, m2, z, deff, ce_fs, ce_asd):
    m_z=(m1+m2)*(1+z)
    mc_z = (m1*m2)**(3/5)/(m1+m2)**(1/5)
    f_isco=1./(6.**0.5*6.*np.pi*m_z)
    Args=np.where(ce_fs<f_isco)
    F=ce_fs[Args]
    Sh=ce_asd[Args]
    A=(5.0*np.pi/24.0)**0.5*(np.pi)**(-7./6.)
    rho=0.0
    for i in range(0,len(F)-1):
        rho+=(F[i]**(-7./3.)/Sh[i]**2+F[i+1]**(-7./3.)/Sh[i+1]**2)*(F[i+1]-F[i])*0.5
    rho=rho*A**2*mc_z**(5./3.)/deff**2
    return rho
#####################################################################################


####################FUNCTION TO INTEGRATE M1 M2 ANALYTICALLY##########################
def integrate_m1m2(fisher_matrix, true_parameter, m1_prior_mean, m1_prior_sigma):
    F = fisher_matrix
    M = m1_prior_mean
    S = m1_prior_sigma
    F00, F01, F02, F03, F11, F12, F13, F22, F23, F33 = F[0,0], F[0,1], F[0,2], F[0,3], F[1,1], F[1,2], F[1,3], F[2,2], F[2,3], F[3,3]
    #F00, F01, F02, F03, F11, F12, F13, F22, F23, F33 = .506, .5026, 485.198, -.00689, .499, 484.027, -0.00686, 529576.277, -7.1, 32.44
    M1, M2, Z, LD = true_parameter[:]
    tld2 = -0.5*(1.*F33 - 1.*F03**2*S**2 - 1.*F13**2*S**2 + 1.*F00*F33*S**2 + 1.*F11*F33*S**2 - 1.*F03**2*F11*S**4 + 2.*F01*F03*F13*S**4 - 1.*F00*F13**2*S**4 - 1.*F01**2*F33*S**4 + 1.*F00*F11*F33*S**4)/(1. + 1.*F00*S**2 + 1.*F11*S**2 - 1.*F01**2*S**4 + 1.*F00*F11*S**4)
    tz2 = -0.5*(1.*F22 - 1.*F02**2*S**2 - 1.*F12**2*S**2 + 1.*F00*F22*S**2 + 1.*F11*F22*S**2 - 1.*F02**2*F11*S**4 + 2.*F01*F02*F12*S**4 - 1.*F00*F12**2*S**4 - 1.*F01**2*F22*S**4 + 1.*F00*F11*F22*S**4)/(1. + 1.*F00*S**2 + 1.*F11*S**2 - 1.*F01**2*S**4 + 1.*F00*F11*S**4)
    tldz = (-1.*(1.*F23 - 1.*F02*F03*S**2 - 1.*F12*F13*S**2 + 1.*F00*F23*S**2 + 1.*F11*F23*S**2 - 1.*F02*F03*F11*S**4 + 1.*F01*F03*F12*S**4 + 1.*F01*F02*F13*S**4 - 1.*F00*F12*F13*S**4 - 1.*F01**2*F23*S**4 + 1.*F00*F11*F23*S**4))/(1. + 1.*F00*S**2 + 1.*F11*S**2 - 1.*F01**2*S**4 + 1.*F00*F11*S**4)
    tz = (1.*(1.*F23*LD - 1.*F02*M - 1.*F12*M + 1.*F02*M1 + 1.*F12*M2 - 1.*F02*F03*LD*S**2 - 1.*F12*F13*LD*S**2 + 1.*F00*F23*LD*S**2 + 1.*F11*F23*LD*S**2 + 1.*F01*F02*M*S**2 - 1.*F02*F11*M*S**2 - 1.*F00*F12*M*S**2 + 1.*F01*F12*M*S**2 + 1.*F02*F11*M1*S**2 - 1.*F01*F12*M1*S**2 - 1.*F01*F02*M2*S**2 + 1.*F00*F12*M2*S**2 - 1.*F02*F03*F11*LD*S**4 + 1.*F01*F03*F12*LD*S**4 + 1.*F01*F02*F13*LD*S**4 - 1.*F00*F12*F13*LD*S**4 - 1.*F01**2*F23*LD*S**4 + 1.*F00*F11*F23*LD*S**4 + 1.*F22*Z - 1.*F02**2*S**2*Z - 1.*F12**2*S**2*Z + 1.*F00*F22*S**2*Z + 1.*F11*F22*S**2*Z - 1.*F02**2*F11*S**4*Z + 2.*F01*F02*F12*S**4*Z - 1.*F00*F12**2*S**4*Z - 1.*F01**2*F22*S**4*Z + 1.*F00*F11*F22*S**4*Z))/(1. + 1.*F00*S**2 + 1.*F11*S**2 - 1.*F01**2*S**4 + 1.*F00*F11*S**4)
    tld = (1.*(1.*F33*LD - 1.*F03*M - 1.*F13*M + 1.*F03*M1 + 1.*F13*M2 - 1.*F03**2*LD*S**2 - 1.*F13**2*LD*S**2 + 1.*F00*F33*LD*S**2 + 1.*F11*F33*LD*S**2 + 1.*F01*F03*M*S**2 - 1.*F03*F11*M*S**2 - 1.*F00*F13*M*S**2 + 1.*F01*F13*M*S**2 +1.*F03*F11*M1*S**2 - 1.*F01*F13*M1*S**2 - 1.*F01*F03*M2*S**2 + 1.*F00*F13*M2*S**2 - 1.*F03**2*F11*LD*S**4 + 2.*F01*F03*F13*LD*S**4 - 1.*F00*F13**2*LD*S**4 - 1.*F01**2*F33*LD*S**4 + 1.*F00*F11*F33*LD*S**4 + 1.*F23*Z - 1.*F02*F03*S**2*Z - 1.*F12*F13*S**2*Z + 1.*F00*F23*S**2*Z + 1.*F11*F23*S**2*Z - 1.*F02*F03*F11*S**4*Z + 1.*F01*F03*F12*S**4*Z + 1.*F01*F02*F13*S**4*Z - 1.*F00*F12*F13*S**4*Z - 1.*F01**2*F23*S**4*Z + 1.*F00*F11*F23*S**4*Z))/(1. + 1.*F00*S**2 + 1.*F11*S**2 - 1.*F01**2*S**4 + 1.*F00*F11*S**4)
    A=-2*tz2
    B=-tldz
    C=-2*tld2
    Fzld = np.array([[A,B],[B,C]])
    ld0 = (A*tld-B*tz)/(A*C - B**2)
    z0 = (C*tz-B*tld)/(A*C - B**2)
    return Fzld,np.linalg.inv(Fzld),np.array([z0,ld0])

############################################################################
######DEFINE PN WAVEFORM TO DOMINANT ORDER OF EACH TERM##################
def Psi_PP(mc_z,q,f):
    return 3./(256.*(2*np.pi*mc_z*f)**(5./3.)) 
def Psi_PPht1(mc_z,q,f):
    return (3*3715./(256.*756.))*(1+q)**(4./5)/(q**(2/5)*mc_z)*(2*np.pi*f)**(-1.)
def Psi_PPht2(mc_z,q,f):
    return (55./(256.*3.))*q**(3./5)/(mc_z*(1+q)**(6./5))*(2*np.pi*f)**(-1.)
def Psi_tidal(mc_z,q,lambdat,f):
    return (-3.*39./8.)*lambdat*(np.pi*f)**(5./3.)*mc_z**(5/3.)*(1+q)**4/q**2.
    
def h(mc_z,q,lambdat,deff,tc,pc,f):
    return mc_z**(5./6.)*f**(-7./6.)*np.exp(1.0j*(Psi_PP(mc_z,q,f)+Psi_PPht1(mc_z,q,f)+Psi_PPht2(mc_z,q,f)+Psi_tidal(mc_z,q,lambdat,f)+2.*np.pi*f*tc-pc-np.pi/4.))/deff
#################################################################################



################################################################################
#########################CALCULATE THE FISHER MATRIX############################
################################################################################
def dh(m1,m2,z,deff,tc,pc,f,lambdat_from_mass):
    delta = 15
    mc_z = ((m1 * m2)**(0.6)/(m1+m2)**0.2)*(1+z)
    q = m1/m2
    lambdat=(1./26)* (lambdat_from_mass(m1)*(1+12*m2/m1)+lambdat_from_mass(m2)*(1+12*m1/m2))
    dhdlogM_cz=h(mc_z,q,lambdat,deff,tc,pc,f)*(5./6. - 5.0j/3 * Psi_PP(mc_z,q,f) + 5.j/3*Psi_tidal(mc_z,q,lambdat,f) -1j * Psi_PPht2(mc_z,q,f) -1.j * Psi_PPht1(mc_z,q,f))
    dhdlogq=1.0j*h(mc_z,q,lambdat,deff,tc,pc,f)*(Psi_PPht1(mc_z,q,f)*(4*q/(5*(1+q))-2./5) + Psi_PPht2(mc_z,q,f)*(3/5-(6*q)/(5*(1+q)))+Psi_tidal(mc_z,q,lambdat,f)*(4*q/(1+q)-2.) )
    dhdlogDeff=h(mc_z,q,lambdat,deff,tc,pc,f)*(-1.)
    dhdlogLam=h(mc_z,q,lambdat,deff,tc,pc,f)*1.0j*Psi_tidal(mc_z,q,lambdat,f)
    dhdtc=h(mc_z,q,lambdat,deff,tc,pc,f)*(-1)*1.0j
    dhdphic=h(mc_z,q,lambdat,deff,tc,pc,f)*2.*np.pi*f*1.0j
    
    dlogmc_zdlogm1 = 3/5 - 1/5 * m1/(m1+m2)
    dlogmc_zdlogm2 = 3./5 - 1/5 * m2/(m1+m2)
    dloglambdatdlogm1 = (m1/(26*lambdat))*((1+12*m2/m1)*(lambdat_from_mass(m1+delta)-lambdat_from_mass(m1-delta))/(2*delta)-12*lambdat_from_mass(m1)*m2/m1**2 + 12 *lambdat_from_mass(m2)/m2)
    dloglambdatdlogm2 = (m2/(26*lambdat))*((1+12*m1/m2)*(lambdat_from_mass(m2+delta)-lambdat_from_mass(m2-delta))/(2*delta)-12*lambdat_from_mass(m2)*m1/m2**2 + 12 *lambdat_from_mass(m1)/m1)
    dhdlogm1 = dhdlogM_cz * dlogmc_zdlogm1 + dhdlogq * 1 + dhdlogLam * dloglambdatdlogm1
    dhdlogm2 = dhdlogM_cz * dlogmc_zdlogm2 + dhdlogq * (-1) + dhdlogLam * dloglambdatdlogm2
    dhdlogz = z/(1+z) * dhdlogM_cz
    S=np.array([dhdlogm1/m1,dhdlogm2/m2,dhdlogz/z,dhdlogDeff,dhdtc,dhdphic])
    return S

def Integrand(m1,m2,z,deff,tc,pc,f,lambdat_from_mass):
     I=np.array(np.outer(dh(m1,m2,z,deff,tc,pc,f,lambdat_from_mass),np.conjugate(dh(m1,m2,z,deff,tc,pc,f,lambdat_from_mass))))
     return 4.0*np.real(I)
     
     
def Fisher(m1,m2,z,deff,tc,pc,lambdat_from_mass,ce_fs,ce_asd):
    S = np.zeros([6,6])
    M_z = (m1 + m2)*(1+z)
    f_isco=1./(6.**0.5*6.*np.pi*M_z)#check 
    Args=np.where(ce_fs<f_isco)
    F=ce_fs[Args]
    Sh=ce_asd[Args]
    A=(5.0*np.pi/24.0)**0.5*(np.pi)**(-7./6.)#??
    #rho=0.0
    for i in range(0,len(F)-1):
        S+=(Integrand(m1,m2,z,deff,tc,pc,F[i],lambdat_from_mass)/Sh[i]**2 +Integrand(m1,m2,z,deff,tc,pc,F[i+1],lambdat_from_mass)/Sh[i+1]**2)*(F[i+1]-F[i])*0.5
    S=S*A**2
    return S
################################################################################


def draw_true(BNSMassDist_mean,BNSMassDist_var,cosmo_true_z_dl_list,z_max=10,a_th=2,b_th=4,optimal=True):
    m1_true=np.random.normal(BNSMassDist_mean,BNSMassDist_var)
    m2_true=np.random.normal(BNSMassDist_mean,BNSMassDist_var)
    z=np.random.beta(3,9)*z_max
    if (optimal):
        th=1
    else:
        th=np.random.beta(a_th,b_th)
        #th=np.random.random()
    d_eff=np.interp(z,cosmo_true_z_dl_list[0],cosmo_true_z_dl_list[1])/th
    return m1_true, m2_true, z, d_eff*MPC_TO_METER, th

    
if __name__ == "__main__":
    np.random.seed(10)
    m,L=np.loadtxt('Mass_Vs_TidalDeformability_SLY.txt',usecols=(0,1),unpack=True)
    m*=1500.
    lambdat_from_mass=interp1d(m,L,kind='cubic')
    ce_fs, ce_asd , et_asd, aligo_asd = np.loadtxt('Amplitude_of_Noise_Spectral_Density.txt', usecols=[0,3,2,1],unpack = True)
    ce_fs *= c**-1.
    ce_asd *= c**0.5
    ce_asd/=ASD_SCALE_FACTOR
    et_asd *= c**0.5
    zarr=np.linspace(0.0001,10.,1000)
    dLarr=luminosity_distance(zarr,H0_inj,Om0_inj)
    cosmo_true_z_dl_list=np.array([zarr,dLarr])
    with h5py.File("E"+str(NEVENTS)+"ASD"+str(ASD_SCALE_FACTOR)+str(INCLUDE_NOISE_IN_FISHER)+".h5", "w") as f:#SNRwoth1000S1e4
        for i in range(NEVENTS):
            m1_true, m2_true, z_true, deff_true, theta_true = draw_true(BNS_POPULATION_MEAN, BNS_POPULATION_SIGMA, cosmo_true_z_dl_list,optimal=OPTIMAL)
            #print (i,z,th)
            fisher = Fisher(m1_true,m2_true,z_true,deff_true,1,1,lambdat_from_mass,ce_fs,ce_asd)
            snr = SNR(m1_true,m2_true,z_true,deff_true,ce_fs,ce_asd)
            
            CoV=np.linalg.inv(fisher[:-2,:-2])
            if (INCLUDE_NOISE_IN_FISHER):
                m1_measured, m2_measured, z_measured, log_deff_measured = np.random.multivariate_normal((np.array([m1_true, m2_true, z_true, np.log(deff_true)])),CoV,1).T
                while m1_measured<M_LOW or m2_measured<M_LOW or m1_measured>M_HIGH or m2_measured>M_HIGH or z_measured<Z_LOW or z_measured>Z_HIGH:
                    print(1)
                    m1_measured, m2_measured, z_measured, log_deff_measured = np.random.multivariate_normal((np.array([m1_true, m2_true, z_true, np.log(deff_true)])),CoV,1).T
                f.create_dataset('measured_parameters'+str(i),data=(np.array([m1_measured, m2_measured, z_measured, log_deff_measured])))
                fisher = Fisher(m1_measured,m2_measured,z_measured,np.exp(log_deff_measured),1,1,lambdat_from_mass,ce_fs,ce_asd)
                snr = SNR(m1_true,m2_true,z_true,deff_true,ce_fs,ce_asd)
            
                CoV=np.linalg.inv(fisher[:-2,:-2])
                
                
            fisher_reduced, CoV_reduced, peak_reduced = integrate_m1m2(fisher[:-2,:-2], np.array([m1_true, m2_true, z_true, np.log(deff_true)]),BNS_POPULATION_MEAN, BNS_POPULATION_SIGMA)        
            #fisher_reduced, CoV_reduced, peak_reduced = integrate_m1m2_flatmass(fisher[:-2,:-2], np.array([m1_true, m2_true, z, np.log(deff)]))
            print (i,snr,m1_true, m2_true, z_true, theta_true)
            f.create_dataset('injected_parameters'+str(i),data=(np.array([m1_true, m2_true, z_true, np.log(deff_true)])))
            f.create_dataset('snr'+str(i),data=snr)
            f.create_dataset('fisher_matrix'+str(i),data=np.array(fisher))
            f.create_dataset('theta'+str(i),data=theta_true)
            
            f.create_dataset('covariance_matrix'+str(i),data=CoV)
            f.create_dataset('reduced_fisher_matrix'+str(i),data=np.array(fisher_reduced))
            f.create_dataset('reduced_covariance_matrix'+str(i),data=CoV_reduced)
            f.create_dataset('reduced_mean'+str(i),data=peak_reduced)
            if (MAKE_LIKELIHOOD_SAMPLES):
                m1,m2,z,logdeff = np.random.multivariate_normal((np.array([m1_true, m2_true, z_true, np.log(deff_true)])),CoV,NSAMPLES).T
                zred, logdeffred = np.random.multivariate_normal(peak_reduced,CoV_reduced,NSAMPLES).T
                f.create_dataset('m1_samples'+str(i),data=m1)
                f.create_dataset('m2_samples'+str(i),data=m2)
                f.create_dataset('log_deff_samples'+str(i),data=logdeff)
                f.create_dataset('z_samples'+str(i),data=z)
                f.create_dataset('reduced_z_samples'+str(i),data=zred)
                f.create_dataset('reduced_log_deff_samples'+str(i),data=logdeffred)
            

'''
def integrate_m1m2_flatmass(fisher_matrix, true_parameter):
    F = fisher_matrix
    F00, F01, F02, F03, F11, F12, F13, F22, F23, F33 = F[0,0], F[0,1], F[0,2], F[0,3], F[1,1], F[1,2], F[1,3], F[2,2], F[2,3], F[3,3]
    #F00, F01, F02, F03, F11, F12, F13, F22, F23, F33 = .506, .5026, 485.198, -.00689, .499, 484.027, -0.00686, 529576.277, -7.1, 32.44
    M1, M2, Z, LD = true_parameter[:]
    tz2 = (-0.5*F02**2*F11)/(F01**2 - 1.*F00*F11) + (1.*F01*F02*F12)/(F01**2 - 1.*F00*F11) - (0.5*F00*F12**2)/(F01**2 - 1.*F00*F11) - 0.5*F22
    tld2 = (-0.5*F03**2*F11)/(F01**2 - 1.*F00*F11) + (1.*F01*F03*F13)/(F01**2 - 1.*F00*F11) - (0.5*F00*F13**2)/(F01**2 - 1.*F00*F11) - 0.5*F33
    tldz = (-1.*F02*F03*F11)/(F01**2 - 1.*F00*F11) + (1.*F01*F03*F12)/(F01**2 - 1.*F00*F11) + (1.*F01*F02*F13)/(F01**2 - 1.*F00*F11) - (1.*F00*F12*F13)/(F01**2 - 1.*F00*F11) - 1.*F23
    tz = (1.*F02*F03*F11*LD)/(F01**2 - 1.*F00*F11) - (1.*F01*F03*F12*LD)/(F01**2 - 1.*F00*F11) - (1.*F01*F02*F13*LD)/(F01**2 - 1.*F00*F11) + (1.*F00*F12*F13*LD)/(F01**2 - 1.*F00*F11) + 1.*F23*LD + 1.*F02*M1 - (1.*F01**2*F02*M1)/(F01**2 - 1.*F00*F11) + (1.*F00*F02*F11*M1)/(F01**2 - 1.*F00*F11) + 1.*F12*M2 - (1.*F01**2*F12*M2)/(F01**2 - 1.*F00*F11) + (1.*F00*F11*F12*M2)/(F01**2 - 1.*F00*F11) + (1.*F02**2*F11*Z)/(F01**2 - 1.*F00*F11) -  (2.*F01*F02*F12*Z)/(F01**2 - 1.*F00*F11) + (1.*F00*F12**2*Z)/(F01**2 - 1.*F00*F11) + 1.*F22*Z
    tld = (1.*F03**2*F11*LD)/(F01**2 - 1.*F00*F11) - (2.*F01*F03*F13*LD)/(F01**2 - 1.*F00*F11) + (1.*F00*F13**2*LD)/(F01**2 - 1.*F00*F11) + 1.*F33*LD + 1.*F03*M1 - (1.*F01**2*F03*M1)/(F01**2 - 1.*F00*F11) + (1.*F00*F03*F11*M1)/(F01**2 - 1.*F00*F11) + 1.*F13*M2 - (1.*F01**2*F13*M2)/(F01**2 - 1.*F00*F11) + (1.*F00*F11*F13*M2)/(F01**2 - 1.*F00*F11) + (1.*F02*F03*F11*Z)/(F01**2 - 1.*F00*F11) - (1.*F01*F03*F12*Z)/(F01**2 - 1.*F00*F11) - (1.*F01*F02*F13*Z)/(F01**2 - 1.*F00*F11) +   (1.*F00*F12*F13*Z)/(F01**2 - 1.*F00*F11) + 1.*F23*Z
    A=-2*tz2
    B=-tldz
    C=-2*tld2
    Fzld = np.array([[A,B],[B,C]])
    ld0 = (A*tld-B*tz)/(A*C - B**2)
    z0 = (C*tz-B*tld)/(A*C - B**2)
    print ([z0,ld0])
    return Fzld,np.linalg.inv(Fzld),np.array([z0,ld0])
    '''