import numpy as np
from numpy import pi
from scipy import stats
from scipy import special
from scipy.interpolate import interp1d
from source_mu import Fisher,SNR
#from source import SNR
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from matplotlib import pyplot as ppl
cosmo=FlatLambdaCDM(70.5,0.2736)
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import root
##import corner
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
    #M=m1+m2
    Mz=m1_z+m2_z
    mu_z=m1_z*m2_z/Mz
    
    pc=np.random.uniform(0,2.*np.pi)
    tc=np.random.uniform(0,1)
    #eta=m1*m2/M**2
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

def z_measured(Mc_z,Lambdat,mu_z):
    eta=(mu_z/Mc_z)**2.5
    Lambda=Lambdat/((1.+7.*eta - 31.*eta**2)*16./13.)
    m_measured=mass(Lambda)
    
    
    
    M_z=Mc_z/eta**(0.6)
    z_measured=M_z/(2.*m_measured) -1.
    return z_measured


#Use KDE to compute marginalized Quasi-Likelihood L(z,DL)
def kernel(Z,DL):
    Dat=np.vstack([Z,DL])
    f=stats.gaussian_kde(Dat)
    return f


#Distribution of Redshifts
def p(z):
    u=z/10.
    return u**2*(1.-u)**9/(special.beta(3,9)*10.)

#Selection Function
def P_det_old(z,DL):
    m1=m1_SLY
    eta=0.25
    Lambdat=1.
    Mc_z=(1.+z)*2.*m1*eta**0.6
    mu_z=m1*(1.+z)/2.
    deff=np.exp(DL)*Mpc
    rho_optimal = SNR(Mc_z,mu_z,deff,ce_fs,ce_asd)**0.5
    y=erfc((Rho_Th-rho_optimal)/2.**0.25)
    
    return y
AA = 1.93
BB = 3.93
from scipy.special import erf
def P_det(z,DL):
    m1=m1_SLY
    eta=0.25
    Lambdat=1.
    Mc_z=(1.+z)*2.*m1*eta**0.6
    mu_z=m1*(1.+z)/2.
    deff=np.exp(DL)*Mpc
    rho_optimal = SNR(Mc_z,mu_z,deff,ce_fs,ce_asd)**0.5
    w = Rho_Th/rho_optimal
    num = erf(AA-BB*w) - erf(AA-BB)
    den = erf(AA) - erf(AA-BB)
    return num/den

#Selection Function for H0 and Omega_m obtained by integrating P_det(Z,DL)*delta(DL-DL(z,H0,Omegam))*p(z)
def P_det_H(H,O):
    n=100
    Z=np.linspace(0.0001,10.0,n)
    DL=np.array([np.log(FlatLambdaCDM(H,O).luminosity_distance(z).value) for z in Z] )
    Sel=np.array([P_det(Z[k],DL[k]) for k in range(len(Z))])
    Sel=np.trapz(Sel,Z)
    return Sel

def choose(S,A):
    return S[A]


def p_r(r):
    
    return r*(1.-r)**3./special.beta(2,4)
#Likelihood of H0 and Omega_m obtained by Integrating L(z,DL)*delta(DL-DL(z,H0,Omegam))*p(z)
def Likelihood(H,O,kerr,NN):
    n=100
    L=0.0
    Z=np.linspace(0.0001,10.0,n)
    DL=np.array([np.log(FlatLambdaCDM(H,O).luminosity_distance(z).value) for z in Z] )
    R=np.linspace(0.0001,1.,10)
    Sel=np.array([[P_det(Z[k],DL[k]-np.log(r)) for k in range(len(Z))]for r in R])
    #Z1,DL1=np.meshgrid(Z,DL)
    #pos=np.vstack([Z1.ravel(),DL1.ravel()])
    for j in range(NN):
        Kernel=kerr[j]
        #Kernel=np.reshape(Kernel(pos).T,Z1.shape)
        #Integrand=np.array([[choose(Kernel.pdf([Z[k],np.exp(DL[k]-np.log(r))]),0)*p(Z[k])*p_r(r) for k in range(len(Z))] for r in R])
        Integrand=np.array([[Kernel.pdf([Z[k],np.exp(DL[k])/r])[0]*p(Z[k])*p_r(r) for k in range(len(Z))]for r in R])/Sel
        #print(Integrand)
        s=np.trapz(Integrand,Z,axis=1)
        s=np.trapz(s,R)
        #s=np.trapz(s,R)
        #ss = np.log(s)
        #print(ss)
        if(s<=0.0):
            continue
        else:
            L+=np.log(s)
    if(L==0.0):
        L=-1.0e20
    print(H,L)
    return(L)


    
#Draw Values for each Event
Z_dat=[]
DL_dat=[]
N=20
z_max=0.0
Z_tr=[]
Hmin=60.
Hmax=80.
HN1=10
HN2=100
H1=np.linspace(Hmin,Hmax,HN1)
H2=np.linspace(Hmin,Hmax,HN2)
Kerr=[]
#Draw Values for N events
for i in range(0,N):
#Introduce SNR Bias    
    while True:
        Th,Mc_z,mu_z,tc,pc,Lambdat,deff,m1,m2,z_true,dl_true=Draw_true(m1_SLY,m2_SLY,Lambda_SLY,dl_dM_SLY,cosmo)
        if( SNR(Mc_z,mu_z,deff,ce_fs,ce_asd)**0.5>Rho_Th):
            break
    #Z_true.append(z_true)
    #DL_True.append(dl_true)
    Mc_z, Mu_z, Lambdat, deff, Th=Draw_Measured(Mc_z, mu_z, Lambdat, deff, Th, z_true)
    Z_measured=np.array([z_measured(Mc_z[i],Lambdat[i],Mu_z[i]) for i in ranfor i in range(0,N):
#Introduce SNR Bias    
    while True:
        Th,Mc_z,mu_z,tc,pc,Lambdat,deff,m1,m2,z_true,dl_true=Draw_true(m1_SLY,m2_SLY,Lambda_SLY,dl_dM_SLY,cosmo)
        if( SNR(Mc_z,mu_z,deff,ce_fs,ce_asd)**0.5>Rho_Th):
            break
    #Z_true.append(z_true)
    #DL_True.append(dl_true)
    Mc_z, Mu_z, Lambdat, deff, Th=Draw_Measured(Mc_z, mu_z, Lambdat, deff, Th, z_true)ge(len(Mc_z))])
    Kerr.append(kernel(Z_measured,deff/Mpc))
    
#print(DL_dat[2])
#Interpolate Likelihood
L_h=lambda x: Likelihood(x,0.2736,Kerr,N)

Lh=np.array([L_h(h) for h in H1])
Lh=np.exp(Lh-max(Lh))

f=interpolate.interp1d(H1,Lh,kind='cubic')
Lh=np.array([f(h) for h in H2])

Lh=Lh/max(Lh)
print(Lh)

#Plot
ppl.plot(H2,Lh)
ppl.axvline(x=70.5)
ppl.xlabel('$H_{0}$ in Km/s/Mpc')
ppl.ylabel(r'$P_{post}(H_{0})$')
ppl.savefig('blah.pdf')