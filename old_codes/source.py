
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:11:03 2019

@author: anarya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:06:50 2019

@author: anarya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:53:57 2019

@author: anarya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:07:39 2019

@author: anarya
"""

import numpy as np
from numpy import pi

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from matplotlib import pyplot as ppl
cosmo=FlatLambdaCDM(70.5,0.2736)

""" Geometricised Units; c=G=1, Mass[1500 meter/Solar Mass], Distance[meter], Time[3.0e8 meter/second]"""

"""Parameters: (ln Mc_z, ln(Lambda),tc,Phi_c,ln d)"""

# standard siren (EOS: SLY)
m1_SLY = m2_SLY = 1.433684*1500. 	  	 
Lambda_SLY=2.664334e+02
PN_order=3.5
dl_dM_SLY=(3.085067e+02-1.997018e+02)/((1.433684e+00  -1.493803)*1500.*2.)

# standard siren (EOS: MS1)
m1_MS1 = m2_MS1 = 1.431954 * 1500.
Lambda_MS1=1.244286e+03
PN_order=3.5
dl_dM_MS1=(1.244286e+03-1.024572e+03 )/(( 1.431954e+00  -1.482695e+00)*1500.*2.)

# standard siren (EOS: AP4)
m1_AP4 = m2_AP4 = 1.431337*1500. 	 
Lambda_AP4=2.285113e+02
PN_order=3.5
dl_dM_AP4=(2.767936e+02-1.891455e+02 )/(( 1.388857  -1.473573)*1500.*2.)

# load noise amplitude spectral density files
ce_fs, ce_asd , et_asd, aligo_asd = np.loadtxt('Amplitude_of_Noise_Spectral_Density.txt', usecols=[0,3,2,1],unpack = True)

# correct units
c=G=1
c1=3.0e8
Mpc=3.086e22

ce_fs *= c1**-1.
ce_asd *= c1**0.5


#Redshift Range
Z=np.logspace(-2,1)



#Point Particle Phase, Truncated at leading order
def Psi_PP(f,Mc_z):
    
    x=(pi*Mc_z*f*G/c**3)**(2./3.)
    s=3./(128.*x**(5./2.))
    return s

#Tidal Contribution to Phase, Truncated at Leading Order
def Psi_tidal(Mc_z,n,l,f):
    

    
    x=(pi*Mc_z*f)**(2./3.)
    
    
    s=(3.*39./(256.*n**2))*l*(x**(5./2.))
    return s

#Derivatives of the Phases
def dPsi_PP_Mc_z(f,Mc_z):        
     return (-5./(3.*Mc_z))*Psi_PP(f,Mc_z)
     
def dPsi_tidal_Mc_z(Mc_z,n,lt,f):
    return (5./(3.*Mc_z))*Psi_tidal(Mc_z,n,lt,f)

def dPsi_tidal_Lambda_tilde(Mc_z,n,lt,f):
    s=Psi_tidal(Mc_z,n,lt,f)/lt
    return s
def dPsi_tidal_eta(Mc_z,n,lt,f):
    s=Psi_tidal(Mc_z,n,lt,f)*(-2./n +(7.-31.*2.*n)*(16./13.)/((1.+7.*n - 31.*n**2)*16./13.))
    
    return s
def dPsi_PP_eta(f,Mc_z,n):
    s=(3./(128.*(np.pi*Mc_z*f)))*(20./9.)*(-2.*743.*n**(-1.-2./5.)/(336.*5)+3.*11.*n**(-1.+3./5.)/(5.*4.))
    return s

#Array of Derivatives of h
def dh(Mc_z,n,lt,f,phc,tc,d):           
    
    s=np.array([Mc_z*(5./(6.*Mc_z)+1.0j*dPsi_PP_Mc_z(f,Mc_z)+1.0j*dPsi_tidal_Mc_z(Mc_z,n,lt,f)),1.0j*lt*dPsi_tidal_Lambda_tilde(Mc_z,n,lt,f),1.0j*n*dPsi_tidal_eta(Mc_z,n,lt,f)+1.0j*n*dPsi_PP_eta(f,Mc_z,n),1.0j*2*pi*f,1.0j,-1.])
    s*=f**(-7./6.)
    return s


#Integrand
def F(Mch,n,l,f,phc,tc,d):         
    SS=dh(Mch,n,l,f,phc,tc,d)
    s=np.outer(SS,np.conjugate(SS))
    
    s=4.*np.real(s)
    return s

#Fisher Matrix and SNR
def Fisher(Mch,n,l,phc,tc,d,f,Sh):
    S=np.zeros([6,6])
    #rho=0.0
    Mz=Mch/(n**0.6)
    fisco=1./(6**0.5*6.*np.pi*Mz)
    f=f[np.where(f<fisco)]
    A = (5.0 * pi / 24.0)**0.5 * ( Mch**2) * (pi*Mch )**(-7.0/6.0) / d

#Trapezoidal Integration:
    
    for i in range(0,len(f)-1):
        
        S=S+(F(Mch,n,l,f[i],phc,tc,d)/Sh[i]**2+F(Mch,n,l,f[i+1],phc,tc,d)/Sh[i+1]**2)*(f[i+1]-f[i])*0.5
       
    return S*A**2

def SNR(Mc_z,n,l,phc,tc,d,f,Sh):
    rho=0.0
    Mz=Mc_z/(n**0.6)
    fisco=1./(6**0.5*6.*np.pi*Mz)
    f=f[np.where(f<fisco)]
    A = (5.0 * pi / 24.0)**0.5 * ( Mc_z**2) * (pi*Mc_z )**(-7.0/6.0) / d
    for i in range(0,len(f)-1):
        
        
        rho=rho+4.*(f[i]**(-7./3.)/Sh[i]**2+f[i+1]**(-7./3.)/Sh[i+1]**2)*(f[i+1]-f[i])*0.5
    return rho*A**2