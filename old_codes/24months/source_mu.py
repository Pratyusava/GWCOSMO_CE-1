#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:08:42 2020

@author: anarya
"""

import numpy as np


def Psi_PP(Mc_z,f):
    return 3./(128.*(np.pi*Mc_z*f)**(5./3.))
def Psi_tidal(Mc_z,Lambdat,mu_z,f):
    return (3.*39./256.)*Lambdat*(np.pi*f)**(5./3.)*Mc_z**(20./3.)/mu_z**5.
def h(Mc_z,Lambdat,mu_z,deff,tc,pc,f):
    return Mc_z**(5./6.)*f**(-7./6.)*np.exp(1.0j*(Psi_PP(Mc_z,f)+Psi_tidal(Mc_z,Lambdat,mu_z,f)+2.*np.pi*f*tc-pc-np.pi/4.))/deff
def dh(Mc_z,Lambdat,mu_z,deff,tc,pc,f):
    cor=(60./(128.*9.))*(1./(np.pi*f))*(-743./(336.*mu_z)+(33./8.)*mu_z**(3./2.)/Mc_z**(5./2.))
    S=np.array([(5./6.+1.0j*20.*Psi_tidal(Mc_z,Lambdat,mu_z,f)/3. -1.0j*5.*Psi_PP(Mc_z,f)/3.),1.0j*cor+5.*1.0j*Psi_tidal(Mc_z,Lambdat,mu_z,f),1.0j*Psi_tidal(Mc_z,Lambdat,mu_z,f),-1.,1.0j*2.*np.pi*f,-1.0j])*h(Mc_z,Lambdat,mu_z,deff,tc,pc,f)
    return S
def Integrand(Mc_z,Lambdat,mu_z,deff,tc,pc,f):
     I=np.array(np.outer(dh(Mc_z,Lambdat,mu_z,deff,tc,pc,f),np.conjugate(dh(Mc_z,Lambdat,mu_z,deff,tc,pc,f))))
     return 4.0*np.real(I)
def Fisher(Mc_z,Lambdat,mu_z,deff,tc,pc,ce_fs,ce_asd):
    S=np.zeros([6,6])
    M_z=Mc_z**(5./2.)/mu_z**(3./2.)
    f_isco=1./(6.**0.5*6.*np.pi*M_z)
    Args=np.where(ce_fs<f_isco)
    F=ce_fs[Args]
    Sh=ce_asd[Args]
    A=(5.0*np.pi/24.0)**0.5*(np.pi)**(-7./6.)
    #rho=0.0
    for i in range(0,len(F)-1):
        S+=(Integrand(Mc_z,Lambdat,mu_z,deff,tc,pc,F[i])/Sh[i]**2+Integrand(Mc_z,Lambdat,mu_z,deff,tc,pc,F[i+1])/Sh[i+1]**2)*(F[i+1]-F[i])*0.5
        #rho+=(F[i]**(-7./3.)/Sh[i]**2+F[i+1]**(7./3.)/Sh[i+1]**2)*(F[i+1]-F[i])*0.5
    #rho=rho*A**2*Mc_z**(5./3.)/deff**2
    S=S*A**2
    return S

    
def SNR(Mc_z,mu_z,deff,ce_fs,ce_asd):
    M_z=Mc_z**(5./2.)/mu_z**(3./2.)
    f_isco=1./(6.**0.5*6.*np.pi*M_z)
    Args=np.where(ce_fs<f_isco)
    F=ce_fs[Args]
    Sh=ce_asd[Args]
    A=(5.0*np.pi/24.0)**0.5*(np.pi)**(-7./6.)
    rho=0.0
    for i in range(0,len(F)-1):
        rho+=(F[i]**(-7./3.)/Sh[i]**2+F[i+1]**(7./3.)/Sh[i+1]**2)*(F[i+1]-F[i])*0.5
    rho=rho*A**2*Mc_z**(5./3.)/deff**2
    return rho