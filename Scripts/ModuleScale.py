#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:51:26 2021

@author: chiche
"""


import os
import h5py
import numpy as np
import Modules.hdf5fileinout as hdf5io
import sys
import glob
import matplotlib.pyplot as plt


# =============================================================================
#                           Energy scale
# =============================================================================
    
def EnergyScale(RefShower, TargetShower):
    
    Nant = RefShower.nant
    ref_energy = RefShower.energy
    target_energy = TargetShower.energy
    
    if(TargetShower.fluctuations):
        
        ref_energy = np.random.normal(ref_energy, 0.1*ref_energy)
    
    kE = target_energy/ref_energy
    
    scaled_traces = TargetShower.traces[:,Nant:]*kE
    
    return scaled_traces, kE

# =============================================================================
#                       Geomagnetic angle scale
# =============================================================================

def GeomagneticScale(RefShower, TargetShower):
    
    Nant = RefShower.nant
    
    ref_alpha = RefShower.get_alpha()
    target_alpha = TargetShower.get_alpha()
    kgeo  = np.sin(target_alpha)/np.sin(ref_alpha)
    ref_zenith = RefShower.zenith
    
    
    LimitZenith = 180
        
    if((ref_zenith>LimitZenith) & (abs(1-kgeo)>0.2)): 
    
        vxb, vxvxb = TargetShower.pos[:,1], TargetShower.pos[:,2]
        cos_eta = vxb/np.sqrt(vxb**2 + vxvxb**2)
        
        Evxb, Evxvxb = TargetShower.traces[:,2*Nant:3*Nant], \
        TargetShower.traces[:,3*Nant:4*Nant] 
        
        w = RefShower.get_w()
                
        w_kxkxb = []
        Ece = []
        Egeo = []
        
        for j in range(len(w)):
            if(abs(vxb[j])<1):
                w_kxkxb.append(w[j])
                Ece.append(abs(Evxvxb[:,j]))
                Egeo.append(abs(Evxb[:,j]))
          
        Evxb_scaled = np.zeros(np.shape(Evxb))
        
        for i in range(len(w)):
            
            diff = abs(w_kxkxb - w[i])
            minimum = np.argmin(diff)
            Evxb_scaled[:,i] = -(Egeo[minimum]*kgeo + Ece[minimum]*cos_eta[i])
            
    else: 
        Evxb_scaled = TargetShower.traces[:,2*Nant:3*Nant]*kgeo
    
    
    return Evxb_scaled, kgeo
    
 
# =============================================================================
#                          Density scaling
# =============================================================================
         
def DensityScale(RefShower, TargetShower):
    
    Nant = RefShower.nant
    
    xmax_target = TargetShower.xmaxpos #= TargetShower.getXmaxPosition()
            
    #XmaxHeight_target, DistDecayXmax = TargetShower._dist_decay_Xmax()  
    XmaxHeight_target =  TargetShower.getSphericalXmaxHeight()  
    XmaxHeight_ref = RefShower.getSphericalXmaxHeight()
            
    rho_ref = TargetShower._getAirDensity(XmaxHeight_ref, "linsley")
    rho_target = TargetShower._getAirDensity(XmaxHeight_target, "linsley")
        
    #krho =  1/np.sqrt(rho_ref/rho_target) # previous implementation
    
    #####fit Egeo #####
    
    def fit_broken_law(rho_bins):

        Phi0 = 1010
        gamma1 = -1.0047
        gamma2 = 0.2222
        rho_break  = 3.5e-4
        beta = 2.5
        rho0 = 1.87e-6
        y = (Phi0*(rho_bins/rho0)**(-gamma1))*(1+ (rho_bins/rho_break)**\
             ((gamma2- gamma1))/beta)**(-beta)
    
        return y
    
    b = 1.194878
    krho_geo = (fit_broken_law(rho_target)/fit_broken_law(rho_ref))
    krho_ce = (rho_target**b)/(rho_ref**b)
    
    scaled_traces_geo = TargetShower.traces[:,2*Nant:3*Nant]*krho_geo
    scaled_traces_ce = TargetShower.traces[:,3*Nant:4*Nant]*krho_ce
    
            
    return scaled_traces_geo, scaled_traces_ce, xmax_target, krho_geo, krho_ce

def DensityScaleBiasCorrected(RefShower, TargetShower):
    
    Nant = RefShower.nant
    
    xmax_target  = TargetShower.getXmaxPosition()
            
    XmaxHeight_target, DistDecayXmax = TargetShower._dist_decay_Xmax()    
    XmaxHeight_ref = RefShower.getSphericalXmaxHeight()
            
    rho_ref = TargetShower._getAirDensity(XmaxHeight_ref, "linsley")
    rho_target = TargetShower._getAirDensity(XmaxHeight_target, "linsley")
        
    #krho =  1/np.sqrt(rho_ref/rho_target) # previous implementation
    
    ############fit Egeo #####
    
    def fit_broken_law(rho_bins):

        Phi0 = 1010
        gamma1 = -1.0047
        gamma2 = 0.2222
        rho_break  = 3.5e-4
        beta = 2.5
        rho0 = 1.87e-6
        y = (Phi0*(rho_bins/rho0)**(-gamma1))*(1+ (rho_bins/rho_break)**\
             ((gamma2- gamma1))/beta)**(-beta)
    
        return y
    
    def fit_whigh(x):

        a = -41852.83493032
        b = 781642.06891597
        c = -2061529.47190934
        d = 2488060.24421785
        e = -1104757.3096756 
        y = a + b*x+ c*x**2 + d*x**3 + e*x**4
        return y

    def fit_wlow(x):

        a = -12181.07849176
        b = 1052230.34544697
        c = -3322672.24089232
        d = 4324392.77382339
        e = -2025673.01986316
        y = a + b*x+ c*x**2 + d*x**3 + e*x**4
        return y
    
    b = 1.194878
    krho_ce = (rho_target**b)/(rho_ref**b)
    CorrectBias = True
    if(CorrectBias):     
         krho_geo = (fit_broken_law(rho_target)/fit_broken_law(rho_ref))
         krho_geo_high = (fit_whigh(rho_target*1e3)/fit_whigh(rho_ref*1e3))
         krho_geo_low = (fit_wlow(rho_target*1e3)/fit_wlow(rho_ref*1e3))
         
         cerangle_ref = RefShower.get_cerenkov_angle()
         cerangle_target = TargetShower.get_cerenkov_angle()
    
         kstretch = cerangle_ref/cerangle_target
         w_target = RefShower.get_w()/kstretch
    
         Traces_Evxb = TargetShower.traces[:,2*Nant:3*Nant]

         Traces_Evxb[:, (w_target<0.75*cerangle_target)] = \
         Traces_Evxb[:, (w_target<0.75*cerangle_target)]*krho_geo_low
         
         Traces_Evxb[:, (w_target>=0.75*cerangle_target) & (w_target<=1.25*cerangle_target)] = \
         Traces_Evxb[:, (w_target>=0.75*cerangle_target) & (w_target<=1.25*cerangle_target)]*krho_geo
         
         Traces_Evxb[:, (w_target>1.25*cerangle_target)] = \
         Traces_Evxb[:, (w_target>1.25*cerangle_target)]*krho_geo_high
         
         scaled_traces_geo = Traces_Evxb
         
    scaled_traces_ce = TargetShower.traces[:,3*Nant:4*Nant]*krho_ce
            
    return scaled_traces_geo, scaled_traces_ce, xmax_target, krho_geo, krho_ce
    
# =============================================================================
#                       Cerenkov Stretch   
# =============================================================================

def CerenkovStretch(RefShower, TargetShower):
    
    Nant = RefShower.nant
    cerangle_ref = RefShower.get_cerenkov_angle()
    cerangle_target = TargetShower.get_cerenkov_angle()
    
    kstretch = cerangle_ref/cerangle_target
    w = RefShower.get_w()/kstretch
        
    v, vxb, vxvxb =  \
    TargetShower.pos[:,0], TargetShower.pos[:,1], TargetShower.pos[:,2]
    eta = np.arctan2(vxvxb, vxb)
    Distplane = TargetShower.distplane  
    d = Distplane*np.tan(w*np.pi/180.0)
    
    vxb_scaled = d*np.cos(eta) 
    vxvxb_scaled = d*np.sin(eta)
    
    scaled_pos = np.array([v,vxb_scaled, vxvxb_scaled]).T
    scaled_traces = TargetShower.traces[:,Nant:]*kstretch
                
    return scaled_pos, scaled_traces, kstretch
    




