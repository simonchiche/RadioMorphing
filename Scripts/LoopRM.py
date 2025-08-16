#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 02:24:48 2021

@author: chiche
"""

#region: Modules
import glob
import numpy as np
from RunRadioMorphing import run
from coreRadiomorphing_ground import extractData
import sys
from TestRadioMorphing.ScalingTestRadiomorphing import Scalingcheck
#from interpolation_test_ground import test_interpolation
from TestRadioMorphing.InterpolationTest.TestFilter import test_interpolation
import matplotlib.pyplot as plt
#from ScalingTests.CompareLDF import CompareLDF, GetRMtraces, ScalingcheckNew
from TestRadioMorphing.ScalingTests.CompareLDF import GetRMtraces, ScalingcheckNew
plt.ion()
##from SaveRMOutputs import SaveRMOutputs
from TestRadioMorphing.ModuleTestRM import GetTargetShowerParamFromZHSsim
#endregion



# Test library to compare the RM results with ZHAireS simulations
##PATH DEFINITIONS
#simulations = glob.glob("./TargetShowers/*.hdf5") 
RMarticlePath = "/Users/chiche/Desktop/RadioMorphingUptoDate/RMarticle"
OutputPath =  RMarticlePath + "/figures/EnergyScaling/"
#simulations = glob.glob(RMarticlePath + "/Simulations/EnergyScaling/*.hdf5") # Energy Scaling
#simulations = glob.glob(RMarticlePath + "/Simulations/AzimuthScaling/*.hdf5") # Energy Scaling
simulations = glob.glob(RMarticlePath + "/Simulations/ShowerFluctuations/*.hdf5") # Energy Scaling

# Flags
ScalingTest = False
InterpolationTest = True

# We loop the RM over the ZHS test library
for Simulated_path in simulations:
    
    energy, zenith, azimuth, xmaxsim, SimulatedShower, Flag = GetTargetShowerParamFromZHSsim(Simulated_path)
    if(Flag): continue # if the shower is already in the reference library, we skip it

# =============================================================================
#                         Radio Morphing launching
# =============================================================================

    TargetShower, RefShower, efield_interpolated, w_interpolated, IndexAll = \
    run(energy, zenith, azimuth, xmaxsim)

# =============================================================================
#                         Radio Morphing tests
# =============================================================================
    
    # Tests of the scaling 
    if(ScalingTest):
        ScalingcheckNew(TargetShower, SimulatedShower, efield_interpolated, OutputPath)

    # Tests of the interpolation
    if(InterpolationTest):
        ResidualPeak, TargetPeak, RefPeak =\
            test_interpolation(SimulatedShower, TargetShower, efield_interpolated, IndexAll, True)
        
        print(np.mean(ResidualPeak[~np.isnan(ResidualPeak)]), \
              np.std(ResidualPeak[~np.isnan(ResidualPeak)]))


        

            


 

    

