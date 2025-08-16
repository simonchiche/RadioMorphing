#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 02:45:28 2021

@author: chiche
"""

Local = True
Lyon = not(Local)

# region: Modules
import h5py
import numpy as np
import Modules.hdf5fileinout as hdf5io
from scaling import myscale
from interpolation_ground import do_interpolation_hdf5
import sys
import copy
from select_plane_ground import select_plane_ground, print_plane
#if(Local):
#    sys.path.append("./InterpolationTest")
#elif(Lyon): 
#    sys.path.append("/sps/trend/chiche/RadiomorphingUptoDate/InterpolationTest")
from Modules.module_signal_process import filter_traces
from Modules.ShowerClass import Shower, extractData, CerenkovStretch
import matplotlib.pyplot as plt
from ModuleCoreRM import TestOnePlane, GetTargetPositionsPerPlane, GenerateTargetShower,  FormatAntennaPositionsforInterpolation, CorrectOmegaAngleBias
# endregion

def process(sim_dir, shower,  out_dir):
    """Rescale and interpolate the radio traces for all antennas 
        - start the Radio Morphing procedure

    Parameters:
    ----------
        sim_dir: str
            path to the simulated traces
        shower: dict
            properties of the requested shower
        antennas: str
            path the requested antenna positions
        out_dir: str
            path where the output traces should be dumped
    """
    # Rescale the simulated showers to the requested one
    #         
# =============================================================================
#                         Shower building
# =============================================================================
    
    # We read the desired positions    
    desired_positionsAll =  np.loadtxt(shower["antennaDir"])
    # We select the reference shower and planes for each target antenna
    IndexAll, DplaneRefAll, PathAll, DplaneTargetAll, XmaxPos = select_plane_ground(desired_positionsAll, shower)
    IndexAllCut = IndexAll
   
    # Initilaization of the outputs
    EfieldAllAntennas, wAllAntennas, IndexAllAntennas, NinterpolatedAll = [], [], [], 0

    # Loop over all selected planes for interpolation
    for i in range(len(DplaneRefAll)):
        
        # We check if more than one plane was selected, if so we select only the antennas associated to the plane
        selected_plane, dplane, desired_positions =\
              GetTargetPositionsPerPlane(IndexAll, DplaneRefAll, DplaneTargetAll, desired_positionsAll, PathAll, i)
        
        # We initalize the Target Shower from the input parameters and the selected reference shower
        Nant, RefShower, TargetShower = GenerateTargetShower(selected_plane, shower, XmaxPos)
        #print_plane(RefShower, TargetShower, dplane)
        
    # =============================================================================
    #                              Scaling
    # =============================================================================

        # Scaling procedure of the  Radio Morphing
        TargetShower, krho_geo = myscale(RefShower, TargetShower)
        
    # =============================================================================
    #                          Preparing Interpolation      
    # =============================================================================
  
        ### à supprimer ?
        TargetShower.pos, TargetShower.traces[:,Nant:], ks = \
        CerenkovStretch(RefShower, TargetShower)
        
    # =============================================================================
    #                 antennas affectation for interpolation
    # =============================================================================
       
        TargetShower.pos, TargetShower.traces = TargetShower.GetinGeographicFrame()
        AntPosforInterpolation, initial_shape = FormatAntennaPositionsforInterpolation(RefShower, TargetShower, desired_positions)
        TargetShower.pos, TargetShower.nant  = AntPosforInterpolation, len(AntPosforInterpolation[:, 0])

    # =============================================================================
    #                 Pre interpolation filter (beta version)
    # =============================================================================
       
        pre_filtering = TargetShower.filter
        if pre_filtering:
            time_sample = int(len(TargetShower.traces[:,0]))
            TargetShower.traces = \
            filter_traces(TargetShower.traces, TargetShower.NantTraces, time_sample)
  
    # =============================================================================
    #                           Interpolation
    # =============================================================================
        
        # We perform the interpolation of the traces
        efield_interpolated, w_interpolated, Ninterpolated = \
        do_interpolation_hdf5(TargetShower, IndexAll[i], VoltageTraces = None, \
        FilteredVoltageTraces = None, antennamin=0,antennamax=initial_shape-1, \
        DISPLAY=False, usetrace="efield")  
    
    # =============================================================================
    #                           Output formating
    # =============================================================================
        
        TargetShower.distplane = dplane # TODO: remove for relase version, only for testing

        NinterpolatedAll = NinterpolatedAll + Ninterpolated
        print("Computing... %d/%d antennas" %(NinterpolatedAll, len(desired_positionsAll)))        
        EfieldAllAntennas =  EfieldAllAntennas + efield_interpolated
        wAllAntennas = wAllAntennas + w_interpolated
        IndexAllAntennas =  IndexAllAntennas + IndexAll[i]

    return 
