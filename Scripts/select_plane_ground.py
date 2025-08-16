#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:08:09 2021

@author: chiche
"""


import numpy as np
import glob
import sys
from Modules.ShowerClass import Shower, extractData, CerenkovStretch
import matplotlib.pyplot as plt
from interpolation_ground import SelectAntennasForInterpolationCorrected, \
GetAntennaAnglesSimon, get_center, ComputeAntennaPhi, get_in_shower_plane
from Modules.ModuleGetXmaxPos import getXmaxPosition

import copy

Local = True
Lyon = not(Local)


def select_zenith(target_zenith):
    
    # zenith of the target shower
    target_zenith = 180 - target_zenith # cosmic ray convention
    
    # zenith of the ref library
    # To modify if we change the ref library
    # TODO: change in Lyon
    zenith_sim  = np.array([65.5, 67.8, 74.8, 79.5, 82.7, 85.0, 86.5])
    
    # We use as reference shower the shower with the closest zenith from the
    # targeted one
    min_index = np.argmin(abs(zenith_sim - target_zenith))
    selected_zenith =  zenith_sim[min_index]
    
    return selected_zenith


def select_azimuth(path, target_azimuth):

    sim = glob.glob(path)
    azimuth_sim = []
    n = len(sim)

    # azimuth of the target shower
    target_azimuth = abs(180  - target_azimuth)
        
    # azimuth of the ref library
    # To modify if we change the ref library
    for i in range(n):
    #azimuth_sim  = np.array([0, 90, 180])
        azRef = float(sim[i].split("_")[-3])
        azimuth_sim.append(azRef)
    
    azimuth_sim = np.array(azimuth_sim)
    
    # We use as reference shower the shower with the closest azimuth from the
    # targeted one
    min_index = np.argmin(abs(azimuth_sim - target_azimuth))
    selected_azimuth =  azimuth_sim[min_index]
    
    return selected_azimuth


def select_path(path, dplane, selected_azimuth):
    
    # Gives the path to the reference simulation
    
    sim = glob.glob(path)
    n = len(sim)
    dsim = []
    path = []
        
    for j in range(n):        
        azsimRef = float(sim[j].split("_")[-3])
        if(int(azsimRef) == int(selected_azimuth)):         
            dsim.append(float(sim[j].split("_")[-1][:-5]))
            path.append(sim[j])
    
    # array containing all the reference planes
    dsim = np.array(dsim)
    path = np.array(path)
    
    PlaneIndex =  np.argsort(-dsim)
    path = path[PlaneIndex]
    dsim =  -np.sort(-dsim)

    # 2D array containing the Xmax-antennas distance (row) 
    # for each plane (line)
    DplaneAll = np.zeros([len(dplane), len(dsim)])
    # Index of all antennas
    AntennasAll = np.arange(0, len(dplane), 1)
    
    for i in range(len(dsim)):
        DiffPlane = abs(dplane - dsim[i])
        DplaneAll[:,i] = DiffPlane
    
    # Index giving the closest plane to each antenna
    ArgAll = np.argmin(DplaneAll, axis = 1)
    
    # Index of Antennas selected for each plane
    IndexAll = []
    # Each selected plane of antennas
    DsimAll = []
    PathAll = []

    # Loop over all the planes
    for i in range(len(dsim)):
        
        #if(AntennasAll[dsim[ArgAll] == dsim[i]] != []):
        IndexAll.append(AntennasAll[dsim[ArgAll] == dsim[i]])
        DsimAll.append(dsim[i])
        PathAll.append(path[i])
      
    IndexAll = np.array(IndexAll, dtype = object)                            
    
    '''
    index_all = np.argsort(abs(dsim - dplane))
    min_index = index_all[0]
    #print(dsim[min_index])
    #min_index = index_all[1] si on veut tester le radiomorphing pour des 
    #plans perp en prenant plan test et plan simulé différent
    '''
    NantDes = len(AntennasAll)
    
        
    return IndexAll, DsimAll, PathAll, NantDes #sim[min_index], dsim[min_index]

def select_pathOld(path, dplane, selected_azimuth):

    # Gives the path to the reference simulation

    sim = glob.glob(path)
    dplane = np.mean(dplane)
    n = len(sim)
    dsim = []

    #azsim_all = np.zeros(n)
    #for i in range(n):
        #azsimRef = float(sim[i].split("_")[-3])
        #if(azsim)
        #azsim_all[i] = (azsimRef - azsimTarget)%180
   # argmin = np.argmin(azsim_all)

    for j in range(n):

        azsimRef = float(sim[j].split("_")[-3])
        if(int(azsimRef) == int(selected_azimuth)):
            dsim.append(float(sim[j].split("_")[-1][:-5]))

    dsim = np.array(dsim)
    index_all = np.argsort(abs(dsim - dplane))

    min_index = index_all[0]
    print(dsim[min_index])    
    #min_index = index_all[1] si on veut tester le radiomorphing pour des
    # plans perp en prenant plan test et plan simulé différent

    return sim[min_index], dsim[min_index]

def select_plane_groundOld(primary, energy, zenith, azimuth, \
                        injection, altitude, fluctuations, cross_check, Xmax, shower):
    
    # We get the closest plane of antenna
    dplane = \
    get_distplane(zenith, azimuth, cross_check[:,0], cross_check[:,1], \
                  cross_check[:,2], Xmax[0], Xmax[1], Xmax[2])
    #dplane = np.mean(dplane)

    # We get the closest reference zenith and and azimuth angle
    target_zenith = select_zenith(zenith)
    
    # We infer the path to the reference shower
    if(Local):
        path = "./../Simulations/SelectedPlane/theta_%.1f/*.hdf5" \
                      %(target_zenith)
    elif(Lyon):
            path = "/sps/trend/chiche/RadiomorphingUptoDate/Simulations/SelectedPlane/theta_%.1f/*.hdf5" \
                      %(target_zenith)
    
    target_azimuth = select_azimuth(path, azimuth)
    
    IndexAll, DsimAll, PathAll, NantDes = select_path(path, dplane, target_azimuth)       
    IndexAll, DsimAll,  = \
    CorrectPlane(IndexAll, DsimAll, PathAll, cross_check, zenith, azimuth, Xmax, NantDes, shower)
    
    print(IndexAll, DsimAll)
    #sys.exit()
    selected_plane, dsim = select_pathOld(path, dplane, target_azimuth)
    Old = True
    if(Old): dplane = np.mean(dplane)
            
    return selected_plane, dplane
        
def select_plane_ground(cross_check, shower):
    

    primary, energy, zenith, azimuth, injection, glevel, fluctuations = shower["primary"], \
    shower["energy"], shower["zenith"], shower["azimuth"], shower["injection"], shower["altitude"],shower["fluctuations"]

    Xmax = getXmaxPosition(primary, energy, fluctuations, azimuth, zenith, glevel, injection)
    #sys.exit(Xmax)

    # We get the closest plane of antenna
    dplane = \
    get_distplane(zenith, azimuth, cross_check[:,0], cross_check[:,1], \
                  cross_check[:,2], Xmax[0], Xmax[1], Xmax[2])
    #dplane = np.mean(dplane)
    
    # We get the closest reference zenith and and azimuth angle
    target_zenith = select_zenith(zenith)
    
    # We infer the path to the reference shower
    if(Local):
        path = "./../Simulations/SelectedPlane/theta_%.1f/*.hdf5" \
                      %(target_zenith)
    elif(Lyon):
            path = "/sps/trend/chiche/RadiomorphingUptoDate/Simulations/SelectedPlane/theta_%.1f/*.hdf5" \
                      %(target_zenith)
    target_azimuth = select_azimuth(path, azimuth)
    
    
    IndexAll, DsimAll, PathAll, NantDes = select_path(path, dplane, target_azimuth)
    #print(DsimAll)      
    IndexAll, DsimAll, PathAll  = \
    CorrectPlane(IndexAll, DsimAll, PathAll, cross_check, zenith, azimuth, Xmax, NantDes, shower)
    
    return IndexAll, DsimAll, PathAll, dplane, Xmax


def get_distplane(zenith, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
     
    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    
    # distance along the x-axis between the antennas postions and Xmax
    x_antenna = x - x_Xmax 
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
     
    # direction of the shower
    uv = np.array([np.sin(zenith)*np.cos(azimuth), \
         np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)])
    # direction of the unit vectors that goes from Xmax to 
    #the position of the antennas
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) 
    distplane = np.dot(np.transpose(u_antenna), uv)
    
    return distplane

def print_plane(RefShower, TargetShower, target_dplane):
    
    print("-----------------------")
    print("Target shower: Energy = %.2f, Zenith = %.2f,  Azimuth = %.2f, \
          Dxmax = %.2d" %(TargetShower.energy, 180 -TargetShower.zenith,\
          TargetShower.azimuth, target_dplane))
    print("")
    print("Ref shower: Energy = %.2f, Zenith = %.2f,  Azimuth = %.2f, \
          Dxmax = %.2d" %(RefShower.energy, 180 -RefShower.zenith,\
          RefShower.azimuth, RefShower.distplane))
    print("-----------------------")
    

# =============================================================================
#                Correct the planes from the antennas selection
# =============================================================================

#The problem here is to find to which plane should be associated to each
# antenna. At this stage of the code, a first association has already been
# performed simply based on the comparions between the distance of the antenna
# to Xmax and the distance of the plane from Xmax. Yet, some antennas that were
# associated to a given plane may in practice rely on the interpolation
# from antennas that are below the ground for this given plane. The function
# below aim to correct this.
    
def CorrectPlane(IndexAll, DsimAll, PathAll, TargetAntennas, \
                 Zenith, Azimuth, xmaxposition, NantDes, shower):
    #print(IndexAll, DsimAll, PathAll)
    #DsimAll =  np.delete(DsimAll, [2])
    #PathAll =  np.delete(PathAll, [2])
    #IndexAll = np.delete(IndexAll, [2])
    #print(DsimAll)
    #print(PathAll)
    #print(IndexAll.shape)
    #print(np.array(DsimAll).shape) 
    #print(np.array(PathAll).shape)
    #print("---------------")
    #print(DsimAll, PathAll)
    #sys.exit()
    IndexAllCorrected = []
    STOP = 0
    ##TODO: add to Lyon
    VoidPlanes = []
    # Loop over each antenna group
    # TODO: remove after tests
    for i in range(len(DsimAll)):
        SelectedPlane = extractData(PathAll[i], False)
        Traces = SelectedPlane.traces
        RefPlane = copy.deepcopy(SelectedPlane) 
        SelectedPlane.pos, SelectedPlane.traces = SelectedPlane.GetinShowerPlane()
        SelectedPlane.primary = shower['primary']
        SelectedPlane.energy = shower['energy']
        SelectedPlane.zenith = shower['zenith']
        SelectedPlane.azimuth = shower['azimuth']
        SelectedPlane.injection = shower['injection']
        SelectedPlane.fluctuations = shower['fluctuations']
        SelectedPlane.filter = shower["filter"]
        Inclination = SelectedPlane.inclination
        SelectedPlane.xmaxpos = xmaxposition 
        Nant = SelectedPlane.InitialShape
        
        SelectedPlane.pos = CerenkovStretch(RefPlane, SelectedPlane, False)[0]  
        SelectedPlane.pos = CerenkovStretch(RefPlane, SelectedPlane, False)[0] 
           
        SelectedPlane.pos, SelectedPlane.traces = SelectedPlane.GetinGeographicFrame()        
        xsim, ysim, zsim = SelectedPlane.pos[:,0], SelectedPlane.pos[:,1], SelectedPlane.pos[:,2]
        position_sims =  np.transpose(np.array([xsim[:Nant], ysim[:Nant], zsim[:Nant]]))
        positions_des = TargetAntennas[IndexAll[i].astype(int)]
        
        
        pos_sims_angles, pos_des_angle, dratio= \
        GetAntennaAnglesSimon(Zenith,Azimuth,xmaxposition,\
        position_sims,positions_des, SelectedPlane.glevel, Inclination)
        #plt.scatter(pos_sims_angles[:,1], pos_sims_angles[:,2])
        #plt.show()
        IndexDplane = []
        #print(DsimAll[i])
        # Loop over each antenna of each group

            
        for j in range(len(IndexAll[i].astype(int))):
            
            SelectedI, SelectedII, SelectedIII, SelectedIV =\
            SelectAntennasForInterpolationCorrected(pos_sims_angles,\
                    pos_des_angle[j], IndexAll[i][j], False, discarded = [])
            
            ConditionI = np.sum(SelectedPlane.traces[:,SelectedI])
            ConditionII = np.sum(SelectedPlane.traces[:,SelectedII])
            ConditionIII = np.sum(SelectedPlane.traces[:,SelectedIII])
            ConditionIV = np.sum(SelectedPlane.traces[:,SelectedIV])
            
            if((ConditionI != 0.0) and (ConditionII != 0.0) and \
               (ConditionIII != 0.0) and (ConditionIV != 0.0)):
                #print(j, IndexAll[i][j], "--", SelectedI, SelectedII, SelectedIII, SelectedIV)
                IndexDplane.append(IndexAll[i][j])
                DisplayResult = False
                if((IndexAll[i][j] == 71) & (DisplayResult)):
                    print("sim")
                    print("index", "phi", "omega", pos_sims_angles[110:130,:])
                    print("des")
                    print("index", "phi", "omega", pos_des_angle[19])
                    plt.scatter(pos_sims_angles[:,1], pos_sims_angles[:,2], s = 0.1)
                    plt.scatter(pos_sims_angles[71,1], pos_sims_angles[71,2], marker = '*')
                    plt.scatter(pos_sims_angles[SelectedI,1], pos_sims_angles[SelectedI,2])
                    plt.scatter(pos_sims_angles[SelectedII,1], pos_sims_angles[SelectedII,2])
                    plt.scatter(pos_sims_angles[SelectedIII,1], pos_sims_angles[SelectedIII,2])
                    plt.scatter(pos_sims_angles[SelectedIV,1], pos_sims_angles[SelectedIV,2])
                    plt.show()
            else:
                #print(i, DsimAll[i])
                IndexAll[i+1] = np.append(IndexAll[i+1],IndexAll[i][j])

        STOP = STOP + len(IndexDplane)
        #print("STOP", STOP)
        ## TODO: add to Lyon
        if(len(IndexDplane)!=0):
            IndexAllCorrected.append(IndexDplane)
        else:
            VoidPlanes.append(i)
            #print("OULALAAAAAAA")
            #print(DsimAll[i])
            #DsimAll =  np.delete(DsimAll, [i])
            #PathAll =  np.delete(PathAll, [i])
        if(STOP==NantDes): break
    #print(IndexAllCorrected, DsimAll)
    ##TODO: add to lyon
    VoidPlanes = np.array(VoidPlanes)
    for k in range(len(VoidPlanes)):
        DsimAll =  np.delete(DsimAll, [k])
        PathAll =  np.delete(PathAll, [k])
        
    TestOnePlane =  False
    if(TestOnePlane):
        print("Enter Plane Index")
        print(DsimAll)
        PlaneIndex = int(input())
        DsimAll = DsimAll[PlaneIndex]
        PathAll = PathAll[PlaneIndex]
    else:
        DsimAll = DsimAll[:len(IndexAllCorrected)]
        PathAll = PathAll[:len(IndexAllCorrected)]

    #sys.exit()
    #print(IndexAllCorrected)
    return IndexAllCorrected, DsimAll, PathAll
    