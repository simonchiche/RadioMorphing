import numpy as np
import copy
from Modules.ShowerClass import Shower, extractData, CerenkovStretch

def TestOnePlane(IndexAll, DplaneRefAll):

#if(TestOnePlane):
    print("Enter Plane Index")
    PlaneIndex = int(input())
    # condition pour choisir un plan qui faisait déjà partie de la 
    # selection de base
    # Condition: si l'indice du plan est plus petit que la longueur de la liste
    if(PlaneIndex<len(IndexAll)): 
        # les positions sélectionées sont les position déjà pré-sélectionnées, associées à ce plane
        IndexAllCut = IndexAll[PlaneIndex]
        #print(IndexAllCut)
    # condition pour choisir un plan qui ne faisait pas déjà partie de la 
    # selection de base. donc à priori un plan assez loin du sol.
    else:
        # Dans ce cas toutes les antennes sont sélectionnées
        IndexAllCut  = np.arange(0,176,1)
    IndexAll = np.arange(0,176,1)
    DplaneRefAll = [DplaneRefAll] 

    return IndexAllCut


def GetTargetPositionsPerPlane(IndexAll, DplaneRefAll, DplaneTargetAll, desired_positionsAll, PathAll, i):
    """Get the target positions for the current plane"""

    if((len(DplaneRefAll)>=1)): 
        selected_plane = PathAll[i]
        dplane = np.mean(DplaneTargetAll[IndexAll[i]])
        desired_positions = desired_positionsAll[IndexAll[i]]
    else:
        selected_plane = PathAll
        dplane = np.mean(DplaneTargetAll[IndexAll])
        desired_positions = desired_positionsAll[IndexAll]
    
    return selected_plane, dplane, desired_positions

def GenerateTargetShower(selected_plane, shower, XmaxPos):
    RefShower = extractData(selected_plane)
    Nant = RefShower.nant
    TargetShower = copy.deepcopy(RefShower) 
    TargetShower.primary = shower['primary']
    TargetShower.energy = shower['energy']
    TargetShower.zenith = shower['zenith']
    TargetShower.azimuth = shower['azimuth']
    TargetShower.injection = shower['injection']
    TargetShower.fluctuations = shower['fluctuations']
    TargetShower.filter = shower["filter"]
    TargetShower.xmaxpos = XmaxPos

    return Nant, RefShower, TargetShower

def FormatAntennaPositionsforInterpolation(RefShower, TargetShower, desired_positions):
# Desired antennas positions
    x, y, z = \
    desired_positions[:,0], desired_positions[:,1], desired_positions[:,2]
    desired_shape = len(x)
    initial_shape = RefShower.InitialShape
    # Antennas position assigned to the target shower
    target_x, target_y, target_z = \
    np.zeros(initial_shape + desired_shape),np.zeros(initial_shape \
            + desired_shape), np.zeros(initial_shape + desired_shape)
    
    # The first antennas are the reference antennas used for the interpolation
    target_x[0:initial_shape] = TargetShower.pos[0:initial_shape,0]
    target_y[0:initial_shape] = TargetShower.pos[0:initial_shape,1]
    target_z[0:initial_shape] = TargetShower.pos[0:initial_shape,2]
    
    # The second antenna group corresponds to the desired antennas positions
    target_x[initial_shape:] = x
    target_y[initial_shape:] = y
    target_z[initial_shape:] = z
    
    # We affect the new positions to the Target Shower
    NewPos = np.transpose(np.array([target_x, target_y, target_z]))

    return NewPos, initial_shape

def CorrectOmegaAngleBias(RefShower, TargetShower, EfieldAllAntennas, wAllAntennas, CorrectBias = False):
    
    if(CorrectBias):
        if((RefShower.zenith - TargetShower.zenith) >0.5):
            cerangle_target = 0.75*TargetShower.get_cerenkov_angle()
            EfieldAllAntennas[wAllAntennas<cerangle_target] = \
            EfieldAllAntennas[wAllAntennas<cerangle_target]/1.25
        
        if((RefShower.zenith - TargetShower.zenith) <-0.5):
            cerangle_target = 0.75*TargetShower.get_cerenkov_angle()
            EfieldAllAntennas[wAllAntennas<cerangle_target] = \
            EfieldAllAntennas[wAllAntennas<cerangle_target]*1.25
    
    return EfieldAllAntennas, wAllAntennas