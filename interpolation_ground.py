'''Script to perform an interpolation between to electric field traces at a desired position
TODO: use magnetic field values and shower core from config-file
'''
import numpy as np
from scipy import signal
from scipy import spatial
import logging
import os
from os.path import split
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from os.path import split, join, realpath
import astropy.units as u
from zhairesppath import GetZHSEffectiveactionIndex, GetEffectiveactionIndex
#ZHAIRESPYTHON=os.environ["ZHAIRESPYTHON"]
#sys.path.append(ZHAIRESPYTHON)
import hdf5fileinout as hdf5io

Local = True
Lyon = not(Local)


def ComputeAntennaPhi(AntennaX,AntennaY, ReferenceDirection=np.array([1,0])):
# WARNING: BE SURE THAT ANTENNA POSITIONS AND ReferenceDirection are with respect to the core of the shower
#this functions computes the angle (in degrees) in the ground plane between the antena position and a reference direction (that could be another antena position
#The optional parameter Cerenkov is used for rescaling the resulting angle to units of the Cerenkov Angle (must be given in degrees))
#The optional parameter Sign asigns (arbitrarily) a sign to the angle depending if it is to the left or to the right (with respect to the x axis) (usefull for plotting
  v1=ReferenceDirection[0:2]
  v2=np.array([AntennaX,AntennaY])
  #Returns the angles in radians between vectors 'v1' and 'v2'
  #v2=v2/np.linalg.norm(v2)
  #print(v1,v2,AntennaX,AntennaY)
  w1 = np.arctan2(v1[1], v1[0])
  w2 = np.arctan2(v2[1], v2[0])
  Angle=((w2-w1)*180/np.pi)
  return Angle


#======================================
def unwrap(phi, ontrue=None):
    """Unwrap the phase so that the absolute difference
      between 2 consecutive phases remains below Pi

    Parameters:
    ----------
        phi: numpy array, float
            phase of the signal trace
        ontrue: str
            printing option, default=None

    Returns:
    ----------
        phi_unwrapped: numpy array, float
            unwarpped phase of the signal trace

    Adapted by E. Hivon (2020-02) from A. Zilles' unwrap
    """
    eps = np.finfo(np.pi).resolution
    thr = np.pi - eps
    pi2 = 2. * np.pi
    phi_unwrapped = np.zeros(phi.shape)
    p0  = phi_unwrapped[0] = phi[0]
    l   = 0
    for i0, p1 in enumerate(phi[1:]):
        i = i0 + 1
        dp = p1 - p0
        if (np.abs(dp) > thr):
            dl = np.floor_divide(abs(dp), pi2) + 1
            if (dp > 0):
                l -= dl
            else:
                l += dl
        phi_unwrapped[i] = p1 + l * pi2
        p0 = p1
        if ontrue is not None:
            print(i, phi[i],           phi[i-1],           abs(phi[i] - phi[i-1]),
                  l, phi_unwrapped[i], phi_unwrapped[i-1], abs(phi_unwrapped[i] - phi_unwrapped[i-1]))

    return phi_unwrapped
#======================================

#original by anne
def unwrap_anne(phi, ontrue=None):
    """Unwrap the phase to a strictly decreasing function.
    Parameters:
    ----------
        phi: numpy array, float
            phase of the signal trace
        ontrue: str
            printing option, default=None
    Returns:
    ----------
        phi_unwrapped: numpy array, float
            unwarpped phase of the signal trace
    """

    phi_unwrapped = np.zeros(phi.shape)
    p0 = phi_unwrapped[0] = phi[0]
    pi2 = 2. * np.pi
    l = 0
    for i0, p1 in enumerate(phi[1:]):
        i = i0 + 1
        if p1 >= p0:
            l += np.floor_divide(p1 - p0, pi2) + 1
        phi_unwrapped[i] = p1 - l * pi2
        p0 = p1
        if ontrue is not None:
            print(i, phi[i], phi[i-1], l, phi_unwrapped[i], abs(phi[i] - phi[i-1]),
                  abs(phi[i] - phi[i-1] + np.pi), abs(phi[i] - phi[i-1] - np.pi), l)
    return phi_unwrapped

def MatiasPhaseInterploation(phi1,w1,phi2,w2):
  phi=np.zeros(phi1.shape)
  for i, p1 in enumerate(phi1[1:]):
    if(np.abs(phi1[i]-phi2[i])>np.pi):
      if(phi1[i]>phi2[i]):
        #print("1 ")
        phi[i]=w1 * (phi1[i]-2*np.pi) + w2 * phi2[i] +2*np.pi
      else:
        #print("2")
        phi[i]=w1 *phi1[i] + w2 * (phi2[i]-2*np.pi) + 2*np.pi
    else:
      #print("3")
      phi[i]=w1*phi1[i] + w2*phi2[i]
    #print(i,phi[i],phi1[i],phi2[i],w1,w2)
  return phi


def interpolate_trace2(interpolate_mode,t1, trace1, x1, t2, trace2, x2, xdes, upsampling=None,  zeroadding=None):
    """Interpolation of signal traces at the specific position in the frequency domain

    The interpolation of traces needs as input antenna position 1 and 2, their traces (filtered or not)
    in one component, their time, and the desired antenna position and returns the trace ( in x,y,z coordinate system) and the time from the desired antenna position.
    Zeroadding and upsampling of the signal are optional functions.

    IMPORTANT NOTE:
    The interpolation of the phases includes the interpolation of the signal arrival time. A linear interpolation implies a plane radio
    emission wave front, which is a simplification as it is hyperbolic in shape. However, the wave front can be estimated as a plane between two simulated observer positions
    for a sufficiently dense grid of observers, as then parts of the wave front are linear on small scales.

    This script bases on the diploma thesis of Ewa Holt (KIT, 2013) in the context of AERA/AUGER. It is based on the interpolation of the amplitude and the pahse in the frequency domain.
    This can lead to misidentifiying of the correct phase. We are working on the interplementaion on a more robust interpolation of the signal time.
    Feel free to include it if you have some time to work on it. The script is completely modular so that single parts can be substitute easily.


    Parameters:
    ----------
            t1: numpy array, float
                time in ns of antenna 1
            trace1: numpy array, float
                single component of the electric field's amplitude of antenna 1
            x1: numpy array, float
                position of antenna 1 in angular coordinates
            t2: numpy array, float
                time in ns of antenna 2
            trace2: numpy array, float
                single component of the electric field's amplitude of antenna 2
            x2: numpy array, float
                position of antenna 2 in angular coordinates
            xdes: numpy arry, float
                antenna position for which trace is desired, in (phi,alpha) in degrees

    Returns:
    ----------
        xnew: numpy array, float
            rough estimation of the time for signal at desired antenna position in ns
        tracedes: numpy array, float
            interpolated electric field component at desired antenna position
    """
    DISPLAY = False
    ERIC=False      #computes the trace with the old unwrapping method

    # calculating weights:
    # if lines are at constant alpha, we weight on phi. If not, we weight on alpha
    #print("new criteria:",np.absolute(x1[2]-x2[2]),x1[2],x2[2])
    #if(np.absolute(x1[2]-x2[2])< 0.05) :
    #  tmp1 = np.absolute(x1[1]-xdes[1])
    #  if(tmp1>180):
    #   tmp1=360-tmp1
    #  tmp2 = np.absolute(x2[1]-xdes[1])
    #  if(tmp2>180):
    #   tmp2=360-tmp2
    #  #print("weight in phi",tmp1,tmp2,x1[1],x2[1],xdes[1])
    #else:
    #  tmp1 = np.absolute(x1[2]-xdes[2])
    #  tmp2 = np.absolute(x2[2]-xdes[2])
    #  #print("weight in alpha",tmp1,tmp2,x1[2])

    #weights, using the geometrical distance
    if(interpolate_mode=="phi"):
      #tmp1 = np.linalg.norm(x1 - xdes)
      #tmp2 = np.linalg.norm(x2 - xdes)
      tmp1 = np.absolute(x1[1]-xdes[1])
      tmp2 = np.absolute(x2[1]-xdes[1])
    elif(interpolate_mode=="alpha"): #using angular distance in alpha
      tmp1 = np.absolute(x1[2]-xdes[2])
      tmp2 = np.absolute(x2[2]-xdes[2])
    else:
       print("unknown interpolation mode")
       weight1=0.5
       weight2=0.5
       

    tmp = 1. / (tmp1 + tmp2)
    weight1 = tmp2 * tmp
    weight2 = tmp1 * tmp

    if np.isinf(weight1):
        print("weight = inf")
        print(x1, x2, xdes)
        weight1 = 1.
        weight2 = 0.
    if np.isnan(weight1):
        print('Attention: projected positions equivalent')
        weight1 = 1.
        weight2 = 0.

    epsilon = np.finfo(float).eps
    if (weight1 > 1. + epsilon) or (weight2 > 1 + epsilon):
        print("weight larger than 1: ", weight1, weight2, x1, x2, xdes, np.linalg.norm(
            x2-x1), np.linalg.norm(x2-xdes), np.linalg.norm(xdes-x1))
    if weight1 + weight2 > 1 + epsilon:
        print("PulseShape_Interpolation.py: order in simulated positions. Check whether ring or ray structure formed first")
        print(weight1, weight2, weight1 + weight2)


    #################################################################################
    # Fourier Transforms  # t in ns, Ex in muV/m, Ey, Ez
    #first antenna
    f = trace1
    xnew = t1

    fsample = 1./((xnew[1]-xnew[0]))  # GHz

    freq = np.fft.rfftfreq(len(xnew), 1./fsample)
    FFT_Ey = np.fft.rfft(f)

    Amp = np.abs(FFT_Ey)
    phi = np.angle(FFT_Ey)

    #############################

    # second antenna
    f2 = trace2
    xnew2 = t2

    fsample2 = 1./((xnew2[1]-xnew2[0]))  # GHz

    freq2 = np.fft.rfftfreq(len(xnew2), 1./fsample2)
    FFT_Ey = np.fft.rfft(f2)

    Amp2 = np.abs(FFT_Ey)
    phi2 = np.angle(FFT_Ey)

    ### Get the pulse sahpe at the desired antenna position

    # get the phase
    phides=MatiasPhaseInterploation(phi,weight1,phi2,weight2)

    if DISPLAY:
        phides2 = phides.copy()

    #check the first frequency.
    eps = np.finfo(float).resolution
    if(phides[0]>np.pi-eps):
      #print(str(eps)+"1 adjusting the first phase "+str(phides[0])+" " + str(phides[0]-np.pi))
      phides[0]=np.pi-eps

    if(phides[0]<-np.pi+eps):
      #print(str(eps)+"2 adjusting the first phase "+str(phides[0])+" " + str(phides[0]+np.pi))
      phides[0]=-np.pi+eps

    #Eric re-unwrap: get -pi to +pi range back and check whether phidesis in between (im not wraping any more, but if by some numerical reason phase is out of range, this fixes it)
    phides = np.mod(phides + np.pi, 2. * np.pi) - np.pi

    if(ERIC):
      phi_unwrapped_eric = unwrap(phi, ontrue)
      phi2_unwrapped_eric = unwrap(phi2, ontrue)
      phides_eric = weight1 * phi_unwrapped_eric + weight2 * phi2_unwrapped_eric
      phides_eric= np.mod(phides_eric + np.pi, 2. * np.pi) - np.pi


    #################################################################################
    ### linear interpolation of the amplitude
    #Amp, Amp2

    Ampdes = weight1 * Amp + weight2 * Amp2
    if(ERIC):
      Ampdes_eric=Ampdes.copy() #we make a copy becouse this is modified next

    if DISPLAY:
        Ampdes2 = Ampdes.copy()

    # inverse FFT for the signal at the desired position
    Ampdes = Ampdes.astype(np.complex64)
    phides = phides.astype(np.complex64)
    Ampdes *= np.exp(1j * phides)
    # trace
    tracedes = (np.fft.irfft(Ampdes))
    tracedes = tracedes.astype(float)

    if(ERIC):
      Ampdes_eric=Ampdes_eric.astype(np.complex64)
      phides_eric=phides_eric.astype(np.complex64)
      Ampdes_eric*= np.exp(1j * phides_eric)
      tracedes_eric=(np.fft.irfft(Ampdes_eric))
      tracedes_eric=tracedes_eric.astype(float)

    #this is a crude interpolation of the time, just to output something (nothing makes more sense at this point)
    tdes=(xnew*weight1+xnew2*weight2)

    if(len(tdes)>len(tracedes) and ERIC==False):
     #tdes=tdes[0:-1] #and this is required becouse the inverse fft returns one less time bin
     #print("interpolate_trace: lenghts are different",len(tdes),len(tracedes))
     tracedes.resize(len(tdes))
     #print("interpolate_trace: tried to fix it",len(tdes),len(tracedes))

    if(ERIC==True):
      if(len(tdes)>len(tracedes_eric)):
         #tdes=tdes[0:-1] #and this is required becouse the inverse fft returns one less time bin
         tracedes_eric.resize(len(tdes))

    # PLOTTING

    if (DISPLAY):
        import matplotlib.pyplot as plt
        plt.rc('font', family='serif', size=12)

        width=7.2
        height=2*width/1.618
        flow=21
        fhigh=121

        ########PHASES
        fig1 = plt.figure(1,figsize=(width,height), facecolor='w', edgecolor='k',dpi=120)
        ax1=fig1.add_subplot(412)
        tmp=ax1.plot(freq*1000, phi,linestyle='-', color='tab:green', label="a",linewidth=2)
        tmp=ax1.plot(freq2*1000, phi2,linestyle='-', color='tab:red', label="b",linewidth=2)
        if(ERIC):
          tmp=ax1.plot(freq2*1000, phides_eric, linestyle='--',color="tab:orange", label="a + b (unwraping)")
        tmp=ax1.plot(freq2*1000, phides, 'k:', label="a + b (this work)")
        #ax1.set_xlabel(r"Frequency (MHz)")
        tmp=ax1.set_ylabel(r"$\theta_{j}[rad]$")
        tmp=ax1.set_xlim(flow,fhigh)

        ######## AMPLITUDES#########################################################################
        ax1=fig1.add_subplot(411)
        tmp=ax1.plot(freq*1000, Amp/1000, linestyle='-', color='tab:green', label="a",linewidth=2)
        tmp=ax1.plot(freq2*1000, Amp2/1000, linestyle='-', color='tab:red', label="b",linewidth=2)
        tmp=ax1.plot(freq2*1000, Ampdes2/1000, linestyle='--',color="tab:orange", label="a + b (unwraping)")
        tmp=ax1.plot(freq2*1000, Ampdes2/1000, 'k:',label="a + b (this work)")
        #tmp=ax1.set_xlabel(r"Frequency (MHz)")
        tmp=ax1.set_ylabel(r"$|S_{j}(k)|[mV/GHz]$")
        tmp=ax1.legend()
        tmp=ax1.set_xlim(flow,fhigh)
        fig1.set_tight_layout(True)

        #UNWRAPPED PHASES #############################################################################
        phi2_unwrapped = unwrap(phi2)
        phi_unwrapped = unwrap(phi)
        phides2_unwrapped= unwrap(phides2)

        ax1 = fig1.add_subplot(413)
        if(ERIC):
          ax1.plot(freq*1000, phi_unwrapped_eric, color='tab:green', label="a",linewidth=2)
          ax1.plot(freq2*1000, phi2_unwrapped_eric, color='tab:red', label="b",linewidth=2)
          ax1.plot(freq2*1000,  weight1 * phi_unwrapped_eric + weight2 * phi2_unwrapped_eric,linestyle='--',color="tab:orange", label="a + b (unwraping)" )
        ax1.plot(freq2*1000,  phides2_unwrapped,'k:', label="a + b (this work)" )
        ax1.set_xlabel(r"Frequency (MHz)")#, fontsize=16)
        ax1.set_ylabel(r"$unwrap. \theta_{j}[rad]$")#, fontsize=16)
        tmp=ax1.set_xlim(flow,fhigh)
        tmp=ax1.set_ylim(-190,-30)

        #SIGNAL #############################################################################################################
        #fig2 = plt.figure(2, dpi=120, facecolor='w', edgecolor='k')
        ax1=fig1.add_subplot(414)
        #ax1.plot(np.real(t1), np.real(trace1), 'g:', label= "antenna 1")
        #ax1.plot(np.real(t2), np.real(trace2), 'b:', label= "antenna 2")
        #ax1.plot(np.real(tdes), np.real(tracedes), 'r-', label= "Synthetized")
        #ax1.plot(np.real(tdes), np.real(tracedes_eric), 'k-', label= "Synthetized with unwraping")

        ax1.plot(np.arange(0,trace1.size), np.real(trace1), linestyle='-', color='tab:green', label="a",linewidth=2)
        ax1.plot(np.arange(0,trace2.size), np.real(trace2), linestyle='-', color='tab:red', label="b",linewidth=2)
        ax1.plot(np.arange(0,tracedes.size), np.real(tracedes), 'k:', label="a + b (this work)")
        if(ERIC):
          ax1.plot(np.arange(0,tracedes_eric.size), np.real(tracedes_eric), linestyle='--',color="tab:orange", label="a + b (unwraping)")

        tmp=ax1.set_ylim(-151,149)
        tmp=ax1.set_xlim(280,1180)
        ax1.set_xlabel(r"Sample")
        ax1.set_ylabel(r"$S_y [\mu V]$")
        #ax1.legend(loc='best')

        plt.show()

    if(ERIC):
        return tdes,tracedes_eric

    else:
        return tdes, tracedes

####################################################################################################################


def get_w(zenith, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
 
    #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
    
    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    
    x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
    u_antenna /= np.linalg.norm(u_antenna, axis =0)
    w = np.arccos(np.dot(np.transpose(u_antenna), uv))
    w = w*180.0/pi # we calculte w in degrees
    
    return w


def get_distplane(zenith, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
 
    #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
    
    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    
    x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
    distplane = np.dot(np.transpose(u_antenna), uv)
    
    
    return distplane

def get_center(distplane, x_Xmax, y_Xmax, z_Xmax, GroundLevel, zenith, azimuth):

    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower

    distground = np.sqrt(x_Xmax**2 + y_Xmax**2 + (z_Xmax-GroundLevel)**2)

    dist_plane_ground = distground - distplane
    core = uv*(dist_plane_ground)
    core[2] = core[2] - GroundLevel
    
    return core

def get_in_shower_plane(x,y,z, core, zenith, inclination,azimuth):
    
    # function that returns the trcaes in the shower plane (v, vxb, vxvxb) from the traces in the geographic plane (x, y, z)
    
    x = x + core[0]
    y = y + core[1] # We move the core position in (0,0,0) before changing the reference frame
    z = z + core[2]
    n = len(x) # number of antennas
    
    # antennas positions in the  shower reference frame (v, vxB, vxvxB)
    v = np.zeros(n)   
    vxb = np.zeros(n)
    vxvxb = np.zeros(n)
    
    # we reper the direction of the shower
    pi = np.pi
    zenith = zenith*pi/180.0 # elevation = zenith - 90°
    inclination = inclination*pi/180.0 # inclination of the magnetic field
    azimuth = azimuth*pi/180.0 # azimuth of the shower
    
    # unit vectors 
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower
    uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field
    
    
    uv_x_uB = np.cross(uv, uB) # unit vector along the vxb direction
    uv_x_uB /= np.linalg.norm(uv_x_uB) # normalisation
    
    uv_x_uvxB  = np.cross(uv, uv_x_uB) # unit vector along the vxvxb direction
    uv_x_uvxB /= np.linalg.norm(uv_x_uB) # normalisation
    
    P = np.transpose(np.array([uv, uv_x_uB, uv_x_uvxB])) # matrix to go from the shower reference frame to the geographic reference frame
    
    P_inv = np.linalg.inv(P) # matrix to go from the geographic reference frame to the shower reference frame
    
    # We calculate the positions in the shower plane
    Position_geo = np.array([x,y,z]) # position in the geographic reference frame
    Position_shower = np.dot(P_inv, Position_geo) # position in the shower reference frame
    
    # We deduce the different components
    v = Position_shower[0, :] 
    vxb = Position_shower[1, :]
    vxvxb =  Position_shower[2, :]
    
    return (v,vxb, vxvxb,)

#def get_phi(vxb, vxvxb):
    
#    phi =np.arctan2(vxvxb,vxb)*180/np.pi
#    
#    phi_sims = abs(phi[0:160])
#    phi_des = abs(phi[160:])
#    
#    phi_sims = phi_sims.astype(int) +1
#    phi_des = phi_des.astype(int)
#
#    return phi_sims, phi_des

###############################################################################

def GetAntennaAnglesSimon(Zenith,Azimuth, xmax_position,positions_sims,positions_des, GroundLevel, Inclination):

# =============================================================================
#                                 loading data
# =============================================================================
    
    x_sims, y_sims, z_sims = positions_sims[:,0], positions_sims[:,1], positions_sims[:,2]
    x_des, y_des, z_des = positions_des[:,0], positions_des[:,1], positions_des[:,2]
    xXmax, yXmax, zXmax = xmax_position[0], xmax_position[1], xmax_position[2]
    
# =============================================================================
#                                 alpha angle
# =============================================================================
   
    w_sims_angles = get_w(Zenith, Azimuth, x_sims, y_sims, z_sims, xXmax, yXmax, zXmax)
    w_des_angles = get_w(Zenith, Azimuth, x_des, y_des, z_des, xXmax, yXmax, zXmax)
    
# =============================================================================
#                               distance ratio
# =============================================================================
    
    distplane = np.mean(get_distplane(Zenith, Azimuth, x_sims, y_sims, z_sims, xXmax, yXmax, zXmax))
    dist_desired = get_distplane(Zenith, Azimuth, x_des, y_des, z_des, xXmax, yXmax, zXmax)
    distanceratio = [distplane/dist_desired]

# =============================================================================
#                                Phi angle
# =============================================================================
    
    center_plane = get_center(distplane, xXmax, yXmax, zXmax, GroundLevel, Zenith, Azimuth)
    x_all, y_all, z_all = np.concatenate((x_sims, x_des)), np.concatenate((y_sims, y_des)),\
    np.concatenate((z_sims, z_des))
    (v, vxb, vxvxb) = get_in_shower_plane(x_all, y_all, z_all, center_plane, Zenith, Inclination, Azimuth)
    #phi_sims, phi_des = get_phi(x_all +center_plane[0], y_all+ center_plane[1])
    Nref = len(x_sims) ## TODO: check
    Ndes = len(x_des)
    phi_sims = np.zeros(Nref)
    phi_des = np.zeros(Ndes)
    for i in np.arange(0,len(positions_sims[:,1])): phi_sims[i]=ComputeAntennaPhi(vxb[i],vxvxb[i])
    for i in np.arange(0,len(positions_des[:,1])): phi_des[i]=ComputeAntennaPhi(vxb[Nref+i],vxvxb[Nref+i])

    #print(phi_sims)
    pos_sims_angles = np.transpose([np.zeros(len(phi_sims)), phi_sims, w_sims_angles])
    pos_des_angles = np.transpose([np.zeros(len(phi_des)), phi_des, w_des_angles])
    
    #print(np.shape(pos_sims_angles))
    #print(np.shape(pos_des_angles))

    
    return pos_sims_angles, pos_des_angles, distanceratio #phi_sims, phi_des 

#gets as input the positions of the simulated antennas in (antid, phi, alpha) and the position of the desired antena in phi,alpha)
#returns 4 list of antenna positions, separated in quadrants in (phi,alpha) arround the desired antenna.
def SelectAntennasForInterpolation(pos_sims_angles,pos_des_angle, i, discarded):

    index_I=[]
    index_II=[]
    index_III=[]
    index_IV=[]
    

    for m in np.arange(0,len(pos_sims_angles)): # desired points as reference
        delta_phi = pos_des_angle[1] - pos_sims_angles[m][1]
        if delta_phi > 180:
            delta_phi = delta_phi -360
        elif delta_phi < -180:
            delta_phi = delta_phi + 360

        delta_alpha = pos_sims_angles[m][2]-pos_des_angle[2] 

        distance=np.sqrt(delta_alpha*delta_alpha+delta_phi*delta_phi) #distance in the Delta alpha, delthaphiplande. Since alpha in covered in much more granularity,

        if delta_phi >= 0. and  delta_alpha >= 0:
            index_I.append((m,delta_phi,delta_alpha,distance))
        if delta_phi >= 0. and  delta_alpha <= 0:
            index_II.append((m,delta_phi,delta_alpha,distance))
        if delta_phi <= 0. and  delta_alpha <= 0:
            index_III.append((m,delta_phi,delta_alpha,distance))
        if delta_phi <= 0. and  delta_alpha >= 0:
            index_IV.append((m,delta_phi,delta_alpha,distance))

    if(False):
        fig74 = plt.figure(4, facecolor='w', edgecolor='k')
        ax1=fig74.add_subplot(111)
        ax1.scatter([item[1] for item in index_I],[item[2] for item in index_I],label="I")
        ax1.scatter([item[1] for item in index_II],[item[2] for item in index_II],label="II")
        ax1.scatter([item[1] for item in index_III],[item[2] for item in index_III],label="III")
        ax1.scatter([item[1] for item in index_IV],[item[2] for item in index_IV],label="IV")
        ax1.legend(loc=2)

    bailoutI=bailoutII=bailoutIII=bailoutIV=0

    if not index_I:
        #print("list - Quadrant 1 - empty")
        bailoutI=1
    if not index_II:
        #print("list - Quadrant 2 - empty")
        bailoutII=1
        index_II=index_I
    if not index_III:
        #print("list - Quadrant 3 - empty")
        bailoutIII=1
        index_III=index_IV
    if not index_IV:
        #print("list - Quadrant 4 - empty")
        bailoutIV=1

    if(bailoutI==1 or bailoutIV==1 or (bailoutII==1 and bailoutIII==0) or (bailoutII==0 and bailoutIII==1)):
      #print("Point is outside of the starshape, discarding", )
      discarded.append(i)
      return -1,-1,-1,-1

    if(bailoutII==1 and bailoutIII==1 and bailoutIV==0 and bailoutI==0):
        #print(" I cannot find antennas with smaller alpha, lets try using only the inner antennas")
        index_II=index_I
        index_III=index_IV

    #now i convert this to numpy arrays to be able to sort them
    index_I=np.array(index_I, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])
    index_II=np.array(index_II, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])
    index_III=np.array(index_III, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])
    index_IV=np.array(index_IV, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])
    #print(index_I[0], "INDEXXI")
    #print(index_IV[0], "INDEXXIV")

    index_I = np.sort(index_I, order=['distance','delta_alpha', 'delta_phi'])
    index_II = np.sort(index_II, order=['distance','delta_alpha', 'delta_phi'])
    index_III = np.sort(index_III, order=['distance','delta_alpha','delta_phi'])
    index_IV = np.sort(index_IV, order=['distance','delta_alpha', 'delta_phi'])

    
    #now, index_I and index_IV are the ones with alpha bigger than thed desired. I want the closest in angle.
    #index_II and index_III are the ones with smaller alphas (so differences will be negative). I want the closest in angle

    #print(index_I[0], "INDEXXI")
    #print(index_IV[0], "INDEXXIV")
    #print("Selected I ",index_I[0][0],"Selected II ", index_II[0][0],"Selected III ", index_III[0][0],"Selected IV ",index_IV[0][0])
    Selected_I=index_I[0][0]
    Selected_II=index_II[0][0]
    Selected_III=index_III[0][0]
    if(bailoutII==1 and bailoutIII==1 and bailoutIV==0 and bailoutI==0): #TODO: check this
        Selected_II=index_II[1][0]
        Selected_III=index_III[1][0]
    Selected_IV=index_IV[0][0]
    '''
    plt.scatter(pos_sims_angles[:,1], pos_sims_angles[:,2], s = 0.1)
    plt.scatter(pos_sims_angles[71,1], pos_sims_angles[71,2], marker = '*')
    plt.scatter(pos_sims_angles[Selected_I,1], pos_sims_angles[Selected_I,2])
    plt.scatter(pos_sims_angles[Selected_II,1], pos_sims_angles[Selected_II,2])
    plt.scatter(pos_sims_angles[Selected_III,1], pos_sims_angles[Selected_III,2])
    plt.scatter(pos_sims_angles[Selected_IV,1], pos_sims_angles[Selected_IV,2])
    plt.show()
    '''
    
    return Selected_I,Selected_II,Selected_III,Selected_IV


def SelectAntennasForInterpolationCorrected(pos_sims_angles,pos_des_angle, i, Display, discarded):
    
# =============================================================================
#                        Initialisation
# =============================================================================
      
    IndexAll = np.arange(0, len(pos_sims_angles))
    DeltaOmegaAll = pos_sims_angles[:,2]-pos_des_angle[2] 
    DeltaPhiAll =  pos_des_angle[1] - pos_sims_angles[:,1]
    DeltaPhiAll = np.where(DeltaPhiAll < 180, DeltaPhiAll, DeltaPhiAll - 360)
    DeltaPhiAll = np.where(DeltaPhiAll > -180, DeltaPhiAll, DeltaPhiAll + 360)
        
    # Ref antenna with the closest omega below the desired one
    NegArg = np.where(DeltaOmegaAll <0, DeltaOmegaAll, -np.inf).argmax()
    # Ref antenna with the closest omega above the desired one
    PosArg = np.where(DeltaOmegaAll >0, DeltaOmegaAll, np.inf).argmin()

# =============================================================================
#                     Antennas inside the starshape
# =============================================================================
    
    # antenna in the inner circle of the starshape
    if(np.where(DeltaOmegaAll <0, DeltaOmegaAll, -np.inf)[NegArg] == -np.inf):
        if(Display): print(i, "Antennas inside the inner starshape circle")
        OmegaNeg, OmegaPos = \
        pos_sims_angles[NegArg,2], pos_sims_angles[PosArg,2]
        
        # Circles of antennas associated with the two selected ones
        ArgNegCircle = np.argsort(abs(OmegaNeg - pos_sims_angles[:,2]))[:8]
        ArgPosCircle = np.argsort(abs(OmegaPos - pos_sims_angles[:,2]))[:8]
        
        ConditionNeg = np.isin(IndexAll, ArgNegCircle)
                        
        ConditionPos = np.isin(IndexAll, ArgPosCircle)
                        
        ConditionPhiPos = DeltaPhiAll>= 0
        ConditionPhiNeg = DeltaPhiAll<= 0
            
        IndexAll_I = IndexAll[ConditionPos & ConditionPhiPos]
        IndexAll_II = IndexAll[ConditionNeg & ConditionPhiPos]
        IndexAll_III = IndexAll[ConditionNeg & ConditionPhiNeg]
        IndexAll_IV =  IndexAll[ConditionPos & ConditionPhiNeg]
        
        ArgI = np.argmin(abs(DeltaPhiAll[IndexAll_I]))
        ArgII = np.argmin(abs(DeltaPhiAll[IndexAll_II]))
        ArgIII = np.argmin(abs(DeltaPhiAll[IndexAll_III]))
        ArgIV = np.argmin(abs(DeltaPhiAll[IndexAll_IV]))
        
        SelectedI, SelectedII, SelectedIII, SelectedIV = \
        IndexAll_I[ArgI], IndexAll_II[ArgII], \
        IndexAll_III[ArgIII], IndexAll_IV[ArgIV]
        
        return SelectedI, SelectedII, SelectedIII, SelectedIV

# =============================================================================
#                     Antennas outside the starshape
# =============================================================================
        
    elif(np.where(DeltaOmegaAll >0, DeltaOmegaAll, np.inf)[PosArg] == +np.inf):
        if(Display): print(i, "Antenna outside of the starshape: can not be interpolated")
        SelectedI, SelectedII, SelectedIII, SelectedIV = -1, -1, -1, -1
        discarded.append(i)
        return SelectedI, SelectedII, SelectedIII, SelectedIV

# =============================================================================
#                  Antennas within the starshape
# =============================================================================
        
    else:
        # Closest Omega value below and above the desired one
        OmegaNeg, OmegaPos = \
        pos_sims_angles[NegArg,2], pos_sims_angles[PosArg,2]
       
        # Circles of antennas associated with the two selected ones
        ArgNegCircle = np.argsort(abs(OmegaNeg - pos_sims_angles[:,2]))[:8]
        ArgPosCircle = np.argsort(abs(OmegaPos - pos_sims_angles[:,2]))[:8]
      
        ConditionNeg = np.isin(IndexAll, ArgNegCircle)                      
        ConditionPos = np.isin(IndexAll, ArgPosCircle)
                             
# =============================================================================
#               Case when the two selected circles are identical
# =============================================================================
        
        if((ConditionNeg == ConditionPos).all()):
            if(abs(DeltaOmegaAll[NegArg])< abs(DeltaOmegaAll[PosArg])):
                PosArg = PosArg + 8 
                if(PosArg>=160): PosArg = 159
                OmegaPos = pos_sims_angles[PosArg,2]
                ArgPosCircle = np.argsort(\
                                abs(OmegaPos - pos_sims_angles[:,2]))[:8]
                ConditionPos = np.isin(IndexAll, ArgPosCircle)                 
        
            else:
                NegArg = NegArg - 8 
                if(NegArg<0): NegArg = 0
                OmegaNeg = pos_sims_angles[NegArg,2]
                ArgNegCircle = np.argsort(\
                                abs(OmegaNeg - pos_sims_angles[:,2]))[:8]
                ConditionNeg = np.isin(IndexAll, ArgNegCircle)
                      
        ConditionPhiPos = DeltaPhiAll>= 0
        ConditionPhiNeg = DeltaPhiAll<= 0
        
        IndexAll_I = IndexAll[ConditionPos & ConditionPhiPos]
        IndexAll_II = IndexAll[ConditionNeg & ConditionPhiPos]
        IndexAll_III = IndexAll[ConditionNeg & ConditionPhiNeg]
        IndexAll_IV =  IndexAll[ConditionPos & ConditionPhiNeg]
        
        ArgI = np.argmin(abs(DeltaPhiAll[IndexAll_I]))
        ArgII = np.argmin(abs(DeltaPhiAll[IndexAll_II]))
        ArgIII = np.argmin(abs(DeltaPhiAll[IndexAll_III]))
        ArgIV = np.argmin(abs(DeltaPhiAll[IndexAll_IV]))
        
        SelectedI, SelectedII, SelectedIII, SelectedIV = \
        IndexAll_I[ArgI], IndexAll_II[ArgII], \
        IndexAll_III[ArgIII], IndexAll_IV[ArgIV]

        return SelectedI, SelectedII, SelectedIII, SelectedIV
    
def PerformInterpolation(EfieldTraces, Selected_I, Selected_II, Selected_III, Selected_IV, distanceratio, pos_sims_angles, pos_des_angle, DesiredT0, tracetype, VoltageTraces, FilteredVoltageTraces, DISPLAY=False, PLOTPAPER=False,i=0):
#InputFilename, where the starshape sim is
#CurrentSignalInfo  (so that i dont have to get it from the file each time the interpolation is done, as usually this will be used inside a loop over the desired antennas.
#CurrentAntennaInfo (so that i dont have to get it from the file each time the interpolation is done, as usually this will be used inside a loop over the desired antennas.
#CurrentEventName (so that i dont have to access it each time the interpolation is done, as usually this will be used inside a loop over the desired antennas.
#Selected_I to IV (the position in de antena index, for the antennas to use in the interpolation. Basically the output of SelectAntennasForInterpolation.
#distanceratio is the distance ratio for the topography correction (distance to xmax from the actual position used on the starshape after the projecton/distance to xmax for the desired antenna)
#pos_sims_angles holds the angular coordinates of each antenna in the format [IndexNr,alpha,phi] (in degrees!)
#pos_des_angle holds the angular coordinate of the desired point, in the format [IndexNr,alpha,phi] (in degrees!)
#DesiredT0 is the arrival time expected for the desired position, taken from the spherical shower front approximation. (the same used for the antennas in the starshape)


    #This was doing the interpolation in angular units for both angles, as it would logically be done. However, this happens to work worse at the cherenkov angle, for some reason
    # probably becouse in very inclined showers theta is very deformed, and and weighting in theta does not represent the actual distance to the antenna, that might be more important
    #(although, by doing it in distance you mess with the polarization)
    
    # TODO: check this part with Matias
    
    tmin = 0#np.min(Time)
    tmax = 2300#np.max(Time)
    tbinsize = 0.5#Time[0]
        
    #tbinsize=hdf5io.GetTimeBinSize(CurrentSignalSimInfo)
    #tmin=hdf5io.GetTimeWindowMin(CurrentSignalSimInfo)
    #tmax=hdf5io.GetTimeWindowMax(CurrentSignalSimInfo)
   # print(tmin, tmin2, tmax, tmax2, tbinsize, tbinsize2)

    ## the interpolation of the pulse shape is performed, in x, y and z component
    
    #print(Selected_I,type(Selected_I), "!!!!!!!")
    
    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_I)
    if(tracetype=='efield'):
      #txt0=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt0 = EfieldTraces[Selected_I] # trace of one of the selected antenna for the interpolation
    elif(tracetype=='voltage'):
      #txt0=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt0 = VoltageTraces[Selected_I]
    elif(tracetype=='filteredvoltage'):
      #txt0=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt0 = FilteredVoltageTraces[Selected_I]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0
  


    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_IV)
    if(tracetype=='efield'):
      #txt1=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt1 = EfieldTraces[Selected_IV]
    elif(tracetype=='voltage'):
      #txt1=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt1 = VoltageTraces[Selected_IV]
    elif(tracetype=='filteredvoltage'):
      #txt1=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt1 = FilteredVoltageTraces[Selected_I]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0

    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_II)
    if(tracetype=='efield'):
      #txt2=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt2 = EfieldTraces[Selected_II]
    elif(tracetype=='voltage'):
     # txt2=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt2 = VoltageTraces[Selected_II]
    elif(tracetype=='filteredvoltage'):
      #txt2=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt2 = FilteredVoltageTraces[Selected_I]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0


    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_III)
    if(tracetype=='efield'):
      #txt3=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt3 = EfieldTraces[Selected_III]
    elif(tracetype=='voltage'):
      #txt3=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt3 = VoltageTraces[Selected_III]
    elif(tracetype=='filteredvoltage'):
      #txt3=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt3 = FilteredVoltageTraces[Selected_III]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0



    #points I and IV have a higher alpha. In case they are different, lets take the average and the same theta of the desired point as the pointa
    meanalpha=(pos_sims_angles[Selected_I][2]+pos_sims_angles[Selected_IV][2])/2.0
    #and theta is already available, is the theta of the desired position
    point_a=np.array([0,pos_des_angle[1],meanalpha])
    #points II and III have a lower alpha. In case they are different, lets take the average and the same theta of the desired point as the pointb
    meanalpha=(pos_sims_angles[Selected_II][2]+pos_sims_angles[Selected_III][2])/2.0
    #and theta is already available
    point_b=np.array([0,pos_des_angle[1],meanalpha])

    point_I=np.array([0,pos_sims_angles[Selected_I][1],pos_sims_angles[Selected_I][2]])
    point_II=np.array([0,pos_sims_angles[Selected_II][1],pos_sims_angles[Selected_II][2]])
    point_III=np.array([0,pos_sims_angles[Selected_III][1],pos_sims_angles[Selected_III][2]])
    point_IV=np.array([0,pos_sims_angles[Selected_IV][1],pos_sims_angles[Selected_IV][2]])


    #print("1st Interpol x")
    tnew1, tracedes1x = interpolate_trace2("phi",txt0.T[0], txt0.T[1], point_I , txt1.T[0], txt1.T[1], point_IV, point_a ,upsampling=None, zeroadding=None)
    #print("1st Interpol y")
    tnew1, tracedes1y = interpolate_trace2("phi",txt0.T[0], txt0.T[2], point_I , txt1.T[0], txt1.T[2], point_IV, point_a ,upsampling=None, zeroadding=None)
    #print("1st Interpol z")
    tnew1, tracedes1z = interpolate_trace2("phi",txt0.T[0], txt0.T[3], point_I , txt1.T[0], txt1.T[3], point_IV, point_a ,upsampling=None, zeroadding=None)

    #print("2nd Interpol x")
    tnew2, tracedes2x = interpolate_trace2("phi",txt2.T[0], txt2.T[1], point_II , txt3.T[0], txt3.T[1], point_III, point_b ,upsampling=None, zeroadding=None)
    #print("2nd Interpol y")
    tnew2, tracedes2y = interpolate_trace2("phi",txt2.T[0], txt2.T[2], point_II , txt3.T[0], txt3.T[2], point_III, point_b ,upsampling=None, zeroadding=None)
    #print("2nd Interpol z")
    tnew2, tracedes2z = interpolate_trace2("phi",txt2.T[0], txt2.T[3], point_II , txt3.T[0], txt3.T[3], point_III, point_b ,upsampling=None, zeroadding=None)

    ###### Get the pulse shape of the desired position from points on a and b
    #print("Interpol x")
    tnew_desiredx, tracedes_desiredx =interpolate_trace2("alpha",tnew1, tracedes1x, point_a, tnew2, tracedes2x, point_b, pos_des_angle, zeroadding=None)
    #print("Interpol y")
    tnew_desiredy, tracedes_desiredy =interpolate_trace2("alpha",tnew1, tracedes1y, point_a, tnew2, tracedes2y, point_b, pos_des_angle, zeroadding=None)
    #print("Interpol z")
    tnew_desiredz, tracedes_desiredz =interpolate_trace2("alpha",tnew1, tracedes1z, point_a, tnew2, tracedes2z, point_b, pos_des_angle, zeroadding=None)


    #aplyinnfg distance ratio
    #print("distanceratio",distanceratio)
    tracedes_desiredx=tracedes_desiredx*distanceratio
    tracedes_desiredy=tracedes_desiredy*distanceratio
    tracedes_desiredz=tracedes_desiredz*distanceratio

    #print("langth traces:", len(txt0.T[0]), len(txt0.T[1]),len(txt1.T[0]),len(txt1.T[1]),len(txt2.T[0]), len(txt2.T[1]),len(txt3.T[0]),len(txt3.T[1]))
    #print("langth interpolated:", len(xnew1), len(tracedes1x),len(xnew2),len(tracedes2x),len(xnew_desiredx), len(tracedes_desiredx))

    #now, lets use timing solution
    ntbins=np.shape(txt0)[0]
    tnew_desiredx=np.linspace(DesiredT0+tmin+tbinsize,DesiredT0+tmin+ntbins*tbinsize,ntbins)
    tnew_desiredy=tnew_desiredx
    tnew_desiredz=tnew_desiredx
    

    if(round((tnew_desiredx[2]-tnew_desiredx[1]),5)!=round(tbinsize,5)):
     print("warning! different tbin sizes",tbinsize,tnew_desiredx[2]-tnew_desiredx[1])
     print(tmin,tmax,tbinsize,ntbins)

    if (len(tracedes_desiredx)!=len(tracedes_desiredy)!=len(tracedes_desiredz)!=len(tnew_desiredx)!=ntbins) :
       print("WARNING: Traces are differnt lenght!",len(tracedes_desiredx),len(tracedes_desiredy),len(tracedes_desiredz),ntbins)


    desired_trace=np.column_stack((tnew_desiredx,tracedes_desiredx,tracedes_desiredy,tracedes_desiredz))


    return desired_trace

def ComputeTimeAntennas(TargetShower):
    
    Nant = TargetShower.InitialShape
    xmaxdist = TargetShower.xmaxdist
    xmaxposition = TargetShower.xmaxpos
    c = 299792458.0
    time2core = (xmaxdist/c)*1e9 # time from xmax to the core for n = 1 
    #if(realXmax): time2core = ((XmaxDistance/(299792458.0))*1e9)
    desired = TargetShower.pos[Nant:,:] # TODO: replace with input positions # interpolated positions in meters [Number_antennas,3]
    positions = TargetShower.pos
    xpoints, ypoints , zpoints  = positions[:,0][Nant:], positions[:,1][Nant:], positions[:,2][Nant:]
    time_all = TargetShower.desired_time # to modify
    t0all= np.zeros(len(desired))
    k =0
    
    for i in range(len(desired)):
        
        nant = i
        nref = GetZHSEffectiveactionIndex(xmaxposition[0],xmaxposition[1],xmaxposition[2],\
        xant=xpoints[nant],yant=ypoints[nant],zant=zpoints[nant],ns=325,kr=-0.1218,stepsize = 20000)
      
        dant = np.sqrt((xmaxposition[0]-xpoints[nant])**2+ (xmaxposition[1]-ypoints[nant])**2 +\
        (xmaxposition[2]-zpoints[nant])**2)
            
        t0 = (dant*nref/(299792458.0))*1e9 - time2core 
        t0all[k] = t0
        k = k +1
    '''
        tmin_all = []
        tmax_all = []


    #for i in range(len(desired)):
        
        #tmin_all.append(abs(t0all[i])- abs(min(time_all[:,i])))
        #tmax_all.append(abs(t0all[i]) - abs(max(time_all[:,i])))
    '''
    
    desiredtime = t0all # TODO: compute that
    '''
    tmin = #np.mean(tmin_all)
    tmax = #np.mean(tmax_all)
    tbin = #time_all[1,0] - time_all[0,0]
    Time = [tbin, tmin, tmax]
    '''
    
    #print(tmin, tmax, tbin)

     
    return desiredtime
    

def do_interpolation_hdf5(TargetShower, IndexAntennas, VoltageTraces, FilteredVoltageTraces, antennamin=0, antennamax=159, DISPLAY=False, usetrace='efield', FillOutliersWithZeros=True):
    '''
    Reads in arrays, looks for neighbours, calls the interpolation and saves the traces

    Parameters:
    ----------
    desired: str
        numpy array of desired antenna positions (x,y,z,t0 info) (in grand coordinates, so , above sea level)
    InputFilename: str
        path to HDF5 simulation file
        The script accepts starshape as well as grid arrays
    antennamin,antennamax:int
        the program is designed to run on the first 160 antennas. If your simulation has more, you can specify a range to be used...but it has to be tested
    EventNumber: int
        number of event in the file to use. you can process only one at a time
    DISPLAY: True/False
        enables printouts and plots
    usetrace: str (note that for now you can only do one at a time, and on different output files)
        efield
        voltage
        filteredvoltage
    overdistance: For compatibilitiy and historical reasons, you can add an additional distance behind xmax for the start of the cone.
    FillOutliersWithZeros: If an antenna is outside of the starshape, you set it to 0. If false, the antenna is skipped (and there is no output to the file)

    Returns:
    ----------
        --
    Saves traces via index infomation in same folder as desired antenna positions


    NOTE: The selection of the neigbours is sufficiently stable, but does not always pick the "best" neigbour, still looking for an idea
    '''

    DISPLAY=0
    #0 dont plot
    #3 plot plot the interpolated traces and the errors in the interpolation (if in DEVELOPMENT)
    #4 plot starshape  (if in DEVELOPMENT)
    #5 plot quadrant selection in alpha,phi


    PLOTPAPER=False #shows only antenna 4 and fixes the axis to accomodate for the plot on the JINST paper

    if(usetrace=="all"):
      print("usetrace is all, looping over all trace types")
      usetracelist=["efield","voltage","filteredvoltage"]
    else:
      usetracelist=[str(usetrace)]

    #Getting Required Information from the InputEvent
    
# =============================================================================
#                       Data loading
# =============================================================================    
    
    # Number of ref ant for the interpolation: 160 by default
    Nant = TargetShower.InitialShape 
    Energy = TargetShower.energy # EeV
    Zenith = TargetShower.zenith # GRANDconventions
    Azimuth = TargetShower.azimuth # GRAND conventions
    Inclination = TargetShower.inclination # degrees
    GroundAltitude = TargetShower.glevel # meters
    xmaxposition = TargetShower.xmaxpos
    desired = TargetShower.pos[Nant:,:]
    PositionsPlane = TargetShower.pos[:Nant,:]
    desiredtime  = ComputeTimeAntennas(TargetShower)
    NantTraces = TargetShower.NantTraces


    # field used for the interpolation
    EfieldTraces = []
    for i in range(Nant):
      Efield = np.array((TargetShower.traces[:,i], TargetShower.traces[:,NantTraces+i], \
      TargetShower.traces[:,2*NantTraces+i], TargetShower.traces[:,3*NantTraces+i])).T 
      EfieldTraces.append(Efield)
         
# =============================================================================
#                       Matias code
# =============================================================================
       
    antennamax=antennamax+1
    positions_sims = PositionsPlane
    #positions_sims = TargetShower.pos

    # Hand over a list file including the antenna positions you would like to have. This could be improved by including an ID.
    positions_des = desired
    
    #DesiredAntennaInfoMeta=hdf5io.CreatAntennaInfoMeta(split(InputFilename)[1],CurrentEventName,AntennaModel="Interpolated") #TODO: discuss that!  
    DesiredIds=np.arange(0, len(positions_des)) #this could be taken from the input file of desired antennas
    #print(DesiredIds)
    DesiredAntx=deepcopy(positions_des.T[0])
    DesiredAnty=deepcopy(positions_des.T[1])
    DesiredAntz=deepcopy(positions_des.T[2]) #this deepcopy bullshit is becouse position_des is later modified by the rotation, and transposition apparently creates a shallow copy (a reference)
    DesiredSlopeA=np.zeros(len(positions_des))
    DesiredSlopeB=np.zeros(len(positions_des))
    #DesiredT0=deepcopy(positions_des.T[3])
    DesiredT0=deepcopy(desiredtime.T)


    positions_des=desired
    pos_sims_angles, pos_des_angles, distanceratio = GetAntennaAnglesSimon(Zenith,Azimuth,xmaxposition,positions_sims,positions_des, GroundAltitude, Inclination)
    remove_antenna=[]
    desired_traceAll = []
    discarded = []
    for i in np.arange(0,len(pos_des_angles)):
        
        if(i%10==0): print("Computing... %d/%d antennas" %(i, len(pos_des_angles)))
        
        #select the four antennas for the inerpolation for this desired antenna
        Selected_I,Selected_II,Selected_III,Selected_IV = SelectAntennasForInterpolationCorrected(pos_sims_angles,pos_des_angles[i], i, True, discarded)
        #print(i, "--", Selected_I,Selected_II,Selected_III,Selected_IV)
        Skip=False
        for tracetype in usetracelist:
            #print("computing for "+tracetype+" on desired antenna "+str(i))
            #If there was a problem selecting the antennas, dont interpolate

            Skip=False
            if(Selected_I==-1 or Selected_II==-1 or Selected_III == -1 or Selected_IV==-1):
                if(FillOutliersWithZeros==True):
                    #print("antenna outside of the starshape, interpolation not performed, filled with 0",i)
                    ntbins = len(EfieldTraces[0][:,0])
                    desired_trace=np.zeros((ntbins,4))
                else:
                    #print("antenna outside of the starshape, interpolation not performed, antenna removed",i)
                    
                    if(Skip==False): #to do it only once
                       Skip=True
                       remove_antenna.append(i)
                       
            else:
                #Do the interpolation
                desired_trace=PerformInterpolation(EfieldTraces, Selected_I, Selected_II, Selected_III, Selected_IV, distanceratio[0][i], pos_sims_angles, pos_des_angles[i],DesiredT0[i], tracetype,VoltageTraces, FilteredVoltageTraces, DISPLAY=DISPLAY, PLOTPAPER=PLOTPAPER, i=i)
                desired_traceAll.append(desired_trace)

        del Selected_I, Selected_II, Selected_III, Selected_IV
        #end of the loop on the desired positions

    #now, lets remove the skiped antennas from the index
    DesiredIds=np.delete(DesiredIds,remove_antenna)
    DesiredAntx=np.delete(DesiredAntx,remove_antenna)
    DesiredAnty=np.delete(DesiredAnty,remove_antenna)
    DesiredAntz=np.delete(DesiredAntz,remove_antenna)
    DesiredSlopeA=np.delete(DesiredSlopeA,remove_antenna)
    DesiredSlopeB=np.delete(DesiredSlopeB,remove_antenna)
    DesiredT0=np.delete(DesiredT0,remove_antenna)
    
    k =0
    efield_interpolated = []
    w_interpolated = []
    discarded.append(-1)
    No_data = np.zeros(len(desired_traceAll))*np.nan 
    
    counter = 1
    for i in range(len(desired)):
        #print(pos_des_angles[i][2])
        if(i != discarded[k]):
            if(Local):
                np.savetxt("./OutputDirectory/DesiredTraces_%d.txt" %IndexAntennas[i], desired_traceAll[i -k])
            elif(Lyon):
                np.savetxt("/sps/trend/chiche/RadiomorphingUptoDate/OutputDirectory/DesiredTraces_%d.txt" %IndexAntennas[i], desired_traceAll[i -k])
            efield_interpolated.append(desired_traceAll[i -k])
            w_interpolated.append(pos_des_angles[i][2])

        else:
            k = k +1
            efield_interpolated.append(No_data)
            #efield_interpolated.append(desired_traceAll[i -k])
            w_interpolated.append(np.nan)
            #w_interpolated.append(pos_des_angles[i][2])
    
    '''
    
    plt.plot(x_interpolated, color ='purple')
    plt.plot(xant4interpolation1, color = 'green')
    plt.plot(xant4interpolation2, color = 'red')
    plt.xlim(400,500)
    plt.legend(["interpolated", "Selected I", "Selected II"])
    plt.show()
    
    plt.plot(y_interpolated, color ='purple')
    plt.plot(yant4interpolation1, color ='green')
    plt.plot(yant4interpolation2, color ='red')
    plt.xlim(400,500)
    plt.legend(["interpolated", "Selected I", "Selected II"])
    plt.show()
    
    plt.plot(z_interpolated, color ='purple')
    plt.plot(zant4interpolation1, color ='green')
    plt.plot(zant4interpolation2, color ='red')
    plt.xlim(400,500)
    plt.legend(["interpolated", "Selected I", "Selected II"])
    plt.show()
    
    print(np.shape(efield_interpolated), " do interpolation")
    
    x_sim, y_sim, z_sim = TargetShower.pos[0:160,0], TargetShower.pos[0:160,1], TargetShower.pos[0:160,2]
    x_des, y_des, z_des = TargetShower.pos[160:,0], TargetShower.pos[160:,1], TargetShower.pos[160:,2]
    
    plt.scatter(x_des,y_des)
    plt.scatter(x_des[8],y_des[8], color = 'purple')
    plt.scatter(x_sim[0],y_sim[0], color ='green')
    plt.scatter(x_sim[16],y_sim[16], color ='red')
    plt.legend(["Array", "interpolated", "Selected I", "Selected II"])
    #plt.xlim(-200, 200)
    #plt.ylim(-200, 200)
    plt.show()

    
    sys.exit()
    '''
    #print(w_interpolated)
    return efield_interpolated, w_interpolated