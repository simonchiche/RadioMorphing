#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:45:27 2022

@author: chiche
"""

import h5py
import numpy as np
import Modules.hdf5fileinout as hdf5io
import sys

class Shower:
    
    def __init__(self, primary, energy, zenith, azimuth, injection_height, \
                 nantennas, inclination, GroundAltitude, Positions, \
                 Traces, XmaxDistance, XmaxPosition, fluctuations):
    
        self.primary = primary
        self.energy = energy
        self.zenith = zenith
        self.azimuth = azimuth
        self.injection = injection_height
        self.nant = nantennas
        self.inclination = inclination
        self.pos = Positions
        self.traces = Traces
        self.desired_time = np.zeros(nantennas) 
        self.glevel = GroundAltitude
        self.xmaxdist = XmaxDistance
        self.xmaxpos = XmaxPosition
        self.fluctuations = fluctuations
        self.distplane = self.get_distplane()
        # hard coded from the reference library
        self.InitialShape = 176 # TODO: check
        self.NantTraces =  nantennas 
        self.filter = False
        

# =============================================================================
#                     GetInShowerPlane Functions
# =============================================================================
        
    def showerdirection(self):
        
        azimuth, zenith = self.azimuth, self.zenith
        zenith = zenith*np.pi/180.0
        azimuth = azimuth*np.pi/180.0
        
        uv = np.array([np.sin(zenith)*np.cos(azimuth), \
                       np.sin(zenith)*np.sin(azimuth), np.cos(zenith)])
        
        return uv
        
        
    def get_distplane(self):
 
    #function that returns "w" at each antenna, i.e. the angle between the 
    #direction that goes from Xmax to the core and the direction that 
    #goes from Xmax to a given antenna
    
        pos = self.pos
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        xmaxpos = self.xmaxpos
        x_Xmax, y_Xmax, z_Xmax = xmaxpos[0], xmaxpos[1], xmaxpos[2]
            
        x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
        y_antenna = y - y_Xmax
        z_antenna = z - z_Xmax
    
        uv = self.showerdirection()
        u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
        distplane = np.dot(np.transpose(u_antenna), uv)
        #print(distplane)
        #print(u_antenna)
        return np.mean(distplane)

    def get_center(self, distplane = 0):

        xmaxpos = self.xmaxpos
        x_Xmax, y_Xmax, z_Xmax = xmaxpos[0], xmaxpos[1], xmaxpos[2]
        GroundLevel = self.glevel
    
        uv = self.showerdirection()

        distground = np.sqrt(x_Xmax**2 + y_Xmax**2 + (z_Xmax-GroundLevel)**2)
        
        distplane = self.distplane

        dist_plane_ground = distground - distplane
        core = -uv*(dist_plane_ground)
        core[2] = core[2] + GroundLevel
            
        return core    
    
    def GetinShowerPlane(self):
        
        # function that returns the trcaes in the shower plane (v, vxb, vxvxb) 
        #from the traces in the geographic plane (x, y, z)

        inclination = self.inclination*np.pi/180.0
        
        pos =  self.pos
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        n = len(x) # number of antennas
        
        # We move the core position in (0,0,0) before changing the 
        #reference frame
        
        core = self.get_center()
        
        x = x - core[0]
        y = y - core[1] 
        z = z - core[2]
        
        Traces = self.traces
        time_sample = len(Traces[:,0])
        
        # antennas positions in the  shower reference frame (v, vxB, vxvxB)
        v = np.zeros(n)   
        vxb = np.zeros(n)
        vxvxb = np.zeros(n)
        
        #Traces in the shower reference frame
        Traces_Ev = np.zeros([time_sample,n])
        Traces_Evxb = np.zeros([time_sample,n])
        Traces_Evxvxb = np.zeros([time_sample,n])
        Time = np.zeros([time_sample, n])
        
        # unit vectors 
        uv = self.showerdirection()
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
        
        # We calulate the traces in the shower plane
        Traces_geo = np.zeros([time_sample,3])
        Traces_shower_temp = np.zeros([3, time_sample])
        
        for i in range(n):
            
            Traces_geo = np.array([Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i + 3*n]])
            
            Traces_shower_temp = np.dot(P_inv, Traces_geo)
            
            Traces_Ev[:,i] = np.transpose(Traces_shower_temp[0,:]) # Ev component of the traces
            Traces_Evxb[:,i] = np.transpose(Traces_shower_temp[1,:]) # Evxb component of the traces
            Traces_Evxvxb[:,i] = np.transpose(Traces_shower_temp[2,:]) # Evxvxb component of the traces
            
            Time[:,i] = Traces[:,i]
        
        # We derive the traces in the shower plane
        
        Traces_sp = np.transpose(np.concatenate((np.transpose(Time), \
        np.transpose(Traces_Ev), np.transpose(Traces_Evxb), np.transpose(Traces_Evxvxb))))
    
        Positions_sp = np.zeros([n,3]) 
        Positions_sp[:,0], Positions_sp[:,1], Positions_sp[:,2] = v, vxb, vxvxb
    
        return Positions_sp, Traces_sp
    
# =============================================================================
#                        shower angles
# =============================================================================
      
    def get_alpha(self):
    
    # function that returns the angle between the direction of the shower and 
    #the direction of the magnetic field 
    
        inclination = self.inclination
        inclination = inclination*np.pi/180.0
        
        # unit vectors    
        uv = self.showerdirection()
        uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field
        cos_alpha = np.dot(uv,uB)
        alpha = np.arccos(cos_alpha) # angle between the direction of the shower and the direction of the magnetic field
        
        return alpha
    
    def get_w(self):
 
        #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
        
        inclination = self.inclination
        xmaxpos = self.xmaxpos
        pos = self.pos
        x_Xmax, y_Xmax, z_Xmax =  xmaxpos[0], xmaxpos[1], xmaxpos[2]
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        
        inclination = inclination*np.pi/180.0
        
        x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
        y_antenna = y - y_Xmax
        z_antenna = z - z_Xmax
        
        uv = self.showerdirection()
        u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
        u_antenna /= np.linalg.norm(u_antenna, axis =0)
        w = np.arccos(np.dot(np.transpose(u_antenna), uv))
        w = w*180.0/np.pi # we calculte w in degrees
    
        return w  
    
# =============================================================================
#                    Get parametrized Xmax position
# =============================================================================
        
    def _getAirDensity(self, _height, model):

        '''Returns the air density at a specific height, using either an 
        isothermal model or the Linsley atmoshperic model as in ZHAireS
    
        Parameters:
        ---------
            h: float
                height in meters
    
        Returns:
        -------
            rho: float
                air density in g/cm3
        '''
    
        if model == "isothermal":
                #Using isothermal Model
                rho_0 = 1.225*0.001    #kg/m^3
                M = 0.028966    #kg/mol
                g = 9.81        #m.s^-2
                T = 288.        #
                R = 8.32        #J/K/mol , J=kg m2/s2
                rho = rho_0*np.exp(-g*M*_height/(R*T))  # kg/m3
    
        elif model == "linsley":
            #Using Linsey's Model
            bl = np.array([1222., 1144., 1305.5948, 540.1778,1])*10  # g/cm2  ==> kg/cm3
            cl = np.array([9941.8638, 8781.5355, 6361.4304, 7721.7016, 1e7])  #m
            hl = np.array([4,10,40,100,113])*1e3  #m
            if (_height>=hl[-1]):  # no more air
                rho = 0
            else:
                hlinf = np.array([0, 4,10,40,100])*1e3  #m
                ind = np.logical_and([_height>=hlinf],[_height<hl])[0]
                rho = bl[ind]/cl[ind]*np.exp(-_height/cl[ind])
                rho = rho[0]*0.001
        else:
            print("#### Error in GetDensity: model can only be isothermal or linsley.")
            return 0
    
        return rho
        
    def getSphericalXmaxHeight(self):
    
        XmaxPosition = self.xmaxpos
        Rearth = 6370949 
        XmaxHeight = np.sqrt((Rearth + XmaxPosition[2])**2 + XmaxPosition[0]**2 +\
                             XmaxPosition[1]**2) - Rearth
    
        return XmaxHeight
    
    def Xmax_param(self):

        #input energy in EeV
        
        primary = self.primary
        energy= self.energy
        fluctuations = self.fluctuations
        
        if(primary == 'Iron'):
            a =65.2
            c =270.6
            
            Xmax = a*np.log10(energy*1e6) + c
            
            if(fluctuations):
                a = 20.9
                b = 3.67
                c = 0.21
                
                sigma_xmax = a + b/energy**c
                Xmax = np.random.normal(Xmax, sigma_xmax)
            
            return Xmax
        
        elif(primary == 'Proton'):
            a = 57.4
            c = 421.9
            Xmax = a*np.log10(energy*1e6) + c
            
            if(fluctuations):
                a = 66.5
                b = 2.84
                c = 0.48
                
                sigma_xmax = a + b/energy**c
                Xmax = np.random.normal(Xmax, sigma_xmax)
            
            return Xmax
        
        else:
            print("missing primary")  
            
    
    
    def _get_CRzenith(self):
        ''' Corrects the zenith angle for CR respecting Earth curvature, zenith seen by observer
            ---fix for CR (zenith computed @ shower core position
        
        Arguments:
        ----------
        zen: float
            GRAND zenith in deg
        injh: float
            injection height wrt to sealevel in m
        GdAlt: float
            ground altitude of array/observer in m (should be substituted)
        
        Returns:
        --------
        zen_inj: float
            GRAND zenith computed at shower core position in deg
            
        Note: To be included in other functions   
        '''
    
        #Note: To be included in other functions
        zen = self.zenith

        GdAlt = self.glevel
        injh = self.injection
                
        Re= 6370949 # m, Earth radius
    
        a = np.sqrt((Re + injh)**2. - (Re+GdAlt)**2 *np.sin(np.pi-np.deg2rad(zen))**2) - (Re+GdAlt)*np.cos(np.pi-np.deg2rad(zen))
        zen_inj = np.rad2deg(np.pi-np.arccos((a**2 +(Re+injh)**2 -Re**2)/(2*a*(Re+injh))))
        
        
        return zen_inj    

            
    def _dist_decay_Xmax(self): 
        ''' Calculate the height of Xmax and the distance injection point to Xmax along the shower axis
        
        Arguments:
        ----------
        zen: float
            GRAND zenith in deg, for CR shower use _get_CRzenith()
        injh2: float
            injectionheight above sealevel in m
        Xmax_primary: float
            Xmax in g/cm2 
            
        Returns:
        --------
        h: float
            vertical Xmax_height in m
        ai: float
            Xmax_distance injection to Xmax along shower axis in m
        '''
        
        zen = self._get_CRzenith()
        injh2 = self.injection
        Xmax_primary = self.Xmax_param() 
        zen2 = np.deg2rad(zen)
        
        hD=injh2
        step=10 #m
        if hD>10000:
            step=100 #m
        Xmax_primary= Xmax_primary#* 10. # g/cm2 to kg/m2: 1g/cm2 = 10kg/m2
        gamma=np.pi-zen2 # counterpart of where it goes to
        Re= 6370949 # m, Earth radius
        X=0.
        i=0.
        h=hD
        ai=0
        while X< Xmax_primary:
            i=i+1
            ai=i*step #100. #m
            hi= -Re+np.sqrt(Re**2. + ai**2. + hD**2. + 2.*Re*hD - 2*ai*np.cos(gamma) *(Re+hD))## cos(gamma)= + to - at 90dg
            deltah= abs(h-hi) #(h_i-1 - hi)= delta h
            h=hi # new height
            rho = self._getAirDensity(hi, "linsley")
            X=X+ rho * step*100. #(deltah*100) *abs(1./np.cos(np.pi-zen2)) # Xmax in g/cm2, slanted = Xmax, vertical/ cos(theta); density in g/cm3, h: m->100cm, np.pi-zen2 since it is defined as where the showers comes from, abs(cosine) so correct for minus values
           
        return h, ai # Xmax_height in m, Xmax_distance in m    
    
    def getGroundXmaxDistance(self):
        
        # zenith in cosmic ray convention here
        
        zenith = (180 -self.zenith)*np.pi/180
        GroundAltitude = self.glevel
        XmaxHeight, DistDecayXmax = self._dist_decay_Xmax()
    
        Rearth = 6370949 
        
        dist = np.sqrt((Rearth+ XmaxHeight)**2 - ((Rearth + GroundAltitude)*np.sin(zenith))**2) \
        - (Rearth + GroundAltitude)*np.cos(zenith)
                
        return dist   
    
    def getXmaxPosition(self):
        
        uv = self.showerdirection()
        showerDistance = self.getGroundXmaxDistance()
        XmaxPosition = -uv*showerDistance 
        XmaxPosition[2] = XmaxPosition[2] + self.glevel  
                
        return XmaxPosition
    
# =============================================================================
#                 Cerenkov angle computation
# =============================================================================

    def GetZHSEffectiveactionIndex(self ,ns=325,kr=-0.1218,stepsize = 20000):
         
        
          XmaxPosition = self.xmaxpos
          core = self.get_center()
          xant, yant, zant = core[0], core[1], core[2]
          x0, y0, z0 = XmaxPosition[0], XmaxPosition[1], XmaxPosition[2]
          #rearth=6371007.0 #new aires
          rearth=6370949.0 #19.4.0
    #     Variable n integral calculation ///////////////////
          R02=x0*x0+y0*y0  #!notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada (entro con injz-zXmax -> injz-z0=zXmax
          h0=(np.sqrt( (z0+rearth)*(z0+rearth) + R02 ) - rearth)/1E3    #!altitude of emission
    
          rh0=ns*np.exp(kr*h0) #!refractivity at emission (this
          n_h0=1+1E-6*rh0 #!n at emission
    #        write(*,*) "n_h0",n_h0,ns,kr,x0,y0,injz-z0,h0,rh0
          modr=np.sqrt(R02)
    
          if(modr > 1000): #! if inclined shower and point more than 20km from core. Using the core as reference distance is dangerous, its invalid in upgoing showers
    
    #         Vector from average point of track to observer.
              ux = xant-x0
              uy = yant-y0
              uz = zant-z0
    
    #         divided in nint pieces shorter than 10km
              nint=int((modr/stepsize)+1)
              kx=ux/nint
              ky=uy/nint       #k is vector from one point to the next
              kz=uz/nint
    #
              currpx=x0
              currpy=y0        #current point (1st is emission point)
              currpz=z0
              currh=h0
    #
              sum=0
              for iii in range(0,nint):
                nextpx=currpx+kx
                nextpy=currpy+ky #!this is the "next" point
                nextpz=currpz+kz
                nextR2=nextpx*nextpx + nextpy*nextpy
                nexth=(np.sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1E3
    #c
                if(np.abs(nexth-currh) > 1E-10  ):
                  sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
                else:
                  sum=sum+np.exp(kr*currh)
    #            endif
    #c
                currpx=nextpx
                currpy=nextpy
                currpz=nextpz  #!Set new "current" point
                currh=nexth
    #c
              avn=ns*sum/nint
    #          print*,"avn:",avn
              n_eff=1+1E-6*avn #!average (effective) n
    #c
          else:
    #c         withouth integral
              hd=zant/1E3 #!detector altitude
    #
              if(np.abs(hd-h0) > 1E-10):
                avn=(ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
    #            print*,"avn2:",avn
              else:
                avn=ns*np.exp(kr*h0)
    #           print*,"avn3:",avn
    #            print *,"Effective n: h0=hd"
    #          endif
              n_eff=1+1E-6*avn #!average (effective) n
    #        endif
    #c     ///////////////////////////////////////////////////
          return n_eff
        
    def get_cerenkov_angle(self):
        
        n_refraction = self.GetZHSEffectiveactionIndex()
        cer_ang = np.rad2deg(np.arccos(1/n_refraction))
        
        return cer_ang
    
# =============================================================================
#                   Get in Geographic Frame
# =============================================================================
    
    def GetinGeographicFrame(self):
        
        # function that returns the trcaes in the shower plane (v, vxb, vxvxb) 
        #from the traces in the geographic plane (x, y, z)
    
        inclination = self.inclination*np.pi/180.0
        
        pos =  self.pos
        v, vxb, vxvxb = pos[:,0], pos[:,1], pos[:,2]
        n = len(vxb) # number of antennas
        
        # We move the core position in (0,0,0) before changing the reference frame
        #core = self.get_center(self.get_distplane)
        
        Traces = self.traces
        time_sample = len(Traces[:,0])
        
        # antennas positions in the  shower reference frame (v, vxB, vxvxB)
        x = np.zeros(n)   
        y = np.zeros(n)
        z = np.zeros(n)
        
        #Traces in the shower reference frame
        Traces_Ex = np.zeros([time_sample,n])
        Traces_Ey = np.zeros([time_sample,n])
        Traces_Ez = np.zeros([time_sample,n])
        Time = np.zeros([time_sample, n])
        
        # unit vectors 
        uv = self.showerdirection()
        uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field 
        
        uv_x_uB = np.cross(uv, uB) # unit vector along the vxb direction
        uv_x_uB /= np.linalg.norm(uv_x_uB) # normalisation
        
        uv_x_uvxB  = np.cross(uv, uv_x_uB) # unit vector along the vxvxb direction
        uv_x_uvxB /= np.linalg.norm(uv_x_uB) # normalisation
        
        P = np.transpose(np.array([uv, uv_x_uB, uv_x_uvxB])) # matrix to go from the shower reference frame to the geographic reference frame
        
        # We calculate the positions in the shower plane
        Position_shower = np.array([v,vxb,vxvxb]) # position in the geographic reference frame
        Position_ground = np.dot(P, Position_shower) # position in the shower reference frame
        
        # We deduce the different components
        x = Position_ground[0, :] 
        y = Position_ground[1, :]
        z =  Position_ground[2, :]
        
        # We calulate the traces in the shower plane
        Traces_shower = np.zeros([time_sample,3])
        Traces_geo_temp = np.zeros([3, time_sample])
        
        for i in range(n):
            
            Traces_shower = np.array([Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i + 3*n]])
            
            Traces_geo_temp = np.dot(P, Traces_shower)
            
            Traces_Ex[:,i] = np.transpose(Traces_geo_temp[0,:]) # Ev component of the traces
            Traces_Ey[:,i] = np.transpose(Traces_geo_temp[1,:]) # Evxb component of the traces
            Traces_Ez[:,i] = np.transpose(Traces_geo_temp[2,:]) # Evxvxb component of the traces
            
            Time[:,i] = Traces[:,i]
        
        # We derive the traces in the shower plane
        
        Traces_ground = np.transpose(np.concatenate((np.transpose(Time), \
        np.transpose(Traces_Ex), np.transpose(Traces_Ey), np.transpose(Traces_Ez))))
        
        core = self.get_center()
        
        x = x + core[0]
        y = y + core[1] 
        z = z + core[2]
    
        Positions_ground = np.zeros([n,3]) # To check
        Positions_ground[:,0], Positions_ground[:,1], Positions_ground[:,2] = x, y, z
        
    
        return Positions_ground, Traces_ground

    
# =============================================================================
#                         Analysis functions
# =============================================================================
    
def CerenkovStretch(RefShower, TargetShower, GroundPlane = True):
    
    if(GroundPlane):
        TargetShower.pos, TargetShower.traces = TargetShower.GetinShowerPlane()
    
    Nant = RefShower.nant
    
    cerangle_ref = RefShower.get_cerenkov_angle()
    cerangle_target = TargetShower.get_cerenkov_angle()
    
    kstretch = cerangle_ref/cerangle_target

    w = RefShower.get_w()/kstretch
    
    v, vxb, vxvxb =  TargetShower.pos[:,0], \
    TargetShower.pos[:,1], TargetShower.pos[:,2]
    eta = np.arctan2(vxvxb, vxb)
    Distplane = TargetShower.distplane  
    d = Distplane*np.tan(w*np.pi/180.0)
    
    vxb_scaled = d*np.cos(eta) 
    vxvxb_scaled = d*np.sin(eta)
    
    scaled_pos = np.array([v,vxb_scaled, vxvxb_scaled]).T
    scaled_traces = TargetShower.traces[:,Nant:]*kstretch
                
    return scaled_pos, scaled_traces, kstretch
    
   
    
def extractData(sim_file, Display = False):
    Local = True
    Lyon = not(Local)
    
    if(Local):
        #print(sim_file, "PATH")
        if(sim_file[0] == "."): 
            simu_path = './' + sim_file
        else: 
            simu_path = sim_file
    elif(Lyon):
        simu_path = sim_file
    if(Display): print(sim_file)
    InputFilename = simu_path
    filehandle = h5py.File(InputFilename, 'r')

    #Shower event access
    RunInfo = hdf5io.GetRunInfo(InputFilename)         
    NumberOfEvents = hdf5io.GetNumberOfEvents(RunInfo)
    EventNumber = NumberOfEvents-1 
    EventName = hdf5io.GetEventName(RunInfo,EventNumber)       
    #All simulations info

    #Shower parameters
    Zenith = hdf5io.GetEventZenith(RunInfo,EventNumber)
    Azimuth = hdf5io.GetEventAzimuth(RunInfo,EventNumber)
    Primary = "electron"
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)
    XmaxDistance = hdf5io.GetEventXmaxDistance(RunInfo,EventNumber)
    #SlantXmax = hdf5io.GetEventSlantXmax(RunInfo,EventNumber) # Slant ?
    Energy = hdf5io.GetEventEnergy(RunInfo,EventNumber)
    #HadronicModel = hdf5io.GetEventHadronicModel(RunInfo,EventNumber)    
    
    #Shower info
    EventInfo = hdf5io.GetEventInfo(InputFilename,EventName)
    XmaxPosition = hdf5io.GetEventXmaxPosition(EventInfo)
    BFieldIncl = hdf5io.GetEventBFieldIncl(EventInfo) #BfieldIncl/Bfieldecl
    #BFieldDecl = hdf5io.GetEventBFieldDecl(EventInfo)
    GroundAltitude = hdf5io.GetGroundAltitude(EventInfo)    
    #Antannas info
    AntennaInfo = hdf5io.GetAntennaInfo(InputFilename,EventName)


    NumberOfAntennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
    IDs_bis = hdf5io.GetAntIDFromAntennaInfo(AntennaInfo)
    X = hdf5io.GetXFromAntennaInfo(AntennaInfo)
    Y = hdf5io.GetYFromAntennaInfo(AntennaInfo)
    Z = hdf5io.GetZFromAntennaInfo(AntennaInfo)
    Positions = np.transpose([X,Y,Z])

    #Traces
    AntennaID = IDs_bis[0]
    Efield_trace = hdf5io.GetAntennaEfield(InputFilename,EventName,AntennaID,OutputFormat="numpy")
    time_sample = len(Efield_trace[:,0])
    
    trace_x = np.zeros([time_sample,NumberOfAntennas])
    trace_y = np.zeros([time_sample,NumberOfAntennas])
    trace_z = np.zeros([time_sample,NumberOfAntennas])
    Time = np.zeros([time_sample,NumberOfAntennas])
    
    for i in range(NumberOfAntennas):
    
        AntennaID = IDs_bis[i]
        path = str(EventName)+"/AntennaTraces/"+str(AntennaID)+"/efield"
        Efield_trace = filehandle[path]
        Time[:,i] = Efield_trace['Time']
        trace_x[:,i] = Efield_trace['Ex']
        trace_y[:,i] = Efield_trace['Ey']
        trace_z[:,i] = Efield_trace['Ez']
               
    Traces = np.transpose(np.concatenate((np.transpose(Time), np.transpose(trace_x), \
                                          np.transpose(trace_y), np.transpose(trace_z))))
              
    Nant = len(X)
    Injection = 1e5 # TODO: get it from the hdf5
    fluctuations = True
        
    
    RefShower = Shower(Primary, Energy, Zenith, Azimuth, Injection, Nant, BFieldIncl, GroundAltitude,
                        Positions, Traces, XmaxDistance, XmaxPosition[0], fluctuations)

    return RefShower  
