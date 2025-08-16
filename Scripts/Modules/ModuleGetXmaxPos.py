import numpy as np

def _get_CRzenith(zenith, glevel, injection):
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
    zen = zenith

    GdAlt = glevel
    injh = injection
            
    Re= 6370949 # m, Earth radius

    a = np.sqrt((Re + injh)**2. - (Re+GdAlt)**2 *np.sin(np.pi-np.deg2rad(zen))**2) - (Re+GdAlt)*np.cos(np.pi-np.deg2rad(zen))
    zen_inj = np.rad2deg(np.pi-np.arccos((a**2 +(Re+injh)**2 -Re**2)/(2*a*(Re+injh))))
    
    
    return zen_inj 

def _getAirDensity(_height, model):

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

def Xmax_param(primary, energy, fluctuations):

    
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
        

def _dist_decay_Xmax(primary, energy, fluctuations, zenith, glevel, injection): 
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
    
    zen = _get_CRzenith(zenith, glevel, injection)
    injh2 = injection
    Xmax_primary = Xmax_param(primary, energy, fluctuations) 
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
        rho = _getAirDensity(hi, "linsley")
        X=X+ rho * step*100. #(deltah*100) *abs(1./np.cos(np.pi-zen2)) # Xmax in g/cm2, slanted = Xmax, vertical/ cos(theta); density in g/cm3, h: m->100cm, np.pi-zen2 since it is defined as where the showers comes from, abs(cosine) so correct for minus values
        
    return h, ai # Xmax_height in m, Xmax_distance in m    

def getGroundXmaxDistance(primary, energy, fluctuations, zenith, glevel, injection):
    
    # zenith in cosmic ray convention here
    XmaxHeight, DistDecayXmax = _dist_decay_Xmax(primary, energy, fluctuations, zenith, glevel, injection)
    zenith = (180 -zenith)*np.pi/180
    GroundAltitude = glevel

    Rearth = 6370949 
    
    dist = np.sqrt((Rearth+ XmaxHeight)**2 - ((Rearth + GroundAltitude)*np.sin(zenith))**2) \
    - (Rearth + GroundAltitude)*np.cos(zenith)
            
    return dist   

def showerdirection(zenith, azimuth):
    
    zenith = zenith*np.pi/180.0
    azimuth = azimuth*np.pi/180.0
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), \
                    np.sin(zenith)*np.sin(azimuth), np.cos(zenith)])
    
    return uv

def getXmaxPosition(primary, energy, fluctuations, azimuth, zenith, glevel, injection):
    
    uv = showerdirection(zenith, azimuth)
    showerDistance = getGroundXmaxDistance(primary, energy, fluctuations, zenith, glevel, injection)
    XmaxPosition = -uv*showerDistance 
    XmaxPosition[2] = XmaxPosition[2] + glevel  
            
    return XmaxPosition

