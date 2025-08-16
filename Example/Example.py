import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import hilbert

TracesPath = "./../OutputDirectory/"
Pos = np.loadtxt("./../DesiredPositions/desired_pos.txt")
Nant = len(glob.glob(TracesPath + "*"))
peak_E = np.zeros(Nant)

for i in range(Nant):
    # Load the traces
    filename = TracesPath + "DesiredTraces_" + str(i) + ".txt"
    t, Ex, Ey, Ez = np.loadtxt(filename, unpack=True)
    Etot = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    
    hilbert_E = abs(hilbert(Etot))
    peak_E[i] = np.max(hilbert_E)


plt.scatter(Pos[:,0], Pos[:,1], c=peak_E, cmap ="jet", s= 10)
cbar = plt.colorbar()
cbar.set_label('Peak Electric Field ($\mu V/m$)', fontsize=12)
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
#plt.savefig("Example_Peak_Electric_Field.pdf", bbox_inches='tight')
plt.show()
