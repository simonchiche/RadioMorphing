import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import hilbert

TracesPath = "./../OutputDirectory/"
Pos = np.loadtxt("./../DesiredPositions/desired_pos.txt")

Nant = len(glob.glob(TracesPath + "*"))
peak_E = np.full(Nant, np.nan)   # NaN pour ignorer les fichiers vides

for i in range(Nant):

    filename = TracesPath + "DesiredTraces_" + str(i) + ".txt"

    # Vérifie que le fichier existe
    if not os.path.isfile(filename):
        continue

    # Vérifie qu’il n’est pas vide
    if os.path.getsize(filename) == 0:
        continue

    try:
        t, Ex, Ey, Ez = np.loadtxt(filename, unpack=True)
    except Exception:
        continue

    Etot = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    hilbert_E = np.abs(hilbert(Etot))
    peak_E[i] = np.max(hilbert_E)


# On ne trace que les points valides
mask = ~np.isnan(peak_E)

plt.scatter(Pos[mask,0], Pos[mask,1],
            c=peak_E[mask], cmap="jet", s=10)

cbar = plt.colorbar()
cbar.set_label('Peak Electric Field ($\\mu V/m$)', fontsize=12)

plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)

lim = 30000
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

plt.show()

