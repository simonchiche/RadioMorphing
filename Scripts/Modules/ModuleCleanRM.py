import numpy as np
import glob
import os

def CleanFolders(outputfiles):
    
    for f in outputfiles:
        os.remove(f)