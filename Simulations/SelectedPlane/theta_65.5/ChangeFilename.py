import glob
import os
import sys
import subprocess
import numpy as np

path_input = glob.glob('./Stshp*')
for i in range(len(path_input)):

    #print(path_input[i])
    split = path_input[i].split("S23d")[1]
    filename = "Stshp_Stshp" + split 
    #print(split1)
    #print(filename)
    cmd1 = 'mv ' +  path_input[i] + ' ' + filename

    wd = os.getcwd()
    print(cmd1)
    p = subprocess.Popen(cmd1, cwd=wd, shell=True)
    stdout, stderr = p.communicate()
    #k = k +1
    #print(k)



