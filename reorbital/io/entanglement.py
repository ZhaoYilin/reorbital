import re
import numpy as np

def read_single_orbital_entropy(filename):
    with open(filename,"r") as f:
        for i,line in enumerate(f):
            if line.strip().startswith("Single-orbital entropies"):
                line = line.replace(" ","")
                line = line.strip('\n')
                line = line.strip('.')
                line = line.split('=')[1]
                line = line.strip("[")
                line = line.strip("]")
                line = line.split(",")
                line = [float(j) for j in line]

                f.close()
                break
    return line

def read_mutual_information(filename,norb):
    with open(filename,"r") as f:
        for line in f:
            if line.strip().startswith("Two-orbital mutual information"):
                matrix = np.zeros((norb//8,norb,8))
                line = next(f)
                line = next(f)
                for i in range(norb//8):
                    line = next(f)
                    line = next(f)
                    line = next(f)
                    for j in range(norb):
                        line = line.strip()
                        line = line.split()
                        line = [float(j) for j in line]
                        matrix[i,j,:]= line
                        line = next(f)
                line = next(f)
                line = next(f)
                line = next(f)
                if norb//8>=2:
                    for i in range(norb//8):
                        matrix[0,:,:] = np.concatenate((matrix[0,:,:],matrix[i+1,:,:]),axis=1)
                matrix = matrix[0,:,:]
                redundante = np.zeros((norb,norb%8))
                for k in range(norb):
                    line = line.strip()
                    line = line.split()
                    line = [float(j) for j in line]
                    redundante[k,:]= line
                    line = next(f)
                matrix = np.concatenate((matrix,redundante),axis=1)
    return matrix
