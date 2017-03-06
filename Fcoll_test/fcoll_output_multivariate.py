# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
from scipy.stats import multivariate_normal
import sys
import time

def load_binary_data(filename, dtype=np.float32): 
     """ 
     We assume that the data was written with write_binary_data() (little endian). 
     """ 
     f = open(filename, "rb") 
     data = f.read() 
     f.close() 
     _data = np.fromstring(data, dtype) 
     if sys.byteorder == 'big':
       _data = _data.byteswap()
     return _data 

def write_binary_data(filename, data, dtype=np.float32):
     """
     Write binary data to a file and make sure it is always in little endian format.
     """
     f = open(filename, "wb")
     _data = np.asarray(data, dtype)
     if sys.byteorder == 'big':
         _data = _data.byteswap()
     f.write(_data)
     f.close()

def Gaussian_3D(coords, centre, width):
    '''
    Takes grid (coords) as arg, along with centre and width of Gaussian. Returns another grid.
    '''
    normal=[]
    power=0.
    
    for i in range(3):
        normal.append(1./(width[i]*(2*np.pi)**0.5))
        power += ((coords[i] - centre[i])/width[i])**2

    normal = linalg.norm(normal)
    result = normal*np.exp(-0.5*power)
    
    return result

#create grid
x_ = np.linspace(0., 255., 256)
y_ = np.linspace(0., 255., 256)
z_ = np.linspace(0., 255., 256)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
pos = np.stack((x, y, z), axis = -1)

#create data
N_peaks = 10
centre_list = np.random.uniform(0., 256., (N_peaks,2))
height_list = np.random.uniform(0., 1., N_peaks)

data = np.zeros((256, 256, 256))

start=time.time()

for i in xrange(N_peaks):
    print i
    data += height_list[i] * multivariate_normal.pdf(pos, [128., 128., 128.], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

end=time.time()
print end - start

outputfile = write_binary_data('Fcoll_output_file', data)

#data1=load_binary_data('Fcoll_output_file')
#print data1.shape
#print data1

