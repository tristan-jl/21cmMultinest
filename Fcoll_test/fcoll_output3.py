# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
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
    power=[]
    
    for i in range(3):
        normal.append(width[i]*(2*np.pi)**0.5)
        power.append((coords[i] - centre[i]/width[i])**2)
    
    normal = 1./linalg.norm(normal)
    power = np.sum(power)

    result = normal*np.exp(-0.5*power)
    
    return result

#create grid
x_ = np.linspace(0., 255., 256)
y_ = np.linspace(0., 255., 256)
z_ = np.linspace(0., 255., 256)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

#create data
N_peaks = 10
centre_list = np.random.uniform(0., 256., (N_peaks,2))
height_list = np.random.uniform(0., 1., N_peaks)

data = np.zeros((256, 256, 256))

start=time.time()

for i in xrange(N_peaks):
    print i
    data += height_list[i] * Gaussian_3D(np.array([x,y,z]), (centre_list[i][0], centre_list[i][1], 128.), (1.,1.,1.))

end=time.time()
print end - start

outputfile = write_binary_data('Fcoll_output_file', data)

#data1=load_binary_data('Fcoll_output_file')
#print data1.shape
#print data1

