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

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

#create grid
grid_length = 64
x_ = np.linspace(0., grid_length - 1., grid_length)
y_ = np.linspace(0., grid_length - 1., grid_length)
z_ = np.linspace(0., grid_length - 1., grid_length)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
pos = np.stack((x, y, z), axis = -1)

#create data
N_peaks = 1000
centre_list = np.random.uniform(0., grid_length, (N_peaks, 3))
height_list = np.random.uniform(0., 1., N_peaks)
rand_matrix_list = np.random.uniform(0., 1., (N_peaks, 3, 3))
data = np.zeros((int(grid_length), int(grid_length), int(grid_length)))

start=time.time()

for i in xrange(N_peaks):
    print i
    cov_matrix = np.dot(rand_matrix_list[i],rand_matrix_list[i].transpose())
    #matrix sometimes singular, ie has very small det (havent seen one equal to 0 yet)
    data += height_list[i] * multivariate_normal.pdf(pos, centre_list[i], cov_matrix, allow_singular = True)

end=time.time()
print end - start

outputfile = write_binary_data('Fcoll_output_file', data)

#data1=load_binary_data('Fcoll_output_file')
#print data1.shape
#print data1

