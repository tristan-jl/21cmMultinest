# -*- coding: utf-8 -*-
import numpy as np
import sys, getopt
import time

USAGE = 'USAGE: fcoll_reader.py -i "<file 1> <file 2>"'

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

files_in = ""

try:
    opts, args = getopt.getopt(sys.argv[1:], "h:i:")
except getopt.GetoptError:
    print USAGE
    sys.exit(2)
# print opts, args
for opt, arg in opts:
    if opt in ("-h", "--h", "--help"):
        print USAGE
        sys.exit()
    elif opt in ("-i"):
        files_in = arg
        files_in = files_in.split()
if files_in == "" or files_in == []:
    print USAGE
    sys.exit()


data1 = load_binary_data(files_in[0])
data2 = load_binary_data(files_in[1])
diff = data1 - data2

print data1 == data2
print diff[diff !=0.]
