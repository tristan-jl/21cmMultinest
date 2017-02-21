import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pymultinest
import sys
import json

def Gaussian_2D(coord, x0, y0, sigma_x, sigma_y, amplitude):
    x, y = coord
    normalisation = 1.

    # for i in range(len(width)):
    #     normalisation *= 1. / (width[i] * np.sqrt(2*np.pi))

    return amplitude * normalisation * np.exp(-0.5 * ( ((x-x0)/sigma_x)**2 + ((y-y0)/sigma_x)**2 ))

def Model(x0, y0, sigma_x, sigma_y, amplitude):
    return Gaussian_2D(xxyy, x0, y0, sigma_x, sigma_y, amplitude)

parameters = ["x0", "y0", "sigma_x", "sigma_y", "amplitude"]
n_params = len(parameters)

array_size = 101

x_range = (-5., 5.)
sigma_range = (0.2, 1.)
amplitude_range = (0.1, 2.,)

x = np.linspace(*x_range, num=array_size)
y = np.linspace(*x_range, num=array_size)
xxyy = np.meshgrid(x, y)
xx, yy = xxyy

num = 1 #8
centre_list = np.random.uniform(*x_range, size=(num,2))
width_list = np.random.uniform(*sigma_range, size=(num,2))
height_list = np.random.uniform(*amplitude_range, size=num)

noise = 0.01


datafile = sys.argv[1]
data = np.loadtxt(datafile)

color_map = LinearSegmentedColormap.from_list('mycmap', ['black', 'red', 'yellow', 'white'])
plt.axis('equal')
plt.pcolormesh(x, y, data, cmap=color_map)

plt.savefig(datafile + "_fig.png")

#Multinest time
number=5 #number of gaussians



def Prior(cube, ndim, nparams):
    """
    Map unit Prior cube onto non-unit paramter space

    Args:
        cube - [x, y, x0, y0, sigma_x, sigma_y, amplitude]
            Each element is an array
        ndim - the number of dimensions of the (hyper)cube
        nparams - WTF
    """
    # x0, y0:
    cube[0], cube[1] = 10.*cube[0] - 5., 10.*cube[1] - 5.
    # sigma_x, sigma_y:
    cube[2], cube[3] = cube[2], cube[3]
    # amplitude:
    cube[4] = 2.*cube[4]



def Loglike(cube, ndim, nparams):
    x0, y0 = cube[0], cube[1]
    xsigma, ysigma = cube[2], cube[3]
    amplitude = cube[4]


    model = Model(x0, y0, xsigma, ysigma, amplitude)
    loglikelihood = (-0.5 * ((model - data) / noise)**2).sum()

    return loglikelihood



# run MultiNest
pymultinest.run(Loglike, Prior, n_params, outputfiles_basename=datafile + '_1_', n_live_points=500, resume=False, verbose=True)
json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

# plt.show()
