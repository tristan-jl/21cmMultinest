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
# xxyy = np.array([[xx[i,j], yy[i,j]] for i in range(0, array_size) for j in range(0, array_size)])

num = 1 #8
centre_list = np.random.uniform(*x_range, size=(num,2))
width_list = np.random.uniform(*sigma_range, size=(num,2))
height_list = np.random.uniform(*amplitude_range, size=num)
#widths=np.ones((num,2))

# print centre_list
# print width_list

# Model = Gaussian_2D
noise = 0.01
# data = np.zeros((array_size, array_size))
# data = np.random.normal(data, noise)
#
# for i in range(num):
#     data += Gaussian_2D(xx, yy, *np.concatenate([centre_list[i], width_list[i], [height_list[i]]]))

datafile = sys.argv[1]
data = np.loadtxt(datafile)

color_map = LinearSegmentedColormap.from_list('mycmap', ['black', 'red', 'yellow', 'white'])
plt.axis('equal')
plt.pcolormesh(x, y, data, cmap=color_map)

#Multinest time
number=5 #number of gaussians

# def Model(number, centres, widths, heights):
#     result = np.zeros((array_size,array_size))
#
#     for i in range(number):
#         result += Gaussian_2D(np.array([xx,yy]), centre_list[i], width_list[i], height_list[i])
#
#     return result


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

    # for i in range(ndim):
    #     cube[i] = cube[i]*10. - 5.


def Loglike(cube, ndim, nparams):
    x0, y0 = cube[0], cube[1]
    xsigma, ysigma = cube[2], cube[3]
    amplitude = cube[4]

    # for i in range(ndim):
    #     if 0 <= i <= (number*2 - 1):
    #         centres[i][0] = cube[2*i]
    #         centres[i][1] = cube[2*i+1]
    #     elif (number*2) <= i <= (number*2-1 + 2*number):
    #         widths[i][0] = cube[2*i]
    #         widths[i][1] = cube[2*i+1]
    #     elif (number*2+3*number) <= i <= (number-1 + 3*number):
    #         heights[i] = cube[i]
    #     else:
    #         print "i wrong index"

    # loglikelihood = np.log(Model(number, centres, widths, heights))

    model = Model(x0, y0, xsigma, ysigma, amplitude)
    loglikelihood = (-0.5 * ((model - data) / noise)**2).sum()

    return loglikelihood



# run MultiNest
pymultinest.run(Loglike, Prior, n_params, outputfiles_basename=datafile + '_1_', n_live_points=500, resume=False, verbose=True)
json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

# plt.show()
plt.savefig(datafile + "_fig.png")
