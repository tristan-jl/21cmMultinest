import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pymultinest
import json
# import sys, getopt

class MN:
    def __init__(self):
        self.parameters = ["x0", "y0", "sigma_x", "sigma_y", "amplitude"]
        self.n_params = len(self.parameters)

        self.array_size = 101

        self.x_range = (-5., 5.)
        self.sigma_range = (0.2, 1.)
        self.amplitude_range = (0.1, 2.,)

        self.x = np.linspace(*self.x_range, num=self.array_size)
        self.y = np.linspace(*self.x_range, num=self.array_size)
        self.xxyy = np.meshgrid(self.x, self.y)
        self.xx, self.yy = self.xxyy

        self.num = 1 #8

    def Gaussian_2D(self, coord, x0, y0, sigma_x, sigma_y, amplitude):
        x, y = coord
        normalisation = 1.

        # for i in range(len(width)):
        #     normalisation *= 1. / (width[i] * np.sqrt(2*np.pi))

        return amplitude * normalisation * np.exp(-0.5 * ( ((x-x0)/sigma_x)**2 + ((y-y0)/sigma_x)**2 ))


    def Model(self, x0, y0, sigma_x, sigma_y, amplitude):
        return self.Gaussian_2D(self.xxyy, x0, y0, sigma_x, sigma_y, amplitude)


    def Prior(self, cube, ndim, nparams):
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


    def Loglike(self, cube, ndim, nparams):
        x0, y0 = cube[0], cube[1]
        xsigma, ysigma = cube[2], cube[3]
        amplitude = cube[4]

        model = self.Model(x0, y0, xsigma, ysigma, amplitude)
        loglikelihood = (-0.5 * ((model - self.data) / self.noise)**2).sum()

        return loglikelihood


    def _plot(self, data):
        plt.figure()
        color_map = LinearSegmentedColormap.from_list('mycmap', ['black', 'red', 'yellow', 'white'])
        plt.axis('equal')
        plt.pcolormesh(self.x, self.y, data, cmap=color_map, vmin=0., vmax=max(self.amplitude_range))
        plt.colorbar()


    def generate_data(self, filename, noise=0.03):
        """
        Create noisy data to run PyMultiNest against. This is saved in the out/ directory. Also saves a plot of the data.

        Args:
            filename (str)
            noise (float)
        """
        x0, y0 = np.random.uniform(*self.x_range, size=(self.num, 2))[0]
        sigma_x, sigma_y = np.random.uniform(*self.sigma_range, size=(self.num, 2))[0]
        amplitude = np.random.uniform(*self.amplitude_range, size=self.num)[0]

        data = self.Model(x0, y0, sigma_x, sigma_y, amplitude)
        data = np.random.normal(data, noise)
        np.savetxt("out/" + filename, data)

        self._plot(data)
        plt.savefig("out/" + filename + "_fig.png")


    def run_sampling(self, datafile, noise=0.01):
        """
        Run PyMultiNest against single Gaussian.

        Args:
            datafile (str): The filename of the data
            noise (float): Loglikelihood noise value
        """
        self.data = np.loadtxt(datafile)
        self.noise = noise

        # run MultiNest
        pymultinest.run(self.Loglike, self.Prior, self.n_params, outputfiles_basename=datafile+'_1_', n_live_points=500, resume=False, verbose=True)
        json.dump(self.parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

        pm_analyser = pymultinest.analyse.Analyzer(self.n_params, outputfiles_basename=datafile+'_1_')
        mode_stats = pm_analyser.get_mode_stats()["modes"][0]
        means = mode_stats["mean"]
        sigmas = mode_stats["sigma"]
        opt_params = np.dstack((means, sigmas))
        fit_data = self.Model(means[0], means[1], means[2], means[3], means[4])
        self._plot(fit_data)
        plt.savefig(datafile + "_1_fig.png")


# USAGE = "To generate data file: \n$ python test.py -o <output data file> \nOr to run PyMultiNest on data file: \n$ python test.py -i <input data file>"
#
# try:
#     opts, args = getopt.getopt(sys.argv[1:], "h:i:o:")
# except getopt.GetoptError:
#     print USAGE
#     sys.exit(2)
# # print opts, args
# for opt, arg in opts:
#     if opt in ("-h", "--h", "--help"):
#         print USAGE
#         sys.exit()
#     elif opt in ("-o"):
#         print "Generating", arg
#         break
#     elif opt in ("-i"):
#         print "Running", arg
#         break
# if not opts:
#     print USAGE
#     sys.exit()
