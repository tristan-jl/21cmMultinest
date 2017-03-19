USAGE = 'EXAMPLE USAGE:\n>>>>A = MN2()\n>>>>A.generate_data("data1")\n>>>>A.run_sampling("out/data1")\n>>>>A.marginals("out/data1")'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pymultinest
import json
# import sys, getopt

class MN2:
    def __init__(self):
        self.parameters = ["x0", "y0"]#["x0a", "y0a", "x0b", "y0b"]#, "sigma_x", "sigma_y", "amplitude"]
        self.n_params = len(self.parameters)

        self.array_size = 101

        self.x_range = (-5., 5.)
        # self.sigma_range = (0.2, 1.)
        # self.amplitude_range = (0.1, 2.,)

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

    def Unimodal_Model(self, x0, y0):#, sigma_x, sigma_y, amplitude):
        """
        Args:
            x0 (float): x coord of centre of Gaussian
            y0 (float): y coord of centre of Gaussian
        """
        other_params = (1., 1., 1.1)
        return self.Gaussian_2D(self.xxyy, x0, y0, *other_params)


    def Multimodal_Model(self, x0, y0):#, sigma_x, sigma_y, amplitude):
        """
        Args:
            x0 (array): list of x coords of centre of Gaussians
            y0 (array): list of y coords of centre of Gaussians
        """
        other_params = (1., 1., 1.1)
        model = np.zeros_like(self.xxyy[0])
        for i in range(len(x0)):
            model += self.Gaussian_2D(self.xxyy, x0[i], y0[i], *other_params)
        return model


    def Prior(self, cube, ndim, nparams):
        """
        Map unit Prior cube onto non-unit paramter space

        Args:
            cube - [x0a, y0a, x0b, y0b]
                Each element is an float
            ndim - the number of dimensions of the (hyper)cube
            nparams - WTF
        """
        # x0, y0:
        cube[0], cube[1] = 10.*cube[0] - 5., 10.*cube[1] - 5.
        # y0:
        # cube[1], cube[3] = 10.*cube[1] - 5., 10.*cube[3] - 5.
        # # sigma_x, sigma_y:
        # cube[2], cube[3] = cube[2], cube[3]
        # # amplitude:
        # cube[4] = 2.*cube[4]


    def Loglike(self, cube, ndim, nparams):
        # x0a, y0b = cube[0], cube[1]
        # x0b, y0b = cube[2], cube[3]
        x0, y0 = cube[0], cube[1]
        # xsigma, ysigma = cube[2], cube[3]
        # amplitude = cube[4]

        model = self.Unimodal_Model(x0, y0)#, x0b, y0b)#, xsigma, ysigma, amplitude)
        loglikelihood = (-0.5 * ((model - self.data) / self.noise)**2).sum()

        return loglikelihood


    def _plot(self, data):
        plt.figure()
        color_map = LinearSegmentedColormap.from_list('mycmap', ['black', 'red', 'yellow', 'white'])
        plt.axis('equal')
        plt.pcolormesh(self.x, self.y, data, cmap=color_map, vmin=0., vmax=2.)#max(self.amplitude_range))
        plt.colorbar()


    def generate_data(self, filename, noise=0.03):
        """
        Create noisy data to run PyMultiNest against. This is saved in the out/ directory. Also saves a plot of the data.

        Args:
            filename (str)
            noise (float)
        """
        x0, y0 = [-0.63, 1.04, 3.8], [0.98, -3.01, 1.2]
        # sigma_x, sigma_y = np.random.uniform(*self.sigma_range, size=(self.num, 2))[0]
        # amplitude = np.random.uniform(*self.amplitude_range, size=self.num)[0]

        data = self.Multimodal_Model(x0, y0)#, sigma_x, sigma_y, amplitude)
        data = np.random.normal(data, noise)
        np.savetxt("out/" + filename, data)

        self._plot(data)
        plt.savefig("out/" + filename + "_fig.png")


    def run_sampling(self, datafile, n_points=500, noise=1., resume=False, mode_tolerance=-1e20, verbose=False):
        """
        Run PyMultiNest against single Gaussian.

        Args:
            datafile (str): The filename of the data
            noise (float): Loglikelihood noise value
        """
        self.data = np.loadtxt(datafile)
        self.noise = noise

        # run MultiNest
        pymultinest.run(self.Loglike, self.Prior, self.n_params, outputfiles_basename=datafile+'_1_', n_live_points=n_points, resume=resume, importance_nested_sampling=False, mode_tolerance=mode_tolerance, verbose=verbose)
        json.dump(self.parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

        self.pm_analyser = pymultinest.analyse.Analyzer(self.n_params, outputfiles_basename=datafile+'_1_')
        mode_stats = self.pm_analyser.get_mode_stats()["modes"]
        # print mode_stats
        n_modes = len(mode_stats)
        means = np.array([mode_stats[i]["mean"] for i in range(n_modes)])
        sigmas = np.array([mode_stats[i]["sigma"] for i in range(n_modes)])
        optimal_params = np.dstack((means, sigmas))
        fit_data = self.Multimodal_Model(means[:,0], means[:,1])#, means[2], means[3])
        self._plot(fit_data)
        plt.savefig(datafile + "_1_fig.png")

        for n in range(len(optimal_params)):
            mode = optimal_params[n]
            print "Mode", n
            for i in range(self.n_params):
                print "  " + self.parameters[i] + ": ", mode[i][0], "+/-", mode[i][1]


    def marginals(self, datafile):
        self.pm_analyser = pymultinest.Analyzer(self.n_params, outputfiles_basename=datafile+'_1_')
        self.pm_marg_modes = pymultinest.PlotMarginalModes(self.pm_analyser)

        fig = plt.figure(figsize=(3*self.n_params, 3*self.n_params))
        for i in range(self.n_params):
            plt.subplot(self.n_params, self.n_params, self.n_params * i + i + 1)
            self.pm_marg_modes.plot_marginal(i, grid_points=100)
            plt.xlabel(self.parameters[i])
            plt.ylabel("Probability")
            # plt.savefig(datafile + "_1_marg_" + str(i) + ".png")
            for j in range(i):
                plt.subplot(self.n_params, self.n_params, self.n_params * j + i + 1)
                self.pm_marg_modes.plot_marginal(j, i, with_ellipses=False) # WITH_ELLIPSES=FALSE!!!!
                plt.xlabel(self.parameters[i])
                plt.ylabel(self.parameters[j])
                # plt.savefig(datafile + "_1_marg_" + str(i) + "_" + str(j) + ".png")
        plt.savefig(datafile + "_1_marg.png")



# print USAGE
A = MN2()
