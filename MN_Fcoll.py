USAGE = '[mpiexec -n 4] python MN_Fcoll.py -i <file in (box)> [-n <number of live points>] [-s <scatter value>] [--resume]'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import json
import sys
# import getopt
import time
try:
    import pymultinest
    ANALYSIS_MODE = False
except:
    import imp
    # import pymultinest.plot as pmt:
    pmt = imp.load_source("plot", "C:/Users/Ronnie/Documents/PyMultiNest/pymultinest/plot.py")
    # import pymultinest.analyse as pma:
    pma = imp.load_source("analyse", "C:/Users/Ronnie/Documents/PyMultiNest/pymultinest/analyse.py")
    ANALYSIS_MODE = True
    exc_info = sys.exc_info()




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


class MN:
    def __init__(self, filename=None):#, run=True, marginals=True):
        """
        Args:
            filename (str): the filename of the box
            run (bool): whether or not to run MultiNest
            marginals (bool): whether or not to plot the marginals
        """
        self.parameters = ["x0", "y0", "z0", "width", "amplitude"]#["x0a", "y0a", "x0b", "y0b"]#, "sigma_x", "sigma_y", "amplitude"]
        self.n_params = len(self.parameters)

        self.array_size = 64

        self.x_range = (0., 64.)
        # self.sigma_range = (0.2, 1.)
        # self.amplitude_range = (0.1, 2.,)

        self.x = np.linspace(*self.x_range, num=self.array_size)
        self.y = np.linspace(*self.x_range, num=self.array_size)
        self.z = np.linspace(*self.x_range, num=self.array_size)

        self.xyz = np.meshgrid(self.x, self.y, self.z)
        self.xx, self.yy, self.zz = self.xyz


        if filename == None:
            raise Exception("No filename specified")
        self.filename = filename
        self.data = load_binary_data(self.filename).reshape((self.array_size, self.array_size, self.array_size))


    def Gaussian_3D(self, coord, x0, y0, z0, width, amplitude):
        x, y, z = coord
        # sigma_x, sigma_y = width, width
        sigma = width
        normalisation = 1.

        # for i in range(len(width)):
        #     normalisation *= 1. / (width[i] * np.sqrt(2*np.pi))

        return amplitude * normalisation * np.exp(-0.5 * ( ((x-x0)/sigma)**2 + ((y-y0)/sigma)**2  + ((z-z0)/sigma)**2 ))


    def Unimodal_Model(self, x0, y0, z0, width, amp):#, sigma_x, sigma_y, amplitude):
        """
        Args:
            x0 (float): x coord of centre of Gaussian
            y0 (float): y coord of centre of Gaussian
            z0 (float): z coord of centre of Gaussian
            width (float): width of Gaussian (i.e. sigma)
        """
        # amp = 1.0
        return self.Gaussian_3D(self.xyz, x0, y0, z0, width, amp)


    def Multimodal_Model(self, x0, y0, z0, width, amp):#, sigma_x, sigma_y, amplitude):
        """
        Args:
            x0 (array): list of x coords of centre of Gaussians
            y0 (array): list of y coords of centre of Gaussians
            z0 (array): list of z coords of centre of Gaussians
            width (array): widths of Gaussians (i.e. sigma)
        """
        # amp = 1.
        model = np.zeros_like(self.xyz[0])
        for i in range(len(x0)):
            model += self.Gaussian_3D(self.xyz, x0[i], y0[i], z0[i], width[i], amp[i])
        return model


    def Prior(self, cube, ndim, nparams):
        """
        Map unit Prior cube onto non-unit paramter space

        Args:
            cube - [x0, y0, z0, width]
            ndim - the number of dimensions of the (hyper)cube
            nparams - WTF
        """
        def transform_centres(i):
            cube[i] = max(self.x_range) * cube[i]
        def transform_widths(i):
            cube[i] = 10. * cube[i]
        # x0, y0, z0:
        for i in [0, 1, 2]:
            transform_centres(i)
        # width:
        transform_widths(3)


    def Loglike(self, cube, ndim, nparams):
        x0, y0, z0 = cube[0], cube[1], cube[2]
        width = cube[3]
        amp = cube[4]

        model = self.Unimodal_Model(x0, y0, z0, width, amp)#, x0b, y0b)#, xsigma, ysigma, amplitude)
        loglikelihood = (-0.5 * ((model - self.data) / self.scatter)**2).sum()

        return loglikelihood


    def _plot(self, data, z_index=32):
        lims = [min(self.x_range), max(self.x_range), min(self.x_range), max(self.x_range)]

        fig = plt.figure()
        sub_fig = fig.add_subplot(111)
        print "Taking a slice along the LOS direction at index="+str(z_index)
        the_slice = data[:,:,z_index]

        frame1 = plt.gca()
        color_map = LinearSegmentedColormap.from_list('mycmap', ['black', 'red', 'yellow', 'white'])
        c_dens = sub_fig.imshow(the_slice, cmap=color_map, extent=lims, origin="lower")
        c_dens.set_clim(vmin=0,vmax=1)
        c_bar = fig.colorbar(c_dens, orientation='vertical')
        # plt.axis('equal')
        plt.xlim(lims[:2])
        plt.ylim(lims[2:])


    def run_sampling(self, marginals=True, n_points=1000, scatter=1., resume=False, mode_tolerance=-1e90, verbose=True, max_iter=0):
        """
        Run PyMultiNest against single Gaussian.

        Args:
            scatter (float): Sampling scatter value
        """
        if ANALYSIS_MODE:
            raise Exception("Could not load full PyMultiNest. Analysis mode only. {0}".format(exc_info))

        self.scatter = scatter

        start_time = time.time()
        # run MultiNest
        pymultinest.run(self.Loglike, self.Prior, self.n_params, outputfiles_basename=self.filename+'_1_', n_live_points=n_points, resume=resume, importance_nested_sampling=False, mode_tolerance=mode_tolerance, verbose=verbose, max_iter=max_iter)

        end_time = time.time()
        print "\nTime taken:", int((end_time - start_time) / 60.), "mins\n"

        json.dump(self.parameters, open(self.filename + '_1_params.json', 'w')) # save parameter names


    def marginals(self):
        if ANALYSIS_MODE:
            self.pm_analyser = pma.Analyzer(self.n_params, outputfiles_basename=self.filename+'_1_')
            self.pm_marg_modes = pmt.PlotMarginalModes(self.pm_analyser)
        else:
            self.pm_analyser = pymultinest.Analyzer(self.n_params, outputfiles_basename=self.filename+'_1_')
            self.pm_marg_modes = pymultinest.PlotMarginalModes(self.pm_analyser)

        fig = plt.figure(figsize=(5*self.n_params, 5*self.n_params))
        for i in range(self.n_params):
            plt.subplot(self.n_params, self.n_params, self.n_params * i + i + 1)
            self.pm_marg_modes.plot_marginal(i, grid_points=100)
            plt.xlabel(self.parameters[i])
            plt.ylabel("Probability")
            # plt.savefig(self.filename + "_1_marg_" + str(i) + ".png")
            for j in range(i):
                plt.subplot(self.n_params, self.n_params, self.n_params * j + i + 1)
                self.pm_marg_modes.plot_marginal(j, i, with_ellipses=False) # WITH_ELLIPSES=FALSE!!!!
                plt.xlabel(self.parameters[j])
                plt.ylabel(self.parameters[i])
                # plt.savefig(self.filename + "_1_marg_" + str(i) + "_" + str(j) + ".png")
        plt.savefig(self.filename + "_1_marg.png")


    def save_modes(self):
        if ANALYSIS_MODE:
            self.pm_analyser = pma.Analyzer(self.n_params, outputfiles_basename=self.filename+'_1_')
        else:
            self.pm_analyser = pymultinest.Analyzer(self.n_params, outputfiles_basename=self.filename+'_1_')

        mode_stats = self.pm_analyser.get_mode_stats()["modes"]
        # print mode_stats
        n_modes = len(mode_stats)
        means = np.array([mode_stats[i]["mean"] for i in range(n_modes)])
        sigmas = np.array([mode_stats[i]["sigma"] for i in range(n_modes)])
        optimal_params = np.dstack((means, sigmas))

        if len(means) == 0:
            print "No modes detected"
        else:
            fit_data = self.Multimodal_Model(means[:,0], means[:,1], means[:,2], means[:,3], means[:,4])
            self._plot(fit_data)
            plt.savefig(self.filename + "_1_fig.png")
            if self.filename[-3:] == "Mpc":
                datafile_split = self.filename.split("_")
                outfile = "_".join(datafile_split[:-2]) + "_MODES_" + "_".join(datafile_split[-2:])
            else:
                outfile = self.filename
            write_binary_data(outfile, fit_data.flatten())

            for n in range(len(optimal_params)):
                mode = optimal_params[n]
                print "Mode", n
                for i in range(self.n_params):
                    print "  " + self.parameters[i] + ": ", mode[i][0], "+/-", mode[i][1]
