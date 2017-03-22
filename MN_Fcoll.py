USAGE = 'EXAMPLE USAGE:\n>>>>A = MN2()\n>>>>A.generate_data("data1")\n>>>>A.run_sampling("out/data1")\n>>>>A.marginals("out/data1")'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pymultinest
import json
import sys#, getopt


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


class MN2:
    def __init__(self, num=4, generate=False, run=False, marginals=False, filename=None):
        """
        Args:
            num (int): the number of Gaussians
            generate (bool): whether or not to generate data
            run (bool): whether or not to run MultiNest
            marginals (bool): whether or not to plot the marginals
            filename (str): the filename in the out/ directory (must be specified if any of the 3 above are True)
        """
        self.parameters = ["x0", "y0", "z0", "width"]#["x0a", "y0a", "x0b", "y0b"]#, "sigma_x", "sigma_y", "amplitude"]
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

        self.num = num #8

        if generate == True and filename != None:
            self.generate_data(filename)
        elif run == True and filename != None:
            if marginals == True:
                self.run_sampling("out/"+filename, marginals=True)
            else:
                self.run_sampling("out/"+filename)
        elif marginals == True and filename != None:
            self.marginals("out/"+filename)
        elif (generate == True or run == True or marginals == True) and filename == None:
            raise Exception("No filename specified")


    def Gaussian_3D(self, coord, x0, y0, z0, width, amplitude):
        x, y, z = coord
        # sigma_x, sigma_y = width, width
        sigma = width
        normalisation = 1.

        # for i in range(len(width)):
        #     normalisation *= 1. / (width[i] * np.sqrt(2*np.pi))

        return amplitude * normalisation * np.exp(-0.5 * ( ((x-x0)/sigma)**2 + ((y-y0)/sigma)**2  + ((z-z0)/sigma)**2 ))


    def Unimodal_Model(self, x0, y0, z0, width):#, sigma_x, sigma_y, amplitude):
        """
        Args:
            x0 (float): x coord of centre of Gaussian
            y0 (float): y coord of centre of Gaussian
            z0 (float): z coord of centre of Gaussian
            width (float): width of Gaussian (i.e. sigma)
        """
        amp = 1.0
        return self.Gaussian_3D(self.xyz, x0, y0, z0, width, amp)


    def Multimodal_Model(self, x0, y0, z0, width):#, sigma_x, sigma_y, amplitude):
        """
        Args:
            x0 (array): list of x coords of centre of Gaussians
            y0 (array): list of y coords of centre of Gaussians
            z0 (array): list of z coords of centre of Gaussians
            width (array): widths of Gaussians (i.e. sigma)
        """
        amp = 1.
        model = np.zeros_like(self.xyz[0])
        for i in range(len(x0)):
            model += self.Gaussian_3D(self.xyz, x0[i], y0[i], z0[i], width[i], amp)
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
            cube[i] = max(self.x_range)/10. * cube[i]
        # x0, y0, z0:
        for i in [0, 1, 2]:
            transform_centres(i)
        # width:
        transform_widths(3)


    def Loglike(self, cube, ndim, nparams):
        x0, y0, z0 = cube[0], cube[1], cube[2]
        width = cube[3]

        model = self.Unimodal_Model(x0, y0, z0, width)#, x0b, y0b)#, xsigma, ysigma, amplitude)
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


    def generate_data(self, filename, noise=0.03):
        """
        Create noisy data to run PyMultiNest against. This is saved in the out/ directory. Also saves a plot of the data.

        Args:
            filename (str)
            noise (float)
        """
        x0, y0, z0 = np.random.uniform(*self.x_range, size=self.num), np.random.uniform(*self.x_range, size=self.num), np.array([32,32,32,32])#np.random.uniform(31., 33., size=self.num)
        width = np.random.uniform(1., 5., self.num)

        self.data = self.Multimodal_Model(x0, y0, z0, width)#, sigma_x, sigma_y, amplitude)
        # data = np.random.normal(data, noise)
        write_binary_data("out/" + filename, self.data.flatten())

        self._plot(self.data)
        plt.savefig("out/" + filename + "_fig.png")

        print x0, "\n", y0, "\n", z0, "\n", width


    def run_sampling(self, datafile, marginals=True, n_points=1000, scatter=4., resume=False, mode_tolerance=-1e90, verbose=False, max_iter=0):
        """
        Run PyMultiNest against single Gaussian.

        Args:
            datafile (str): The filename of the data
            scatter (float): Sampling scatter value
        """
        self.data = load_binary_data(datafile).reshape((self.array_size, self.array_size, self.array_size))
        self.scatter = scatter

        # run MultiNest
        pymultinest.run(self.Loglike, self.Prior, self.n_params, outputfiles_basename=datafile+'_1_', n_live_points=n_points, resume=resume, importance_nested_sampling=False, mode_tolerance=mode_tolerance, verbose=verbose, max_iter=max_iter)
        json.dump(self.parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

        self.pm_analyser = pymultinest.analyse.Analyzer(self.n_params, outputfiles_basename=datafile+'_1_')
        mode_stats = self.pm_analyser.get_mode_stats()["modes"]
        # print mode_stats
        n_modes = len(mode_stats)
        means = np.array([mode_stats[i]["mean"] for i in range(n_modes)])
        sigmas = np.array([mode_stats[i]["sigma"] for i in range(n_modes)])
        optimal_params = np.dstack((means, sigmas))

        if len(means) == 0:
            print "No modes detected"
            return 0

        fit_data = self.Multimodal_Model(means[:,0], means[:,1], means[:,2], means[:,3])#, means[2], means[3])
        self._plot(fit_data)
        plt.savefig(datafile + "_1_fig.png")

        for n in range(len(optimal_params)):
            mode = optimal_params[n]
            print "Mode", n
            for i in range(self.n_params):
                print "  " + self.parameters[i] + ": ", mode[i][0], "+/-", mode[i][1]

        if marginals == True:
            self.marginals(datafile)


    def marginals(self, datafile):
        self.pm_analyser = pymultinest.Analyzer(self.n_params, outputfiles_basename=datafile+'_1_')
        self.pm_marg_modes = pymultinest.PlotMarginalModes(self.pm_analyser)

        fig = plt.figure(figsize=(5*self.n_params, 5*self.n_params))
        for i in range(self.n_params):
            plt.subplot(self.n_params, self.n_params, self.n_params * i + i + 1)
            self.pm_marg_modes.plot_marginal(i, grid_points=100)
            plt.xlabel(self.parameters[i])
            plt.ylabel("Probability")
            # plt.savefig(datafile + "_1_marg_" + str(i) + ".png")
            for j in range(i):
                plt.subplot(self.n_params, self.n_params, self.n_params * j + i + 1)
                self.pm_marg_modes.plot_marginal(j, i, with_ellipses=False) # WITH_ELLIPSES=FALSE!!!!
                plt.xlabel(self.parameters[j])
                plt.ylabel(self.parameters[i])
                # plt.savefig(datafile + "_1_marg_" + str(i) + "_" + str(j) + ".png")
        plt.savefig(datafile + "_1_marg.png")



# print USAGE
