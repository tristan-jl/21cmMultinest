import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as font_manager
# import pymultinest
import json
import sys#, getopt
import imp
# from MN_Fcoll.MN2 import Gaussian_3D
# from MN_Fcoll.MN2 import Unimodal_Model
# from MN_Fcoll.MN2 import Multimodal_Model
# import pymultinest.plot as pmt:
pmt = imp.load_source("plot", "C:/Users/Ronnie/Documents/PyMultiNest/pymultinest/plot.py")
# import pymultinest.analyse as pma:
pma = imp.load_source("analyse", "C:/Users/Ronnie/Documents/PyMultiNest/pymultinest/analyse.py")


font_path = 'C:\Windows\Fonts\Roboto-Regular.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=11)
title_fontsize = 13
tick_fontsize = 11
def set_ax_font(ax):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(font_prop)
        label.set_fontsize(tick_fontsize)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


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
    def __init__(self, num=4, filename=None):
        """
        Args:
            num (int): the number of Gaussians
            generate (bool): whether or not to generate data
            run (bool): whether or not to run MultiNest
            marginals (bool): whether or not to plot the marginals
            filename (str): the filename in the out/ directory (must be specified if any of the 3 above are True)
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

        self.num = num

        if filename != None:
            self.data = load_binary_data(filename).reshape((self.array_size, self.array_size, self.array_size))





    def marginals(self, datafile, num=None, ij=None, savefig=False):
        self.pm_analyser = pma.Analyzer(self.n_params, outputfiles_basename=datafile+'_1_')
        self.pm_marg_modes = pmt.PlotMarginalModes(self.pm_analyser)

        if (num == None and ij == None):
            fig = plt.figure(figsize=(5*self.n_params, 5*self.n_params), dpi=72)
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
            # plt.savefig(datafile + "_1_marg.png")
            plt.show()
        elif isinstance(num, int):
            fig = plt.figure(dpi=72)
            ax = fig.add_subplot(111)
            self.pm_marg_modes.plot_marginal(num, grid_points=100)
            plt.xlabel(self.parameters[num], fontproperties=font_prop)
            plt.ylabel("Probability", fontproperties=font_prop)
            set_ax_font(ax)

            if savefig == False:
                plt.show()
            else:
                plt.savefig("marginals_" + str(num) + ".png", dpi=200, bbox_inches="tight")
                plt.close()

        elif ij != None:
            i = ij[0]
            j = ij[1]
            fig = plt.figure(dpi=72)
            ax = fig.add_subplot(111, aspect="equal")
            self.pm_marg_modes.plot_marginal(j, i, with_ellipses=False) # WITH_ELLIPSES=FALSE!!!!
            plt.xlabel(self.parameters[j], fontproperties=font_prop)
            plt.ylabel(self.parameters[i], fontproperties=font_prop)
            set_ax_font(ax)
            # im = ax.images #this is a list of all images that have been plotted
            # cb = im[-1].colorbar
            # cb.remove()
            # plt.draw()
            if i in (0, 1, 2):
                plt.ylim((0, 64))
            elif i == 3:
                plt.ylim((0.1, 4))
            elif i == 4:
                plt.ylim((0, 1))
            if j in (0, 1, 2):
                plt.xlim((0, 64))
            elif j == 3:
                plt.xlim((0.1, 4))
            elif j == 4:
                plt.xlim((0, 1))

            if savefig == False:
                plt.show()
            else:
                plt.savefig("marginals_" + str(i) + "_" + str(j) + ".png", dpi=200, bbox_inches="tight")
                plt.close()

        # set_ax_font(ax)
        # # plt.show()
        # plt.savefig("marginals_" + )
