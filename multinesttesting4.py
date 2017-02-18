from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import numpy as np
import os
import threading, subprocess
from sys import platform
import scipy as sp
from scipy import stats

if not os.path.exists("chains"): os.mkdir("chains")
def show(filepath): 
	""" open the output (pdf) file for the user """
	if os.name == 'mac' or platform == 'darwin': subprocess.call(('open', filepath))
	elif os.name == 'nt' or platform == 'win32': os.startfile(filepath)
	elif platform.startswith('linux') : subprocess.call(('xdg-open', filepath))

array_size = 11
nopeaks = 2
mu = np.random.uniform(2.5, 10., (nopeaks, 2))
sigma = np.random.uniform(0.2, 2.0, nopeaks)


def Gaussian_2D(coords, centre, width):
    norm=1./((2*np.pi)*width**2)
    result_gaussian = norm*np.exp(-0.5*(((coords[0]-centre[0])/width)**2 + ((coords[1]-centre[1])/width)**2))
    
    return result_gaussian

x=np.linspace(0., 10., array_size)
y=x
xx,yy=np.meshgrid(x,y)

data=np.zeros((array_size,array_size))

data=Gaussian_2D(np.array([xx, yy]), np.array([2.5, 2.5]), 1.) + Gaussian_2D(np.array([xx, yy]), np.array([7.5, 7.5]), 1.)
noise=0.1

def Model(centre1, centre2, width):
    return Gaussian_2D(np.array([xx, yy]), centre1, width) + Gaussian_2D(np.array([xx, yy]), centre2, width)

def Prior(cube, ndim, nparams):
    for i in range(ndim):
        cube[i] = cube[i]*10.        

def Loglike(cube, ndim, nparams):
    #centre=np.array(cube[0], cube[1])
    model=Model(np.array([cube[0], cube[1]]), np.array([cube[3], cube[4]]), 1.)
    loglikelihood = (-0.5 * ((model - data) / noise)**2).sum()
    return loglikelihood

# number of dimensions our problem has
parameters = ["x1", "y1", "x2", "y2"]
n_params = len(parameters)

# we want to see some output while it is running
progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/'); progress.start()
threading.Timer(2, show, ["chains/phys_live.points.pdf"]).start() # delayed opening
# run MultiNest
pymultinest.run(Loglike, Prior, n_params, importance_nested_sampling = False, resume = False, verbose = True, sampling_efficiency = 'model', n_live_points = 5000, outputfiles_basename='chains/')
# ok, done. Stop our progress watcher
progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='chains/')
s = a.get_stats()

import json
# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
	json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
	json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

import matplotlib.pyplot as plt
plt.clf()

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
	plt.subplot(n_params, n_params, n_params * i + i + 1)
	p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
	plt.ylabel("Probability")
	plt.xlabel(parameters[i])
	
	for j in range(i):
		plt.subplot(n_params, n_params, n_params * j + i + 1)
		#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
		p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
		plt.xlabel(parameters[i])
		plt.ylabel(parameters[j])

plt.savefig("chains/marginals_multinest.pdf") #, bbox_inches='tight')
show("chains/marginals_multinest.pdf")

for i in range(n_params):
	outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
	p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
	plt.ylabel("Probability")
	plt.xlabel(parameters[i])
	plt.savefig(outfile, format='pdf', bbox_inches='tight')
	plt.close()
	
	outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
	p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
	plt.ylabel("Cumulative probability")
	plt.xlabel(parameters[i])
	plt.savefig(outfile, format='pdf', bbox_inches='tight')
	plt.close()

print("Take a look at the pdf files in chains/") 

print("mus=", mu, "sigmas=", sigma)
