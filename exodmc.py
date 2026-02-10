#!/usr/bin/env python
# coding: utf-8

__author__ = 'Mariangela Bonavita'
__version__ = 'v2.0'
__all__ = ['exodmc']


import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import colors as mcolors
from matplotlib.ticker import ScalarFormatter
from scipy import interpolate
from numpy import random as rn
import scipy.ndimage as ndimage
from scipy.special import erf
import copy
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)




##########################################
###              EXO-DMC               ###
### EXOplanet Detection Map Calculator ###
##########################################

"""



Class: exodmc

Monte Carlo tool for the statistical analysis of exoplanet surveys results.
Combines the information on the target stars with the instrument detection limits
to estimate the probability of detection of a given synthetic planet population,
ultimately generating detection probability maps.

Credits: Mariangela Bonavita, Vito Squicciarini, Kellen Lawson

Parameters:
star_ID: list, required. ID of the target(s)
star_dist: list, required. Target(s) distance, in pc

Methods:

1) set_grid
Change range or resolution of grid over companions are generated.
	- x_min: float, optional. Lower limit for grid x axis (default = 1)
	- x_max: float, optional. Upper limit for grid x axis (default = 1000)
	- nx: int, optional. Number of steps in the grid x axis (default = 100)
	- xlog: boolean, optional. If True the x axis will be uniformly spaced in log
	- y_min: float, optional. Lower limit for grid y axis (default = 0.5)
	- y_max: float, optional. Upper limit for grid y axis (default = 75)
	- ny: int, optional. Number of steps in the grid y axis (default = 100)
	- ylog: boolean, optional. If True the y axis will be uniformly spaced in log
	- ngen: float, optional. Number of orbital elements sets to be generated for each point in the grid.
		All orbital parameters are uniformly distributed by default, except for the eccentricity. Default: 1000.
	- e_params: dict, optional. Specifies the parameters needed to define the eccentricity distribution. If used, the following keys can/must be present:
            - shape: string, required. Desired eccentricity distribution. Can be uniform ('uniform') or Gaussian ('gauss')
            - mean: float, optional. Only used if shape = 'gauss'. Mean of the gaussian eccentricity distribution.
            - sigma: float, optional. Only used if shape = 'gauss'. Standard deviation of the gaussian eccentricity distribution.
            - min: float, optional. Only used if shape = 'uniform'. Lower eccentricity value to be considered.
            - max: float, optional. Only used if shape = 'uniform'. Upper eccentricity value to be considered.
            Default: 'shape' = 'gauss', 'mean' = 0, 'sigma' = 0.3.
	- i_params: dict, optional. Specifies the parameters needed to define the inclination distribution. If used, the following keys can/must be present:
            - shape: string, required. Desired inclination distribution. Can be uniform in cos(i) ('cos_i') or Gaussian ('gauss')
            - mean: float, optional. Only used if shape = 'gauss'. Mean of the gaussian inclination distribution [rad].
            - sigma: float, optional. Only used if shape = 'gauss'. Standard deviation of the gaussian inclination distribution [rad].
            Default: 'shape' = 'cos_i'.
        - rho_visibility: dict, optional. Correction to be applied to account for the non-complete field-of-view coverage at all separations. If used, the following keys must be present:
            - separation: numpy array. Array of separations [arcsec].
            - visibility: numpy array. Fraction of the circumference, defined by a radius equal to separation, that is within the field of view.   

2) DImode
	Estimates the detection probability map for DI data.
	Parameters:
	- xlim: list, required. Detection limit(s) projected separations (arcsec)
	- ylim: list, required. Minimum detectable mass (in Mjup) at each xlim.
	- lxunit: string, optional. Unit for xlim, default is arcseconds ('as').
		Can also be set to 'au' or 'mas'
	- verbose: if True (default), the code provide the runtime for each target
	- plot: if True (default) a .png file with the detection probability map is produced for each target


"""

class exodmc(object):

    def __init__(self, star_ID, star_dist, **kwargs):

        if isinstance(star_ID,(list)) is not True: star_ID=[star_ID]
        if isinstance(star_dist,(list)) is not True: star_dist=[star_dist]

        self.ID = star_ID
        self.dpc = star_dist # in pc
        self.set_grid(**kwargs)

    def set_grid(self, x_min=0.1, x_max=1000., nx=100, logx=False, y_min=0.1, y_max=100., ny=100, logy=False, ngen=1000, 
                 e_params={'shape': 'gauss', 'mean': 0, 'sigma': 0.3},
                 i_params={'shape': 'cos_i'},
                 rho_visibility=None
                 ):

        self.x_min = x_min
        self.x_max = x_max
        self.x_nsteps = nx
        self.logx = logx

        self.y_min = y_min
        self.y_max = y_max
        self.y_nsteps = ny
        self.logy = logy

        self.norb = ngen
        self.e_dist = e_params['shape']
        self.e_mu = e_params['mean']
        self.e_sigma = e_params['sigma']
        if 'min' in e_params:
            self.e_min = e_params['min']
        else:
            self.e_min = 0
        if 'max' in e_params:
            self.e_max = e_params['max']
        else:
            self.e_max = 1
        
        self.i_dist = i_params['shape']
        if 'mean' in i_params:
            self.i_mu = i_params['mean']
            self.i_sigma = i_params['sigma']

        self.sma = np.linspace(self.x_min, self.x_max, self.x_nsteps)
        if self.logx is True: self.sma = np.logspace(np.log10(self.x_min), np.log10(self.x_max), self.x_nsteps)

        self.M2 = np.linspace(self.y_min, self.y_max, self.y_nsteps)	# range of M2 in Mjup
        if self.logy is True: self.M2 = np.logspace(np.log10(self.y_min), np.log10(self.y_max), self.y_nsteps)

        if self.e_dist=='gauss':
            self.ecc = cropped_gaussian(self.e_mu, self.e_sigma, self.norb)
        elif self.e_dist == 'uniform':
            self.ecc = self.e_min + (self.e_max - self.e_min) * rn.random_sample(self.norb)
        else:
            raise ValueError("e_params['shape'] must be 'gauss' or 'uniform'.")
            
        ecc_expr = (1+self.ecc)/(1-self.ecc)
        self.Omega_Node = rn.random_sample(self.norb)*2.*np.pi # Longitude of node ranges between 0 and 2pi
        self.Omega_Peri = rn.random_sample(self.norb)*2.*np.pi # Longitude of periastron ranges between 0 and 2pi
        self.omega = self.Omega_Peri-self.Omega_Node
        
        if self.i_dist == 'cos_i':
            cosi = 2*rn.random_sample(self.norb) -1.
            self.irad = np.arccos(cosi)
        elif self.i_dist == 'gauss':
            self.irad = np.abs(rn.normal(self.i_mu, self.i_sigma, self.norb))
        else:
            raise ValueError("e_params['shape'] must be 'gauss' or 'cos_i'.")
        
        self.T0 = rn.random_sample(self.norb) # T peri in fraction of period

        # Thiele—Innes elements

        A1=(np.cos(self.Omega_Peri)*np.cos(self.Omega_Node))-(np.sin(self.Omega_Peri)*np.sin(self.Omega_Node)*np.cos(self.irad))
        B1=(np.cos(self.Omega_Peri)*np.sin(self.Omega_Node))+(np.sin(self.Omega_Peri)*np.cos(self.Omega_Node)*np.cos(self.irad))
        F1=(-1*np.sin(self.Omega_Peri)*np.cos(self.Omega_Node))-(np.cos(self.Omega_Peri)*np.sin(self.Omega_Node)*np.cos(self.irad))
        G1=(-1*np.sin(self.Omega_Peri)*np.sin(self.Omega_Node))+(np.cos(self.Omega_Peri)*np.cos(self.Omega_Node)*np.cos(self.irad))

        self.M=rn.random_sample(self.norb)*2*np.pi # mean anomaly
        E0=self.M + np.sin(self.M) * self.ecc + ((self.ecc**2)/2)*np.sin(2*self.M)
        self.E1 = iter_eccentric_anomaly(self.M, self.ecc)
        self.nurad=2*np.arctan((np.sqrt(ecc_expr))*np.tan(self.E1/2.))

        x1=np.cos(self.E1)-self.ecc[np.newaxis,:]
        y1=np.sqrt(1-self.ecc[np.newaxis,:]**2)*np.sin(self.E1)
        y2=(B1[np.newaxis,:]*x1 + G1[np.newaxis,:]*y1)
        x2=(A1[np.newaxis,:]*x1 + F1[np.newaxis,:]*y1)

        # radius vector and projected separation (arcsecs)
        self.rad=(np.sqrt(x2**2 + y2**2)).T
        self.rho=(self.rad[:,np.newaxis]*(self.sma[:,np.newaxis]/self.dpc)).T # projected separation in AU

        if rho_visibility is not None:
            if isinstance(rho_visibility, dict) == False:
                raise TypeError('The argument "rho_visibility" must be a dictionaty with keywords "separation" and "visibility"')
            if ('separation' not in rho_visibility.keys()) | ('visibility' not in rho_visibility.keys()):
                raise TypeError('The argument "rho_visibility" must be a dictionaty with keywords "separation" and "visibility"')

            vis_f = interpolate.interp1d(rho_visibility['separation'], rho_visibility['visibility'], bounds_error=False, fill_value=(1, 0))
            self.rho_visibility = vis_f(self.rho)


    def DImode(self, xlim, ylim, lxunit='as', lyunit='Mjup', verbose=True, plot=True, savefig=True):

            ns=np.size(self.dpc)
            detmap = []
            self.detflag = []
            if isinstance(xlim,(np.ndarray)) is not True: xlim=np.array(xlim)
            if isinstance(ylim,(np.ndarray)) is not True: ylim=np.array(ylim)
            if ns == 1:
                    xlim=[xlim]
                    ylim=[ylim]
            for ll in range(ns):

                    start = time.time()
                    det=np.zeros((self.x_nsteps, self.y_nsteps, self.norb))

                    if lxunit == 'au': xlim[ll] = xlim[ll]/self.dpc[ll]
                    if lxunit == 'mas': xlim[ll] = xlim[ll]/1000.
                        
                    if 'rho_visibility' in self.__dict__.keys():
                        values = self.rho_visibility[ll]
                    else:
                        values = np.ones_like(self.rho[0])
                        

                    s=np.array(np.where(ylim[ll] < self.y_max))
                    if np.size(s) > 1:
                            max_mass=np.nanmax(ylim[ll][s])
                            min_mass=np.nanmin(ylim[ll][s])

                    rlim=np.interp(self.rho[ll], xlim[ll], ylim[ll], right=np.nan,left=np.nan)
                    mm=np.where((self.M2 > min_mass) & (self.M2 < max_mass))
                    for i in range(self.x_nsteps):
                            ff=np.where((self.rho[ll,i].any() < np.min(xlim[ll])) & (self.rho[ll,i].any() > np.max(xlim[ll])))
                                
                            for j in range(self.y_nsteps):
                                    if (self.M2[j] > min_mass and self.M2[j] < max_mass):
                                            index=np.where(rlim[i] < self.M2[j])
                                            if np.size(index) > 1: 
                                                det[i,j,index] = values[i, index] #1 self.rho_visibility[ll][i] self.rho[ll][i]
                            if np.size(ff) > 1: 
                                det[i,j,ff] = 0
                            mm = np.where(self.M2 > max_mass)[0]
                            if np.size(mm) != 0: 
                                det[:,mm,:] = np.tile(det[:,mm[0]-1,:][:,np.newaxis,:], [np.size(mm),1])
                    map=np.sum(det, axis=2)/self.norb
                    detmap.append(map)
                    self.detflag.append(det)
                    end = time.time()
                    hours, rem = divmod(end-start, 3600)
                    minutes, seconds = divmod(rem, 60)
                    if verbose is True: print(self.ID[ll], "time elapsed - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

                    if plot is True:
                            fig = plt.figure(figsize=(8, 6))
                            plt.rc('font', family='serif', size='20')
                            plt.rc('text', usetex=True)
                            plt.rc('xtick', labelsize='15')
                            plt.rc('ytick', labelsize='15')
                            ax = fig.add_axes([0.15, 0.15, 0.8, 0.7])
                            ax.set_xscale('log')
                            ax.set_yscale('log')
                            ax.set_ylabel(" Mass (M$_{Jup}$)")
                            ax.set_xlabel(" Semi major axis (au) ")

                            levels=[10,20,50,70,90,95,99,100]
                            norm = mcolors.Normalize(0, 100)
                            cf0=ax.contourf(self.sma, self.M2, map.T*100, norm=norm, levels=np.arange(0,100.01,0.1), extend='neither', cmap='Blues', antialiased=False, zorder=0)
                            contours = plt.contour(self.sma, self.M2, map.T*100, levels, cmap='bone', zorder=1, linewidths=1)
                            CB = plt.colorbar(cf0,  extend='both', cmap='Blues', ticks=levels)
                            CB.add_lines(contours)
                            CB.set_ticks(levels)
                            CB.ax.set_yticklabels([r"{:.0f}$\%$".format(i) for i in CB.get_ticks()]) # set ticks of your format

                            plt.title(self.ID[ll])
                            if savefig is True: plt.savefig(self.ID[ll]+'_detprob.png', dpi=300)
                            #plt.show()
            return detmap


def iter_eccentric_anomaly(M, e, nIter=100):
    """
    Iteratively numerically estimate eccentric anomaly from mean anomaly
    M (radians) and eccentricity e. 100 iterations is way more than we need,
    but it's not a bottleneck so we are not worried about it.
    Credits: Kellen Lawson
    """
    E = copy.deepcopy(M)
    for _ in range(nIter):
        delta = (M - (E - e*np.sin(E))) / (1 - e*np.cos(E))
        E += delta
    return E

def cropped_gaussian(mu, sigma, norb, limits=[0, 1]):
    """
    Defines a set of nord points, distributed according to a cropped Gaussian defined
    by (mu, sigma) and bound between two boundaries specified by the list "limits".
    Unlike mirroring (e.g., np.abs()) or flooring (np.crop()) function, this function
    retains the Gaussianity of the returned array.
    Credits: Vito Squicciarini
    """

    if (mu < limits[0]) | (mu > limits[1]):
        raise ValueError(r'mu must be between {limits[0]} and {limits[1]}')
    if sigma > 5 * (limits[1] - limits[0]):
        return np.ones(norb)/(limits[1] - limits[0])
    
    Phi = lambda x, mu, sigma: 0.5 * (1 + erf((x-mu)/(np.sqrt(2) * sigma)))
    expected_invalid = Phi(limits[0], mu, sigma) + (limits[1] - Phi(1, mu, sigma))
    scaling_factor = int(2/(1 - expected_invalid)) #doubled to be sure
    a1 = rn.normal(mu, sigma, scaling_factor * norb)
    a1 = a1[(a1 > limits[0]) & (a1 < limits[1])][:norb]
    return a1
