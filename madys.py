import sys
import copy
import warnings
from astropy.utils.exceptions import AstropyWarning
import logging
import numpy as np
from pathlib import Path
import os
from evolution import *
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from astropy.constants import M_jup,M_sun,R_jup,R_sun
import time
from astropy.coordinates import Angle, SkyCoord, ICRS, Galactic, FK4, FK5, Latitude, Longitude,Galactocentric, galactocentric_frame_defaults
from astropy import units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
import csv
from astropy.table import Table, vstack
from astropy.io import ascii
from tabulate import tabulate
import math
import h5py
from astropy.io import fits
Vizier.TIMEOUT = 100000000 # rise the timeout for Vizier
from astropy.table import Table, Column, vstack, hstack, MaskedColumn
from astropy.io import ascii
from tap import (GaiaArchive, TAPVizieR, resolve, QueryStr, timeit)
from json import JSONDecodeError
from astropy.io.votable.exceptions import E19
from astroquery.gaia import Gaia
gaia = GaiaArchive()
vizier = TAPVizieR()

"""
Class: MADYS

Tool for age and mass determination of young stellar and substellar objects. Given a list of stars:
- it retrieves and cross-matches photometry from Gaia and 2MASS
- corrects for interstellar extinction
- assesses the quality of each photometric measurement
- uses reliable photometric data to derive ages and masses (and optionally, other physical parameters too) of individual stars.
MADYS allows a selection of one among 13 theoretical models, many of which with several tunable parameters (metallicity, rotational velocity, etc).
Check the provided manual for additional details on general working, customizable settings and allowed inputs.

MADYS can work in two modes, differing in the shape of input data:
 (mode 1) uses just a list of targets;
 (mode 2) uses a Table containing both the target list and photometric data.
Parameters that are only used in mode 1 will be labeled with (1), and similarly for mode 2. Parameters common to both modes will be labeled with (1,2).

Parameters:
- file (1): string, required. Full path of the file containing target names
- file (2): astropy.table.table.Table, required. Table containing target names and photometric data.
- mock_file (2): string, required. Full path of the non-existing file where the Table would come from if in mode 1. Used to extract the working path and to name the outputs after it.
- id_type (1): string, optional. Type of IDs provided: must be one among 'DR2','EDR3' or 'other'. Default: 'DR2'
- get_phot (1): bool or string, optional. Set to:
        -True: to query the provided IDs;
        -False: to recover photometric data from a previous execution; the filename and path must match the default one.
        -string: full path of the file to load photometric data from. The file should come from a previous execution.
  Default: True.
- save_phot (1): bool, optional. Set to True to create a .csv file with the retrieved photometry. Default: True.
- ext_map (1,2): string, optional. Extinction map used between 'leike' or 'stilism'. Default: 'leike'.
- ebv (1,2): float or numpy array, optional. If set, uses the i-th element of the array as E(B-V) for the i-th star. Default: not set, computes E(B-V) through the map instead.
- max_tmass_q (1): worst 2MASS photometric flag ('ph_qual') still considered reliable. Possible values, ordered by decreasing quality: 'A','B','C','D','E','F','U','X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.
- max_wise_q (1): worst ALLWISE photometric flag ('ph_qual2') still considered reliable. Possible values, ordered by decreasing quality: 'A','B','C','U','Z','X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.

Attributes:
- file: corresponding to either file (1) or mock_file (2).
- path: working path, where all inputs and outputs are present.
- log_file: name of the log_file. Open it for details on the process outcome.
- phot_table (1): Table containing all retrieved data.
- abs_phot: absolute magnitudes in the required filters.
- abs_phot_err: errors on absolute magnitudes in the required filters.
- filters: set of filters, given either by filters of Gaia DR2+EDR3 + 2MASS (1) or by column names (2) 

Methods:

1) get_agemass
Estimates age and mass of individual stars by comparison with isochrone grids.
    Parameters:
    - model: string, required. Chosen model of isochrone grids. Use MADYS.info_models() for further information on the available models.
    - mass_range: list, optional. A two-element list with minimum and maximum mass to consider (M_sun). Default: [0.01,1.4]
    - age_range: list or numpy array, optional. It can be either:
            - a two-element list with minimum and maximum age to consider for the whole sample (Myr);
            - a 1D numpy array, so that the i-th age (Myr) is used as fixed age for the i-th star;
            - a 2D numpy array with 2 columns. The i-th row defines (lower_age,upper_age) range in which one or more solutions are found for the i-th star.
            - a 2D numpy array with 3 columns. The i-th row is used as fixed (mean_age,lower_age,upper_age) for the i-th star.
      Default: [1,1000]
    - n_steps: list, optional. Number of (mass, age) steps of the interpolated grid. Default: [1000,500].
    - verbose: bool, optional. Set to True to save the results in a file. Default: True.
    - feh: float, optional. Selects [Fe/H] of the isochrone set. Default: None (=0.00, solar metallicity).
    - he: float, optional. Selects helium fraction Y of the isochrone set. Default: None (=solar Y).    
    - afe: float, optional. Selects alpha enhancement [a/Fe] of the isochrone set. Default: None (=0.00).
    - v_vcrit: float, optional. Selects rotational velocity of the isochrone set. Default: None (=0.00, non-rotating).
    - fspot: float, optional. Selects fraction of stellar surface covered by star spots. Default: None (=0.00).
    - B: int, optional. Set to 1 to turn on the magnetic field (only for Dartmouth models). Default: 0.
    - ph_cut: float, optional. Maximum  allowed photometric uncertainty [mag]. Default: 0.2.
    - m_unit: string, optional. Unit of measurement of the resulting mass. Choose either 'm_sun' or 'm_jup'. Default: 'm_sun'.
    - phys_param: bool, optional. Set to True to estimate, in addition to mass and age, also radius, effective temperature, surface gravity and luminosity. Default: False.
    Output:
    - dic: a dictionary containing the following keys:
        (case 1: if age_range is a list or a 1D numpy array)
        - ages: numpy array. Final age estimates [Myr].
        - masses: numpy array. Final mass estimates [M_sun or M_jup].
        - ages_err: numpy array. Error on age estimates [Myr].
        - masses_err: numpy array. Error on mass estimates [M_sun or M_jup].
        - ebv: numpy array. Adopted/computed E(B-V), one element per star [mag].
        - radii: numpy array. Final radius estimates [R_sun or R_jup]. Only returned if phys_param=True.
        - radii_err: numpy array. Final radius estimates [R_sun or R_jup]. Only returned if phys_param=True.
        - logg: numpy array. Final surface gravity estimates [log10([cm s-2])]. Only returned if phys_param=True.
        - logg_err: numpy array. Error on surface gravity estimates [log10([cm s-2])]. Only returned if phys_param=True.
        - logL: numpy array. Final luminosity estimates [log10([L_sun])]. Only returned if phys_param=True.
        - logL_err: numpy array. Error on luminosity estimates [log10([L_sun])]. Only returned if phys_param=True.
        - Teff: numpy array. Final effective temperature estimates [K]. Only returned if phys_param=True.
        - Teff_err: numpy array. Error on effective temperature estimates [K]. Only returned if phys_param=True.
       (case 2: if age_range is a 2D numpy array)
        - ages: numpy array. Final age estimates [Myr].
        - masses: numpy array. Final mass estimates [M_sun or M_jup].
        - ages_min: numpy array. Minimum age given by the user [Myr].
        - ages_max: numpy array. Maximum age given by the user [Myr].
        - masses_err_m: numpy array. Difference between 'masses' and masses computed at age=a_min [M_sun or M_jup].
        - masses_err_p: numpy array. Difference between masses computed at age=a_max and 'masses' [M_sun or M_jup].
        - radii: numpy array. Final radius estimates [R_sun or R_jup]. Only returnedif phys_param=True.
        - logg: numpy array. Final surface gravity estimates [log10([cm s-2])]. Only returnedif phys_param=True.
        - logL: numpy array. Final luminosity estimates [log10([L_sun])]. Only returnedif phys_param=True.
        - Teff: numpy array. Final effective temperature estimates [K]. Only returnedif phys_param=True.
        - radii_err_m: numpy array. Difference between 'radii' and radii computed at age=a_min [R_sun or R_jup].
        - radii_err_p: numpy array. Difference between radii computed at age=a_max and 'radii' [R_sun or R_jup].
        - logg_err_m: numpy array. Difference between 'logg' and logg computed at age=a_min.
        - logg_err_p: numpy array. Difference between logg computed at age=a_max and 'logg'.
        - logL_err_m: numpy array. Difference between 'logL' and logL computed at age=a_min.
        - logL_err_p: numpy array. Difference between logL computed at age=a_max and 'logL'.
        - Teff_err_m: numpy array. Difference between 'Teff' and Teff computed at age=a_min [K].
        - Teff_err_p: numpy array. Difference between Teff computed at age=a_max and 'Teff' [K].
    
2) CMD
Draws a color-magnitude diagram (CMD) containing both the measured photometry and a set of theoretical isochrones.
    Parameters:
    - col: string, required. Quantity to be plotted along the x axis (e.g.: 'G' or 'G-K')
    - mag: string, required. Quantity to be plotted along the y axis (e.g.: 'G' or 'G-K')
    - model: string, required. Chosen model of isochrone grids. Use MADYS.info_models() for further information on the available models.
    - mass_range: list, optional. A two-element list with minimum and maximum mass to consider (M_sun). Default: [0.01,1.4]
    - age_range: list or numpy array, optional. It can be either:
            - a two-element list with minimum and maximum age to consider for the whole sample (Myr);
            - a 1D numpy array, so that the i-th age (Myr) is used as fixed age for the i-th star;
            - a 2D numpy array with 3 columns. The i-th row is used as fixed (mean_age,lower_age,upper_age) for the i-th star.
      Default: [1,1000]
    - n_steps: list, optional. Number of (mass, age) steps of the interpolated grid. Default: [1000,500].
    - feh: float, optional. Selects [Fe/H] of the isochrone set. Default: None (=0.00, solar metallicity).
    - afe: float, optional. Selects alpha enhancement [a/Fe] of the isochrone set. Default: None (=0.00).
    - v_vcrit: float, optional. Selects rotational velocity of the isochrone set. Default: None (=0.00, non-rotating).
    - fspot: float, optional. Selects fraction of stellar surface covered by star spots. Default: None (=0.00).
    - B: int, optional. Set to 1 to turn on the magnetic field (only for Dartmouth models). Default: 0.
    - he: float, optional. Selects helium fraction Y of the isochrone set. Default: None (=solar Y).    
    - ids: list or numpy array of integers, optional. Array of indices, selects the subset of input data to be drawn.
    - plot_ages: numpy array or bool, optional. It can be either:
            - a numpy array containing the ages (in Myr) of the isochrones to be plotted;
            - False, not to plot any isochrone.
      Default: [1,3,5,10,20,30,100,200,500,1000].
    - plot_masses: numpy array or bool, optional. It can be either:
            - a numpy array containing the masses (in M_sun) of the tracks to be plotted.             - a numpy array containing the ages (in Myr) of the isochrones to be plotted;
            - False, not to plot any track.
      Default: 0.1,0.3,0.5,0.7,0.85,1.0,1.3,2].
    - stick_to_points: bool, optional. Zooms the view on data points so that the axes are defined as [min(data)-0.1,max(data)+0.1]. Default: False.
    - x_range: list, optional. A two-element list with minimum and maximum value for the x axis. Similar to pyplot's xlim.
    - y_range: list, optional. A two-element list with minimum and maximum value for the y axis. Similar to pyplot's ylim.
    - groups: list or numpy array of integers, optional. Draws different groups of stars in different colors. The i-th element is a number, indicating to which group the i-th star belongs. Default: None.
    - group_list: list or numpy array of strings, optional. Names of the groups defined by the 'groups' keyword. No. of elements must match the no. of groups. Default: None.
    - label_points: bool, optional. Draws a label next to each point, specifying its row index. Default: True.
    - tofile: bool or string, optional. If True, saves the output to as .png image. To change the file name, provide a string as full path of the output file. Default: False.

3) interstellar_ext
Computes the reddening/extinction in a custom band, given the position of a star.
No parameter is strictly required, but at one between RA and l, one between dec and b, one between par and d must be supplied.
    Parameters:
    - ra: float or numpy array, optional. Right ascension of the star(s) [deg].
    - dec: float or numpy array, optional. Declination of the star(s) [deg].
    - l: float or numpy array, optional. Galactic longitude of the star(s) [deg].
    - b: float or numpy array, optional. Galactic latitude of the star(s) [deg].
    - par: float or numpy array, optional. Parallax of the star(s) [mas].
    - d: float or numpy array, optional. Distance of the star(s) [pc].
    - ext_map: string, optional. Extinction map to be used: must be 'leike' or 'stilism'. Default: 'leike'.
    - color: string, optional. Band in which the reddening/extinction is desired. Default: B-V.
    - error: bool, optional. Computes also the uncertainty on the estimate. Default: False.
    Output:
    - ext: float or numpy array. Best estimate of reddening/extinction for each star.
    (only if error==True:)
    (- err: float or numpy array. Uncertainty on the best estimate of reddening/extinction for each star.)

4) app_to_abs_mag
Turns one or more apparent magnitude(s) into absolute magnitude(s).
    Parameters:
    - app_mag: float, list or numpy array (1D or 2D), required. Input apparent magnitude(s).
      If a 2D numpy array, each row corresponds to a star, each row to a certain band.
    - parallax: float, list or 1D numpy array, required. Input parallax(es).
    - app_mag_error: float, list or numpy array (1D or 2D), optional. Error on apparent magnitude(s); no error estimation if ==None. Default: None.
    - parallax_error: float, list or 1D numpy array, optional. Error on parallax(es); no error estimation if ==None. Default: None.
    - ebv: float, list or 1D numpy array, optional. E(B-V) affecting input magnitude(s); assumed null if ==None. Default: None.
    - filters: list or 1D numpy array, optional. Names of the filters; length must equal no. of columns of app_mag. Default: None.
    Output:
    - abs_mag: float or numpy array. Absolute magnitudes, same shape as app_mag.
    (only if app_mag_error!=None and parallax_error!=None)
    (- abs_err: float or numpy array. Propagated uncertainty on abs_mag.)
    
5) info_models
Prints info about available models for MADYS.
Informs about 1) the calling sequence; 2) the customizable parameters; 3) age and mass range; 4) adopted solar metallicity and helium content; 5) literature reference.
    Parameters:
    - model: string, optional. Use it to print info about a specific model. If None, prints info about all the available models. Default: None.

6) plot_2D_ext
Plots the integrated absorption in a given region of the sky, by creating a 2D projection at constant distance of an extinction map.
No parameter is strictly required, but at one between RA and l, one between dec and b, one between par and d must be supplied.
    Parameters:
    - ra: 2-element list, optional. Minimum and maximum right ascension of the sky region [deg].
    - dec: 2-element list, optional. Minimum and maximum declination of the sky region [deg].
    - l: 2-element list, optional. Minimum and maximum galactic longitude of the sky region [deg].
    - b: 2-element list, optional. Minimum and maximum galactic latitude of the sky region [deg].
    - par: float or int, optional. Parallax corresponding to the depth of the integration [mas].
    - d: float or int, optional. Maximum distance, i.e. depth of the integration [pc].
    - ext_map: string, optional. Extinction map to be used: must be 'leike' or 'stilism'. Default: 'leike'.
    - color: string, optional. Band in which the reddening/extinction is desired. Default: B-V.
    - n: int, optional. No. of steps along each axis. The final grid will have size [n*n]. Default: 50.
    - reverse_xaxis: bool, optional. If True, reverses the x axis in the plot. Default: False.
    - reverse_yaxis: bool, optional. If True, reverses the x axis in the plot. Default: False.
    - tofile: string, optional. Full path of the output file where the plot will be saved to. Default: None.
    Output: no output is returned, but the plot is shown in the current window.

7) info_filters
Prints info about filters/physical quantity in MADYS.
Informs about 1) the name of the quantity and its basic reference; 2) the models in which the quantity is available.
    Parameters:
    - filt: string, optional. Use it to print info about a specific quantity. If None, prints info about all the available quantities. Default: None.
    - model: string, optional. If provided, returns True if the quantity is available in the model, False otherwise. Default: None.

"""

class MADYS(object):
    filt={'bessell':{'B':0.4525,'V':0.5525,'U':0.3656,'Ux':0.3656,'Bx':0.4537,'R':0.6535,'I':0.8028,
                     'B_J':1.22,'B_H':1.63,'B_K':2.19,'L':3.45,'Lp':3.80,'M':4.75,'N':10.4},
          'gaia':{'G':0.6419,'G2':0.6419,'Gbp':0.5387,'Gbp2':0.5387,'Grp':0.7667,'Grp2':0.7667},
          '2mass':{'J':1.2345,'H':1.6393,'K':2.1757},
          'panstarrs':{'g':0.4957,'r':0.6211,'i':0.7522,'z':0.8671,'y':0.9707},
          'wise':{'W1':3.3172,'W2':4.5501,'W3':11.7281,'W4':22.0883},
          'hst':{'H_F110W':1.11697,'H_F160W':1.52583,'H_F090M':0.903471,'H_F165M':1.648026,
                 'H_F187W':1.871407,'H_F205W':2.063609,'H_F207M':2.082083,'H_F222M':2.217527,
                 'H_F237M':2.369093,'H_F253M':0.254890,'H_F300W':0.298519,'H_F336W':0.334429,
                 'H_F346M':0.347452,'H_F439W':0.431175,'H_F555W':0.544294,'H_F606W':0.600127,
                 'H_F675W':0.671771,'H_F785LP':0.868737,'H_F814W':0.799594},
          'jwst_miri_c':{'MIRI_c_F1065C':10.562839,'MIRI_c_F1140C':11.310303,'MIRI_c_F1550C':15.516774,'MIRI_c_F2300C':22.644644},
          'jwst_miri_p':{'MIRI_p_F560W':5.635257,'MIRI_p_F770W':7.639324,'MIRI_p_F1000W':9.953116,'MIRI_p_F1130W':11.308501,
                       'MIRI_p_F1280W':12.810137,'MIRI_p_F1500W':15.063507,'MIRI_p_F1800W':17.983723,'MIRI_p_F2100W':20.795006,
                       'MIRI_p_F2550W':25.364004},
          'jwst_nircam_c210':{'NIRCAM_c210_F182M':1.83932,'NIRCAM_c210_F187N':1.8740,'NIRCAM_c210_F200W':1.9694,
                             'NIRCAM_c210_F210M':2.09152,'NIRCAM_c210_F212N':2.1210},         
          'jwst_nircam_c335':{'NIRCAM_c335_F250M':2.50085,'NIRCAM_c335_F300M':2.98178,'NIRCAM_c335_F322W2':3.07505,
                              'NIRCAM_c335_F335M':3.35376,'NIRCAM_c335_F356W':3.52897,'NIRCAM_c335_F360M':3.61512,
                              'NIRCAM_c335_F410M':4.0720,'NIRCAM_c335_F430M':4.27752,'NIRCAM_c335_F444W':4.34419,
                              'NIRCAM_c335_F460M':4.62676,'NIRCAM_c335_F480M':4.81229},
          'jwst_nircam_c430':{'NIRCAM_c430_F250M':2.50085,'NIRCAM_c430_F300M':2.98178,'NIRCAM_c430_F322W2':3.07505,
                              'NIRCAM_c430_F335M':3.35376,'NIRCAM_c430_F356W':3.52897,'NIRCAM_c430_F360M':3.61512,
                              'NIRCAM_c430_F410M':4.0720,'NIRCAM_c430_F430M':4.27752,'NIRCAM_c430_F444W':4.34419,
                              'NIRCAM_c430_F460M':4.62676,'NIRCAM_c430_F480M':4.81229},
          'jwst_nircam_cswb':{'NIRCAM_cSWB_F182M': 1.83932,'NIRCAM_cSWB_F187N': 1.8740,'NIRCAM_cSWB_F200W':1.96947,
                              'NIRCAM_cSWB_F210M':2.09152,'NIRCAM_cSWB_F212N':2.1210},
          'jwst_nircam_clwb':{'NIRCAM_cLWB_F250M':2.50085,'NIRCAM_cLWB_F277W':2.72889,'NIRCAM_cLWB_F300M':2.98178,
                              'NIRCAM_cLWB_F335M':3.35376,'NIRCAM_cLWB_F356W':3.52897,'NIRCAM_cLWB_F360M':3.61512,
                              'NIRCAM_cLWB_F410M':4.0720,'NIRCAM_cLWB_F430M':4.27752,'NIRCAM_cLWB_F444W':4.34419,
                              'NIRCAM_cLWB_F460M':4.62676,'NIRCAM_cLWB_F480M':4.81229},
          'jwst_nircam_pa':{'NIRCAM_p_F070Wa':0.70401,'NIRCAM_p_F090Wa':0.90045,'NIRCAM_p_F115Wa':1.15039,
                            'NIRCAM_p_F140Ma':1.40402,'NIRCAM_p_F150Wa':1.49409,'NIRCAM_p_F150W2a':1.54231,
                            'NIRCAM_p_F162Ma':1.62491,'NIRCAM_p_F164Na':1.6450,'NIRCAM_p_F182Ma':1.83932,
                            'NIRCAM_p_F187Na':1.8740,'NIRCAM_p_F200Wa':1.96947,'NIRCAM_p_F210Ma':2.09152,
                            'NIRCAM_p_F212Na':2.1210,'NIRCAM_p_F250Ma':2.50085,'NIRCAM_p_F277Wa':2.72889,
                            'NIRCAM_p_F300Ma':2.98178,'NIRCAM_p_F322W2a':3.07505,'NIRCAM_p_F323Na':3.2370,
                            'NIRCAM_p_F335Ma':3.35376,'NIRCAM_p_F356Wa':3.52897,'NIRCAM_p_F360Ma':3.61512,
                            'NIRCAM_p_F405Na':4.0520,'NIRCAM_p_F410Ma':4.0720,'NIRCAM_p_F430Ma':4.27752,
                            'NIRCAM_p_F444Wa':4.34419,'NIRCAM_p_F460Ma':4.62676,'NIRCAM_p_F466Na':4.6540,
                            'NIRCAM_p_F470Na':4.7080,'NIRCAM_p_F480Ma':4.81229},
          'jwst_nircam_pab':{'NIRCAM_p_F070Wab':0.70401,'NIRCAM_p_F090Wab':0.90045,'NIRCAM_p_F115Wab':1.15039,
                             'NIRCAM_p_F140Mab':1.40402,'NIRCAM_p_F150Wab':1.49409,'NIRCAM_p_F150W2ab':1.54231,
                             'NIRCAM_p_F162Mab':1.62491,'NIRCAM_p_F164Nab':1.6450,'NIRCAM_p_F182Mab':1.83932,
                             'NIRCAM_p_F187Nab':1.8740,'NIRCAM_p_F200Wab':1.96947,'NIRCAM_p_F210Mab':2.09152,
                             'NIRCAM_p_F212Nab':2.1210,'NIRCAM_p_F250Mab':2.50085,'NIRCAM_p_F277Wab':2.72889,
                             'NIRCAM_p_F300Mab':2.98178,'NIRCAM_p_F322W2ab':3.07505,'NIRCAM_p_F323Nab':3.2370,
                             'NIRCAM_p_F335Mab':3.35376,'NIRCAM_p_F356Wab':3.52897,'NIRCAM_p_F360Mab':3.61512,
                             'NIRCAM_p_F405Nab':4.0520,'NIRCAM_p_F410Mab':4.0720,'NIRCAM_p_F430Mab':4.27752,
                             'NIRCAM_p_F444Wab':4.34419,'NIRCAM_p_F460Mab':4.62676,'NIRCAM_p_F466Nab':4.6540,
                             'NIRCAM_p_F470Nab':4.7080,'NIRCAM_p_F480Mab':4.81229},
          'jwst_nircam_pb':{'NIRCAM_p_F070Wb':0.70401,'NIRCAM_p_F090Wb':0.90045,'NIRCAM_p_F115Wb':1.15039,
                            'NIRCAM_p_F140Mb':1.40402,'NIRCAM_p_F150Wb':1.49409,'NIRCAM_p_F150W2b':1.54231,
                            'NIRCAM_p_F162Mb':1.62491,'NIRCAM_p_F164Nb':1.6450,'NIRCAM_p_F182Mb':1.83932,
                            'NIRCAM_p_F187Nb':1.8740,'NIRCAM_p_F200Wb':1.96947,'NIRCAM_p_F210Mb':2.09152,
                            'NIRCAM_p_F212Nb':2.1210,'NIRCAM_p_F250Mb':2.50085,'NIRCAM_p_F277Wb':2.72889,
                            'NIRCAM_p_F300Mb':2.98178,'NIRCAM_p_F322W2b':3.07505,'NIRCAM_p_F323Nb':3.2370,
                            'NIRCAM_p_F335Mb':3.35376,'NIRCAM_p_F356Wb':3.52897,'NIRCAM_p_F360Mb':3.61512,
                            'NIRCAM_p_F405Nb':4.0520,'NIRCAM_p_F410Mb':4.0720,'NIRCAM_p_F430Mb':4.27752,
                            'NIRCAM_p_F444Wb':4.34419,'NIRCAM_p_F460Mb':4.62676,'NIRCAM_p_F466Nb':4.6540,
                            'NIRCAM_p_F470Nb':4.7080,'NIRCAM_p_F480Mb':4.81229},
          'jwst_niriss_c':{'NIRISS_c_F277W':2.72889,'NIRISS_c_F380M':3.828358,'NIRISS_c_F430M':4.27752,'NIRISS_c_F480M':4.81229},
          'jwst_niriss_p':{'NIRISS_p_F090W':0.90045,'NIRISS_p_F115W':1.15039,'NIRISS_p_F140M':1.40402,'NIRISS_p_F150W':1.49409,
                         'NIRISS_p_F200W':1.96947,'NIRISS_p_F158M':1.586646,'NIRISS_p_F277W':2.72889,'NIRISS_p_F356W':3.52897,
                         'NIRISS_p_F380M':3.828358,'NIRISS_p_F430M':4.27752,'NIRISS_p_F444W':4.34419,'NIRISS_p_F480M':4.81229},
          'cfht':{'CFHT_H':1.624354,'CFHT_J':1.251872,'CFHT_K':2.143400,'CFHT_Y':1.024151,'CFHT_Z':0.879313,'CFHT_CH4ON':1.690964,'CFHT_CH4OFF':1.588350},
          'mko':{'MKO_H':1.622935,'MKO_J':1.245693,'MKO_K':2.193904,'MKO_Lp':3.757162,'MKO_Mp':4.683029,'MKO_Y':1.02},
          'ukirt':{'UKIDSS_h':1.6313,'UKIDSS_j':1.2483,'UKIDSS_k':2.2010,'UKIDSS_y':1.0305,'UKIDSS_z':0.8817},
          'spitzer':{'IRAC1':3.537841,'IRAC2':4.478049,'IRAC3':5.696177,'IRAC4':7.797839,'MIPS24':23.593461,'MIPS70':70.890817,
                     'MIPS160':155.0,'IRSblue':15.766273,'IRSred':22.476444},
          'sphere':{'SPH_H':1.625,'SPH_H2':1.593,'SPH_H3':1.667,'SPH_H4':1.733,'SPH_J':1.245,'SPH_J2':1.190,'SPH_J3':1.273,'SPH_K':2.182,
                    'SPH_K1':2.110,'SPH_K2':2.251,'SPH_NDH':1.593,'SPH_Y':1.043,'SPH_Y2':1.022,'SPH_Y3':1.076},
          'kepler':{'D51':0.510,'Kp':0.630335},
          'hipparcos':{'Hp':0.502506},
          'tess':{'I_c':0.769758},
          'skymapper':{'SM_g':0.507519,'SM_i':0.776798,'SM_r':0.613844,'SM_u':0.349336,'SM_v':0.383593,'SM_z':0.914599},
          'tycho':{'T_B':0.4280,'T_V':0.5340},      
          'hr':{'logg':np.nan,'logL':np.nan,'radius':np.nan,'Teff':np.nan}      
         } #filters, surveys, mean wavelength
    
    def __init__(self, file, **kwargs):
        if isinstance(file,Table): self.file = kwargs['mock_file']
        else: self.file = file
        self.path = os.path.dirname(self.file)     #working path        
        self.sample_name_ext()

        if isinstance(file,Table):            
            col0=file.colnames
            kin=np.array(['parallax','parallax_err','ra','dec','name'])
            col=np.setdiff1d(np.unique(np.char.replace(col0,'_err','')),kin)
            col_err=np.array([i+'_err' for i in col])
            self.filters=np.array(col)
            self.GaiaID=file['name']
        else:
            self.surveys = kwargs['surveys'] if 'surveys' in kwargs else ['gaia','2mass']
            if 'gaia' not in self.surveys: self.surveys.append('gaia')
            filters=[]
            for i in range(len(self.surveys)): filters.extend(list(MADYS.filt[self.surveys[i]]))
            self.filters=np.array(filters) # self.filters=np.array(['G','Gbp','Grp','G2','Gbp2','Grp2','J','H','K'])
            self.__id_type = kwargs['id_type'] if 'id_type' in kwargs else 'DR2'            
            get_phot = kwargs['get_phot'] if 'get_phot' in kwargs else True
            save_phot = kwargs['save_phot'] if 'save_phot' in kwargs else True
            
            self.ID=self.read_IDs()
            if self.__id_type!='other': self.GaiaID = self.ID
            else: self.get_gaia()
        
        logging.shutdown() 
        self.log_file = Path(self.path) / (self.__sample_name+'_log.txt')
        if os.path.exists(self.log_file): os.remove(self.log_file)
        self.__logger = MADYS.setup_custom_logger('madys',self.log_file)
        ext_map = kwargs['ext_map'] if 'ext_map' in kwargs else 'leike'
        
        if isinstance(file,Table):
            n=len(col)
            nst=len(file)
            self.abs_phot=np.full([nst,n],np.nan)
            self.abs_phot_err=np.full([nst,n],np.nan)
            for i in range(n):
                self.abs_phot[:,i]=file[col[i]]               
                self.abs_phot_err[:,i]=file[col_err[i]]
                
            self.__logger.info('Program started')
            self.__logger.info('Input type: custom table')
            self.__logger.info('Filters required: '+','.join(self.filters))
            self.__logger.info('No. of stars: '+str(nst))
                
            self.ebv=np.zeros(len(file))
            if 'ebv' in kwargs:
                self.ebv=kwargs['ebv']
                self.__logger.info('Extinction type: provided by the user')
            elif ('ra' in col0) & ('dec' in col0) & ('parallax' in col0):
                self.par=file['parallax']
                self.ebv=MADYS.interstellar_ext(ra=file['ra'],dec=file['dec'],par=self.par,ext_map=ext_map,logger=self.__logger)
                self.__logger.info('Extinction type: computed using '+ext_map+' extinction map')
            if 'parallax' in col0:
                self.par=file['parallax']
                self.par_err=file['parallax_err']
                self.app_phot=copy.deepcopy(self.abs_phot)
                self.app_phot_err=copy.deepcopy(self.abs_phot_err)
                self.abs_phot,self.abs_phot_err=MADYS.app_to_abs_mag(self.abs_phot,self.par,app_mag_error=self.abs_phot_err,parallax_error=self.par_err,ebv=self.ebv,filters=col)
                self.__logger.info('Input photometry: apparent, converted to absolute.')
            else:
                self.__logger.info('Input photometry: no parallax provided, assumed absolute.')
                
        else:            
            nst=len(self.ID)                

            self.__logger.info('Program started')
            self.__logger.info('Input file: list of IDs')
            
            self.__logger.info('No. of stars: '+str(nst))
            self.__logger.info('Looking for photometry in the surveys: '+','.join(['gaia','2mass']))
                
            if get_phot==True: 
                self.__logger.info('Starting data query...')
                self.get_phot(save_phot)
                self.__logger.info('Data query: ended.')                
            elif get_phot==False:
                filename=os.path.join(self.path,(self.__sample_name+'_phot_table.csv'))
                if os.path.exists(filename):
                    self.phot_table=ascii.read(filename, format='csv')
                    self.__logger.info('Data recovered from a previous execution. File: '+filename)
                else: 
                    self.__logger.warning('get_phot is set to False but the file '+filename+' was not found. Program ended.')
                    raise ValueError('get_phot is set to False but the file '+filename+' was not found. Set get_phot=True to query the provided IDs, or get_phot=full_file_path to recover them from an input file.')
            else:
                filename=get_phot
                if os.path.exists(filename):
                    self.phot_table=ascii.read(filename, format='csv')
                    self.__logger.info('Data recovered from a previous execution. File: '+filename)
                else: 
                    self.__logger.warning('The provided file '+filename+' was not found. Program ended.')
                    raise ValueError('The provided file file '+filename+' was not found. Set get_phot=True to query the provided IDs, or check the file name.')
                    
            self.good_phot=self.check_phot(**kwargs)
            
            
            nf=len(self.filters) #self.filters

            query_keys={'G':'edr3_gmag_corr','Gbp':'edr3_phot_bp_mean_mag','Grp':'edr3_phot_rp_mean_mag','G2':'dr2_phot_g_mean_mag',
                        'Gbp2':'dr2_phot_bp_mean_mag','Grp2':'dr2_phot_rp_mean_mag','J':'j_m','H':'h_m','K':'ks_m',
                        'W1':'w1mpro','W2':'w2mpro','W3':'w3mpro','W4':'w4mpro',
                        'G_err':'edr3_phot_g_mean_mag_error','Gbp_err':'edr3_phot_bp_mean_mag_error','Grp_err':'edr3_phot_rp_mean_mag_error',
                        'G2_err':'dr2_g_mag_error','Gbp2_err':'dr2_bp_mag_error','Grp2_err':'dr2_rp_mag_error',
                        'J_err':'j_msigcom','H_err':'h_msigcom','K_err':'ks_msigcom',
                        'W1_err':'w1mpro_error','W2_err':'w2mpro_error','W3_err':'w3mpro_error','W4_err':'w4mpro_error',
                        'g':'ps1_g','r':'ps1_r','i':'ps1_i','z':'ps1_z','y':'ps1_y','g_err':'ps1_g_error',
                        'r_err':'ps1_r_error','i_err':'ps1_i_error','z_err':'ps1_z_error','y_err':'ps1_y_error'}

            phot=np.full([nst,nf],np.nan)
            phot_err=np.full([nst,nf],np.nan)
            for i in range(nf):
                phot[:,i]=self.good_phot[query_keys[self.filters[i]]].filled(np.nan)
                phot_err[:,i]=self.good_phot[query_keys[self.filters[i]+'_err']].filled(np.nan)

            self.app_phot=phot
            self.app_phot_err=phot_err
            ra=np.array(self.good_phot['ra'].filled(np.nan))
            dec=np.array(self.good_phot['dec'].filled(np.nan))
            par=np.array(self.good_phot['edr3_parallax'].filled(np.nan))
            par_err=np.array(self.good_phot['edr3_parallax_error'].filled(np.nan))
            u,=np.where(np.isnan(par))
            if len(u)>0:
                par2=np.array(self.good_phot['dr2_parallax'].filled(np.nan))
                par_err2=np.array(self.good_phot['dr2_parallax_error'].filled(np.nan))
                u2,=np.where((np.isnan(par)) & (np.isnan(par2)==False))
                par=np.where(np.isnan(par),par2,par)
                par_err=np.where(np.isnan(par_err),par_err2,par_err)
                for i in range(len(u2)):
                    self.__logger.info('Invalid parallax in Gaia EDR3 for star '+str(self.ID[u2[i]][0])+', using DR2 instead')                    
            if 'ebv' in kwargs:
                self.ebv=kwargs['ebv']
                self.__logger.info('Extinction type: provided by the user')
            else:
                tt0=time.perf_counter()  #CANCELLA
                self.ebv=self.interstellar_ext(ra=ra,dec=dec,par=par,ext_map=ext_map,logger=self.__logger)
                tt1=time.perf_counter()  #CANCELLA
                print('Time for the computation of extinctions: ',tt1-tt0,' s') #CANCELLA
                self.__logger.info('Extinction type: computed using '+ext_map+' extinction map')
            self.abs_phot,self.abs_phot_err=self.app_to_abs_mag(self.app_phot,par,app_mag_error=self.app_phot_err,parallax_error=par_err,ebv=self.ebv,filters=self.filters)
            self.__logger.info('Input photometry: apparent, converted to absolute')
            self.par=par #OCCHIO, HIDE ATTRIBUTE
            self.par_err=par_err #OCCHIO, HIDE ATTRIBUTE

        logging.shutdown() 
        
    def sample_name_ext(self):
        sample_name=os.path.split(self.file)[1]
        i=0
        while sample_name[i]!='.': i=i+1
        self.__ext=sample_name[i:]
        self.__sample_name=sample_name[:i]
        
    def read_IDs(self):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if self.__ext=='.csv':
                IDtab = pd.read_csv(self.file)
            else:
                IDtab = pd.read_csv(self.file, skipinitialspace=True,sep='\s{2,}',engine='python')

            col=IDtab.columns

            if len(col)>1:
                if 'id' in col: ID = IDtab['id']
                elif 'ID' in col: ID = IDtab['ID']
                elif 'source_id' in col: ID = IDtab['source_id']
                else: raise ValueError('The columns with IDs was not found! Check that a "ID" or "source_id" column is present, and try again.')    
            else: ID=IDtab[col[0]]

            if self.__id_type=='DR2':
                for i in range(len(ID)): 
                    if 'Gaia DR2' not in str(ID.loc[i]):
                        ID.loc[i]='Gaia DR2 '+str(ID.loc[i])
            elif self.__id_type=='EDR3':
                for i in range(len(ID)):
                    if 'Gaia EDR3' not in str(ID.loc[i]):
                        ID.loc[i]='Gaia EDR3 '+str(ID.loc[i])

            if isinstance(ID,pd.Series): ID=ID.to_frame()
            ID=Table.from_pandas(ID)
            ID.rename_column(ID.colnames[0], 'ID')
        
        return ID
    
    def get_gaia(self):
        ns=len(self.ID)
        self.GaiaID=['']*ns
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(ns):
                found=False
                res=Simbad.query_objectids(self.ID[i])
                for rr in res:
                    if str(rr[0]).startswith('Gaia DR2'): 
                        self.GaiaID[i]=str(rr[0])#
                        found=True
                        break
                if found==False: self.GaiaID[i]='Gaia DR2 0000'

    def list_chunk(self,ind=None,key_name=None,id_list=None,equality='=',quote_mark=False):
        query_list=''
        
        print(ind)
        print(key_name)
        print(id_list)
        print(equality)
        print(quote_mark)
        
        if (type(key_name)==type(None)) & (type(id_list)==type(None)):
            id_str = 'Gaia EDR3 ' if self.__id_type=='EDR3' else 'Gaia DR2 '
            id_sea = 'edr3.source_id' if self.__id_type=='EDR3' else 'dr2xmatch.dr2_source_id'    
            if type(ind)==type(None): ind=len(self.GaiaID)
            if quote_mark:
                if isinstance(ind,np.ndarray):
                    for i in ind:
                        id=str(self.GaiaID[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+" '"+id+"' OR "
                else:        
                    for i in range(ind):
                        id=str(self.GaiaID[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality++" '"+id+"' OR "
            else:
                if isinstance(ind,np.ndarray):
                    for i in ind:
                        id=str(self.GaiaID[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+' '+id+' OR '
                else:        
                    for i in range(ind):
                        id=str(self.GaiaID[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+' '+id+' OR '
        elif type(key_name)==type(None):
            id_str = 'Gaia EDR3 ' if self.__id_type=='EDR3' else 'Gaia DR2 '
            id_sea = 'edr3.source_id' if self.__id_type=='EDR3' else 'dr2xmatch.dr2_source_id'    
            if type(ind)==type(None): ind=len(id_list)
            if quote_mark:
                if isinstance(ind,np.ndarray):
                    for i in ind:
                        id=str(id_list[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+" '"+id+"' OR "
                else:        
                    for i in range(ind):
                        id=str(id_list[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+" '"+id+"' OR "
            else:
                if isinstance(ind,np.ndarray):
                    for i in ind:
                        id=str(id_list[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+' '+id+' OR '
                else:        
                    for i in range(ind):
                        id=str(id_list[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+' '+id+' OR '
        else:
            id_sea = key_name
            if type(ind)==type(None): ind=len(id_list)
            if quote_mark:
                if isinstance(ind,np.ndarray):
                    for i in ind:
                        id=str(id_list[i])
                        query_list+=id_sea+' '+equality+" '"+id+"' OR "
                else:        
                    for i in range(ind):
                        id=str(self.GaiaID[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+" '"+id+"' OR "
            else:
                if isinstance(ind,np.ndarray):
                    for i in ind:
                        id=str(id_list[i])
                        query_list+=id_sea+' '+equality+' '+id+' OR '
                else:        
                    for i in range(ind):
                        id=str(self.GaiaID[i]).split(id_str)[1]
                        query_list+=id_sea+' '+equality+' '+id+' OR '
                
        query_list=query_list[:-4]
        return query_list

    def query_string(self,query_list,surveys=None):
        qstr1=''
        qstr2=''
        qstr3=''
        surveys = surveys if type(surveys)!=type(None) else self.surveys

        if 'gaia' in surveys:
            qstr1+='    edr3.designation as edr3_id, dr2.designation as dr2_id, '
            qstr2+="""
            edr3.ra as ra, edr3.dec as dec,
            edr3.ref_epoch as edr3_epoch, edr3.parallax as edr3_parallax,
            edr3.parallax_error as edr3_parallax_error, edr3.parallax_over_error as edr3_parallax_over_error,
            edr3.pmra as edr3_pmra, edr3.pmra_error as edr3_pmra_error,
            edr3.pmdec as edr3_pmdec, edr3.pmdec_error as edr3_pmdec_error,
            edr3.ra_dec_corr as edr3_ra_dec_corr, edr3.ra_parallax_corr as edr3_ra_parallax_corr,
            edr3.ra_pmra_corr as edr3_ra_pmra_corr, edr3.ra_pmdec_corr as edr3_ra_pmdec_corr,
            edr3.dec_parallax_corr as edr3_dec_parallax_corr,
            edr3.dec_pmra_corr as edr3_dec_pmra_corr, edr3.dec_pmdec_corr as edr3_dec_pmdec_corr,
            edr3.parallax_pmra_corr as edr3_parallax_pmra_corr, edr3.parallax_pmdec_corr as edr3_parallax_pmdec_corr,
            edr3.pmra_pmdec_corr as edr3_pmra_pmdec_corr, edr3.phot_g_mean_mag as edr3_phot_g_mean_mag,
            edr3.phot_g_mean_flux as edr3_phot_g_mean_flux, edr3.phot_g_mean_flux_error as edr3_phot_g_mean_flux_error,
            edr3.phot_bp_mean_flux as edr3_phot_bp_mean_flux, edr3.phot_bp_mean_flux_error as edr3_phot_bp_mean_flux_error,
            edr3.phot_bp_mean_mag as edr3_phot_bp_mean_mag,
            edr3.phot_rp_mean_flux as edr3_phot_rp_mean_flux, edr3.phot_rp_mean_flux_error as edr3_phot_rp_mean_flux_error,
            edr3.phot_rp_mean_mag as edr3_phot_rp_mean_mag,
            edr3.bp_rp as edr3_bp_rp, edr3.phot_bp_rp_excess_factor as edr3_phot_bp_rp_excess_factor,
            edr3.ruwe as edr3_ruwe, edr3.astrometric_params_solved as edr3_astrometric_params_solved,
            dr2.ref_epoch as dr2_epoch, dr2.ra as dr2_ra, dr2.dec as dr2_dec,
            dr2.parallax as dr2_parallax,
            dr2.parallax_error as dr2_parallax_error, dr2.parallax_over_error as dr2_parallax_over_error,
            dr2.pmra as dr2_pmra, dr2.pmra_error as dr2_pmra_error,
            dr2.pmdec as dr2_pmdec, dr2.pmdec_error as dr2_pmdec_error,
            dr2.ra_dec_corr as dr2_ra_dec_corr, dr2.ra_parallax_corr as dr2_ra_parallax_corr,
            dr2.ra_pmra_corr as dr2_ra_pmra_corr, dr2.ra_pmdec_corr as dr2_ra_pmdec_corr,
            dr2.dec_parallax_corr as dr2_dec_parallax_corr,
            dr2.dec_pmra_corr as dr2_dec_pmra_corr, dr2.dec_pmdec_corr as dr2_dec_pmdec_corr,
            dr2.parallax_pmra_corr as dr2_parallax_pmra_corr, dr2.parallax_pmdec_corr as dr2_parallax_pmdec_corr,
            dr2.pmra_pmdec_corr as dr2_pmra_pmdec_corr, dr2.phot_g_mean_mag as dr2_phot_g_mean_mag,
            dr2.phot_g_mean_flux as dr2_phot_g_mean_flux, dr2.phot_g_mean_flux_error as dr2_phot_g_mean_flux_error,
            dr2.phot_bp_mean_flux as dr2_phot_bp_mean_flux, dr2.phot_bp_mean_flux_error as dr2_phot_bp_mean_flux_error,
            dr2.phot_bp_mean_mag as dr2_phot_bp_mean_mag,
            dr2.phot_rp_mean_flux as dr2_phot_rp_mean_flux, dr2.phot_rp_mean_flux_error as dr2_phot_rp_mean_flux_error,
            dr2.phot_rp_mean_mag as dr2_phot_rp_mean_mag,
            dr2.bp_rp as dr2_bp_rp, dr2.phot_bp_rp_excess_factor as dr2_phot_bp_rp_excess_factor,
            dr2ruwe.ruwe as dr2_ruwe, dr2.astrometric_params_solved as dr2_astrometric_params_solved,
            dr2.radial_velocity, dr2.radial_velocity_error,"""
            qstr3+="""
            from
                gaiaedr3.gaia_source as edr3
            LEFT OUTER JOIN
                gaiaedr3.dr2_neighbourhood AS dr2xmatch
                ON edr3.source_id = dr2xmatch.dr3_source_id
            LEFT OUTER JOIN
                gaiadr2.gaia_source as dr2
                ON dr2xmatch.dr2_source_id = dr2.source_id
            LEFT OUTER JOIN
                gaiadr2.ruwe as dr2ruwe
                ON dr2xmatch.dr2_source_id = dr2ruwe.source_id
            """
        if '2mass' in surveys: 
            qstr1+='tmass.designation as tmass_id, '
            qstr2+="""
            tmass.j_m, tmass.j_msigcom,
            tmass.h_m, tmass.h_msigcom,
            tmass.ks_m, tmass.ks_msigcom,
            tmass.ph_qual,
            tmass.ra as tmass_ra, tmass.dec as tmass_dec,"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiaedr3.tmass_psc_xsc_best_neighbour AS tmassxmatch
                ON edr3.source_id = tmassxmatch.source_id
            LEFT OUTER JOIN
                gaiadr1.tmass_original_valid AS tmass
                ON tmassxmatch.original_ext_source_id = tmass.designation"""

        if 'wise' in surveys: 
            qstr1+='allwise.designation as allwise_id, '
            qstr2+="""
            allwise.w1mpro, allwise.w1mpro_error,
            allwise.w2mpro,allwise.w2mpro_error,
            allwise.w3mpro,allwise.w3mpro_error,
            allwise.w4mpro,allwise.w4mpro_error,
            allwise.cc_flags, allwise.ext_flag, allwise.var_flag, allwise.ph_qual as ph_qual_2, allwise.tmass_key,
            allwise.ra as wise_ra, allwise.dec as wise_dec"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiaedr3.allwise_best_neighbour AS allwisexmatch
                ON edr3.source_id = allwisexmatch.source_id
            LEFT OUTER JOIN
                gaiadr1.allwise_original_valid AS allwise
                ON allwisexmatch.original_ext_source_id = allwise.designation
            """   
        if 'apass' in surveys: 
            qstr1+='apass.recno as apassdr9_id, '
            qstr2+="""
            apass.b_v, apass.e_b_v, apass.vmag, apass.e_vmag, apass.bmag, apass.e_bmag, apass.r_mag, apass.e_r_mag, apass.i_mag, apass.e_i_mag,"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiaedr3.apassdr9_best_neighbour AS apassxmatch
                ON edr3.source_id = apassxmatch.source_id
            LEFT OUTER JOIN
                external.apassdr9 AS apass
                ON apassxmatch.clean_apassdr9_oid = apass.recno
            """
        if 'sloan' in surveys:
            qstr1+='sloan.objid as sloan_id, '
            qstr2+="""
            sloan.u, sloan.err_u, sloan.g, sloan.err_g, sloan.r, sloan.err_r, sloan.i, sloan.err_i, sloan.u, sloan.err_u,"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiaedr3.sdssdr13_best_neighbour AS sloanxmatch
                ON edr3.source_id = sloanxmatch.source_id
            LEFT OUTER JOIN
                external.sdssdr13_photoprimary as sloan
                ON sloanxmatch.clean_sdssdr13_oid = sloan.objid        
            """
        if 'panstarrs' in surveys: 
            qstr1+='ps1xmatch.source_id as panstarrs_id, '        
            qstr2+="""
            panstarrs.g_mean_psf_mag as ps1_g, g_mean_psf_mag_error as ps1_g_error, panstarrs.r_mean_psf_mag as ps1_r, r_mean_psf_mag_error as ps1_r_error,
            panstarrs.i_mean_psf_mag as ps1_i, i_mean_psf_mag_error as ps1_i_error, panstarrs.z_mean_psf_mag as ps1_z, z_mean_psf_mag_error as ps1_z_error,
            panstarrs.y_mean_psf_mag as ps1_y, y_mean_psf_mag_error as ps1_y_error,"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiaedr3.panstarrs1_best_neighbour AS ps1xmatch
                ON edr3.source_id = ps1xmatch.source_id
            LEFT OUTER JOIN
                gaiadr2.panstarrs1_original_valid AS panstarrs
                ON ps1xmatch.original_ext_source_id = panstarrs.obj_id            
            """

        if qstr2.rstrip().endswith(','): qstr2=qstr2.rstrip()[:-1]

        qstr0="""
        select all
        """        
        qstr4="""
        WHERE """+query_list
        qstr=qstr0+qstr1+qstr2+qstr3+qstr4

        return qstr

    def get_phot(self, save_phot, verbose=True):
        data=[]
        start = time.time()

        n_chunks=1
        nst=len(self.GaiaID)
        print('no. of stars: ',nst)
        done=np.zeros(nst,dtype=bool)
        nit=0
        while (np.sum(done)<nst) & (nit<10):
            todo,=np.where(done==False)
            st=int(len(todo)/n_chunks)
            for i in range(n_chunks):
                todo_c=todo[i*st:(i+1)*st]
                query_list=self.list_chunk(todo_c)
                qstr=self.query_string(query_list)
                try:
                    adql = QueryStr(qstr,verbose=False)
                    t=gaia.query(adql)        
                    data.append(t)
                    done[todo_c]=True
                except JSONDecodeError: 
                    continue
            n_chunks*=2
            nit+=1
            if nit>9: raise RuntimeError('Perhaps '+nst+' stars are too many?')                
                
        if len(data)>1: t=vstack(data)
        else: t=data[0]
        print('len before: ',len(t))
        t=self.fix_double_entries(t)
        print('len after: ',len(t))
        
        t=self.fix_2mass(t)

        data=[]
        with np.errstate(divide='ignore',invalid='ignore'):        
            edr3_gmag_corr, edr3_gflux_corr = self.correct_gband(t.field('edr3_bp_rp'), t.field('edr3_astrometric_params_solved'), t.field('edr3_phot_g_mean_mag'), t.field('edr3_phot_g_mean_flux'))
            edr3_bp_rp_excess_factor_corr = self.edr3_correct_flux_excess_factor(t.field('edr3_bp_rp'), t.field('edr3_phot_bp_rp_excess_factor'))
            edr3_g_mag_error, edr3_bp_mag_error, edr3_rp_mag_error = self.gaia_mag_errors(t.field('edr3_phot_g_mean_flux'), t.field('edr3_phot_g_mean_flux_error'), t.field('edr3_phot_bp_mean_flux'), t.field('edr3_phot_bp_mean_flux_error'), t.field('edr3_phot_rp_mean_flux'), t.field('edr3_phot_rp_mean_flux_error'))
            dr2_bp_rp_excess_factor_corr = self.dr2_correct_flux_excess_factor(t.field('dr2_phot_g_mean_mag'), t.field('dr2_bp_rp'), t.field('dr2_phot_bp_rp_excess_factor'))
            dr2_g_mag_error, dr2_bp_mag_error, dr2_rp_mag_error = self.gaia_mag_errors(t.field('dr2_phot_g_mean_flux'), t.field('dr2_phot_g_mean_flux_error'), t.field('dr2_phot_bp_mean_flux'), t.field('dr2_phot_bp_mean_flux_error'), t.field('dr2_phot_rp_mean_flux'), t.field('dr2_phot_rp_mean_flux_error'))
            t_ext=Table([edr3_gmag_corr, edr3_gflux_corr, edr3_bp_rp_excess_factor_corr, edr3_g_mag_error, edr3_bp_mag_error, edr3_rp_mag_error, dr2_bp_rp_excess_factor_corr, dr2_g_mag_error, dr2_bp_mag_error, dr2_rp_mag_error],
                names=['edr3_gmag_corr', 'edr3_gflux_corr','edr3_phot_bp_rp_excess_factor_corr', 'edr3_phot_g_mean_mag_error', 'edr3_phot_bp_mean_mag_error', 'edr3_phot_rp_mean_mag_error', 'dr2_phot_bp_rp_excess_factor_corr', 'dr2_g_mag_error', 'dr2_bp_mag_error', 'dr2_rp_mag_error'],
                units=["mag", "'electron'.s**-1", "", "mag", "mag", "mag", "", "mag", "mag", "mag"],
                descriptions=['EDR3 G-band mean mag corrected as per Riello et al. (2021)', 'EDR3 G-band mean flux corrected as per Riello et al. (2021)', 'EDR3 BP/RP excess factor corrected as per Riello et al. (2021)','EDR3 Error on G-band mean mag', 'EDR3 Error on BP-band mean mag', 'EDR3 Error on RP-band mean mag', 'DR2 BP/RP excess factor corrected as per Squicciarini et al. (2021)', 'DR2 Error on G-band mean mag', 'DR2 Error on BP-band mean mag', 'DR2 Error on RP-band mean mag'])
    #        data.append(hstack([self.ID[i], t, t_ext]))
            data.append(hstack([self.ID, t, t_ext]))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
    #        if len(data)>1:
    #            self.phot_table=vstack(data)
    #        else: self.phot_table=data[0]                
            self.phot_table=vstack(data)
        if save_phot == True:
            filename=os.path.join(self.path,(self.__sample_name+'_phot_table.csv'))
            ascii.write(self.phot_table, filename, format='csv', overwrite=True)
        if verbose == True:
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Total time needed to retrieve photometry for "+ np.str(len(self.GaiaID))+ " targets: - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))            

            
    def fix_double_entries(self,t,index=None):        
        
        if type(index)==type(None): index=np.arange(len(self.GaiaID))
                
        n=len(index)
        
        if (len(t)<2) & (n<2): return t
        else:
            t=Table(t,masked=True)            
            n_t=len(t)
            id_type=self.__id_type

            w,=np.where(t['dr2_id']=='')
            t['dr2_id'].mask[w]=True
            t['dr2_id']=t['dr2_id'].filled('Gaia DR2 0000')
            
            w,=np.where(t['edr3_id']=='')
            t['edr3_id'].mask[w]=True
            t['edr3_id']=t['edr3_id'].filled('Gaia EDR3 0000')            

            gaia2_col=np.array([str(i).split('Gaia DR2')[1] for i in t['dr2_id']])
            gaia3_col=np.array([str(i).split('Gaia EDR3')[1] for i in t['edr3_id']])
            ind=[]
            p_mask=[]
            t_mask=[]
            if id_type=='EDR3':
                cols=['dr2_id', 'dr2_epoch', 'dr2_ra', 'dr2_dec', 'dr2_parallax', 'dr2_parallax_error', 'dr2_parallax_over_error', 'dr2_pmra', 'dr2_pmra_error', 'dr2_pmdec', 'dr2_pmdec_error', 'dr2_ra_dec_corr', 'dr2_ra_parallax_corr', 'dr2_ra_pmra_corr', 'dr2_ra_pmdec_corr', 'dr2_dec_parallax_corr', 'dr2_dec_pmra_corr', 'dr2_dec_pmdec_corr', 'dr2_parallax_pmra_corr', 'dr2_parallax_pmdec_corr', 'dr2_pmra_pmdec_corr', 'dr2_phot_g_mean_mag', 'dr2_phot_g_mean_flux', 'dr2_phot_g_mean_flux_error', 'dr2_phot_bp_mean_flux', 'dr2_phot_bp_mean_flux_error', 'dr2_phot_bp_mean_mag', 'dr2_phot_rp_mean_flux', 'dr2_phot_rp_mean_flux_error', 'dr2_phot_rp_mean_mag', 'dr2_bp_rp', 'dr2_phot_bp_rp_excess_factor', 'dr2_ruwe', 'dr2_astrometric_params_solved', 'radial_velocity', 'radial_velocity_error']
                for i in range(n):
                    id=str(self.GaiaID[index[i]]).split('Gaia EDR3')[1]
                    w,=np.where(id==gaia3_col)
                    if len(w)==1: 
                        ind.extend(w)
                        if gaia2_col[w]==' 0000': t['dr2_id'].mask[w]=True
                    elif len(w)==0: 
                        ind.append(0)
                        t_mask.append(i)
                    else: 
                        w1,=np.where(id==gaia2_col[w])
                        if len(w1)==1: ind.extend(w[w1])
                        else: 
                            ind.append(0)
                            p_mask.append(i)
            else:
                cols=['edr3_id', 'ra', 'dec', 'edr3_epoch', 'edr3_parallax', 'edr3_parallax_error', 'edr3_parallax_over_error', 'edr3_pmra', 'edr3_pmra_error', 'edr3_pmdec', 'edr3_pmdec_error', 'edr3_ra_dec_corr', 'edr3_ra_parallax_corr', 'edr3_ra_pmra_corr', 'edr3_ra_pmdec_corr', 'edr3_dec_parallax_corr', 'edr3_dec_pmra_corr', 'edr3_dec_pmdec_corr', 'edr3_parallax_pmra_corr', 'edr3_parallax_pmdec_corr', 'edr3_pmra_pmdec_corr', 'edr3_phot_g_mean_mag', 'edr3_phot_g_mean_flux', 'edr3_phot_g_mean_flux_error', 'edr3_phot_bp_mean_flux', 'edr3_phot_bp_mean_flux_error', 'edr3_phot_bp_mean_mag', 'edr3_phot_rp_mean_flux', 'edr3_phot_rp_mean_flux_error', 'edr3_phot_rp_mean_mag', 'edr3_bp_rp', 'edr3_phot_bp_rp_excess_factor', 'edr3_ruwe', 'edr3_astrometric_params_solved']
                for i in range(n):
                    id=str(self.GaiaID[index[i]]).split('Gaia DR2')[1]
                    w,=np.where(id==gaia2_col)
                    if len(w)==1: 
                        ind.extend(w)
                        if gaia3_col[w]==' 0000': t['edr3_id'].mask[w]=True
                    elif len(w)==0:
                        ind.append(0)
                        t_mask.append(i)
                    else: 
                        w1,=np.where(id==gaia3_col[w])
                        if len(w1)==1: ind.extend(w[w1])
                        else: 
                            ind.append(0)
                            p_mask.append(i)
            ind=np.array(ind)
            t=t[ind]
            if len(t_mask)>0:
                t_mask=np.array(t_mask)
                for j in t_mask:
                    for i in t.columns:
                        t[i].mask[j]=True        
            if len(p_mask)>0:
                p_mask=np.array(p_mask)
                for j in p_mask:
                    for i in cols:
                        t[i].mask[j]=True        
            return t

    def check_phot(self,**kwargs):

        t=copy.deepcopy(self.phot_table)
        t=Table(t, masked=True, copy=False)
        
        dr2_q = self.dr2_quality(t.field('dr2_phot_bp_rp_excess_factor_corr'),t.field('dr2_phot_g_mean_mag'))
        t['dr2_phot_bp_mean_mag'].mask[~dr2_q]=True
        t['dr2_phot_rp_mean_mag'].mask[~dr2_q]=True
        t['dr2_phot_bp_mean_mag'].fill_value = np.nan
        t['dr2_phot_rp_mean_mag'].fill_value = np.nan
        
        edr3_q = self.edr3_quality(t.field('edr3_phot_bp_rp_excess_factor_corr'),t.field('edr3_phot_g_mean_mag'))
        t['edr3_phot_bp_mean_mag'].mask[~edr3_q]=True
        t['edr3_phot_rp_mean_mag'].mask[~edr3_q]=True
        t['edr3_phot_bp_mean_mag'].fill_value = np.nan
        t['edr3_phot_rp_mean_mag'].fill_value = np.nan
        
        if '2mass' in self.surveys:
            if 'max_tmass_q' in kwargs:
                max_tmass_q=kwargs['max_tmass_q']
            else: max_tmass_q='A'
            tm_q = self.tmass_quality(t.field('ph_qual'),max_q=max_tmass_q)
            t['j_m'].mask[~tm_q[0]]=True
            t['h_m'].mask[~tm_q[1]]=True
            t['ks_m'].mask[~tm_q[2]]=True
            t['j_m'].fill_value = np.nan
            t['h_m'].fill_value = np.nan
            t['ks_m'].fill_value = np.nan
            
        if 'wise' in self.surveys:
            if 'max_wise_q' in kwargs:
                max_wise_q=kwargs['max_wise_q']
            else: max_wise_q='A'
            wise_q = self.allwise_quality(t.field('cc_flags'),t.field('ph_qual_2'),max_q=max_wise_q)
            t['w1mpro'].mask[~wise_q[0]]=True
            t['w2mpro'].mask[~wise_q[1]]=True
            t['w3mpro'].mask[~wise_q[2]]=True
            t['w4mpro'].mask[~wise_q[3]]=True
            t['w1mpro'].fill_value = np.nan
            t['w2mpro'].fill_value = np.nan
            t['w3mpro'].fill_value = np.nan
            t['w4mpro'].fill_value = np.nan
        
        return t

    def get_agemass(self, model, **kwargs):
        model=(str.lower(model)).replace('-','_')        

        w,=np.where(self.filters=='G')
        if 'mass_range' in kwargs: self.mass_range=MADYS.get_mass_range(kwargs['mass_range'],model,dtype='mass')
        elif len(w)==1: self.mass_range=MADYS.get_mass_range(self.abs_phot[:,w],model)
        else: self.mass_range=MADYS.get_mass_range([1e-6,1e+6],model)
        kwargs['mass_range']=self.mass_range

        self.age_range = kwargs['age_range'] if 'age_range' in kwargs else [1,1000]
        self.n_steps = kwargs['n_steps'] if 'n_steps' in kwargs else [1000,500]
        verbose = kwargs['verbose'] if 'verbose' in kwargs else True
        self.feh = kwargs['feh'] if 'feh' in kwargs else None
        self.he = kwargs['he'] if 'he' in kwargs else None
        self.afe = kwargs['afe'] if 'afe' in kwargs else None
        self.v_vcrit = kwargs['v_vcrit'] if 'v_vcrit' in kwargs else None
        self.fspot = kwargs['fspot'] if 'fspot' in kwargs else None
        self.B = kwargs['B'] if 'B' in kwargs else 0
        self.ph_cut = kwargs['ph_cut'] if 'ph_cut' in kwargs else 0.2
        m_unit=kwargs['m_unit'] if 'm_unit' in kwargs else 'm_sun'
        phys_param=kwargs['phys_param'] if 'phys_param' in kwargs else False
        hot_points=kwargs['hot_points'] if 'hot_points' in kwargs else False

        self.__logger.info('Starting age determination...')
        filt=np.concatenate([self.filters,['logg','Teff','logL','radius']]) if phys_param else self.filters


        iso_mass,iso_age,iso_filt,iso_data=MADYS.load_isochrones(model,filt,logger=self.__logger,**kwargs)        
        
        self.__logger.info('Isochrones for model '+model+' correctly loaded')
        iso_mass_log=np.log10(iso_mass)
        iso_age_log=np.log10(iso_age)
        
        if phys_param:
            phys_filt=['logg','Teff','logL','radius']
            w_p=MADYS.where_v(phys_filt,iso_filt)
            w_d=MADYS.complement_v(w_p,len(iso_filt))
            iso_filt=np.delete(iso_filt,w_p)
            phys_data=iso_data[:,:,w_p]
            iso_data=iso_data[:,:,w_d]
            self.__logger.info('Estimation of physical parameters (radius, Teff, log(L), log(g))? Yes')
        else: self.__logger.info('Estimation of physical parameters (radius, Teff, log(L), log(g))? No')

        mass_range_str=["%.2f" % s for s in self.mass_range]
        try:
            age_range_str=["%s" % s for s in self.age_range]
        except TypeError: age_range_str=[str(self.age_range)]

        self.__logger.info('Input parameters for the model: mass range = ['+','.join(mass_range_str)+'] M_sun; age range = ['+','.join(age_range_str)+'] Myr')
        if self.feh==None: self.__logger.info('Metallicity: solar (use MADYS.info_models('+model+') for details).')
        else: self.__logger.info('Metallicity: [Fe/H]='+str(self.feh)+' (use MADYS.info_models('+model+') for details).')
        if self.he==None: self.__logger.info('Helium content: solar (use MADYS.info_models('+model+') for details).')
        else: self.__logger.info('Helium content: Y='+str(self.he)+' (use MADYS.info_models('+model+') for details).')
        if self.afe==None: self.__logger.info('Alpha enhancement: [a/Fe]=0.00')
        else: self.__logger.info('Alpha enhancement: [a/Fe]='+str(self.afe))
        if self.v_vcrit==None: self.__logger.info('Rotational velocity: 0.00 (non-rotating model).')
        else: self.__logger.info('Rotational velocity: '+str(self.v_vcrit)+' * v_crit.')
        if self.fspot==None: self.__logger.info('Spot fraction: f_spot=0.00.')
        else: self.__logger.info('Spot fraction: f_spot='+str(self.fspot))
        if self.B==0: self.__logger.info('Magnetic model? No')
        else: self.__logger.info('Magnetic model? Yes')

        self.__logger.info('Maximum allowed photometric uncertainty: '+str(self.ph_cut)+' mag')
        self.__logger.info('Mass unit of the results: '+m_unit)
        self.__logger.info('Age unit of the results: Myr')

        phot=self.abs_phot
        phot_err=self.abs_phot_err            

        l0=phot.shape
        xlen=l0[0] #no. of stars
        ylen=len(iso_filt) #no. of filters

        filt2=MADYS.where_v(iso_filt,self.filters)

        phot=phot[:,filt2]
        phot_err=phot_err[:,filt2]
        red=np.zeros([l0[0],len(filt2)]) #reddening
        for i in range(len(filt2)): 
            red[:,i]=MADYS.extinction(self.ebv,self.filters[filt2[i]])
        app_phot=self.app_phot[:,filt2]-red #a semi-apparent magnitude (already de-extinct. Only needs correction for distance)
        app_phot_err=self.app_phot_err[:,filt2]

        l=iso_data.shape

        m_fit=np.full(xlen,np.nan)
        m_min=np.full(xlen,np.nan)
        m_max=np.full(xlen,np.nan) 


        if l[1]==1: #just one age is present in the selected set of isochrones (e.g. pm13)
            a_fit=iso_age[0]+np.zeros(xlen)
            a_min=a_fit
            a_max=a_fit
            i_age=np.zeros(xlen)
            case=1
        elif isinstance(self.age_range,np.ndarray): 
            if len(self.age_range.shape)==1: #the age is fixed for each star
                case=1
                if len(self.age_range)!=xlen:
                    self.__logger.error('The number of stars is not equal to the number of input ages. Check the length of your input ages')
                    raise ValueError('The number of stars ('+str(xlen)+') is not equal to the number of input ages ('+str(len(self.age_range))+').')
                a_fit=self.age_range
                a_min=self.age_range
                a_max=self.age_range
                i_age=np.arange(0,xlen,dtype=int)            
            elif len(self.age_range[0])==2: #the age is to be found within the specified interval
                case=2
                if len(self.age_range)!=xlen:
                    self.__logger.error('The number of stars is not equal to the number of input ages. Check the length of your input ages')
                    raise ValueError('The number of stars ('+str(xlen)+') is not equal to the number of input ages ('+str(len(self.age_range))+').')
                i_age=np.zeros(self.age_range.shape,dtype=int)
                for i in range(xlen):
                    i_age[i,:]=MADYS.closest(iso_age,self.age_range[i,:])
                a_fit=np.full(xlen,np.nan)
                a_min=np.full(xlen,np.nan)
                a_max=np.full(xlen,np.nan)            
                ravel_indices=lambda i,j,j_len: j+j_len*i
            elif len(self.age_range[0])==3: #the age is fixed, and age_min and age_max are used to compute errors
                case=3
                if len(self.age_range)!=xlen:
                    self.__logger.error('The number of stars is not equal to the number of input ages. Check the length of your input ages')
                    raise ValueError('The number of stars ('+str(xlen)+') is not equal to the number of input ages ('+str(len(self.age_range))+').')
                i_age=np.zeros(self.age_range.shape,dtype=int)
                for i in range(xlen):
                    i_age[i,:]=MADYS.closest(iso_age,self.age_range[i,:])
                a_fit=iso_age[i_age[:,0]]
                a_min=iso_age[i_age[:,1]]
                a_max=iso_age[i_age[:,2]]
        else: #the program is left completely unconstrained
            case=4
            a_fit=np.full(xlen,np.nan)
            a_min=np.full(xlen,np.nan)
            a_max=np.full(xlen,np.nan)

        if phys_param:
            radius_fit=np.full(xlen,np.nan)
            radius_min=np.full(xlen,np.nan)
            radius_max=np.full(xlen,np.nan)
            logL_fit=np.full(xlen,np.nan)
            logL_min=np.full(xlen,np.nan)
            logL_max=np.full(xlen,np.nan)
            logg_fit=np.full(xlen,np.nan)
            logg_min=np.full(xlen,np.nan)
            logg_max=np.full(xlen,np.nan)
            Teff_fit=np.full(xlen,np.nan)
            Teff_min=np.full(xlen,np.nan)
            Teff_max=np.full(xlen,np.nan)
            if (case==2) | (case==4):
                phys_nan=np.isnan(phys_data)
                phys_data2=np.where(phys_nan,0,phys_data)
                logg_f = RectBivariateSpline(iso_mass_log,iso_age_log,phys_data2[:,:,0])
                Teff_f = RectBivariateSpline(iso_mass_log,iso_age_log,phys_data2[:,:,1])
                logL_f = RectBivariateSpline(iso_mass_log,iso_age_log,phys_data2[:,:,2])
                radius_f = RectBivariateSpline(iso_mass_log,iso_age_log,phys_data2[:,:,3])    


        all_maps=[] #a list of chi2 maps
        all_sol=[] #a list of dictionaries, with all the solutions found for each star

        iso_data_r=iso_data.reshape([l[0]*l[1],l[2]])
        l_r=l[0]*l[1]
        sigma=np.full((l_r,ylen),np.nan)
        chi2_min=np.full(xlen,np.nan)

        mem_err=True
        if hot_points==True:
            try:
                hot_p=np.zeros([l[0],l[1],xlen])
                mem_err=False
            except MemoryError: mem_err=True

        with np.errstate(divide='ignore',invalid='ignore'):        
            if case==3:
                sigma0=np.full(([l[0],ylen]),np.nan)
                sigma=np.full(([l[0],l[1],ylen]),np.nan)
                for i in range(xlen):
                    w,=np.where(MADYS.is_phot_good(phot[i,:],phot_err[i,:],max_phot_err=self.ph_cut))
                    if len(w)==0: 
                        self.__logger.info('All magnitudes for star '+str(i)+' have an error beyond the maximum allowed threshold ('+str(self.ph_cut)+' mag): age and mass determinations was not possible.')
                        all_sol.append({})
                        all_maps.append([])
                        continue
                    i00=i_age[i,0]
                    b=np.zeros(len(w),dtype=bool)
                    for h in range(len(w)):
                        ph=phot[i,w[h]]
                        sigma0[:,w[h]]=((iso_data[:,i00,w[h]]-ph)/phot_err[i,w[h]])**2
                        ii=np.nanargmin(sigma0[:,w[h]])
                        if abs(iso_data[ii,i00,w[h]]-ph)<0.2: b[h]=True #if the best theor. match is more than 0.2 mag away, discards it
                    if np.sum(b)==0:
                        self.__logger.info('All magnitudes for star '+str(i)+' are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.')
                        all_sol.append({})
                        all_maps.append([])
                        continue                        
                    w2=w[b]
                    if len(w2)>1:
                        chi2=np.sum(sigma0[:,w2],axis=1)/(np.sum(np.isnan(iso_data[:,i00,w2])==False,axis=1)-1)
                    else:
                        chi2=np.sum(sigma0[:,w2],axis=1)
                        
                    all_maps.append(chi2) # no. of degrees of freedom = no. filters - two parameters (age and mass)                
                    est,ind=MADYS.min_v(chi2)
                    chi2_min[i]=est
                    ind=ind[0]
                    
                    daa=MADYS.closest(iso_mass_log,[iso_mass_log[ind]-0.3,iso_mass_log[ind]+0.3])
                    n_try=1000
                    n_est=n_try*(i_age[i,2]-i_age[i,1]+1)
                    ind_array=np.zeros(n_est,dtype=int)
                    k=0
                    for l in range(n_try):
                        phot1,phot_err1=MADYS.app_to_abs_mag(app_phot[i,w2]+app_phot_err[i,w2]*np.random.normal(size=len(w2)),self.par[i]+self.par_err[i]*np.random.normal(),app_mag_error=app_phot_err[i,w2],parallax_error=self.par_err[i])
                        for j in range(i_age[i,1],i_age[i,2]+1):
                            for h in range(len(w2)):
                                sigma[daa[0]:daa[1],j,w2[h]]=((iso_data[daa[0]:daa[1],j,w2[h]]-phot1[h])/phot_err1[h])**2
                            if len(w2)>1:
                                chi2=np.nansum(sigma[daa[0]:daa[1],j,w2],axis=1)/(np.sum(np.isnan(iso_data[daa[0]:daa[1],j,w2])==False,axis=1)-1)
                            else:
                                chi2=np.nansum(sigma[daa[0]:daa[1],j,w2],axis=1)
                            
                            ind_array[k]=daa[0]+np.nanargmin(chi2)
                            k+=1
                    
                    m_min[i],m_fit[i],m_max[i]=10**np.percentile(iso_mass_log[ind_array],[16,50,84])
                    if phys_param:
                        rep_ages=np.tile(np.arange(i_age[i,1],i_age[i,2]+1),n_try)
                        logg_min[i],logg_fit[i],logg_max[i]=np.percentile(phys_data[ind_array,rep_ages,0],[16,50,84])
                        Teff_min[i],Teff_fit[i],Teff_max[i]=10**np.percentile(np.log10(phys_data[ind_array,rep_ages,1]),[16,50,84])
                        logL_min[i],logL_fit[i],logL_max[i]=np.percentile(phys_data[ind_array,rep_ages,2],[16,50,84])
                        radius_min[i],radius_fit[i],radius_max[i]=10**np.percentile(np.log10(phys_data[ind_array,rep_ages,3]),[16,50,84])
                if phys_param:
                    dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                         'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                         'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,
                         'radii':radius_fit, 'radii_min': radius_min, 'radii_max': radius_max,
                         'logg':logg_fit, 'logg_min': logg_min, 'logg_max': logg_max,
                         'logL':logL_fit, 'logL_min': logL_min, 'logL_max': logL_max,
                         'Teff':Teff_fit, 'Teff_min': Teff_min, 'Teff_max': Teff_max,
                         'iso_mass':iso_mass, 'iso_age':iso_age,
                         'model':model}
                else:
                    dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                         'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                         'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,
                         'iso_mass':iso_mass, 'iso_age':iso_age,
                         'model':model}
                        
            elif case==1:
                sigma=np.full(([l[0],1,ylen]),np.nan)
                for i in range(xlen):
                    w,=np.where(MADYS.is_phot_good(phot[i,:],phot_err[i,:],max_phot_err=self.ph_cut))
                    if len(w)==0: 
                        self.__logger.info('All magnitudes for star '+str(i)+' have an error beyond the maximum allowed threshold ('+str(self.ph_cut)+' mag): age and mass determinations was not possible.')
                        all_sol.append({})
                        all_maps.append([])
                        continue
                    i00=i_age[i]

                    b=np.zeros(len(w),dtype=bool)
                    for h in range(len(w)):
                        ph=phot[i,w[h]]
                        sigma[:,0,w[h]]=((iso_data[:,i00,w[h]]-ph)/phot_err[i,w[h]])**2
                        ii=np.nanargmin(sigma[:,0,w[h]])
                        if abs(iso_data[ii,i00,w[h]]-ph)<0.2: b[h]=True #if the best theor. match is more than 0.2 mag away, discards it
                    if np.sum(b)==0:
                        self.__logger.info('All magnitudes for star '+str(i)+' are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.')
                        all_sol.append({})
                        all_maps.append([])
                        continue
                    w2=w[b]
                    
                    if len(w2)>1:
                        chi2=np.nansum(sigma[:,0,w2],axis=1)/(np.sum(np.isnan(iso_data[:,i00,w2])==False,axis=1)-1) #no. of degrees of freedom = no. filters - one parameter (mass) 
                    else:
                        chi2=np.nansum(sigma[:,0,w2],axis=1)
                    
                    all_maps.append(chi2)                
                    est,ind=MADYS.min_v(chi2)
                    chi2_min[i]=est/(len(w2)-2)
                    m_fit[i]=iso_mass[ind[0]]
                    a_fit[i]=iso_age[i00]
                    
                    n_try=1000
                    m_f1=np.zeros(n_try)
                    a_f1=np.zeros(n_try)
                                                
                    if phys_param:
                        logg_fit[i]=phys_data[ind[0],i00,0]
                        Teff_fit[i]=phys_data[ind[0],i00,1]
                        logL_fit[i]=phys_data[ind[0],i00,2]
                        radius_fit[i]=phys_data[ind[0],i00,3]
                        logg_f1=np.zeros(n_try)
                        Teff_f1=np.zeros(n_try)
                        logL_f1=np.zeros(n_try)
                        radius_f1=np.zeros(n_try)
                        for j in range(n_try):
                            phot1,phot_err1=MADYS.app_to_abs_mag(app_phot[i,w2]+app_phot_err[i,w2]*np.random.normal(size=len(w2)),self.par[i]+self.par_err[i]*np.random.normal(),app_mag_error=app_phot_err[i,w2],parallax_error=self.par_err[i])                            
                            for h in range(len(w2)):
                                sigma[:,0,w2[h]]=((iso_data[:,i00,w2[h]]-phot1[h])/phot_err1[h])**2
                            cr1=np.sum(sigma[:,:,w2],axis=2)
                            est1,ind1=MADYS.min_v(cr1)
                            m_f1[j]=iso_mass_log[ind1[0]]
                            a_f1[j]=iso_age_log[i00]
                            logg_f1[j]=phys_data[ind1[0],i00,0]
                            Teff_f1[j]=phys_data[ind1[0],i00,1]
                            logL_f1[j]=phys_data[ind1[0],i00,2]
                            radius_f1[j]=phys_data[ind1[0],i00,3]
                        s_logg=np.std(logg_f1,ddof=1)
                        logg_min[i]=logg_fit[i]-s_logg
                        logg_max[i]=logg_fit[i]+s_logg
                        s_logL=np.std(logL_f1,ddof=1)
                        logL_min[i]=logL_fit[i]-s_logL
                        logL_max[i]=logL_fit[i]+s_logL
                        s_Teff=np.std(np.log10(Teff_f1),ddof=1)
                        Teff_min[i]=10**(np.log10(Teff_fit[i])-s_teff)
                        Teff_max[i]=10**(np.log10(Teff_fit[i])+s_teff)
                        s_radius=np.std(np.log10(radius_f1),ddof=1)
                        radius_min[i]=10**(np.log10(radius_fit[i])-s_radius)
                        radius_max[i]=10**(np.log10(radius_fit[i])+s_radius)
                    else:
                        for j in range(n_try):
                            phot1,phot_err1=MADYS.app_to_abs_mag(app_phot[i,w2]+app_phot_err[i,w2]*np.random.normal(size=len(w2)),self.par[i]+self.par_err[i]*np.random.normal(),app_mag_error=app_phot_err[i,w2],parallax_error=self.par_err[i])                            
                            for h in range(len(w2)):
                                sigma[:,0,w2[h]]=((iso_data[:,i00,w2[h]]-phot1[h])/phot_err1[h])**2
                            cr1=np.sum(sigma[:,:,w2],axis=2)
                            est1,ind1=MADYS.min_v(cr1)
                            m_f1[j]=iso_mass_log[ind1[0]]
                            a_f1[j]=iso_age_log[i00]
                    m_min[i]=10**(np.log10(m_fit[i])-np.std(m_f1,ddof=1))
                    m_max[i]=10**(np.log10(m_fit[i])+np.std(m_f1,ddof=1))
                    a_min[i]=10**(np.log10(a_fit[i])-np.std(a_f1,ddof=1))
                    a_max[i]=10**(np.log10(a_fit[i])+np.std(a_f1,ddof=1))            
                        
                if phys_param:
                    dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                         'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                         'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,
                         'radii':radius_fit, 'radii_min': radius_min, 'radii_max': radius_max,
                         'logg':logg_fit, 'logg_min': logg_min, 'logg_max': logg_max,
                         'logL':logL_fit, 'logL_min': logL_min, 'logL_max': logL_max,
                         'Teff':Teff_fit, 'Teff_min': Teff_min, 'Teff_max': Teff_max,
                         'iso_mass':iso_mass, 'iso_age':iso_age,
                         'model':model}
                else:
                    dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                         'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                         'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,
                         'iso_mass':iso_mass, 'iso_age':iso_age,
                         'model':model}

            else:
                n_try=1000
                for i in range(xlen):
                    w,=np.where(MADYS.is_phot_good(phot[i,:],phot_err[i,:],max_phot_err=self.ph_cut))
                    if len(w)==0: 
                        self.__logger.info('All magnitudes for star '+str(i)+' have an error beyond the maximum allowed threshold ('+str(self.ph_cut)+' mag): age and mass determinations was not possible.')
                        all_sol.append({})
                        all_maps.append([])
                        continue
                    b=np.zeros(len(w),dtype=bool)

                    if case==2:                    
                        use_i=[]
                        for j in range(i_age[i,0],i_age[i,1]+1):
                            new_i=ravel_indices(np.arange(0,l[0]),j,l[1])
                            use_i.extend(new_i) #ages to be used
                        use_i=np.array(use_i,dtype=int)
                    else: use_i=np.arange(0,l_r,dtype=int)                

                    for h in range(len(w)):
                        ph=phot[i,w[h]]
                        sigma[use_i,w[h]]=((iso_data_r[use_i,w[h]]-ph)/phot_err[i,w[h]])**2
                        if np.nanmin(np.abs(iso_data_r[use_i,w[h]]-ph))<0.2: b[h]=True #if the best theor. match is more than 0.2 mag away, discards it      
                    w2=w[b]
                    if len(w2)<3:
                        if np.sum(b)==0:
                            self.__logger.info('All magnitudes for star '+str(i)+' are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.')
                        else:
                            self.__logger.info('Less than three good filters for star '+str(i)+': use a less strict error threshold, or consider adopting an age range to have at least a mass estimate.')
                        all_sol.append({})
                        all_maps.append([])
                        continue #at least 3 filters needed for the fit
                    if len(use_i)<l_r: sigma[MADYS.complement_v(use_i,l_r),:]=np.nan
                    chi2=np.nansum(sigma[:,w2],axis=1)/(np.sum(np.isnan(iso_data_r[:,w2])==False,axis=1)-2)
                    
                    all_maps.append(chi2.reshape([l[0],l[1]])) #two parameters: age and mass
                    ind=np.nanargmin(chi2)
                    crit1=np.sort(sigma[ind,w2])
                    crit2=np.sort(iso_data_r[ind,w2])
                    g_sol=[]
                    chi_sol=[]
                    if (crit1[2]<9) | (crit2[2]<0.1): #the 3rd best sigma < 3 or the 3rd best solution closer than 0.1 mag  
                        w_ntb,=np.where(chi2<1000)
                        chi2_red=chi2[w_ntb]
                        if len(w_ntb)==0:
                            self.__logger.info('No good fits could be found for star '+str(i)+'. Returning nan.')
                            all_sol.append({})
                            all_maps.append([])
                            continue
                        gsol,=np.where(chi2_red<(chi2[ind]+2.3)) #68.3% C.I.. Use delta chi2=4.61,6.17,11.8 for 90%,95.4%,99.73% C.I.
                        g_sol.append(w_ntb[gsol])
                        chi_sol.append(chi2[w_ntb[gsol]])
                        if phys_param:                        
                            for j in range(n_try):
                                phot1,phot_err1=MADYS.app_to_abs_mag(app_phot[i,w2]+app_phot_err[i,w2]*np.random.normal(size=len(w2)),self.par[i]+self.par_err[i]*np.random.normal(),app_mag_error=app_phot_err[i,w2],parallax_error=self.par_err[i])
                                for h in range(len(w2)):
                                    sigma[w_ntb,w2[h]]=((iso_data_r[w_ntb,w2[h]]-phot1[h])/phot_err1[h])**2
                                sigma_red=sigma[w_ntb,:]
                                chi2=np.sum(sigma_red[:,w2],axis=1)/(np.sum(np.isnan(iso_data_r[:,w2])==False,axis=1)-2)
                                ind=np.argmin(chi2)
                                gsol,=np.where(chi2<(chi2[ind]+2.3)) #68.3% C.I.. Use delta chi2=4.61,6.17,11.8 for 90%,95.4%,99.73% C.I.
                                g_sol.append(w_ntb[gsol])
                                chi_sol.append(chi2[gsol])
                            g_sol=np.concatenate(g_sol)
                            chi_sol=np.concatenate(chi_sol)
                            chi2_min[i]=np.min(chi_sol)
                            i_ma=np.unravel_index(g_sol,(l[0],l[1]))

                            ma0=np.zeros([l[0],l[1]])
                            np.add.at(ma0,i_ma,1)                        
                            i_ma0=np.where(ma0>100)
                            ma=np.zeros([l[0],l[1]],dtype=bool)
                            ma[i_ma0]=True # ma[i_ma]=True

                            labeled, _ = label(ma, np.ones((3, 3), dtype=np.int))
                            labeled_r=labeled.ravel()        

                            n_gr=np.max(labeled)

                            com=np.array(center_of_mass(ma,labeled,range(1,n_gr+1)))
                            wn,=np.where((labeled_r==0) & (ma0.ravel()!=0))
                            wn1,wn2=np.unravel_index(wn,ma.shape)

                            lab_n=np.zeros(len(wn1))
                            for kk in range(len(wn1)):
                                lab_n[kk]=np.argmin((wn1[kk]-com[:,0])**2+(wn2[kk]-com[:,1])**2)+1
                            labeled[wn1,wn2]=lab_n                        
                            labeled_r=labeled.ravel()        

                            mship=labeled_r[g_sol]            
                            wei_norm=np.sum(1/chi_sol)
                            fam=np.zeros([n_gr,5])

                            if mem_err==False: np.add.at(hot_p[:,:,i],i_ma,1/chi_sol)                        

                            for jj in range(n_gr):
                                w_gr,=np.where(mship==(jj+1))
                                n_gr0=len(w_gr)
                                if n_gr0==1:
                                    fam[jj,0]=iso_mass_log[i_ma[0][w_gr]]
                                    fam[jj,2]=iso_age_log[i_ma[1][w_gr]]
                                else:
                                    fam[jj,0]=np.average(iso_mass_log[i_ma[0][w_gr]],weights=1/chi_sol[w_gr])
                                    fam[jj,1]=np.sqrt(np.average((iso_mass_log[i_ma[0][w_gr]]-fam[jj,0])**2,weights=1/chi_sol[w_gr])*n_gr0/(n_gr0-1))
                                    fam[jj,2]=np.average(iso_age_log[i_ma[1][w_gr]],weights=1/chi_sol[w_gr])
                                    fam[jj,3]=np.sqrt(np.average((iso_age_log[i_ma[1][w_gr]]-fam[jj,2])**2,weights=1/chi_sol[w_gr])*n_gr0/(n_gr0-1))
                                fam[jj,4]=np.sum(1/chi_sol[w_gr])/wei_norm
                            fam=MADYS.merge_fam(fam)

                            n_gr=len(fam)            
                            i_s=np.argmax(fam[:,4])    

                            ival=np.array([[fam[i_s,0]-fam[i_s,1],fam[i_s,0],fam[i_s,0]+fam[i_s,1]],
                                          [fam[i_s,2]-fam[i_s,3],fam[i_s,2],fam[i_s,2]+fam[i_s,3]]])                        

                            logg_fit[i],logg_min[i],logg_max[i]=logg_f(ival[0,1],ival[1,1]),np.nanmin(logg_f(ival[0,:],ival[1,:])),np.nanmax(logg_f(ival[0,:],ival[1,:]))
                            Teff_fit[i],Teff_min[i],Teff_max[i]=Teff_f(ival[0,1],ival[1,1]),np.nanmin(Teff_f(ival[0,:],ival[1,:])),np.nanmax(Teff_f(ival[0,:],ival[1,:]))
                            logL_fit[i],logL_min[i],logL_max[i]=logL_f(ival[0,1],ival[1,1]),np.nanmin(logL_f(ival[0,:],ival[1,:])),np.nanmax(logL_f(ival[0,:],ival[1,:]))
                            radius_fit[i],radius_min[i],radius_max[i]=radius_f(ival[0,1],ival[1,1]),np.nanmin(radius_f(ival[0,:],ival[1,:])),np.nanmax(radius_f(ival[0,:],ival[1,:]))

                            m_fit[i],m_min[i],m_max[i]=10**fam[i_s,0],10**(fam[i_s,0]-fam[i_s,1]),10**(fam[i_s,0]+fam[i_s,1])
                            a_fit[i],a_min[i],a_max[i]=10**fam[i_s,2],10**(fam[i_s,2]-fam[i_s,3]),10**(fam[i_s,2]+fam[i_s,3])                                

                            if n_gr>1:
                                self.__logger.info('More than one region of the (mass,age) space is possible for star '+str(i)+'.')                            
                                self.__logger.info('Possible solutions for star'+str(i)+':')
                                m_all=np.zeros([n_gr,3])
                                a_all=np.zeros([n_gr,3])
                                logg_all=np.zeros([n_gr,3])
                                Teff_all=np.zeros([n_gr,3])
                                logL_all=np.zeros([n_gr,3])
                                radius_all=np.zeros([n_gr,3])
                                for jj in range(n_gr):
                                    m_all[jj,:]=[10**fam[jj,0],10**(fam[jj,0]-fam[jj,1]),10**(fam[jj,0]+fam[jj,1])]
                                    a_all[jj,:]=[10**fam[jj,2],10**(fam[jj,2]-fam[jj,3]),10**(fam[jj,2]+fam[jj,3])]

                                    ival=np.array([[fam[jj,0]-fam[jj,1],fam[jj,0],fam[jj,0]+fam[jj,1]],
                                                  [fam[jj,2]-fam[jj,3],fam[jj,2],fam[jj,2]+fam[jj,3]]])                        

                                    logg_all[jj,:]=[logg_f(ival[0,1],ival[1,1]),np.nanmin(logg_f(ival[0,:],ival[1,:])),np.nanmax(logg_f(ival[0,:],ival[1,:]))]
                                    Teff_all[jj,:]=[Teff_f(ival[0,1],ival[1,1]),np.nanmin(Teff_f(ival[0,:],ival[1,:])),np.nanmax(Teff_f(ival[0,:],ival[1,:]))]
                                    logL_all[jj,:]=[logL_f(ival[0,1],ival[1,1]),np.nanmin(logL_f(ival[0,:],ival[1,:])),np.nanmax(logL_f(ival[0,:],ival[1,:]))]
                                    radius_all[jj,:]=[radius_f(ival[0,1],ival[1,1]),np.nanmin(radius_f(ival[0,:],ival[1,:])),np.nanmax(radius_f(ival[0,:],ival[1,:]))]                                

                                    Mi,Mip,Mim='{:.3f}'.format(m_all[jj,0]),'{:.3f}'.format(m_all[jj,1]),'{:.3f}'.format(m_all[jj,2])
                                    Ai,Aip,Aim='{:.3f}'.format(a_all[jj,0]),'{:.3f}'.format(a_all[jj,1]),'{:.3f}'.format(a_all[jj,2])
                                    self.__logger.info('M='+Mi+'('+Mip+','+Mim+') M_sun, t='+Ai+'('+Aip+','+Aim+') Myr (prob='+str(fam[jj,4])+')')
                                dic={'masses':m_all,'ages':a_all,'logg':logg_all,'Teff':Teff_all,'logL':logL_all,'radii':radius_all,'prob':fam[:,4].ravel()}
                                all_sol.append(dic)
                            else: all_sol.append({'masses':np.array([m_fit[i],m_min[i],m_max[i]]),
                                                 'ages':np.array([a_fit[i],a_min[i],a_max[i]]),
                                                 'logg':np.array([logg_fit[i],logg_min[i],logg_max[i]]),
                                                 'Teff':np.array([Teff_fit[i],Teff_min[i],Teff_max[i]]),
                                                 'logL':np.array([logL_fit[i],logL_min[i],logL_max[i]]),
                                                 'radii':np.array([radius_fit[i],radius_min[i],radius_max[i]]),
                                                 'prob':fam[:,4].ravel()})                            
                        else:
                            for j in range(n_try):
                                phot1,phot_err1=MADYS.app_to_abs_mag(app_phot[i,w2]+app_phot_err[i,w2]*np.random.normal(size=len(w2)),self.par[i]+self.par_err[i]*np.random.normal(),app_mag_error=app_phot_err[i,w2],parallax_error=self.par_err[i])
                                for h in range(len(w2)):
                                    sigma[w_ntb,w2[h]]=((iso_data_r[w_ntb,w2[h]]-phot1[h])/phot_err1[h])**2
                                sigma_red=sigma[w_ntb,:]
                                chi2=np.sum(sigma_red[:,w2],axis=1)/(np.sum(np.isnan(iso_data_r[:,w2])==False,axis=1)-2)
                                ind=np.argmin(chi2)
                                gsol,=np.where(chi2<(chi2[ind]+2.3)) #68.3% C.I.. Use delta chi2=4.61,6.17,11.8 for 90%,95.4%,99.73% C.I.
                                g_sol.append(w_ntb[gsol])
                                chi_sol.append(chi2[gsol])
                            g_sol=np.concatenate(g_sol)
                            chi_sol=np.concatenate(chi_sol)
                            i_ma=np.unravel_index(g_sol,(l[0],l[1]))
                            ma=np.zeros([l[0],l[1]],dtype=bool)
                            ma[i_ma]=True

                            labeled, _ = label(ma, np.ones((3, 3), dtype=np.int))
                            labeled_r=labeled.ravel()
                            mship=labeled_r[g_sol]                
                            n_gr=np.max(labeled)
                            wei_norm=np.sum(1/chi_sol)
                            fam=np.zeros([n_gr,5])
                            for jj in range(n_gr):
                                w_gr,=np.where(mship==(jj+1))
                                n_gr0=len(w_gr)
                                if n_gr0==1:
                                    fam[jj,0]=iso_mass_log[i_ma[0][w_gr]]
                                    fam[jj,2]=iso_age_log[i_ma[1][w_gr]]
                                else:
                                    fam[jj,0]=np.average(iso_mass_log[i_ma[0][w_gr]],weights=1/chi_sol[w_gr])
                                    fam[jj,1]=np.sqrt(np.average((iso_mass_log[i_ma[0][w_gr]]-fam[jj,0])**2,weights=1/chi_sol[w_gr])*n_gr0/(n_gr0-1))
                                    fam[jj,2]=np.average(iso_age_log[i_ma[1][w_gr]],weights=1/chi_sol[w_gr])
                                    fam[jj,3]=np.sqrt(np.average((iso_age_log[i_ma[1][w_gr]]-fam[jj,2])**2,weights=1/chi_sol[w_gr])*n_gr0/(n_gr0-1))
                                fam[jj,4]=np.sum(1/chi_sol[w_gr])/wei_norm

                            fam=MADYS.merge_fam(fam)
                            n_gr=len(fam)            
                            i_s=np.argmax(fam[:,4])                    

                            m_fit[i],m_min[i],m_max[i]=10**fam[i_s,0],10**(fam[i_s,0]-fam[i_s,1]),10**(fam[i_s,0]+fam[i_s,1])
                            a_fit[i],a_min[i],a_max[i]=10**fam[i_s,2],10**(fam[i_s,2]-fam[i_s,3]),10**(fam[i_s,2]+fam[i_s,3])                


                            if n_gr>1:
                                self.__logger.info('More than one region of the (mass,age) space is possible for star '+str(i)+'.')                            
                                self.__logger.info('Possible solutions for star'+str(i)+':')
                                m_all=np.zeros([n_gr,3])
                                a_all=np.zeros([n_gr,3])
                                logg_all=np.zeros([n_gr,3])
                                Teff_all=np.zeros([n_gr,3])
                                logL_all=np.zeros([n_gr,3])
                                radius_all=np.zeros([n_gr,3])
                                for jj in range(n_gr):
                                    m_all[jj,:]=[10**fam[jj,0],10**(fam[jj,0]-fam[jj,1]),10**(fam[jj,0]+fam[jj,1])]
                                    a_all[jj,:]=[10**fam[jj,2],10**(fam[jj,2]-fam[jj,3]),10**(fam[jj,2]+fam[jj,3])]

                                    Mi,Mip,Mim='{:.3f}'.format(m_all[jj,0]),'{:.3f}'.format(m_all[jj,1]),'{:.3f}'.format(m_all[jj,2])
                                    Ai,Aip,Aim='{:.3f}'.format(a_all[jj,0]),'{:.3f}'.format(a_all[jj,1]),'{:.3f}'.format(a_all[jj,2])
                                    self.__logger.info('M='+Mi+'('+Mip+','+Mim+') M_sun, t='+Ai+'('+Aip+','+Aim+') Myr (prob='+str(fam[jj,4])+')')
                                dic={'masses':m_all,'ages':a_all,'prob':fam[:,4].ravel()}
                                all_sol.append(dic)
                            else: all_sol.append({'masses':np.array([m_fit[i],m_min[i],m_max[i]]),
                                                 'ages':np.array([a_fit[i],a_min[i],a_max[i]]),
                                                 'prob':fam[:,4].ravel()})

                    else: all_sol.append({})
                    print(i,m_fit[i],m_min[i],m_max[i],a_fit[i],a_min[i],a_max[i])

                if phys_param:
                    if mem_err: 
                        dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                             'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                             'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,
                             'radii':radius_fit, 'radii_min': radius_min, 'radii_max': radius_max,
                             'logg':logg_fit, 'logg_min': logg_min, 'logg_max': logg_max,
                             'logL':logL_fit, 'logL_min': logL_min, 'logL_max': logL_max,
                             'Teff':Teff_fit, 'Teff_min': Teff_min, 'Teff_max': Teff_max,
                             'all_solutions':all_sol, 'iso_mass':iso_mass, 'iso_age':iso_age,
                             'model':model}
                    else:
                        dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                         'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                         'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,'hot_points':hot_p,
                         'radii':radius_fit, 'radii_min': radius_min, 'radii_max': radius_max,
                         'logg':logg_fit, 'logg_min': logg_min, 'logg_max': logg_max,
                         'logL':logL_fit, 'logL_min': logL_min, 'logL_max': logL_max,
                         'Teff':Teff_fit, 'Teff_min': Teff_min, 'Teff_max': Teff_max,
                         'all_solutions':all_sol, 'iso_mass':iso_mass, 'iso_age':iso_age,
                         'model':model}

                else:
                    if mem_err:
                        dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                             'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                             'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,
                             'all_solutions':all_sol, 'iso_mass':iso_mass, 'iso_age':iso_age,
                             'model':model}
                    else:
                        dic={'ages':a_fit, 'ages_min':a_min, 'ages_max':a_max,
                             'masses':m_fit, 'masses_min':m_min, 'masses_max':m_max,
                             'ebv':self.ebv, 'chi2_min':chi2_min, 'all_maps':all_maps,'hot_points':hot_p,
                             'all_solutions':all_sol, 'iso_mass':iso_mass, 'iso_age':iso_age,
                             'model':model}

        if m_unit.lower()=='m_jup':
            m_fit*=M_sun.value/M_jup.value
            if 'i_age' in locals():
                m_min*=M_sun.value/M_jup.value
                m_max*=M_sun.value/M_jup.value
            if phys_param:
                radius_fit*=R_sun.value/R_jup.value
                if 'radius_err' in locals():
                    radius_err*=R_sun.value/R_jup.value
                else:
                    radius_min*=R_sun.value/R_jup.value
                    radius_max*=R_sun.value/R_jup.value

        if verbose:
            try:
                if type(self.GaiaID)==Table: star_names=self.GaiaID['ID'].value
                else: star_names=self.GaiaID.value
            except:
                if type(self.GaiaID)==Table: star_names=self.GaiaID['ID']
                else: star_names=self.GaiaID
            filename=os.path.join(self.path,str(self.__sample_name+'_ages_'+model+'.txt'))
            f=open(filename, "w+")

            if phys_param:
                f.write(tabulate(np.column_stack((star_names,m_fit,m_min,m_max,a_fit,a_min,a_max,self.ebv,radius_fit,radius_min,radius_max,logg_fit,logg_min,logg_max,logL_fit,logL_min,logL_max,Teff_fit,Teff_min,Teff_max)),
                                 headers=['ID','MASS','MASS_MIN','MASS_MAX','AGE','AGE_MIN','AGE_MAX','E(B-V)','RADIUS','RADIUS_MIN','RADIUS_MAX','LOG(G)','LOG(G)_MIN','LOG(G)_MAX','LOG(L)','LOG(L)_MIN','LOG(L)_MAX','TEFF','TEFF_MIN','TEFF_MAX'], tablefmt='plain', stralign='right', numalign='right', floatfmt=(None,".2f",".2f",".2f",".2f",".2f",".2f",".3f",".2f",".2f",".2f",".2f",".2f",".2f",".2f",".2f",".2f",".2f",".2f",".2f")))
            else:
                f.write(tabulate(np.column_stack((star_names,m_fit,m_min,m_max,a_fit,a_min,a_max,self.ebv)),
                                 headers=['ID','MASS','MASS_MIN','MASS_MAX','AGE','AGE_MIN','AGE_MAX','E(B-V)'], tablefmt='plain', stralign='right', numalign='right', floatfmt=(None,".2f",".2f",".2f",".2f",".2f",".2f",".3f")))
            f.close()

            self.__logger.info('Age determination ended. Results saved in '+filename)
        else:
            self.__logger.info('Age determination ended. Results not saved in any file because "verbose" is set to False.')

        logging.shutdown()


        return dic
                
    @staticmethod
    def axis_range(col_name,col_phot,stick_to_points=False):
        print(col_name,col_phot)
        try:
            len(col_phot)
            cmin=np.min(col_phot)-0.1
            cmax=np.min([70,max(col_phot)])+0.1
        except TypeError:
            cmin=col_phot-0.1
            cmax=np.min([70,col_phot])+0.1
        
        if stick_to_points:
            dic1={'G':[cmax,cmin], 'Gbp':[cmax,cmin], 'Grp':[cmax,cmin],
                'J':[cmax,cmin], 'H':[cmax,cmin], 'K':[cmax,cmin],
                'W1':[cmax,cmin], 'W2':[cmax,cmin], 'W3':[cmax,cmin],
                'W4':[cmax,cmin], 'K1mag':[cmax,cmin], 'K2mag':[cmax,cmin],
                'G-J':[cmin,cmax],
                'G-H':[cmin,cmax], 'G-K':[cmin,cmax],
                'G-W1':[cmin,cmax], 'G-W2':[cmin,cmax],
                'G-W3':[cmin,cmax], 'G-W4':[cmin,cmax],
                'J-H':[cmin,cmax], 'J-K':[cmin,cmax],
                'H-K':[cmin,cmax], 'Gbp-Grp':[cmin,cmax],
                'K1mag-K2mag':[cmin,cmax]
                }
        else:
            dic1={'G':[max(15,cmax),min(1,cmin)], 'Gbp':[max(15,cmax),min(1,cmin)], 'Grp':[max(15,cmax),min(1,cmin)],
                'J':[max(10,cmax),min(0,cmin)], 'H':[max(10,cmax),min(0,cmin)], 'K':[max(10,cmax),min(0,cmin)],
                'W1':[max(10,cmax),min(0,cmin)], 'W2':[max(10,cmax),min(0,cmin)], 'W3':[max(10,cmax),min(0,cmin)],
                'W4':[max(10,cmax),min(0,cmin)], 'K1mag':[max(19,cmax),min(6,cmin)], 'K2mag':[max(19,cmax),min(6,cmin)],
                'G-J':[min(0,cmin),max(5,cmax)],
                'G-H':[min(0,cmin),max(5,cmax)], 'G-K':[min(0,cmin),max(5,cmax)],
                'G-W1':[min(0,cmin),max(6,cmax)], 'G-W2':[min(0,cmin),max(6,cmax)],
                'G-W3':[min(0,cmin),max(10,cmax)], 'G-W4':[min(0,cmin),max(12,cmax)],
                'J-H':[min(0,cmin),max(1,cmax)], 'J-K':[min(0,cmin),max(1.5,cmax)],
                'H-K':[min(0,cmin),max(0.5,cmax)], 'Gbp-Grp':[min(0,cmin),max(5,cmax)],
#                'K1mag-K2mag':[min(-3,cmin),max(2,cmax)]
                }

        try:
            xx=dic1[col_name]
        except KeyError:
            if '-' in col_name:
                if cmax-cmin>5: x=[cmin,cmax]
                else: xx=np.nanmean(col_phot)+[-3,3]
            else: 
                if cmax-cmin>5: x=[cmax,cmin]
                else: xx=np.nanmean(col_phot)+[3,-3]

        return xx 
    
    def CMD(self,col,mag,model,ids=None,**kwargs):

        def filter_model(model,col):
            if model in ['bt_settl','starevol','spots','dartmouth','ames_cond',
                         'ames_dusty','bt_nextgen','nextgen','bhac15','geneva',
                         'nextgen','pm13']:
                if col=='G': col2='G2'
                elif col=='Gbp': col2='Gbp2'
                elif col=='Grp': col2='Grp2'
                elif col=='G-Gbp': col2='G2-Gbp2'
                elif col=='Gbp-G': col2='Gbp2-G2'
                elif col=='G-Grp': col2='G2-Gbp2'
                elif col=='Grp-G': col2='Grp2-G2'
                elif col=='Gbp-Grp': col2='Gbp2-Grp2'
                elif col=='Grp-Gbp': col2='Grp2-Gbp2'
                elif 'G-' in col: col2=col.replace('G-','G2-')            
                elif col[-2:]=='-G': col2=col.replace('-G','-G2')
                else: col2=col        
            else: col2=col
            return col2
        
        if '-' in col:
            col_n=filter_model(model,col).split('-')
            c1,=np.where(self.filters==col_n[0])
            c2,=np.where(self.filters==col_n[1])
            col1,col1_err=self.abs_phot[:,c1],self.abs_phot_err[:,c1]
            col2,col2_err=self.abs_phot[:,c2],self.abs_phot_err[:,c2]
            col_data=col1-col2
            col_err=np.sqrt(col1_err**2+col2_err**2)
        else:
            c1,=np.where(self.filters==filter_model(model,col))
            col_data,col_err=self.abs_phot[:,c1],self.abs_phot_err[:,c1]
        if '-' in mag:
            mag_n=filter_model(model,mag).split('-')
            m1,=np.where(self.filters==mag_n[0])
            m2,=np.where(self.filters==mag_n[1])
            mag1,mag1_err=self.abs_phot[:,m1],self.abs_phot_err[:,m1]
            mag2,mag2_err=self.abs_phot[:,m2],self.abs_phot_err[:,m2]
            mag_data=mag1-mag2
            mag_err=np.sqrt(mag1_err**2+mag2_err**2)
        else:
            m1,=np.where(self.filters==filter_model(model,mag))
            mag_data,mag_err=self.abs_phot[:,m1],self.abs_phot_err[:,m1]
            
        
        wG,=np.where(self.filters=='G')
        if 'mass_range' in kwargs: mass_r=MADYS.get_mass_range(kwargs['mass_range'],model,dtype='mass')
        elif len(wG)==1: mass_r=MADYS.get_mass_range(self.abs_phot[:,wG],model)
        else: mass_r=MADYS.get_mass_range([1e-6,1e+6],model)
        print(mass_r)
            
        iso=MADYS.load_isochrones(model,self.filters,logger=self.__logger,mass_range=mass_r,**kwargs)

        col_data=col_data.ravel()
        mag_data=mag_data.ravel()
        col_err=col_err.ravel()
        mag_err=mag_err.ravel()

        if type(ids)!=type(None):
            col_data=col_data[ids]
            mag_data=mag_data[ids]
            col_err=col_err[ids]
            mag_err=mag_err[ids]
            ebv1=self.ebv[ids]
        else:
            ebv1=self.ebv
        
        plot_ages=np.array([1,3,5,10,20,30,100,200,500,1000])
        plot_masses=np.array([0.1,0.3,0.5,0.7,0.85,1.0,1.3,2])
        stick_to_points=False
        tofile=False
        
        if 'stick_to_points' in kwargs: stick_to_points=kwargs['stick_to_points']
        if 'tofile' in kwargs: tofile=kwargs['tofile']
        if 'plot_masses' in kwargs: plot_masses=kwargs['plot_masses']
        if 'plot_ages' in kwargs: plot_ages=kwargs['plot_ages']

        try:
            len(col_err)
            col_err=col_err.ravel()
            mag_err=mag_err.ravel()
        except TypeError:
            pass
        
        x=col_data
        y=mag_data
        x_axis=col
        y_axis=mag
        ebv=ebv1
        x_error=col_err
        y_error=mag_err
        
        label_points = kwargs['label_points'] if 'label_points' in kwargs else True        
        groups = kwargs['groups'] if 'groups' in kwargs else None
        group_names = kwargs['group_names'] if 'group_names' in kwargs else None
        
        isochrones=iso[3]
        iso_ages=iso[1]
        iso_filters=iso[2]
        iso_masses=iso[0]

        #changes names of Gaia_DR2 filters to EDR3
        if 'G2' in iso_filters: 
            w=MADYS.where_v(['G2','Gbp2','Grp2'],iso_filters)
            iso_filters[w]=['G','Gbp','Grp']

        #axes ranges
        if 'stick_to_points' in kwargs: stick_to_points=kwargs['stick_to_points']
        else: stick_to_points=False
        
        if 'x_range' in kwargs: x_range=kwargs['x_range']
        else: x_range=MADYS.axis_range(x_axis,x,stick_to_points=stick_to_points)
        if 'y_range' in kwargs: y_range=kwargs['y_range']
        else: y_range=MADYS.axis_range(y_axis,y,stick_to_points=stick_to_points)

            
        print(x_range)
        print(y_range)
            
        #finds color/magnitude isochrones to plot
        if '-' in x_axis: 
            col_n=x_axis.split('-')
            w1,=np.where(iso_filters==col_n[0])
            w2,=np.where(iso_filters==col_n[1])
            col_th=isochrones[:,:,w1]-isochrones[:,:,w2]
        else: 
            w1,=np.where(iso_filters==x_axis)
            col_th=isochrones[:,:,w1]
        if '-' in y_axis: 
            mag_n=y_axis.split('-')
            w1,=np.where(iso_filters==mag_n[0])
            w2,=np.where(iso_filters==mag_n[1])
            mag_th=isochrones[:,:,w1]-isochrones[:,:,w2]
        else: 
            w1,=np.where(iso_filters==y_axis)
            mag_th=isochrones[:,:,w1]


        n=len(isochrones) #no. of grid masses
        tot_iso=len(isochrones[0]) #no. of grid agesge
        npo=MADYS.n_elements(x) #no. of stars
        nis=len(plot_ages) #no. of isochrones to be plotted

        fig, ax = plt.subplots(figsize=(16,12))

        x_ext=MADYS.extinction(self.ebv,x_axis)
        y_ext=MADYS.extinction(self.ebv,y_axis)

        arr=[x_range[0]+0.2*(x_range[1]-x_range[0]),y_range[0]+0.1*(y_range[1]-y_range[0]),-np.median(x_ext),-np.median(y_ext)]
        ax.quiver(arr[0],arr[1],arr[2],arr[3],label='dereddening',scale=1,scale_units='xy',angles='xy')

        x1=x
        y1=y

        if type(plot_ages)!=bool:
            for i in range(len(plot_ages)):
                ii=MADYS.closest(iso_ages,plot_ages[i])
                plt.plot(col_th[:,ii],mag_th[:,ii],label=str(plot_ages[i])+' Myr')

        if type(plot_masses)!=bool:
            with np.errstate(divide='ignore',invalid='ignore'):
                for i in range(len(plot_masses)):
                    im=MADYS.closest(iso_masses,plot_masses[i])
                    plt.plot(col_th[im,:],mag_th[im,:],linestyle='dashed',color='gray')
                    c=0
                    while (np.isfinite(col_th[im,c])==0) | (np.isfinite(mag_th[im,c])==0) | ((col_th[im,c]<x_range[0]) | (col_th[im,c]>x_range[1])) & ((mag_th[im,c]<y_range[0]) | (mag_th[im,c]>y_range[1])): 
                        c+=1
                        if c==len(col_th[im,:]): break
                    if c<len(col_th[im,:]):
                        plt.annotate(str(plot_masses[i]),(col_th[im,c],mag_th[im,c]),size='large')

        if (type(groups)==type(None)):        
            if (type(x_error)==type(None)) & (type(y_error)==type(None)):
                plt.scatter(x1, y1, s=50, facecolors='none', edgecolors='black')
            else: plt.errorbar(x1, y1, yerr=y_error, xerr=x_error, fmt='o', color='black')
        else:
            nc=max(groups)
            colormap = plt.cm.gist_ncar
            colorst = [colormap(i) for i in np.linspace(0, 0.9,nc+1)]       
            for j in range(nc+1):
                w,=np.where(groups==j)
                if len(w)>0:  
                    if (type(x_error)==type(None)) & (type(y_error)==type(None)):
                        plt.scatter(x1[w], y1[w], s=50, facecolors='none', edgecolors=colorst[j], label=group_names[j])
                    else: plt.errorbar(x1[w], y1[w], yerr=y_error[w], xerr=x_error[w], fmt='o', color=colorst[j], label=group_names[j])

        if label_points==True:
            po=(np.linspace(0,npo-1,num=npo,dtype=int)).astype('str')
            for i, txt in enumerate(po):
                ax.annotate(txt, (x1[i], y1[i]))

        plt.ylim(y_range)
        plt.xlim(x_range)
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)
        plt.legend()
        if tofile==False:
            plt.show()
        elif tofile==True:
            img_file=Path(self.path) / (self.__sample_name+'_'+col+'_'+mag+'_'+model+'.png')            
            plt.savefig(img_file)
            plt.close(fig)    
        else:
            plt.savefig(tofile)
            plt.close(fig)    

        return None     

        
    @staticmethod
    def dr2_correct_flux_excess_factor(phot_g_mean_mag, bp_rp, phot_bp_rp_excess_factor):
        if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor) or np.isscalar(phot_g_mean_mag):
            bp_rp = np.float64(bp_rp)
            phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
            phot_g_mean_mag = np.float64(phot_g_mean_mag)
        if bp_rp.shape != phot_bp_rp_excess_factor.shape or bp_rp.shape != phot_g_mean_mag.shape or phot_g_mean_mag.shape != phot_bp_rp_excess_factor.shape:
            raise ValueError('Function parameters must be of the same shape!')
        a0=lambda x: -1.121221*np.heaviside(-(x-0.5),0)-1.1244509*np.heaviside(-(x-3.5),0)-(-1.1244509*np.heaviside(-(x-0.5),0))-0.9288966*np.heaviside(x-3.5,1)
        a1=lambda x: 0.0505276*np.heaviside(-(x-0.5),0)+0.0288725*np.heaviside(-(x-3.5),0)-(0.0288725*np.heaviside(-(x-0.5),0))-0.168552*np.heaviside(x-3.5,1)
        a2=lambda x: -0.120531*np.heaviside(-(x-0.5),0)-0.0682774*np.heaviside(-(x-3.5),0)-(-0.0682774*np.heaviside(-(x-0.5),0))
        a3=lambda x: 0.00795258*np.heaviside(-(x-3.5),0)-(0.00795258*np.heaviside(-(x-0.5),0))
        a4=lambda x: -0.00555279*np.heaviside(-(x-0.5),0)-0.00555279*np.heaviside(-(x-3.5),0)-(-0.00555279*np.heaviside(-(x-0.5),0))-0.00555279*np.heaviside(x-3.5,1)
        C1 = phot_bp_rp_excess_factor + a0(bp_rp)+a1(bp_rp)*bp_rp+a2(bp_rp)*bp_rp**2+a3(bp_rp)*bp_rp**3+a4(bp_rp)*phot_g_mean_mag #final corrected factor
        return C1

    @staticmethod
    def edr3_correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
        if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
            bp_rp = np.float64(bp_rp)
            phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)

        if bp_rp.shape != phot_bp_rp_excess_factor.shape:
            raise ValueError('Function parameters must be of the same shape!')

        do_not_correct = np.isnan(bp_rp)
        bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
        greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
        redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)

        correction = np.zeros_like(bp_rp)
        correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange], 2)
        correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange], 2) \
                - 0.005879*np.power(bp_rp[greenrange], 3)
        correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]

        return phot_bp_rp_excess_factor - correction

    @staticmethod
    def gaia_mag_errors(phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error):
        sigmaG_0 = 0.0027553202
        sigmaGBP_0 = 0.0027901700
        sigmaGRP_0 = 0.0037793818

        phot_g_mean_mag_error = np.sqrt((-2.5/np.log(10)*phot_g_mean_flux_error/phot_g_mean_flux)**2 + sigmaG_0**2)
        phot_bp_mean_mag_error = np.sqrt((-2.5/np.log(10)*phot_bp_mean_flux_error/phot_bp_mean_flux)**2 + sigmaGBP_0**2)
        phot_rp_mean_mag_error = np.sqrt((-2.5/np.log(10)*phot_rp_mean_flux_error/phot_rp_mean_flux)**2 + sigmaGRP_0**2)

        return phot_g_mean_mag_error, phot_bp_mean_mag_error, phot_rp_mean_mag_error

    @staticmethod
    def correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux):
        if np.isscalar(bp_rp) or np.isscalar(astrometric_params_solved) or np.isscalar(phot_g_mean_mag) \
                    or np.isscalar(phot_g_mean_flux):
            bp_rp = np.float64(bp_rp)
            astrometric_params_solved = np.int64(astrometric_params_solved)
            phot_g_mean_mag = np.float64(phot_g_mean_mag)
            phot_g_mean_flux = np.float64(phot_g_mean_flux)

        if not (bp_rp.shape == astrometric_params_solved.shape == phot_g_mean_mag.shape == phot_g_mean_flux.shape):
            raise ValueError('Function parameters must be of the same shape!')

        do_not_correct = np.isnan(bp_rp) | (phot_g_mean_mag<13) | (astrometric_params_solved == 31)
        bright_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>=13) & (phot_g_mean_mag<=16)
        faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>16)
        bp_rp_c = np.clip(bp_rp, 0.25, 3.0)

        correction_factor = np.ones_like(phot_g_mean_mag)
        correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + \
            0.01740*np.power(bp_rp_c[faint_correct],2) - 0.00253*np.power(bp_rp_c[faint_correct],3)
        correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + \
            0.01747*np.power(bp_rp_c[bright_correct],2) - 0.00277*np.power(bp_rp_c[bright_correct],3)


        gmag_corrected = phot_g_mean_mag - 2.5*np.log10(correction_factor)
        gflux_corrected = phot_g_mean_flux * correction_factor

        return gmag_corrected, gflux_corrected

    @staticmethod
    def dr2_quality(dr2_bp_rp_excess_factor,dr2_phot_g_mean_mag):
        s1=lambda x: 0.004+8e-12*x**7.55
        with np.errstate(invalid='ignore'):
            q1=np.where(abs(dr2_bp_rp_excess_factor)>3*s1(dr2_phot_g_mean_mag),False,True)
        q1[dr2_bp_rp_excess_factor.mask]=False
        return q1

    @staticmethod
    def edr3_quality(edr3_bp_rp_excess_factor,edr3_phot_g_mean_mag):
        s1=lambda x: 0.0059898+8.817481e-12*x**7.618399
        with np.errstate(invalid='ignore'):
            q1=np.where(abs(edr3_bp_rp_excess_factor)>3*s1(edr3_phot_g_mean_mag),False,True)
        q1[edr3_bp_rp_excess_factor.mask]=False
        return q1

    @staticmethod    
    def tmass_quality(ph_qual,max_q='A'):
        q=np.array(['A','B','C','D','E','F','U','X'])
        w,=np.where(q==max_q)[0]
        n=len(ph_qual)
        qJ=np.zeros(n,dtype=bool)
        qH=np.zeros(n,dtype=bool)
        qK=np.zeros(n,dtype=bool)
        for i in range(n):
            if ph_qual.mask[i]==True: continue
            if bool(ph_qual[i])==False: continue
            if ph_qual[i][0] in q[0:w+1]: qJ[i]=True
            if ph_qual[i][1] in q[0:w+1]: qH[i]=True
            if ph_qual[i][2] in q[0:w+1]: qK[i]=True
        return qJ,qH,qK
    
    @staticmethod    
    def allwise_quality(cc_flags,ph_qual2,max_q='A'):
        q=np.array(['A','B','C','U','Z','X'])    
        w,=np.where(q==max_q)[0]
        n=len(ph_qual2)
        qW1=np.zeros(n,dtype=bool)
        qW2=np.zeros(n,dtype=bool)
        qW3=np.zeros(n,dtype=bool)
        qW4=np.zeros(n,dtype=bool)
        for i in range(n):
            if (ph_qual2.mask[i]==True) | (cc_flags.mask[i]==True): continue
            if (bool(ph_qual2[i])==False) | (bool(cc_flags[i])==False): continue
            if (ph_qual2[i][0] in q[0:w+1]) & (cc_flags[i][0]=='0'): qW1[i]=True
            if (ph_qual2[i][1] in q[0:w+1]) & (cc_flags[i][1]=='0'): qW2[i]=True
            if (ph_qual2[i][2] in q[0:w+1]) & (cc_flags[i][2]=='0'): qW3[i]=True
            if (ph_qual2[i][3] in q[0:w+1]) & (cc_flags[i][3]=='0'): qW4[i]=True
        return qW1,qW2,qW3,qW4
    
    def fix_2mass(self,t):
        
        #cerca le stelle senza entry 2MASS
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            tt1=time.perf_counter()
            if 'wise' in self.surveys:
                w,=np.where((t['j_m'].mask==True) & (t['tmass_key'].mask==False)) #stelle con cross-match 2MASS
                w1=w
                i1=np.arange(len(w))
                i2=i1
                t1=t
            else:
                w,=np.where(t['j_m'].mask==True)

                if len(w)==0: return t
                print('Stars without 2MASS photometry: ',len(w))

            #effettua una ricerca su WISE per queste stesse stelle
                query_list=self.list_chunk(w) #SE WISE è GIà NELLE SURVEYS, EVITARE DI FARE QUESTA RICERCA E RECUPERARE I DATI GIà PRESENTI
    #            qstr=self.query_string(query_list,surveys=['gaia','wise'])
                qstr=self.query_string('',surveys=['gaia','wise'])
        
                t1=self.divide_query(qstr,id_list=np.array(self.GaiaID['ID'])[w],n_it_max=10,engine='gaia')
    #            adql = QueryStr(qstr,verbose=False)
    #            t1=gaia.query(adql)            

                t1=self.fix_double_entries(t1,index=w)

                w1,=np.where(t1['tmass_key'].mask==False) #stelle con cross-match 2MASS

                if self.__id_type=='EDR3': 
                    id_res=np.array(t1['edr3_id'][w1])
                    l=np.array(t['edr3_id'][w])            
                else: 
                    id_res=np.array(t1['dr2_id'][w1])
                    l=np.array(t['dr2_id'][w])        

                __, i1, i2 = np.intersect1d(id_res,l,return_indices=True) #cross-match tra t[w] (=stelle della vecchia ricerca senza 2MASS) e t1[w1] (=stelle della nuova ricerca con 2MASS)        

            print('Stars without 2MASS photometry with ALLWISE cross-match: ',len(i1))
            if len(i1)>0:
            #ricerca le stelle t1[w1[i1]] sul catalogo ALLWISE principale
                wise_ids = np.array(t1['allwise_id'][w1[i1]])
#                query_list=''
#                for ii in range(len(i1)):
#                    id=wise_ids[ii]
#                    query_list+='allwise.AllWISE LIKE '+"'"+id+"'"+' OR '
#                query_list=query_list[:-4]

#                qstr = """
#                 SELECT all
#                     allwise.AllWISE, allwise.RAJ2000, allwise.DEJ2000, allwise.W1mag, allwise.W2mag, 
#                 allwise.W3mag, allwise.W4mag, allwise.Jmag, allwise.e_Jmag, allwise.Hmag, allwise.e_Hmag, 
#                 allwise.Kmag, allwise.e_Kmag, allwise.ccf, allwise.d2M, allwise."2Mkey"
#                 FROM "II/328/allwise" as allwise
#                where """ + query_list

#                adql = QueryStr(qstr,verbose=False)                
#                res = vizier.query(adql)

                qstr = """
                 SELECT all
                     allwise.AllWISE, allwise.RAJ2000, allwise.DEJ2000, allwise.W1mag, allwise.W2mag, 
                 allwise.W3mag, allwise.W4mag, allwise.Jmag, allwise.e_Jmag, allwise.Hmag, allwise.e_Hmag, 
                 allwise.Kmag, allwise.e_Kmag, allwise.ccf, allwise.d2M, allwise."2Mkey"
                 FROM "II/328/allwise" as allwise
                where """

                res=self.divide_query(qstr,id_list=wise_ids,n_it_max=10,engine='vizier',key_name='allwise.AllWISE',equality='LIKE',quote_mark=True)

                wise_res = np.array(res['AllWISE'])
                __, i3, i4 = MADYS.intersect1d_rep1(wise_ids,wise_res) #cross-match tra t[w] (=stelle della vecchia ricerca senza 2MASS) e t1[w1] (=stelle della nuova ricerca con 2MASS)

                names=['j_m', 'h_m','ks_m','j_msigcom', 'h_msigcom','ks_msigcom']
                t_ext=Table([res['Jmag'][i4],res['Hmag'][i4],res['Kmag'][i4],res['e_Jmag'][i4],res['e_Hmag'][i4],res['e_Kmag'][i4]],
                    names=names,
                    units=["mag","mag","mag","mag","mag","mag"])
                for i in names: t[i][w[i2[i3]]]=t_ext[i]

                #devo recuperare i quality flags di 2MASS, le coordinate e l'ID!

                n_chunks=1
                nst=len(i3)
                qstr0 = """
                 SELECT ALL
                  tmass.RAJ2000, tmass.DEJ2000, tmass."2MASS", tmass.Jmag, tmass.e_Jmag, tmass.Hmag, tmass.e_Hmag, 
                 tmass.Kmag, tmass.e_Kmag, tmass.Qflg, tmass.Rflg, tmass.Bflg, tmass.Cflg, tmass.Xflg, tmass.Aflg, 
                 tmass.Cntr
                 FROM "II/246/out" as tmass
                 WHERE """

                done=np.zeros(nst,dtype=bool)
                nit=0
                data=[]
                while (np.sum(done)<nst) & (nit<10):
                    todo,=np.where(done==False)
                    st=int(len(todo)/n_chunks)
                    for i in range(n_chunks):
                        todo_c=todo[i*st:(i+1)*st]

                        query_list=''
                        for ii in todo_c:
                            jstr='tmass.Jmag > '+'{:.3f}'.format(t_ext['j_m'][ii]-0.001)+' AND tmass.Jmag <'+'{:.3f}'.format(t_ext['j_m'][ii]+0.001)
                            hstr='tmass.Hmag > '+'{:.3f}'.format(t_ext['h_m'][ii]-0.001)+' AND tmass.Hmag <'+'{:.3f}'.format(t_ext['h_m'][ii]+0.001)
                            kstr='tmass.Kmag > '+'{:.3f}'.format(t_ext['ks_m'][ii]-0.001)+' AND tmass.Kmag <'+'{:.3f}'.format(t_ext['ks_m'][ii]+0.001)
                            query_list+=' ( '+jstr+' AND '+hstr+' AND '+kstr+' ) OR'
                        query_list=query_list[:-3]
                        qstr=qstr0+query_list
                        try:
                            adql = QueryStr(qstr,verbose=False)                
                            res = vizier.query(adql)
                            data.append(res)
                            done[todo_c]=True
                        except: continue
                    n_chunks*=2
                    nit+=1
                    if nit>9: raise RuntimeError('Perhaps '+nst+' stars are too many?')

                if len(data)>1: res=vstack(data)
                else: res=data[0]

                cntr_res = np.array(res['Cntr'])
                cntr_old = np.array(t1['tmass_key'][w1[i1]])
                __, i5, i6 = MADYS.intersect1d_rep1(cntr_old,cntr_res) #cross-match tra t[w] (=stelle della vecchia ricerca senza 2MASS) e t1[w1] (=stelle della nuova ricerca con 2MASS)

                if self.__id_type=='EDR3':
                    dd=3600*MADYS.ang_dist(t['ra'][w[i2[i5]]].value-(t['edr3_epoch'][w[i2[i5]]].value-2000)*t['edr3_pmra'][w[i2[i5]]].value/3.6e+6,t['dec'][w[i2[i5]]].value-(t['edr3_epoch'][w[i2[i5]]].value-2000)*t['edr3_pmdec'][w[i2[i5]]].value/3.6e+6,res['RAJ2000'][i6].value,res['DEJ2000'][i6].value)
                else:
                    dd=3600*MADYS.ang_dist(t['dr2_ra'][w[i2[i5]]].value-(t['dr2_epoch'][w[i2[i5]]].value-2000)*t['dr2_pmra'][w[i2[i5]]].value/3.6e+6,t['dr2_dec'][w[i2[i5]]].value-(t['dr2_epoch'][w[i2[i5]]].value-2000)*t['dr2_pmdec'][w[i2[i5]]].value/3.6e+6,res['RAJ2000'][i6].value,res['DEJ2000'][i6].value)

                w_ncm,=np.where(dd>0.7)

                names=['tmass_id','ph_qual','tmass_ra','tmass_dec']
                t_ext=Table([res['_2MASS'][i6],res['Qflg'][i6],res['RAJ2000'][i6],res['DEJ2000'][i6]],
                    names=names,
                    units=["","","deg","deg"])
                t_ext['ph_qual']=MaskedColumn(t_ext['ph_qual'],dtype=object)
                for i in names: t[i][w[i2[i5]]]=t_ext[i]
                    
                names=['tmass_id','ph_qual','tmass_ra','tmass_dec','j_m', 'h_m','ks_m','j_msigcom', 'h_msigcom','ks_msigcom']
                for i in names: t[i].mask[w[i2[i5[w_ncm]]]]=True

                print("Recovered stars (2MASS source within 0.7''): ",len(i5)-len(w_ncm))
            else:
                i5=[]
                w_ncm=[]

            tt2=time.perf_counter()
            print('Time for ALLWISE recovery: ',tt2-tt1,' s')


            w,=np.where(t['j_m'].mask==True)
            if len(w)==0: return t
            print('Stars still without 2MASS: ',len(w))

            key1='_2MASS'
            l=[]
            ind=[]

            for j in range(len(w)):
                found=False
                x=Simbad.query_objectids(t['edr3_id'][w[j]])
                if type(x)==type(None): x=Simbad.query_objectids(t['dr2_id'][w[j]])            
                if type(x)!=type(None):
                    for rr in x:
                        if str(rr[0]).startswith('2MASS'): 
                            key2=str(rr[0]).replace('2MASS J','')
                            found=True
                            break
                if found: 
                    l.append(key2)
                    ind.append(w[j])
                if time.perf_counter()-tt2>180: #exceeded 3 minutes
                    print('exceeded 3 minutes')
                    break

            if len(l)>0:
                ind=np.array(ind)

                query_list=''
                for i in range(len(l)):
                    id=l[i]
                    query_list+='tmass."2MASS" = '+"'"+id+"'"+' OR '
                query_list=query_list[:-4]            

                qstr = """
                select all
                    tmass.RAJ2000, tmass.DEJ2000, tmass."2MASS", tmass.Jmag, tmass.e_Jmag, tmass.Hmag, tmass.e_Hmag, 
                    tmass.Kmag, tmass.e_Kmag, tmass.Qflg
                from 
                    "II/246/out" as tmass
                where """ + query_list

                adql = QueryStr(qstr,verbose=False)

                res = vizier.query(adql)

                l=np.array(l)
                id_res=np.array(res['_2MASS'])
                __, i1, i2 = np.intersect1d(id_res,l,return_indices=True)

                names=['j_m', 'h_m','ks_m','j_msigcom', 'h_msigcom','ks_msigcom','ph_qual','tmass_ra','tmass_dec','tmass_id']
                t_ext=Table([res['Jmag'][i1],res['Hmag'][i1],res['Kmag'][i1],res['e_Jmag'][i1],res['e_Hmag'][i1],res['e_Kmag'][i1],res['Qflg'][i1],res['RAJ2000'][i1],res['DEJ2000'][i1],res['_2MASS'][i1]],
                    names=names,
                    units=["mag","mag","mag","mag","mag","mag","","deg","deg",""])
                t_ext['ph_qual']=MaskedColumn(t_ext['ph_qual'],dtype=object)
                for i in names: t[i][ind[i2]]=t_ext[i]     
            else: i2=[]

            tt3=time.perf_counter()
            print("Recovered stars (2MASS cross-match in Simbad)",len(i2))
            print('Total number of stars with recovered 2MASS photometry: ',len(i2)+len(i5)-len(w_ncm))
            print('Time for individual Simbad query: ',tt3-tt2,' s')

        return t
    
    @staticmethod    
    def fix_edr3(t):
        if t['ra'].mask==False: return t

        ra0=t['dr2_ra'].value[0]
        dec0=t['dr2_dec'].value[0]
        pmra0=t['dr2_pmra'].value[0]
        pmdec0=t['dr2_pmdec'].value[0]
        ep=t['dr2_epoch'].value[0]
        ra1=ra0+(2016-ep)*pmra0/3.6e+6
        dec1=dec0+(2016-ep)*pmdec0/3.6e+6

        r=np.sqrt((pmra0/1000)**2+(pmdec0/1000)**2)*1.5*(2016-ep)
        if np.isnan(r): return t
        r=str(r)+'s'
        v = Vizier(columns=["*", "+_r"], catalog="I/350/gaiaedr3")
        no_res=False
        try:
            res=v.query_region(SkyCoord(ra=ra1, dec=dec1,unit=(u.deg, u.deg),frame='icrs'),width=r,catalog=["I/350/gaiaedr3"])[0]
        except IndexError: 
            no_res=True
            l=0
        else:
            l=len(res)
            if l>1:
                no_res=True
                if np.sum(res['pmRA'].mask)<l:
                    w=np.argmin(np.abs(res['pmRA']-pmra0))
                    res=res[w]
                    if np.abs(res['Gmag']-t['dr2_phot_g_mean_mag'].value[0])<0.2: no_res=False
                elif np.sum(res['pmDE'].mask)<l:
                    w=np.argmin(np.abs(res['pmDE']-pmdec0))
                    res=res[w]
                    if np.abs(res['Gmag']-t['dr2_phot_g_mean_mag'].value[0])<0.2: no_res=False
                elif np.sum(res['Plx'].mask)<l:
                    w=np.argmin(np.abs(res['Plx']-t['dr2_parallax'].value[0]))
                    res=res[w]
                    if np.abs(res['Gmag']-t['dr2_phot_g_mean_mag'].value[0])<0.2: no_res=False
        finally:
            if (no_res==False) & (l>0):
                id=str(res['Source'])
                qstr="""
                select all
                edr3.designation as edr3_id,
                edr3.ra as ra, edr3.dec as dec,
                edr3.ref_epoch as edr3_epoch, edr3.parallax as edr3_parallax,
                edr3.parallax_error as edr3_parallax_error, edr3.parallax_over_error as edr3_parallax_over_error,
                edr3.pmra as edr3_pmra, edr3.pmra_error as edr3_pmra_error,
                edr3.pmdec as edr3_pmdec, edr3.pmdec_error as edr3_pmdec_error,
                edr3.ra_dec_corr as edr3_ra_dec_corr, edr3.ra_parallax_corr as edr3_ra_parallax_corr,
                edr3.ra_pmra_corr as edr3_ra_pmra_corr, edr3.ra_pmdec_corr as edr3_ra_pmdec_corr,
                edr3.dec_parallax_corr as edr3_dec_parallax_corr,
                edr3.dec_pmra_corr as edr3_dec_pmra_corr, edr3.dec_pmdec_corr as edr3_dec_pmdec_corr,
                edr3.parallax_pmra_corr as edr3_parallax_pmra_corr, edr3.parallax_pmdec_corr as edr3_parallax_pmdec_corr,
                edr3.pmra_pmdec_corr as edr3_pmra_pmdec_corr, edr3.phot_g_mean_mag as edr3_phot_g_mean_mag,
                edr3.phot_g_mean_flux as edr3_phot_g_mean_flux, edr3.phot_g_mean_flux_error as edr3_phot_g_mean_flux_error,
                edr3.phot_bp_mean_flux as edr3_phot_bp_mean_flux, edr3.phot_bp_mean_flux_error as edr3_phot_bp_mean_flux_error,
                edr3.phot_bp_mean_mag as edr3_phot_bp_mean_mag,
                edr3.phot_rp_mean_flux as edr3_phot_rp_mean_flux, edr3.phot_rp_mean_flux_error as edr3_phot_rp_mean_flux_error,
                edr3.phot_rp_mean_mag as edr3_phot_rp_mean_mag,
                edr3.bp_rp as edr3_bp_rp, edr3.phot_bp_rp_excess_factor as edr3_phot_bp_rp_excess_factor,
                edr3.ruwe as edr3_ruwe, edr3.astrometric_params_solved as edr3_astrometric_params_solved
                from
                    gaiaedr3.gaia_source as edr3
                WHERE edr3.source_id = """ + id
                adql = QueryStr(qstr,verbose=False)
                try:
                    t2=gaia.query(adql)
                except:
                    pass
                else:
                    col=t2.colnames
                    for i in range(len(col)):
                        t[col[i]]=t2[col[i]]
            return t                

    #################################################################
    # UTILITY FUNCTIONS
    @staticmethod
    def n_elements(x):
        size = 1
        for dim in np.shape(x): size *= dim
        return size

    @staticmethod
    def where_v(elements,array,approx=False):

        if isinstance(array,list): array=np.array(array)
        try:
            dd=len(elements)
            if isinstance(elements,list): elements=np.array(elements)
            dim=len(elements.shape)
        except TypeError: dim=0

        if approx==True:
            if dim==0: 
                w=MADYS.closest(array,elements)
                return w
            ind=np.zeros(len(elements),dtype=np.int16)
            for i in range(len(elements)):
                ind[i]=MADYS.closest(array,elements[i])
            return ind
        else:
            if dim==0: 
                w,=np.where(array==elements)
                return w
            ind=np.zeros(len(elements),dtype=np.int16)
            for i in range(len(elements)):
                w,=np.where(array==elements[i])
                if len(w)==0: ind[i]=len(array) #so it will raise an error
                else: ind[i]=w[0]
            return ind

    @staticmethod
    def closest(array,value):
        '''Given an "array" and a (list of) "value"(s), finds the j(s) such that |array[j]-value|=min((array-value)).
        "array" must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.'''
        n = len(array)
        nv=MADYS.n_elements(value)
        if nv==1:
            if (value < array[0]):
                return 0
            elif (value > array[n-1]):
                return n-1
            jl = 0# Initialize lower
            ju = n-1# and upper limits.
            while (ju-jl > 1):# If we are not yet done,
                jm=(ju+jl) >> 1# compute a midpoint with a bitshift
                if (value >= array[jm]):
                    jl=jm# and replace either the lower limit
                else:
                    ju=jm# or the upper limit, as appropriate.
                # Repeat until the test condition is satisfied.
            if (value == array[0]):# edge cases at bottom
                return 0
            elif (value == array[n-1]):# and top
                return n-1
            else:
                jn=jl+np.argmin([value-array[jl],array[jl+1]-value])
                return jn
        else:
            jn=np.zeros(nv,dtype='int32')
            for i in range(nv):
                if (value[i] < array[0]): jn[i]=0
                elif (value[i] > array[n-1]): jn[i]=n-1
                else:
                    jl = 0# Initialize lower
                    ju = n-1# and upper limits.
                    while (ju-jl > 1):# If we are not yet done,
                        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
                        if (value[i] >= array[jm]):
                            jl=jm# and replace either the lower limit
                        else:
                            ju=jm# or the upper limit, as appropriate.
                        # Repeat until the test condition is satisfied.
                    if (value[i] == array[0]):# edge cases at bottomlog
                        jn[i]=0
                    elif (value[i] == array[n-1]):# and top
                        jn[i]=n-1
                    else:
                        jn[i]=jl+np.argmin([value[i]-array[jl],array[jl+1]-value[i]])
            return jn
    
    @staticmethod
    def min_v(a,absolute=False):
        """
        handles n-dimensional array with nan, returning the minimum valid value and its n-dim index
        can return the minimum absolute value, too

        e.g., a=[[3,1,4],[-7,-0.1,8]]
        min_v(a) returns -7 and (1,0)
        min_v(a,key=abs) returns -0.1 and (1,1)
        """

        if absolute: ind = np.unravel_index(np.nanargmin(abs(a), axis=None), a.shape)
        else: ind = np.unravel_index(np.nanargmin(a, axis=None), a.shape)

        return a[ind],ind

    @staticmethod
    def file_search(files):
        """
        given one or more files, returns 1 if all of them exist, 0 otherwise
        if n_elements(files)==1:

        input:
            files: a string, a list of strings or a numpy array of strings,
                specifying the full path of the file(s) whose existence is questioned

        usage:
            c=[file1,file2,file3]
            file_search(c)=1 if all files exist, 0 otherwise

        notes:
        case insensitive.

        """
        if (isinstance(files,str)) | (isinstance(files,Path)):
            try:
                open(files,'r')
            except IOError:
                return 0
        else:
            for i in range(MADYS.n_elements(files)):
                try:
                    open(files[i],'r')
                except IOError:
                    return 0
        return 1

    @staticmethod
    def split_if_nan(a):
        ind=[]
        res=[]
        for s in np.ma.clump_unmasked(np.ma.masked_invalid(a)):
            ind.append(s)
            res.append(a[s])    
        return res,ind

    @staticmethod
    def complement_v(arr,n):
        compl=np.full(n,True)
        compl[arr]=False
        compl,=np.where(compl==True)
        return compl

    #################################################################
    # ASTRONOMICAL FUNCTIONS

    @staticmethod
    def Wu_line_integrate(f,x0,x1,y0,y1,z0,z1,layer=None,star_id=None,logger=None):
        dim=f.shape
        while np.max(np.abs([x1,y1,z1]))>2*np.max(dim):
            x1=x0+(x1-x0)/2
            y1=y0+(y1-y0)/2
            z1=z0+(z1-z0)/2
        n=int(10*np.ceil(abs(max([x1-x0,y1-y0,z1-z0],key=abs))))    
        ndim=len(dim)

        x=np.floor(np.linspace(x0,x1,num=n)).astype(int)
        y=np.floor(np.linspace(y0,y1,num=n)).astype(int)
        
        if type(layer)==type(None):
            if ndim==2:
                d10=np.sqrt((x1-x0)**2+(y1-y0)**2) #distance 
                w_g,=np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]))
                w_g=np.insert(w_g+1,0,0)
                w_f=np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])            
                w,=np.where((x[w_g]<dim[0]) & (y[w_g]<dim[1]))
                if (len(w)<len(w_g)) & (logger!=None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')                
                w2=w_g[w]
                I=np.sum(f[x[w2],y[w2]]*w_f[w])
            elif ndim==3:
                d10=np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2) #distance        
                z=np.floor(np.linspace(z0,z1,num=n)).astype(int)           
                w_g,=np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]) | (z[1:]!=z[:-1]))
                w_g=np.insert(w_g+1,0,0)
                w_f=np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])            
                w,=np.where((x[w_g]<dim[0]) & (y[w_g]<dim[1]) & (z[w_g]<dim[2]) & (x[w_g]>=0) & (y[w_g]>=0) & (z[w_g]>=0))
                if (len(w)<len(w_g)) & (logger!=None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')                
                w2=w_g[w]
                I=np.sum(f[x[w2],y[w2],z[w2]]*w_f[w])
        else:
            if ndim==3:
                d10=np.sqrt((x1-x0)**2+(y1-y0)**2) #distance
                w_g,=np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]))
                w_g=np.insert(w_g+1,0,0)
                w_f=np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])            
                w,=np.where((x[w_g]<dim[1]) & (y[w_g]<dim[2]))
                if (len(w)<len(w_g)) & (logger!=None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')                
                w2=w_g[w]
                I=np.sum(f[layer,x[w2],y[w2]]*w_f[w])
            elif ndim==4:
                d10=np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2) #distance        
                z=np.floor(np.linspace(z0,z1,num=n)).astype(int)
                w_g,=np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]) | (z[1:]!=z[:-1]))
                w_g=np.insert(w_g+1,0,0)
                w_f=np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])            
                w,=np.where((x[w_g]<dim[1]) & (y[w_g]<dim[2]) & (z[w_g]<dim[3]))
                if (len(w)<len(w_g)) & (logger!=None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')                
                w2=w_g[w]
                I=np.sum(f[layer,x[w2],y[w2],z[w2]]*w_f[w])

        return I/n*d10

    @staticmethod
    def interstellar_ext(ra=None,dec=None,l=None,b=None,par=None,d=None,ext_map='leike',color='B-V',error=False,logger=None):

        if (ext_map=='leike') & (error==False): fname='leike_mean_std.h5'
        elif (ext_map=='leike') & (error==True): fname='leike_samples.h5'
        if (ext_map=='stilism'): fname='stilism_feb2019.h5'

        folder = os.path.dirname(os.path.realpath(__file__))

        paths=[x[0] for x in os.walk(folder)]
        found = False
        for path in paths:
            if (Path(path) / fname).exists():
                map_path = path
                found = True
                break
        if not found:
    #        raise ValueError('Extinction map not found! Setting extinction to zero.')
            print('Extinction map not found! Setting extinction to zero.')
            if MADYS.n_elements(ra)>1: ebv=np.zeros(MADYS.n_elements(ra))
            else: ebv=0.
            return ebv

        fits_image_filename=os.path.join(map_path,fname)
        f = h5py.File(fits_image_filename,'r')
        if ext_map=='leike': 
            x=np.arange(-370.,370.)
            y=np.arange(-370.,370.)
            z=np.arange(-270.,270.)
            if error==False: obj='mean'
            else: obj='dust_samples'
            data = f[obj][()]
        elif ext_map=='stilism': 
            x=np.arange(-3000.,3005.,5)
            y=np.arange(-3000.,3005.,5)
            z=np.arange(-400.,405.,5)    
            data = f['stilism']['cube_datas'][()]

        if type(ra)==type(None) and type(l)==type(None): raise NameError('One between RA and l must be supplied!')
        if type(dec)==type(None) and type(b)==type(None): raise NameError('One between dec and b must be supplied!')
        if type(par)==type(None) and type(d)==type(None): raise NameError('One between parallax and distance must be supplied!')
        if type(ra)!=type(None) and type(l)!=type(None): raise NameError('Only one between RA and l must be supplied!')
        if type(dec)!=type(None) and type(b)!=type(None): raise NameError('Only one between dec and b must be supplied!')
        if type(par)!=type(None) and type(d)!=type(None): raise NameError('Only one between parallax and distance must be supplied!')

        sun=[MADYS.closest(x,0),MADYS.closest(z,0)]

        #  ;Sun-centered Cartesian Galactic coordinates (right-handed frame)
        if type(d)==type(None): 
            try:
                len(par)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    par=np.where(par<0,np.nan,par)
            except TypeError:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if par<0: par=np.nan
            d=1000./par #computes heliocentric distance, if missing
        if type(ra)!=type(None): #equatorial coordinates
            c1 = SkyCoord(ra=ra*u.degree, dec=dec*u.degree,
                                distance=d*u.pc,
                                frame='icrs')
        else:
            c1 = SkyCoord(l=l*u.degree, b=b*u.degree, #galactic coordinates
                                distance=d*u.pc,
                                frame='galactic')

        galactocentric_frame_defaults.set('pre-v4.0')
        gc1 = c1.transform_to(Galactocentric)
        x0=(gc1.x+gc1.galcen_distance).value #X is directed to the Galactic Center
        y0=gc1.y.value #Y is in the sense of rotation
        z0=(gc1.z-gc1.z_sun).value #Z points to the north Galactic pole    

        px=MADYS.closest(x,x0)
        py=MADYS.closest(y,y0)
        pz=MADYS.closest(z,z0)

        dist=x[1]-x[0]

        try:
            len(px)        
            wx,=np.where(px<len(x)-1)
            wy,=np.where(py<len(y)-1)
            wz,=np.where(pz<len(z)-1)
            px2=px.astype(float)
            py2=py.astype(float)    
            pz2=pz.astype(float)
            px2[wx]=(x0[wx]-x[px[wx]])/dist+px[wx]
            py2[wy]=(y0[wy]-y[py[wy]])/dist+py[wy]
            pz2[wz]=(z0[wz]-z[pz[wz]])/dist+pz[wz]    
            ebv=np.full(len(x0),np.nan)
            if ext_map=='stilism':
                for i in range(MADYS.n_elements(x0)):
                    if np.isnan(px2[i])==0:
                        ebv[i]=dist*MADYS.Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i],star_id=i,logger=logger)/3.16
            elif ext_map=='leike':
                if error==False: 
                    for i in range(MADYS.n_elements(x0)):
                        if np.isnan(px2[i])==0:
                            ebv[i]=dist*(2.5*MADYS.Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i],star_id=i,logger=logger)*np.log10(np.exp(1)))/3.16/0.789
                else:
                    dim=data.shape
                    ebv0=np.full([len(x0),dim[0]],np.nan)
                    ebv_s=np.full(len(x0),np.nan)
                    for i in range(MADYS.n_elements(x0)):
                        if np.isnan(px2[i])==0:
                            for k in range(dim[0]):
                                ebv0[i,k]=dist*(2.5*MADYS.Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i],layer=k,star_id=i,logger=logger)*np.log10(np.exp(1)))/3.16/0.789
                        ebv[i]=np.mean(ebv0[i,:])
                        ebv_s[i]=np.std(ebv0[i,:],ddof=1) #sample std dev                
        except TypeError:
            if px<len(x)-1: px2=(x0-x[px])/dist+px
            else: px2=px
            if py<len(y)-1: py2=(y0-y[py])/dist+py
            else: py2=py
            if pz<len(z)-1: pz2=(z0-z[pz])/dist+pz
            else: pz2=pz
            if isinstance(px2,np.ndarray): px2=px2[0]
            if isinstance(py2,np.ndarray): py2=py2[0]
            if isinstance(pz2,np.ndarray): pz2=pz2[0]
            if ext_map=='stilism': ebv=dist*MADYS.Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2,star_id=0,logger=logger)/3.16
            elif ext_map=='leike': 
                if error==False:
                    if np.isnan(px2)==0:
                        ebv=dist*(2.5*MADYS.Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2,star_id=0,logger=logger)*np.log10(np.exp(1)))/3.16/0.789
                    else: return np.nan
                else:
                    dim=data.shape
                    ebv0=np.zeros(dim[0])
                    if np.isnan(px2)==0:
                        for k in range(dim[0]):
                            ebv0[k]=dist*(2.5*MADYS.Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2,layer=k)*np.log10(np.exp(1)))/3.16/0.789
                    else: return np.nan,np.nan
                    ebv=np.mean(ebv0)
                    ebv_s=np.std(ebv0,ddof=1) #sample std dev

        if color=='B-V': 
            if error==False:
                return ebv
            else: return ebv,ebv_s
        else: 
            if error==False:
                return MADYS.extinction(ebv,color)
            else:
                return MADYS.extinction(ebv,color),MADYS.extinction(ebv_s,color)

    @staticmethod
    def app_to_abs_mag(app_mag,parallax,app_mag_error=None,parallax_error=None,ebv=None,filters=None):
        if isinstance(app_mag,list): app_mag=np.array(app_mag)
        if (isinstance(parallax,list)) | (isinstance(parallax,Column)): parallax=np.array(parallax,dtype=float)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dm=5*np.log10(100./parallax) #modulo di distanza

        try:
            dd=len(app_mag)
            dim=len(app_mag.shape)
        except TypeError: dim=0

        if dim <= 1:
            abs_mag=app_mag-dm
            if type(ebv)!=type(None):
                if dim==0: red=MADYS.extinction(ebv,filters[0])
                else: red=np.array([MADYS.extinction(ebv,filt) for filt in filters])
                abs_mag-=red
            if (type(app_mag_error)!=type(None)) & (type(parallax_error)!=type(None)): 
                if isinstance(app_mag_error,list): app_mag_error=np.array(app_mag_error)
                if (isinstance(parallax_error,list)) | (isinstance(parallax_error,Column)): parallax_error=np.array(parallax_error,dtype=float)
                total_error=np.sqrt(app_mag_error**2+(5/np.log(10)/parallax)**2*parallax_error**2)
                result=(abs_mag,total_error)
            else: result=abs_mag
        else: #  se è 2D, bisogna capire se ci sono più filtri e se c'è anche l'errore fotometrico
            l=app_mag.shape
            abs_mag=np.empty([l[0],l[1]])
            for i in range(l[1]): abs_mag[:,i]=app_mag[:,i]-dm
            if type(parallax_error)!=type(None):
                if isinstance(app_mag_error,list): app_mag_error=np.array(app_mag_error)
                if (isinstance(parallax_error,list)) | (isinstance(parallax_error,Column)): parallax_error=np.array(parallax_error,dtype=float)
                total_error=np.empty([l[0],l[1]])
                for i in range(l[1]): 
                    total_error[:,i]=np.sqrt(app_mag_error[:,i]**2+(5/np.log(10)/parallax)**2*parallax_error**2)
                result=(abs_mag,total_error)
            else: result=abs_mag
            if type(ebv)!=type(None):
                red=np.zeros([l[0],l[1]]) #reddening
                for i in range(l[1]): 
                    red[:,i]=MADYS.extinction(ebv,filters[i])
                abs_mag-=red        
        return result #se l'input è un array 1D, non c'è errore ed è un unico filtro

    @staticmethod
    def extinction(ebv,col):
        A_law={'B':1.317,'V':1,
             'G':0.789,'G2':0.789,'Gbp':1.002,'Gbp2':1.002,'Grp':0.589,'Grp2':0.589,
             'J':0.243,'H':0.131,'K':0.078,
             'g':1.155,'r':0.843,'i':0.628,'z':0.487,'y':0.395,
             'H_F110W':0.2966,'H_F160W':0.1556,
             'NIRCAM_c210_F182M':0.1058,'NIRCAM_c210_F187N':0.1018,'NIRCAM_c210_F200W':0.0919,
             'NIRCAM_cSWB_F182M':0.1058,'NIRCAM_cSWB_F187N':0.1018,'NIRCAM_cSWB_F200W':0.0919,
             'NIRCAM_p_F070Wa':0.6919,'NIRCAM_p_F070Wab':0.6919,'NIRCAM_p_F070Wb':0.6919,'NIRCAM_p_F090Wa':0.4523,
             'NIRCAM_p_F090Wab':0.4523,'NIRCAM_p_F090Wb':0.4523,'NIRCAM_p_F115Wa':0.2785,'NIRCAM_p_F115Wab':0.2785,
             'NIRCAM_p_F115Wb':0.2785,'NIRCAM_p_F140Ma':0.1849,'NIRCAM_p_F140Mab':0.1849,'NIRCAM_p_F140Mb':0.1849,
             'NIRCAM_p_F150Wa':0.1618,'NIRCAM_p_F150Wab':0.1618,'NIRCAM_p_F150Wb':0.1618,'NIRCAM_p_F150W2a':0.1519,
             'NIRCAM_p_F150W2ab':0.1519,'NIRCAM_p_F150W2b':0.1519,'NIRCAM_p_F162Ma':0.1361,'NIRCAM_p_F162Mab':0.1361,
             'NIRCAM_p_F162Mb':0.1361,'NIRCAM_p_F164Na':0.1329,'NIRCAM_p_F164Nab':0.1329,'NIRCAM_p_F164Nb':0.1329,
             'NIRCAM_p_F182Ma':0.1058,'NIRCAM_p_F182Mab':0.1058,'NIRCAM_p_F182Mb':0.1058,'NIRCAM_p_F187Na':0.1018,
             'NIRCAM_p_F187Nab':0.1018,'NIRCAM_p_F187Nb':0.1018,'NIRCAM_p_F200Wa':0.0919,'NIRCAM_p_F200Wab':0.0919,
             'NIRCAM_p_F200Wb':0.0919,'NIRISS_p_F090W':0.4523,'NIRISS_p_F115W':0.2785,'NIRISS_p_F140M':0.1849,
             'NIRISS_p_F150W':0.1618,'NIRISS_p_F200W':0.0919 
            } #absorption coefficients

        def abs_curve(col,p1=1,p2=2,p3=6.5,p4=40):        
            found=False
            n_s=len(list(MADYS.filt.keys()))
            surv=np.array(list(MADYS.filt.keys()))
            for i in range(n_s):
                try:
                    x=MADYS.filt[surv[i]][col]
                    found=True
                    break
                except KeyError:
                    continue
            if found==False: raise NameError('Filter '+col+' not found!')

            ec1=lambda x: 1.+0.7499*x-0.1086*x**2-0.08909*x**3+0.02905*x**4+0.01069*x**5+0.001707*x**6-0.001002*x**7
            ec2=lambda x: (0.3722)*x**(-2.07)

            B=0.366
            alpha=1.48
            S1=0.06893
            S2=0.02684
            l01=9.865
            g01=2.507
            a01=-0.232
            l02=19.973
            g02=16.989
            a02=-0.273

            g1=lambda x: 2*g01/(1+np.exp(a01*(x-l01)))
            g2=lambda x: 2*g02/(1+np.exp(a02*(x-l02)))
            d1=lambda x: (g1(x)/l01)**2/((x/l01-l01/x)**2+(g1(x)/l01)**2)
            d2=lambda x: (g2(x)/l02)**2/((x/l02-l02/x)**2+(g2(x)/l02)**2)        
            k=lambda x: B*x**(-alpha)+S1*d1(x)+S2*d2(x)
            q=lambda x: (x-p2)/(p3-p2)        
            step=lambda sta, sto, x: np.heaviside(x-sta,0)-np.heaviside(x-sto,0)
            wc=lambda x: step(0,p1,x)*ec1(1/x-1.82)+step(p1,p2,x)*ec2(x)+step(p2,p3,x)*(q(x)*k(x)+(1-q(x))*ec2(x))+step(p3,p4,x)*k(x)

            return wc(x)

        if '-' in col:
            c1,c2=col.split('-')
            try: A1=A_law[c1]
            except KeyError: A1=abs_curve(c1)
            try: A2=A_law[c2]
            except KeyError: A2=abs_curve(c2)            
            A=A1-A2
        else:
            try: A=A_law[col]
            except KeyError: A=abs_curve(col)

        return 3.16*A*ebv

    @staticmethod
    def is_phot_good(phot,phot_err,max_phot_err=0.1):
        if type(phot)==float: dim=0
        else:
            l=phot.shape
            dim=len(l)
        if dim<=1: 
            with np.errstate(invalid='ignore'):
                gs=(np.isnan(phot)==False) & (phot_err < max_phot_err) & (abs(phot) < 70)
        else:
            gs=np.zeros([l[0],l[1]])
            with np.errstate(invalid='ignore'):
                for i in range(l[1]): gs[:,i]=(np.isnan(phot[:,i])==False) & (phot_err[:,i] < max_phot_err)
        return gs

    @staticmethod
    def model_name(model,feh=None,afe=None,v_vcrit=None,fspot=None,B=0,he=None):
#        param={'model':model,'feh':0.0,'afe':0.0,'v_vcrit':0.0,'fspot':0.0,'B':0}
        if model=='bt_settl': model2=model
        elif model=='mist':
            feh_range=np.array([-4.,-3.5,-3.,-2.5,-2,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5])
            afe_range=np.array([0.0])
            vcrit_range=np.array([0.0,0.4])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
#                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(afe)!=type(None):
                i=np.argmin(abs(afe_range-afe))
                afe0=afe_range[i]
#                param['afe']=afe0
                if afe0<0: s='m'
                else: s='p'
                afe1="{:.1f}".format(abs(afe0))            
                model2+='_'+s+afe1
            else: model2+='_p0.0'
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
#                param['v_vcrit']=v_vcrit0
                if v_vcrit0<0: s='m'
                else: s='p'
                v_vcrit1="{:.1f}".format(abs(v_vcrit0))            
                model2+='_'+s+v_vcrit1
            else: model2+='_p0.0'
        elif model=='parsec':
            feh_range=np.array([-1.0,-0.75,-0.5,-0.25,0.0,0.13,0.25,0.5,0.75,1.00])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
#                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
        elif model=='starevol':
            feh_range=np.array([-0.826,-0.349,-0.224,-0.127,-0.013,0.152,0.288])
            vcrit_range=np.array([0.0,0.2,0.4,0.6])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
#                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_m0.01'
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
#                param['v_vcrit']=v_vcrit0
                if v_vcrit0<0: s='m'
                else: s='p'
                v_vcrit1="{:.1f}".format(abs(v_vcrit0))            
                model2+='_'+s+v_vcrit1
            else: model2+='_p0.0'
        elif model=='spots':
            fspot_range=np.array([0.00,0.17,0.34,0.51,0.68,0.85])
            if type(fspot)!=type(None):
                i=np.argmin(abs(fspot_range-fspot))
                fspot0=fspot_range[i]
#                param['fspot']=fspot0
                fspot1="{:.2f}".format(abs(fspot0))            
                model2=model+'_p'+fspot1
            else: model2=model+'_p0.00'
        elif model=='dartmouth':
            feh_range=np.array([0.0])
            afe_range=np.array([0.0])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
#                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(afe)!=type(None):
                i=np.argmin(abs(afe_range-afe))
                afe0=afe_range[i]
#                param['afe']=afe0
                if afe0<0: s='m'
                else: s='p'
                afe1="{:.1f}".format(abs(afe0))            
                model2+='_'+s+afe1
            else: model2+='_p0.0'
            if B==0: 
                model2+='_nomag'
            else: 
                model2+='_mag'
#                param['B']=1  
        elif model=='yapsi':
            feh_range=np.array([-1.44,-0.94,-0.44,0.06,0.36])
            he_range=np.array([0.25,0.28,0.31,0.34,0.37])
            mod_vals=np.array([['X0p749455_Z0p000545','X0p719477_Z0p000523','X0p689499_Z0p000501','X0p659520_Z0p000480','X0p629542_Z0p000458'],
                             ['X0p748279_Z0p001721','X0p718348_Z0p001652','X0p688417_Z0p001583','X0p658485_Z0p001515','X0p628554_Z0p001446'],
                             ['X0p744584_Z0p005416','X0p714801_Z0p005199','X0p685018_Z0p004982','X0p655234_Z0p004766','X0p625451_Z0p004549'],
                             ['X0p733138_Z0p016862','X0p703812_Z0p016188','X0p674487_Z0p015513','X0p645161_Z0p014839','X0p615836_Z0p014164'],
                             ['X0p717092_Z0p032908','X0p688408_Z0p031592','X0p659725_Z0p030275','X0p631041_Z0p028959','X0p602357_Z0p027643']])    
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
            else: i=3
            if type(he)!=type(None):
                j=np.argmin(abs(he_range-he))
            else: j=1
            model2='yapsi_'+mod_vals[i,j]            
        elif 'sb12' in model:
            feh_range=np.array([0.00,0.48])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
                feh1="{:.2f}".format(abs(feh0))
                model2=model+'_p'+feh1
            else: model2=model+'_p0.00'
        else: model2=model
#        return model2,param
        return model2
    
    @staticmethod
    def get_isochrone_filter(model,filt): #MODIFICARE NOMI, RENDERLI COERENTI CON IL DIZIONARIO E AGGIUNGERE FILTRI MANCANTI
        if model=='bt_settl':
            dic={'G':'G2018','Gbp':'G2018_BP','Grp':'G2018_RP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10','U':'U','B':'B',
                 'V':'V','R':'R','I':'i','g':'g_p1','r':'r_p1','i':'i_p1',
                 'z':'z_p1','y':'y_p1','SPH_Y':'B_Y','SPH_J':'B_J','SPH_H':'B_H',
                 'SPH_K':'B_Ks','SPH_H2':'D_H2','SPH_H3':'D_H3','SPH_H4':'D_H4',
                 'SPH_J2':'D_J2','SPH_J3':'D_J3','SPH_K1':'D_K1','SPH_K2':'D_K2',
                 'SPH_Y2':'D_Y2','SPH_Y3':'D_Y3',
                 'Teff':'Teff','logL':'logL','logg':'logg','radius':'radius'}
        elif model=='ames_cond':
            dic={'G':'G','Gbp':'G_BP','Grp':'G_BP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10',
                 'g':'g_p1','r':'r_p1','i':'i_p1','z':'z_p1','y':'y_p1',
                 'SPH_Y':'B_Y','SPH_J':'B_J','SPH_H':'B_H','SPH_K':'B_Ks','SPH_H2':'D_H2','SPH_H3':'D_H3',
                 'SPH_H4':'D_H4','SPH_J2':'D_J2','SPH_J3':'D_J3','SPH_K1':'D_K1','SPH_K2':'D_K2',
                 'SPH_Y2':'D_Y2','SPH_Y3':'D_Y3','H_F090M':'F090M','H_F110W':'F110W',
                 'H_F160W':'F160W','H_F165M':'F165M','H_F187W':'F187W','H_F205W':'F205W','H_F207M':'F207M',
                 'H_F222M':'F222M','H_F237M':'F237M','H_F253M':'F253M','H_F300W':'F300W','H_F336W':'F336W',
                 'H_F346M':'F346M','H_F439W':'F439W','H_F555W':'F555W','H_F606W':'F606W','H_F675W':'F675W',
                 'H_F785LP':'F785LP','H_F814W':'F814W',
                 'Teff':'Teff','logL':'logL','logg':'logg','radius':'radius'}
        elif model=='ames_dusty':
            dic={'G':'G','Gbp':'G_BP','Grp':'G_BP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10',
                 'g':'g_p1','r':'r_p1','i':'i_p1','z':'z_p1','y':'y_p1',
                 'SPH_Y':'B_Y','SPH_J':'B_J','SPH_H':'B_H','SPH_K':'B_Ks','SPH_H2':'D_H2','SPH_H3':'D_H3',
                 'SPH_H4':'D_H4','SPH_J2':'D_J2','SPH_J3':'D_J3','SPH_K1':'D_K1','SPH_K2':'D_K2',
                 'SPH_Y2':'D_Y2','SPH_Y3':'D_Y3','H_F090M':'F090M','H_F110W':'F110W',
                 'H_F160W':'F160W','H_F165M':'F165M','H_F187W':'F187W','H_F205W':'F205W','H_F207M':'F207M',
                 'H_F222M':'F222M','H_F237M':'F237M','H_F253M':'F253M','H_F300W':'F300W','H_F336W':'F336W',
                 'H_F346M':'F346M','H_F439W':'F439W','H_F555W':'F555W','H_F606W':'F606W','H_F675W':'F675W',
                 'H_F785LP':'F785LP','H_F814W':'F814W',
                 'Teff':'Teff','logL':'logL','logg':'logg','radius':'radius'}
        elif model=='nextgen':
            dic={'G':'G2018','Gbp':'G2018_BP','Grp':'G2018_RP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10',
                 'g':'g_p1','r':'r_p1','i':'i_p1','z':'z_p1','y':'y_p1',
                 'SPH_Y':'B_Y','SPH_J':'B_J','SPH_H':'B_H','SPH_K':'B_Ks','SPH_H2':'D_H2','SPH_H3':'D_H3',
                 'SPH_H4':'D_H4','SPH_J2':'D_J2','SPH_J3':'D_J3','SPH_K1':'D_K1','SPH_K2':'D_K2',
                 'SPH_Y2':'D_Y2','SPH_Y3':'D_Y3',
                 'Teff':'Teff','logL':'logL','logg':'logg','radius':'radius'}
        elif model=='mist':
            dic={'G':'Gaia_G_EDR3','Gbp':'Gaia_BP_EDR3','Grp':'Gaia_RP_EDR3',
                 'J':'2MASS_J','H':'2MASS_H','K':'2MASS_Ks',
                 'W1':'WISE_W1','W2':'WISE_W2','W3':'WISE_W3','W4':'WISE_W4',
                 'U':'Bessell_U','B':'Bessell_B','V':'Bessell_V','R':'Bessell_R','I':'Bessell_I',
                 'Kp':'Kepler_Kp','D51':'Kepler_D51','Hp':'Hipparcos_Hp',
                 'T_B':'Tycho_B','T_V':'Tycho_V','I_c':'TESS',
                 'Teff':'Teff','logL':'log_L','logg':'log_g','radius':'radius'}
        elif model=='parsec':
            dic={'G':'Gmag','Gbp':'G_BPmag','Grp':'G_RPmag',                 
                 'J':'Jmag','H':'Hmag','K':'Ksmag','IRAC1':'IRAC_3.6mag',
                 'IRAC2':'IRAC_4.5mag','IRAC3':'IRAC_5.8mag','IRAC4':'IRAC_8.0mag',                 
                 'MIPS24':'MIPS_24mag', 'MIPS70':'MIPS_70mag', 'MIPS160':'MIPS_160mag',
                 'W1':'W1mag','W2':'W2mag','W3':'W3mag','W4':'W4mag',
                 'Ux':'UXmag','Bx':'BXmag','B':'Bmag','V':'Vmag','R':'Rmag','I':'Imag',
                 'B_J':'Jmag','B_H':'Hmag','B_K':'Kmag','L':'Lmag','Lp':"L'mag",'M':'Mmag',
                 'SM_u':'umag','SM_v':'vmag','SM_g':'gmag','SM_r':'rmag','SM_i':'imag','SM_z':'zmag',
                 'Teff':'Teff','logL':'logL','logg':'logg','radius':'radius'}            
        elif model=='spots':
            dic={'G':'G_mag','Gbp':'BP_mag','Grp':'RP_mag',                 
                 'J':'J_mag','H':'H_mag','K':'K_mag',
                 'B':'B_mag','V':'V_mag','R':'Rc_mag','I':'Ic_mag',
                 'W1':'W1_mag',
                 'Teff':'Teff','logL':'log(L/Lsun)','logg':'log(g)','radius':'radius'}
        elif model=='dartmouth':
            dic={'B': 'jc_B','V': 'jc_V','R': 'jc_R','I': 'jc_I',
                 'G':'gaia_G','Gbp':'gaia_BP','Grp':'gaia_RP',
                 'J':'2mass_J','H':'2mass_H','K':'2mass_K',
                 'Teff':'Teff','logL':'log(L)','logg':'log(g)','radius':'radius'}     
        elif model=='starevol':
            dic={'U':'M_U','B':'M_B','V':'M_V','R':'M_R','I':'M_I',
                 'J':'M_J','H':'M_H','K':'M_K','G':'M_G','Gbp':'M_Gbp','Grp':'M_Grp',
                 'Teff':'Teff','logL':'logL','logg':'logg','radius':'R'}
        elif model=='bhac15':
            dic={'G':'G','Gbp':'G_BP','Grp':'G_RP','J':'Mj','H':'Mh','K':'Mk',
                 'g':'g_p1','r':'r_p1','i':'i_p1','z':'z_p1','y':'y_p1',
                 'V':'Mv','R':'Mr','I':'Mi','B_J':'Mj','B_H':'Mh','B_K':'Mk','L':'Ml','Lp':'Mll','M':'Mm',
                 'SPH_Y':'B_Y','SPH_J':'B_J','SPH_H':'B_H','SPH_K':'B_Ks','SPH_H2':'D_H2','SPH_H3':'D_H3',
                 'SPH_H4':'D_H4','SPH_J2':'D_J2','SPH_J3':'D_J3','SPH_K1':'D_K1','SPH_K2':'D_K2',
                 'SPH_Y2':'D_Y2','SPH_Y3':'D_Y3','IRAC1':'IRAC1','IRAC2':'IRAC2',
                 'IRAC3':'IRAC3','IRAC4':'IRAC4','IRSblue':'IRSblue','IRSred':'IRSred',                 
                 'MIPS24':'MIPS24', 'MIPS70':'MIPS70', 'MIPS160':'MIPS160',
                 'UKIDSS_y':'My', 'UKIDSS_z':'Mz', 'UKIDSS_j':'Mj', 'UKIDSS_h':'Mh', 'UKIDSS_k':'Mk',
                 'CFHT_Z':'Z','CFHT_Y':'Y','CFHT_J':'J','CFHT_H':'H','CFHT_K':'Ks','CFHT_CH4ON':'CH4_ON',
                 'CFHT_CH4OFF':'CH4_OFF','NIRCAM_p_F070Wa':'F070Wa','NIRCAM_p_F090Wa':'F090Wa',
                 'NIRCAM_p_F115Wa':'F115Wa','NIRCAM_p_F140Ma':'F140Ma','NIRCAM_p_F150Wa':'F150Wa',
                 'NIRCAM_p_F150W2a':'F150W2a','NIRCAM_p_F162a':'F162Ma','NIRCAM_p_F164Na':'F164Na',
                 'NIRCAM_p_F182Ma':'F182Ma','NIRCAM_p_F187Na':'F187Na','NIRCAM_p_F200Wa':'F200Wa',
                 'NIRCAM_p_F210Ma':'F210Ma','NIRCAM_p_F212Na':'F212Na','NIRCAM_p_F250Ma':'F250Ma',
                 'NIRCAM_p_F277Wa':'F277Wa','NIRCAM_p_F300Ma':'F300Ma','NIRCAM_p_F322W2a':'F322W2a',
                 'NIRCAM_p_F323Na':'F323Na','NIRCAM_p_F335Ma':'F335Ma','NIRCAM_p_F356Wa':'F356Wa',
                 'NIRCAM_p_F360Ma':'F360Ma','NIRCAM_p_F405Na':'F405Na','NIRCAM_p_F410Ma':'F410Ma',
                 'NIRCAM_p_F430Ma':'F430Ma','NIRCAM_p_F444Wa':'F444Wa','NIRCAM_p_F460Ma':'F460Ma',
                 'NIRCAM_p_F466Na':'F466Na','NIRCAM_p_F470Na':'F470Na','NIRCAM_p_F480Ma':'F480Ma',
                 'NIRCAM_p_F070Wb':'F070Wb','NIRCAM_p_F090Wb':'F090Wb','NIRCAM_p_F115Wb':'F115Wb',
                 'NIRCAM_p_F140Mb':'F140Mb','NIRCAM_p_F150Wb':'F150Wb','NIRCAM_p_F150W2b':'F150W2b',
                 'NIRCAM_p_F162Mb':'F162Mb','NIRCAM_p_F164Nb':'F164Nb','NIRCAM_p_F182Mb':'F182Mb',
                 'NIRCAM_p_F187Nb':'F187Nb','NIRCAM_p_F200Wb':'F200Wb','NIRCAM_p_F210Mb':'F210Mb',
                 'NIRCAM_p_F212Nb':'F212Nb','NIRCAM_p_F250Mb':'F250Mb','NIRCAM_p_F277Wb':'F277Wb',
                 'NIRCAM_p_F300Mb':'F300Mb','NIRCAM_p_F322W2b':'F322W2b','NIRCAM_p_F323Nb':'F323Nb',
                 'NIRCAM_p_F335Mb':'F335Mb','NIRCAM_p_F356Wb':'F356Wb','NIRCAM_p_F360Mb':'F360Mb',
                 'NIRCAM_p_F405Nb':'F405Nb','NIRCAM_p_F410Mb':'F410Mb','NIRCAM_p_F430Mb':'F430Mb',
                 'NIRCAM_p_F444Wb':'F444Wb','NIRCAM_p_F460Mb':'F460Mb','NIRCAM_p_F466Nb':'F466Nb',
                 'NIRCAM_p_F470Nb':'F470Nb','NIRCAM_p_F480Mb':'F480Mb','NIRCAM_p_F070Wab':'F070Wab',
                 'NIRCAM_p_F090Wab':'F090Wab','NIRCAM_p_F115Wab':'F115Wab','NIRCAM_p_F140Mab':'F140Mab',
                 'NIRCAM_p_F150Wab':'F150Wab','NIRCAM_p_F150W2ab':'F150W2ab','NIRCAM_p_F162Mab':'F162Mab',
                 'NIRCAM_p_F164Nab':'F164Nab','NIRCAM_p_F182Mab':'F182Mab','NIRCAM_p_F187Nab':'F187Nab',
                 'NIRCAM_p_F200Wab':'F200Wab','NIRCAM_p_F210Mab':'F210Mab','NIRCAM_p_F212Nab':'F212Nab',
                 'NIRCAM_p_F250Mab':'F250Mab','NIRCAM_p_F277Wab':'F277Wab','NIRCAM_p_F300Mab':'F300Mab',
                 'NIRCAM_p_F322W2ab':'F322W2ab','NIRCAM_p_F323Nab':'F323Nab','NIRCAM_p_F335Mab':'F335Mab',
                 'NIRCAM_p_F356Wab':'F356Wab','NIRCAM_p_F360Mab':'F360Mab','NIRCAM_p_F405Nab':'F405Nab',
                 'NIRCAM_p_F410Mab':'F410Mab','NIRCAM_p_F430Mab':'F430Mab','NIRCAM_p_F444Wab':'F444Wab',
                 'NIRCAM_p_F460Mab':'F460Mab','NIRCAM_p_F466Nab':'F466Nab','NIRCAM_p_F470Nab':'F470Nab',
                 'NIRCAM_p_F480Mab':'F480Mab','MIRI_p_F560W':'F560W','MIRI_p_F770W':'F770W',
                 'MIRI_p_F1000W':'F1000W','MIRI_p_F1130W':'F1130W','MIRI_p_F1280W':'F1280W',
                 'MIRI_p_F1500W':'F1500W','MIRI_p_F1800W':'F1800W','MIRI_p_F2100W':'F2100W','MIRI_p_F2550W':'F2550W',
                 'Teff':'Teff','logL':'L/Ls','logg':'g','radius':'radius'}
        elif 'atmo2020' in model:
            dic={'MKO_Y':'MKO_Y','MKO_J':'MKO_J','MKO_H':'MKO_H','MKO_K':'MKO_K','MKO_Lp':'MKO_Lp','MKO_Mp':'MKO_Mp',
                 'W1':'W1','W2':'W2','W3':'W3','W4':'W4',
                 'IRAC1':'IRAC_CH1','IRAC2':'IRAC_CH2',
                 'NIRCAM_p_F070Wa':'NIRCAM-F070W','NIRCAM_p_F090Wa':'NIRCAM-F090W','NIRCAM_p_F115Wa':'NIRCAM-F115W',
                 'NIRCAM_p_F140Ma':'NIRCAM-F140M','NIRCAM_p_F150Wa':'NIRCAM-F150W','NIRCAM_p_F150W2a':'NIRCAM-F150W2',
                 'NIRCAM_p_F162Ma':'NIRCAM-F162M','NIRCAM_p_F164Na':'NIRCAM-F164N','NIRCAM_p_F182Ma':'NIRCAM-F182M',
                 'NIRCAM_p_F187Na':'NIRCAM-F187N','NIRCAM_p_F200Wa':'NIRCAM-F200W','NIRCAM_p_F210Ma':'NIRCAM-F210M',
                 'NIRCAM_p_F212Na':'NIRCAM-F212N','NIRCAM_p_F250Ma':'NIRCAM-F250M','NIRCAM_p_F277Wa':'NIRCAM-F277W',
                 'NIRCAM_p_F300Ma':'NIRCAM-F300M','NIRCAM_p_F322W2a':'NIRCAM-F322W2','NIRCAM_p_F323Na':'NIRCAM-F323N',
                 'NIRCAM_p_F335Ma':'NIRCAM-F335M','NIRCAM_p_F356Wa':'NIRCAM-F356W','NIRCAM_p_F360Ma':'NIRCAM-F360M',
                 'NIRCAM_p_F405Na':'NIRCAM-F405N','NIRCAM_p_F410Ma':'NIRCAM-F410M','NIRCAM_p_F430Ma':'NIRCAM-F430M',
                 'NIRCAM_p_F444Wa':'NIRCAM-F444W','NIRCAM_p_F460Ma':'NIRCAM-F460M','NIRCAM_p_F466Na':'NIRCAM-F466N',
                 'NIRCAM_p_F470Na':'NIRCAM-F470N','NIRCAM_p_F480Ma':'NIRCAM-F480M','NIRCAM_p_F070Wab':'NIRCAM-F070W',
                 'NIRCAM_p_F090Wab':'NIRCAM-F090W','NIRCAM_p_F115Wab':'NIRCAM-F115W','NIRCAM_p_F140Mab':'NIRCAM-F140M',
                 'NIRCAM_p_F150Wab':'NIRCAM-F150W','NIRCAM_p_F150W2ab':'NIRCAM-F150W2','NIRCAM_p_F162Mab':'NIRCAM-F162M',
                 'NIRCAM_p_F164Nab':'NIRCAM-F164N','NIRCAM_p_F182Mab':'NIRCAM-F182M','NIRCAM_p_F187Nab':'NIRCAM-F187N',
                 'NIRCAM_p_F200Wab':'NIRCAM-F200W','NIRCAM_p_F210Mab':'NIRCAM-F210M','NIRCAM_p_F212Nab':'NIRCAM-F212N',
                 'NIRCAM_p_F250Mab':'NIRCAM-F250M','NIRCAM_p_F277Wab':'NIRCAM-F277W','NIRCAM_p_F300Mab':'NIRCAM-F300M',
                 'NIRCAM_p_F322W2ab':'NIRCAM-F322W2','NIRCAM_p_F323Nab':'NIRCAM-F323N','NIRCAM_p_F335Mab':'NIRCAM-F335M',
                 'NIRCAM_p_F356Wab':'NIRCAM-F356W','NIRCAM_p_F360Mab':'NIRCAM-F360M','NIRCAM_p_F405Nab':'NIRCAM-F405N',
                 'NIRCAM_p_F410Mab':'NIRCAM-F410M','NIRCAM_p_F430Mab':'NIRCAM-F430M','NIRCAM_p_F444Wab':'NIRCAM-F444W',
                 'NIRCAM_p_F460Mab':'NIRCAM-F460M','NIRCAM_p_F466Nab':'NIRCAM-F466N','NIRCAM_p_F470Nab':'NIRCAM-F470N',
                 'NIRCAM_p_F480Mab':'NIRCAM-F480M','NIRCAM_p_F070Wb':'NIRCAM-F070W','NIRCAM_p_F090Wb':'NIRCAM-F090W',
                 'NIRCAM_p_F115Wb':'NIRCAM-F115W','NIRCAM_p_F140Mb':'NIRCAM-F140M','NIRCAM_p_F150Wb':'NIRCAM-F150W',
                 'NIRCAM_p_F150W2b':'NIRCAM-F150W2','NIRCAM_p_F162Mb':'NIRCAM-F162M','NIRCAM_p_F164Nb':'NIRCAM-F164N',
                 'NIRCAM_p_F182Mb':'NIRCAM-F182M','NIRCAM_p_F187Nb':'NIRCAM-F187N','NIRCAM_p_F200Wb':'NIRCAM-F200W',
                 'NIRCAM_p_F210Mb':'NIRCAM-F210M','NIRCAM_p_F212Nb':'NIRCAM-F212N','NIRCAM_p_F250Mb':'NIRCAM-F250M',
                 'NIRCAM_p_F277Wb':'NIRCAM-F277W','NIRCAM_p_F300Mb':'NIRCAM-F300M','NIRCAM_p_F322W2b':'NIRCAM-F322W2',
                 'NIRCAM_p_F323Nb':'NIRCAM-F323N','NIRCAM_p_F335Mb':'NIRCAM-F335M','NIRCAM_p_F356Wb':'NIRCAM-F356W',
                 'NIRCAM_p_F360Mb':'NIRCAM-F360M','NIRCAM_p_F405Nb':'NIRCAM-F405N','NIRCAM_p_F410Mb':'NIRCAM-F410M',
                 'NIRCAM_p_F430Mb':'NIRCAM-F430M','NIRCAM_p_F444Wb':'NIRCAM-F444W','NIRCAM_p_F460Mb':'NIRCAM-F460M',
                 'NIRCAM_p_F466Nb':'NIRCAM-F466N','NIRCAM_p_F470Nb':'NIRCAM-F470N','NIRCAM_p_F480Mb':'NIRCAM-F480M',
                 'MIRI_p_F560W':'MIRI-F560W','MIRI_p_F770W':'MIRI-F770W','MIRI_p_F1000W':'MIRI-F1000W','MIRI_p_F1130W':'MIRI-F1130W',
                 'MIRI_p_F1280W':'MIRI-F1280W','MIRI_p_F1500W':'MIRI-F1500W','MIRI_p_F1800W':'MIRI-F1800W',
                 'MIRI_p_F2100W':'MIRI-F2100W','MIRI_p_F2550W':'MIRI-F2550W',
                 'NIRISS_p_F090W':'NIRISS-F090W','NIRISS_p_F115W':'NIRISS-F115W','NIRISS_p_F140M':'NIRISS-F140M',
                 'NIRISS_p_F150W':'NIRISS-F150W','NIRISS_p_F158M':'NIRISS-F158M','NIRISS_p_F200W':'NIRISS-F200W',
                 'NIRISS_p_F277W':'NIRISS-F277W','NIRISS_p_F356W':'NIRISS-F356W','NIRISS_p_F380M':'NIRISS-F380M',
                 'NIRISS_p_F430M':'NIRISS-F430M','NIRISS_p_F444W':'NIRISS-F444W','NIRISS_p_F480M':'NIRISS-F480M',
                 'MIRI_c_F1065C':'MIRI-F1065C','MIRI_c_F1140C':'MIRI-F1140C','MIRI_c_F1550C':'MIRI-F1550C',
                 'MIRI_c_F2300C':'MIRI-F2300C','NIRCAM_c210_F182M':'NIRCAM-F182M','NIRCAM_c210_F187N':'NIRCAM-F187N',
                 'NIRCAM_c210_F200W':'NIRCAM-F200W','NIRCAM_c210_F210M':'NIRCAM-F210M','NIRCAM_c210_F212N':'NIRCAM-F212N',
                 'NIRCAM_c335_F250M':'NIRCAM-F250M','NIRCAM_c335_F300M':'NIRCAM-F300M',
                 'NIRCAM_c335_F322W2':'NIRCAM-F322W2','NIRCAM_c335_F335M':'NIRCAM-F335M',
                 'NIRCAM_c335_F356W':'NIRCAM-F356W','NIRCAM_c335_F360M':'NIRCAM-F360M',
                 'NIRCAM_c335_F410M':'NIRCAM-F410M','NIRCAM_c335_F430M':'NIRCAM-F430M',
                 'NIRCAM_c335_F444W':'NIRCAM-F444W','NIRCAM_c335_F460M':'NIRCAM-F460M',
                 'NIRCAM_c335_F480M':'NIRCAM-F480M','NIRCAM_c430_F250M':'NIRCAM-F250M',
                 'NIRCAM_c430_F300M':'NIRCAM-F300M','NIRCAM_c430_F322W2':'NIRCAM-F322W2',
                 'NIRCAM_c430_F335M':'NIRCAM-F335M','NIRCAM_c430_F356W':'NIRCAM-F356W',
                 'NIRCAM_c430_F360M':'NIRCAM-F360M','NIRCAM_c430_F410M':'NIRCAM-F410M',
                 'NIRCAM_c430_F430M':'NIRCAM-F430M','NIRCAM_c430_F444W':'NIRCAM-F444W',
                 'NIRCAM_c430_F460M':'NIRCAM-F460M','NIRCAM_c430_F480M':'NIRCAM-F480M',
                 'NIRCAM_cLWB_F250M':'NIRCAM-F250M','NIRCAM_cLWB_F277W':'NIRCAM-F277W',
                 'NIRCAM_cLWB_F300M':'NIRCAM-F300M','NIRCAM_cLWB_F335M':'NIRCAM-F335M',
                 'NIRCAM_cLWB_F356W':'NIRCAM-F356W','NIRCAM_cLWB_F360M':'NIRCAM-F360M',
                 'NIRCAM_cLWB_F410M':'NIRCAM-F410M','NIRCAM_cLWB_F430M':'NIRCAM-F430M',
                 'NIRCAM_cLWB_F444W':'NIRCAM-F444W','NIRCAM_cLWB_F460M':'NIRCAM-F460M',
                 'NIRCAM_cLWB_F480M':'NIRCAM-F480M','NIRCAM_cSWB_F182M':'NIRCAM-F182M',
                 'NIRCAM_cSWB_F187N':'NIRCAM-F187N','NIRCAM_cSWB_F200W':'NIRCAM-F200W',
                 'NIRCAM_cSWB_F210M':'NIRCAM-F210M','NIRCAM_cSWB_F212N':'NIRCAM-F212N',
                 'NIRISS_c_F277W':'NIRISS-F277W','NIRISS_c_F380M':'NIRISS-F380M',
                 'NIRISS_c_F430M':'NIRISS-F430M','NIRISS_c_F480M':'NIRISS-F480M',                 
                 'Teff':'Teff','logL':'Luminosity','logg':'Gravity','radius':'Radius'}
        elif model=='geneva':
            dic={'V':'V','U':'U','B':'B','R':'R','I':'I',
                 'J':'J', 'H':'H', 'K':'K','G':'G','Gbp':'Gbp','Grp':'Grp',
                 'Teff':'Teff','logL':'logL','logg':'g_pol','radius':'r_pol'}
        elif model=='sonora_bobcat':
            dic={'J':'J','H':'H','K':'Ks','W1':'W1',
                'W2':'W2','W3':'W3','W4':'W4',
                 'Teff':'Teff','logL':'logL','logg':'logg','radius':'radius'}
        elif model=='yapsi':
            dic={'Teff':'Teff','logL':'log(L/Lsun)','logg':'log_g','radius':'radius'}
        elif ('sb12' in model) & ('hot' in model):
            dic={'B_J':'hJmag','B_H':'hHmag','B_K':'hKmag','radius':'hRad', 'Lp':'hLmag', 'M':'hMmag', 'N':'hNmag'}
        elif ('sb12' in model) & ('cold' in model):
            dic={'B_J':'cJmag','B_H':'cHmag','B_K':'cKmag','radius':'cRad', 'Lp':'cLmag', 'M':'cMmag', 'N':'cNmag'}
        elif model=='b97':
            dic={'Teff':'T_eff','logL':'log_L','logg':'log_g','radius':'R'}
        elif model=='pm13':
            dic={'Teff':'Teff','logL':'logL','logg':'log_g','radius':'R_Rsun',
                'V':'Mv','B':'B','Gbp':'Bp','Grp':'Rp','G':'M_G','U':'U','R':'Rc','I':'Ic',
                'H':'H','J':'M_J','K':'M_Ks','W1':'W1','W2':'W2','W3':'W3','W4':'W4'}
        try:
            return dic[filt]
        except KeyError: return 'nan'
    
    @staticmethod
    def fix_filters(filters,model,mode='collapse'):
        mod2=['bt_settl','starevol','spots','dartmouth','ames_cond','ames_dusty','bt_nextgen','nextgen','bhac15','geneva']
        mod3=['mist','parsec']        
        if mode=='collapse':
            filters=np.where(filters=='G2','G', filters)
            filters=np.where(filters=='Gbp2','Gbp', filters)
            filters=np.where(filters=='Grp2','Grp', filters)
            filters=np.unique(filters)
        elif mode=='replace':
            if model in mod2: 
                filters=np.where(filters=='G','G2', filters)
                filters=np.where(filters=='Gbp','Gbp2', filters)
                filters=np.where(filters=='Grp','Grp2', filters)
        return filters
    
    @staticmethod
    def load_isochrones(model,filters,n_steps=[1000,500], **kwargs):

        folder = os.path.dirname(os.path.realpath(__file__))

        add_search_path(folder)
        for x in os.walk(folder):
            add_search_path(x[0])

        logger=kwargs['logger'] if 'logger' in kwargs else None

        mass_range=kwargs['mass_range'] if 'mass_range' in kwargs else [0.01,1.4]
        age_range=kwargs['age_range'] if 'age_range' in kwargs else [1,1000]
        B=kwargs['B'] if 'B' in kwargs else 0
        feh=kwargs['feh'] if 'feh' in kwargs else None
        afe=kwargs['afe'] if 'afe' in kwargs else None
        v_vcrit=kwargs['v_vcrit'] if 'v_vcrit' in kwargs else None
        fspot=kwargs['fspot'] if 'fspot' in kwargs else None
        he=kwargs['he'] if 'he' in kwargs else None

        model_code=MADYS.model_name(model,feh=feh,afe=afe,v_vcrit=v_vcrit,fspot=fspot,B=B,he=he)
    #    model_code,param=model_name(model,feh=feh,afe=afe,v_vcrit=v_vcrit,fspot=fspot,B=B)
    #    param['mass_range']=mass_range
    #    param['age_range']=age_range

        
        fnew=MADYS.fix_filters(filters,model,mode='collapse')
        nf=len(fnew)
        c=0

        surveys=MADYS.filters_to_surveys(fnew)
        
        n1=n_steps[0]
        mnew=M_sun.value/M_jup.value*np.exp(np.log(0.999*mass_range[0])+(np.log(1.001*mass_range[1])-np.log(0.999*mass_range[0]))/(n1-1)*np.arange(n1))

        try: len(age_range)
        except TypeError:
            anew=age_range
            n2=1
            case=1
        else:
            if isinstance(age_range,list):
                n2=n_steps[1]
                anew=np.exp(np.log(1.0001*age_range[0])+(np.log(0.9999*age_range[1])-np.log(1.0001*age_range[0]))/(n2-1)*np.arange(n2))
                case=2
            elif isinstance(age_range,np.ndarray):
                if len(age_range.shape)==1:
                    anew=np.array(age_range)
                    n2=len(anew)
                    case=3
                elif len(age_range[0])==3:
                    age0=np.unique(age_range.ravel())    
                    age1=(age0[:-1]+(age0[1:]-age0[:-1])/4)
                    age2=(age0[:-1]+(age0[1:]-age0[:-1])/2)
                    age3=(age0[:-1]+3*(age0[1:]-age0[:-1])/4)
                    anew=np.sort(np.concatenate((age0,age1,age2,age3)))
                    #anew=np.unique(age_range.ravel())
                    n2=len(anew)
                    case=4
                elif len(age_range[0])==2:
                    n2=n_steps[1]
                    anew=np.exp(np.log(1.0001*np.nanmin(age_range))+(np.log(0.9999*np.nanmax(age_range))-np.log(1.0001*np.nanmin(age_range)))/(n2-1)*np.arange(n2))
                    case=6                
            else: raise TypeError('Only scalar, list or numpy arrays are valid inputs for the keyword "age_range".')
        if model_code=='pm13': case=5
            
                
        iso_f=np.full(([n1,n2,nf]), np.nan) #final matrix    
        found=np.zeros(nf,dtype=bool)
#        c=0
        if case==5:
            for i in range(len(surveys)):
#                if c==nf: break
                if np.sum(found)==nf: break
                try:
                    masses, ages, v0, data0 = model_data(surveys[i],model_code)
                except ValueError: #if the survey is not found for the isochrone set, its filters are set to NaN
                    print(surveys[i],model_code)
                    if logger!=None:
                        logger.info('Survey '+surveys[i]+' not found for model '+model+'. Setting all related filters to nan.')
                    continue
                iso=np.full([n1,len(ages),len(fnew)],np.nan)
                for j in range(len(fnew)):
                    iso_filter=MADYS.get_isochrone_filter(model,fnew[j])
                    w,=np.where(v0==iso_filter) #leaves NaN if the filter is not found
                    if (len(w)>0) & (found[j]==False):
                        found[j]=True
                        k=0
                        gv = np.isfinite(data0[:,k,w]).ravel() #interpolates along mass
                        m0=masses[gv]
                        if len(m0)>1:
                            f=interp1d(m0,data0[gv,k,w],kind='linear',fill_value=np.nan,bounds_error=False)
                            iso_f[:,k,j]=f(mnew)
                        if np.sum(found)==nf: break
            anew=ages
        elif case>1:
            for i in range(len(surveys)):
#                if c==nf: break
                if np.sum(found)==nf: break
                try:
                    masses, ages, v0, data0 = model_data(surveys[i],model_code)
                except ValueError: #if the survey is not found for the isochrone set, its filters are set to NaN
                    print(surveys[i],model_code)
                    if logger!=None:
                        logger.info('Survey '+surveys[i]+' not found for model '+model+'. Setting all related filters to nan.')
                    continue
                iso=np.full([n1,len(ages),len(fnew)],np.nan)
                for j in range(len(fnew)):
                    iso_filter=MADYS.get_isochrone_filter(model,fnew[j])
                    w,=np.where(v0==iso_filter) #leaves NaN if the filter is not found
                    if (len(w)>0) & (found[j]==False):
                        found[j]=True
                        for k in range(len(ages)): #interpolates along mass
                            gv = np.isfinite(data0[:,k,w]).ravel()
                            m0=masses[gv]
                            if len(m0)>1:
                                f=interp1d(m0,data0[gv,k,w],kind='linear',fill_value=np.nan,bounds_error=False)
                                iso[:,k,j]=f(mnew)
                        for k in range(n1):  #interpolates along age
                            gv, igv = MADYS.split_if_nan((iso[k,:,j]).ravel())
                            for l in range(len(gv)):
                                a0=ages[igv[l]]
                                an,=np.where((anew>0.95*a0[0]) & (anew<1.05*a0[-1]))
                                if len(an)==0: continue
                                if len(a0)>2:
                                    f=interp1d(a0,iso[k,igv[l],j],kind='linear',fill_value='extrapolate',bounds_error=False)
                                    iso_f[k,an,j]=f(anew[an])
#                                    iso_f[k,an,c]=f(anew[an])
                                elif len(a0)==2:
                                    f=lambda x: iso[k,igv[l].start,j]+(iso[k,igv[l].stop,j]-iso[k,igv[l].start,j])/(a0[1]-a0[0])*x
                                    iso_f[k,an,j]=f(anew[an])
#                                    iso_f[k,an,c]=f(anew[an])
                                elif len(a0)==1: iso_f[k,an,j]=iso[k,igv[l],j]
#                                elif len(a0)==1: iso_f[k,an,c]=iso[k,igv[l],j]
#                        c+=1
#                        if c==nf: break
                        if np.sum(found)==nf: break
        else:
            for i in range(len(surveys)):
                if c==nf: break
                try:
                    masses, ages, v0, data0 = model_data(surveys[i],model_code)
                except ValueError: #if the survey is not found for the isochrone set, its filters are set to NaN
                    if logger!=None:
                        logger.info('Survey '+surveys[i]+' not found for model '+model+'. Setting all related filters to nan.')
                    continue
                iso=np.full([n1,len(ages),len(fnew)],np.nan)
                for j in range(len(fnew)):
                    iso_filter=MADYS.get_isochrone_filter(model,fnew[j])
                    w,=np.where(v0==iso_filter) #leaves NaN if the filter is not found                
                    if (len(w)>0) & (found[j]==False):
                        found[j]=True
                        for k in range(len(ages)): #interpolates along mass
                            gv = np.isfinite(data0[:,k,w]).ravel()
                            m0=masses[gv]
                            if len(m0)>1:
                                f=interp1d(m0,data0[gv,k,w],kind='linear',fill_value=np.nan,bounds_error=False)
                                iso[:,k,j]=f(mnew)
                        for k in range(n1):  #interpolates along age
                            gv, igv = MADYS.split_if_nan((iso[k,:,j]).ravel())
                            for l in range(len(gv)):
                                a0=ages[igv[l]]
                                if (anew>0.95*a0[0]) & (anew<1.05*a0[-1]):
                                    if len(a0)>2:
                                        f=interp1d(a0,iso[k,igv[l],j],kind='linear',fill_value='extrapolate',bounds_error=False)
                                        iso_f[k,0,j]=f(anew)
#                                        iso_f[k,0,c]=f(anew)
                                    elif len(a0)==2:
                                        f=lambda x: iso[k,igv[l].start,j]+(iso[k,igv[l].stop,j]-iso[k,igv[l].start,j])/(a0[1]-a0[0])*x
                                        iso_f[k,0,j]=f(anew)
#                                        iso_f[k,0,c]=f(anew)
                                    elif len(a0)==1: iso_f[k,0,j]=iso[k,igv[l],j]
#                                    elif len(a0)==1: iso_f[k,0,c]=iso[k,igv[l],j]
#                        c+=1
#                        if c==nf: break
                        if np.sum(found)==nf: break

        mnew=M_jup.value/M_sun.value*mnew
        if n2==1: anew=np.array([anew])        
        fnew=np.array(fnew)
        if 'radius' in fnew:
            w=MADYS.where_v(['radius'],fnew)
            iso_f[:,:,w]*=R_jup.value/R_sun.value            
        fnew=MADYS.fix_filters(fnew,model,mode='replace')

        return mnew,anew,fnew,iso_f

    @staticmethod
    def ang_deg(ang,form='hms'):    
        ang2=ang.split(' ')
        ang2=ang2[0]+form[0]+ang2[1]+form[1]+ang2[2]+form[2]
        return ang2

    @staticmethod
    def merge_fam(x):
        if (len(x.shape)<2) | ((x.shape)[0]<2): return x
        w0,=np.where(x[:,4]<1e-3)
        if len(w0)>1:
            x=np.delete(x,w0,0)
            x[:,4]/=np.sum(x[:,4])
            if (x.shape)[0]<2: return x

        while True:
            for i in range(len(x)):
                found=0
                for j in range(i+1,len(x)):
                    err_i_m,err_j_m=np.max([x[i,1],0.009]),np.max([x[j,1],0.009])
                    err_i_a,err_j_a=np.max([x[i,3],0.009]),np.max([x[j,3],0.009])
                    dd=(x[i,0]-x[j,0])**2/(err_i_m**2+err_j_m**2)+(x[i,2]-x[j,2])**2/(err_i_a**2+err_j_a**2)
                    if dd<8:
                        x[i,1]=np.sqrt((x[i,4]*x[i,1])**2+(x[j,4]*x[j,1])**2)
                        if x[i,1]==0: x[i,1]=np.abs(x[i,0]-x[j,0])
                        x[i,0]=np.average([x[i,0],x[j,0]],weights=[x[i,4],x[j,4]])
                        x[i,3]=np.sqrt((x[i,4]*x[i,3])**2+(x[j,4]*x[j,3])**2)
                        if x[i,3]==0: x[i,3]=np.abs(x[i,2]-x[j,2])
                        x[i,2]=np.average([x[i,2],x[j,2]],weights=[x[i,4],x[j,4]])
                        x[i,4]+=x[j,4]
                        x=np.delete(x,j,0)
                        found=1
                        break
                if found: break
            if found==0: break
        w,=np.where(np.isnan(x[:,0])==False)
        return x[w,:]

    @staticmethod
    def intersect_arr(x,y):
        if np.max(x)<np.min(y): return np.array([])
        else: return [np.max([np.min(x),np.min(y)]),np.min([np.max(x),np.max(y)])]

    @staticmethod
    def get_mass_range(data,model,dtype='mag'):

        if model=='bt_settl': mass_range=[0.01,1.4]
        elif model=='mist': mass_range=[0.1,150]
        elif model=='parsec': mass_range=[0.09,350]
        elif model=='starevol': mass_range=[0.2,1.5]
        elif model=='spots': mass_range=[0.1,1.3]
        elif model=='dartmouth': mass_range=[0.105,1.05]
        elif model=='yapsi': mass_range=[0.15,5]
        elif 'sb12' in model: mass_range=[0.001, 0.01]
        elif 'atmo' in model: mass_range=[0.001,0.075]
        elif 'b97' in model: mass_range=[0.001,0.01]
        elif model=='bhac15': mass_range=[0.01,1.4]
        elif model=='geneva': mass_range=[0.8,120]
        elif model=='ames_cond': mass_range=[0.005,1.4]
        elif model=='ames_dusty': mass_range=[0.005,1.4]
        elif model=='nextgen': mass_range=[0.01,1.4]
        elif model=='sonora_bobcat': mass_range=[0.0005,0.1]

        if dtype=='mass':
            sample_r=[np.nanmin(data),np.nanmax(data)]
            return MADYS.intersect_arr(mass_range,[sample_r[0],sample_r[1]])
        else:
            m_G_data=np.array([[-5.8,84],[-5.2,52],[-4.8,27],[-3.5,15],[-2.6,10],[-1.19,5.4],[-0.99,5.1],
                           [-0.84,4.7],[-0.54,4.3],[-0.39,3.92],[-0.01,3.38],[0.515,2.75],[0.615,2.68],
                           [1.00,2.18],[1.16,2.05],[1.345,1.98],[1.69,1.86],[1.92,1.93],[1.98,1.88],
                           [2.09,1.83],[2.19,1.77],[2.27,1.81],[2.37,1.75],[2.51,1.61],[2.69,1.50],[2.89,1.46],
                           [2.99,1.44],[3.10,1.38],[3.26,1.33],[3.56,1.25],[3.66,1.21],[3.90,1.18],[4.105,1.13],
                           [4.195,1.08],[4.325,1.06],[4.462,1.03],[4.635,1.00],[4.703,0.99],[4.757,0.985],
                           [4.801,0.98],[4.914,0.97],[5.006,0.95],[5.098,0.94],[5.34,0.90],[5.553,0.88],
                           [5.65,0.86],[5.83,0.82],[6.20,0.78],[6.53,0.73],[6.83,0.70],[7.02,0.69],
                           [7.57,0.64],[7.74,0.62],[8.03,0.59],[8.16,0.57],[8.44,0.54],[8.82,0.50],
                           [8.98,0.47],[9.29,0.44],[9.67,0.40],[10.05,0.37],[10.87,0.27],[11.21,0.23],
                           [12.04,0.184],[12.45,0.162],[13.35,0.123],[14.26,0.102],[14.40,0.093],
                           [14.72,0.090],[15.20,0.088],[15.20,0.085],[15.90,0.080],[16.20,0.079],
                           [16.40,0.078],[16.60,0.077],[16.90,0.076],[17.30,0.075]])

            G_m=interp1d(m_G_data[:,0],m_G_data[:,1],fill_value=(84,0.075),bounds_error=False)
            sample_r=G_m([np.nanmax(data),np.nanmin(data)])   
            return MADYS.intersect_arr(mass_range,[0.6*sample_r[0],1.4*sample_r[1]])    
    
    #################################################################
    # TO CREATE LOG FILES

    @staticmethod
    def setup_custom_logger(name,file,mode='a'):
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(file, mode=mode)
        handler.setFormatter(formatter)
    #    screen_handler = logging.StreamHandler(stream=sys.stdout)
    #    screen_handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if (logger.hasHandlers()):
            logger.handlers.clear()        
        logger.addHandler(handler)
    #    logger.addHandler(screen_handler)    

        return logger    

    @staticmethod
    def info_models(model=None):

            folder = os.path.dirname(os.path.realpath(__file__))

            paths=[x[0] for x in os.walk(folder)]
            found = False
            for path in paths:
                if (Path(path) / 'isochrones').exists():
                    found = True
                    path=Path(path) / 'isochrones'
                    folders=[x[0] for x in os.walk(path)]
                    break
                        
            if type(model)==type(None):
                print('AVAILABLE MODELS FOR MADYS')
                print('')
            else:
                model=str.lower(model)
                def fix_mod(model):
                    dic={'atmo2020':'atmo','atmo2020_ceq':'atmo',
                         'atmo2020_neq_s':'atmo','atmo2020_neq_w':'atmo','bhac 15':'bhac15',
                         'ames dusty':'dusty','ames_dusty':'dusty',
                         'ames cond':'cond','ames_cond':'cond', 
                         'bt_settl':'settl','bt-settl':'settl','bt settl':'settl',
                         'sonora bobcat':'sonora', 'sonora_bobcat':'sonora'
                        }
                    try: 
                        return dic[model]
                    except KeyError:
                        return model
                model=fix_mod(model)
                found=False
                
            f_mods=[]
            for path in folders:
                if (Path(path) / 'info.txt').exists():
                    info=Path(path) / 'info.txt'
                    if type(model)!=type(None):
                        if model not in str.lower(path): 
                            if 'starevol' in str.lower(path): f_mods.append('starevol')
                            elif 'atmo' in str.lower(path): f_mods.append('atmo2020')
                            elif 'bhac15' in str.lower(path): f_mods.append('bhac15')
                            elif 'mist' in str.lower(path): f_mods.append('mist')
                            elif 'parsec' in str.lower(path): f_mods.append('parsec')
                            elif 'cond' in str.lower(path): f_mods.append('ames_cond')
                            elif 'dusty' in str.lower(path): f_mods.append('ames_dusty')
                            elif 'nextgen' in str.lower(path): f_mods.append('nextgen')
                            elif 'settl' in str.lower(path): f_mods.append('bt_settl')
                            elif 'spots' in str.lower(path): f_mods.append('spots')
                            elif 'dartmouth' in str.lower(path): f_mods.append('dartmouth')
                            elif 'geneva' in str.lower(path): f_mods.append('geneva')
                            elif 'sonora' in str.lower(path): f_mods.append('sonora_bobcat')
                            elif 'yapsi' in str.lower(path): f_mods.append('yapsi')
                            elif 'sb12' in str.lower(path): f_mods.append('sb12')
                            elif 'b97' in str.lower(path): f_mods.append('b97')
                            elif 'pm13' in str.lower(path): f_mods.append('pm13')
                            continue
                        else: found=True
                    with open(info) as f:
                        print(f.read())
                    files=[x[2] for x in os.walk(path)]
                    if 'starevol' in str.lower(path):
                        expr = re.compile('Isochr_Z(.+)_Vini(.+)_t(.+).dat')
                        z=[]
                        v_vcrit=[]            
                        for j in range(len(files[0])):
                            string=files[0][j]
                            m=expr.match(string)
                            if (m is not None):
                                z.append(float(m.group(1)))
                                v_vcrit.append(m.group(2))
                                #t=m.group(3)
                        feh=["%.2f" % np.log10(np.array(d)/0.0134) for d in z]
                        feh=np.unique(feh)
                        v_vcrit=np.unique(v_vcrit)
                        print('Available metallicities: [Fe/H]=[',','.join(feh),']')
                        print('Available velocities/v_crit:[',','.join(v_vcrit),']')
                        print("CALL IT AS: 'starevol'")
                    elif 'atmo' in str.lower(path): 
                        print("CALL IT AS: 'atmo2020_ceq'/'atmo2020_neq_s'/'atmo2020_neq_w'")                                        
                    elif 'bhac15' in str.lower(path): 
                        print("CALL IT AS: 'bhac15'")
                    elif 'mist' in str.lower(path):
                        expr = re.compile('MIST_v1.2_feh_(.+)_afe_(.+)_vvcrit(.+)_.+')
                        feh=[]
                        afe=[]
                        v_vcrit=[]                    
                        for j in range(len(files[0])):
                            string=files[0][j]
                            m=expr.match(string)
                            if (m is not None):
                                feh0=m.group(1).replace('m','-')
                                feh.append(feh0.replace('p',''))
                                afe0=m.group(2).replace('m','-')
                                afe.append(afe0.replace('p',''))
                                v_vcrit.append(m.group(3))
                        feh=np.unique(feh)
                        afe=np.unique(afe)
                        v_vcrit=np.unique(v_vcrit)
                        print('Available metallicities: [Fe/H]= [',','.join(feh),']')
                        print("Available alpha enh's: [Fe/H]= [",','.join(afe),']')
                        print('Available velocities/v_crit: [',','.join(v_vcrit),']')
                        print("CALL IT AS: 'mist'")
                    elif 'parsec' in str.lower(path):
                        expr = re.compile('GAIA_EDR3_feh_(.+).txt')
                        feh=[]
                        for j in range(len(files[0])):
                            string=files[0][j]
                            m=expr.match(string)
                            if (m is not None):
                                feh0=m.group(1).replace('m','-')
                                feh.append(feh0.replace('p',''))
                        feh=np.unique(feh)
                        print('Available metallicities: [Fe/H]= [',','.join(feh),']')
                        print("CALL IT AS: 'parsec'")
                    elif ('ames' in str.lower(path)) & ('cond' in str.lower(path)):
                        print("CALL IT AS: 'ames_cond'")
                    elif ('ames' in str.lower(path)) & ('dusty' in str.lower(path)):
                        print("CALL IT AS: 'ames_dusty'")                    
                    elif 'nextgen' in str.lower(path):
                        print("CALL IT AS: 'nextgen'")
                    elif 'settl' in str.lower(path):
                        print("CALL IT AS: 'bt_settl'")                    
                    elif 'spots' in str.lower(path):
                        expr = re.compile('f(.+)_all')
                        fspot=[]
                        for j in range(len(files[0])):
                            string=files[0][j] 
                            m=expr.match(string)
                            if (m is not None):
                                fspot.append(m.group(1)[0]+'.'+m.group(1)[1:])
                        print('Available spot fractions= [',','.join(fspot),']')
                        print("CALL IT AS: 'spots'")
                    elif 'dartmouth' in str.lower(path):
                        print("CALL IT AS: 'dartmouth'")
                    elif 'geneva' in str.lower(path):
                        print("CALL IT AS: 'geneva'")                    
                    elif 'sonora' in str.lower(path):
                        print("CALL IT AS: 'sonora_bobcat'")                    
                    elif 'yapsi' in str.lower(path):
                        print("CALL IT AS: 'yapsi'")                    
                    elif 'sb12' in str.lower(path): 
                        print("CALL IT AS: 'sb12_hy_cold'/'sb12_hy_hot'/'sb12_cf_cold'/'sb12_cf_hot'")                                        
                    elif 'b97' in str.lower(path): 
                        print("CALL IT AS: 'b97'")                                        
                    elif 'pm13' in str.lower(path): 
                        print("CALL IT AS: 'pm13'") 
                    print('--------------------------------------------------------------------------------------')
            if found==False: 
                mess='The inserted model does not exist. Check the spelling and try again. Available models: '+','.join(f_mods)
                raise ValueError(mess)
                
    @staticmethod
    def plot_2D_ext(ra=None,dec=None,l=None,b=None,par=None,d=None,color='G',n=50,reverse_xaxis=False,reverse_yaxis=False,tofile=None,ext_map='leike'):

        if type(d)==type(None): 
            if type(par)==type(None): raise NameError('Exactly one between d and par must be supplied!') 
            d=1000/par
        elif (type(d)!=type(None)) & (type(par)!=type(None)):
            raise NameError('Exactly one between d and par must be supplied!')

        dist=np.full(n**2,d)
        if (type(ra)!=type(None)) & (type(dec)!=type(None)) & (type(l)==type(None)) & (type(b)==type(None)):
            a2=np.linspace(ra[0],ra[1],n)
            d2=np.linspace(dec[0],dec[1],n)
            coo2,coo1=np.meshgrid(d2,a2)
            aa=coo1.ravel()
            dd=coo2.ravel()
            ee=MADYS.interstellar_ext(ra=aa,dec=dd,d=dist,color=color,ext_map=ext_map)
            col_name=[r'$\alpha [^\circ]$',r'$\delta [^\circ]$']
        elif (type(ra)==type(None)) & (type(dec)==type(None)) & (type(l)!=type(None)) & (type(b)!=type(None)):
            a2=np.linspace(l[0],l[1],n)
            d2=np.linspace(b[0],b[1],n)
            coo2,coo1=np.meshgrid(d2,a2)
            aa=coo1.ravel()
            dd=coo2.ravel()
            ee=MADYS.interstellar_ext(l=aa,b=dd,d=d,color=color,ext_map=ext_map)
            col_name=[r'$l [^\circ]$',r'$b [^\circ]$']
        else: raise NameError('Exactly one pair between (ra, dec) and (l,b) must be supplied!')       

        E2=ee.reshape(n,n)
        fig, ax = plt.subplots(figsize=(12,12))
        CS = ax.contourf(coo1, coo2, E2, 100)
        cbar = fig.colorbar(CS)
        ax.set_xlabel(col_name[0],fontsize=15)
        ax.set_ylabel(col_name[1],fontsize=15)
        if '-' in color: cbar.ax.set_ylabel(color+' reddening [mag]')
        else: cbar.ax.set_ylabel(color+'-band extinction [mag]',fontsize=15)
        if reverse_xaxis: plt.gca().invert_xaxis()
        if reverse_yaxis: plt.gca().invert_yaxis()
        if type(tofile)!=type(None): plt.savefig(tofile)
        plt.show()

    @staticmethod
    def filters_to_surveys(filters):    
        surveys=[]

        n_s=len(list(MADYS.filt.keys()))
        surv=np.array(list(MADYS.filt.keys()))
        for i in surv:
            s=np.array(list(MADYS.filt[i].keys()))
            if len(np.intersect1d(filters,s))>0: surveys.append(i)

        return surveys

    @staticmethod
    def info_filters(filt=None,model=None):
        
        folder = os.path.dirname(os.path.realpath(__file__))

        files=[]
        i=0
        found = False
        for x in os.walk(folder): 
            for j in range(len(x[2])): 
                if 'info_filters' in x[2][j]:
                    ff=os.path.join(x[0],x[2][j])
                    found = True
                    break

        if found == False: raise NameError('File info_filters.txt not found in your working path. Please insert it and run this function again.')

        # read column headers and number of values
        p_cols = re.compile("\s*Quantity name:\s+'(.+)'")
        p_sp = re.compile("--*")

        # get column names
        cols  = []
        file = open(ff, 'r')    
        line = file.readline()
        found = False

        if filt!=None:    
            while True:
                m = p_cols.match(line)
                if (m is not None):
                    cols.append(m.group(1))
                    if m.group(1)==filt:
                        print(line)
                        found = True
                line = file.readline()         #reads next line
                if found:
                    m1 = p_sp.match(line)
                    if (m1 is not None):
                        break
                    else: 
                        print(' '.join(line.splitlines()))
                if bool(line)==False: 
                    break
            if found==False:
                cols=np.array(cols)
                raise NameError("Quantity '"+str(filt)+"' not found: check the spelling and try again. Available filters and physical parameters: "+', '.join(cols)+'.')
            if model!=None:   
                surv=MADYS.filters_to_surveys([filt])
                model_c=MADYS.model_name(model)

                add_search_path(folder)
                for x in os.walk(folder):
                    add_search_path(x[0])      
                try:
                    model_data(surv[0],model_c)
                    return True
                except ValueError: #if the survey is not found for the isochrone set, its filters are set to NaN
                    return False

        else:
            while True:
                if bool(line)==False: 
                    break
                else: print(' '.join(line.splitlines()))                    
                line = file.readline()         #reads next line

        file.close()    
        
    @staticmethod
    def plot_chi2_map(result,indices,tofile=False,limits=None):

        iso_mass=result['iso_mass']
        iso_age=result['iso_age']
        model=result['model']
        AA,MM = np.meshgrid(iso_age,iso_mass)

        try: len(indices)
        except TypeError: indices=np.array([indices])

        if type(limits)!=type(None):
            limits=np.array(limits)
            if len(limits.shape)==1: limits=np.tile(limits, (len(indices), 1))

        p=0
        for i in indices:
            print('Star '+str(i))
            chi2=result['all_maps'][i]
            try:
                m_sol=result['all_solutions'][i]['masses']
            except KeyError:
                print('No solution was found for star '+str(i)+'. Check the log for details.')
                p+=1
                continue

            m_sol=result['all_solutions'][i]['masses']

            best=np.nanmin(chi2)
            arg_best=np.nanargmin(chi2)
            rav_chi2=chi2.ravel()

            inv_chi2=1/chi2
            plt.figure(figsize=(12,12))
            levels = 10**np.linspace(np.log10(best), np.log10(best+15), 10)
            h = plt.contourf(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm_r) #cmap=cm.coolwarm_r, extend='min') #100)
            CB = plt.colorbar(h,ticks=levels,format='%.1f')
            CB.set_label(r'$\chi^2$', rotation=270)
            m_range=[np.min(m_sol)*0.9,np.max(m_sol)*1.1]
            plt.xlim(m_range)
            plt.yscale('log')
            i70,i85=np.argmin(np.abs(iso_mass-m_range[0])),np.argmin(np.abs(iso_mass-m_range[1]))
            for j in range(i70,i85):
                plt.plot([iso_mass[j],iso_mass[j]],[iso_age[0],iso_age[-1]],color='white',linewidth=0.3)
            for j in range(len(iso_age)):
                plt.plot([iso_mass[i70],iso_mass[i85]],[iso_age[j],iso_age[j]],color='white',linewidth=0.3)

            if type(limits)!=type(None):
                if limits[p,0]!=None: plt.plot([limits[p,0],limits[p,0]],[iso_age[0],iso_age[-1]],color='white',linewidth=1)
                if limits[p,1]!=None: plt.plot([limits[p,1],limits[p,1]],[iso_age[0],iso_age[-1]],color='white',linewidth=1)
                if limits[p,2]!=None: plt.plot([iso_mass[i70],iso_mass[i85]],[limits[p,2],limits[p,2]],color='white',linewidth=1)
                if limits[p,3]!=None: plt.plot([iso_mass[i70],iso_mass[i85]],[limits[p,3],limits[p,3]],color='white',linewidth=1)

            plt.ylabel(r'$\log_{10}$(age)')
            plt.xlabel(r'mass ($M_\odot$)')
            plt.title(r'$\chi^2$ map for star '+str(i)+', '+str.upper(model))
            if tofile: plt.savefig(tofile)
            plt.show()    
            p+=1
        return

    @staticmethod
    def intersect1d_rep1(x,y):
        x1=copy.deepcopy(x)
        y1=copy.deepcopy(y)    
        r=[]
        i_1=[]
        i_2=[]
        while 1:
            r0 , i1, i2 = np.intersect1d(x1,y1,return_indices=True)
            x1[i1]='999799' if isinstance(x1[0],str) else 999799
            if len(i1)==0: break
            i_1.append(i1)
            i_2.append(i2)
            r.append(r0)
        i_1=np.concatenate(i_1)
        i_2=np.concatenate(i_2)
        r=np.concatenate(r)    
        return r,i_1,i_2

    @staticmethod
    def ang_dist(ra1,dec1,ra2,dec2,error=False):      
        try:
            ra1.unit
            dist=2*np.arcsin(np.sqrt(np.sin((dec2-dec1)/2.)**2+np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2.)**2)).to(u.deg)
            if error:
                a=dec2
                b=dec1
                d=ra2
                e=ra1
                a_err=dec2_err
                b_err=dec1_err
                d_err=ra2_err
                e_err=ra1_err
                ddec2=(np.sin(a-b)-2*np.cos(b)*np.sin(a)*np.sin((d-e)/2)**2)/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                ddec1=(np.sin(a-b)-2*np.cos(a)*np.sin(b)*np.sin((d-e)/2)**2)/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                dra2=(np.cos(a)*np.cos(b)*np.sin(d-e))/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((-e+d)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                dra1=-(np.cos(a)*np.cos(b)*np.sin(d-e))/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((-e+d)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                err=np.sqrt(dra1**2*e_err**2+dra2**2*d_err**2+ddec1**2*b_err**2+ddec2**2*a_err**2)
                return dist.value,err.value
            else: return dist.value    
        except:
            dist=2*np.arcsin(np.sqrt(np.sin((dec2-dec1)/2.*u.degree)**2+np.cos(dec2*u.degree)*np.cos(dec1*u.degree)*np.sin((ra2-ra1)/2.*u.degree)**2)).to(u.deg)
            if error:
                a=dec2*u.degree
                b=dec1*u.degree
                d=ra2*u.degree
                e=ra1*u.degree
                a_err=dec2_err*u.degree
                b_err=dec1_err*u.degree
                d_err=ra2_err*u.degree
                e_err=ra1_err*u.degree
                ddec2=(np.sin(a-b)-2*np.cos(b)*np.sin(a)*np.sin((d-e)/2)**2)/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                ddec1=(np.sin(a-b)-2*np.cos(a)*np.sin(b)*np.sin((d-e)/2)**2)/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                dra2=(np.cos(a)*np.cos(b)*np.sin(d-e))/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((-e+d)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                dra1=-(np.cos(a)*np.cos(b)*np.sin(d-e))/(2*np.sqrt(-np.cos(a)*np.cos(b)*np.sin((-e+d)/2)**2+np.cos((a-b)/2)**2)*np.sqrt(np.cos(a)*np.cos(b)*np.sin((d-e)/2)**2+np.sin((a-b)/2)**2))
                err=np.sqrt(dra1**2*e_err**2+dra2**2*d_err**2+ddec1**2*b_err**2+ddec2**2*a_err**2)
                return dist.value,err.value
            else: return dist.value
    
    def divide_query(self,query,key_name=None,id_list=None,n_it_max=10,engine='gaia',equality='=',quote_mark=False):

        if engine=='gaia': f=gaia.query
        elif engine=='vizier': f=TAPVizieR().query

        n_chunks=1
        nst=len(id_list) if type(id_list)!=type(None) else len(self.GaiaID) #dev'essere self.GaiaID nel caso della ricerca principale
        print('no. of stars: ',nst)
        done=np.zeros(nst,dtype=bool)
        nit=0
        data=[]
        while (np.sum(done)<nst) & (nit<10):
            todo,=np.where(done==False)
            st=int(len(todo)/n_chunks)
            for i in range(n_chunks):
                todo_c=todo[i*st:(i+1)*st]
                query_list=self.list_chunk(todo_c,key_name=key_name,id_list=id_list,equality=equality,quote_mark=quote_mark)
                qstr=query+query_list
                try:
                    print(qstr)
                    adql = QueryStr(qstr,verbose=False)
                    t=f(adql)
                    data.append(t)
                    done[todo_c]=True
                except JSONDecodeError: 
                    continue
            n_chunks*=2
            nit+=1
            if nit>(n_it_max-1): raise RuntimeError('Perhaps '+nst+' stars are too many?')                

        if len(data)>1: t=vstack(data)
        else: t=data[0]
        return t
