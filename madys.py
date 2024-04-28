"""

MADYS
Tool for age and mass determination of young stellar and substellar objects.
Reference: Squicciarini & Bonavita (2022), A&A 666, A15
Given a list of stars, it:
- retrieves and cross-matches photometry from several catalogs;
- corrects for interstellar extinction;
- assesses the quality of each photometric measurement;
- uses reliable photometric data to derive physical parameters (notably ages and masses)
of individual stars.
In the current release, MADYS allows a selection of one among 20 theoretical models,
many of which with several customizable parameters (metallicity, rotational velocity,
etc). Have a look to the GitHub repository and to the Readthedocs page for additional details.

Classes:
- SampleObject
- FitParams
- IsochroneGrid
- ModelHandler
- CurveObject

"""
import sys
import os
madys_path = os.path.dirname(os.path.realpath(__file__))
import copy
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage import label, center_of_mass
import time
from astropy import units as u
from astropy.constants import M_jup,M_sun,R_jup,R_sun
from astropy.coordinates import SkyCoord, Galactocentric, galactocentric_frame_defaults
from astropy.io import ascii, fits
from astropy.io.votable.exceptions import E19
from astropy.table import Table, Column, vstack, hstack, MaskedColumn
from astroquery.simbad import Simbad
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate
from json import JSONDecodeError
import gzip
import urllib
import shutil
import pickle
import re
from xml.parsers.expat import ExpatError
from http.client import RemoteDisconnected

try:
    from tap import (GaiaArchive, QueryStr)
except ModuleNotFoundError:
    os.system('pip install git+https://github.com/mfouesneau/tap')
    from tap import (GaiaArchive, QueryStr)
gaia = GaiaArchive()

dt = h5py.special_dtype(vlen=str)
MADYS_VERSION = 'v1.3.0'



def closest(array, value):
    """Given an "array" and a (list of) "value"(s), finds the j(s) such that |array[j]-value|=min((array-value)).
    "array" must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that "value" is out of range (below and above, respectively)."""
    n = len(array)
    if hasattr(value, '__len__') == False:
        if (value < array[0]):
            return 0
        elif (value > array[n-1]):
            return n-1
        jl = 0
        ju = n-1
        while (ju-jl > 1):
            jm = (ju+jl) >> 1
            if (value >= array[jm]):
                jl = jm
            else:
                ju = jm
        if (value == array[0]):
            return 0
        elif (value == array[n-1]):
            return n-1
        else:
            jn = jl + np.argmin([value-array[jl], array[jl+1]-value])
            return jn
    else:
        nv = len(value)
        jn = np.zeros(nv, dtype='int32')
        for i in range(nv):
            if (value[i] < array[0]): 
                jn[i] = 0
            elif (value[i] > array[n-1]): 
                jn[i] = n-1
            else:
                jl = 0
                ju = n-1
                while (ju-jl > 1):
                    jm=(ju+jl) >> 1
                    if (value[i] >= array[jm]):
                        jl = jm
                    else:
                        ju = jm
                if (value[i] == array[0]):
                    jn[i] = 0
                elif (value[i] == array[n-1]):
                    jn[i] = n-1
                else:
                    jn[i] = jl+np.argmin([value[i]-array[jl], array[jl+1]-value[i]])
        return jn

def where_v(elements, array, approx=False, assume_sorted=True):

    """Returns the index of the first occurrence of each of the "elements" in the
    "array". If approx==True, the indices of closest matchs
    are returned instead (in this case, the array is supposed to be sorted. Use
    assume_sorted=False instead.)"""

    if isinstance(array, list): array = np.array(array)
    try:
        dd = len(elements)
        if isinstance(elements, list): 
            elements = np.array(elements)
        dim = len(elements.shape)
    except TypeError: dim = 0

    if approx == True:
        if assume_sorted == False:
            i_sort = np.argsort(array)
            array2 = array[i_sort]
        else:
            array2 = array
        if dim == 0:
            w = closest(array2, elements)
            return w
        ind = np.zeros(len(elements), dtype=np.int16)
        for i in range(len(elements)):
            ind[i] = closest(array2,elements[i])
        if assume_sorted:
            return ind
        else:
            return i_sort[ind]
    else:
        if dim == 0:
            w, = np.where(array == elements)
            return w
        ind = np.zeros(len(elements), dtype=np.int16)
        for i in range(len(elements)):
            w, = np.where(array == elements[i])
            if len(w) == 0: 
                ind[i] = len(array)
            else: 
                ind[i] = w[0]
                
        return ind

def nansumwrapper(a, axis=None, **kwargs):

    """Wrapper for np.nansum. Unlike np.nansum, returns np.nan (and
    not 0) if all the elements of "a" are np.nan."""

    ma = np.isnan(a) == False
    sa = np.nansum(ma, axis=axis)
    sm = np.nansum(a, axis=axis,**kwargs)
    sm = np.where(sa == 0, np.nan, sm)

    return sm

def repr_table(table):

    """Returns the correct __repr__ of an astropy Table."""

    aa = {}
    for col in list(table.columns):
        aa[col] = table[col].data
    r = repr(aa)
    r = r.replace('array', 'np.array')
    try:
        eval('nan')
    except NameError:
        r = r.replace('nan', 'np.nan')
    return "Table("+r+")"

def info_filters(x=None):

    """Provides info on all available filters (if no argument is provided) or on a specific filter (specified as an argument)."""
    
    if x is None:

        temp_filters = copy.deepcopy(stored_data['filters'])
        del temp_filters['logT'], temp_filters['logR'], temp_filters['logL'], temp_filters['logg']

        print('Available filters for MADYS: ')
        print('')
        print(', '.join(temp_filters)+'.')
        print('')
        print('Available physical parameters for MADYS: ')
        print(', '.join(['logg','logL','logR','logT'])+'.')
    else:
        try:
            print(stored_data['filters'][x]['description'])
            if (x!='logT') & (x!='logL') & (x!='logg') & (x!='logR'):
                print('Wavelength: '+'{:.3f}'.format(stored_data['filters'][x]['wavelength'])+' micron')
                print('Absolute extinction A(l)/A(V): '+'{:.3f}'.format(stored_data['filters'][x]['A_coeff']))
        except KeyError:
            raise ValueError("Quantity '"+x+"' not found: check the spelling and try again. Available filters and physical parameters: "+', '.join(stored_data['filters'])+'.')
            
def make_logo():
    
    zenodo_path = list(stored_data['complete_model_list'].values())[0].split('files')[0]
    logo = """

                                                    Welcome to

     .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
    | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
    | | ____    ____ | || |      __      | || |  ________    | || |  ____  ____  | || |    _______   | |
    | ||_   \  /   _|| || |     /  \     | || | |_   ___ `.  | || | |_  _||_  _| | || |   /  ___  |  | |
    | |  |   \/   |  | || |    / /\ \    | || |   | |   `. \ | || |   \ \  / /   | || |  |  (__ \_|  | |
    | |  | |\  /| |  | || |   / ____ \   | || |   | |    | | | || |    \ \/ /    | || |   '.___`-.   | |
    | | _| |_\/_| |_ | || | _/ /    \ \_ | || |  _| |___.' / | || |    _|  |_    | || |  |`\____) |  | |
    | ||_____||_____|| || ||____|  |____|| || | |________.'  | || |   |______|   | || |  |_______.'  | |
    | |              | || |              | || |              | || |              | || |              | |
    | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
     '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 

                                                 version: {0}
                                     @ V. Squicciarini & M. Bonavita
                             code repository: https://github.com/vsquicciarini/madys
                             model repository: {1}
                             documentation: https://madys.readthedocs.io/en/latest/


     """.format(MADYS_VERSION, zenodo_path)
    
    return logo
    
def add_mag(mag1, mag2):
    
    """Adds two magnitudes."""
    
    return -2.5*np.log10(10**(-0.4*mag1)+10**(-0.4*mag2))

    
stored_data = {'models': {'data': {}, 'parameters':{}}}


class ModelHandler(object):

    """
    Class: madys.ModelHandler

    Class that handles data and metadata of the files in the /isochrone path.
    Input:
    - model_grid: string, required. Selected isochrone grid model. Use ModelHandler.available() to return a list of available models.
    - resolve: bool, optional. If True, searchs for the location of the correct file. If False, it recovers it from the dictionary 'stored_data'. Default: False.

    Attributes:
    - file: string. Full path to the local file corresponding to model_grid;
    - age_range: 2-element list. Minimum and maximum age of the grid [Myr].
    - mass_range: 2-element list. Minimum and maximum age mass of the grid [M_sun].
    - header: string. Verbose description of the model_grid.
    - feh: float. [Fe/H] of the grid.
    - he: float. Helium content of the grid.
    - afe: float. Alpha enhancement [a/Fe] of the grid.
    - v_vcrit: float. Rotational velocity of the grid.
    - fspot: float. Fraction of stellar surface covered by star spots.
    - B: int. Whether magnetic fields are included (1) or not (0) in the grid.

    Built-in methods:

    1) __repr__
    Returns self.header.

    Methods (use help() to have more detailed info):

    1) get_contents
    Returns the full or part of the content of self.file, i.e. the data+metadata associated to a model_grid.

    2) get_data
    Wrapper for get_contents() that only returns the data associated to a model_grid.

    3) available
    Prints info about locally available models.

    4) download_model
    Given a model grid, it downloads it.

    """

    def __init__(self, model_grid, resolve=False):

        self.model_grid = model_grid
        if resolve:
            found = False
            fname = model_grid+'.h5'
            for root, dirs, files in os.walk(os.path.join(madys_path,'isochrones')):
                for name in files:
                    if os.path.isfile(os.path.join(root, fname)):
                        found = True
                        true_path = root
                        break
            if not found:
                raise ValueError('File {0} for model {1} does not exists. Are you sure it is in your search path?'.format(fname, model_grid))
            self.file = os.path.join(true_path,fname)
        else:
            try:
                self.file = stored_data['local_model_list'][model_grid]
            except KeyError:
                raise ValueError('Model {0} not found. Are you sure it is in your search path?'.format(model_grid))
        
        model_info = self.get_contents(get_isochrone_ranges=True, get_headers=True, get_attributes=True)
        
        attrs = model_info['attributes']
        self.B, self.afe, self.feh, self.fspot, self.he = attrs['B'], attrs['afe'], attrs['feh'], attrs['fspot'], attrs['he']
        self.header = model_info['headers']
        self.mass_range, self.age_range =  model_info['agemass_range']

    def __repr__(self):
        return self.header[0]

    def get_data(self):
        """
        Wrapper for ModelHandler.get_contents(get_data = True)['data'], i.e. with only 'get_data' set to True.
            Input: None.
            Output:
            - masses: numpy array. The n_m masses of the grid [M_sun].
            - ages: numpy array. The n_a ages of the grid [Myr].
            - filters: numpy array. List of the n_f filters in the grid.
            - dic: dictionary. Contains all the metadata of the file, excluding the headers.
            - data: numpy array with shape (n_m,n_a,n_f). Isochrone grid.
        """
        
        return self.get_contents(get_data=True)['data']
        
    def get_contents(self, get_data=False, get_isochrone_ranges=False, get_headers=False, get_attributes=False):
        """
        Returns (part of) the content of self.file, i.e. the data associated to a model_grid.
            Input: 
            - get_data: bool, optional. Whether to return (mass, age, filters, data) of the grid. Default=False.
            - get_isochrone_ranges: bool, optional. Whether to return (mass_range, age_range, filters of the grid. Default=False.
            - get_headers: bool, optional. Whether to return (header, version_header) of the grid. Default=False.
            - get_attributes: bool, optional. Whether to return all the attributes of the grid BUT the headers. Default=False.
            Output:
            - results: dictionary. Depending on the truth value of the named keywords, it can or cannot contain the following keywords:
                - attributes: dictionary. Contains all the metadata of the file, excluding the headers.
                - headers: tuple. Contains the model suite header and the (usually empty) model version header.
                - agemass_range: tuple. Two elements: 
                    - a 2-element list with [min(mass), max(mass)], measured in M_sun;
                    - a 2-element list with [min(age), max(age)], measured in Myr.
                - filters: numpy array. List of the n_f filters in the grid.
                - data: tuple containing the following:
                    - masses: numpy array. The n_m masses of the grid [M_sun].
                    - ages: numpy array. The n_a ages of the grid [Myr].
                    - filters: numpy array. List of the n_f filters in the grid.
                    - dic: dictionary. Contains all the metadata of the file, excluding the headers.
                    - data: numpy array with shape (n_m,n_a,n_f). Isochrone grid.
        """

        results = {}
        with h5py.File(self.file, "r") as f:
            if get_attributes:
                dic = {}
                for i in f.attrs.keys():
                    if (i == 'header') | (i == 'version_header'): continue
                    dic[i] = f.attrs[i]
                results['attributes'] = dic
            if get_headers:
                header = f.attrs['header']
                for j in range(10):
                    header = header.replace(' #','#')

                try:
                    version_header = f.attrs['version_header']
                    for j in range(10):
                        version_header = version_header.replace(' #','#')

                    headers = (header.rstrip(), version_header.rstrip())
                except KeyError:
                    headers = (header.rstrip(), """""")
                results['headers'] = headers
            if get_isochrone_ranges:
                m = f.get('masses')[:]
                a = f.get('ages')[:]
                fi = f.get('filters')[:]
                fi = np.array(fi, dtype='str')
                results['agemass_range'] = ([np.min(m),np.max(m)], [np.min(a),np.max(a)])
                results['filters'] = fi
            if get_data:
                if self.model_grid in stored_data['models']['data'].keys():
                    results['data'] = stored_data['models']['data'][self.model_grid]
                else:
                    with h5py.File(self.file, "r") as f:
                        m = f.get('masses')[:]
                        a = f.get('ages')[:]
                        fi = f.get('filters')[:]
                        dat = f.get('data')[:]
                        dic = {}
                        for i in f.attrs.keys():
                            if (i == 'header') | (i == 'version_header'): continue
                            dic[i] = f.attrs[i]
                        fi = np.array(fi, dtype='str')

                    data = (m, a, fi, dic, dat)
                    stored_data['models']['data'][self.model_grid] = data
                    results['data'] = data

            return results

    @classmethod
    def available(cls, key=None, verbose=True):
        """
        A class method.
        Prints info about locally available models.
        Informs about 1) the calling sequence; 2) age and mass ranges; 3) list of filters;
        4) available parameters / adopted parameters; 5) literature reference.
            Input:
            - key: string, optional. If selected, searchs the corresponding model. If None, prints info about all the available model suites. Default: None.
              It can be either:
                1) 'full_model_list';
                2) a valid model_family;
                3) a valid model_suite;
                4) a valid model_version.
            - verbose: bool, optional. If True, prints all information about a specific or all models. If False, only raises an error if the model is not found. Default: True.
            Output: none, but the method prints:
                1) key not provided: mass range, age range, available parameters, calling sequence for all the model versions belonging to each model suite;
                2) key='full_model_list': taxonomic classification of the complete set of models available on Zenodo;
                3) key is a model_suite: mass range, age range, available parameters, calling sequence for all the model versions belonging to each model suite;
                4) key is a model_family: mass range, age range, available parameters, calling sequence for all the model versions belonging to each model family;
                5) key is a model_version: mass range, age range, available parameters, calling sequence of the specified model version.
        """

        attrs_list, header_list, version_header_list = [], [], []
        mass_list, age_list, filter_list = [], [], []
        for root, dirs, files in os.walk(madys_path):
            for name in files:
                if root.endswith('extinction'): 
                    continue
                if name.endswith('h5'):
                    model_grid = ModelHandler(name[:-3])

                    model_info = model_grid.get_contents(get_isochrone_ranges=True, get_headers=True, get_attributes=True)

                    headers = model_info['headers']
                    agemass_ranges = model_info['agemass_range']

                    header_list.append(headers[0])
                    version_header_list.append(headers[1])
                    mass_list.append(agemass_ranges[0])
                    age_list.append(agemass_ranges[1])
                    attrs_list.append(model_info['attributes'])
                    filter_list.append(model_info['filters'])

        model_families = np.array([attr['model_family'].lower() for attr in attrs_list])
        model_suites = np.array([attr['model_suite'].lower() for attr in attrs_list])
        model_versions = np.array([attr['model_version'].lower() for attr in attrs_list])
        feh_list = np.array([attr['feh'] for attr in attrs_list])
        he_list = np.array([attr['he'] for attr in attrs_list])
        afe_list = np.array([attr['afe'] for attr in attrs_list])
        B_list = np.array([attr['B'] for attr in attrs_list])
        v_vcrit_list = np.array([attr['v_vcrit'] for attr in attrs_list])
        fspot_list = np.array([attr['fspot'] for attr in attrs_list])

        unique_names = np.unique(model_suites)
        __, i1, i2 = np.intersect1d(unique_names, model_suites, return_indices=True)

        if key is None:
            print('Available models for MADYS: ')
            print('')
            for i in i2:
                print(header_list[i][:-1])
                surveys = list(np.unique([stored_data['filters'][k]['survey'] for k in filter_list[i]]))
                d = 1
                while np.sum(np.around(mass_list[i],d)==0) > 0:
                    d += 1
                    if d == 8: break
                mass_range = mass_list[i]
                print(f'# Mass range (M_sun): [{mass_range[0]:.{d}f}, {mass_range[1]:.{d}f}]')
                print('# Age range (Myr): '+str(list(np.around(age_list[i],1))))
                print('# Available photometric systems: ')
                print('# '+', '.join(surveys))
                w, = np.where(np.array(model_suites) == model_suites[i])
                feh = np.unique(feh_list[w]).astype(str)
                print('# Available metallicities: ['+','.join(feh)+']')
                v_vcrit = np.unique(v_vcrit_list[w]).astype(str)
                print('# Available rotational velocities: ['+','.join(v_vcrit)+']')
                afe = np.unique(afe_list[w]).astype(str)
                print('# Available alpha enhancements: ['+','.join(afe)+']')
                B = np.unique(B_list[w]).astype(str)
                print('# Available magnetic field strengths: ['+','.join(B)+']')
                fspot = np.unique(fspot_list[w]).astype(str)
                print('# Available spot fractions: ['+','.join(fspot)+']')
                he = np.unique(he_list[w]).astype(str)
                print('# Available helium contents: ['+','.join(he)+']')
                print('# Model family: {0}'.format(ModelHandler._get_model_tree_info(model_versions[i], 'model_family')))
                print('# Model suite: {0}'.format(ModelHandler._get_model_tree_info(model_versions[i], 'model_suite')))
                if version_header_list[i] != '':
                    versions = np.unique(np.array(model_versions)[w])
                    print("# Call it as: '"+"'/ '".join(versions)+"'")
                else: print("# Call it as: '"+model_suites[i]+"'")
        elif key == 'full_model_list':
            print('Full list of models available on Zenodo: ')

            zenodo_grids = list(stored_data['complete_model_list'].keys())
            zenodo_versions = np.array([i.split('_')[0] for i in zenodo_grids])
            zenodo_families = np.array([i.lower() for i in ModelHandler._get_model_tree_info(zenodo_versions, 'model_family')])
            zenodo_suites = np.array([i.lower() for i in ModelHandler._get_model_tree_info(zenodo_versions, 'model_suite')])

            unique_families = np.unique(zenodo_families)
            for fam in unique_families:
                print('Model family: '+fam)
                w1, = np.where(zenodo_families == fam)
                unique_suites = np.unique(zenodo_suites[w1])
                for sui in unique_suites:
                    print('    Model suite: '+sui)
                    w2, = np.where(zenodo_suites==sui)
                    unique_versions = np.unique(zenodo_versions[w2])
                    for ver in unique_versions:
                        print('        Model version: '+ver)
                        w3, = np.where(zenodo_versions==ver)
                        for gri in w3:
                            print('            Model grid: '+zenodo_grids[gri])
                print('')
        else:
            key = key.lower()
            __, if1, if2 = np.intersect1d(model_families,[key], return_indices=True)
            __, in1, in2 = np.intersect1d(model_suites,[key], return_indices=True)
            __, is1, is2 = np.intersect1d(model_versions,[key], return_indices=True)

            zenodo_grids = list(stored_data['complete_model_list'].keys())
            zenodo_versions = np.array([i.split('_')[0] for i in zenodo_grids])
            zenodo_families = np.array([i.lower() for i in ModelHandler._get_model_tree_info(zenodo_versions,'model_family')])

            if len(in1)>0:
                if verbose:
                    in1=in1[0]
                    print(header_list[in1][:-1])
                    surveys = list(np.unique([stored_data['filters'][k]['survey'] for k in filter_list[in1]]))
                    d=1
                    while np.sum(np.around(mass_list[in1],d)==0)>0:
                        d+=1
                        if d==8: break
                    mass_range = mass_list[in1]
                    print(f'# Mass range (M_sun): [{mass_range[0]:.{d}f}, {mass_range[1]:.{d}f}]')
                    print('# Age range (Myr): '+str(list(np.around(age_list[in1],1))))
                    print('# Available photometric systems: ')
                    print('# '+', '.join(surveys))
                    w, = np.where(np.array(model_suites)==model_suites[in1])
                    feh = np.unique(feh_list[w]).astype(str)
                    print('# Available metallicities: ['+','.join(feh)+']')
                    v_vcrit = np.unique(v_vcrit_list[w]).astype(str)
                    print('# Available rotational velocities: ['+','.join(v_vcrit)+']')
                    afe = np.unique(afe_list[w]).astype(str)
                    print('# Available alpha enhancements: ['+','.join(afe)+']')
                    B = np.unique(B_list[w]).astype(str)
                    print('# Available magnetic field strengths: ['+','.join(B)+']')
                    fspot = np.unique(fspot_list[w]).astype(str)
                    print('# Available spot fractions: ['+','.join(fspot)+']')
                    he = np.unique(he_list[w]).astype(str)
                    print('# Available helium contents: ['+','.join(he)+']')
                    print('# Model family: {0}'.format(ModelHandler._get_model_tree_info(model_versions[in1],'model_family')))
                    print('# Model suite: {0}'.format(ModelHandler._get_model_tree_info(model_versions[in1],'model_suite')))
                    if version_header_list[in1] != '':
                        versions = np.unique(np.array(model_versions)[w])
                        print("# Call it as: '"+"'/ '".join(versions)+"'")
                    else: print("# Call it as: '"+model_suites[in1]+"'")
            elif len(is1)>0:
                if verbose:
                    is1 = is1[0]
                    w, = np.where(np.array(model_versions) == model_versions[is1])
                    print(version_header_list[is1][:-1])
                    print(header_list[is1][1:-1])
                    surveys = list(np.unique([stored_data['filters'][k]['survey'] for k in filter_list[is1]]))
                    d=1
                    while np.sum(np.around(mass_list[is1],d) == 0) > 0:
                        d += 1
                        if d == 8: break
                    mass_range = mass_list[is1]
                    print(f'# Mass range (M_sun): [{mass_range[0]:.{d}f}, {mass_range[1]:.{d}f}]')
                    print('# Age range (Myr): '+str(list(np.around(age_list[is1],1))))
                    print('# Available photometric systems: ')
                    print('# '+', '.join(surveys))
                    feh = np.unique(feh_list[w]).astype(str)
                    print('# Available metallicities: ['+','.join(feh)+']')
                    v_vcrit = np.unique(v_vcrit_list[w]).astype(str)
                    print('# Available rotational velocities: ['+','.join(v_vcrit)+']')
                    afe = np.unique(afe_list[w]).astype(str)
                    print('# Available alpha enhancements: ['+','.join(afe)+']')
                    B = np.unique(B_list[w]).astype(str)
                    print('# Available magnetic field strengths: ['+','.join(B)+']')
                    fspot = np.unique(fspot_list[w]).astype(str)
                    print('# Available spot fractions: ['+','.join(fspot)+']')
                    he = np.unique(he_list[w]).astype(str)
                    print('# Available helium contents: ['+','.join(he)+']')
                    versions = np.unique(np.array(model_versions)[w])
                    print("# Call it as: '"+model_versions[is1]+"'")
            elif len(if1)>0:
                if verbose:
                    if1 = if1[0]
                    print('Model family: '+model_families[if1])
                    w, = np.where(np.array(model_families) == model_families[if1])
                    versions = np.unique(np.array(model_versions)[w])
                    __, is01, is02 = np.intersect1d(np.array(model_versions)[w], versions, return_indices=True)
                    for i in range(len(is01)):
                        print(version_header_list[w[is01[i]]][:-1])
                        print(header_list[w[is01[i]]][1:-1])
                        surveys = list(np.unique([stored_data['filters'][k]['survey'] for k in filter_list[w[is01[i]]]]))
                        d=1
                        while np.sum(np.around(mass_list[w[is01[i]]],d) == 0) > 0:
                            d += 1
                            if d == 8: break
                        mass_range = mass_list[w[is01[i]]]
                        print(f'# Mass range (M_sun): [{mass_range[0]:.{d}f}, {mass_range[1]:.{d}f}]')
                        print('# Age range (Myr): '+str(list(np.around(age_list[w[is01[i]]],1))))
                        print('# Available photometric systems: ')
                        print('# '+', '.join(surveys))
                        ww, = np.where(np.array(model_versions)==versions[is02[i]])
                        feh = np.unique(feh_list[ww]).astype(str)
                        print('# Available metallicities: ['+','.join(feh)+']')
                        v_vcrit = np.unique(v_vcrit_list[ww]).astype(str)
                        print('# Available rotational velocities: ['+','.join(v_vcrit)+']')
                        afe = np.unique(afe_list[ww]).astype(str)
                        print('# Available alpha enhancements: ['+','.join(afe)+']')
                        B = np.unique(B_list[ww]).astype(str)
                        print('# Available magnetic field strengths: ['+','.join(B)+']')
                        fspot = np.unique(fspot_list[ww]).astype(str)
                        print('# Available spot fractions: ['+','.join(fspot)+']')
                        he = np.unique(he_list[ww]).astype(str)
                        print('# Available helium contents: ['+','.join(he)+']')
                        print("# Call it as: '"+versions[is02[i]]+"'")
            elif key in zenodo_versions:
                print('The model '+key+' is not available in your working path.')
                print('However, there are model grids on Zenodo, which you can download via ModelHandler.download_model().')
                w, = np.where(zenodo_versions==key)
                print('Available model grids within this model version:')
                print(', '.join(np.array(zenodo_grids)[w]))
            elif key in zenodo_families:
                print('The selected model "'+key+'" is a family of models. No related grid is available in your working path.')
                key_list = np.array(list(stored_data['model_families'].keys()))
                key_list_lower = np.array([i.lower() for i in key_list])
                w, = np.where(key_list_lower == key)
                available_suites = stored_data['model_families'][key_list[w[0]]]
                for suite in available_suites.keys():
                    versions = available_suites[suite]
                    for version in versions:
                        ModelHandler.available(version)
                        print('')

            else:
                mess='The inserted model does not exist. Check the spelling and try again. Available models: '+', '.join(unique_names)+'.'
                raise ValueError(mess)

    @staticmethod
    def _get_model_tree_info(model_version,info):

        model_tree = stored_data['model_families']

        if (info != 'model_family') & (info != 'model_suite'):
            raise ValueError('Select one between "model_family" and "model_suite".')

        if isinstance(model_version, str):
            found = False
            for model_family in model_tree.keys():
                model_suites = model_tree[model_family]
                for model_suite in model_suites.keys():
                    model_versions = model_suites[model_suite]
                    if model_version in model_versions:
                        found = True
                        break
                if found: break

            if found: 
                if info == 'model_family': return model_family
                elif info == 'model_suite': return model_suite
                return model_family, model_suite
            else: raise ValueError('Invalid model provided: {0}'.format(model_version))
        else:
            l = []
            for i in range(len(model_version)):
                l.append(ModelHandler._get_model_tree_info(model_version[i],info))
            return l                
                     
    @staticmethod
    def _available_parameters(model_suite_or_version):

        if model_suite_or_version in stored_data['models']['parameters'].keys():
            return stored_data['models']['parameters'][model_suite_or_version]
        else:
            attrs_list = []
            for root, dirs, files in os.walk(madys_path):
                for name in files:
                    if root.endswith('extinction'): continue
                    if name.endswith('h5'):
                        model_grid = ModelHandler(name[:-3])
                        attrs_list.append(model_grid.get_contents(get_attributes=True)['attributes'])
            model_versions = [attr['model_version'] for attr in attrs_list]
            model_suites = [attr['model_suite'] for attr in attrs_list]
            feh_list = np.array([attr['feh'] for attr in attrs_list])
            he_list = np.array([attr['he'] for attr in attrs_list])
            afe_list = np.array([attr['afe'] for attr in attrs_list])
            B_list = np.array([attr['B'] for attr in attrs_list])
            v_vcrit_list = np.array([attr['v_vcrit'] for attr in attrs_list])
            fspot_list = np.array([attr['fspot'] for attr in attrs_list])

            __, is1, __ = np.intersect1d(model_suites, [model_suite_or_version], return_indices=True)
            __, is2, __ = np.intersect1d(model_versions, [model_suite_or_version], return_indices=True)

            if len(is1) > 0:
                is1 = is1[0]
                w, = np.where(np.array(model_suites) == model_suites[is1])
                feh = np.unique(feh_list[w])
                v_vcrit = np.unique(v_vcrit_list[w])
                afe = np.unique(afe_list[w])
                B = np.unique(B_list[w])
                fspot = np.unique(fspot_list[w])
                he = np.unique(he_list[w])
            elif len(is2) > 0:
                is2 = is2[0]
                w, = np.where(np.array(model_versions) == model_versions[is2])
                feh = np.unique(feh_list[w])
                v_vcrit = np.unique(v_vcrit_list[w])
                afe = np.unique(afe_list[w])
                B = np.unique(B_list[w])
                fspot = np.unique(fspot_list[w])
                he = np.unique(he_list[w])
            else:
                raise ValueError('Model '+model_suite_or_version+' not found!')

            res = {'feh':feh, 'he':he, 'afe':afe, 
                   'v_vcrit':v_vcrit, 'fspot':fspot, 'B':B}
            stored_data['models']['parameters'][model_suite_or_version] = res
            return res

    @classmethod
    def _model_list(cls, dtype='grid', return_type='list'):

        attrs_list = []
        file_list = []
        for root, dirs, files in os.walk(madys_path):
            for name in files:
                if root.endswith('extinction'): continue
                if name.endswith('h5'):
                    model = ModelHandler(name[:-3],resolve=True)
                    attrs_list.append(model.get_contents(get_attributes = True)['attributes'])
                    file_list.append(model.file)

        try:
            res=[attr['model_'+dtype] for attr in attrs_list]
        except KeyError: raise KeyError("Valid arguments for dtype: 'grid', 'version', 'name', 'family'. ")

        if return_type=='dict':
            unique_models = np.unique(res)
            __, i1, i2 = np.intersect1d(unique_models,np.array(res),return_indices=True)
            dic = {}
            res = np.array(res)
            for i, r in enumerate(res[i2]): dic[r] = file_list[i2[i]]
            return dic
        else: return np.unique(res)

    @staticmethod
    def _version_to_grid(model_version, model_params):

        code_dict={'mist':'211000','starevol':'201000','spots':'200200','dartmouth':'21000Y',
                   'yapsi':'200020','pm13':'000000','parsec2':'202000'}
        try:
            code=code_dict[model_version]
        except KeyError:
            code='200000'
        keys=['feh', 'afe', 'v_vcrit', 'fspot', 'he', 'B']

        if model_version == 'starevol':
            def_params = {'feh':-0.01, 'afe': 0.00, 'v_vcrit': 0.00, 'fspot': 0.00, 'B':0, 'he':0.27}
        else:
            def_params = {'feh':0.00, 'afe': 0.00, 'v_vcrit': 0.00, 'fspot': 0.00, 'B':0, 'he':0.27}
        
        for key in def_params.keys():
            if key not in model_params: model_params[key]=def_params[key]

        model_grid=model_version+''
        for i,k in enumerate(keys):
            if code[i]=='0': continue
            elif code[i]=='Y':
                if model_params[k]==0: model_grid+='_B0'
                else: model_grid+='_B1'
            else:
                value=model_params[k]
                s='m' if value<0 else 'p'
                ff="{:."+code[i]+"f}"
                s1=ff.format(abs(value))
                model_grid+='_'+s+s1

        return model_grid

    @staticmethod
    def _grid_to_version(model_grid):

        code_dict={'mist':'211000','starevol':'201000','spots':'200200','dartmouth':'21000Y',
                   'yapsi':'200020','pm13':'000000','parsec2':'202000'}

        split_model = model_grid.split('_')
        model_version = split_model[0]
        try:
            code=code_dict[split_model[0]]
        except KeyError:
            code='200000'

        sort_par = ['feh','afe','v_vcrit','fspot','he','B']

        c=1
        model_params={}
        for i,k in enumerate(sort_par):
            if code[i]=='0': continue
            elif code[i]=='Y':
                model_params[k]=int(split_model[c][1:])
                c+=1
            else:
                model_params[k]=float(split_model[c].replace('p','+').replace('m','-'))
                c+=1

        return model_version,model_params

    @staticmethod
    def _check_updates():
        PIK=os.path.join(madys_path,'utils','models.pkl')
        with open(PIK,'rb') as f:
            local_model_db=pickle.load(f)
            local_model_families=pickle.load(f)
            local_zenodo_record=pickle.load(f)
        current_release_model_list=list(local_model_db.keys())
        try:
            tmp=urllib.request.urlretrieve('https://github.com/vsquicciarini/madys/raw/main/utils/models.pkl')[0]
            with open(tmp,'rb') as f:
                github_model_db=pickle.load(f)
                github_model_families=pickle.load(f)
                github_zenodo_record=pickle.load(f)
            github_model_list=list(github_model_db.keys())
            if len(github_model_list)>len(current_release_model_list):
                print('New models found on the GitHub repository. MADYS is updating its model list. Please wait...')
                urllib.request.urlretrieve('https://github.com/vsquicciarini/madys/raw/main/utils/models.pkl',PIK)
                print('Done.')
                stored_data['complete_model_list'] = github_model_db
            elif local_zenodo_record!=github_zenodo_record:
                print('The Zenodo record appears to have changed. MADYS is updating its model list. Please wait...')
                urllib.request.urlretrieve('https://github.com/vsquicciarini/madys/raw/main/utils/models.pkl',PIK)
                print('Done.')
                stored_data['complete_model_list'] = github_model_db
                stored_data['model_families'] = github_model_families
            else:
                stored_data['complete_model_list'] = local_model_db
                stored_data['model_families'] = local_model_families
        except:
                print('It was not possible to check the GitHub page for updates. Using local list of models.')
                stored_data['complete_model_list'] = local_model_db
                stored_data['model_families'] = local_model_families

    @staticmethod
    def _load_local_models(reload=False):
        if ('local_model_list' not in stored_data.keys()) | (reload==True):
            stored_data['local_model_list'] = ModelHandler._model_list(return_type='dict')

    @staticmethod
    def _find_model_grid(model_version, start_params):

        local_model_list = list(stored_data['local_model_list'].keys())
        
        if isinstance(start_params,dict):
            model_params1 = ModelHandler._find_match(model_version,start_params,list(stored_data['complete_model_list'].keys()))
            
            sol1 = ModelHandler._version_to_grid(model_version,model_params1)
            
            if len(local_model_list)==0:
                print('No model '+model_version+' found in your local path. The best-matching model for your request would be '+sol1+'.')
                while 1:
                    value = input("Do you want me to download this model? Selecting 'N' will end the program. [Y/N]:\n")
                    if str.lower(value)=='y':
                        print('Downloading the model...')
                        break
                    elif str.lower(value)=='n':
                        raise KeyboardInterrupt('No analysis is possible if the model is not present locally. Execution ended.')
                    else:
                        print("Invalid choice. Please select 'Y' or 'N'.")
                if str.lower(value)=='y':
                    ModelHandler.download_model(sol1)
                    return sol1
            elif sol1 not in local_model_list:
                model_versions = np.array([mod.split('_')[0] for mod in local_model_list])
                w, = np.where(model_versions == model_version)
                n_m=len(w)
                if n_m==0:
                    print('No model '+model_version+' found in your local path. The best-matching model for your request would be '+sol1+'.')
                    while 1:
                        value = input("Do you want me to download this model? Selecting 'N' will end the program. [Y/N]:\n")
                        if str.lower(value)=='y':
                            print('Downloading the model...')
                            break
                        elif str.lower(value)=='n':
                            raise KeyboardInterrupt('No analysis is possible if the model is not present locally. Execution ended.')
                        else:
                            print("Invalid choice. Please select 'Y' or 'N'.")
                    if str.lower(value)=='y':
                        ModelHandler.download_model(sol1)
                        return sol1
                else:
                    true_model_list = list(np.array(local_model_list)[w])
                    
                    chi2 = np.zeros(n_m)
                    for q in range(n_m):
                        par = ModelHandler._grid_to_version(true_model_list[q])[1]
                        for k in par.keys():
                            chi2[q] += (par[k]-model_params1[k])**2
                    arg_min = np.argmin(chi2)
                    sol2 = true_model_list[arg_min]
                    sol2_dict = ModelHandler._grid_to_version(true_model_list[arg_min])[1]
                    

                    print('The closest model (M1) to the input has: '+str(sol2_dict)+', but a closer match (M2) was found in the MADYS database:')
                    print(str(model_params1)+'.')
                    while 1:
                        value = input("Do you want me to download this model? Press 'N' to use M1 instead. [Y/N]:\n")
                        if str.lower(value)=='y':
                            print('Downloading the model...')
                            break
                        elif str.lower(value)=='n':
                            break
                        else:
                            print("Invalid choice. Please select 'Y' or 'N'.")
                    if str.lower(value)=='y':
                        ModelHandler.download_model(sol1)
                        return sol1
                    else: return sol2
            else:
                return sol1

        elif isinstance(start_params,list):

            try:
                ModelHandler._find_match(model_version,start_params[0],list(stored_data['complete_model_list'].keys()))
            except ValueError as e:
                msg = 'Model '+model_version+' does not exist. Check the spelling, and re-run MADYS again.'
                e.args = (msg,)
                raise

            sol1_list=[]
            for i in range(len(start_params)):
                model_params1 = ModelHandler._find_match(model_version,start_params[i],list(stored_data['complete_model_list'].keys()))
                sol1 = ModelHandler._version_to_grid(model_version,model_params1)
                sol1_list.append(sol1)

            __, i1, i2 = np.intersect1d(sol1_list,local_model_list,return_indices=True)
            lacking_models = np.setdiff1d(sol1_list,local_model_list)

            if len(lacking_models)==0:
                pass
            else:
                if len(i1)>0:
                    print('Based on input parameters, it seems that some model grids are better suited to (a portion of) your sample than the grids that are locally available.')
                    print('Locally available grids: '+','.join(local_model_list))
                    print('MADYS suggests you to download the following additional grids: '+', '.join(lacking_models))
                else:
                    print('No grid for model version '+model_version+' was found locally.')
                    print('Based on input parameters, MADYS suggests you to download the following grids: '+', '.join(lacking_models))
                n_mod = len(lacking_models)
                for i in range(n_mod):
                    print('Select code {0} to download model {1}'.format(i,lacking_models[i]))
                while 1:
                    print('Type all the desired codes by separating them with a comma: e.g., 0, 2, 3. Type -1 not to download any model, 999 to download all the models.')
                    value = input("Please provide codes:\n")
                    values=[int(v) for v in value.split(',')]
                    if (np.max(values)>=n_mod) & (np.max(values)!=999):
                        print('Invalid choice: only {0} models are available'.format(n_mod))
                    elif np.min(values)<-1:
                        print('Invalid choice: type -1 if you prefer not to download any model.')
                    elif (np.min(values)==-1) & (len(values)>1):
                        print('Invalid choice: type just -1 (no comma) if you prefer not to download any model.')
                    else: break
                if (np.max(values)==999):
                    values = np.arange(0,n_mod)
                if (np.min(values)==-1) & (len(i1)==0):
                    msg = """You decided not to download any grid for model_version """+model_version+""".
                However, the relative folder is empty, so MADYS does not have any model to compare data with.
                Re-run the program, downloading at least one model when prompted.
                Program ended."""
                elif np.min(values)==-1:
                    return
                else:
                    for value in values:
                        dl_model = lacking_models[value]
                        print('Downloading model '+dl_model+'...')
                        ModelHandler.download_model(dl_model)
                    print('Download ended.')
        else: raise ValueError('Invalid input provided. Valid input types: dictionary, list.')

    @staticmethod
    def download_model(model_grid, verbose=True):
        """
        A static method. Given a model grid, it downloads it.
        The file is downloaded from Zenodo and put into the correct local directory.
        The model list is automatically updated: there's no need to restart the program.
            Input:
            - model_grid: string, required. Model grid to be downloaded.
            Output: none.
        """
        
        model_version = ModelHandler._grid_to_version(model_grid)[0]
        model_family = ModelHandler._get_model_tree_info(model_version,'model_family')

        download_path = os.path.join(madys_path,'isochrones',model_family)
        fname = os.path.join(download_path,model_grid+'.h5')
        if os.path.isfile(fname): return
        if os.path.isdir(download_path)==False: os.makedirs(download_path)
        try:
            urllib.request.urlretrieve(stored_data['complete_model_list'][model_grid], fname)
            f=h5py.File(fname, 'r')
            if verbose: print('Model correctly downloaded.')
        except Exception as e:
            if os.path.isfile(fname): os.remove(fname)
            msg = """
            The file was not downloaded correctly. Possible reasons:
                1) a connection error;
                2) the file url has changed.
            Please download the file manually, and insert it in the following path:
            """+download_path+"""
            Then run MADYS again.
            """
            e.args = (msg,)
            raise
        ModelHandler._load_local_models(reload=True)

    @staticmethod
    def _find_match(model_version, model_params, model_list, approximate=False):

        model_versions = np.array([mod.split('_')[0] for mod in model_list])
        w, = np.where(model_versions == model_version)
        n_m = len(w)
        true_model_list = list(np.array(model_list)[w])

        if len(w)==0:
            raise ValueError('Model '+model_version+' does not exist. Are you sure you have spelled it correctly?')

        if approximate:
            def_params = {'feh':0.00, 'afe': 0.00, 'v_vcrit': 0.00, 'fspot': 0.00, 'B':0, 'he':0.27}
            for key in def_params.keys():
                if key not in model_params: model_params[key]=def_params[key]
            chi2 = np.zeros(n_m)
            for q in range(n_m):
                par = ModelHandler._grid_to_version(true_model_list[q])[1]
                for k in par.keys():
                    chi2[q] += (par[k] - model_params[k])**2
            arg_min = np.argmin(chi2)
            sol2 = true_model_list[arg_min]
            sol2_dict = ModelHandler._grid_to_version(true_model_list[arg_min])[1]
            return sol2_dict
        else:
            res={}
            used_keys = list(ModelHandler._grid_to_version(true_model_list[0])[1].keys())
            for k in used_keys:
                if k in model_params:
                    param_array = np.array([ModelHandler._grid_to_version(j)[1][k] for j in true_model_list])
                    arg = np.argmin(abs(param_array-model_params[k]))
                    res[k] = param_array[arg]

            return res

    @staticmethod
    def _load_filters():
        PIK = os.path.join(madys_path, 'utils', 'filters.pkl')
        with open(PIK,'rb') as f:
            filter_dict = pickle.load(f)
        stored_data['filters'] = filter_dict


class IsochroneGrid(object):

    """
    Class: madys.IsochroneGrid

    Class that creates a set of theoretical isochrones.
    Input:
    - model_version: string, required. Selected model_version of the isochrone grid. Use ModelHandler.available() for further information on available model versions.
    - filters: list or numpy array, required. Set of filters of the final grid. Use info_filters() for further information on available filters.
    - mass_range: list or numpuy array, optional. It can be either:
            1) a two-element list with minimum and maximum mass within the grid (M_sun);
            2) a numpy array, giving the desired mass steps (M_sun) at which the model is evaluated.
      Default: [0.01, 1.4].
    - age_range: list or numpy array, optional. It can be either:
            1) a 1D numpy array, so that the i-th age (Myr) is used as fixed age for the i-th star;
            2) a 2D numpy array with 2 columns. The i-th row defines (lower_age,upper_age) range in which one or more solutions are found for the i-th star.
            3) a 2D numpy array with 3 columns. The i-th row is used as (mean_age,lower_age,upper_age) for the i-th star; mean_age is used as in case 2), and [lower_age, upper_age] are used as in case 3).
            4) a two-element list with minimum and maximum age to consider for the whole sample (Myr);
      Default: [1,1000].
    - n_steps: list, optional. Number of (mass, age) steps of the interpolated grid. Default: [1000,500].
    - n_try: int, optional. Number of Monte Carlo iteractions for each star. Default: 1000.
    - feh: float, optional. Selects [Fe/H] of the isochrone set. Default: 0.00 (=solar metallicity).
    - he: float, optional. Selects helium fraction Y of the isochrone set. Default: solar Y (depends on the model).
    - afe: float, optional. Selects alpha enhancement [a/Fe] of the isochrone set. Default: 0.00.
    - v_vcrit: float, optional. Selects rotational velocity of the isochrone set. Default: 0.00 (non-rotating).
    - fspot: float, optional. Selects fraction of stellar surface covered by star spots. Default: 0.00.
    - B: int, optional. Set to 1 to turn on the magnetic field (only for Dartmouth models). Default: 0.
    - fill_value: array-like or (array-like, array_like) or “extrapolate”, optional. How the interpolation over mass deals with values outside the original range. Default: np.nan. See scipy.interpolate.interp1d for details.
    - logger: logger, optional. A logger returned by SampleObject._setup_custom_logger(). Default: None.

    Attributes:
    - ages: numpy array. The n_a ages of the grid [Myr].
    - masses: numpy array. The n_m masses of the grid [M_sun].
    - filters: numpy array. List of the n_f filters in the grid.
    - data: numpy array with shape (n_m,n_a,n_f). Isochrone grid.
    - model_version: string. Input model version.
    - model_grid: string. Model_grid corresponding to the unique set of parameters specified for the selected model_version.
    - file: string. Full path to the .h5 file containing the dataset.
    - n_steps: two-element list: [n_m,n_a].
    - mass_range: list or numpy array. See above.
    - age_range: list or numpy array. See above.
    - feh: float. See above.
    - he: float. See above.
    - afe: float. See above.
    - v_vcrit: float. See above.
    - fspot: float. See above.
    - B: int. See above.

    Built-in methods:

    1) __eq__
    Two IsochroneGrid instances are considered equal if their model_code are equal.

    2) __repr__
    Returns a string 's' corresponding to the user's input.
    It can be executed through eval(s).

    Methods (use help() to have more detailed info):

    1) plot_isochrones
    Defines a IsochroneGrid object and draws the selected CMD.

    2) plot_iso_grid
    Plots theoretical magnitudes in a given band for a certain model, or the magnitude difference between two models.

    """

    def __init__(self, model_version, filters, **kwargs):

        logger=kwargs['logger'] if 'logger' in kwargs else None
        self.model_version = str.lower(model_version)
        self.mass_range=kwargs['mass_range'] if 'mass_range' in kwargs else [0.01,1.4]
        self.age_range=kwargs['age_range'] if 'age_range' in kwargs else [1,1000]
        self.n_steps=kwargs['n_steps'] if 'n_steps' in kwargs else [1000,500]
        self.__fill_value=kwargs['fill_value'] if 'fill_value' in kwargs else np.nan
        search_model=kwargs['search_model'] if 'search_model' in kwargs else True

        model_params = {'feh':0.00, 'afe': 0.00, 'v_vcrit': 0.00, 'fspot': 0.00, 'B':0, 'he':0.27}
        for key in model_params.keys():
            if key in kwargs: model_params[key]=kwargs[key]

        if search_model:
            ModelHandler._find_model_grid(model_version,model_params)

        model_params = self._get_model_parameters(model_params)
        for key in model_params.keys(): self.__setattr__(key,model_params[key])

        self.model_grid = ModelHandler._find_model_grid(model_version,model_params)
        try:
            model_obj = ModelHandler(self.model_grid)
            masses, ages, v0, __, data0 = model_obj.get_data()
        except ValueError:
            raise ValueError('Model '+self.model_grid+' not found!')
            if logger!=None:
                logger.error('Model '+self.model_grid+' not found! Program ended.')
        self.file = model_obj.file

        try:
            len(filters)
        except TypeError: 
            filters = np.array([filters])
        self.filters = np.array(filters)
        self.filters = IsochroneGrid._fix_filters(self.filters, model_obj.file)
        fnew = self.filters
        nf = len(fnew)
        
        if isinstance(self.mass_range,list):
            n1 = self.n_steps[0]
            mnew = np.exp(np.log(0.999*self.mass_range[0])+(np.log(1.001*self.mass_range[1])-np.log(0.999*self.mass_range[0]))/(n1-1)*np.arange(n1))
        else:
            mnew = np.sort(self.mass_range)
            n1 = len(mnew)

        try: len(self.age_range)
        except TypeError:
            anew = np.array([self.age_range])
            n2 = 1
            case = 1
        else:
            if isinstance(self.age_range,list):
                n2 = self.n_steps[1]
                anew = np.exp(np.log(1.0001*self.age_range[0])+(np.log(0.9999*self.age_range[1])-np.log(1.0001*self.age_range[0]))/(n2-1)*np.arange(n2))
                case = 2
            elif isinstance(self.age_range, np.ndarray):
                if len(self.age_range.shape) == 1:
                    anew = np.array(self.age_range)
                    n2 = len(anew)
                    case = 3
                elif len(self.age_range[0]) == 3:
                    age0 = np.unique(self.age_range.ravel())
                    age1 = (age0[:-1]+(age0[1:]-age0[:-1])/4)
                    age2 = (age0[:-1]+(age0[1:]-age0[:-1])/2)
                    age3 = (age0[:-1]+3*(age0[1:]-age0[:-1])/4)
                    anew = np.sort(np.concatenate((age0,age1,age2,age3)))
                    n2 = len(anew)
                    case = 4
                elif len(self.age_range[0])==2:
                    n2 = self.n_steps[1]
                    anew = np.exp(np.log(1.0001*np.nanmin(self.age_range))+(np.log(0.9999*np.nanmax(self.age_range))-np.log(1.0001*np.nanmin(self.age_range)))/(n2-1)*np.arange(n2))
                    case = 6
            else: raise TypeError('Only scalar, list or numpy arrays are valid inputs for the keyword "age_range".')
        if self.model_grid == 'pm13':
            n2 = 1
            case = 5

        iso = np.full([n1,len(ages),nf], np.nan)
        iso_f = np.full(([n1,n2,nf]), np.nan)
        
        w_interp, = np.where((anew>0.95*ages[0]) & (anew<1.05*ages[-1]))
        
        if len(w_interp)>0:
            age_interp = anew[w_interp]
        else: 
            self.masses=mnew
            self.ages=anew
            self.data=iso_f
        
        if case == 5:
            for j in range(nf):
                w, = np.where(v0 == fnew[j])
                if len(w)>0:
                    k=0
                    gv = np.isfinite(data0[:,k,w]).ravel()
                    if np.sum(gv)>2:
                        m0=masses[gv]
                        f=interp1d(m0,data0[gv,k,w],kind='linear',fill_value=self.__fill_value,bounds_error=False)
                        iso_f[:,k,j]=f(mnew)
                else:
                    warnings.warn('Filter '+fnew[j]+' not available for model '+self.model_grid+'. Setting the corresponding row to nan.')
                    if logger!=None:
                        logger.info('Filter '+fnew[j]+' not available for model '+self.model_grid+'. Setting the corresponding row to nan.')
                    continue
            anew=ages
        elif case > 1:
            for j in range(nf):
                w, = np.where(v0 == fnew[j])
                if len(w) > 0:
                    for k in range(len(ages)):
                        gv = np.isfinite(data0[:,k,w]).ravel()
                        if np.sum(gv)>2:
                            m0 = masses[gv]
                            f = interp1d(m0, data0[gv,k,w],
                                         kind='linear',
                                         fill_value=self.__fill_value,
                                         bounds_error=False)
                            iso[:,k,j]=f(mnew)
                    for k in range(n1):
                        mask = np.isfinite(iso[k,:,j]).ravel()
                        if np.sum(mask)>2:
                            f = interp1d(ages[mask], iso[k,mask,j], 
                                         kind='linear', fill_value=np.nan,
                                         bounds_error=False)
                            iso_f[k,w_interp,j]=f(age_interp)
                else:
                    warnings.warn('Filter '+fnew[j]+' not available for model '+self.model_grid+'. Setting the corresponding row to nan.')
                    if logger != None:
                        logger.info('Filter '+fnew[j]+' not available for model '+self.model_grid+' Setting the corresponding row to nan.')
                    continue
        else:
            for j in range(nf):
                w, = np.where(v0 == fnew[j])
                if len(w)>0:
                    for k in range(len(ages)):
                        gv = np.isfinite(data0[:,k,w]).ravel()
                        m0=masses[gv]
                        if len(m0)>1:
                            f=interp1d(m0, data0[gv,k,w], kind='linear',
                                       fill_value=self.__fill_value,
                                       bounds_error=False)
                            iso[:,k,j]=f(mnew)
                    for k in range(n1):
                        mask = np.isfinite(iso[k,:,j]).ravel()
                        if np.sum(mask) > 2:
                            f = interp1d(ages[mask], iso[k,mask,j],
                                         kind='linear', fill_value=np.nan,
                                         bounds_error=False)
                            iso_f[k,w_interp,j]=f(age_interp)
                else:
                    warnings.warn('Filter '+fnew[j]+' not available for model '+self.model_grid+'. Setting the corresponding row to nan.')
                    if logger != None:
                        logger.info('Filter '+fnew[j]+' not available for model '+self.model_grid+'. Setting the corresponding row to nan.')
                    continue

        if hasattr(anew,'__len__') == False: 
            anew = np.array([anew])
        fnew = np.array(fnew)

        self.masses = mnew
        self.ages = anew
        self.data = iso_f

    def __eq__(self, other):
        return self.model_grid == other.model_grid

    def __repr__(self):
        s="IsochroneGrid('"+self.model_version+"', "+repr(list(self.filters))+", "
        kwargs=['mass_range','age_range','n_steps','_IsochroneGrid__fill_value','feh','B','afe','v_vcrit','fspot','he']
        for i in kwargs:
            if isinstance(self.__dict__[i],list):
                l=[str(j) for j in self.__dict__[i]]
                s+=i+'=['+','.join(l)+']'
            elif isinstance(self.__dict__[i],np.ndarray): s+=i+'=np.'+np.array_repr(self.__dict__[i])
            else: s+=i+'='+str(self.__dict__[i])
            s+=', '
        if s.endswith(', '): s=s[:-2]
        s+=')'
        s=s.replace('_IsochroneGrid__fill_value','fill_value')
        s=s.replace('=nan','=np.nan')

        return s

    def _get_model_parameters(self, model_params):

        model_p = ModelHandler._available_parameters(self.model_version)
        feh_range, he_range, afe_range, v_vcrit_range, fspot_range, B_range = model_p['feh'], model_p['he'], model_p['afe'], model_p['v_vcrit'], model_p['fspot'], model_p['B']

        keys = ['feh', 'afe', 'v_vcrit', 'fspot', 'he', 'B']
        res = {}
        for i, k in enumerate(keys):
            value_l = eval(k+'_range')
            arg = np.argmin(abs(value_l-model_params[k]))
            value = value_l[arg]
            res[k] = value

        return res

    @staticmethod
    def _fix_filters(filter_array, fname):

        filter_array = np.array(filter_array)

        with h5py.File(fname,"r") as f:
            fi = f.get('filters')[:]
            fi = np.array(fi,dtype='str')

        if 'Gbp2' in fi:
            filter_array = np.where(filter_array == 'G', 'G2', filter_array)
            filter_array = np.where(filter_array == 'Gbp', 'Gbp2', filter_array)
            filter_array = np.where(filter_array == 'Grp', 'Grp2', filter_array)
            filter_array = np.unique(filter_array)
        elif 'G' in fi:
            filter_array = np.where(filter_array == 'G2', 'G', filter_array)
            filter_array = np.where(filter_array == 'Gbp2', 'Gbp', filter_array)
            filter_array = np.where(filter_array == 'Grp2', 'Grp', filter_array)

        return np.unique(filter_array)    
    
    def _get_mass_range(data, model_version, dtype='mag', **kwargs):

        if 'secondary_contrast' in kwargs:
            contr = kwargs['secondary_contrast']
            contr_values = list(contr.values())[0]
            if hasattr(contr_values, '__len__') == False:
                contr_values = [contr_values]
            data2 = [data]
            for val in contr_values:
                if np.isfinite(val):
                    data2.append(data + val)
            data = np.concatenate(data2)

        if model_version == 'starevol':
            model_params = {'feh':-0.01, 'afe': 0.00, 'v_vcrit': 0.00, 'fspot': 0.00, 'B':0, 'he':0.24}
        else:
            model_params = {'feh':0.00, 'afe': 0.00, 'v_vcrit': 0.00, 'fspot': 0.00, 'B':0, 'he':0.24}

        for key in model_params.keys():
            if key in kwargs: model_params[key]=kwargs[key]

        model_grid = ModelHandler._version_to_grid(model_version,model_params)
        mod = ModelHandler(model_grid)
        mass_range = mod.mass_range

        if dtype=='mass':
            sample_r=[np.nanmin(data),np.nanmax(data)]
            m=IsochroneGrid._intersect_arr(mass_range,[sample_r[0],sample_r[1]])
            if isinstance(m,list): final_m = m
            else: final_m = m.tolist()
        else:
            filt = np.array(['G','K'])
            iso = IsochroneGrid(model_version, filt, n_steps=[100, 75],
                                mass_range = [0.001, 30], **kwargs)
            iso_data = iso.data
            iso_m = iso.masses
            iso_a = iso.ages
            sh = iso_data.shape[0:2]

            w, = np.where((np.isnan(data[:,0])==False) & (np.isnan(data[:,1])==False))
            if len(w)>0:
                i1, i2, i3, i4 = np.nanargmin(data[w,0]), np.nanargmax(data[w,0]), np.nanargmin(data[w,0]-data[w,1]), np.nanargmax(data[w,0]-data[w,1])
                i1, i2, i3, i4 = w[i1],w[i2],w[i3],w[i4]

                a1 = np.nanargmin((iso_data[:,:,0]-data[i1,0])**2+(iso_data[:,:,1]-data[i1,1])**2)
                im1, __ = np.unravel_index(a1,sh)
                a2 = np.nanargmin((iso_data[:,:,0]-data[i2,0])**2+(iso_data[:,:,1]-data[i2,1])**2)
                im2, __ = np.unravel_index(a2,sh)
                a3 = np.nanargmin((iso_data[:,:,0]-data[i3,0])**2+(iso_data[:,:,1]-data[i3,1])**2)
                im3, __ = np.unravel_index(a3,sh)
                a4 = np.nanargmin((iso_data[:,:,0]-data[i4,0])**2+(iso_data[:,:,1]-data[i4,1])**2)
                im4, __ = np.unravel_index(a4,sh)
                im = np.array([im1,im2,im3,im4])
                sample_r = [np.nanmin(iso_m[im]),np.nanmax(iso_m[im])]

                m = IsochroneGrid._intersect_arr(mass_range, [0.6*sample_r[0], 1.4*sample_r[1]])
            else: m = mass_range

            if isinstance(m, list): final_m = m
            else: final_m = m.tolist()

            if 'secondary_q' in kwargs:
                q = kwargs['secondary_q']
                if hasattr(q, '__len__') == False:
                    q = [q]
                min_q = np.nanmin(q)
                if min_q == 0: min_q = 1e-2
                final_m = [min_q*final_m[0], final_m[1]]

            final_m = IsochroneGrid._intersect_arr(mass_range, final_m)

        return final_m
        
    @staticmethod
    def _alter_isochrones_for_binarity(grid, q=None, delta_mag_dict=None):

        has_companion = (q is not None) | (delta_mag_dict is not None)
        if has_companion == False:
            return grid.data, None, None, None, None

        grid_mass, grid_age, grid_filt, grid_data = grid.masses, grid.ages, grid.filters, grid.data

        all_iso_data, all_mass_index_B, all_use_B, all_q_eff = [], [], [], []

        if q is not None:

            if hasattr(q, '__len__') == False:
                q = [q]

            unique_q = np.unique(q)
            index_of_iso = where_v(q, unique_q)
            no_iso = len(unique_q)

            for q_i in unique_q:

                if q_i > 0:
                    new_grid_data = copy.deepcopy(grid_data)

                    i_q = np.nanargmin(np.abs(grid_mass[0]/grid_mass-q_i))
                    q_eff = grid_mass[0]/grid_mass[i_q]
                    q_cond = np.abs((q_eff-q_i)/q_i < 0.1) & np.abs((q_eff-q_i) < 0.05)
                    if q_cond:
                        if i_q > 0:
                            new_grid_data[i_q:,:,:] = add_mag(grid_data[i_q:,:,:], grid_data[:-i_q,:,:])
                        else:
                            new_grid_data[i_q:,:,:] = add_mag(grid_data, grid_data)

                    mass_index_B = where_v(grid_mass*q_i, grid_mass, approx=True)
                    use_B = np.abs(grid_mass[mass_index_B]-grid_mass*q_i)/grid_mass < 0.05

                    all_iso_data.append(new_grid_data)
                    all_mass_index_B.append(mass_index_B)
                    all_use_B.append(use_B)
                    all_q_eff.append(q_eff)

                else:
                    all_iso_data.append(grid_data)
                    all_mass_index_B.append(None)
                    all_use_B.append(None)
                    all_q_eff.append(0.)

        elif delta_mag_dict is not None:

            if isinstance(delta_mag_dict, dict) == False:
                raise TypeError('Argument "secondary_contrast" must be a dictionary.')
            contrast_filter_used = list(delta_mag_dict.keys())
            if len(contrast_filter_used) > 1:
                raise ValueError('Only one contrast must be provided in "secondary_contrast".')
            contrast_filter_used = contrast_filter_used[0]

            delta_mag = delta_mag_dict[contrast_filter_used]
            w_G, = np.where(grid_filt == contrast_filter_used)
            if len(w_G) == 0:
                raise ValueError('Filter {0} not recognized in argument "secondary_contrast". Make sure to use one of the following filters: {1}.'.format(contrast_filter_used, iso_filt))

            if hasattr(delta_mag, '__len__') == False:
                delta_mag = [delta_mag]

            unique_delta_mag = np.unique(delta_mag)
            index_of_iso = where_v(delta_mag, unique_delta_mag)
            no_iso = len(unique_delta_mag)

            for delta_mag_i in unique_delta_mag:

                if np.isfinite(delta_mag_i):

                    new_grid_data = copy.deepcopy(grid_data)

                    corr_factor = -2.5*np.log10((1+10**(-0.4*delta_mag_i)))

                    mass_index_B, use_B = np.zeros([len(grid_mass),len(grid_age)], dtype = int), np.zeros([len(grid_mass),len(grid_age)], dtype = bool)
                    for j in range(len(grid_age)):
                        i_G = where_v((grid_data[:,j,w_G]+delta_mag_i).ravel(), grid_data[:,j,w_G].ravel(), approx=True, assume_sorted=False)
                        mask = np.abs(grid_data[:,j,w_G].ravel()+delta_mag_i-grid_data[i_G,j,w_G].ravel()) < 0.05
                        mass_index_B[:,j] = i_G
                        use_B[:,j] = mask

                    grid_data2 = np.zeros_like(grid_data)
                    for k in range(len(grid_filt)):
                        for j in range(len(grid_age)):
                            grid_data2[:,j,k] = np.where(use_B[:,j], grid_data[mass_index_B[:,j],j,k], np.inf)
                        if k==w_G:
                            new_grid_data[:,:,w_G] += corr_factor
                        else:
                            new_grid_data[:,:,k] = add_mag(grid_data[:,:,k], grid_data2[:,:,k])

                    all_iso_data.append(new_grid_data)
                    all_mass_index_B.append(mass_index_B)
                    all_use_B.append(use_B)
                    all_q_eff.append(None)

                else:
                    all_iso_data.append(grid_data)
                    all_mass_index_B.append(None)
                    all_use_B.append(None)
                    all_q_eff.append(None)


        if no_iso == 1:
            all_iso_data, all_mass_index_B, all_use_B = all_iso_data[0], all_mass_index_B[0], all_use_B[0]
            all_q_eff, index_of_iso = all_q_eff[0], None
        else:
            all_q_eff = np.array(all_q_eff)

        return all_iso_data, all_mass_index_B, all_use_B, all_q_eff, index_of_iso
        

    ############################################# plotting functions #########################################

    @classmethod
    def plot_isochrones(cls, col, mag, model_version, ax, **kwargs):

        """
        A class method. Defines a IsochroneGrid object and draws the selected CMD.
        Similar to SampleObject.CMD, but draws only the isochrones over an existing figure.
            Input:
            - col: string, required. Quantity to be plotted along the x axis (e.g.: 'G' or 'G-K')
            - mag: string, required. Quantity to be plotted along the y axis (e.g.: 'G' or 'G-K')
            - model_version: string, required. Selected model_version. Use ModelHandler.available() for further information.
            - ax: AxesSubplot, required. Axis object where the isochrones will be drawn upon.
            - plot_ages: numpy array or bool, optional. It can be either:
                    - a numpy array containing the ages (in Myr) of the isochrones to be plotted;
                    - False, not to plot any isochrone.
              Default: [1,3,5,10,20,30,100,200,500,1000].
            - plot_masses: numpy array or bool, optional. It can be either:
                    - a numpy array containing the masses (in M_sun) of the tracks to be plotted.
                    - False, not to plot any track.
              Default: [0.1,0.3,0.5,0.7,0.85,1.0,1.3,2].
            - all valid keywords for IsochroneGrid().
        """

        filters = []

        if '-' in col:
            col_n = col.split('-')
            filters.extend(col_n)
        else:
            filters.append(col)
        if '-' in mag:
            mag_n = mag.split('-')
            filters.extend(mag_n)
        else:
            filters.append(mag)

        filters = np.array(filters)

        if 'mass_range' in kwargs: 
            mass_r = IsochroneGrid._get_mass_range(kwargs['mass_range'], 
                                                   model_version, dtype='mass',
                                                   **kwargs)
        else: mass_r = IsochroneGrid._get_mass_range([1e-6,1e+6],
                                                     model_version,
                                                     dtype='mass', **kwargs)
        kwargs['mass_range'] = mass_r

        if 'plot_ages' in kwargs:
            plot_ages = kwargs['plot_ages']
            if isinstance(plot_ages, bool) == False:
                plot_ages = np.array(plot_ages) 
        else: plot_ages = np.array([1,3,5,10,20,30,100,200,500,1000])
        
        if 'plot_masses' in kwargs:
            plot_masses = kwargs['plot_masses']
            if isinstance(plot_masses, bool) == False:
                plot_masses = np.array(plot_masses) 
        else: plot_masses = np.array([0.1,0.3,0.5,0.7,0.85,1.0,1.3,2])

        x_axis=col
        y_axis=mag

        kwargs['age_range']=plot_ages
        iso=IsochroneGrid(model_version,filters,**kwargs)
        isochrones=iso.data
        iso_ages=iso.ages
        iso_filters=iso.filters
        iso_masses=iso.masses
        
        n_ages = len(iso_ages)

        #changes names of Gaia_DR2 filters to DR3
        if 'G2' in iso_filters:
            w=where_v(['G2'],iso_filters)
            iso_filters[w]=['G']
        if 'Gbp2' in iso_filters:
            w=where_v(['Gbp2'],iso_filters)
            iso_filters[w]=['Gbp']
        if 'Grp2' in iso_filters:
            w=where_v(['Grp2'],iso_filters)
            iso_filters[w]=['Grp']

        #finds color/magnitude isochrones to plot
        if '-' in x_axis:
            col_n = x_axis.split('-')
            w1, = np.where(iso_filters == col_n[0])
            w2, = np.where(iso_filters == col_n[1])
            col_th = isochrones[:,:,w1]-isochrones[:,:,w2]
        else:
            w1, = np.where(iso_filters == x_axis)
            col_th = isochrones[:,:,w1]
        if '-' in y_axis:
            mag_n = y_axis.split('-')
            w1, = np.where(iso_filters == mag_n[0])
            w2, = np.where(iso_filters == mag_n[1])
            mag_th = isochrones[:,:,w1]-isochrones[:,:,w2]
        else:
            w1, = np.where(iso_filters == y_axis)
            mag_th = isochrones[:,:,w1]

        if type(plot_masses) != bool:
            kwargs['mass_range'] = plot_masses
            kwargs2 = copy.deepcopy(kwargs)
            if 'age_range' in kwargs2: 
                del kwargs2['age_range']
            trk = IsochroneGrid(model_version, filters, 
                                age_range = [np.min(plot_ages), np.max(plot_ages)],
                                **kwargs2)
            tracks = trk.data
            trk_ages = trk.ages
            trk_filters = trk.filters
            trk_masses = trk.masses
            if 'G2' in trk_filters:
                w = where_v(['G2'],trk_filters)
                trk_filters[w] = ['G']
            if 'Gbp2' in trk_filters:
                w = where_v(['Gbp2'],trk_filters)
                trk_filters[w] = ['Gbp']
            if 'Grp2' in trk_filters:
                w = where_v(['Grp2'],trk_filters)
                trk_filters[w] = ['Grp']
            if '-' in x_axis:
                w1, = np.where(trk_filters == col_n[0])
                w2, = np.where(trk_filters == col_n[1])
                col_th_t = tracks[:,:,w1] - tracks[:,:,w2]
            else:
                w1, = np.where(trk_filters == x_axis)
                col_th_t = tracks[:,:,w1]
            if '-' in y_axis:
                w1, = np.where(trk_filters == mag_n[0])
                w2, = np.where(trk_filters == mag_n[1])
                mag_th_t = tracks[:,:,w1] - tracks[:,:,w2]
            else:
                w1, = np.where(trk_filters == y_axis)
                mag_th_t = tracks[:,:,w1]

        n = len(isochrones)
        tot_iso = len(isochrones[0])
        nis = len(plot_ages)

        if type(plot_ages) != bool:
            if n_ages > 1:
                for i in range(len(plot_ages)):
                    ii = closest(iso_ages, plot_ages[i])
                    ax.plot(col_th[:,ii], mag_th[:,ii], label=str(plot_ages[i])+' Myr')
            else:
                ax.plot(col_th[:,0], mag_th[:,0], label='model')

        if type(plot_masses) != bool:
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(len(plot_masses)):
                    im = closest(trk_masses, plot_masses[i])
                    ax.plot(col_th_t[im,:], mag_th_t[im,:], linestyle='dashed', color='gray')
                    c = 0
                    while (np.isfinite(col_th_t[im,c]) == 0) | (np.isfinite(mag_th_t[im,c]) == 0):
                        c += 1
                        if c == len(col_th_t[im,:]): 
                            break
                    if c < len(col_th_t[im,:]):
                        an = ax.annotate(str(plot_masses[i]), (col_th_t[im,c],mag_th_t[im,c]), size='large')
                        an.set_in_layout(False)
        return None

    @classmethod
    def plot_iso_grid(cls, col, model_version, reverse_xaxis=False, reverse_yaxis=False, tofile=None, **kwargs):

        """
        A class method. Defines a IsochroneGrid object.
        Plots theoretical magnitudes in a given band for a certain model, or the magnitude difference between two models.
        The returned magnitudes are shown as f(age, mass), i.e. as a color map in the (age, mass) grid.
            Input:
            - col: string, required. Quantity to be plotted in a color scale as along the x axis (e.g.: 'G' or 'G-K').
            - model_version: string or 2-element list, required. If a string, it shows the data for the model_grid uniquely identified from the model_version + customizable parameters.
              If a list, it shows the difference between model[0] and model[1].
            - reverse_xaxis: bool, optional. Reverses the x axis. Default: False.
            - reverse_yaxis: bool, optional. Reverses the y axis. Default: False.
            - x_log: bool, optional. Sets the mass axis scale as logarithmic. Default: True.
            - y_log: bool, optional. Sets the age axis scale as logarithmic. Default: True.
            - levels: list, optional. Contour levels to be overplotted on the map. Default: not set.
            - fontsize: string, optional. Size of ticks, labels of axes, contours, etc. Default: 15.
            - cmap: string, optional. Color map of f(age,mass). Default: 'viridis_r'.
            - tofile: string, optional. Full path to the output .png file. Default: False (no file is saved).
            - show_plot: bool, optional. Whether to show the plot on screen or not (useful when saving the output to a file). Default: True.

            - all valid keywords for IsochroneGrid().
        """

        model_params = {}
        for i in ['feh','afe','v_vcrit','he','fspot','B']:
            if i in kwargs: model_params[i] = kwargs[i]

        x_log = kwargs['x_log'] if 'x_log' in kwargs else True
        y_log = kwargs['y_log'] if 'y_log' in kwargs else True
        f = list(stored_data['filters'].keys())
        filt = []

        if isinstance(model_version, list):
            for mod in model_version:
                ModelHandler._find_model_grid(mod, model_params)
                if mod == 'atmo2020': 
                    raise ValueError("Please use one among the following: 'atmo2020-ceq', 'atmo2020-neq-s', 'atmo2020-neq-w'. ")
                elif mod == 'sb12': 
                    raise ValueError("Please use one among the following: 'sb12-hy-cold','sb12-hy-hot','sb12-cf-cold','sb12-cf-hot'. ")
            if '-' in col:
                filt.extend(col.split('-'))
                filt = np.array(filt)

                for ff in filt:
                    if ff not in f: 
                        raise ValueError('Filter '+ff+' does not exist. Use info_filters() to know the available filters.')

                iso = IsochroneGrid(model_version[0], filt, **kwargs)
                iso_mass, iso_age, iso_filt1, iso_data1 = iso.masses, iso.ages, iso.filters, iso.data
                if 'G2' in iso_filt1:
                    w = where_v(['G2'],iso_filt1)
                    iso_filt1[w] = ['G']
                if 'Gbp2' in iso_filt1:
                    w = where_v(['Gbp2'],iso_filt1)
                    iso_filt1[w] = ['Gbp']
                if 'Grp2' in iso_filt1:
                    w = where_v(['Grp2'],iso_filt1)
                    iso_filt1[w] = ['Grp']
                w1_1, = np.where(iso_filt1 == filt[0])
                w2_1, = np.where(iso_filt1 == filt[1])
                l = iso_data1.shape
                iso2 = IsochroneGrid(model_version[1], filt, **kwargs)
                iso_mass2, iso_age2, iso_filt2, iso_data2 = iso2.masses, iso2.ages, iso2.filters, iso2.data
                if 'G2' in iso_filt2:
                    w = where_v(['G2'], iso_filt2)
                    iso_filt2[w] = ['G']
                if 'Gbp2' in iso_filt2:
                    w = where_v(['Gbp2'], iso_filt2)
                    iso_filt2[w] = ['Gbp']
                if 'Grp2' in iso_filt2:
                    w = where_v(['Grp2'], iso_filt2)
                    iso_filt2[w] = ['Grp']
                w1_2, = np.where(iso_filt2 == filt[0])
                w2_2, = np.where(iso_filt2 == filt[1])
                data1 = (iso_data1[:,:,w1_1]-iso_data1[:,:,w2_1]).reshape([l[0],l[1]])
                data2 = (iso_data2[:,:,w1_2]-iso_data2[:,:,w2_2]).reshape([l[0],l[1]])
                data = data1 - data2
            else:
                if col not in f: 
                    raise ValueError('Filter '+col+' does not exist. Use info_filters() to know the available filters.')
                try:
                    len(col)
                    filt = np.array(col)
                except TypeError: 
                    filt = np.array([col])
                
                iso = IsochroneGrid(model_version[0], filt, **kwargs)
                iso_mass, iso_age, iso_filt1, data1 = iso.masses, iso.ages, iso.filters, iso.data
                iso2 = IsochroneGrid(model_version[1],filt,**kwargs)
                iso_mass2, iso_age2, iso_filt2, data2 = iso2.masses, iso2.ages, iso2.filters, iso2.data
                l = data1.shape
                data1 = data1.reshape([l[0], l[1]])
                data2 = data2.reshape([l[0], l[1]])
                data = data1 - data2
        else:
            ModelHandler._find_model_grid(model_version, model_params)
            
            if model_version=='atmo2020': 
                raise ValueError("Please use one among the following: 'atmo2020-ceq', 'atmo2020-neq-s', 'atmo2020-neq-w'. ")
            elif model_version=='sb12': 
                raise ValueError("Please use one among the following: 'sb12-hy-cold','sb12-hy-hot','sb12-cf-cold','sb12-cf-hot'. ")
                
            if '-' in col:
                filt.extend(col.split('-'))
                filt = np.array(filt)
                
                if 'mass_range' not in kwargs:
                    kwargs['mass_range'] = IsochroneGrid._get_mass_range([0.001,1.4], model_version, 
                                                                         dtype='mass', **model_params)
                
                iso = IsochroneGrid(model_version, filt, **kwargs)
                iso_mass, iso_age, iso_filt, iso_data = iso.masses, iso.ages, iso.filters, iso.data
                if 'G2' in iso_filt:
                    w = where_v(['G2'],iso_filt)
                    iso_filt[w] = ['G']
                if 'Gbp2' in iso_filt:
                    w = where_v(['Gbp2'],iso_filt)
                    iso_filt[w] = ['Gbp']
                if 'Grp2' in iso_filt:
                    w = where_v(['Grp2'],iso_filt)
                    iso_filt[w] = ['Grp']
                w1, = np.where(iso_filt == filt[0])
                w2, = np.where(iso_filt == filt[1])
                l = iso_data.shape
                data = (iso_data[:,:,w1]-iso_data[:,:,w2]).reshape([l[0],l[1]])
            else:
                if col not in f: 
                    raise ValueError('Filter '+col+' does not exist. Use info_filters() to know the available filters.')
                try:
                    len(col)
                    filt = np.array(col)
                except TypeError: 
                    filt = np.array([col])
                    
                if 'mass_range' not in kwargs:
                    kwargs['mass_range'] = IsochroneGrid._get_mass_range([0.001,1.4], model_version, 
                                                                         dtype='mass', **model_params)
                    
                iso = IsochroneGrid(model_version, filt, **kwargs)
                iso_mass, iso_age, iso_filt, data = iso.masses, iso.ages, iso.filters, iso.data
                l = data.shape
                data = data.reshape([l[0], l[1]])

        im_min, im_max = 0, l[0] - 1
        if np.isnan(nansumwrapper(data[0,:])) == False:
            im_min = 0
            if np.isnan(nansumwrapper(data[-1,:])):
                i = l[0] - 1
                while 1:
                    s = nansumwrapper(data[i,:])
                    if np.isnan(s) == False: 
                        break
                    i -= 1
                im_max = i
            else: 
                im_max = l[0]
        elif np.isnan(nansumwrapper(data[l[0]-1,:])) == False:
            im_max = l[0] - 1
            if np.isnan(nansumwrapper(data[0,:])):
                i = 0
                while 1:
                    s = nansumwrapper(data[i,:])
                    if np.isnan(s) == False: 
                        break
                    i += 1
                im_min = i
            else: 
                im_min = 0
        ia_min = 0
        ia_max = l[1] - 1
        if np.isnan(nansumwrapper(data[im_min:im_max,0])) == False:
            ia_min = 0
            if np.isnan(nansumwrapper(data[im_min:im_max,-1])):
                i = l[1] - 1
                while 1:
                    s = nansumwrapper(data[im_min:im_max,i])
                    if np.isnan(s) == False:
                        break
                    i -= 1
                ia_max = i
            else: 
                ia_max = l[0]
        elif np.isnan(nansumwrapper(data[im_min:im_max,l[1]-1])) == False:
            ia_max = l[0] - 1
            if np.isnan(nansumwrapper(data[im_min:im_max,0])):
                i = 0
                while 1:
                    s = nansumwrapper(data[im_min:im_max,i])
                    if np.isnan(s) == False:
                        break
                    i += 1
                ia_min = i
            else: ia_min = 0

        fontsize = kwargs['fontsize'] if 'fontsize' in kwargs else 15
        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'viridis_r'

        fig, ax = plt.subplots(figsize=(12,12))
        if len(iso_age[ia_min:ia_max])>1:
            ax.contour(iso_mass[im_min:im_max], iso_age[ia_min:ia_max], 
                       (data[im_min:im_max,ia_min:ia_max]).T, 100, cmap=cmap)
            CS = ax.contourf(iso_mass[im_min:im_max], iso_age[ia_min:ia_max], 
                             (data[im_min:im_max,ia_min:ia_max]).T, 100, cmap=cmap)
        else:
            CS = ax.imshow(data[im_min:im_max,ia_min:ia_max].T, cmap='autumn_r',
                           extent=[iso_mass[im_min],iso_mass[im_max-1],999,1001], aspect='auto')
            ax.spines['left'].set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(CS, cax=cax)

        ax.set_xlabel(r'mass [$M_\odot$]',fontsize=fontsize)
        ax.set_ylabel('age [Myr]',fontsize=fontsize)
        cbar.ax.set_ylabel(col,fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        if reverse_xaxis: ax.invert_xaxis()
        if reverse_yaxis: ax.invert_yaxis()
        if x_log:
            ax.set_xscale('log')
        if y_log:
            ax.set_yscale('log')
        if 'levels' in kwargs:
            CS2 = ax.contour(CS,levels=kwargs['levels'], colors='white')
            ax.clabel(CS2, CS2.levels, fmt='%1.2f', fontsize=fontsize)

        if tofile is not None: 
            plt.savefig(tofile)
            
        show_plot = kwargs['show_plot'] if 'show_plot' in kwargs else True
        if show_plot: 
            plt.show()
        else: 
            plt.close()

    @staticmethod
    def _intersect_arr(x, y):
        if np.max(x)<np.min(y): 
            return np.array([])
        else: 
            return [np.max([np.min(x),np.min(y)]), np.min([np.max(x),np.max(y)])]


class SampleObject(object):

    """
    Class: madys.SampleObject

    Class that creates, handles and obtains physical parameters for lists of n young stellar and substellar objects.
    Check the documentation for additional details on general functioning, customizable settings and various examples.

    An instance can be initialized in two modes, differing in the shape of input data:
     (mode 1) uses just a list of targets;
     (mode 2) uses a Table containing both the target list and photometric data.
    Parameters that are only used in mode 1 are labeled with (1), and similarly for mode 2. Parameters common to both modes are not labeled.
    Starting from v1.1.0, an additional init mode exists, mediated by the function import_from_file(), that recovers data from a previous execution.

    Input:
    - file (1): string or list, required. It can be either:
        - a string, giving the full path to the file containing target names;
        - a list of IDs. Gaia IDs must begin by 'Gaia DR2 ' or 'Gaia DR3'.
    - file (2): astropy Table, required. Table containing target names and photometric data. See documentation for examples of valid inputs.
    - ext_map: string, required. Extinction map used. Select one among 'leike', 'stilism' and None. Not required (set to None) if ebv is provided.
    - mock_file: string, optional. Only used if file is a list or a table. Full path to a fictitious file, used to extract the working path and to name the outputs after it. If not set and verbose>=1, verbose changes to 0.
    - surveys (1): list, optional. List of surveys where to extract photometric data from. Default: ['gaia','2mass'].
    - id_type (1): string, required. Type of IDs provided: must be one among 'DR2','DR3','HIP' or 'other'.
    - ebv: float or numpy array, optional. If set, uses the i-th element of the array as E(B-V) for the i-th star. Default: not set, computes E(B-V) through the map instead.
    - ebv_err: float or numpy array, optional. Error on ebv, it should have its same type. If not set or None, no error is assumed. Default: not set.
    - max_tmass_q (1): worst 2MASS photometric flag ('ph_qual') still considered reliable. Possible values, ordered by decreasing quality: 'A','B','C','D','E','F','U','X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.
    - max_wise_q (1): worst ALLWISE photometric flag ('ph_qual2') still considered reliable. Possible values, ordered by decreasing quality: 'A','B','C','U','Z','X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.
    - verbose: int, optional. Degree of verbosity of the various tasks performed by MADYS. It can be:
        - 0: no file is saved, nothing is printed on the screen;
        - 1: a .csv file with retrieved information is saved (1), few info are printed on the screen;
        - 2: in addition to the output of 1, a log file is created;
        - 3: in addition to the output of 2, .csv files are created when executing SampleObject.get_params().
      Default: 2. However, if file is a list or a table and mock_file is not set, it is forcingly set to 0.

    Attributes:
    - file: string. Corresponding to either 'file' (1) or 'mock_file' (2).
    - verbose: int. Corresponding to input 'verbose'.
    - path: string. Working path, where all inputs and outputs are present.
    - log_file: string. Name of the log_file. Open it for details on the process outcome.
    - phot_table: astropy Table. It contains all input (1) or retrieved (2) data.
    - good_phot (1): astropy Table. Same as 'phot_table', but with discarded photometry being masked.
    - quality_table (1): astropy Table. It contains all information (e.g., photometric flags) needed to assess the quality and the acceptance/rejection of photometric measurements.
    - abs_phot: numpy array. Absolute magnitudes in the required filters.
    - abs_phot_err: numpy array. Errors on absolute magnitudes in the required filters.
    - abs_phot: numpy array. Apparent magnitudes in the required filters.
    - abs_phot_err: numpy array. Errors on apparent magnitudes in the required filters.
    - par: numpy array. Parallaxes of the objects.
    - par_err: numpy array. Errors on parallaxes.
    - ebv: float or numpy array. Corresponding to input 'ebv' or to computed E(B-V) reddening.
    - ebv_err: float or numpy array. Corresponding to input 'ebv_err'.
    - filters: list. Set of filters, given either by the filters of 'surveys' (1) or by column names (2).
    - surveys: list. Surveys used to extract photometric data.
    - mode: int. The execution mode.
    - ID: astropy Table. Original set of IDs.
    - GaiaID: astropy Table. Gaia IDs (original or recovered). If original, they can come from DR3 or DR2. If recovered, they always come from DR3.
    - log_file: Path object. Full path to the log file. It only exists if verbose>=2.

    Built-in methods:

    1) __getitem__
    SampleObject instances can be indexed like numpy arrays.
    Have a look at the documentation for additional details.

    2) __len__
    The len of a SampleObject instance is equal to the number of objects in the original list.

    3) __repr__
    Returns a string 's' corresponding to the user's input.
    It can be executed through eval(s).

    4) __str__
    Returns a verbose representation of the calling sequence.

    4) __eq__
    Two SampleObject instances are considered equal if the queried objects (as specified by the attribute ID) are the same.
    Two instances with the same object names but a different order will yield False in the comparison. This is done to avoid
    possible misinterpretations of the results based on indexing.
    
    Methods (use help() to have more detailed info):

    1) get_params
    Estimates age, mass, radius, Teff and logg for each object in the sample by comparison with isochrone grids.

    2) CMD
    Draws a color-magnitude diagram (CMD) containing both the measured photometry and a set of theoretical isochrones.

    3) plot_photometry
    Similar to CMD, but draws only photometric data over an existing figure.

    4) interstellar_ext
    Computes the reddening/extinction in a custom band, given the position of a star.

    5) extinction
    Converts one or more B-V color excess(es) into absorption(s) in the required photometric band.

    6) app_to_abs_mag
    Turns one or more apparent magnitude(s) into absolute magnitude(s).

    7) plot_2D_ext
    Plots the integrated absorption in a given region of the sky, by creating a 2D projection at constant distance of an extinction map.

    8) ang_dist
    Computes the angular distance between two sky coordinates or two equal-sized arrays of positions.
    
    9) export_to_file
    Exports the instance to a .h5 file. Complementary to import_from_file(), which performs the opposite operation.

    10) import_from_file
    Alternative initializer for the class. It imports the instance from a .h5 file. Complementary to export_to_file(), which performs the opposite operation. 

    11) merge
    Merges two or more SampleObject instances into a single one. 
    """

    def __init__(self, file, **kwargs):
        
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else 2
        
        if self.verbose>0:
            if (isinstance(file,Table)) | (isinstance(file,list)):
                try:
                    self.file = kwargs['mock_file']
                except KeyError:
                    self.verbose=0
                    self.file=''
            else: self.file = file
            if self.verbose>0:
                self.path = os.path.dirname(self.file)
                self._sample_name_ext()
        else:
            if (isinstance(file,Table)) | (isinstance(file,list)): self.file = ''
            else:
                self.file = file
                self.path = os.path.dirname(self.file)
                self._sample_name_ext()
        
        if isinstance(self.file, list):
            self.file = [str(i).replace(u'\xa0', u' ') for i in self.file]
        if isinstance(file, list):
            file = [str(i).replace(u'\xa0', u' ') for i in file]
            
        stop_init = kwargs['stop_init'] if 'stop_init' in kwargs else False
        if stop_init:
            if self.verbose>1:
                logging.shutdown()
                self.log_file = os.path.join(self.path,self.__sample_name+'_log.txt')
                if os.path.exists(self.log_file): os.remove(self.log_file)
                self.__logger = SampleObject._setup_custom_logger('madys',self.log_file)
                
            else: self.__logger = None
            
            return
        
        
        if 'ext_map' not in kwargs:
            if 'ebv' in kwargs:
                ext_map = None
            else:
                print("The keyword 'ext_map' must be set to 'leike', 'stilism' or None. ")
                while 1:
                    ext_map = input("Please insert a value among 'leike', 'stilism' or 'None' (without quotation marks):\n")
                    if ext_map not in ['None','leike','stilism']:
                        print("Invalid choice. Please insert a value among 'leike', 'stilism' or 'None' (without quotation marks):\n")
                    else: break
        else: ext_map = kwargs['ext_map']

        if ext_map == 'None': ext_map = None
        if ext_map not in [None,'leike','stilism']: raise ValueError("'ext_map' must be set to 'leike', 'stilism' or None.")
        SampleObject._download_ext_map(ext_map)
            
        self.__input = copy.deepcopy(kwargs)
        self.__input['ext_map'] = ext_map
        self.__input['file'] = file
        self.__madys_version = MADYS_VERSION


        if isinstance(file,Table):
            self.mode=2
            self.phot_table = file
            col0 = file.colnames
            kin = np.array(['parallax','parallax_err','ra','dec','id','ID','source_id','object_name'])
            col = np.setdiff1d(np.unique(np.char.replace(col0,'_err','')),kin)
            col_err = np.array([i+'_err' for i in col])
            self.filters = np.array(col)
            self.GaiaID = copy.deepcopy(file)
            if 'id' in self.GaiaID.columns: 
                self.GaiaID.keep_columns('id')
            elif 'ID' in self.GaiaID.columns: 
                self.GaiaID.keep_columns('ID')
            elif 'source_id' in self.GaiaID.columns: 
                self.GaiaID.keep_columns('source_id')
            elif 'object_name' in self.GaiaID.columns: 
                self.GaiaID.keep_columns('object_name')
            else: raise ValueError('The columns with IDs was not found! Check that a "ID"/"id"/"source_id"/"object_name" column is present, and try again.')
            self.ID = self.GaiaID
            n = len(col)
            nst = len(file)
        else:
            self.mode = 1
            self.surveys = kwargs['surveys'] if 'surveys' in kwargs else ['gaia','2mass']
            if 'gaia' not in self.surveys: self.surveys.append('gaia')
            filters = []
            available_surveys = list(np.unique([stored_data['filters'][k]['survey'] for k in stored_data['filters'].keys()]))
            for i in range(len(self.surveys)):
                if self.surveys[i] not in available_surveys:
                    raise ValueError("Survey "+self.surveys[i]+' not available. Available surveys: '+','.join(available_surveys)+'.')
                else:
                    w = [stored_data['filters'][k]['survey'] == self.surveys[i] for k in stored_data['filters'].keys()]
                    filters.extend(np.array(list(stored_data['filters'].keys()))[w])
            self.filters = np.array(filters)
            if 'id_type' in kwargs: 
                id_type = kwargs['id_type']
            else: 
                raise ValueError("'id_type' not provided! Select one among: 'DR2', 'DR3', 'HIP', 'other'.")
            self.__id_type = id_type
            if isinstance(id_type, str) == False: 
                raise ValueError("'id_type' must be a string.")
            elif id_type not in ['DR2','DR3','HIP','other']:
                raise ValueError("Invalid value for 'id_type'. Select one among: 'DR2', 'DR3', 'HIP', 'other'.")

            if isinstance(file,list): self.ID=Table({'ID':file})
            else: self.ID=self._read_IDs()
            if self.__id_type in ['DR2', 'DR3']: self.GaiaID = self.ID
            else: self._get_survey_names('Gaia DR3')
                
            nst = len(self.ID)

        if self.verbose>1:
            logging.shutdown()
            self.log_file = os.path.join(self.path,self.__sample_name+'_log.txt')
            if os.path.exists(self.log_file): os.remove(self.log_file)
            self.__logger = SampleObject._setup_custom_logger('madys',self.log_file)
        else: self.__logger = None

        if isinstance(file, Table):
            self.abs_phot = np.full([nst,n], np.nan)
            self.abs_phot_err = np.full([nst,n], np.nan)
            for i in range(n):
                self.abs_phot[:,i] = file[col[i]]
                self.abs_phot_err[:,i] = file[col_err[i]]

            self._print_log('info',logo)
            self._print_log('info',f'Program started. Running madys {MADYS_VERSION}.')
            self._print_log('info','Input type: custom table.')
            self._print_log('info',r'Filters required: {0}.'.format(', '.join(self.filters)))
            self._print_log('info',f'No. of stars: {nst}.')

            self.ebv_err = kwargs['ebv_err'] if 'ebv_err' in kwargs else None
            self.ebv = np.zeros(len(file))
            if 'ebv' in kwargs:
                self.ebv = kwargs['ebv']
                self._print_log('info','Extinction type: provided by the user.')
            elif ('ra' in col0) & ('dec' in col0) & ('parallax' in col0):
                self.par = file['parallax']
                self.ebv = SampleObject.interstellar_ext(ra=file['ra'], dec=file['dec'], par=self.par,
                                                         ext_map=ext_map, logger=self.__logger)
                if ext_map is None:
                    self._print_log('info',"Extinction neglected, because 'ext_map' was set to None.")
                else: self._print_log('info',f'Extinction type: computed using {ext_map} extinction map.')
            elif ('l' in col0) & ('b' in col0) & ('parallax' in col0):
                self.par = file['parallax']
                self.ebv = SampleObject.interstellar_ext(l=file['l'], b=file['b'], par=self.par,
                                                         ext_map=ext_map, logger=self.__logger)
                if ext_map is None:
                    self._print_log('info',"Extinction neglected, because 'ext_map' was set to None.")
                else: self._print_log('info',f'Extinction type: computed using {ext_map} extinction map.')
            if 'parallax' in col0:
                self.par = file['parallax']
                self.par_err = file['parallax_err']
                self.app_phot = copy.deepcopy(self.abs_phot)
                self.app_phot_err = copy.deepcopy(self.abs_phot_err)
                self.abs_phot, self.abs_phot_err = SampleObject.app_to_abs_mag(self.app_phot, self.par,
                                                                               app_mag_error=self.app_phot_err,
                                                                               parallax_error=self.par_err,
                                                                               ebv=self.ebv,ebv_error=self.ebv_err,
                                                                               filters=col)
                self._print_log('info','Input photometry: apparent, converted to absolute.')
            else:
                self._print_log('info','Input photometry: no parallax provided, assumed absolute.')

        else:
            self._print_log('info',logo)
            self._print_log('info',f'Program started. Running madys {MADYS_VERSION}.')
            self._print_log('info','Input file: list of IDs.')
            self._print_log('info',f'No. of stars: {nst}.')
            self._print_log('info',r'Looking for photometry in the surveys: {0}.'.format(', '.join(['gaia','2mass'])))

            self._print_log('info','Starting data query...')
            
            n_it, n_it_max = 0, 10
            search_done = False
            while (search_done == False) & (n_it < n_it_max):
                try:
                    self._get_phot()
                    search_done = True
                except ConnectionError:
                    msg = 'Search stopped: connection issues are impeding the query. Usually they are temporary: we suggest trying again in a few minutes.'
                    self._print_log('error', msg)
                    raise RuntimeError(msg) from None
                
                n_it += 1
                
                
            self._print_log('info','Data query: ended.')

            self.good_phot, self.quality_table = self._check_phot(**kwargs)

            nf=len(self.filters)

            query_keys={'G':'dr3_gmag_corr','Gbp':'dr3_phot_bp_mean_mag','Grp':'dr3_phot_rp_mean_mag','G2':'dr2_phot_g_mean_mag',
                        'Gbp2':'dr2_phot_bp_mean_mag','Grp2':'dr2_phot_rp_mean_mag','J':'j_m','H':'h_m','K':'ks_m',
                        'W1':'w1mpro','W2':'w2mpro','W3':'w3mpro','W4':'w4mpro',
                        'G_err':'dr3_phot_g_mean_mag_error','Gbp_err':'dr3_phot_bp_mean_mag_error','Grp_err':'dr3_phot_rp_mean_mag_error',
                        'G2_err':'dr2_g_mag_error','Gbp2_err':'dr2_bp_mag_error','Grp2_err':'dr2_rp_mag_error',
                        'J_err':'j_msigcom','H_err':'h_msigcom','K_err':'ks_msigcom',
                        'W1_err':'w1mpro_error','W2_err':'w2mpro_error','W3_err':'w3mpro_error','W4_err':'w4mpro_error',
                        'g':'ps1_g','r':'ps1_r','i':'ps1_i','z':'ps1_z','y':'ps1_y','g_err':'ps1_g_error',
                        'r_err':'ps1_r_error','i_err':'ps1_i_error','z_err':'ps1_z_error','y_err':'ps1_y_error',
                        'SDSS_g':'sdss_g','SDSS_r':'sdss_r','SDSS_i':'sdss_i','SDSS_z':'sdss_z','SDSS_u':'sdss_u','SDSS_g_err':'sdss_g_error',
                        'SDSS_r_err':'sdss_r_error','SDSS_i_err':'sdss_i_error','SDSS_z_err':'sdss_z_error','SDSS_u_err':'sdss_u_error'}

            phot = np.full([nst,nf],np.nan)
            phot_err = np.full([nst,nf],np.nan)
            for i in range(nf):
                try:
                    phot[:,i] = self.good_phot[query_keys[self.filters[i]]].filled(np.nan)
                    phot_err[:,i] = self.good_phot[query_keys[self.filters[i]+'_err']].filled(np.nan)
                except (ValueError,TypeError):
                    phot[:,i] = MaskedColumn(self.good_phot[query_keys[self.filters[i]]],dtype=float).filled(np.nan)
                    phot_err[:,i] = MaskedColumn(self.good_phot[query_keys[self.filters[i]+'_err']],dtype=float).filled(np.nan)
                except KeyError:
                    continue

            self.app_phot = phot
            self.app_phot_err = phot_err
            ra = np.array(self.good_phot['ra'].filled(np.nan))
            dec = np.array(self.good_phot['dec'].filled(np.nan))
            par = np.array(self.good_phot['dr3_parallax'].filled(np.nan))
            par_err = np.array(self.good_phot['dr3_parallax_error'].filled(np.nan))
            u, = np.where(np.isnan(par))
            if len(u)>0:
                par2 = np.array(self.good_phot['dr2_parallax'].filled(np.nan))
                par_err2 = np.array(self.good_phot['dr2_parallax_error'].filled(np.nan))
                u2, = np.where((np.isnan(par)) & (np.isnan(par2) == False))
                par = np.where(np.isnan(par),par2,par)
                par_err = np.where(np.isnan(par_err),par_err2,par_err)
                for i in range(len(u2)):
                    self._print_log('info',f'Invalid parallax in Gaia DR3 for star {self.ID[u2[i]][0]}, using DR2 instead.')
                u, = np.where(np.isnan(par))
                if len(u)>0:
                    par2 = np.array(self.good_phot['hipparcos_parallax'].filled(np.nan))
                    par_err2 = np.array(self.good_phot['hipparcos_parallax_error'].filled(np.nan))
                    u2, = np.where((np.isnan(par)) & (np.isnan(par2) == False))
                    par = np.where(np.isnan(par),par2,par)
                    par_err = np.where(np.isnan(par_err),par_err2,par_err)
                    for i in range(len(u2)):
                        self._print_log('info',f'Invalid parallax in Gaia DR3 and DR2 for star {self.ID[u2[i]][0]}, using Hipparcos instead.')
                    
            self.ebv_err = kwargs['ebv_err'] if 'ebv_err' in kwargs else None
            if 'ebv' in kwargs:
                self.ebv = kwargs['ebv']
                self._print_log('info','Extinction type: provided by the user')
                self.phot_table['ebv'] = self.ebv
                self.phot_table['ebv_err'] = self.ebv_err
            else:
                tt0 = time.perf_counter()
                self.ebv = self.interstellar_ext(ra=ra, dec=dec, par=par, 
                                                 ext_map=ext_map,
                                                 logger=self.__logger)
                tt1 = time.perf_counter()
                if self.verbose>1: 
                    print('Time for the computation of extinctions: {0:.2f} s.'.format(tt1-tt0))
                if ext_map is None:
                    self._print_log('info',"Extinction neglected, because 'ext_map' was set to None.")
                else: self._print_log('info',f'Extinction type: computed using {ext_map} extinction map.')
                self.phot_table['ebv'] = self.ebv
                self.phot_table['ebv_err'] = self.ebv_err

            for col in ['ebv', 'ebv_err']:
                self.phot_table[col].unit = 'mag'
            self.abs_phot,self.abs_phot_err = self.app_to_abs_mag(self.app_phot, par,
                                                                  app_mag_error=self.app_phot_err,
                                                                  parallax_error=par_err,
                                                                  ebv=self.ebv, 
                                                                  ebv_error=self.ebv_err, 
                                                                  filters=self.filters)
            self._print_log('info','Input photometry: apparent, converted to absolute.')
            self.par = par
            self.par_err = par_err
            if self.verbose > 0:
                filename = os.path.join(self.path, (self.__sample_name+'_phot_table.csv'))
                ascii.write(self.phot_table, filename, format='csv', overwrite=True)

        if self.verbose>1: 
            logging.shutdown()

    def __getitem__(self,i):

        if isinstance(i,str): return self.__dict__[i]
        elif isinstance(i,int): i = [i]

        new = copy.deepcopy(self)
        for j in new.__dict__.keys():
            try:
                if isinstance(new.__dict__[j],str): continue
                elif j == 'surveys': continue
                elif j == 'filters': continue
                elif j == 'additional_outputs': continue
                elif j == '_SampleObject__input':
                    new_input = {}
                    for key in new.__dict__[j]:
                        if key == 'file':
                            try:
                                new_input[key] = list(np.array(new.__dict__[j][key])[i])
                            except IndexError:
                                new_input[key] = list(np.array(new.ID['ID'])[i])
                        else:
                            new_input[key] = new.__dict__[j][key]
                    new.__dict__[j] = new_input
                    continue
                elif hasattr(new.__dict__[j], '__len__') == False: continue
                new.__dict__[j]=new.__dict__[j][i]
            except TypeError:
                continue
        n=len(new.abs_phot)
        if len(new.abs_phot.shape) == 1:
            new.abs_phot = new.abs_phot.reshape([1,n])
            new.abs_phot_err = new.abs_phot_err.reshape([1,n])
            new.app_phot = new.app_phot.reshape([1,n])
            new.app_phot_err = new.app_phot_err.reshape([1,n])
            new.par = np.array([new.par])
            new.par_err = np.array([new.par_err])
            if 'ID' in new.__dict__.keys():
                new.ID = Table(new.ID)
            new.GaiaID = Table(new.GaiaID)
            if 'phot_table' in new.__dict__.keys():
                new.good_phot = Table(new.good_phot)
                new.phot_table = Table(new.phot_table)
                new.quality_table = Table(new.quality_table)

        return new

    def __len__(self):
        return len(self.par)

    def __repr__(self):

        if self.mode == 1:
            s='SampleObject('
            l=self._SampleObject__input['file']
            if isinstance(l,list):
                s+="['"+"','".join(l)+"'], "
            else:
                s+=f"'{l}', "
                s=s.replace('\\','/')

            ext_map = self._SampleObject__input['ext_map']
            s+=f"ext_map='{ext_map}', "

            for i in self._SampleObject__input:
                if i == 'file': 
                    continue
                elif i == 'ext_map': 
                    continue
                elif i == 'mock_file':
                    s += f"{i}='{self._SampleObject__input[i]}'"
                    s = s.replace('\\','/')
                elif isinstance(self._SampleObject__input[i],str):
                    s += f"{i}='{str(self._SampleObject__input[i])}'"
                elif isinstance(self._SampleObject__input[i],list):
                    l = [f"'{j}'" for j in self._SampleObject__input[i]]
                    s += f"{i}=[{','.join(l)}]"
                elif isinstance(self._SampleObject__input[i],np.ndarray): 
                    s += i+'=np.'+np.array_repr(self._SampleObject__input[i])
                else: 
                    s += i+'='+str(self._SampleObject__input[i])
                s += ', '
            if s.endswith(', '): 
                s = s[:-2]
            s += ')'
            s = s.replace('=nan','=np.nan')
            return s
        
        elif self.mode == 2:
            s = 'SampleObject('+repr_table(self.phot_table)+', '

            ext_map = self._SampleObject__input['ext_map']
            s += f'ext_map={ext_map}, '

            for i in self._SampleObject__input:
                if isinstance(self._SampleObject__input[i],list):
                    l = [f"'{j}'" for j in self._SampleObject__input[i]]
                    s += f"{i}=[{','.join(l)}]"
                elif isinstance(self._SampleObject__input[i],np.ndarray): 
                    s+=i+'=np.'+np.array_repr(self._SampleObject__input[i])
                elif i == 'file': continue
                elif i == 'mock_file':
                    s += f"{i}='{str(self._SampleObject__input[i])}'"
                    s = s.replace('\\','/')
                elif i == 'ext_map': continue
                elif isinstance(self._SampleObject__input[i],str): 
                    s += f"{i}='{str(self._SampleObject__input[i])}'"
                else: s += f"{i}={str(self._SampleObject__input[i])}"
                s += ', '
            if s.endswith(', '):
                s = s[:-2]
            s += ')'
            s = s.replace('=nan','=np.nan')
            return s
        else: 
            raise ValueError('Has the value for self.mode been modified? Restore it to 1 or 2.')

    def __str__(self):

        if self.mode == 1:
            s = 'A SampleObject instance, mode 1 \n'
            l = self._SampleObject__input['file']
            if isinstance(l,list):
                s += 'Input IDs: '+"['"+"','".join(l)+"'] \n"
            else:
                s += "Input file: '"+l+"' \n"
                s = s.replace('\\','/')
            s+='Settings: '
            for i in self._SampleObject__input:
                if i == 'file': continue
                elif i == 'mock_file':
                    s += i+'='+"'"+self._SampleObject__input[i]+"'"
                    s = s.replace('\\','/')
                elif isinstance(self._SampleObject__input[i],str):
                    s += i+"='"+str(self._SampleObject__input[i])+"'"
                elif isinstance(self._SampleObject__input[i],list):
                    l = [str(j) for j in self._SampleObject__input[i]]
                    s += i+'=['+','.join(l)+']'
                elif isinstance(self._SampleObject__input[i],np.ndarray): 
                    s += i+'=np.'+np.array_repr(self._SampleObject__input[i])
                else: s += i+'='+str(self._SampleObject__input[i])
                s+=', '
            if s.endswith(', '): 
                s = s[:-2]
            s = s.replace('=nan','=np.nan')
            return s
        elif self.mode == 2:
            s = 'A SampleObject instance, mode 2 \n'
            s += 'Input data: '+repr_table(self.phot_table)+' \n'
            s += 'Settings: '
            for i in self._SampleObject__input:
                if isinstance(self._SampleObject__input[i],list):
                    l = [str(j) for j in self._SampleObject__input[i]]
                    s += i+'=['+','.join(l)+']'
                elif isinstance(self._SampleObject__input[i],np.ndarray): 
                    s+=i+'=np.'+np.array_repr(self._SampleObject__input[i])
                elif i == 'file': 
                    continue
                elif i == 'mock_file':
                    s += i+'='+"'"+self._SampleObject__input[i]+"'"
                    s = s.replace('\\','/')
                else: s += i+'='+str(self._SampleObject__input[i])
                s += ', '
            if s.endswith(', '): 
                s = s[:-2]
            s = s.replace('=nan','=np.nan')
            return s
        else: raise ValueError('Has the value for self.mode been modified? Restore it to 1 or 2.')
    
    def __eq__(self, other):
        return np.array_equal(np.array(self.ID['ID']), np.array(other.ID['ID']))
            

    @staticmethod
    def _convert_table_ndarray(data, columns=None, dtypes=None, masked=None, mask=None):

        if isinstance(data, Table):
            masked = data.masked
            mask = data.mask
            data = data.filled()

            columns = data.columns
            array = data.as_array()
            dtypes = [array.dtype[i] for i in range(len(array.dtype))]

            n_columns, n_rows = len(columns), len(data)
            y = np.zeros([n_rows,n_columns], dtype = dt)
            for i in range(n_rows):
                for j in range(n_columns):
                    y[i,j] = array[i][j]

            y = y.astype(str).astype(object)
            columns = np.array(list(columns)).astype(dt)
            dtypes = np.array(dtypes).astype(str).astype(dt)

            return y, columns, dtypes, masked, mask

        elif isinstance(data, np.ndarray):
            if mask is not None:
                t = Table(data, dtype = dtypes, names = columns, masked = True)
                t.mask = mask
            else:
                t = Table(data, dtype = dtypes, names = columns)

            return t
        
    def export_to_file(self, output_file):
        
        """
        Saves an existing SampleObject instance to a .h5 file.
            Input:
            - output_file: string, required. Full path to the output .h5 file.
            Output:
            - no output is returned apart from the .h5 file.
        """

        with h5py.File(output_file, 'w') as hf:

            for i in self.__dict__:

                if isinstance(self[i], np.ndarray):
                    if isinstance(self[i][0],str):
                        values = np.array(self[i], dtype=dt) 
                    else: values = self[i]

                    dset = hf.create_dataset(i, data = values, compression='gzip', compression_opts=9)
                elif isinstance(self[i], Table):

                    array, columns, dtypes, masked, mask = SampleObject._convert_table_ndarray(self[i])
                    dset = hf.create_dataset(i, data = array, compression='gzip', compression_opts=9)
                    if mask is not None:
                        dset = hf.create_dataset(i+'_mask', data = mask, compression='gzip', compression_opts=9)
                    hf.attrs[i+'_columns'] = columns
                    hf.attrs[i+'_dtypes'] = dtypes
                    hf.attrs[i+'_masked'] = masked

                else:
                    try:
                        hf.attrs[i] = self[i]
                    except TypeError:
                        hf.attrs[i] = str(self[i])
            hf.attrs['class'] = type(self).__name__

            if self.mode == 2:
                array, columns, dtypes, masked, mask = SampleObject._convert_table_ndarray(self['_SampleObject__input']['file'])
                name = 'input_table'
                dset = hf.create_dataset(name, data = array, compression='gzip', compression_opts=9)
                if mask is not None:
                    dset = hf.create_dataset(name+'_mask', data = mask, compression='gzip', compression_opts=9)
                hf.attrs[name+'_columns'] = columns
                hf.attrs[name+'_dtypes'] = dtypes
                hf.attrs[name+'_masked'] = masked

    @classmethod
    def import_from_file(cls, file, verbose=2):
        
        """
        Alternative initializer for the SampleObject class.
        It creates an instance from a valid .h5 file.
            Input:
            - file: string, required. Full path to the input .h5 file.
            - verbose: int, optional. Degree of verbosity of everything related to the output instance. It can be:
                - 0: no file is saved, nothing is printed on the screen;
                - 1: a .csv file with retrieved information is saved (1), few info are printed on the screen;
                - 2: in addition to the output of 1, a log file is created;
                - 3: in addition to the output of 2, .csv files are created when executing SampleObject.get_params().
              Default: 2.
            Output:
            - instance: a SampleObject instance.
        """

        if file.endswith('.h5') == False:
            raise TypeError('Input file must be a .h5 file.')

        if isinstance(verbose, int) == False:
            raise TypeError("Keyword 'verbose' must be an integer between 0 and 3.")
        elif (verbose < 0) | (verbose > 3):
            raise ValueError("Keyword 'verbose' must be an integer between 0 and 3.")

        replacements = {'array': 'np.array', 'nan': 'np.nan'}
        dic = {}
        with h5py.File(file,"r") as hf:

            if hf.attrs['class'] != 'SampleObject':
                raise ValueError('The provided file does not appear to be an instance of the SampleObject class.')

            dataset_names = []
            for dataset in hf.values():
                dataset_names.append(str(dataset).split('"')[1])

            for name in dataset_names:
                if name+'_columns' in hf.attrs.keys():
                    array = hf.get(name)[:]
                    columns = hf.attrs[name+'_columns']
                    dtypes = hf.attrs[name+'_dtypes']
                    masked = hf.attrs[name+'_masked']
                    if name+'_mask' in dataset_names:
                        mask = hf.get(name+'_mask')[:]
                    else: mask = None

                    t = SampleObject._convert_table_ndarray(array, columns, dtypes, masked, mask)

                    dic[name] = t
                elif name.endswith('_mask'): continue
                else:
                    dic[name] = hf.get(name)[:]

            for i in hf.attrs.keys():
                if i.endswith('_columns'): continue
                elif i.endswith('_dtypes'): continue
                elif i.endswith('_masked'): continue
                dic[i] = hf.attrs[i]

        for key in dic.keys():
            if isinstance(dic[key], np.ndarray):
                if isinstance(dic[key][0], bytes):
                    dic[key] = dic[key].astype(str)
            elif isinstance(dic[key], str):
                try:
                    dic[key] = eval(dic[key])
                except:
                    continue
        if dic['mode'] == 2:
            dic['_SampleObject__input'] = eval(dic['_SampleObject__input'].split(", 'file'")[0]+'}')
            dic['_SampleObject__input']['file'] = dic['input_table']
            del dic['input_table']

        if 'mock_file' in dic['_SampleObject__input'].keys():
            dic['_SampleObject__input']['mock_file'] = file

        instance = SampleObject(dic['_SampleObject__input']['file'], verbose = verbose, stop_init = True, mock_file = file)

        not_to_be_updated = ['_SampleObject__logger', 'file', '_SampleObject__sample_name', '_SampleObject__sample_path', 'log_file']
        for key in dic.keys():
            if key in not_to_be_updated:continue
            instance.__dict__[key] = dic[key]
        
        
        instance._print_log('info',logo)
        instance._print_log('info',f'Program started. Running madys {MADYS_VERSION}.')
        instance._print_log('info',f'Input type: instance recovered from an existing file: {instance.file}.')
        
        if instance['_SampleObject__madys_version'] != MADYS_VERSION:
            old_version = instance['_SampleObject__madys_version']
            instance._print_log('warning',f'This file was generated with a different version of madys ({old_version}). Beware, for undesired behaviours or bugs might occur because of compatibility problems.')
            warnings.warn(f'This file was generated with a different version of madys ({old_version}). Beware, for undesired behaviours or bugs might occur because of compatibility problems.')
        
        instance._print_log('info',r'Filters required: {0}.'.format(', '.join(instance.filters)))
        instance._print_log('info',f'No. of stars: {len(instance)}.')
        if 'ebv' in dic['_SampleObject__input'].keys():
            instance._print_log('info','Extinction type: provided by the user.')
        elif (dic['_SampleObject__input']['ext_map'] is None) | (dic['_SampleObject__input']['ext_map'] == 'None'):
            instance._print_log('info',"Extinction neglected, because 'ext_map' was set to None.")
        else: instance._print_log('info',r'Extinction type: computed using {0} extinction map.'.format(dic['_SampleObject__input']['ext_map']))

        return instance
        
    @classmethod
    def merge(cls, list_of_instances, indices=None):
        
        """
        Merges two or more SampleObject instances into a single one.
            Input:
            - list_of_instances: list, required. List of SampleObject instances.
            - indices: list, optional. List of numpy array of indices that will be used to reorder the final star list.
              No repeated elements must be present in any of the arrays, nor in the concatenated array of arrays.
              The total number of elements of the concatenated array must match the length of the final object.
              Ex.: taken two instances with 3 and 2 elements, a valid 'indices' list would be [np.array([4,2,0]),np.array([1,3])].
              The first element of the first input instance will be sent to row index 4, the second one to 2, and so on.
            Output:
            - instance: the merged SampleObject instance.
        This method is currently supported only for mode 1. The following keywords must be equal for all instances:
        'mode', 'surveys', 'filters'. Additionally, the MADYS version that produced the instances must be the same.
        
        """


        n = len(list_of_instances)
        check_equal = ['_SampleObject__madys_version', 'mode', 'surveys', 'filters']

        for i in range(1, n):
            if list_of_instances[0]._SampleObject__input['ext_map'] != list_of_instances[i]._SampleObject__input['ext_map']:
                raise ValueError(f"Databases 0 and {i} can't be merged: the ext_map employed is different.")

        for col in check_equal:
            for i in range(1, n):
                try:
                    if list_of_instances[0].__dict__[col] != list_of_instances[i].__dict__[col]:
                        raise ValueError(f"Databases 0 and {i} can't be merged: keyword {col} assumes different values.")
                except ValueError:
                    if (list_of_instances[0].__dict__[col] != list_of_instances[i].__dict__[col]).any():
                        raise ValueError(f"Databases 0 and {i} can't be merged: keyword {col} assumes different values.")

        n_obj = np.sum([len(obj) for obj in list_of_instances])

        if indices is not None:
            if isinstance(indices, list) == False:
                raise TypeError("'indices' must be a list!")
            if len(indices) != n:
                raise ValueError("'indices' must be an array with the same number of elements as 'list_of_instances'.")
            for i, el in enumerate(indices):
                if isinstance(el, np.ndarray) == False:
                    raise TypeError("Each element of 'indices' must be a numpy array.")
                elif 'int' not in str(type(el[0])):
                    raise TypeError("Each element of 'indices' must be a numpy array of type int.")
                elif len(el) != len(list_of_instances[i]):
                    raise ValueError("Each element of 'indices' must have the same size as its corresponding instance in 'list_of_instances'.")
                elif len(np.unique(el)) < len(el):
                    raise ValueError("Repeated entry in the following index array: {0}. Every element should be unique.".format(el))
                elif len(np.unique(el)) < len(el):
                    raise ValueError("Repeated entry in the following index array: {0}. Every element should be unique.".format(el))

            conc_indices = np.concatenate(indices)
            if len(np.unique(conc_indices)) < n_obj:
                raise ValueError("Repeated entry in the final index array: {0}. Every element should be unique.".format(conc_indices))


        instance = copy.deepcopy(list_of_instances[0])

        for attr in instance.__dict__.keys():
            val = instance.__dict__[attr]
            if attr in check_equal: continue
            elif attr == '_SampleObject__input':
                continue
            elif (isinstance(val, Table)) | (isinstance(val, Column)):
                instance.__dict__[attr] = vstack([inst.__dict__[attr] for inst in list_of_instances])
                if indices is not None:
                    instance.__dict__[attr] = instance.__dict__[attr][conc_indices]
            elif isinstance(val, np.ndarray):
                if len(val.shape) == 1:
                    instance.__dict__[attr] = np.concatenate([inst.__dict__[attr] for inst in list_of_instances])
                    if indices is not None:
                        instance.__dict__[attr] = instance.__dict__[attr][conc_indices]
                elif len(val.shape) == 2:
                    instance.__dict__[attr] = np.vstack([inst.__dict__[attr] for inst in list_of_instances])
                    if indices is not None:
                        instance.__dict__[attr] = instance.__dict__[attr][conc_indices, :]
                else:
                    raise ValueError(f'Unrecognized data shape: {type(val)}')

        instance._SampleObject__logger = None
        instance.verbose = 0
        instance.file = ''

        instance._SampleObject__input['file'] = list(np.array(instance.ID['ID']))
        if len(np.unique([inst._SampleObject__id_type for inst in list_of_instances])) > 1: 
            instance._SampleObject__id_type = 'other'
            instance._SampleObject__input['id_type'] = 'other'

        return instance
            

    ############################################# catalog queries ############################################

    def _sample_name_ext(self):
        sample_p,sample_name=os.path.split(self.file)
        i=0
        while sample_name[i]!='.': i=i+1
        self.__ext=sample_name[i:]
        self.__sample_name=sample_name[:i]
        self.__sample_path=self.file.split(self.__ext)[0]

    def _read_IDs(self):

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
                elif 'object_name' in col: ID = IDtab['object_name']
                else: raise ValueError('The columns with IDs was not found! Check that a "ID"/"id"/"source_id"/"object_name" column is present, and try again.')
            else: ID=IDtab[col[0]]

            if self.__id_type=='DR2':
                for i in range(len(ID)):
                    st_id=str(ID.loc[i])
                    if 'Gaia DR3' in st_id:
                        raise ValueError("id_type='DR2' but the star "+st_id+" is from Gaia DR3.")
                    elif 'Gaia DR2' not in st_id:
                        ID.loc[i]='Gaia DR2 '+st_id
            elif self.__id_type=='DR3':
                for i in range(len(ID)):
                    st_id=str(ID.loc[i])
                    if 'Gaia DR2' in st_id:
                        raise ValueError("id_type='DR3' but the star "+st_id+" is from Gaia DR2.")
                    elif 'Gaia DR3' not in st_id:
                        ID.loc[i]='Gaia DR3 '+st_id

            if isinstance(ID,pd.Series): ID=ID.to_frame()
            ID=Table.from_pandas(ID)
            ID.rename_column(ID.colnames[0], 'ID')

        return ID

    def _get_simbad_names(self):
        ns=len(self.ID)
        results = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(ns):
                res=Simbad.query_objectids(self.ID[i])
                try:
                    results.append(np.array(res['ID'].astype(str)))
                except TypeError:
                    results.append(np.array(['']))

        self.SimbadIDs = results
        return results

    def _get_survey_names(self, survey='Gaia DR3', write_id_attribute=True):

        if 'SimbadIDs' not in self.__dict__.keys():
            self._get_simbad_names()

        if write_id_attribute:
            dic = {'Gaia DR3': 'Gaia', 'Gaia DR2': 'Gaia',
                   '2MASS':'tmass', 'HIP': 'Hip'}
        else:
            dic = {'Gaia DR3': 'DR3', 'Gaia DR2': 'DR2',
                   '2MASS':'tmass', 'HIP': 'Hip'}

        ns = len(self.ID)
        newID = [''] * ns
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(ns):
                found = False
                res = self.SimbadIDs[i]
                for rr in res:
                    if str(rr).startswith(survey):
                        newID[i] = str(rr)
                        found = True
                        break
                if found == False:
                    if survey == '2MASS':
                        newID[i] = survey+' J0000'
                    else:
                        newID[i] = survey+' 0000'

        self.__dict__[dic[survey]+'ID'] = Table({'ID':newID})
    
    
###################################################

    def _list_chunk(self, ind=None, key_name=None, 
                    id_list=None, equality='=',
                    quote_mark=False, force_id_type=None):

        query_list=''

        if force_id_type is not None:
            id_type = force_id_type
        else:
            id_type = self.__id_type

        if (key_name is None) & (id_list is None):

            if id_type == 'DR2':
                id_str = 'Gaia DR2 '
                id_sea = 'dr2.source_id'
                if 'DR2ID' in self.__dict__.keys():
                    used_id = self.DR2ID
                else:
                    used_id = self.GaiaID
            elif id_type == 'HIP':
                id_str = 'HIP'
                id_sea = 'hipparcos_newreduction.hip'
                used_id = self.ID
            elif id_type == '2MASS':
                id_str = '2MASS J'
                id_sea = 'tmass.designation'
                used_id = self.tmassID
            else:
                id_str = 'Gaia DR3 '
                id_sea = 'dr3.source_id'
                if 'DR3ID' in self.__dict__.keys():
                    used_id = self.DR3ID
                else:
                    used_id = self.GaiaID

            if ind is None: 
                ind = len(self.GaiaID)
            if quote_mark:
                if isinstance(ind, np.ndarray):
                    for i in ind:
                        id = str(used_id[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+" '"+id+"' OR "
                else:
                    for i in range(ind):
                        id = str(used_id[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+" '"+id+"' OR "
            else:
                if isinstance(ind, np.ndarray):
                    for i in ind:
                        id = str(used_id[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+' '+id+' OR '
                else:
                    for i in range(ind):
                        id = str(used_id[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+' '+id+' OR '
        elif key_name is None:
            if id_type == 'DR2':
                id_str = 'Gaia DR2 '
                id_sea = 'dr2.source_id'
            elif id_type == 'HIP':
                id_str = 'HIP'
                id_sea = 'hipparcos_newreduction.hip'
            elif id_type == '2MASS':
                id_str = '2MASS J'
                id_sea = 'tmass.designation'
            else:
                id_str = 'Gaia DR3 '
                id_sea = 'dr3.source_id'

            if ind is None: 
                ind = len(id_list)
            if quote_mark:
                if isinstance(ind, np.ndarray):
                    for i in ind:
                        id = str(id_list[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+" '"+id+"' OR "
                else:
                    for i in range(ind):
                        id = str(id_list[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+" '"+id+"' OR "
            else:
                if isinstance(ind, np.ndarray):
                    for i in ind:
                        id = str(id_list[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+' '+id+' OR '
                else:
                    for i in range(ind):
                        id = str(id_list[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+' '+id+' OR '
        else:
            if id_type == 'HIP':
                used_id = self.ID
            else:
                used_id = self.GaiaID

            id_sea = key_name
            if ind is None: ind = len(id_list)
            if quote_mark:
                if isinstance(ind, np.ndarray):
                    for i in ind:
                        id = str(id_list[i])
                        query_list += id_sea+' '+equality+" '"+id+"' OR "
                else:
                    for i in range(ind):
                        id = str(used_id[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+" '"+id+"' OR "
            else:
                if isinstance(ind, np.ndarray):
                    for i in ind:
                        id = str(id_list[i])
                        query_list += id_sea+' '+equality+' '+id+' OR '
                else:
                    for i in range(ind):
                        id = str(used_id[i]).split(id_str)[1].strip()
                        query_list += id_sea+' '+equality+' '+id+' OR '

        query_list = query_list[:-4]

        return query_list

    def _query_string(self, query_list, surveys=None, force_id_type=None):
        
        qstr1, qstr2, qstr3 = '', '', ''
        surveys = surveys if surveys is not None else self.surveys

        if force_id_type is not None:
            id_type = force_id_type
        else:
            id_type = self.__id_type

        if 'gaia' in surveys:
            qstr1+='    dr3.designation as dr3_id, dr2.designation as dr2_id, '
            qstr2+="""
            dr3.ref_epoch as dr3_epoch, dr3.ra as ra, dr3.dec as dec,
            dr3.ra_error as ra_error, dr3.dec_error as dec_error,
            dr3.parallax as dr3_parallax,
            dr3.parallax_error as dr3_parallax_error, dr3.parallax_over_error as dr3_parallax_over_error,
            dr3.pmra as dr3_pmra, dr3.pmra_error as dr3_pmra_error,
            dr3.pmdec as dr3_pmdec, dr3.pmdec_error as dr3_pmdec_error,
            dr3.ra_dec_corr as dr3_ra_dec_corr, dr3.ra_parallax_corr as dr3_ra_parallax_corr,
            dr3.ra_pmra_corr as dr3_ra_pmra_corr, dr3.ra_pmdec_corr as dr3_ra_pmdec_corr,
            dr3.dec_parallax_corr as dr3_dec_parallax_corr,
            dr3.dec_pmra_corr as dr3_dec_pmra_corr, dr3.dec_pmdec_corr as dr3_dec_pmdec_corr,
            dr3.parallax_pmra_corr as dr3_parallax_pmra_corr, dr3.parallax_pmdec_corr as dr3_parallax_pmdec_corr,
            dr3.pmra_pmdec_corr as dr3_pmra_pmdec_corr, dr3.phot_g_mean_mag as dr3_phot_g_mean_mag,
            dr3.phot_g_mean_flux as dr3_phot_g_mean_flux, dr3.phot_g_mean_flux_error as dr3_phot_g_mean_flux_error,
            dr3.phot_bp_mean_flux as dr3_phot_bp_mean_flux, dr3.phot_bp_mean_flux_error as dr3_phot_bp_mean_flux_error,
            dr3.phot_bp_mean_mag as dr3_phot_bp_mean_mag,
            dr3.phot_rp_mean_flux as dr3_phot_rp_mean_flux, dr3.phot_rp_mean_flux_error as dr3_phot_rp_mean_flux_error,
            dr3.phot_rp_mean_mag as dr3_phot_rp_mean_mag,
            dr3.bp_rp as dr3_bp_rp, dr3.phot_bp_rp_excess_factor as dr3_phot_bp_rp_excess_factor,
            dr3.ruwe as dr3_ruwe, dr3.astrometric_params_solved as dr3_astrometric_params_solved,
            dr2.ref_epoch as dr2_epoch, dr2.ra as dr2_ra, dr2.dec as dr2_dec,
            dr2.ra_error as dr2_ra_error, dr2.dec_error as dr2_dec_error,
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
            dr2.radial_velocity, dr2.radial_velocity_error,
            hipparcos_newreduction.hip as hipparcos_id,hipparcos_newreduction.ra as hipparcos_ra,
            hipparcos_newreduction.dec as hipparcos_dec,hipparcos_newreduction.plx as hipparcos_parallax,
            hipparcos_newreduction.pm_ra as hipparcos_pmra,hipparcos_newreduction.pm_de as hipparcos_pmdec,
            hipparcos_newreduction.e_ra_rad as hipparcos_ra_error,
            hipparcos_newreduction.e_de_rad as hipparcos_dec_error,
            hipparcos_newreduction.e_plx as hipparcos_parallax_error,
            hipparcos_newreduction.e_pm_ra as hipparcos_pmra_error,
            hipparcos_newreduction.e_pm_de as hipparcos_pmdec_error,"""

            if id_type == 'HIP':
                qstr3+="""
                FROM
                    public.hipparcos_newreduction AS hipparcos_newreduction
                LEFT OUTER JOIN
                    gaiadr3.hipparcos2_best_neighbour AS dr3hip
                    ON dr3hip.original_ext_source_id = hipparcos_newreduction.hip
                LEFT OUTER JOIN
                    gaiadr3.gaia_source AS dr3
                    ON dr3.source_id = dr3hip.source_id
                LEFT OUTER JOIN
                    gaiadr2.hipparcos2_best_neighbour AS dr2hip
                    ON dr2hip.original_ext_source_id = hipparcos_newreduction.hip
                LEFT OUTER JOIN
                    gaiadr2.gaia_source AS dr2
                    ON dr2.source_id = dr2hip.source_id
                LEFT OUTER JOIN
                    gaiadr2.ruwe AS dr2ruwe
                    ON dr2.source_id = dr2ruwe.source_id"""
                
            elif id_type == 'DR2':
                qstr3+="""
                FROM
                    gaiadr2.gaia_source as dr2
                LEFT OUTER JOIN
                    gaiadr3.dr2_neighbourhood AS dr2xmatch
                    ON dr2.source_id = dr2xmatch.dr2_source_id
                LEFT OUTER JOIN
                    gaiadr3.gaia_source as dr3
                    ON dr2xmatch.dr3_source_id = dr3.source_id
                LEFT OUTER JOIN
                    gaiadr2.ruwe as dr2ruwe
                    ON dr2.source_id = dr2ruwe.source_id
                LEFT OUTER JOIN
                    gaiadr2.hipparcos2_best_neighbour as dr2hip
                    ON dr2.source_id = dr2hip.source_id
                LEFT OUTER JOIN
                    public.hipparcos_newreduction AS hipparcos_newreduction
                    ON dr2hip.original_ext_source_id = hipparcos_newreduction.hip"""
                
            else:
                qstr3+="""
                FROM
                    gaiadr3.gaia_source as dr3
                LEFT OUTER JOIN
                    gaiadr3.dr2_neighbourhood AS dr2xmatch
                    ON dr3.source_id = dr2xmatch.dr3_source_id
                LEFT OUTER JOIN
                    gaiadr2.gaia_source as dr2
                    ON dr2xmatch.dr2_source_id = dr2.source_id
                LEFT OUTER JOIN
                    gaiadr2.ruwe as dr2ruwe
                    ON dr2xmatch.dr2_source_id = dr2ruwe.source_id
                LEFT OUTER JOIN
                    gaiadr3.hipparcos2_best_neighbour as dr3hip
                    ON dr3.source_id = dr3hip.source_id
                LEFT OUTER JOIN
                    public.hipparcos_newreduction AS hipparcos_newreduction
                    ON dr3hip.original_ext_source_id = hipparcos_newreduction.hip"""

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
                gaiadr3.tmass_psc_xsc_best_neighbour AS tmassxmatch
                ON dr3.source_id = tmassxmatch.source_id
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
            allwise.ra as wise_ra, allwise.dec as wise_dec,"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiadr3.allwise_best_neighbour AS allwisexmatch
                ON dr3.source_id = allwisexmatch.source_id
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
                gaiadr3.apassdr9_best_neighbour AS apassxmatch
                ON dr3.source_id = apassxmatch.source_id
            LEFT OUTER JOIN
                external.apassdr9 AS apass
                ON apassxmatch.clean_apassdr9_oid = apass.recno
            """
        if 'sdss' in surveys:
            qstr1+='sdss.objid as sdss_id, '
            qstr2+="""
            sdss.u as sdss_u, sdss.err_u as sdss_u_error, sdss.g as sdss_g, sdss.err_g as sdss_g_error, 
            sdss.r as sdss_r, sdss.err_r as sdss_r_error, sdss.i as sdss_i, sdss.err_i as sdss_i_error,
            sdss.z as sdss_z, sdss.err_z as sdss_z_error,"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiadr3.sdssdr13_best_neighbour AS sdssxmatch
                ON dr3.source_id = sdssxmatch.source_id
            LEFT OUTER JOIN
                external.sdssdr13_photoprimary as sdss
                ON sdssxmatch.original_ext_source_id = sdss.objid
            """
        if 'panstarrs' in surveys:
            qstr1+='ps1xmatch.source_id as panstarrs_id, '
            qstr2+="""
            panstarrs.g_mean_psf_mag as ps1_g, g_mean_psf_mag_error as ps1_g_error, panstarrs.r_mean_psf_mag as ps1_r, r_mean_psf_mag_error as ps1_r_error,
            panstarrs.i_mean_psf_mag as ps1_i, i_mean_psf_mag_error as ps1_i_error, panstarrs.z_mean_psf_mag as ps1_z, z_mean_psf_mag_error as ps1_z_error,
            panstarrs.y_mean_psf_mag as ps1_y, y_mean_psf_mag_error as ps1_y_error,"""
            qstr3+="""
            LEFT OUTER JOIN
                gaiadr3.panstarrs1_best_neighbour AS ps1xmatch
                ON dr3.source_id = ps1xmatch.source_id
            LEFT OUTER JOIN
                gaiadr2.panstarrs1_original_valid AS panstarrs
                ON ps1xmatch.original_ext_source_id = panstarrs.obj_id
            """

        if qstr2.rstrip().endswith(','): qstr2=qstr2.rstrip()[:-1]

        qstr0="""
        select
        """
        qstr4="""
        WHERE """+query_list
        qstr=qstr0+qstr1+qstr2+qstr3+qstr4

        return qstr

    def _query_hipparcos(self):

        hip_ids = np.array([hip.replace('HIP','').strip() for hip in self.ID['ID']])
        hip_string = ''
        for hip in hip_ids:
            hip_string += 'hipparcos_newreduction.hip = '+hip+' OR '
        hip_string = hip_string[:-4]

        qstr0="""
        select
            hipparcos_newreduction.hip as hipparcos_id,hipparcos_newreduction.ra as hipparcos_ra,
            hipparcos_newreduction.dec as hipparcos_dec,hipparcos_newreduction.plx as hipparcos_parallax,
            hipparcos_newreduction.pm_ra as hipparcos_pmra,hipparcos_newreduction.pm_de as hipparcos_pmdec,
            hipparcos_newreduction.e_ra_rad as hipparcos_ra_error,
            hipparcos_newreduction.e_de_rad as hipparcos_dec_error,
            hipparcos_newreduction.e_plx as hipparcos_parallax_error,
            hipparcos_newreduction.e_pm_ra as hipparcos_pmra_error,
            hipparcos_newreduction.e_pm_de as hipparcos_pmdec_error
            from
                public.hipparcos_newreduction as hipparcos_newreduction
        """

        new_colnames = {'hip': 'hipparcos_id', 
                        'ra': 'hipparcos_ra', 'dec': 'hipparcos_dec',
                        'plx': 'hipparcos_parallax',
                        'e_plx': 'hipparcos_parallax_error',
                        'pm_ra': 'hipparcos_pmra',
                        'pm_de': 'hipparcos_pmdec',
                        'e_pm_ra': 'hipparcos_pmra_error',
                        'e_pm_de': 'hipparcos_pmdec_error',
                       }


        qstr1="""
        WHERE """+hip_string
        qstr=qstr0+qstr1

        adql = QueryStr(qstr,verbose=False)
        t=gaia.query(adql)

        return t

    def _fix_hipparcos(self, t):

        if t.masked == False:
            t = Table(t, masked=True)

        if self.__id_type == 'HIP':

            gaia_colnames = ['dr3_id', 'dr2_id']
            gaia_release = ['DR3', 'DR2']

            for k in range(2):

                try:
                    w_fix, = np.where((t[gaia_colnames[k]].mask) | (t[gaia_colnames[k]] == ''))
                except AttributeError:
                    continue

                if (len(w_fix) == 0):
                    continue

                self._get_survey_names('Gaia '+gaia_release[k], write_id_attribute=False)

                gaia_ids = np.array(self.__dict__[gaia_release[k]+'ID']['ID'])

                t_hip = self._make_gaia_query(force_id_type=gaia_release[k], ind=w_fix)

                if len(t_hip) == 0:
                    continue

                t_hip_gaia_ids = np.array(t_hip[gaia_colnames[k]])
                __, i1, i2 = np.intersect1d(t_hip_gaia_ids, gaia_ids, return_indices = True)

                if len(i1)>0:
                    n_stars = len(t)
                    for col in t_hip.columns:
                        if 'hipparcos' in col: continue
                        column = np.array(t[col])
                        column[i2] = t_hip[col][i1]
                        if 'id' not in col:
                            try:
                                column = column.astype(float)
                            except (TypeError, ValueError):
                                pass
                        t[col] = column

                    w, = np.where(t[gaia_colnames[k]] == 'Gaia {0} 0000'.format(gaia_release[k]))
                    for col in t_hip.columns:
                        if 'hipparcos' in col: continue
                        if gaia_release[~k+2].lower() in col: 
                            continue
                        t[col].mask[w] = True    
                    w, = np.where(t[gaia_colnames[~k+2]] == 'Gaia {0} 0000'.format(gaia_release[~k+2]))
                    for col in t_hip.columns:
                        if gaia_release[~k+2].lower() in col:
                            t[col].mask[w] = True    


        else:
            w_fix, = np.where(t['hipparcos_id'].mask)
            if len(w_fix) == 0: return t

            self._get_survey_names('HIP')
            hip_ids = np.array(self.HipID['ID'])
            t_hip = self._make_gaia_query(force_id_type='HIP',ind=w_fix, id_list = hip_ids)

            if len(t_hip) == 0:
                return t

            for gaia_dr in ['dr2', 'dr3']:
                w_null_dr3, = np.where(t_hip[gaia_dr+'_id'] == '')
                if len(w_null_dr3):
                    for col in t_hip.columns:
                        if col.startswith(gaia_dr):
                            t_hip[col].mask[w_null_dr3] = True

            t_hip_ids = np.array(['HIP '+str(i) for i in t_hip['hipparcos_id']])
            __, i1, i2 = np.intersect1d(t_hip_ids, hip_ids, return_indices = True)

            if len(i1)>0:
                n_stars = len(t)
                for col in t_hip.columns:
                    old_mask = t[col].mask
                    t[col][i2] = np.where(t_hip[col].mask[i1], t[col][i2], t_hip[col][i1])
                    t[col] = MaskedColumn(t[col])
                    t[col].mask[i2] = np.where(t_hip[col].mask[i1], old_mask[i2], t[col].mask[i2])

        return t

    def _fix_double_entries(self, t, index=None):

        if index is None: 
            index = np.arange(len(self.GaiaID))
        n = len(index)
        if (len(t) < 2) & (n < 2): 
            return t

        t = Table(t,masked=True)
        n_t = len(t)
        id_type = self.__id_type

        w, = np.where(t['dr2_id'] == '')
        t['dr2_id'].mask[w] = True
        t['dr2_id'] = t['dr2_id'].filled('Gaia DR2 0000')

        w, = np.where(t['dr3_id'] == '')
        t['dr3_id'].mask[w] = True
        t['dr3_id'] = t['dr3_id'].filled('Gaia DR3 0000')

        hip_col = np.array(t['hipparcos_id'], dtype = str)
        gaia2_col = np.array([str(i).split('Gaia DR2')[1] for i in t['dr2_id']])
        gaia3_col = np.array([str(i).split('Gaia DR3')[1] for i in t['dr3_id']])
        ind = []
        p_mask = []
        t_mask = []
        if id_type == 'DR2':
            cols = ['dr3_id', 'ra', 'dec', 'dr3_epoch', 
                    'dr3_parallax', 'dr3_parallax_error', 
                    'dr3_parallax_over_error', 'dr3_pmra', 
                    'dr3_pmra_error', 'dr3_pmdec', 'dr3_pmdec_error', 
                    'dr3_ra_dec_corr', 'dr3_ra_parallax_corr', 
                    'dr3_ra_pmra_corr', 'dr3_ra_pmdec_corr', 
                    'dr3_dec_parallax_corr', 'dr3_dec_pmra_corr', 
                    'dr3_dec_pmdec_corr', 'dr3_parallax_pmra_corr', 
                    'dr3_parallax_pmdec_corr', 'dr3_pmra_pmdec_corr', 
                    'dr3_phot_g_mean_mag', 'dr3_phot_g_mean_flux', 
                    'dr3_phot_g_mean_flux_error', 'dr3_phot_bp_mean_flux', 
                    'dr3_phot_bp_mean_flux_error', 'dr3_phot_bp_mean_mag', 
                    'dr3_phot_rp_mean_flux', 'dr3_phot_rp_mean_flux_error', 
                    'dr3_phot_rp_mean_mag', 'dr3_bp_rp', 
                    'dr3_phot_bp_rp_excess_factor', 'dr3_ruwe', 
                    'dr3_astrometric_params_solved']
            for i in range(n):
                id = str(self.GaiaID[index[i]]).split('Gaia DR2')[1]
                w, = np.where(id==gaia2_col)
                if len(w) == 1:
                    ind.extend(w)
                    if gaia3_col[w] == ' 0000': 
                        t['dr3_id'].mask[w]=True
                elif len(w) == 0:
                    ind.append(0)
                    t_mask.append(i)
                else:
                    w1, = np.where(id == gaia3_col[w])
                    if len(w1) == 1: 
                        ind.extend(w[w1])
                    else:
                        ind.append(w[0])
                        p_mask.append(i)
        elif id_type == 'HIP':
            cols=['dr3_id', 'ra', 'dec', 'dr3_epoch', 
                  'dr3_parallax', 'dr3_parallax_error', 
                  'dr3_parallax_over_error', 'dr3_pmra', 
                  'dr3_pmra_error', 'dr3_pmdec', 
                  'dr3_pmdec_error', 'dr3_ra_dec_corr', 
                  'dr3_ra_parallax_corr', 'dr3_ra_pmra_corr', 
                  'dr3_ra_pmdec_corr', 'dr3_dec_parallax_corr', 
                  'dr3_dec_pmra_corr', 'dr3_dec_pmdec_corr', 
                  'dr3_parallax_pmra_corr', 'dr3_parallax_pmdec_corr', 
                  'dr3_pmra_pmdec_corr', 'dr3_phot_g_mean_mag', 
                  'dr3_phot_g_mean_flux', 'dr3_phot_g_mean_flux_error', 
                  'dr3_phot_bp_mean_flux', 'dr3_phot_bp_mean_flux_error', 
                  'dr3_phot_bp_mean_mag', 'dr3_phot_rp_mean_flux', 
                  'dr3_phot_rp_mean_flux_error', 'dr3_phot_rp_mean_mag', 
                  'dr3_bp_rp', 'dr3_phot_bp_rp_excess_factor', 'dr3_ruwe', 
                  'dr3_astrometric_params_solved', 'dr2_id', 'dr2_epoch',
                  'dr2_ra', 'dr2_dec', 'dr2_parallax',
                  'dr2_parallax_error', 'dr2_parallax_over_error', 
                  'dr2_pmra', 'dr2_pmra_error', 'dr2_pmdec', 
                  'dr2_pmdec_error', 'dr2_ra_dec_corr', 
                  'dr2_ra_parallax_corr', 'dr2_ra_pmra_corr', 
                  'dr2_ra_pmdec_corr', 'dr2_dec_parallax_corr', 
                  'dr2_dec_pmra_corr', 'dr2_dec_pmdec_corr', 
                  'dr2_parallax_pmra_corr', 'dr2_parallax_pmdec_corr', 
                  'dr2_pmra_pmdec_corr', 'dr2_phot_g_mean_mag', 
                  'dr2_phot_g_mean_flux', 'dr2_phot_g_mean_flux_error', 
                  'dr2_phot_bp_mean_flux', 'dr2_phot_bp_mean_flux_error', 
                  'dr2_phot_bp_mean_mag', 'dr2_phot_rp_mean_flux', 
                  'dr2_phot_rp_mean_flux_error', 'dr2_phot_rp_mean_mag', 
                  'dr2_bp_rp', 'dr2_phot_bp_rp_excess_factor', 'dr2_ruwe', 
                  'dr2_astrometric_params_solved', 'radial_velocity', 
                  'radial_velocity_error']
            for i in range(n):
                id = str(self.ID[index[i]]).split('HIP')[1].strip()
                w, = np.where(id == hip_col)
                if len(w) == 1:
                    ind.extend(w)
                    if gaia3_col[w] == ' 0000': t['dr3_id'].mask[w]=True
                    if gaia2_col[w] == ' 0000': t['dr2_id'].mask[w]=True
                elif len(w) == 0:
                    ind.append(0)
                    t_mask.append(i)
                else:
                    w1, = np.where(id == gaia3_col[w])
                    if len(w1) == 1: ind.extend(w[w1])
                    else:
                        ind.append(w[0])
                        p_mask.append(i)
        else:
            cols=['dr2_id', 'dr2_epoch', 'dr2_ra', 'dr2_dec', 
                  'dr2_parallax', 'dr2_parallax_error', 
                  'dr2_parallax_over_error', 'dr2_pmra', 
                  'dr2_pmra_error', 'dr2_pmdec', 'dr2_pmdec_error', 
                  'dr2_ra_dec_corr', 'dr2_ra_parallax_corr', 
                  'dr2_ra_pmra_corr', 'dr2_ra_pmdec_corr', 
                  'dr2_dec_parallax_corr', 'dr2_dec_pmra_corr', 
                  'dr2_dec_pmdec_corr', 'dr2_parallax_pmra_corr', 
                  'dr2_parallax_pmdec_corr', 'dr2_pmra_pmdec_corr', 
                  'dr2_phot_g_mean_mag', 'dr2_phot_g_mean_flux', 
                  'dr2_phot_g_mean_flux_error', 'dr2_phot_bp_mean_flux', 
                  'dr2_phot_bp_mean_flux_error', 'dr2_phot_bp_mean_mag', 
                  'dr2_phot_rp_mean_flux', 'dr2_phot_rp_mean_flux_error', 
                  'dr2_phot_rp_mean_mag', 'dr2_bp_rp', 
                  'dr2_phot_bp_rp_excess_factor', 'dr2_ruwe', 
                  'dr2_astrometric_params_solved', 'radial_velocity', 
                  'radial_velocity_error']
            for i in range(n):
                id = str(self.GaiaID[index[i]]).split('Gaia DR3')[1]
                w, = np.where(id == gaia3_col)
                if len(w) == 1:
                    ind.extend(w)
                    if gaia2_col[w] == ' 0000': 
                        t['dr2_id'].mask[w]=True
                elif len(w) == 0:
                    ind.append(0)
                    t_mask.append(i)
                else:
                    w1, = np.where(id == gaia2_col[w])
                    if len(w1) == 1: 
                        ind.extend(w[w1])
                    else:
                        ind.append(w[0])
                        p_mask.append(i)
        ind = np.array(ind)
        t = t[ind]
        if len(t_mask)>0:
            t_mask = np.array(t_mask)
            for j in t_mask:
                for i in t.columns:
                    t[i].mask[j] = True
        if len(p_mask)>0:
            p_mask = np.array(p_mask)
            for j in p_mask:
                for i in cols:
                    t[i].mask[j] = True
        return t

    def _make_gaia_query(self, force_id_type=None, ind=None, id_list=None):

        data = []
        nst = len(self.GaiaID)
        n_chunks = 1
        if ind is None:
            done = np.zeros(nst, dtype=bool)
        else:
            done = np.ones(nst, dtype=bool)
            done[ind] = 0    

        nit, nit_max = 0, 10
        while (np.sum(done) < nst) & (nit < nit_max):
            todo, = np.where(done==False)
            st = int(len(todo)/n_chunks)
            c, n_corr_err = 0, 0
            while 1:
                todo_c = todo[c*st:(c+1)*st]
                query_list = self._list_chunk(todo_c, 
                                              force_id_type=force_id_type, 
                                              id_list=id_list)
                qstr = self._query_string(query_list, 
                                          force_id_type=force_id_type)
                try:
                    adql = QueryStr(qstr, verbose=False)
                    t = gaia.query(adql)
                    data.append(t)
                    done[todo_c]=True
                    n_conn_err = 0
                    c+=1
                except JSONDecodeError:
                    continue
                except ExpatError:
                    time.sleep(0.3)
                    continue
                except RuntimeError as e:
                    nit += 1
                    time.sleep(0.5)
                    if nit > (nit_max-1):
                        if 'Query error.' in repr(e):
                            msg = """
                            The following query appears to be wrong: 
                            \n\n {0} 
                            \n\nThis is weird: it might be related to a modification of the syntax used in Gaia SQL queries.
                            \n\nTry to manually input it in https://gea.esac.esa.int/archive/ --> Search --> Advanced (ADQL) to check if this fails.
                            \n * If it fails, MADYS needs to be updated following the new default syntax.
                            \n * If it does not fail, a compatibility problem of the tap dipendency in madys has probably arisen. 
                            \nIn any case, contact the creators of MADYS and they will inspect the problem and come up with a solution for it.
                            """.format(adql)
                            raise RuntimeError(msg) from None
                        else:
                            raise RuntimeError(e) from None
                    continue
                except (ConnectionError, RemoteDisconnected):
                    time.sleep(1)
                    n_conn_err += 1
                if c >= n_chunks: 
                    break
                if n_conn_err > (nit_max-1): 
                    raise ConnectionError('Too many connection errors. Did you check your connection?')
            n_chunks *= 2
            nit += 1
            if nit > (nit_max-1): 
                raise RuntimeError('Perhaps '+str(nst)+' stars are too many?')

        if len(data) > 1:
            t = vstack(data)
        else: 
            t = data[0]

        if len(t) == 0:
            qstr = self._query_string('dr3.source_id = 6057914176870184064', force_id_type='DR3')
            adql = QueryStr(qstr,verbose=False)
            t = gaia.query(adql)
            for col in t.columns:
                col_type = type(t[col].filled()[0])
                if col == 'dr3_id':
                    t[col] = 'Gaia DR3 0000'
                elif col == 'dr2_id':
                    t[col] = 'Gaia DR2 0000'
                elif col == 'tmass_id':
                    t[col] = ''
                elif col_type in [int, np.int16, np.int32]:
                    t[col] = 0
                else:
                    t[col] = np.nan
                t[col].mask = True

        return t

    def _get_phot(self):
        start = time.time()

        t = self._make_gaia_query()

        t=self._fix_double_entries(t)

        if len(t) == 0:
            message = "The required objects do not appear to be in Gaia. MADYS can't be run in mode 1 with no Gaia data."
            self._print_log('error', message)
            raise ValueError(message) from None


        t=self._fix_2mass(t)

        t=self._fix_hipparcos(t)


        ra_G3, dec_G3 = SampleObject._astropy_column_to_numpy_array(t['ra']), SampleObject._astropy_column_to_numpy_array(t['dec'])
        ra_H, dec_H = SampleObject._astropy_column_to_numpy_array(t['hipparcos_ra']), SampleObject._astropy_column_to_numpy_array(t['hipparcos_dec'])
        ra_G3_err, dec_G3_err = SampleObject._astropy_column_to_numpy_array(t['ra_error']), SampleObject._astropy_column_to_numpy_array(t['dec_error'])
        ra_H_err, dec_H_err = SampleObject._astropy_column_to_numpy_array(t['hipparcos_ra_error']), SampleObject._astropy_column_to_numpy_array(t['hipparcos_dec_error'])

        ra_G2, dec_G2 = SampleObject._astropy_column_to_numpy_array(t['dr2_ra']), SampleObject._astropy_column_to_numpy_array(t['dr2_dec'])
        ra_G2_err, dec_G2_err = SampleObject._astropy_column_to_numpy_array(t['dr2_ra_error']), SampleObject._astropy_column_to_numpy_array(t['dr2_dec_error'])

        time_base = np.where(np.isnan(ra_G3), 24.25, 24.75)
        ra_G, dec_G = np.where(np.isnan(ra_G3), ra_G2, ra_G3), np.where(np.isnan(dec_G3), dec_G2, dec_G3)
        ra_G_err, dec_G_err = np.where(np.isnan(ra_G3_err), ra_G2_err, ra_G3_err), np.where(np.isnan(dec_G3_err), dec_G2_err, dec_G3_err)

        long_term_pmra, long_term_pmdec = (ra_G-ra_H)*np.cos((dec_G)*np.pi/180)*3.6e+6/time_base, (dec_G-dec_H)*3.6e+6/time_base
        long_term_pmra_err = np.sqrt(ra_G_err**2+ra_H_err**2)/time_base
        long_term_pmdec_err = np.sqrt(dec_G_err**2+dec_H_err**2)/time_base
        t['gaia_hip_pmra'] = long_term_pmra
        t['gaia_hip_pmra_error'] = long_term_pmra_err
        t['gaia_hip_pmdec'] = long_term_pmdec
        t['gaia_hip_pmdec_error'] = long_term_pmdec_err
        for column in ['gaia_hip_pmra', 'gaia_hip_pmra_error', 'gaia_hip_pmdec', 'gaia_hip_pmdec_error', 'hipparcos_pmra', 'hipparcos_pmra_error', 'hipparcos_pmdec', 'hipparcos_pmdec_error']:
            t[column].unit = u.mas/u.yr
        for column in ['hipparcos_parallax', 'hipparcos_parallax_error']:
            t[column].unit = u.mas
        for column in ['hipparcos_ra', 'hipparcos_dec', 'hipparcos_ra_error', 'hipparcos_dec_error']:
            t[column].unit = u.deg

        t = SampleObject._move_column_to_index(t, 'hipparcos_id', 'ra')

        data=[]
        with np.errstate(divide='ignore',invalid='ignore'):
            dr3_gmag_corr, dr3_gflux_corr = self._correct_gband(t.field('dr3_bp_rp'), t.field('dr3_astrometric_params_solved'), t.field('dr3_phot_g_mean_mag'), t.field('dr3_phot_g_mean_flux'))
            dr3_bp_rp_excess_factor_corr = self._dr3_correct_flux_excess_factor(t.field('dr3_bp_rp'), t.field('dr3_phot_bp_rp_excess_factor'))
            dr3_g_mag_error, dr3_bp_mag_error, dr3_rp_mag_error = self._gaia_mag_errors(t.field('dr3_phot_g_mean_flux'), t.field('dr3_phot_g_mean_flux_error'), t.field('dr3_phot_bp_mean_flux'), t.field('dr3_phot_bp_mean_flux_error'), t.field('dr3_phot_rp_mean_flux'), t.field('dr3_phot_rp_mean_flux_error'))
            dr2_bp_rp_excess_factor_corr = self._dr2_correct_flux_excess_factor(t.field('dr2_phot_g_mean_mag'), t.field('dr2_bp_rp'), t.field('dr2_phot_bp_rp_excess_factor'))
            dr2_g_mag_error, dr2_bp_mag_error, dr2_rp_mag_error = self._gaia_mag_errors(t.field('dr2_phot_g_mean_flux'), t.field('dr2_phot_g_mean_flux_error'), t.field('dr2_phot_bp_mean_flux'), t.field('dr2_phot_bp_mean_flux_error'), t.field('dr2_phot_rp_mean_flux'), t.field('dr2_phot_rp_mean_flux_error'))
            t_ext=Table([dr3_gmag_corr, dr3_gflux_corr, dr3_bp_rp_excess_factor_corr, dr3_g_mag_error, dr3_bp_mag_error, dr3_rp_mag_error, dr2_bp_rp_excess_factor_corr, dr2_g_mag_error, dr2_bp_mag_error, dr2_rp_mag_error],
                names=['dr3_gmag_corr', 'dr3_gflux_corr','dr3_phot_bp_rp_excess_factor_corr', 'dr3_phot_g_mean_mag_error', 'dr3_phot_bp_mean_mag_error', 'dr3_phot_rp_mean_mag_error', 'dr2_phot_bp_rp_excess_factor_corr', 'dr2_g_mag_error', 'dr2_bp_mag_error', 'dr2_rp_mag_error'],
                units=["mag", "", "", "mag", "mag", "mag", "", "mag", "mag", "mag"],
                descriptions=['dr3 G-band mean mag corrected as per Riello et al. (2021)', 'dr3 G-band mean flux corrected as per Riello et al. (2021)', 'dr3 BP/RP excess factor corrected as per Riello et al. (2021)','dr3 Error on G-band mean mag', 'dr3 Error on BP-band mean mag', 'dr3 Error on RP-band mean mag', 'DR2 BP/RP excess factor corrected as per Squicciarini et al. (2021)', 'DR2 Error on G-band mean mag', 'DR2 Error on BP-band mean mag', 'DR2 Error on RP-band mean mag'])
            t_ext['dr3_gflux_corr'].unit = u.electron/u.s
            data.append(hstack([self.ID, t, t_ext]))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.phot_table=vstack(data)
        if self.verbose>1:
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            message = "Total time needed to retrieve photometry for "+ str(len(self.GaiaID))+ " targets: - {:0>2}:{:0>2}:{:05.2f}.".format(int(hours),int(minutes),seconds)
            self._print_log('info', message)
            print(message)

    def _fix_2mass(self, t):

        w_fix, = np.where(t['tmass_id'] == '')
        if len(w_fix) == 0:
            return t

        self._get_survey_names('2MASS')
        id_string = self._list_chunk(force_id_type = '2MASS', quote_mark = True, ind = w_fix)

        tmass_ids = np.array(self.tmassID['ID'])

        query_2mass = """    SELECT
                         tmass.designation AS tmass_id,
                         tmass.j_m, tmass.j_msigcom,
                         tmass.h_m, tmass.h_msigcom,
                         tmass.ks_m, tmass.ks_msigcom,
                         tmass.ph_qual,
                         tmass.ra AS tmass_ra, tmass.dec AS tmass_dec
                         FROM
                             gaiadr1.tmass_original_valid AS tmass
                         WHERE """+id_string

        adql = QueryStr(query_2mass,verbose=False)
        t_tmass=gaia.query(adql)    

        if len(t_tmass) == 0:
            return t

        queried_tmass_ids = np.array(['2MASS J'+i for i in t_tmass['tmass_id']])
        __, i1, i2 = np.intersect1d(queried_tmass_ids, tmass_ids, return_indices = True)

        if len(i1)>0:
            n_stars = len(t)
            for col in t_tmass.columns:
                t[col][i2] = t_tmass[col][i1]

            return t


            for col in t_tmass.columns:

                if 'hipparcos' in col: continue
                column = np.array(t[col])
                column[i2] = t_hip[col][i1]
                if 'id' not in col:
                    try:
                        column = column.astype(float)
                    except (TypeError, ValueError):
                        pass
                t[col] = column

            w, = np.where(t['dr3_id'] == 'Gaia DR3 0000')
            for col in t_hip.columns:
                if 'hipparcos' in col: continue
                if 'dr2' in col: continue
                t[col].mask[w] = True    
            w, = np.where(t['dr2_id'] == 'Gaia DR2 0000')
            for col in t_hip.columns:
                if 'dr2' in col:
                    t[col].mask[w] = True    

        return t
        
    @staticmethod
    def _move_column_to_index(table, column_name, index):
        #if index is an integer, moves the column to the index position
        #if it's a column name, it moves the column before that column
        new_order = np.array(list(table.columns))
        new_order = new_order[new_order != column_name]
        if isinstance(index, str):
            ind, = np.where(new_order == index)[0]
            new_order = np.concatenate((new_order[:ind],[column_name],new_order[ind:]))
        else:
            new_order = np.concatenate((new_order[:index],[column_name],new_order[index:]))
        return table[list(new_order)]
        
    @staticmethod
    def _astropy_column_to_numpy_array(column):
        try:
            return np.array(column.filled(np.nan))
        except AttributeError:
            return np.array(column)
            

    def _divide_query(self, query, key_name=None,
                      id_list=None, n_it_max=10,
                      equality='=', quote_mark=False):

        f = gaia.query

        n_chunks = 1
        nst = len(id_list) if id_list is not None else len(self.GaiaID)
        done = np.zeros(nst, dtype=bool)
        nit=0
        data=[]
        while (np.sum(done) < nst) & (nit < 10):
            todo, = np.where(done == False)
            st = int(len(todo)/n_chunks)
            for i in range(n_chunks):
                todo_c = todo[i*st:(i+1)*st]
                query_list = self._list_chunk(todo_c, key_name=key_name,
                                              id_list=id_list,
                                              equality=equality,
                                              quote_mark=quote_mark)
                qstr = query+query_list
                try:
                    adql = QueryStr(qstr,verbose=False)
                    t = f(adql)
                    data.append(t)
                    done[todo_c]=True
                except (JSONDecodeError, RuntimeError):
                    continue
            n_chunks *= 2
            nit += 1
            if nit > (n_it_max - 1): 
                raise RuntimeError('Perhaps '+str(nst)+' stars are too many?')

        if len(data) > 1:
            t=vstack(data)
        else: 
            t = data[0]
        return t


    ############################################# data handling and quality assessment #######################

    def _check_phot(self, **kwargs):

        t=copy.deepcopy(self.phot_table)
        t=Table(t, masked=True, copy=False)

        dr2_q, max_dr2_excess = self._dr2_quality(t.field('dr2_phot_bp_rp_excess_factor_corr'),
                                                  t.field('dr2_phot_g_mean_mag'))
        t['dr2_phot_bp_mean_mag'].mask[~dr2_q]=True
        t['dr2_phot_rp_mean_mag'].mask[~dr2_q]=True
        t['dr2_phot_bp_mean_mag'].fill_value = np.nan
        t['dr2_phot_rp_mean_mag'].fill_value = np.nan

        dr3_q, max_dr3_excess = self._dr3_quality(t.field('dr3_phot_bp_rp_excess_factor_corr'),
                                                  t.field('dr3_phot_g_mean_mag'))
        t['dr3_phot_bp_mean_mag'].mask[~dr3_q]=True
        t['dr3_phot_rp_mean_mag'].mask[~dr3_q]=True
        t['dr3_phot_bp_mean_mag'].fill_value = np.nan
        t['dr3_phot_rp_mean_mag'].fill_value = np.nan
        
        quality_dict = {}
        quality_dict['dr2_phot_bp_rp_excess_factor_corr'] = t['dr2_phot_bp_rp_excess_factor_corr']
        quality_dict['dr2_bp_rp_acceptance_threshold'] = max_dr2_excess
        quality_dict['dr2_bp_rp_retained'] = dr2_q
        quality_dict['dr3_phot_bp_rp_excess_factor_corr'] = t['dr3_phot_bp_rp_excess_factor_corr']
        quality_dict['dr3_bp_rp_acceptance_threshold'] = max_dr3_excess
        quality_dict['dr3_bp_rp_retained'] = dr3_q

        if '2mass' in self.surveys:
            if 'max_tmass_q' in kwargs:
                max_tmass_q = kwargs['max_tmass_q']
            else: max_tmass_q = 'A'
            tm_q = self._tmass_quality(t.field('ph_qual'), max_q=max_tmass_q)
            t['j_m'].mask[~tm_q[0]] = True
            t['h_m'].mask[~tm_q[1]] = True
            t['ks_m'].mask[~tm_q[2]] = True
            t['j_m'].fill_value = np.nan
            t['h_m'].fill_value = np.nan
            t['ks_m'].fill_value = np.nan

            quality_dict['tmass_quality'] = t['ph_qual']
            quality_dict['tmass_J_retained'] = tm_q[0]
            quality_dict['tmass_H_retained'] = tm_q[1]
            quality_dict['tmass_K_retained'] = tm_q[2]
        
        if 'wise' in self.surveys:
            if 'max_wise_q' in kwargs:
                max_wise_q = kwargs['max_wise_q']
            else: max_wise_q ='A'
            wise_q = self._allwise_quality(t.field('cc_flags'), 
                                           t.field('ph_qual_2'),
                                           max_q=max_wise_q)
            t['w1mpro'].mask[~wise_q[0]] = True
            t['w2mpro'].mask[~wise_q[1]] = True
            t['w3mpro'].mask[~wise_q[2]] = True
            t['w4mpro'].mask[~wise_q[3]] = True
            t['w1mpro'].fill_value = np.nan
            t['w2mpro'].fill_value = np.nan
            t['w3mpro'].fill_value = np.nan
            t['w4mpro'].fill_value = np.nan

            quality_dict['allwise_contamination_flag'] = t['cc_flags']
            quality_dict['allwise_quality'] = t['ph_qual_2']
            quality_dict['allwise_W1_retained'] = wise_q[0]
            quality_dict['allwise_W2_retained'] = wise_q[1]
            quality_dict['allwise_W3_retained'] = wise_q[2]
            quality_dict['allwise_W4_retained'] = wise_q[3]
            
        quality_table = Table(quality_dict)
        
        return t, quality_table

    @staticmethod
    def _dr2_correct_flux_excess_factor(phot_g_mean_mag, bp_rp, phot_bp_rp_excess_factor):
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
        C1 = phot_bp_rp_excess_factor + a0(bp_rp)+a1(bp_rp)*bp_rp+a2(bp_rp)*bp_rp**2+a3(bp_rp)*bp_rp**3+a4(bp_rp)*phot_g_mean_mag
        return C1

    @staticmethod
    def _dr3_correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
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
    def _gaia_mag_errors(phot_g_mean_flux, phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error):
        sigmaG_0 = 0.0027553202
        sigmaGBP_0 = 0.0027901700
        sigmaGRP_0 = 0.0037793818

        phot_g_mean_mag_error = np.sqrt((-2.5/np.log(10)*phot_g_mean_flux_error/phot_g_mean_flux)**2 + sigmaG_0**2)
        phot_bp_mean_mag_error = np.sqrt((-2.5/np.log(10)*phot_bp_mean_flux_error/phot_bp_mean_flux)**2 + sigmaGBP_0**2)
        phot_rp_mean_mag_error = np.sqrt((-2.5/np.log(10)*phot_rp_mean_flux_error/phot_rp_mean_flux)**2 + sigmaGRP_0**2)

        return phot_g_mean_mag_error, phot_bp_mean_mag_error, phot_rp_mean_mag_error

    @staticmethod
    def _correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux):
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
    def _dr2_quality(dr2_bp_rp_excess_factor,dr2_phot_g_mean_mag):
        s1=lambda x: 0.004+8e-12*x**7.55
        max_possible_excess = 3*s1(dr2_phot_g_mean_mag)
        with np.errstate(invalid='ignore'):
            q1 = abs(dr2_bp_rp_excess_factor) <= max_possible_excess
        q1[dr2_bp_rp_excess_factor.mask]=False
        return q1, max_possible_excess

    @staticmethod
    def _dr3_quality(dr3_bp_rp_excess_factor,dr3_phot_g_mean_mag):
        s1=lambda x: 0.0059898+8.817481e-12*x**7.618399
        max_possible_excess = 3*s1(dr3_phot_g_mean_mag)
        with np.errstate(invalid='ignore'):
            q1 = abs(dr3_bp_rp_excess_factor) <= max_possible_excess
        q1[dr3_bp_rp_excess_factor.mask]=False
        return q1, max_possible_excess

    @staticmethod
    def _tmass_quality(ph_qual,max_q='A'):
        q = np.array(['A','B','C','D','E','F','U','X'])
        w, = np.where(q == max_q)[0]
        n = len(ph_qual)
        qJ = np.zeros(n, dtype=bool)
        qH = np.zeros(n, dtype=bool)
        qK = np.zeros(n, dtype=bool)
        for i in range(n):
            if ph_qual.mask[i] == True: 
                continue
            if bool(ph_qual[i]) == False: 
                continue
            if ph_qual[i][0] in q[0:w+1]: 
                qJ[i]=True
            if ph_qual[i][1] in q[0:w+1]: 
                qH[i]=True
            if ph_qual[i][2] in q[0:w+1]: 
                qK[i]=True
                
        return qJ, qH, qK

    @staticmethod
    def _allwise_quality(cc_flags, ph_qual2, max_q='A'):
        q = np.array(['A','B','C','U','Z','X'])
        w, = np.where(q == max_q)[0]
        n = len(ph_qual2)
        qW1 = np.zeros(n, dtype=bool)
        qW2 = np.zeros(n, dtype=bool)
        qW3 = np.zeros(n, dtype=bool)
        qW4 = np.zeros(n, dtype=bool)
        for i in range(n):
            if (ph_qual2.mask[i] == True) | (cc_flags.mask[i] == True): 
                continue
            if (bool(ph_qual2[i]) == False) | (bool(cc_flags[i]) == False): 
                continue
            if (ph_qual2[i][0] in q[0:w+1]) & (cc_flags[i][0]=='0'): 
                qW1[i] = True
            if (ph_qual2[i][1] in q[0:w+1]) & (cc_flags[i][1]=='0'): 
                qW2[i] = True
            if (ph_qual2[i][2] in q[0:w+1]) & (cc_flags[i][2]=='0'): 
                qW3[i] = True
            if (ph_qual2[i][3] in q[0:w+1]) & (cc_flags[i][3]=='0'): 
                qW4[i] = True
                
        return qW1, qW2, qW3, qW4

    @staticmethod
    def _is_phot_good(phot, phot_err, max_phot_err=0.1):
        if type(phot) == float: 
            dim=0
        else:
            l = phot.shape
            dim = len(l)
        if dim <= 1:
            with np.errstate(invalid='ignore'):
                gs = (np.isnan(phot) == False) & (phot_err < max_phot_err) & (abs(phot) < 70)
        else:
            gs = np.zeros([l[0],l[1]])
            with np.errstate(invalid='ignore'):
                for i in range(l[1]): 
                    gs[:,i] = (np.isnan(phot[:,i]) == False) & (phot_err[:,i] < max_phot_err)
        return gs
    
    @staticmethod
    def _mask_bad_photometry(phot, phot_err, max_phot_err=0.1, data=None, errors=None):

        if data is None:
            new_phot, new_phot_err = copy.deepcopy(phot), copy.deepcopy(phot_err)
        else:
            new_phot, new_phot_err = copy.deepcopy(data), copy.deepcopy(errors)
            
        if isinstance(phot, float):
            if (np.isnan(phot)) | (phot_err > max_phot_err):
                new_phot, new_phot_err = np.nan, np.nan
        else:
            mask = (np.isnan(phot)) | (phot_err > max_phot_err)
            new_phot[mask], new_phot_err[mask] = np.nan, np.nan

        return new_phot, new_phot_err
    

    ############################################# isochrones and age-mass computation ########################

    @staticmethod
    def _merge_solutions(x):
        if (len(x.shape) < 2) | ((x.shape)[0] < 2): return x
        w0, = np.where(x[:,4] < 1e-3)
        if len(w0)>1:
            x = np.delete(x,w0,0)
            x[:,4] /= np.sum(x[:,4])
            if (x.shape)[0] < 2: 
                return x

        while True:
            for i in range(len(x)):
                found = 0
                for j in range(i+1, len(x)):
                    err_i_m,err_j_m = np.max([x[i,1],0.009]),np.max([x[j,1],0.009])
                    err_i_a,err_j_a = np.max([x[i,3],0.009]),np.max([x[j,3],0.009])
                    dd = (x[i,0]-x[j,0])**2/(err_i_m**2+err_j_m**2)+(x[i,2]-x[j,2])**2/(err_i_a**2+err_j_a**2)
                    if dd < 8:
                        x[i,1] = np.sqrt((x[i,4]*x[i,1])**2+(x[j,4]*x[j,1])**2)
                        if x[i,1] == 0:
                            x[i,1] = np.abs(x[i,0]-x[j,0])
                        x[i,0] = np.average([x[i,0],x[j,0]], weights=[x[i,4],x[j,4]])
                        x[i,3] = np.sqrt((x[i,4]*x[i,3])**2+(x[j,4]*x[j,3])**2)
                        if x[i,3] == 0:
                            x[i,3] = np.abs(x[i,2]-x[j,2])
                        x[i,2] = np.average([x[i,2],x[j,2]], weights=[x[i,4],x[j,4]])
                        x[i,4] += x[j,4]
                        x = np.delete(x,j,0)
                        found = 1
                        break
                if found: break
            if found == 0: 
                break
        w, = np.where(np.isnan(x[:,0]) == False)
        return x[w,:]

    def _get_agemass(self, model_version, **kwargs):

        GK = where_v(np.array(['G', 'K']), self.filters)

        if 'mass_range' in kwargs:
            kwargs['mass_range'] = IsochroneGrid._get_mass_range(kwargs['mass_range'], model_version, dtype='mass', **kwargs)
        elif np.max(GK) < len(self.filters): 
            mass_range = IsochroneGrid._get_mass_range(self.abs_phot[:,GK], model_version, **kwargs)
            kwargs['mass_range'] = mass_range
        else: 
            kwargs['mass_range'] = IsochroneGrid._get_mass_range([1e-6,1e+6], model_version, dtype='mass', **kwargs)

        self.ph_cut = kwargs['ph_cut'] if 'ph_cut' in kwargs else 0.2
        m_unit = kwargs['m_unit'] if 'm_unit' in kwargs else 'm_sun'
        additional_columns = kwargs['additional_columns'] if 'additional_columns' in kwargs else []
        save_maps = kwargs['save_maps'] if 'save_maps' in kwargs else False
        n_try = kwargs['n_try'] if 'n_try' in kwargs else 1000
        q = kwargs['secondary_q'] if 'secondary_q' in kwargs else None
        delta_mag_dict = kwargs['secondary_contrast'] if 'secondary_contrast' in kwargs else None
        has_companion = (q is not None) | (delta_mag_dict is not None)

        n_objects = len(self)

        if q is not None:
            if hasattr(q, '__len__'):
                if len(q) != n_objects:
                    raise ValueError("Argument 'secondary_q' must be either a scalar or an array with the same length as the target list.")
        if delta_mag_dict is not None:
            key = list(delta_mag_dict.keys())[0]
            val = delta_mag_dict[key]
            if hasattr(val, '__len__'):
                if len(val) != n_objects:
                    raise ValueError("Argument 'secondary_contrast' must be either a scalar or an array with the same length as the target list.")

        default_columns = ['logT','logL','logR','logg']
        output_columns = np.union1d(default_columns,additional_columns)        

        self._print_log('info','Starting age determination...')
        filt = np.concatenate([self.filters,output_columns])

        th_model = IsochroneGrid(model_version, filt, logger=self.__logger, search_model=False, **kwargs)
        iso_mass, iso_age, iso_filt, iso_data = th_model.masses, th_model.ages, th_model.filters, th_model.data

        self._print_log('info',f'Isochrones for model {model_version} correctly loaded.')
        iso_mass_log = np.log10(iso_mass)
        iso_age_log = np.log10(iso_age)

        filters_to_fix = np.setdiff1d(filt,iso_filt)
        fixed_filters = copy.deepcopy(self.filters)

        if len(filters_to_fix) > 0:
            filt = IsochroneGrid._fix_filters(filt, th_model.file)
            output_columns = IsochroneGrid._fix_filters(output_columns, th_model.file)
            fixed_filters = IsochroneGrid._fix_filters(fixed_filters, th_model.file)

        filters_to_fix = np.setdiff1d(filt, iso_filt)
        assert len(filters_to_fix)==0

        filter_index = where_v(output_columns,iso_filt)
        phys_data = iso_data[:,:,filter_index] #phys_data ordered as output_columns
        self.additional_outputs = output_columns
        n_params = len(filter_index)

        self_filter_index=where_v(fixed_filters,iso_filt)
        iso_data=iso_data[:,:,self_filter_index]
        iso_filt=iso_filt[self_filter_index] #iso_data ordered as self.filters

        self._print_log('info',r'Estimation of the following parameters will be attempted: {0}.'.format(', '.join(output_columns)))

        mass_range_str = ["%.2f" % s for s in th_model.mass_range]
        try:
            age_range_str = ["%s" % s for s in th_model.age_range]
        except TypeError: age_range_str=[str(th_model.age_range)]

        self._print_log('info',r'Input parameters for the model: mass range = [{0}] M_sun; age range = [{1}] Myr.'.format(', '.join(mass_range_str),', '.join(age_range_str)))
        if th_model.feh==0.0: self._print_log('info',f'Metallicity: solar (use SampleObject.info_models({model_version}) for details).')
        else: self._print_log('info',f'Metallicity: [Fe/H]={th_model.feh} (use SampleObject.info_models({model_version}) for details).')
        self._print_log('info',f'Helium content: Y={th_model.he} (use SampleObject.info_models({model_version}) for details).')
        if th_model.afe==0.0: self._print_log('info','Alpha enhancement: [a/Fe]=0.00.')
        else: self._print_log('info',f'Alpha enhancement: [a/Fe]={th_model.afe}.')
        if th_model.v_vcrit==0.0: self._print_log('info','Rotational velocity: 0.00 (non-rotating model).')
        else: self._print_log('info',f'Rotational velocity: {th_model.v_vcrit}*v_crit.')
        self._print_log('info',f'Spot fraction: f_spot={th_model.fspot}.')
        if th_model.B==0: self._print_log('info','Magnetic model? No.')
        else: self._print_log('info','Magnetic model? Yes.')

        self._print_log('info',f'Maximum allowed photometric uncertainty: {self.ph_cut} mag.')
        self._print_log('info',f'Mass unit of the results: {m_unit}.')
        self._print_log('info','Age unit of the results: Myr.')

        phot, phot_err = self.abs_phot, self.abs_phot_err

        l, l0 = iso_data.shape, phot.shape
        n_filters, n_masses, n_ages = len(iso_filt), l[0], l[1]

        true_filter_index = where_v(iso_filt, self.filters)

        phot, phot_err = phot[:,true_filter_index], phot_err[:,true_filter_index]
        red = np.zeros([n_objects,len(true_filter_index)])

        for i in range(len(true_filter_index)):
            red[:,i] = SampleObject.extinction(self.ebv,self.filters[true_filter_index[i]])
        if self.ebv_err is not None:
            ebv_err = self.ebv_err
            red_err = np.zeros([n_objects,len(true_filter_index)])

            for i in range(len(true_filter_index)):
                red_err[:,i] = SampleObject.extinction(self.ebv_err,self.filters[true_filter_index[i]])
            app_phot_err = np.sqrt(self.app_phot_err[:,true_filter_index]**2+red_err**2)
        else:
            app_phot_err = self.app_phot_err[:,true_filter_index]
            ebv_err = 0*self.ebv

        app_phot = self.app_phot[:,true_filter_index]-red

        phot, phot_err = SampleObject._mask_bad_photometry(phot, phot_err, max_phot_err = self.ph_cut)
        app_phot, app_phot_err = SampleObject._mask_bad_photometry(phot, phot_err, data = app_phot, errors = app_phot_err, max_phot_err = self.ph_cut)

        code = np.full(n_objects, 5, dtype=int)
        m_fit, m_min, m_max = np.full(n_objects,np.nan), np.full(n_objects,np.nan), np.full(n_objects,np.nan)
        additional_params = np.full([n_objects,3,n_params], np.nan)

        if has_companion:
            additional_params_B = np.full([n_objects,3,n_params], np.nan)
            m_B_fit, m_B_min, m_B_max = np.full(n_objects,np.nan), np.full(n_objects,np.nan), np.full(n_objects,np.nan)

        th_model.data, th_model.filters = iso_data, iso_filt

        iso_data_for_fit_list, mass_index_B_list, use_B_list, q_eff_list, index_of_iso = IsochroneGrid._alter_isochrones_for_binarity(th_model, q, delta_mag_dict)

        if n_ages==1: #just one age is present in the selected set of isochrones (e.g. pm13)
            a_fit=iso_age[0]+np.zeros(n_objects)
            a_min, a_max = a_fit, a_fit
            i_age = np.zeros(n_objects,dtype=int)
            case = 1
        elif isinstance(th_model.age_range,np.ndarray):
            if len(th_model.age_range.shape)==1: #the age is fixed for each star
                case = 1
                if len(th_model.age_range) != n_objects:
                    self._print_log('error','The number of stars is not equal to the number of input ages. Check the length of your input ages.')
                    raise ValueError(f'The number of stars ({n_objects}) is not equal to the number of input ages ({len(th_model.age_range)}).')
                a_fit = th_model.age_range
                a_min, a_max = a_fit, a_fit
                i_age = np.arange(0, n_objects, dtype=int)
            elif len(th_model.age_range[0])==2: #the age is to be found within the specified interval
                case = 2
                if len(th_model.age_range) != n_objects:
                    self._print_log('error','The number of stars is not equal to the number of input ages. Check the length of your input ages.')
                    raise ValueError(f'The number of stars ({n_objects}) is not equal to the number of input ages ({len(th_model.age_range)}).')
                i_age = np.zeros(th_model.age_range.shape, dtype=int)
                for i in range(n_objects):
                    i_age[i,:]=closest(iso_age,th_model.age_range[i,:])
                a_fit, a_min, a_max = np.full(n_objects,np.nan), np.full(n_objects,np.nan), np.full(n_objects,np.nan)
                ravel_indices = lambda i, j, j_len: j+j_len*i
            elif len(th_model.age_range[0])==3: #the age is fixed, and age_min and age_max are used to compute errors
                case = 3
                if len(th_model.age_range) != n_objects:
                    self._print_log('error','The number of stars is not equal to the number of input ages. Check the length of your input ages.')
                    raise ValueError(f'The number of stars ({n_objects}) is not equal to the number of input ages ({len(th_model.age_range)}).')
                i_age = np.zeros(th_model.age_range.shape, dtype=int)
                for i in range(n_objects):
                    i_age[i,:] = closest(iso_age, th_model.age_range[i,:])
                a_fit = iso_age[i_age[:,0]]
                a_min = iso_age[i_age[:,1]]
                a_max = iso_age[i_age[:,2]]
        else: #the program is left completely unconstrained
            case = 4
            a_fit, a_min, a_max = np.full(n_objects,np.nan), np.full(n_objects,np.nan), np.full(n_objects,np.nan)

        if (case == 2) | (case == 4):
            phys_nan = np.isnan(phys_data)
            phys_data2 = np.where(phys_nan, 0, phys_data)
            interp_params = []
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for p in range(n_params):
                    if np.isnan(np.nanmin(phys_data[:,:,p])):
                        interp_params.append(lambda x,y: np.nan)
                    else:
                        interp_params.append(RectBivariateSpline(iso_mass_log,iso_age_log,phys_data2[:,:,p]))

        all_maps, hot_p, all_solutions = [], [], []
        all_solutions_B = []

        chi2_min = np.full(n_objects, np.nan)
        len_sample = kwargs['n_tot'] if 'n_tot' in kwargs else len(self)

        more_than_one_iso = False
        if has_companion:
            if index_of_iso is not None:
                more_than_one_iso = True

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if case==3:

                n_try_safe = 2000
                n_loops = int(n_try // n_try_safe)

                sigma0 = np.full(([n_masses,n_filters]), np.nan)

                phot1, phot_err1 = np.zeros(phot.shape + (n_try,)), np.zeros(phot.shape + (n_try,))
                for j in range(n_try): phot1[:,:,j], phot_err1[:,:,j] = SampleObject.app_to_abs_mag(app_phot+app_phot_err*np.random.normal(size=phot.shape),self.par+self.par_err*np.random.normal(size=n_objects),app_mag_error=app_phot_err,parallax_error=self.par_err)

                for i in range(n_objects):

                    if more_than_one_iso:
                        index_iso_i = index_of_iso[i]
                        iso_data_for_fit, mass_index_B, use_B, q_eff = iso_data_for_fit_list[index_iso_i], mass_index_B_list[index_iso_i], use_B_list[index_iso_i], q_eff_list[index_iso_i]
                    else:
                        iso_data_for_fit, mass_index_B, use_B, q_eff = iso_data_for_fit_list, mass_index_B_list, use_B_list, q_eff_list

                    self.done += 1
                    if time.perf_counter()-self.t1 > 60:
                        time_left = (time.perf_counter()-self.t0)/self.done*(len_sample-self.done)
                        print('Program running. Done: '+str(self.done)+'/'+str(len_sample)+' ('+'{:.1f}'.format(self.done/len_sample*100)+ '%). Estimated time left: '+'{:.0f}'.format(time_left)+' s.')
                        self.t1 = time.perf_counter()

                    w, = np.where(np.isnan(phot[i,:]) == False)
                    if len(w)==0:
                        self._print_log('info',f'All magnitudes for star {i} have an error beyond the maximum allowed threshold ({self.ph_cut} mag): age and mass determinations was not possible.')
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])
                        code[i] = 1
                        continue
                    i00 = i_age[i,0]
                    b = np.zeros(len(w), dtype=bool)
                    for h in range(len(w)):
                        single_phot = phot[i,w[h]]
                        sigma0[:,w[h]] = ((iso_data_for_fit[:,i00,w[h]]-single_phot)/phot_err[i,w[h]])**2
                        try:
                            ii = np.nanargmin(sigma0[:,w[h]])
                            if abs(iso_data_for_fit[ii,i00,w[h]]-single_phot)<0.2: b[h]=True #if the best theor. match is more than 0.2 mag away, discards it
                        except ValueError: b[h]=True
                    if np.sum(b) == 0:
                        self._print_log('info',f'All magnitudes for star {i} are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.')
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])
                        code[i] = 2
                        continue
                    w2 = w[b]

                    if len(w2)>1:
                        chi2 = nansumwrapper(sigma0[:,w2],axis=1)/(np.sum(np.isnan(iso_data_for_fit[:,i00,w2])==False,axis=1)-1)
                    else:
                        chi2 = sigma0[:,w2].ravel()

                    if save_maps: all_maps.append(chi2)
                    value, index = SampleObject._min_v(chi2)
                    chi2_min[i] = value                    
                    index = index[0]

                    nominal_mass = iso_mass[index]

                    used_mass_index = closest(iso_mass_log,[iso_mass_log[index]-0.3,iso_mass_log[index]+0.3])
                    n_est = n_try*(i_age[i,2]-i_age[i,1]+1)
                    ind_array = np.zeros(n_est, dtype=int)                

                    iso_data_r = iso_data_for_fit[used_mass_index[0]:used_mass_index[1]][:, i_age[i,1]:i_age[i,2]+1][:,:,w2]
                    if np.sum(np.isnan(np.nanmin(iso_data_r, axis = 0))) > 0:
                        self._print_log('info',f'All magnitudes for star {i} are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.')
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])
                        code[i]=2
                        continue

                    if n_loops == 0:
                        sigma = ((iso_data_r[:,:,:,None]-phot1[i,w2,:])/phot_err1[i,w2,:])**2
                    else:
                        sigma = []
                        for l in range(n_loops+1):
                            if l < n_loops:
                                indices = np.arange(l*n_try_safe,(l+1)*n_try_safe)
                            elif (n_try - n_try_safe*n_loops) > 0:
                                indices = np.arange(l*n_try_safe,n_try)
                            else:
                                continue
                            phot2, phot_err2 = phot1[:,:,indices], phot_err1[:,:,indices]
                            sigma.append(((iso_data_r[:,:,:,None]-phot2[i,w2,:])/phot_err2[i,w2,:])**2)
                        sigma = np.concatenate(sigma, axis=3)

                    if len(w2) > 1:
                        chi2_denom = np.sum(np.isnan(iso_data_for_fit[used_mass_index[0]:used_mass_index[1],:,w2][:, i_age[i,1]:i_age[i,2]+1])==False,axis=2)-1
                        chi2 = nansumwrapper(sigma, axis=2)/chi2_denom[:, :, None]
                    else:
                        chi2 = nansumwrapper(sigma, axis=2)

                    ind_array = (used_mass_index[0]+np.nanargmin(chi2, axis=0))
                    m_perc = np.nanpercentile(iso_mass_log[ind_array], [16, 84], axis = 1)
                    m_min[i], m_fit[i], m_max[i] = 10**np.nanmin(m_perc), nominal_mass, 10**np.nanmax(m_perc)

                    if delta_mag_dict is not None:

                        if mass_index_B is not None:
                            ind_array_B = np.zeros_like(ind_array)
                            for k in range(n_ages):
                                ind_array_B[k,:] = mass_index_B[:,k].ravel()[ind_array[k,:]]

                            m_perc_B = np.nanpercentile(iso_mass_log[ind_array_B], [16, 84], axis = 1)
                            nominal_mass_B = iso_mass[mass_index_B[index,i00]]

                            m_B_min[i], m_B_fit[i], m_B_max[i] = 10**np.nanmin(m_perc_B), nominal_mass_B, 10**np.nanmax(m_perc_B)
                        else:
                            m_B_min[i], m_B_fit[i], m_B_max[i] = 0, 0, 0


                    code[i]=0

                    for p in range(n_params):
                        age_vector = np.arange(i_age[i,1],i_age[i,2]+1)
                        rep_ages = np.tile(age_vector,n_try).reshape(len(age_vector), n_try)

                        param_estimates = phys_data[ind_array,rep_ages,p]
                        param_perc = np.nanpercentile(param_estimates, [16, 84], axis = 1)

                        additional_params[i,:,p] = np.nanmin(param_perc), phys_data[index,i00,p], np.nanmax(param_perc)

                        if delta_mag_dict is not None:
                            if mass_index_B is not None:
                                param_estimates_B = phys_data[ind_array_B,rep_ages,p]
                                param_perc_B = np.nanpercentile(param_estimates_B, [16, 84], axis = 1)

                                additional_params_B[i,:,p] = np.nanmin(param_perc_B), phys_data[mass_index_B[index,i00],i00,p], np.nanmax(param_perc_B)
                        elif q is not None:
                            if mass_index_B is not None:
                                ind_array_B = mass_index_B[ind_array]
                                param_estimates_B = phys_data[ind_array_B,rep_ages,p]
                                param_perc_B = np.nanpercentile(param_estimates_B, [16, 84], axis = 1)

                                additional_params_B[i,:,p] = np.nanmin(param_perc_B), phys_data[mass_index_B[index],i00,p], np.nanmax(param_perc_B)


                dic={'age':a_fit, 'age_min':a_min, 'age_max':a_max,
                     'mass':m_fit, 'mass_min':m_min, 'mass_max':m_max,
                     'ebv':self.ebv, 'ebv_err':ebv_err, 'chi2_min':chi2_min, 'chi2_maps':all_maps, 'weight_maps':hot_p
                     }

                for p,param in enumerate(output_columns):
                    if param.startswith('log'):
                        if param=='logR':
                            dic['radius'] = 10**additional_params[:,1,p].ravel()
                            dic['radius_min'] = 10**additional_params[:,0,p].ravel()
                            dic['radius_max'] = 10**additional_params[:,2,p].ravel()
                        elif param=='logT':
                            dic['Teff'] = 10**additional_params[:,1,p].ravel()
                            dic['Teff_min'] = 10**additional_params[:,0,p].ravel()
                            dic['Teff_max'] = 10**additional_params[:,2,p].ravel()
                        else:
                            dic[param] = additional_params[:,1,p].ravel()
                            dic[param+'_min'] = additional_params[:,0,p].ravel()
                            dic[param+'_max'] = additional_params[:,2,p].ravel()
                    else:
                        dic['synth_'+param] = additional_params[:,1,p].ravel()
                        dic['synth_'+param+'_min'] = additional_params[:,0,p].ravel()
                        dic['synth_'+param+'_max'] = additional_params[:,2,p].ravel()

                if q is not None:
                    if more_than_one_iso:
                        q_eff_ordered = q_eff_list[index_of_iso]
                    else:
                        q_eff_ordered = q_eff_list
                    m_B_fit = dic['mass']*q_eff_ordered
                    m_B_min = dic['mass_min']*q_eff_ordered
                    m_B_max = dic['mass_max']*q_eff_ordered

                if has_companion:
                    dic['mass_B'] = m_B_fit
                    dic['mass_B_min'] = m_B_min
                    dic['mass_B_max'] = m_B_max

                    for p,param in enumerate(output_columns):
                        if param.startswith('log'):
                            if param=='logR':
                                dic['radius_B'] = 10**additional_params_B[:,1,p].ravel()
                                dic['radius_B_min'] = 10**additional_params_B[:,0,p].ravel()
                                dic['radius_B_max'] = 10**additional_params_B[:,2,p].ravel()
                            elif param=='logT':
                                dic['Teff_B'] = 10**additional_params_B[:,1,p].ravel()
                                dic['Teff_B_min'] = 10**additional_params_B[:,0,p].ravel()
                                dic['Teff_B_max'] = 10**additional_params_B[:,2,p].ravel()
                            else:
                                dic[param+'_B'] = additional_params_B[:,1,p].ravel()
                                dic[param+'_B_min'] = additional_params_B[:,0,p].ravel()
                                dic[param+'_B_max'] = additional_params_B[:,2,p].ravel()
                        else:
                            dic['synth_'+param+'_B'] = additional_params_B[:,1,p].ravel()
                            dic['synth_'+param+'_B_min'] = additional_params_B[:,0,p].ravel()
                            dic['synth_'+param+'_B_max'] = additional_params_B[:,2,p].ravel()

                dic['fit_status'] = code

            elif case == 1:
                sigma = np.full(([n_masses,1,n_filters]),np.nan)
                for i in range(n_objects):
                    self.done+=1

                    if more_than_one_iso:
                        index_iso_i = index_of_iso[i]
                        iso_data_for_fit, mass_index_B, use_B, q_eff = iso_data_for_fit_list[index_iso_i], mass_index_B_list[index_iso_i], use_B_list[index_iso_i], q_eff_list[index_iso_i]
                    else:
                        iso_data_for_fit, mass_index_B, use_B, q_eff = iso_data_for_fit_list, mass_index_B_list, use_B_list, q_eff_list

                    if time.perf_counter()-self.t1>60:
                        time_left=(time.perf_counter()-self.t0)/self.done*(len_sample-self.done)
                        print('Program running. Done: '+str(self.done)+'/'+str(len_sample)+' ('+'{:.1f}'.format(self.done/len_sample*100)+ '%). Estimated time left: '+'{:.0f}'.format(time_left)+' s.')
                        self.t1=time.perf_counter()
                    w, = np.where(np.isnan(phot[i,:]) == False)
                    if len(w)==0:
                        self._print_log('info',f'All magnitudes for star {i} have an error beyond the maximum allowed threshold ({self.ph_cut} mag): age and mass determinations was not possible.')
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])
                        code[i]=1
                        continue
                    i00 = i_age[i]

                    acceptable_delta_mag = np.nanmin(np.abs(iso_data_for_fit[:, i00, :]-phot[i,:]), axis=0) < 0.2
                    n_acceptable_delta_mag = np.nansum(acceptable_delta_mag)
                    if n_acceptable_delta_mag == 0:
                        self._print_log('info',f'All magnitudes for star {i} are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.')
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])
                        code[i] = 2
                        continue

                    sigma[:,0,:] = ((iso_data_for_fit[:,i00,:]-phot[i])/phot_err[i])**2

                    if n_acceptable_delta_mag > 1:
                        chi2=nansumwrapper(sigma[:,0,acceptable_delta_mag],axis=1)/(np.sum(np.isnan(iso_data_for_fit[:,i00,acceptable_delta_mag])==False,axis=1)-1) #no. of degrees of freedom = no. filters - one parameter (mass)
                    else:
                        chi2=sigma[:,0,acceptable_delta_mag].ravel()

                    if save_maps: all_maps.append(chi2)
                    value, index = SampleObject._min_v(chi2)

                    if delta_mag_dict is not None:
                        if mass_index_B is not None:
                            index_B = mass_index_B[index[0],i00]
                        else: index_B = None
                    elif q is not None:
                        if mass_index_B is not None:
                            index_B = mass_index_B[index[0]]
                        else: index_B = None

                    chi2_min[i] = value/(n_acceptable_delta_mag-2)
                    m_fit[i] = iso_mass[index[0]]
                    if delta_mag_dict is not None:
                        if index_B is not None:
                            m_B_fit[i] = iso_mass[index_B]
                        else:
                            m_B_fit[i] = 0
                    a_fit[i] = iso_age[i00]
                    code[i] = 0

                    m_values = np.zeros(n_try)
                    if delta_mag_dict is not None:
                        m_values_B = np.zeros(n_try)

                    additional_params_samples = np.zeros([n_try,n_params])
                    for p in range(n_params):
                        additional_params[i,1,p] = phys_data[index[0],i00,p]

                    if has_companion:
                        if index_B is not None:
                            additional_params_B_samples = np.zeros([n_try,n_params])
                            for p in range(n_params):
                                additional_params_B[i,1,p] = phys_data[index_B,i00,p]

                    for j in range(n_try):
                        phot1, phot_err1 = SampleObject.app_to_abs_mag(app_phot[i,:]+app_phot_err[i,:]*np.random.normal(size=n_filters),self.par[i]+self.par_err[i]*np.random.normal(),app_mag_error=app_phot_err[i,:],parallax_error=self.par_err[i])
                        for h in range(n_filters):
                            sigma[:,0,h] = ((iso_data_for_fit[:,i00,h]-phot1[h])/phot_err1[h])**2
                        metrics = np.sum(sigma[:, :, acceptable_delta_mag], axis=2)
                        value, index = SampleObject._min_v(metrics)
                        m_values[j] = iso_mass_log[index[0]]
                        for p in range(n_params):
                            additional_params_samples[j,p] = phys_data[index[0],i00,0]

                        if has_companion:
                            if index_B is not None:
                                index_B = mass_index_B[index[0]].ravel()
                                if delta_mag_dict is not None:
                                    m_values_B[j] = iso_mass_log[index_B]
                                for p in range(n_params):
                                    additional_params_B_samples[j,p] = phys_data[index_B,i00,0]

                    for p in range(n_params):
                        additional_params_samples_std = np.std(additional_params_samples[:,p],ddof=1)
                        additional_params[i, 0, p] = additional_params[i, 1, p] - additional_params_samples_std
                        additional_params[i, 2, p] = additional_params[i, 1, p] + additional_params_samples_std

                    if has_companion:
                        if index_B is not None:
                            for p in range(n_params):
                                additional_params_B_samples_std = np.std(additional_params_B_samples[:,p],ddof=1)
                                additional_params_B[i, 0, p] = additional_params_B[i, 1, p] - additional_params_B_samples_std
                                additional_params_B[i, 2, p] = additional_params_B[i, 1, p] + additional_params_B_samples_std

                    m_min[i]=10**(np.log10(m_fit[i])-np.std(m_values,ddof=1))
                    m_max[i]=10**(np.log10(m_fit[i])+np.std(m_values,ddof=1))                
                    if delta_mag_dict is not None:
                        m_B_min[i]=10**(np.log10(m_B_fit[i])-np.std(m_values_B,ddof=1))
                        m_B_max[i]=10**(np.log10(m_B_fit[i])+np.std(m_values_B,ddof=1))

                dic={'age':a_fit, 'age_min':a_min, 'age_max':a_max,
                     'mass':m_fit, 'mass_min':m_min, 'mass_max':m_max,
                     'ebv':self.ebv, 'ebv_err':ebv_err, 'chi2_min':chi2_min, 'chi2_maps':all_maps, 'weight_maps':hot_p
                     }

                for p,param in enumerate(output_columns):
                    if param.startswith('log'):
                        if param=='logR':
                            dic['radius'] = 10**additional_params[:,1,p].ravel()
                            dic['radius_min'] = 10**additional_params[:,0,p].ravel()
                            dic['radius_max'] = 10**additional_params[:,2,p].ravel()
                        elif param=='logT':
                            dic['Teff'] = 10**additional_params[:,1,p].ravel()
                            dic['Teff_min'] = 10**additional_params[:,0,p].ravel()
                            dic['Teff_max'] = 10**additional_params[:,2,p].ravel()
                        else:
                            dic[param] = additional_params[:,1,p].ravel()
                            dic[param+'_min'] = additional_params[:,0,p].ravel()
                            dic[param+'_max'] = additional_params[:,2,p].ravel()
                    else:
                        dic['synth_'+param] = additional_params[:,1,p].ravel()
                        dic['synth_'+param+'_min'] = additional_params[:,0,p].ravel()
                        dic['synth_'+param+'_max'] = additional_params[:,2,p].ravel()

                if q is not None:
                    if more_than_one_iso:
                        q_eff_ordered = q_eff_list[index_of_iso]
                    else:
                        q_eff_ordered = q_eff_list
                    m_B_fit = dic['mass']*q_eff_ordered
                    m_B_min = dic['mass_min']*q_eff_ordered
                    m_B_max = dic['mass_max']*q_eff_ordered

                if has_companion:
                    dic['mass_B'] = m_B_fit
                    dic['mass_B_min'] = m_B_min
                    dic['mass_B_max'] = m_B_max

                    for p,param in enumerate(output_columns):
                        if param.startswith('log'):
                            if param=='logR':
                                dic['radius_B'] = 10**additional_params_B[:,1,p].ravel()
                                dic['radius_B_min'] = 10**additional_params_B[:,0,p].ravel()
                                dic['radius_B_max'] = 10**additional_params_B[:,2,p].ravel()
                            elif param=='logT':
                                dic['Teff_B'] = 10**additional_params_B[:,1,p].ravel()
                                dic['Teff_B_min'] = 10**additional_params_B[:,0,p].ravel()
                                dic['Teff_B_max'] = 10**additional_params_B[:,2,p].ravel()
                            else:
                                dic[param+'_B'] = additional_params_B[:,1,p].ravel()
                                dic[param+'_B_min'] = additional_params_B[:,0,p].ravel()
                                dic[param+'_B_max'] = additional_params_B[:,2,p].ravel()
                        else:
                            dic['synth_'+param+'_B'] = additional_params_B[:,1,p].ravel()
                            dic['synth_'+param+'_B_min'] = additional_params_B[:,0,p].ravel()
                            dic['synth_'+param+'_B_max'] = additional_params_B[:,2,p].ravel()    

                dic['fit_status'] = code

            else:
                n_masses_x_ages = n_masses*n_ages
                sigma = np.full((n_masses_x_ages,n_filters),np.nan)
                phot1, phot_err1 = np.zeros(phot.shape + (n_try,)), np.zeros(phot.shape + (n_try,))
                for j in range(n_try): phot1[:,:,j], phot_err1[:,:,j] = SampleObject.app_to_abs_mag(app_phot+app_phot_err*np.random.normal(size=phot.shape),self.par+self.par_err*np.random.normal(size=n_objects),app_mag_error=app_phot_err,parallax_error=self.par_err)

                for i in range(n_objects):

                    self.done+=1

                    if more_than_one_iso:
                        index_iso_i = index_of_iso[i]
                        iso_data_for_fit, mass_index_B, use_B, q_eff = iso_data_for_fit_list[index_iso_i], mass_index_B_list[index_iso_i], use_B_list[index_iso_i], q_eff_list[index_iso_i]
                    else:
                        iso_data_for_fit, mass_index_B, use_B, q_eff = iso_data_for_fit_list, mass_index_B_list, use_B_list, q_eff_list

                    iso_data_r = iso_data_for_fit.reshape([n_masses_x_ages,l[2]])

                    if time.perf_counter()-self.t1>60:
                        time_left=(time.perf_counter()-self.t0)/self.done*(len_sample-self.done)
                        print('Program running. Done: '+str(self.done)+'/'+str(len_sample)+' ('+'{:.1f}'.format(self.done/len_sample*100)+ '%). Estimated time left: '+'{:.0f}'.format(time_left)+' s.')
                        self.t1=time.perf_counter()
                    w, = np.where(np.isnan(phot[i,:]) == False)
                    if len(w) == 0:
                        self._print_log('info',f'All magnitudes for star {i} have an error beyond the maximum allowed threshold ({self.ph_cut} mag): age and mass determinations was not possible.')
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])
                        code[i] = 1
                        continue

                    if case == 2:
                        use_i = []
                        for j in range(i_age[i,0], i_age[i,1]+1):
                            new_i=ravel_indices(np.arange(0,n_masses), j, n_ages)
                            use_i.extend(new_i) #ages to be used
                        use_i = np.array(use_i,dtype=int)
                    else: 
                        use_i = np.arange(0,n_masses_x_ages,dtype=int)

                    sigma[use_i,:] = ((iso_data_r[use_i,:]-phot[i,:])/phot_err[i,:])**2
                    acceptable_delta_mag = np.nanmin(np.abs(iso_data_r[use_i,:]-phot[i,:]), axis=0) < 0.2
                    n_acceptable_delta_mag = np.nansum(acceptable_delta_mag)
                    if n_acceptable_delta_mag < 3:
                        if n_acceptable_delta_mag == 0:
                            self._print_log('info',f'All magnitudes for star {i} are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.')
                        else:
                            self._print_log('info',f'Less than three good filters for star {i}: use a less strict error threshold, or consider adopting an age range to have at least a mass estimate.')
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])
                        code[i]=2
                        continue #at least 3 filters needed for the fit


                    if len(use_i)<n_masses_x_ages: 
                        sigma[SampleObject._complement_v(use_i,n_masses_x_ages),:] = np.nan
                    chi2 = nansumwrapper(sigma[:, acceptable_delta_mag], axis=1)/(np.sum(np.isnan(iso_data_r[:, acceptable_delta_mag]) == False, axis=1)-2)

                    min_chi2_ind = np.nanargmin(chi2)
                    condition_1 = np.sort(sigma[min_chi2_ind,acceptable_delta_mag])
                    condition_2 = np.sort(np.abs(iso_data_r[min_chi2_ind,acceptable_delta_mag]-phot[i,acceptable_delta_mag]))

                    chi2_threshold = 500
                    good_solutions_indices, good_solutions_chi2 = [], []

                    if (condition_1[2]<9) | (condition_2[2]<0.1): #the 3rd best sigma < 3 or the 3rd best solution closer than 0.1 mag
                        accept_chi2, = np.where(chi2 < chi2_threshold)
                        if len(accept_chi2) == 0:
                            self._print_log('info',f'No good fits could be found for star {i}. Returning nan.')
                            all_solutions.append({})
                            all_solutions_B.append({})
                            all_maps.append([])
                            hot_p.append([])
                            code[i]=3
                            continue
                        if save_maps: all_maps.append(chi2.reshape([n_masses,n_ages]))
                        chi2_red=chi2[accept_chi2]
                        keep, = np.where(chi2_red<(chi2[min_chi2_ind]+2.3)) #68.3% C.I
                        good_solutions_indices.append(accept_chi2[keep])
                        good_solutions_chi2.append(chi2[accept_chi2[keep]])


                        iso_data_red = iso_data_r[accept_chi2]
                        sigma_red_all = ((iso_data_red[:,acceptable_delta_mag,None]-phot1[i,acceptable_delta_mag,:])/phot_err1[i,acceptable_delta_mag,:])**2
                        chi2_denom = np.sum(np.isnan(iso_data_red[:,acceptable_delta_mag])==False,axis=1)-2
                        chi2 = np.sum(sigma_red_all, axis=1)/chi2_denom[:, None]
                        chi2_mins = np.nanmin(chi2, axis=0)
                        condition_3 = (chi2-chi2_mins[None,:]) < 2.3
                        keep, isol = np.where(condition_3) #68.3% C.I.
                        good_solutions_indices.append(accept_chi2[keep])
                        good_solutions_chi2.append(chi2[condition_3].ravel())

                        good_solutions_indices = np.concatenate(good_solutions_indices)
                        good_solutions_chi2 = np.concatenate(good_solutions_chi2)

                        chi2_min[i] = np.min(good_solutions_chi2)
                        i_ma = np.unravel_index(good_solutions_indices,(n_masses,n_ages))

                        ma0 = np.zeros([n_masses,n_ages])
                        np.add.at(ma0,i_ma,1)
                        i_ma0 = np.where(ma0>(n_try/10))
                        ma = np.zeros([n_masses,n_ages],dtype=bool)
                        ma[i_ma0] = True

                        labeled, _ = label(ma, np.ones((3, 3), dtype=int))
                        labeled_r = labeled.ravel()

                        n_groups = np.max(labeled)

                        com = np.array(center_of_mass(ma,labeled,range(1,n_groups+1)))
                        wn, = np.where((labeled_r == 0) & (ma0.ravel() != 0))
                        wn1, wn2 = np.unravel_index(wn,ma.shape)

                        lab_n = np.zeros(len(wn1))
                        for kk in range(len(wn1)):
                            lab_n[kk] = np.argmin((wn1[kk]-com[:,0])**2+(wn2[kk]-com[:,1])**2)+1
                        labeled[wn1,wn2] = lab_n
                        labeled_r = labeled.ravel()

                        mship = labeled_r[good_solutions_indices]
                        wei_norm = np.sum(1/good_solutions_chi2)
                        group_statistics = np.zeros([n_groups,5])

                        if save_maps:
                            hp = np.zeros([n_masses,n_ages])
                            np.add.at(hp,i_ma,1/good_solutions_chi2)
                            hot_p.append(hp/np.nansum(hp))

                        if has_companion:

                            group_statistics_B = np.zeros([n_groups,5])
                            for jj in range(n_groups):
                                w_gr, = np.where(mship==(jj+1))
                                n_groups0 = len(w_gr)
                                mass_index, age_index = i_ma[0][w_gr], i_ma[1][w_gr]

                                if delta_mag_dict is not None:
                                    if mass_index_B is not None:
                                        mass_index_Bstar = mass_index_B[(mass_index, age_index)]                    
                                elif q is not None:
                                    if mass_index_B is not None:
                                        mass_index_Bstar = mass_index_B[mass_index]

                                if n_groups0 == 1:
                                    group_statistics[jj,0] = iso_mass_log[mass_index]
                                    group_statistics[jj,2] = iso_age_log[age_index]
                                    if mass_index_B is not None:
                                        group_statistics_B[jj,0] = iso_mass_log[mass_index_Bstar]
                                        group_statistics_B[jj,2] = group_statistics[jj,2]
                                else:
                                    group_statistics[jj,0] = np.average(iso_mass_log[mass_index], weights = 1/good_solutions_chi2[w_gr])
                                    group_statistics[jj,1] = np.sqrt(np.average((iso_mass_log[mass_index]-group_statistics[jj,0])**2, weights = 1/good_solutions_chi2[w_gr])*n_groups0/(n_groups0-1))
                                    group_statistics[jj,2] = np.average(iso_age_log[age_index],weights=1/good_solutions_chi2[w_gr])
                                    group_statistics[jj,3] = np.sqrt(np.average((iso_age_log[age_index]-group_statistics[jj,2])**2, weights = 1/good_solutions_chi2[w_gr])*n_groups0/(n_groups0-1))
                                    if mass_index_B is not None:
                                        group_statistics_B[jj,0] = np.average(iso_mass_log[mass_index_Bstar], weights = 1/good_solutions_chi2[w_gr])
                                        group_statistics_B[jj,1] = np.sqrt(np.average((iso_mass_log[mass_index_Bstar]-group_statistics_B[jj,0])**2, weights = 1/good_solutions_chi2[w_gr])*n_groups0/(n_groups0-1))
                                        group_statistics_B[jj,2] = group_statistics[jj,2]
                                        group_statistics_B[jj,3] = group_statistics[jj,3]

                                group_statistics[jj,4] = np.sum(1/good_solutions_chi2[w_gr])/wei_norm
                                group_statistics_B[jj,4] = group_statistics[jj,4]
                            group_statistics_B = SampleObject._merge_solutions(group_statistics_B)

                        else:
                            for jj in range(n_groups):
                                w_gr, = np.where(mship==(jj+1))
                                n_groups0 = len(w_gr)
                                mass_index, age_index = i_ma[0][w_gr], i_ma[1][w_gr]
                                if n_groups0 == 1:
                                    group_statistics[jj,0] = iso_mass_log[mass_index]
                                    group_statistics[jj,2] = iso_age_log[age_index]
                                else:
                                    group_statistics[jj,0] = np.average(iso_mass_log[mass_index], weights = 1/good_solutions_chi2[w_gr])
                                    group_statistics[jj,1] = np.sqrt(np.average((iso_mass_log[mass_index]-group_statistics[jj,0])**2, weights = 1/good_solutions_chi2[w_gr])*n_groups0/(n_groups0-1))
                                    group_statistics[jj,2] = np.average(iso_age_log[age_index],weights=1/good_solutions_chi2[w_gr])
                                    group_statistics[jj,3] = np.sqrt(np.average((iso_age_log[age_index]-group_statistics[jj,2])**2, weights = 1/good_solutions_chi2[w_gr])*n_groups0/(n_groups0-1))

                                group_statistics[jj,4] = np.sum(1/good_solutions_chi2[w_gr])/wei_norm

                        group_statistics = SampleObject._merge_solutions(group_statistics)                        

                        n_groups = len(group_statistics)
                        i_s = np.argmax(group_statistics[:,4])

                        ival = np.array([[group_statistics[i_s,0]-group_statistics[i_s,1],group_statistics[i_s,0],group_statistics[i_s,0]+group_statistics[i_s,1]],
                                         [group_statistics[i_s,2]-group_statistics[i_s,3],group_statistics[i_s,2],group_statistics[i_s,2]+group_statistics[i_s,3]]])

                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            for p in range(n_params):
                                additional_params[i,1,p],additional_params[i,0,p],additional_params[i,2,p]=interp_params[p](ival[0,1],ival[1,1]),np.nanmin(interp_params[p](ival[0,:],ival[1,:])),np.nanmax(interp_params[p](ival[0,:],ival[1,:]))

                        m_fit[i], m_min[i], m_max[i] = 10**group_statistics[i_s,0], 10**(group_statistics[i_s,0]-group_statistics[i_s,1]), 10**(group_statistics[i_s,0]+group_statistics[i_s,1])
                        a_fit[i], a_min[i], a_max[i] = 10**group_statistics[i_s,2], 10**(group_statistics[i_s,2]-group_statistics[i_s,3]), 10**(group_statistics[i_s,2]+group_statistics[i_s,3])

                        code[i] = 0

                        if has_companion:
                            n_groups_B = len(group_statistics_B)
                            i_s_B = np.argmax(group_statistics_B[:,4])

                            ival_B = np.array([[group_statistics_B[i_s_B,0]-group_statistics_B[i_s_B,1],group_statistics_B[i_s_B,0],group_statistics_B[i_s_B,0]+group_statistics_B[i_s_B,1]],
                                             [group_statistics_B[i_s_B,2]-group_statistics_B[i_s_B,3],group_statistics_B[i_s_B,2],group_statistics_B[i_s_B,2]+group_statistics_B[i_s_B,3]]])

                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                if mass_index_B is not None:
                                    for p in range(n_params):
                                        additional_params_B[i,1,p],additional_params_B[i,0,p],additional_params_B[i,2,p]=interp_params[p](ival_B[0,1],ival_B[1,1]),np.nanmin(interp_params[p](ival_B[0,:],ival_B[1,:])),np.nanmax(interp_params[p](ival_B[0,:],ival_B[1,:]))

                            if delta_mag_dict is not None:
                                if mass_index_B is not None:
                                    m_B_fit[i], m_B_min[i], m_B_max[i] = 10**group_statistics_B[i_s_B,0], 10**(group_statistics_B[i_s_B,0]-group_statistics_B[i_s_B,1]), 10**(group_statistics_B[i_s_B,0]+group_statistics_B[i_s_B,1])
                                else:
                                    m_B_fit[i], m_B_min[i], m_B_max[i] = 0., 0., 0.


                        if delta_mag_dict is not None:
                            if mass_index_B is not None:
                                mass_index_Bstar = mass_index_B[(mass_index, age_index)]                    
                        elif q is not None:
                            if mass_index_B is not None:
                                mass_index_Bstar = mass_index_B[mass_index]


                        if n_groups > 1:
                            self._print_log('info',f'More than one region of the (mass,age) space is possible for star {i}.')
                            self._print_log('info',f'Possible solutions for star {i}:')
                            m_all = np.zeros([n_groups,3])
                            a_all = np.zeros([n_groups,3])

                            additional_params_all = np.zeros([n_groups,3,n_params])

                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')

                                for jj in range(n_groups):
                                    m_all[jj,:] = [10**group_statistics[jj,0],10**(group_statistics[jj,0]-group_statistics[jj,1]),10**(group_statistics[jj,0]+group_statistics[jj,1])]
                                    a_all[jj,:] = [10**group_statistics[jj,2],10**(group_statistics[jj,2]-group_statistics[jj,3]),10**(group_statistics[jj,2]+group_statistics[jj,3])]

                                    ival = np.array([[group_statistics[jj,0]-group_statistics[jj,1],group_statistics[jj,0],group_statistics[jj,0]+group_statistics[jj,1]],
                                                  [group_statistics[jj,2]-group_statistics[jj,3],group_statistics[jj,2],group_statistics[jj,2]+group_statistics[jj,3]]])

                                    for p in range(n_params):
                                        interp_f = interp_params[p]
                                        additional_params_all[jj,:,p] = [float(interp_f(ival[0,1],ival[1,1])),np.nanmin(interp_f(ival[0,:],ival[1,:])),np.nanmax(interp_f(ival[0,:],ival[1,:]))]

                                    Mi, Mip, Mim = '{:.3f}'.format(m_all[jj,0]), '{:.3f}'.format(m_all[jj,1]), '{:.3f}'.format(m_all[jj,2])
                                    Ai, Aip, Aim = '{:.1f}'.format(a_all[jj,0]), '{:.1f}'.format(a_all[jj,1]), '{:.1f}'.format(a_all[jj,2])
                                    self._print_log('info',f'M={Mi} ({Mip}, {Mim}) M_sun, t={Ai} ({Aip}, {Aim}) Myr (prob={group_statistics[jj,4]:.2f}).')

                                if has_companion:

                                    additional_params_B_all = np.zeros([n_groups_B,3,n_params])

                                    m_all_B = np.zeros([n_groups_B,3])
                                    a_all_B = np.zeros([n_groups_B,3])
                                    for jj in range(n_groups_B):
                                        m_all_B[jj,:] = [10**group_statistics_B[jj,0], 10**(group_statistics_B[jj,0]-group_statistics_B[jj,1]), 10**(group_statistics_B[jj,0]+group_statistics_B[jj,1])]
                                        a_all_B[jj,:] = [10**group_statistics_B[jj,2], 10**(group_statistics_B[jj,2]-group_statistics_B[jj,3]), 10**(group_statistics_B[jj,2]+group_statistics_B[jj,3])]

                                        ival_B = np.array([[group_statistics_B[jj,0]-group_statistics_B[jj,1], 
                                                            group_statistics_B[jj,0],
                                                            group_statistics_B[jj,0]+group_statistics_B[jj,1]],
                                                           [group_statistics_B[jj,2]-group_statistics_B[jj,3], 
                                                            group_statistics_B[jj,2],
                                                            group_statistics_B[jj,2]+group_statistics_B[jj,3]]])

                                        for p in range(n_params):
                                            interp_f = interp_params[p]
                                            additional_params_B_all[jj,:,p] = [float(interp_f(ival_B[0,1],ival_B[1,1])),np.nanmin(interp_f(ival_B[0,:],ival_B[1,:])),np.nanmax(interp_f(ival_B[0,:],ival_B[1,:]))]

                                        Mi_B, Mip_B, Mim_B = '{:.3f}'.format(m_all_B[jj,0]), '{:.3f}'.format(m_all_B[jj,1]), '{:.3f}'.format(m_all_B[jj,2])
                                        self._print_log('info',f'M_B={Mi_B} ({Mip_B}, {Mim_B}) M_sun (prob={group_statistics_B[jj,4]:.2f}).')


                            dic={'mass':m_all,'age':a_all}

                            for p,param in enumerate(output_columns):
                                if param.startswith('log'):
                                    if param=='logR':
                                        dic['radius'] = 10**additional_params_all[:,:,p]
                                    elif param=='logT':
                                        dic['Teff'] = 10**additional_params_all[:,:,p]
                                    else:
                                        dic[param] = additional_params_all[:,:,p]
                                else:
                                    dic['synth_'+param] = additional_params_all[:,:,p]

                            dic['prob'] = group_statistics[:,4].ravel()

                            all_solutions.append(dic)

                            if has_companion:
                                dic_B = {'mass':m_all_B,'age':a_all_B}

                                for p,param in enumerate(output_columns):
                                    if param.startswith('log'):
                                        if param=='logR':
                                            dic_B['radius'] = 10**additional_params_B_all[:,:,p]
                                        elif param=='logT':
                                            dic_B['Teff'] = 10**additional_params_B_all[:,:,p]
                                        else:
                                            dic_B[param] = additional_params_B_all[:,:,p]
                                    else:
                                        dic_B['synth_'+param] = additional_params_B_all[:,:,p]

                                dic_B['prob'] = group_statistics_B[:,4].ravel()

                                all_solutions_B.append(dic_B)

                        else:
                            dic = {'mass':np.array([m_fit[i],m_min[i],m_max[i]]),
                                   'age':np.array([a_fit[i],a_min[i],a_max[i]])}

                            for p,param in enumerate(output_columns):
                                if param.startswith('log'):
                                    if param=='logR':
                                        dic['radius'] = np.array([10**additional_params[i,1,p],10**additional_params[i,0,p],10**additional_params[i,2,p]]).ravel()
                                    elif param=='logT':
                                        dic['Teff'] = np.array([10**additional_params[i,1,p],10**additional_params[i,0,p],10**additional_params[i,2,p]]).ravel()
                                    else:
                                        dic[param] = np.array([additional_params[i,1,p],additional_params[i,0,p],additional_params[i,2,p]]).ravel()
                                else:
                                    dic['synth_'+param] = np.array([additional_params[i,1,p],additional_params[i,0,p],additional_params[i,2,p]]).ravel()
                            dic['prob'] = group_statistics[:,4].ravel()
                            all_solutions.append(dic)

                            if has_companion:

                                if q is not None:
                                    dic_B = {'mass':np.array([m_fit[i]*q_eff,m_min[i]*q_eff,m_max[i]*q_eff]),
                                           'age':np.array([a_fit[i],a_min[i],a_max[i]])}
                                else:
                                    dic_B = {'mass':np.array([m_B_fit[i],m_B_min[i],m_B_max[i]]),
                                           'age':np.array([a_fit[i],a_min[i],a_max[i]])}

                                for p,param in enumerate(output_columns):
                                    if param.startswith('log'):
                                        if param=='logR':
                                            dic_B['radius'] = np.array([10**additional_params_B[i,1,p],10**additional_params_B[i,0,p],10**additional_params_B[i,2,p]]).ravel()
                                        elif param=='logT':
                                            dic_B['Teff'] = np.array([10**additional_params_B[i,1,p],10**additional_params_B[i,0,p],10**additional_params_B[i,2,p]]).ravel()
                                        else:
                                            dic_B[param] = np.array([additional_params_B[i,1,p],additional_params_B[i,0,p],additional_params_B[i,2,p]]).ravel()
                                    else:
                                        dic_B['synth_'+param] = np.array([additional_params_B[i,1,p],additional_params_B[i,0,p],additional_params_B[i,2,p]]).ravel()
                                dic_B['prob'] = group_statistics_B[:,4].ravel()
                                all_solutions_B.append(dic_B)

                    else:
                        code[i]=4
                        all_solutions.append({})
                        all_solutions_B.append({})
                        all_maps.append([])
                        hot_p.append([])


                dic={'age':a_fit, 'age_min':a_min, 'age_max':a_max,
                     'mass':m_fit, 'mass_min':m_min, 'mass_max':m_max,
                     'ebv':self.ebv, 'ebv_err':ebv_err, 'chi2_min':chi2_min, 'chi2_maps':all_maps, 'weight_maps':hot_p
                     }

                for p,param in enumerate(output_columns):
                    if param.startswith('log'):
                        if param=='logR':
                            dic['radius'] = 10**additional_params[:,1,p].ravel()
                            dic['radius_min'] = 10**additional_params[:,0,p].ravel()
                            dic['radius_max'] = 10**additional_params[:,2,p].ravel()
                        elif param=='logT':
                            dic['Teff'] = 10**additional_params[:,1,p].ravel()
                            dic['Teff_min'] = 10**additional_params[:,0,p].ravel()
                            dic['Teff_max'] = 10**additional_params[:,2,p].ravel()
                        else:
                            dic[param] = additional_params[:,1,p].ravel()
                            dic[param+'_min'] = additional_params[:,0,p].ravel()
                            dic[param+'_max'] = additional_params[:,2,p].ravel()
                    else:
                        dic['synth_'+param] = additional_params[:,1,p].ravel()
                        dic['synth_'+param+'_min'] = additional_params[:,0,p].ravel()
                        dic['synth_'+param+'_max'] = additional_params[:,2,p].ravel()
                dic['all_solutions'] = all_solutions
                if has_companion:
                    dic['all_solutions_B'] = all_solutions_B

                if q is not None:

                    if more_than_one_iso:
                        q_eff_ordered = q_eff_list[index_of_iso]
                    else:
                        q_eff_ordered = q_eff_list
                    m_B_fit = dic['mass']*q_eff_ordered
                    m_B_min = dic['mass_min']*q_eff_ordered
                    m_B_max = dic['mass_max']*q_eff_ordered

                if has_companion:
                    dic['mass_B'] = m_B_fit
                    dic['mass_B_min'] = m_B_min
                    dic['mass_B_max'] = m_B_max

                    for p,param in enumerate(output_columns):
                        if param.startswith('log'):
                            if param=='logR':
                                dic['radius_B'] = 10**additional_params_B[:,1,p].ravel()
                                dic['radius_B_min'] = 10**additional_params_B[:,0,p].ravel()
                                dic['radius_B_max'] = 10**additional_params_B[:,2,p].ravel()
                            elif param=='logT':
                                dic['Teff_B'] = 10**additional_params_B[:,1,p].ravel()
                                dic['Teff_B_min'] = 10**additional_params_B[:,0,p].ravel()
                                dic['Teff_B_max'] = 10**additional_params_B[:,2,p].ravel()
                            else:
                                dic[param+'_B'] = additional_params_B[:,1,p].ravel()
                                dic[param+'_B_min'] = additional_params_B[:,0,p].ravel()
                                dic[param+'_B_max'] = additional_params_B[:,2,p].ravel()
                        else:
                            dic['synth_'+param+'_B'] = additional_params_B[:,1,p].ravel()
                            dic['synth_'+param+'_B_min'] = additional_params_B[:,0,p].ravel()
                            dic['synth_'+param+'_B_max'] = additional_params_B[:,2,p].ravel()

                dic['fit_status'] = code


        if m_unit.lower() == 'm_jup':
            dic['mass'] *= M_sun.value/M_jup.value
            dic['mass_min'] *= M_sun.value/M_jup.value
            dic['mass_max'] *= M_sun.value/M_jup.value
            dic['radius'] *= R_sun.value/R_jup.value
            dic['radius_min'] *= R_sun.value/R_jup.value
            dic['radius_max'] *= R_sun.value/R_jup.value
            if 'mass_B' in dic.keys():
                dic['mass_B'] *= M_sun.value/M_jup.value
                dic['mass_B_min'] *= M_sun.value/M_jup.value
                dic['mass_B_max'] *= M_sun.value/M_jup.value
                dic['radius_B'] *= R_sun.value/R_jup.value
                dic['radius_min'] *= R_sun.value/R_jup.value
                dic['radius_max'] *= R_sun.value/R_jup.value
            if 'all_solutions' in dic.keys():
                for i in range(len(dic['all_solutions'])):
                    if 'mass' in dic['all_solutions'][i].keys():
                        dic['all_solutions'][i]['mass']*=M_sun.value/M_jup.value
            r_unit = 'r_jup'
        else: r_unit = 'r_sun'

        if save_maps == False:
            del dic['chi2_maps']
            del dic['weight_maps']

        dic['feh'] = np.full(n_objects,th_model.feh)
        dic['he'] = np.full(n_objects,th_model.he)
        dic['afe'] = np.full(n_objects,th_model.afe)
        dic['v_vcrit'] = np.full(n_objects,th_model.v_vcrit)
        dic['fspot'] = np.full(n_objects,th_model.fspot)
        dic['B'] = np.full(n_objects,th_model.B)
        if self.verbose > 0:
            dic['sample_name'] = self.__sample_name
            dic['path'] = self.__sample_path
        l = list(self.GaiaID[self.GaiaID.columns[0].name])
        dic['objects'] = np.array(l)
        dic['exec_command'] = np.array(list([repr(th_model)])*n_objects, dtype=object)
        dic['fitting_mode'] = case
        dic['model_grid'] = np.array(list([th_model.__dict__['model_grid']])*n_objects, dtype=object)
        dic['is_true_fit'] = True
        dic['units'] = {'mass':m_unit, 'age':'Myr', 'Teff': 'K', 'radius': r_unit}

        return FitParams(dic)

    def get_params(self, model_version, **kwargs):

        """
        Estimates age, mass, radius, Teff and logg for each object in the sample by comparison with isochrone grids.
            Input:
            - model_version: string, required. Selected isochrone grid model. Use ModelHandler.available() for further information on available models.
            - mass_range: list, optional. A two-element list with minimum and maximum mass within the grid (M_sun). Default: not set; the mass_range is the intersection between a rough mass estimate based on G and K magnitudes and the dynamical range of the model itself.
            - age_range: list or numpy array, optional. It can be either:
                    1) a two-element list with minimum and maximum age to consider for the whole sample (Myr);
                    2) a 1D numpy array, so that the i-th age (Myr) is used as fixed age for the i-th star;
                    3) a 2D numpy array with 2 columns. The i-th row defines (lower_age,upper_age) range in which one or more solutions are found for the i-th star.
                    4) a 2D numpy array with 3 columns. The i-th row is used as (mean_age,lower_age,upper_age) for the i-th star; mean_age is used as in case 2), and [lower_age, upper_age] are used as in case 3).
              Default: [1,1000]
            - n_steps: list, optional. Number of (mass, age) steps of the interpolated grid. Default: [1000,500].
            - n_try: int, optional. Number of Monte Carlo iteractions for each star. Default: 1000.
            - feh: float or numpy array, optional. Selects [Fe/H] of the isochrone set. If numpy array, the i-th element refers to the i-th star. Default: 0.00 (solar metallicity).
            - he: float, optional. Selects helium fraction Y of the isochrone set. If numpy array, the i-th element refers to the i-th star. Default: solar Y (different for each model).
            - afe: float, optional. Selects alpha enhancement [a/Fe] of the isochrone set. If numpy array, the i-th element refers to the i-th star. Default: 0.00.
            - v_vcrit: float, optional. Selects rotational velocity of the isochrone set. If numpy array, the i-th element refers to the i-th star. Default: 0.00 (non-rotating).
            - fspot: float, optional. Selects fraction of stellar surface covered by star spots. If numpy array, the i-th element refers to the i-th star. Default: 0.00.
            - B: int, optional. Set to 1 to turn on the magnetic field (only for Dartmouth models). If numpy array, the i-th element refers to the i-th star. Default: 0.
            - secondary_q: float or list or numpy array, optional. Secondary-to-primary mass ratios (i.e., under the assumption that the targets are unresolved binaries). Zero values correspond to single stars. If list or numpy array, the i-th element refers to the i-th star. Default: None (=single star model).
            - secondary_contrast: dict, optional. Secondary-to-primary contrast in the band specified by the key name (i.e., under the assumption that the targets are unresolved binaries). np.inf values correspond to single stars.
              Only one key must be provided. The corresponding value must be a float, list or numpy array.
              If list or numpy array, the i-th element refers to the i-th star. Default: None (=single star model).
            - fill_value: array-like or (array-like, array_like) or “extrapolate”, optional. How the interpolation over mass deals with values outside the original range. Default: np.nan. See scipy.interpolate.interp1d for details.
            - ph_cut: float, optional. Maximum  allowed photometric uncertainty [mag]. Data with a larger error will be ignored. Default: 0.2.
            - m_unit: string, optional. Unit of measurement of the resulting mass. Choose either 'm_sun' or 'm_jup'. Default: 'm_sun'.
            - save_maps: bool, optional. Set to True to save chi2 and weight maps for each star. Not recommended if n_star is big (let's say, >1000). Default: False.
            - logger: logger, optional. A logger returned by SampleObject._setup_custom_logger(). Default: self.__logger.
            Output:
            - a madys.FitParams object.
        """

        def renew_kwargs(kwa, w):
            kw = copy.deepcopy(kwa)

            if 'secondary_q' in kw:
                if hasattr(kw['secondary_q'], '__len__'):
                    kw['secondary_q'] = kw['secondary_q'][w]

            if 'secondary_contrast' in kw:
                key, val = list(kw['secondary_contrast'].keys())[0], np.array(list(kw['secondary_contrast'].values())).ravel()
                if len(val) > 1:
                    val = val[w]
                    kw['secondary_contrast'] = {key: val}

            if 'age_range' not in kw: 
                return kw
            else:
                age_range = kw['age_range']
                if isinstance(age_range, np.ndarray):
                    if len(age_range.shape) == 1:
                        kw['age_range'] = age_range[w]
                    elif len(age_range[0]) == 2:
                        kw['age_range'] = age_range[w]
                    elif len(age_range[0]) == 3:
                        kw['age_range'] = age_range[w]
                return kw

        for arg in ['secondary_q']:
            if arg in kwargs:
                if hasattr(kwargs[arg], '__len__'):
                    kwargs[arg] = np.array(kwargs[arg])        

        original_kwargs = copy.deepcopy(kwargs)
        p = np.array(['feh','he','afe','v_vcrit','fspot','B'])
        k = np.sum([i in kwargs for i in p])

        skip = False
        if k > 0:
            cust = np.zeros(6)
            for i in range(6):
                try:
                    cust[i] = isinstance(kwargs[p[i]], np.ndarray)
                    cust[i] = (len(kwargs[p[i]])>1)
                    if (len(kwargs[p[i]])!=len(self)): 
                        raise ValueError('The number of '+p[i]+' ('+str(len(kwargs[p[i]]))+') is not equal to the number of stars ('+str(len(self))+').')
                except (KeyError, TypeError): 
                    continue
            if np.sum(cust) < 0.1: 
                skip = True
        else:
            skip = True

        if skip:
            dic = {}
            for kw in p:
                if kw in kwargs: dic[kw] = kwargs[kw]
            model_params = ModelHandler._find_match(model_version, dic, 
                                                    list(stored_data['complete_model_list'].keys()))
        else:
            model_params = []
            for i in range(len(self)):
                dic = {}
                for k in p:
                    if k in kwargs: 
                        try:
                            dic[k] = kwargs[k][i]
                        except TypeError:
                            dic[k] = kwargs[k]

                model_params1 = ModelHandler._find_match(model_version, dic,
                                                         list(stored_data['complete_model_list'].keys()))
                sol1 = ModelHandler._version_to_grid(model_version, model_params1)
                model_params.append(model_params1)

        ModelHandler._find_model_grid(model_version, model_params)

        try:
            model_p = ModelHandler._available_parameters(model_version)
        except ValueError as e:
            msg = """You decided not to download any grid for model_version """+model_version+""".
            However, the relative folder is empty, so MADYS does not have any model to compare data with.
            Re-run the program, downloading at least one model when prompted.
            Program ended."""
            e.args = (msg,)
            raise

        n_st = len(self)

        self.t0 = time.perf_counter()
        self.t1 = time.perf_counter()
        self.done = 0

        if skip == False:
            comb = np.zeros([len(self), 6])
            w, = np.where(cust == 1)

            for j in range(n_st):
                kw_i = {}
                for k in p[w]: 
                    kw_i[k] = kwargs[k][j]
                bf_params = ModelHandler._find_match(model_version, kw_i, 
                                                     list(stored_data['local_model_list'].keys()),
                                                     approximate=True)
                for i in w: 
                    comb[j,i] = bf_params[p[i]]

            comb_u = np.vstack(list({tuple(row) for row in comb}))

            if len(comb_u.shape)==1:
                for i in w:
                    kwargs[p[i]]=comb_u[i]
                res = self._get_agemass(model_version, **kwargs)
            else:
                for j in range(len(comb_u)):
                    w_an, = np.where(np.sum(comb_u[j] == comb, axis=1) == 6)
                    for i in w:
                        kwargs[p[i]] = comb[w_an[0],i]
                    kwargs2 = renew_kwargs(kwargs,w_an)
                    res_i = self[w_an]._get_agemass(model_version, 
                                                    n_tot=len(self),
                                                    **kwargs2)
                    self.done += len(w_an)
                    if j == 0: 
                        res = res_i.empty_like(n_st)
                    res[w_an] = res_i
        else:
            kwargs = copy.deepcopy(kwargs)
            for kw in p:
                if kw in kwargs: 
                    try:
                        kwargs[kw] = model_params[kw]
                    except KeyError:
                        print('Input parameter {0} is not supported by model {1} and was not used.'.format(kw,model_version))
            res = self._get_agemass(model_version, **kwargs)

        print('Execution ended. Elapsed time: '+'{:.0f}'.format(time.perf_counter()-self.t0)+' s.')

        del self.t0, self.t1, self.done

        for col_name in ['id', 'ID', 'source_id', 'object_name']:
            try:
                original_names = np.array(self.ID[col_name], dtype=str)
                found = True
            except KeyError:
                continue
            if found: 
                break
        if found == False: 
            original_names = np.array(self.ID, dtype=str)

        res.input_parameters = original_kwargs
        res.original_IDs = original_names

        if self.verbose == 3:
            filename = os.path.join(self.path, str(self.__sample_name+'_ages_'+model_version+'.txt'))
            res.to_file(filename, check_verbose=True)
            self._print_log('info',f'Age determination ended. Results saved to {filename}.')
            logging.shutdown()
        elif self.verbose == 2:
            self._print_log('info','Age determination ended. Results not saved to any file because "verbose" is set to 2.')
            logging.shutdown()

        return res
    


    ############################################# plotting functions #########################################

    @staticmethod
    def _axis_range(col_name, col_phot):
        try:
            len(col_phot)
            cmin = np.min(col_phot) - 0.1
            cmax = np.min([70, max(col_phot)]) + 0.1
        except TypeError:
            cmin = col_phot-0.1
            cmax = np.min([70, col_phot]) + 0.1

        dic1={'G':[max(15,cmax),min(1,cmin)], 'Gbp':[max(15,cmax),min(1,cmin)], 'Grp':[max(15,cmax),min(1,cmin)],
            'J':[max(10,cmax),min(0,cmin)], 'H':[max(10,cmax),min(0,cmin)], 'K':[max(10,cmax),min(0,cmin)],
            'W1':[max(10,cmax),min(0,cmin)], 'W2':[max(10,cmax),min(0,cmin)], 'W3':[max(10,cmax),min(0,cmin)],
            'W4':[max(10,cmax),min(0,cmin)], 'SPH_K1':[max(19,cmax),min(6,cmin)], 'SPH_K2':[max(19,cmax),min(6,cmin)],
            'G-J':[min(0,cmin),max(5,cmax)],
            'G-H':[min(0,cmin),max(5,cmax)], 'G-K':[min(0,cmin),max(5,cmax)],
            'G-W1':[min(0,cmin),max(6,cmax)], 'G-W2':[min(0,cmin),max(6,cmax)],
            'G-W3':[min(0,cmin),max(10,cmax)], 'G-W4':[min(0,cmin),max(12,cmax)],
            'J-H':[min(0,cmin),max(1,cmax)], 'J-K':[min(0,cmin),max(1.5,cmax)],
            'H-K':[min(0,cmin),max(0.5,cmax)], 'Gbp-Grp':[min(0,cmin),max(5,cmax)],
            }

        try:
            xx=dic1[col_name]
        except KeyError:
            if '-' in col_name:
                if cmax-cmin > 5: 
                    x = [cmin, cmax]
                else: 
                    xx = np.nanmean(col_phot) + [-3,3]
            else:
                if cmax-cmin > 5: 
                    x = [cmax, cmin]
                else: 
                    xx = np.nanmean(col_phot) + [3,-3]

        return xx

    def CMD(self, col, mag, model_version, ids=None, **kwargs):

        """
        Draws a color-magnitude diagram (CMD) containing both the measured photometry and a set of theoretical isochrones.
        It's a combination of IsochroneGrid.plot_isochrones() and SampleObject.plot_photometry().
            Input:
            - col: string, required. Quantity to be plotted along the x axis (e.g.: 'G' or 'G-K')
            - mag: string, required. Quantity to be plotted along the y axis (e.g.: 'G' or 'G-K')
            - model_version: string, required. Selected model_version. Use ModelHandler.available() for further information on the available models.
            - plot_ages: numpy array or bool, optional. It can be either:
                    - a numpy array containing the ages (in Myr) of the isochrones to be plotted;
                    - False, not to plot any isochrone.
              Default: [1,3,5,10,20,30,100,200,500,1000].
            - plot_masses: numpy array or bool, optional. It can be either:
                    - a numpy array containing the masses (in M_sun) of the tracks to be plotted.
                    - False, not to plot any track.
              Default: [0.1,0.3,0.5,0.7,0.85,1.0,1.3,2].
            - all valid keywords of a IsochroneGrid object, optional.
            - ids: list or numpy array of integers, optional. Array of indices, selects the subset of input data to be drawn.
            - xlim: list, optional. A two-element list with minimum and maximum value for the x axis.
            - ylim: list, optional. A two-element list with minimum and maximum value for the y axis.
            - groups: list or numpy array of integers, optional. Draws different groups of stars in different colors. The i-th element is a number, indicating to which group the i-th star belongs. Default: None.
            - group_list: list or numpy array of strings, optional. Names of the groups defined by the 'groups' keyword. No. of elements must match the no. of groups. Default: None.
            - label_points: bool, optional. Draws a label next to each point, specifying its row index. Default: True.
            - figsize: tuple or list, optional. Figure size. Default: (16,12).
            - tofile: bool or string, optional. If True, saves the output to as .png image. To change the file name, provide a string as full path to the output file. Default: False.
            - close: bool, optional. If False, it does not close the figure and returns the fig and ax objects from plt.subplots(). Useful to manually overplot something on top of the CMD. Default: False.
            Output: (only if close = True)
            - fig: a matplotlib.figure.Figure instance.
            - ax: a matplotlib.axes.Axes instance.
        """
        
        model_params={}
        for i in ['feh','afe','v_vcrit','he','fspot','B']:
            if i in kwargs: model_params[i]=kwargs[i]
        ModelHandler._find_model_grid(model_version,model_params)
        
        model_params2 = ModelHandler._find_match(model_version,model_params,list(stored_data['complete_model_list'].keys()))
        kwargs2 = copy.deepcopy(kwargs)
        for key in model_params2.keys(): 
            if kwargs2[key] != model_params2[key]:
                print('Value {0} not available for input parameter {1}. Using value {2} instead.'.format(kwargs[key],key,model_params2[key]))
                kwargs2[key] = model_params2[key]        

        figsize = kwargs['figsize'] if 'figsize' in kwargs else (16,12)

        fig, ax = plt.subplots(figsize=figsize)
        IsochroneGrid.plot_isochrones(col,mag,model_version,ax,**kwargs2)

        errors = kwargs['errors'] if 'errors' in kwargs else None
        ids = kwargs['ids'] if 'ids' in kwargs else None
        tofile = kwargs['tofile'] if 'tofile' in kwargs else False
        close = kwargs['close'] if 'close' in kwargs else True

        x, y = SampleObject.plot_photometry(col,mag,ax,self,errors=None,ids=None,return_points=True,**kwargs2)

        #axes ranges
        xlim = kwargs['xlim'] if 'xlim' in kwargs else SampleObject._axis_range(col,x)
        ylim = kwargs['ylim'] if 'ylim' in kwargs else SampleObject._axis_range(mag,y)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xlabel(col, fontsize=18)
        ax.set_ylabel(mag, fontsize=18)
        ax.legend()

        if close == False:
            return fig, ax
        else:
            if tofile==False:
                plt.show()
            elif tofile==True:
                try:
                    img_file=os.path.join(self.path,self.__sample_name+'_'+col+'_'+mag+'_'+model_version+'.png')
                except AttributeError: raise ValueError("No working path was defined in the analysis. Please pass an explicit path of the output file as argument to 'tofile'.")
                plt.savefig(img_file)
                plt.close(fig)
            else:
                plt.savefig(tofile)
                plt.close(fig)

            return None


    @staticmethod
    def plot_photometry(col, mag, ax, data,
                        errors=None, ids=None,
                        **kwargs):

        """
        Similar to CMD, but draws only photometric data over an existing figure.
            Input:
            - col: string, required. Quantity to be plotted along the x axis (e.g.: 'G' or 'G-K').
            - mag: string, required. Quantity to be plotted along the y axis (e.g.: 'G' or 'G-K').
            - ax: AxesSubplot, required. Axis object where the isochrones will be drawn upon.
            - data: SampleObject instance or numpy array, required. It can be either:
                    - a MADYS instance;
                    - a 2D numpy array; the first row will be plotted on the 'col' axis, the second row on the 'mag' axis.
            - errors: numpy array, optional. Use it only if data is a numpy array. Contains the errors associated to data (first row: errors on 'col', second row: errors on 'mag').
            - ids: list or numpy array of integers, optional. Array of indices, selects the subset of input data to be drawn.
            - groups: list or numpy array of integers, optional. Draws different groups of stars in different colors. The i-th element is a number, indicating to which group the i-th star belongs. Default: None.
            - group_list: list or numpy array of strings, optional. Names of the groups defined by the 'groups' keyword. No. of elements must match the no. of groups. Default: None.
            - label_points: bool, optional. Draws a label next to each point, specifying its row index. Default: True.
            - s: int, optional. Size of the drawn circles. It follows the scaling conventions of plt.scatter(). Default: 50.
            - alpha: float, optional. Opacity (between 0 and 1) of the drawn circles. Default: 1 (completely opaque).
            - return_points: bool, optional. If True, returns the plotted points as arrays. Default: False.
            Output:
            - x: numpy array. x coordinates of the plotted points. Only returned if return_points=True.
            - y: numpy array. y coordinates of the plotted points. Only returned if return_points=True.
        """        

        if 'SampleObject' in str(type(data)):
            self = data
            if '-' in col:
                col_n = col.split('-')
                c1, = np.where(self.filters==col_n[0])
                c2, = np.where(self.filters==col_n[1])
                col1, col1_err = self.abs_phot[:,c1], self.app_phot_err[:,c1]
                col2, col2_err = self.abs_phot[:,c2], self.app_phot_err[:,c2]
                col_data = col1 - col2
                if type(errors) == type(False):
                    if errors == False:
                        col_err = None
                else: col_err = np.sqrt(col1_err**2 + col2_err**2)
            else:
                c1, = np.where(self.filters==col)
                col_data, col_err = self.abs_phot[:,c1], self.abs_phot_err[:,c1]
                if type(errors) == type(False):
                    if errors == False:
                        col_err = None
            if '-' in mag:
                mag_n = mag.split('-')
                m1, = np.where(self.filters == mag_n[0])
                m2, = np.where(self.filters == mag_n[1])
                mag1, mag1_err = self.abs_phot[:,m1], self.app_phot_err[:,m1]
                mag2, mag2_err = self.abs_phot[:,m2], self.app_phot_err[:,m2]
                mag_data = mag1 - mag2
                if type(errors) == type(False):
                    if errors == False:
                        mag_err = None
                else: mag_err = np.sqrt(mag1_err**2 + mag2_err**2)
            else:
                m1, = np.where(self.filters == mag)
                mag_data, mag_err = self.abs_phot[:,m1], self.abs_phot_err[:,m1]
                if type(errors) == type(False):
                    if errors == False:
                        mag_err = None
            if col_err is not None:
                col_err = col_err.ravel()
            if mag_err is not None: 
                mag_err = mag_err.ravel()
        else:
            col_data = data[0,:]
            mag_data = data[1,:]
            if errors is not None:
                col_err = errors[0,:]
                mag_err = errors[1,:]
                col_err = col_err.ravel()
                mag_err = mag_err.ravel()
            else:
                col_err = None
                mag_err = None

        col_data = col_data.ravel()
        mag_data = mag_data.ravel()

        if ids is not None:
            col_data = col_data[ids]
            mag_data = mag_data[ids]
            col_err = col_err[ids]
            mag_err = mag_err[ids]

        x = col_data
        y = mag_data
        x_axis = col
        y_axis = mag

        label_points = kwargs['label_points'] if 'label_points' in kwargs else True
        groups = kwargs['groups'] if 'groups' in kwargs else None
        group_names = kwargs['group_names'] if 'group_names' in kwargs else None
        if (groups is not None) & (group_names is None):
            raise ValueError("Keyword 'group_names' must be set if groups is set. Please provide group names for the groups you are drawing in different colors.") from None
        elif group_names is not None:
            if len(group_names) != len(np.unique(groups)):
                raise ValueError(f"You provided {len(group_names)} groups but I see {len(np.unique(groups))} groups in the 'groups' vector.")

        size_scatter = kwargs['s'] if 's' in kwargs else 50
        size_errorbar = (size_scatter/400)**0.5*20
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 1

        npo = len(x) if hasattr(x,'__len__') else 1

        if groups is None:
            if (col_err is None) & (mag_err is None):
                ax.scatter(x, y, s=size_scatter, 
                           facecolors='none', 
                           edgecolors='black', 
                           alpha = alpha)
            else: ax.errorbar(x, y, yerr=mag_err, 
                              xerr=col_err, fmt='o', 
                              color='black', 
                              ms=size_errorbar, 
                              alpha=alpha)
        else:
            nc = max(groups)
            colormap = plt.cm.gist_ncar
            colorst = [colormap(i) for i in np.linspace(0, 0.9,nc+1)]
            for j in range(nc+1):
                w, = np.where(groups == j)
                if len(w) > 0:
                    if (col_err is None) & (mag_err is None):
                        ax.scatter(x[w], y[w], s=size_scatter, 
                                   facecolors='none', edgecolors=colorst[j], 
                                   label=group_names[j], alpha=alpha)
                    else: 
                        ax.errorbar(x[w], y[w], yerr=mag_err[w], 
                                    xerr=col_err[w], fmt='o', 
                                    color=colorst[j], label=group_names[j], 
                                    ms=size_errorbar, alpha=alpha)

        if label_points == True:
            po = (np.linspace(0, npo-1, num=npo, dtype=int)).astype('str')
            for i, txt in enumerate(po):
                an = ax.annotate(txt, (x[i], y[i]))
                an.set_in_layout(False)

        if 'return_points' in kwargs:
            if kwargs['return_points'] == True: 
                return x, y

    @staticmethod
    def plot_2D_ext(ra=None, dec=None, l=None, b=None,
                    par=None, d=None, color='G', 
                    n=50, tofile=None, ext_map='leike',
                    cmap='viridis', **kwargs):
        
        """
        Plots the integrated absorption in a given region of the sky, by creating a 2D projection at constant distance of an extinction map.
        No parameter is strictly required, but at one between RA and l, one between dec and b, one between par and d must be supplied.
            Input:
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
            - tofile: string, optional. Full path to the output file where the plot will be saved to. Default: None.
            - cmap: string, optional. A valid colormap for the contour plot. Default: 'viridis'.
            - size: int, optional. Size of axis labels and ticks. Default: 15.
            - colorbar: bool, optional. Whether to show a clorbar or not. Default: True.
            - ax: None or Axes object, optional. If nont None, draws the figure over an axisting 'ax'. Default: None.
            Output: no output is returned, but the plot is drawn or overplotted in the current window.
        """

        if d is None:
            if par is None: 
                raise NameError('Exactly one between d and par must be supplied!')
            d = 1000/par
        elif (d is not None) & (par is not None):
            raise NameError('Exactly one between d and par must be supplied!')

        dist = np.full(n**2,d)
        if (ra is not None) & (dec is not None) & (l is None) & (b is None):
            a2 = np.linspace(ra[0],ra[1],n)
            d2 = np.linspace(dec[0],dec[1],n)
            coo2, coo1 = np.meshgrid(d2,a2)
            aa = coo1.ravel()
            dd = coo2.ravel()
            ee = SampleObject.interstellar_ext(ra=aa, dec=dd, 
                                               d=dist, color=color, 
                                               ext_map=ext_map)
            col_name = [r'$\alpha [^\circ]$',r'$\delta [^\circ]$']
        elif (ra is None) & (dec is None) & (l is not None) & (b is not None):
            a2 = np.linspace(l[0],l[1],n)
            d2 = np.linspace(b[0],b[1],n)
            coo2, coo1 = np.meshgrid(d2,a2)
            aa = coo1.ravel()
            dd = coo2.ravel()
            ee = SampleObject.interstellar_ext(l=aa, b=dd, 
                                               d=d, color=color, 
                                               ext_map=ext_map)
            col_name = [r'$l [^\circ]$',r'$b [^\circ]$']
        else: 
            raise NameError('Exactly one pair between (ra, dec) and (l,b) must be supplied!')

        E2 = ee.reshape(n,n)

        size = kwargs['fontsize'] if 'fontsize' in kwargs else 15
        col_bar = kwargs['colorbar'] if 'colorbar' in kwargs else True
        oplot = kwargs['oplot'] if 'oplot' in kwargs else False
        ax = kwargs['ax'] if 'ax' in kwargs else None

        close = False
        if ax == None:
            fig, ax = plt.subplots(figsize=(12,12))
            close = True
        ax.contour(coo1, coo2, E2, 100, cmap=cmap)
        CS = ax.contourf(coo1, coo2, E2, 100, cmap=cmap)
        if col_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(CS, cax=cax)
            if '-' in color: cbar.ax.set_ylabel(color+' reddening [mag]',fontsize=size)
            else: cbar.ax.set_ylabel(color+'-band extinction [mag]',fontsize=size)
            cbar.ax.xaxis.set_tick_params(labelsize=size)
            cbar.ax.yaxis.set_tick_params(labelsize=size)

        ax.set_xlabel(col_name[0],fontsize=size)
        ax.set_ylabel(col_name[1],fontsize=size)
        ax.xaxis.set_tick_params(labelsize=size)
        ax.yaxis.set_tick_params(labelsize=size)
        if 'reverse_xaxis' in kwargs:
            if kwargs['reverse_xaxis']==True: 
                ax.invert_xaxis()
        if 'reverse_yaxis' in kwargs:
            if kwargs['reverse_yaxis']==True: 
                ax.invert_yaxis()
        if tofile is not None: 
            plt.savefig(tofile)
        if close: 
            plt.show()


    ############################################# extinction #################################################

    @staticmethod
    def _download_ext_map(ext_map):

        if ext_map is None: return

        path_ext=os.path.join(madys_path,'extinction')
        if os.path.exists(path_ext) is False: os.mkdir(path_ext) # if the folder does not exist, it creates it
        opl = os.path.join(path_ext,'leike_mean_std.h5')
        ops = os.path.join(path_ext,'stilism_feb2019.h5')

        #if chosen map is not there it fetches it from cds
        if ((ext_map == 'leike') & (os.path.exists(opl) is False)):
            print('You selected the map by Leike et al. (2020), but the file '+opl+' seems missing. ')
            while 1:
                value = input("Do you want me to download the map (size=2.2 GB)? [Y/N]:\n")
                if str.lower(value)=='y':
                    print('Downloading the map...')
                    break
                elif str.lower(value)=='n':
                    break
                else:
                    print("Invalid choice. Please select 'Y' or 'N'.")
            if str.lower(value)=='n': raise KeyboardInterrupt('Please restart the program, setting ext_map=None.')
            else:
                urllib.request.urlretrieve('https://cdsarc.cds.unistra.fr/ftp/J/A+A/639/A138/mean_std.h5.gz', opl+'.gz')
                with gzip.open(opl+'.gz', 'r') as f_in, open(opl, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
                os.remove(opl+'.gz')
                print('Extinction map correctly downloaded.')

        if ((ext_map == 'stilism') & (os.path.exists(ops) is False)):
            print('You selected the map by Lallement et al. (2019), but the file '+ops+' seems missing. ')
            print('')
            while 1:
                value = input("Do you want me to download the map (size=770 MB)? [Y/N]:\n")
                if str.lower(value)=='y':
                    print('Downloading the map...')
                    break
                elif str.lower(value)=='n':
                    break
                else:
                    print("Invalid choice. Please select 'Y' or 'N'.")
            if str.lower(value)=='n': raise KeyboardInterrupt('Please restart the program, setting ext_map=None.')
            else:
                urllib.request.urlretrieve('http://cdsarc.u-strasbg.fr/ftp/J/A+A/625/A135/map3D_GAIAdr2_feb2019.h5.gz', ops+'.gz')
                with gzip.open(ops+'.gz', 'r') as f_in, open(ops, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
                os.remove(ops+'.gz')
                print('Extinction map correctly downloaded.')

    @staticmethod
    def _wu_line_integrate(f, x0, x1, y0, y1, 
                           z0, z1, layer=None,
                           star_id=None, logger=None):
        dim = f.shape
        while np.max(np.abs([x1,y1,z1]))>2*np.max(dim):
            x1 = x0 + (x1-x0)/2
            y1 = y0 + (y1-y0)/2
            z1 = z0 + (z1-z0)/2
        n = int(10*np.ceil(abs(max([x1-x0,y1-y0,z1-z0],key=abs))))
        ndim = len(dim)

        x = np.floor(np.linspace(x0, x1, num=n)).astype(int)
        y = np.floor(np.linspace(y0, y1, num=n)).astype(int)

        if layer is None:
            if ndim == 2:
                d10 = np.sqrt((x1-x0)**2+(y1-y0)**2)
                w_g, = np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]))
                w_g = np.insert(w_g+1,0,0)
                w_f = np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])
                w, = np.where((x[w_g]<dim[0]) & (y[w_g]<dim[1]))
                if (len(w) < len(w_g)) & (logger != None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')
                w2 = w_g[w]
                I = np.sum(f[x[w2],y[w2]]*w_f[w])
            elif ndim == 3:
                d10 = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
                z = np.floor(np.linspace(z0,z1,num=n)).astype(int)
                w_g, = np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]) | (z[1:]!=z[:-1]))
                w_g = np.insert(w_g+1,0,0)
                w_f = np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])
                w, = np.where((x[w_g]<dim[0]) & (y[w_g]<dim[1]) & (z[w_g]<dim[2]) & (x[w_g]>=0) & (y[w_g]>=0) & (z[w_g]>=0))
                if (len(w) < len(w_g)) & (logger != None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')
                w2 = w_g[w]
                I = np.sum(f[x[w2],y[w2],z[w2]]*w_f[w])
        else:
            if ndim == 3:
                d10 = np.sqrt((x1-x0)**2+(y1-y0)**2)
                w_g, = np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]))
                w_g = np.insert(w_g+1,0,0)
                w_f = np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])
                w, = np.where((x[w_g]<dim[1]) & (y[w_g]<dim[2]))
                if (len(w) < len(w_g)) & (logger != None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')
                w2 = w_g[w]
                I = np.sum(f[layer,x[w2],y[w2]] * w_f[w])
            elif ndim == 4:
                d10 = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
                z = np.floor(np.linspace(z0,z1,num=n)).astype(int)
                w_g, = np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]) | (z[1:]!=z[:-1]))
                w_g = np.insert(w_g+1,0,0)
                w_f = np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])
                w, = np.where((x[w_g]<dim[1]) & (y[w_g] < dim[2]) & (z[w_g] < dim[3]))
                if (len(w) < len(w_g)) & (logger != None):
                    logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')
                w2 = w_g[w]
                I = np.sum(f[layer,x[w2],y[w2],z[w2]] * w_f[w])

        return I/n*d10

    @staticmethod
    def interstellar_ext(ra=None, dec=None, l=None, b=None,
                         par=None, d=None, ext_map='leike',
                         color='B-V', error=False, logger=None):

        """
        Computes the reddening/extinction in a custom band, given the position of a star.
        No parameter is strictly required, but at one between RA and l, one between dec and b, one between par and d must be supplied.
            Input:
            - ra: float or numpy array, optional. Right ascension of the star(s) [deg].
            - dec: float or numpy array, optional. Declination of the star(s) [deg].
            - l: float or numpy array, optional. Galactic longitude of the star(s) [deg].
            - b: float or numpy array, optional. Galactic latitude of the star(s) [deg].
            - par: float or numpy array, optional. Parallax of the star(s) [mas].
            - d: float or numpy array, optional. Distance of the star(s) [pc].
            - ext_map: string, optional. Extinction map to be used: must be 'leike' or 'stilism'. Default: 'leike'.
            - color: string, optional. Band in which the reddening/extinction is desired. Default: 'B-V'.
            - error: bool, optional. Computes also the uncertainty on the estimate. Default: False.
            Output:
            - ext: float or numpy array. Best estimate of reddening/extinction for each star.
            - err: float or numpy array, returned only if error==True. Uncertainty on the best estimate of reddening/extinction for each star.
        """        

        if (ra is None) & (l is None): 
            raise NameError('One between RA and l must be supplied!')
        if (dec is None) & (b is None): 
            raise NameError('One between dec and b must be supplied!')
        if (par is None) & (d is None): 
            raise NameError('One between parallax and distance must be supplied!')
        if (ra is not None) & (l is not None): 
            raise NameError('Only one between RA and l must be supplied!')
        if (dec is not None) & (b is not None): 
            raise NameError('Only one between dec and b must be supplied!')
        if (par is not None) & (d is not None): 
            raise NameError('Only one between parallax and distance must be supplied!')

        SampleObject._download_ext_map(ext_map)

        if ext_map is None:
            if ra is not None:
                ext = np.zeros_like(ra)
                if isinstance(ra,np.ndarray) == False:
                    ext = 0.0
            elif l is not None:
                ext = np.zeros_like(l)
                if isinstance(l, np.ndarray) == False:
                    ext = 0.0
            if error: 
                return ext, ext
            else: 
                return ext

        if (ext_map == 'leike') & (error == False): 
            fname = 'leike_mean_std.h5'
        elif (ext_map == 'leike') & (error == True): 
            fname = 'leike_samples.h5'
        if (ext_map == 'stilism'): 
            fname = 'stilism_feb2019.h5'

        paths = [x[0] for x in os.walk(madys_path)]
        found = False
        for path in paths:
            if os.path.isfile(os.path.join(path, fname)):
                map_path = path
                found = True
                break
        if not found:
            print('Extinction map not found! Setting extinction to zero.')
            if hasattr(ra, '__len__'): 
                ebv = np.zeros(len(ra))
            else: ebv = 0.
            return ebv

        fits_image_filename = os.path.join(map_path,fname)
        f = h5py.File(fits_image_filename,'r')
        if ext_map == 'leike':
            x = np.arange(-370.,370.)
            y = np.arange(-370.,370.)
            z = np.arange(-270.,270.)
            if error == False: 
                obj = 'mean'
            else: 
                obj = 'dust_samples'
            data = f[obj][()]
        elif ext_map == 'stilism':
            x = np.arange(-3000.,3005.,5)
            y = np.arange(-3000.,3005.,5)
            z = np.arange(-400.,405.,5)
            data = f['stilism']['cube_datas'][()]

        sun = [closest(x,0), closest(z,0)]

        if d is None:
            try:
                len(par)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    par = np.where(par<0,np.nan,par)
            except TypeError:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if par < 0: 
                        par = np.nan
            d = 1000./par
        if ra is not None:
            c1 = SkyCoord(ra=ra*u.degree, dec=dec*u.degree,
                          distance=d*u.pc, 
                          frame='icrs')
        else:
            c1 = SkyCoord(l=l*u.degree, b=b*u.degree,
                          distance=d*u.pc,
                          frame='galactic')

        galactocentric_frame_defaults.set('pre-v4.0')
        gc1 = c1.transform_to(Galactocentric)
        x0 = (gc1.x+gc1.galcen_distance).value #X is directed to the Galactic Center
        y0 = gc1.y.value #Y is in the sense of rotation
        z0 = (gc1.z-gc1.z_sun).value #Z points to the north Galactic pole

        px = closest(x, x0)
        py = closest(y, y0)
        pz = closest(z, z0)

        dist = x[1] - x[0]

        try:
            len(px)
            wx, = np.where(px < len(x) - 1)
            wy, = np.where(py < len(y) - 1)
            wz, = np.where(pz < len(z) - 1)
            px2 = px.astype(float)
            py2 = py.astype(float)
            pz2 = pz.astype(float)
            px2[wx] = (x0[wx]-x[px[wx]])/dist+px[wx]
            py2[wy] = (y0[wy]-y[py[wy]])/dist+py[wy]
            pz2[wz] = (z0[wz]-z[pz[wz]])/dist+pz[wz]
            ebv = np.full(len(x0),np.nan)
            if ext_map == 'stilism':
                for i in range(len(x0)):
                    if np.isnan(px2[i]) == 0:
                        ebv[i] = dist / 3.16 * SampleObject._wu_line_integrate(data, sun[0],
                                                                               px2[i], sun[0],
                                                                               py2[i], sun[1],
                                                                               pz2[i], star_id=i,
                                                                               logger=logger)
            elif ext_map == 'leike':
                c = dist * 2.5 * np.log10(np.exp(1)) / 3.16 / 0.789
                if error == False:
                    for i in range(len(x0)):
                        if np.isnan(px2[i])==0:
                            ebv[i] = c * SampleObject._wu_line_integrate(data,
                                                                         sun[0],
                                                                         px2[i],
                                                                         sun[0],
                                                                         py2[i],
                                                                         sun[1],
                                                                         pz2[i],
                                                                         star_id=i,
                                                                         logger=logger)
                else:
                    dim = data.shape
                    ebv0 = np.full([len(x0),dim[0]], np.nan)
                    ebv_s = np.full(len(x0), np.nan)
                    for i in range(len(x0)):
                        if np.isnan(px2[i]) == 0:
                            for k in range(dim[0]):
                                ebv0[i,k] = c * SampleObject._wu_line_integrate(data,
                                                                                sun[0],
                                                                                px2[i],
                                                                                sun[0],
                                                                                py2[i],
                                                                                sun[1],
                                                                                pz2[i],
                                                                                layer=k,
                                                                                star_id=i,
                                                                                logger=logger)
                        ebv[i] = np.mean(ebv0[i,:])
                        ebv_s[i] = np.std(ebv0[i,:], ddof=1)
        except TypeError:
            if px < len(x)-1: 
                px2 = (x0-x[px])/dist + px
            else: 
                px2 = px
            if py < len(y)-1: 
                py2 = (y0-y[py])/dist + py
            else: 
                py2 = py
            if pz < len(z)-1: 
                pz2 = (z0-z[pz])/dist + pz
            else: 
                pz2 = pz
            if isinstance(px2, np.ndarray):
                px2 = px2[0]
            if isinstance(py2, np.ndarray):
                py2 = py2[0]
            if isinstance(pz2, np.ndarray):
                pz2 = pz2[0]
            if ext_map == 'stilism':
                c = dist / 3.16
                ebv = SampleObject._wu_line_integrate(data,
                                                      sun[0],
                                                      px2,
                                                      sun[0],
                                                      py2,
                                                      sun[1],
                                                      pz2,
                                                      star_id=0,
                                                      logger=logger)
            elif ext_map == 'leike':
                c = dist * 2.5 * np.log10(np.exp(1)) / 3.16 / 0.789
                if error == False:
                    if np.isnan(px2) == 0:
                        ebv = c * SampleObject._wu_line_integrate(data,
                                                                  sun[0],
                                                                  px2,
                                                                  sun[0],
                                                                  py2,
                                                                  sun[1],
                                                                  pz2,
                                                                  star_id=0,
                                                                  logger=logger)
                    else: return np.nan
                else:
                    dim = data.shape
                    ebv0 = np.zeros(dim[0])
                    if np.isnan(px2) == 0:
                        for k in range(dim[0]):
                            ebv0[k] = c * SampleObject._wu_line_integrate(data,
                                                                          sun[0],
                                                                          px2,
                                                                          sun[0],
                                                                          py2,
                                                                          sun[1],
                                                                          pz2,
                                                                          layer=k)
                    else: 
                        return np.nan, np.nan
                    ebv = np.mean(ebv0)
                    ebv_s = np.std(ebv0, ddof=1)

        if color == 'B-V':
            if error == False:
                return ebv
            else: return ebv, ebv_s
        else:
            if error == False:
                return SampleObject.extinction(ebv, color)
            else:
                return SampleObject.extinction(ebv, color), SampleObject.extinction(ebv_s, color)

    @staticmethod
    def extinction(ebv, col):
        
        """
        Converts one or more B-V color excess(es) into absorption(s) in the required photometric band.
            Input:
            - ebv: float or numpy array, required. Input color excess(es).
            - col: string, required. Name of the photometric band of interest. Use info_filters() for further information on the available bands.
            Output:
            - ext: float or numpy array. Absorption(s) in the band 'col'.
        """
        
        if '-' in col:
            c1,c2=col.split('-')
            A1=stored_data['filters'][c1]['A_coeff']
            A2=stored_data['filters'][c2]['A_coeff']
            A=A1-A2
        else:
            A=stored_data['filters'][col]['A_coeff']

        return 3.16*A*ebv

    ############################################# other astronomical functions ###############################

    @staticmethod
    def app_to_abs_mag(app_mag,parallax,app_mag_error=None,parallax_error=None,ebv=None,ebv_error=None,filters=None):
        
        """
        Turns one or more apparent magnitude(s) into absolute magnitude(s).
            Input:
            - app_mag: float, list or numpy array (1D or 2D), required. Input apparent magnitude(s).
              If a 2D numpy array, each row corresponds to a star, each row to a certain band.
            - parallax: float, list or 1D numpy array, required. Input parallax(es).
            - app_mag_error: float, list or numpy array (1D or 2D), optional. Error on apparent magnitude(s); no error estimation if ==None. Default: None.
            - parallax_error: float, list or 1D numpy array, optional. Error on parallax(es); no error estimation if ==None. Default: None.
            - ebv: float, list or 1D numpy array, optional. E(B-V) affecting input magnitude(s); assumed null if ==None. Default: None.
            - ebv_error: float, list or 1D numpy array, optional. Error on E(B-V); assumed null if ==None. Default: None.
            - filters: list or 1D numpy array, optional. Names of the filters; length must equal no. of columns of app_mag. Default: None.
            Output:
            - abs_mag: float or numpy array. Absolute magnitudes, same shape as app_mag.
            - abs_err: float or numpy array, returned only if app_mag_error!=None and parallax_error!=None. Propagated uncertainty on abs_mag.
        """
        
        if isinstance(app_mag, list): 
            app_mag = np.array(app_mag)
        if (isinstance(parallax, list)) | (isinstance(parallax,Column)): 
            parallax = np.array(parallax, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dm = 5 * np.log10(100./parallax)
        try:
            dim = len(app_mag.shape)
        except (TypeError, AttributeError): 
            dim=0

        if dim <= 1:
            abs_mag = app_mag - dm
            if filters is not None:
                __, i1, i2 = np.intersect1d(np.array(filters),
                                            np.array(['logR','logT','logg','logL']),
                                            return_indices=True)
            else: 
                i1 = []
            if len(i1) > 0:
                if (app_mag_error is not None) & (parallax_error is not None): 
                    return (app_mag, app_mag_error)
                else: 
                    return app_mag
            if ebv is not None:
                if dim == 0: 
                    red = SampleObject.extinction(ebv, filters[0])
                else: 
                    red = np.array([SampleObject.extinction(ebv, filt) for filt in filters])
                abs_mag -= red
            if ebv_error is not None:
                if dim == 0: 
                    red_error = SampleObject.extinction(ebv_error, filters[0])
                else: 
                    red_error = np.array([SampleObject.extinction(ebv_error,filt) for filt in filters])
            else: 
                red_error = 0
            if (app_mag_error is not None) & (parallax_error is not None):
                if isinstance(app_mag_error, list): 
                    app_mag_error = np.array(app_mag_error)
                if (isinstance(parallax_error, list)) | (isinstance(parallax_error, Column)): 
                    parallax_error = np.array(parallax_error,dtype=float)
                total_error = np.sqrt(app_mag_error**2 + (5/np.log(10)/parallax)**2*parallax_error**2 + red_error**2)
                result = (abs_mag, total_error)
            else: 
                result = abs_mag
        else:
            l = app_mag.shape
            abs_mag = np.empty([l[0],l[1]])
            if filters is not None:
                __, i1, i2 = np.intersect1d(np.array(filters),
                                            np.array(['logR','logT','logg','logL']),
                                            return_indices=True)
            else: 
                i1 = []
            if len(i1) > 0:
                abs_mag[:,SampleObject._complement_v(i1,l[1])] = app_mag[:,SampleObject._complement_v(i1,l[1])] - dm
                abs_mag[:,i1] = app_mag[:,i1]
            else:
                for i in range(l[1]): 
                    abs_mag[:,i] = app_mag[:,i] - dm
            if parallax_error is not None:
                if isinstance(app_mag_error, list): 
                    app_mag_error = np.array(app_mag_error)
                if (isinstance(parallax_error, list)) | (isinstance(parallax_error, Column)): 
                    parallax_error = np.array(parallax_error, dtype=float)
                total_error = np.empty([l[0],l[1]])
                red_error = np.zeros([l[0],l[1]])
            if ebv_error is not None:
                if isinstance(ebv_error, list): 
                    ebv_error = np.array(ebv_error)
                for i in range(l[1]):
                    red_error[:,i] = SampleObject.extinction(ebv_error,filters[i])
            for i in range(l[1]):
                total_error[:,i] = np.sqrt(app_mag_error[:,i]**2 + (5/np.log(10)/parallax)**2*parallax_error**2 + red_error[:,i]**2)
            if len(i1)>0: 
                total_error[:,i1] = app_mag_error[:,i1]
            result = (abs_mag, total_error)
            if ebv is not None:
                red = np.zeros([l[0], l[1]])
                for i in range(l[1]):
                    red[:,i] = SampleObject.extinction(ebv, filters[i])
                abs_mag -= red

        return result

    @staticmethod
    def ang_dist(ra1, dec1, ra2, dec2, 
                 ra1_err=0.0, dec1_err=0.0,
                 ra2_err=0.0, dec2_err=0.0,
                 error=False):

        """
        Computes the angular distance between two sky coordinates or two equal-sized arrays of positions.
            Input:
            - ra1: float or numpy array, required. Right ascension of the first star. If unitless, it is interpreted as if measured in degrees.
            - dec1: float or numpy array, required. Declination of the first star. If unitless, it is interpreted as if measured in degrees.
            - ra2: float or numpy array, required. Right ascension of the second star. If unitless, it is interpreted as if measured in degrees.
            - dec2: float or numpy array, required. Declination of the second star. If unitless, it is interpreted as if measured in degrees.
            - ra1_err: float or numpy array, required if error==True. Error on RA, first star. If unitless, it is interpreted as if measured in degrees. Default: None.
            - dec1_err: float or numpy array, required if error==True. Error on dec, first star. If unitless, it is interpreted as if measured in degrees. Default: None.
            - ra2_err: float or numpy array, required if error==True. Error on RA, second star. If unitless, it is interpreted as if measured in degrees. Default: None.
            - dec2_err: float or numpy array, required if error==True. Error on dec, second star. If unitless, it is interpreted as if measured in degrees. Default: None.
            - error: bool, optional. If True, propagates the errors into the final estimate(s).
            Output:
            - dist: float or numpy array. Distance between (each couple of) coordinates [deg].
            - err: float or numpy array, returned only if error==True. Uncertainty on the angular distance for (each couple of) coordinates [deg].
        """
        
        ra1, dec1, ra2, dec2 = np.ma.filled(ra1, fill_value=np.nan), np.ma.filled(dec1, fill_value=np.nan), np.ma.filled(ra2, fill_value=np.nan), np.ma.filled(dec2, fill_value=np.nan)
        try:
            ra1.unit
        except AttributeError:
            ra1 *= u.degree
            dec1 *= u.degree
            ra2 *= u.degree
            dec2 *= u.degree
            dec2_err *= u.degree
            dec1_err *= u.degree
            ra2_err *= u.degree
            ra1_err *= u.degree
        dist = 2*np.arcsin(np.sqrt(np.sin((dec2-dec1)/2.)**2+np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2.)**2)).to(u.deg)
        if error:
            ra1_err, dec1_err, ra2_err, dec2_err = np.ma.filled(ra1_err,fill_value=np.nan), np.ma.filled(dec1_err,fill_value=np.nan), np.ma.filled(ra2_err,fill_value=np.nan), np.ma.filled(dec2_err,fill_value=np.nan)
            ddec2 = (np.sin(dec2-dec1)-2*np.cos(dec1)*np.sin(dec2)*np.sin((ra2-ra1)/2)**2)/(2*np.sqrt(-np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.cos((dec2-dec1)/2)**2)*np.sqrt(np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.sin((dec2-dec1)/2)**2))
            ddec1 = (np.sin(dec2-dec1)-2*np.cos(dec2)*np.sin(dec1)*np.sin((ra2-ra1)/2)**2)/(2*np.sqrt(-np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.cos((dec2-dec1)/2)**2)*np.sqrt(np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.sin((dec2-dec1)/2)**2))
            dra2 = (np.cos(dec2)*np.cos(dec1)*np.sin(ra2-ra1))/(2*np.sqrt(-np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.cos((dec2-dec1)/2)**2)*np.sqrt(np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.sin((dec2-dec1)/2)**2))
            dra1 = -(np.cos(dec2)*np.cos(dec1)*np.sin(ra2-ra1))/(2*np.sqrt(-np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.cos((dec2-dec1)/2)**2)*np.sqrt(np.cos(dec2)*np.cos(dec1)*np.sin((ra2-ra1)/2)**2+np.sin((dec2-dec1)/2)**2))
            err = np.sqrt(dra1**2*e_err**2+dra2**2*ra2_err**2+ddec1**2*dec1_err**2+ddec2**2*dec2_err**2)
            return dist.value, err.value
        else: return dist.value

    ############################################# general functions ##########################################

    @staticmethod
    def _min_v(a, absolute=False):
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
    def _complement_v(arr, n):
        compl = np.full(n, True)
        compl[arr] = False
        compl, = np.where(compl == True)
        return compl

    @staticmethod
    def _intersect1d_rep1(x, y):
        x1 = copy.deepcopy(x)
        y1 = copy.deepcopy(y)
        r, i_1, i_2 = [], [], []
        while 1:
            r0, i1, i2 = np.intersect1d(x1, y1, return_indices=True)
            x1[i1] = '999799' if isinstance(x1[0], str) else 999799
            if len(i1) == 0:
                break
            i_1.append(i1)
            i_2.append(i2)
            r.append(r0)
        i_1 = np.concatenate(i_1)
        i_2 = np.concatenate(i_2)
        r = np.concatenate(r)
        return r, i_1, i_2

    @staticmethod
    def _setup_custom_logger(name, file, mode='a'):
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(file, mode=mode)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.addHandler(handler)

        return logger

    def _print_log(self, ltype, message):
        if self.verbose < 2: return
        else:
            if ltype == 'info':
                self.__logger.info(message)
            elif ltype == 'warning':
                self.__logger.warning(message)
            elif ltype == 'error':
                self.__logger.error(message)


class FitParams(object):

    """
    Class: madys.FitParams

    Class that contains the results of an analysis performed upon a SampleObject instance.
    Created when get_params() is executed upon a sample object.
    It can be accessed like a dictionary.

    Attributes:
        - age: numpy array. Final age estimates [Myr].
        - age_min: numpy array. Minimum age (given by the user or derived) [Myr].
        - age_max: numpy array. Maximum age (given by the user or derived) [Myr].
        - mass: numpy array. Final mass estimates [M_sun or M_jup].
        - mass_min: numpy array. Minimum mass estimates [M_sun or M_jup].
        - mass_max: numpy array. Maximum mass estimates [M_sun or M_jup].
        - ebv: numpy array. Adopted/computed E(B-V), one element per star [mag].
        - ebv_err: numpy array. Error on adopted/computed E(B-V), one element per star [mag].
        - chi2_min: numpy array. Reduced chi2 of best-fit solutions.
        - radius: numpy array. Final radius estimates [R_sun or R_jup]. Only returned if phys_param=True.
        - radius_min: numpy array. Minimum radius estimates [R_sun or R_jup]. Only returned if phys_param=True.
        - radius_max: numpy array. Maximum radius estimates [R_sun or R_jup]. Only returned if phys_param=True.
        - logg: numpy array. Final surface gravity estimates [log10([cm s-2])]. Only returned if phys_param=True.
        - logg_min: numpy array. Minimum surface gravity estimates [log10([cm s-2])]. Only returned if phys_param=True.
        - logg_max: numpy array. Maximum surface gravity estimates [log10([cm s-2])]. Only returned if phys_param=True.
        - logL: numpy array. Final luminosity estimates [log10([L_sun])]. Only returned if phys_param=True.
        - logL_min: numpy array. Minimum luminosity estimates [log10([L_sun])]. Only returned if phys_param=True.
        - logL_max: numpy array. Maximum luminosity estimates [log10([L_sun])]. Only returned if phys_param=True.
        - Teff: numpy array. Final effective temperature estimates [K]. Only returned if phys_param=True.
        - Teff_min: numpy array. Minimum effective temperature estimates [K]. Only returned if phys_param=True.
        - Teff_max: numpy array. Maximum effective temperature estimates [K]. Only returned if phys_param=True.
        - fit_status: numpy array. Flag for the outcome of the fitting process, one element per star.
            0 - Successful fit.
            1 - All magnitudes for the star have an error beyond the maximum allowed threshold: age and mass determinations was not possible.
            2 - All magnitudes for the star are more than 0.2 mag away from their best theoretical match. Check age and mass range of the theoretical grid, or change the model if the current one does not cover the expected age/mass range for this star.
            3 - No point with chi2<1000 was found for the star.
            4 - The third closest filter in the best-fit solution is more than 3 sigma away from its theoretical match, and the third closest magnitude to its theoretical match is more than 0.1 mag away.
            5 - Undefined error.
        - chi2_maps: list. Only present if save_maps=True in the parent analysis. Contains one 2D numpy array per star; matrix elements are reduced chi2 estimates for grid points, using nominal data.
        - weight_maps: list. Only present if save_maps=True in the parent analysis. Contains one 2D numpy array per star; matrix elements are the weight of grid points, as used to obtain the final family of solutions.
        - all_solutions: list. Contains a dictionary per star, with all possible solutions providing an accettable fit to data.
        - feh: float. [Fe/H] of the grid.
        - he: float. Helium content of the grid.
        - afe: float. Alpha enhancement [a/Fe] of the grid.
        - v_vcrit: float. Rotational velocity of the grid.
        - fspot: float. Fraction of stellar surface covered by star spots.
        - B: int. Whether magnetic fields are included (1) or not (0) in the grid.
        - sample_name: string. Only returned if verbose>0. Name of the sample file, without extension.
        - path: string. Only returned if verbose>0. Full path to the sample file, without extension.
        - objects: numpy array. Names of analyzed objects. Corresponds to self.Gaia_ID of the parent SampleObject instance.
        - original_IDs: numpy array. Original IDs of analyzed objects. Corresponds to self.ID of the parent SampleObject instance.
        - exec_command: numpy array. Each entry is the __repr__ of the IsochroneGrid object used within SampleObject.get_params.
        - fitting_mode: int. Fitting mode of the parent get_params process. It can be either:
            1: the age was set a priori to a single value, or the selected model_version only has one age; corresponding to case 2) for the keyword 'age_range' from SampleObject.get_params.
            2: the age was to be found within the specified interval; corresponding to case 1) or 3) for the keyword 'age_range' from SampleObject.get_params.
            3: the age was fixed, and age_min and age_max were used to compute errors; corresponding to case 4) for the keyword 'age_range' from SampleObject.get_params.
            4: the program was left completely free to explore the entire age range.
        - model_grid: numpy array. Each entry is the model_version used to fit the corresponding star with SampleObject.get_params.
        - is_true_fit: bool. Whether the instance comes directly from a fit, or if it's an average of different model estimates.
        - units: dict. Physical units of mass, radius and Teff.
        - input_parameters: dict. Input parameters used to initialize the parent SampleObject.get_params() process.
        - a triple of numpy arrays (value, value_min, value_max) for every additional quantity specified through the 'additional_columns' keyword in the parent SampleObject.get_params() process.

    Built-in methods:

    1) __getitem__
    FitParam instances can be indexed like pandas dataframes, i.e. with a numpy array/list/int or with a string containing the column name.
    In the former case, the sample is cut according to the selected indices.
    In the latter case, the corresponding key is returned.
    See documentation for additional details.

    2) __setitem__
    FitParam instances support item assignment.
    Check the documentation for additional details.

    3) __len__
    The len of a FitParams instance is equal to the number of objects in the original list.
    
    4) __repr__
    Verbose reproduction of the execution command ran by the user. If coming from an average of different
    FitParams instance, it lists all of them.
    
    5) __eq__
    Two FitParams instances are defined to be equal if their object list and their attributes 'objects' and 'exec_command' are equal.

    Methods (use help() to have more detailed info):

    1) empty_like
    Starting from an existing instance, creates an empty FitParams instance with a given dimension.

    2) to_file
    Saves the instance in a .csv file.

    3) to_table
    Turns the main attributes of a FitParams instance into columns of an astropy Table.

    4) pprint
    Enables fancy print of a FitParams instance.

    5) plot_maps
    Plots (reduced chi2 / weight) maps of one or more stars as a f(mass,age) color map.

    6) average_results
    Averages the results of two or more FitParams instances coming from the same list of objects.
    
    7) export_to_file
    Exports the instance to a .h5 file. Complementary to import_from_file(), which performs the opposite operation.

    8) import_from_file
    Alternative initializer for the class. It imports the instance from a .h5 file. Complementary to export_to_file(), which performs the opposite operation.    

    """

    def __init__(self,dic):
        for i in dic.keys():
            self.__dict__[i]=dic[i]

    def __len__(self):
        return len(self.age)

    def __getitem__(self, i):
        new = copy.deepcopy(self)
        if isinstance(i,str):
            return self.__dict__[i]
        for j in new.__dict__.keys():
            try:
                if isinstance(new.__dict__[j], str):
                    continue
                elif hasattr(new.__dict__[j], '__len__') == False:
                    new.__dict__[j] = new.__dict__[j]
                else:
                    new.__dict__[j] = new.__dict__[j][i]
            except TypeError:
                continue
        return new

    def __setitem__(self, i, other):
        self.age[i] = other.age

        for j in self.__dict__.keys():
            if isinstance(self.__dict__[j],str): self.__dict__[j] = other.__dict__[j]
            elif j in ['all_solutions', 'chi2_maps', 'weight_maps', 'exec_command', 'model_grid']:
                try:
                    self.__dict__[j][i] = other.__dict__[j]
                except TypeError:
                    for k in range(len(i)):
                        self.__dict__[j][i[k]] = other.__dict__[j][k]
            elif isinstance(self.__dict__[j], np.ndarray):
                self.__dict__[j][i] = other.__dict__[j]
            elif j in ['is_true_fit', 'input_parameters']:
                self.__dict__[j] = other.__dict__[j]

    def __repr__(self):
        if self.is_true_fit:
            model_version, __ = ModelHandler._grid_to_version(self.model_grid[0])
            input_parameters_string = repr(self.input_parameters).replace('{','').replace('}','').replace(':',' =')

            s = "your_input_object.FitParams('{0}', {1})".format(model_version, input_parameters_string)
        else:
            n_models = len(self.model_grid)
            s = """This object is the average of {0} independent parameter estimates: """.format(n_models)
            for i in range(n_models):
                model_version, __ = ModelHandler._grid_to_version(self.model_grid[i][0])
                input_parameters_string = repr(self.input_parameters[i]).replace('{','').replace('}','').replace(':',' =')
                s += "\nyour_input_object.FitParams('{0}', {1})".format(model_version, input_parameters_string)

        return s
    
    def __eq__(self, other):
        
        cond1 = np.array_equal(self.objects, other.objects)
        cond2 = np.array_equal(self.exec_command, other.exec_command)
        if cond1 & cond2:
            return True
        else: return False
    

    def empty_like(self,n, fill_value=0):

        """
        Starting from an existing instance, creates an empty FitParams instance with a given dimension.
            Input:
            - n: int, required. len() of the new instance.
            - fill_value: float, optional. Fill value of empty entries. Default: 0.
            Output:
            - new: an empty FitParams instance.
        """
        
        new = copy.deepcopy(self)
        if fill_value == 0:
            f = lambda x, n, fill: np.zeros_like(x,shape=n)
        else:
            f = lambda x, n, fill: np.full_like(x,fill,shape=n)
            
        for j in new.__dict__.keys():
            if isinstance(new.__dict__[j],str): continue
            elif j=='all_solutions':
                new.__dict__[j]=[{} for i in range(n)]
            elif isinstance(new.__dict__[j],np.ndarray):
                new.__dict__[j]=f(new.__dict__[j], n, fill_value)
            elif j=='chi2_maps':
                new.__dict__[j]=[[] for i in range(n)]
            elif j=='weight_maps':
                new.__dict__[j]=[[] for i in range(n)]
            elif j=='exec_command':
                new.__dict__[j]=[[] for i in range(n)]
            elif j=='model_grid':
                new.__dict__[j]=[[] for i in range(n)]
        return new
    
    def to_file(self, filename, check_verbose=False):

        """
        Saves the results contained in the instance to a .csv or an ascii file.
            Input:
            - filename: string, required. Full path to output file.
            - check_verbose: bool, optional. If set to True, it checks if the degree of verbosity is >0. Default: False.
            
            Output: besides the .csv/ascii file, no output is returned.
            The output file will be:
            - a fixed-width ascii file if the extension is not equal to '.csv';
            - a comma-separated .csv file otherwise.

        """

        if check_verbose:
            try:
                self['sample_name']
            except KeyError: raise ValueError('verbose=0, so the results cannot be saved to a file.')

        found = False
        original_names = self['original_IDs']
        Gaia_names = np.array(self['objects'], dtype = str)

        tab = self.to_table()

        if np.sum(original_names == Gaia_names) < len(self):
            tab['input_name'] = original_names

        if filename.endswith('.csv'):        
            ascii.write(tab, filename, format='csv', fast_writer=False, overwrite=True)  

        else:
            tab.round(2)
            ascii.write(tab, filename, format='fixed_width', fast_writer=False, overwrite=True, delimiter=None)

    def export_to_file(self, output_file):
        
        """
        Saves an existing FitParams instance to a .h5 file.
            Input:
            - output_file: string, required. Full path to the output .h5 file.
            Output:
            - no output is returned apart from the .h5 file.
        """


        with h5py.File(output_file, 'w') as hf:

            for i in self.__dict__:

                if isinstance(self[i], np.ndarray):
                    if isinstance(self[i][0],str):
                        values = np.array(self[i], dtype=dt) 
                    else: values = self[i]

                    dset = hf.create_dataset(i, data = values, compression='gzip', compression_opts=9)
                elif i in ['chi2_maps', 'weight_maps']:
                    dset = hf.create_dataset(i, data = self[i], compression='gzip', compression_opts=9)
                else:
                    try:
                        hf.attrs[i] = self[i]
                    except TypeError:
                        hf.attrs[i] = str(self[i])
            hf.attrs['class'] = type(self).__name__

    @classmethod
    def import_from_file(cls, file):
        
        """
        Alternative initializer for the FitParams class.
        It creates an instance from a valid .h5 file.
            Input:
            - file: string, required. Full path to the input .h5 file.
            Output:
            - instance: a FitParams instance.
        """

        if file.endswith('.h5') == False:
            raise TypeError('Input file must be a .h5 file.')
            
        replacements = {'array': 'np.array', 'nan': 'np.nan'}
        dic = {}
        with h5py.File(file,"r") as hf:
            
            if hf.attrs['class'] != 'FitParams':
                raise ValueError('The provided file does not appear to be an instance of the FitParams class.')
            
            for dataset in hf.values():
                name = str(dataset).split('"')[1]
                if name in ['all_solutions']:
                    dic[name] = hf.get(name)
                else:
                    dic[name] = hf.get(name)[:]
            for i in hf.attrs.keys():
                val = hf.attrs[i]
                if isinstance(val, str):
                    for r in replacements.keys():
                        val = val.replace(r, replacements[r])
                    dic[i] = eval(val)
                else: dic[i] = val

        for key in dic.keys():
            if isinstance(dic[key], np.ndarray):
                if isinstance(dic[key][0], bytes):
                    dic[key] = dic[key].astype(str)

        if 'class' in dic.keys():
            del dic['class']
            
        return FitParams(dic)
    
    def to_table(self, **kwargs):

        """
        Turns the main attributes of a FitParams instance into columns of an astropy Table.
            Input:
            - round: int, required. Rounds all table entries to a number 'round' of digits.
            Output: an astropy table with columns:
               'objects', 'age', 'age_min', 'age_max', 'mass', 'mass_min', 'mass_max',
               'ebv', 'radius', 'radius_min', 'radius_max', 'logg', 'logg_min', 'logg_max', 'logL',
               'logL_min', 'logL_max', 'Teff', 'Teff_min', 'Teff_max', 'fit_status'
               + all the additional requested syntethic photometry.
        """

        round_digits = None
        if 'round' in kwargs:
            round_digits = kwargs['round']
            del kwargs['round']

        t = {}

        columns = np.array(list(self.__dict__.keys()))
        to_remove = ['chi2_min', 'chi2_maps', 'feh', 'he', 'afe', 'v_vcrit', 'fspot', 
                     'B', 'exec_command', 'model_grid', 'fitting_mode', 'objects',
                     'weight_maps', 'all_solutions', 'all_solutions_B',
                     'is_true_fit', 'input_parameters', 'original_IDs']
        columns = np.concatenate((['objects'], columns[~np.in1d(columns, to_remove)]))

        for i in columns:
            if isinstance(self[i], str):
                continue
            elif isinstance(self[i], dict):
                continue
            try:
                len(self[i])
                t[i] = self[i]
            except KeyError:
                continue
            except TypeError:
                if (type(self[i]) == float) | (type(self[i]) == int):
                    self.__dict__[i] = np.full_like(self['mass'], self[i])
                else:
                    del self.__dict__[i]

        tab = Table(t, **kwargs)

        if round_digits is not None:
            tab.round(round_digits)

        return tab
    
    def pprint(self, mode=None, **kwargs):

        """
        Enables fancy print of a FitParams instance.
            Input:
            - mode: string, optional. Use:
                - 'all': to return all the rows via astropy.table's pprint_all();
                - 'in_notebook': to return an interactive print of the table in a Jupyter Notebook via astropy.table's show_in_notebook.
                - None: to return a simple astropy Table with default options (e.g. max no. of rows).
            Output: astropy Table.
        """
        
        tab = self.to_table(**kwargs)
        if mode == 'all':
            return tab.pprint_all()
        elif mode == 'in_notebook':
            return tab.show_in_notebook(css='%3.1f')
        else:
            return tab
        
    def plot_maps(self, indices=None, tofile=False, dtype='chi2'):

        """
        Plots (reduced chi2 / weight) maps of one or more stars as a f(mass,age) color map.
            Input:
            - indices: list, optional. Indices of the stars of interest, ordered as in the original list. Default: numpy.arange(n_stars), i.e. all stars.
            - tofile: bool, string or list, optional. It can be set to either:
                  - (only if verbose=0) True, to save the plots as .png images in the same path where the analysis was performed.
                  - (if len(indices)=1) a string, indicating the full path to the output file.
                  - (if len(indices)>1 or indices not set) a list; each entry should be drafted as in the case above.
              Default: False.
            - dtype: string, optional. Use 'chi2' to plot chi2 maps, 'weights' to plot weight maps. Default: 'chi2'.
            Output: no output is returned, but the plot is shown in the current window.
        """

        if self['is_true_fit'] == False:
            raise ValueError("This instance comes from an average of models, so there's no {0} map to plot.".format(dtype))
                             
        if indices is None: 
            indices = np.arange(len(self))

        try: 
            len(indices)
        except TypeError: 
            indices = np.array([indices])

        if dtype == 'chi2':
            key = 'chi2_maps'
        elif dtype == 'weights':
            key = 'weight_maps'
        else: 
            raise ValueError("Invalid value for 'dtype'. dtype must be either 'chi2' or 'weights'.")


        if ((self['fitting_mode']==1) | (self['fitting_mode']==3)) & (key=='weight_maps'):
            raise ValueError('No weight map is returned under fitting mode '+str(self['fitting_mode']))

        if np.max(indices) >= len(self):
            raise IndexError('index '+str(int(np.max(indices)))+' is out of bounds for axis 0 with size '+str(len(self)))

        try:
            self[key]
        except KeyError:
            raise KeyError('No '+dtype+' maps present. Perhaps get_params was used with save_maps=False?')

        if hasattr(self['fit_status'],'__len__') == False:
            if self['fit_status'] != 0:
                print('No solution was found for star '+str(i)+'. Check the log for details.')
                return
            m_sol = self['all_solutions']['mass']

            chi2 = self[key]

            th_model = eval(self['exec_command'])
            iso_mass = th_model.masses
            iso_age = th_model.ages
            model = th_model.model_version
            AA, MM = np.meshgrid(iso_age, iso_mass)

            if dtype == 'chi2':
                best = np.nanmin(chi2)
                arg_best = np.nanargmin(chi2)

                plt.figure(figsize=(12,12))
                levels = 10**np.linspace(np.log10(best), np.log10(best+15), 10)
                plt.contour(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm_r)
                h = plt.contourf(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm_r)
                CB = plt.colorbar(h,ticks=levels,format='%.1f')
                CB.set_label(r'$\chi^2$', rotation=270)
                m_range=[np.min(m_sol)*0.9,np.max(m_sol)*1.1]
                plt.xlim(m_range)
                plt.yscale('log')
                i70, i85 = np.argmin(np.abs(iso_mass-m_range[0])), np.argmin(np.abs(iso_mass-m_range[1]))
                for j in range(i70,i85):
                    plt.plot([iso_mass[j],iso_mass[j]],[iso_age[0],iso_age[-1]],color='white',linewidth=0.3)
                for j in range(len(iso_age)):
                    plt.plot([iso_mass[i70],iso_mass[i85]],[iso_age[j],iso_age[j]],color='white',linewidth=0.3)
                plt.title(r'$\chi^2$ map for star 0, '+str.upper(model))
            elif dtype == 'weights':
                best = np.nanmax(chi2)
                arg_best = np.nanargmax(chi2)

                plt.figure(figsize=(12,12))
                levels = np.linspace(0, best, 10)
                plt.contour(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm)
                h = plt.contourf(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm)
                CB = plt.colorbar(h,ticks=levels,format='%.3f')
                CB.set_label('weight', rotation=270)
                m_range = [np.min(m_sol)*0.9,np.max(m_sol)*1.1]
                plt.xlim(m_range)
                plt.yscale('log')
                i70, i85 = np.argmin(np.abs(iso_mass-m_range[0])),np.argmin(np.abs(iso_mass-m_range[1]))
                for j in range(i70,i85):
                    plt.plot([iso_mass[j], iso_mass[j]], [iso_age[0], iso_age[-1]],
                             color='white', linewidth=0.3)
                for j in range(len(iso_age)):
                    plt.plot([iso_mass[i70], iso_mass[i85]], [iso_age[j], iso_age[j]],
                             color='white', linewidth=0.3)
                plt.title('weight map for star 0, '+str.upper(model))

            plt.ylabel(r'age [Myr]')
            plt.xlabel(r'mass [$M_\odot$]')

            if isinstance(tofile, bool):
                if tofile:
                    file = self['path']+'_'+dtype+'_map_star0_'+model+'.png'
                    plt.savefig(file)
            else:
                plt.savefig(tofile)

            plt.show()

        else:
            p=0
            for i in indices:

                try:
                    m_sol=self['all_solutions'][i]['mass']
                except KeyError:
                    print('No solution was found for star '+str(i)+'. Check the log for details.')
                    p+=1
                    continue

                chi2 = self[key][i]
                m_sol = self['all_solutions'][i]['mass']

                th_model = eval(self['exec_command'][i])
                iso_mass = th_model.masses
                iso_age = th_model.ages
                model = th_model.model_version
                AA, MM = np.meshgrid(iso_age, iso_mass)

                if dtype == 'chi2':
                    best = np.nanmin(chi2)
                    arg_best = np.nanargmin(chi2)

                    plt.figure(figsize=(12,12))
                    levels = 10**np.linspace(np.log10(best), np.log10(best+15), 10)
                    plt.contour(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm_r)
                    h = plt.contourf(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm_r)
                    CB = plt.colorbar(h,ticks=levels, format='%.1f')
                    CB.set_label(r'$\chi^2$', rotation=270)
                    m_range = [np.min(m_sol)*0.9, np.max(m_sol)*1.1]
                    plt.xlim(m_range)
                    plt.yscale('log')
                    i70, i85 = np.argmin(np.abs(iso_mass-m_range[0])), np.argmin(np.abs(iso_mass-m_range[1]))
                    for j in range(i70, i85):
                        plt.plot([iso_mass[j], iso_mass[j]], [iso_age[0], iso_age[-1]], 
                                 color='white', linewidth=0.3)
                    for j in range(len(iso_age)):
                        plt.plot([iso_mass[i70], iso_mass[i85]], [iso_age[j], iso_age[j]],
                                 color='white', linewidth=0.3)
                    plt.title(r'$\chi^2$ map for star '+str(i)+', '+str.upper(model))
                elif dtype == 'weights':
                    best = np.nanmax(chi2)
                    arg_best = np.nanargmax(chi2)

                    plt.figure(figsize=(12,12))
                    levels = np.linspace(0, best, 10)
                    plt.contour(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm)
                    h = plt.contourf(MM, AA, chi2, levels=levels, extend='both', cmap=cm.coolwarm)
                    CB = plt.colorbar(h,ticks=levels,format='%.3f')
                    CB.set_label('weight', rotation=270)
                    m_range = [np.min(m_sol)*0.9, np.max(m_sol)*1.1]
                    plt.xlim(m_range)
                    plt.yscale('log')
                    i70, i85 = np.argmin(np.abs(iso_mass-m_range[0])), np.argmin(np.abs(iso_mass-m_range[1]))
                    for j in range(i70,i85):
                        plt.plot([iso_mass[j], iso_mass[j]], [iso_age[0], iso_age[-1]],
                                 color='white', linewidth=0.3)
                    for j in range(len(iso_age)):
                        plt.plot([iso_mass[i70], iso_mass[i85]], [iso_age[j], iso_age[j]],
                                 color='white', linewidth=0.3)
                    plt.title('weight map for star '+str(i)+', '+str.upper(model))

                plt.ylabel(r'age [Myr]')
                plt.xlabel(r'mass [$M_\odot$]')

                if isinstance(tofile,bool):
                    if tofile:
                        try:
                            file = self['path']+'_'+dtype+'_map_star'+str(i)+'_'+model+'.png'
                            plt.savefig(file)
                        except TypeError:
                            raise TypeError('verbose is set to 0, so MADYS does not have a path to save files in. Either restart your analysis with verbose>0, or set tofile equal to a list of file names, one per star.')
                else:
                    if isinstance(tofile,list):
                        plt.savefig(tofile[i])
                    else:
                        raise TypeError('You requested multiple outputs, hence the argument of tofile must be a list of file names.')

                plt.show()

                p += 1
                
    @staticmethod
    def _apply_cuts_to_instances(list_of_instances, cuts = None):

        if (cuts is not None) & (isinstance(cuts, dict) == False):
            raise TypeError("'cuts' must be a dictionary. Please provide a correct input.")
        elif cuts is None: return list_of_instances
    

        indices = list(cuts.keys())
        for i in range(len(indices)):
            if isinstance(indices[i], int) == False:
                raise TypeError("The keys of 'cuts' must be of type int! Please check your input dictionary.")

        n_instances = len(list_of_instances)
        if np.max(indices) >= n_instances:
            raise ValueError('Conditions for the i={0} instance in the input list have been provided, but the list only contains {1} elements!'.format(np.max(indices), n_instances))

        valid_params = ['mass', 'age', 'Teff', 'logg', 'radius', 'logL']

        new_list_of_instances = []
        if cuts is not None:
            for j in range(n_instances):
                if j not in indices:
                    new_list_of_instances.append(list_of_instances[j])
                    continue
                instance = list_of_instances[j]
                n = len(instance)
                if isinstance(cuts[j], str):
                    conditions = [cuts[j]]
                else:
                    conditions = cuts[j]

                keep = np.ones(n, dtype=bool)
                for cond in conditions:
                    p1 = re.compile('\s*([a-zA-Z]+)\s*(>*<*=*!*)\s*([0-9]+\.*[0-9]*)\s*')
                    m1 = p1.match(cond)
                    if m1 is not None:
                        col = m1.group(1)
                        sign = m1.group(2)
                        value = float(m1.group(3))
                        if sign == '=': sign = '=='
                        s = "instance[col]{0}value".format(sign)
                        try:
                            keep[~eval(s)] = False
                        except (SyntaxError, TypeError):
                            raise ValueError('One of the expressions in {0} was not recognized. At least one sign is not valid.'.format(cuts)) from None
                        except KeyError:
                            raise KeyError('The following parameter does not exist: {0}. Please check the syntax of your input. Valid parameters: {1}'.format(col,valid_params)) from None

                    else:
                        raise ValueError('Expression {0} not recognized.'.format(cond))

                new_instance = instance.empty_like(len(instance), fill_value = np.nan)
                w, = np.where(keep)
                new_instance[w] = instance[w]
                
                for keyword in ['model_grid', 'exec_command', 'objects', 'ebv', 'ebv_err', 'original_IDs']:
                    new_instance.__dict__[keyword] = instance.__dict__[keyword]
                new_instance.__dict__['fit_status'] = np.where(keep, instance.__dict__['fit_status'], 5)

                new_list_of_instances.append(new_instance)

        return new_list_of_instances
    
    @staticmethod
    def average_results(results, minimum_error=None, cuts=None):

        """
        Averages the results of two or more FitParams instances coming from the same list of objects.
        This is especially useful for model comparison.
        Warning: the average assumes that every parameter of the best-fit solution can be approximated by a normal distribution
        and that parameter uncertainties across different instances are equivalent (i.e., an arithmetic mean is performed).
        These two approximations might not hold in every case, especially if the age is not well constrained. Hence, it is
        adviced to use this function with caution.
            Input:
            - results: list or tuple, required. Every element should represent a FitParams instance; all instances should come from the analysis of the same SampleObject instance.
            - minimum_error: float, optional. Relative error (between 0 and 1) that will be assigned by default to every final parameter if its error is smaller than minimum_error. Default: None (=0).
            - cuts: dict, optional. Conditions to apply to each input instance: only stars satisfying all these conditions will be kept.
              Example: we want to average three datasets, but we only want to consider hotter stars than 2000 K for the first model,
              and stars between 0.3 and 0.7 M_sun for the second model; no condition is needed for the third model.
              Multiple conditions are handled through lists. We would write something like that:
              cuts = {0: 'Teff > 2000', 1: ['mass > 0.3', 'mass < 0.7']}.
              Default: None (no condition applied, i.e. all information is retained).
            Output:
            - new_instance: FitParams instance with average values for the parameters. Please notice that the format of some attributes is different from the usual one
              because they incorporate information coming from more than a single get_params() execution.
        """
        
        all_cols = np.unique(np.concatenate([list(res.__dict__.keys()) for res in results]))

        def _get_avg_column(input_results, col, minimum_error = None):

            results = copy.deepcopy(input_results)

            for res in results:
                if col not in res.__dict__.keys():
                    res.__dict__[col] = np.full(len(res),np.nan)
                if (col+'_max' not in res.__dict__.keys()) & (col+'_max' in all_cols):
                    res.__dict__[col+'_max'] = np.full(len(res),np.nan)
                if (col+'_min' not in res.__dict__.keys()) & (col+'_min' in all_cols):
                    res.__dict__[col+'_min'] = np.full(len(res),np.nan)

            nominal_values = [res[col] for res in results]
            nominal_errp = [(res[col+'_max']-res[col]).ravel() for res in results]
            nominal_errm = [(-res[col+'_min']+res[col]).ravel() for res in results]
            nominal_err = np.nanmean([nominal_errp,nominal_errm],axis=0)

            n_models = np.sum([np.isnan(nom)==False for nom in nominal_values],axis=0)
            sqrt_dof = np.sqrt(n_models-1)
            avg = np.nanmean(nominal_values,axis=0)
            avg_min = np.where(sqrt_dof>0, avg - np.nanstd(nominal_values,axis=0)/sqrt_dof, avg - np.nanmax(nominal_err,axis=0))
            avg_max = np.where(sqrt_dof>0, avg + np.nanstd(nominal_values,axis=0)/sqrt_dof, avg + np.nanmax(nominal_err,axis=0))
            
            if minimum_error is not None:
                if (minimum_error<0) | (minimum_error>1):
                    raise ValueError("Invalid value provided for the keyword 'minimum_error'. Plese provide a value between 0 and 1 because this value is a relative error.")
                avg_min = np.where((avg-avg_min)/avg < minimum_error, (1-minimum_error) * avg, avg_min)
                avg_max = np.where((avg_max-avg)/avg < minimum_error, (1+minimum_error) * avg, avg_max)

            return avg, avg_min, avg_max

        if (isinstance(results, list) == False) & (isinstance(results, tuple) == False):
            raise TypeError('Only lists or tuples of FitParams instances are valid inputs.')

        n = len(results)
        for i in range(n):
            if 'FitParams' not in str(type(results[i])):
                raise TypeError('The provided list or tuple must only contain FitParams instances.')
        if n == 1: return results[0]

        lens = [len(i) for i in results]
        if len(np.unique(lens)) > 1:
            msg = """The number of stars in the provided instances is the following:
            \n{0}
            \nThe star list is supposed to be the same in all FitParams instances to be averaged!

            """.format(lens)
            raise ValueError(msg)

        ids_equal = [list(res['objects'])==list(results[0]['objects']) for res in results]
        if np.sum(ids_equal) < len(results):
            raise ValueError('The star IDs appear to be different. Are you sure the FitParams instances are derived from the same SampleObject instance?')

        if cuts is not None:
            results = FitParams._apply_cuts_to_instances(results, cuts)

        new_instance = results[0].empty_like(len(results[0]))    
        cols = np.array(['age', 'mass', 'radius', 'logg', 'logL', 'Teff',
                         'mass_B', 'radius_B', 'logg_B', 'logL_B', 'Teff_B'])
        n_stars = len(results[0])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            for i, col in enumerate(cols):
                if col in all_cols:
                    avg, avg_min, avg_max = _get_avg_column(results, col, minimum_error = minimum_error)
                    new_instance.__dict__[col] = avg
                    new_instance.__dict__[col+'_min'] = avg_min
                    new_instance.__dict__[col+'_max'] = avg_max

            for col in all_cols:
                if (col.startswith('synth')) & (col.endswith('max')==False) & (col.endswith('min')==False):
                    avg, avg_min, avg_max = _get_avg_column(results, col, minimum_error = minimum_error)
                    new_instance.__dict__[col] = avg
                    new_instance.__dict__[col+'_min'] = avg_min
                    new_instance.__dict__[col+'_max'] = avg_max

            new_instance.__dict__['ebv'] = np.nanmean([res['ebv'] for res in results], axis=0)
            new_instance.__dict__['ebv_err'] = np.std([res['ebv'] for res in results], axis=0)
            new_instance.__dict__['fit_status'] = np.nanmin([res['fit_status'] for res in results], axis=0)
            new_instance.__dict__['objects'] = results[0]['objects']

            if np.sum(['all_solutions' in res.__dict__.keys() for res in results]) == len(results):
                for col in ['all_solutions']:
                    values = []
                    for i in range(n_stars):
                        values.append([res[col][i] for res in results])
                    new_instance.__dict__[col] = values                
            
            for col in ['model_grid','exec_command']:
                values = [res[col] for res in results]
                new_instance.__dict__[col] = values                

            new_instance.__dict__['input_parameters'] = [res['input_parameters'] for res in results]
            new_instance.__dict__['original_IDs'] = results[0]['original_IDs']

        new_instance.__dict__['is_true_fit'] = False

        for key in ['B', 'feh', 'he', 'afe', 'v_vcrit', 'weight_maps', 'chi2_maps', 'fitting_mode']:
            try:
                del new_instance.__dict__[key]
            except KeyError:
                continue

        return new_instance
                
                
class CurveObject(object):

    """
    Class: madys.CurveObject

    Class that turns a contrast/flux limit curve from a direct imaging survey into a mass limit curve.
    Check the documentation for additional details on general functioning, customizable settings and various examples.

    New in version v1.1.0.

    Input:
    - file: string, required. Full path to the .fits file containing the input curve;
    - file_type: string, required. It can be either:
        - 'contrast_separation', if a 1D curve(separation) with shape (n_points, 2) is provided; the first column is assumed to
          correspond to contrasts, the second one to separations;
        - 'contrast_map', if a 2D curve(x, y) is provided. A 3D map (lambda, x, y) is accepted too.
    - stellar_parameters: dict, required. A dictionary containing information for the star under consideration. The following keywords must be present:
        - 'parallax': float. Stellar parallax [mas];
        - 'parallax_error': float. Uncertainty on stellar parallax [mas];
        - 'app_mag': float. Apparent magnitude of the star in the 'band' band [mag];
        - 'app_mag_error': float. Uncertainty on 'app_mag' [mag];
        - 'band': string. Filter which the map refers to. It should be a valid filter name for MADYS;
        - 'age': float. Stellar age [Myr];
        - 'age_error': float. Uncertainty on stellar age [Myr].
        For extinction, two possibility exist. Either it is set explicitly through the following doublet of keywords:
        - 'ebv': float. E(B-V) reddening for the star [mag];
        - 'ebv_error': float. Uncertainty on E(B-V) [mag];
        or coordinates must be specified to allow the program to estimate E(B-V):
        - 'ra': float. Right ascension of the star [deg];
        - 'dec': float. Declination of the star [deg].
        - 'ext_map': string, optional. 3D extinction map used for the computation. Possible values: 'leike', 'stilism'. Default: 'leike'.
        In the latter case, the error on ebv is always set to 0.
    - data_unit: string, optional. Choose 'magnitude' if the map is expressed in magnitudes, 'flux' if in flux contrast. Default: 'flux'.
    - rescale_flux = float, optional. Renormalization constant the flux is to be multiplied by. Default: 1.
    
    Attributes:
    - file: string. Corresponding to input 'file'.
    - file_type: string. Corresponding to input 'file_type'.
    - stellar_parameters: dict. Corresponding to input 'stellar_parameters'.
    - data_unit: string. Corresponding to input 'data_unit'.
    - rescale_flux: float. Corresponding to input 'rescale_flux'.
    - contrasts: numpy array. Renormalized contrast curve (flux ratio).
    - contrasts_mag: numpy array. Renormalized contrast curve (magnitude contrast).
    - abs_phot: numpy array. Absolute magnitudes in the required filters.
    - abs_phot_err: numpy array. Uncertainties on absolute magnitudes in the required filters.
    - abs_phot: numpy array. Apparent magnitudes in the required filters.
    - abs_phot_err: numpy array. Uncertainties on apparent magnitudes in the required filters.
    - mag_limits: numpy array. Limit absolute magnitudes corresponding to input curve.
    - mag_limits_err: numpy array. Uncertainties on limit absolute magnitudes.
    - mag_limits_app: numpy array. Limit apparent magnitudes corresponding to input curve.
    - mag_limits_app_err: numpy array. Uncertainties on limit apparent magnitudes.
    - separations: numpy array. Input separations, if 'file_type'='contrast_separation'; zero-filled array otherwise.
    - band: string. Input stellar_parameters['band'].

    Built-in methods:

    1) __len__
    The len of a CurveObject instance is equal to the number of unique values of contrasts_mag (rounded to 2 digits).

    Methods (use help() to have more detailed info):

    1) compute_mass_limits
    Converts the magnitude limit map stored in a CurveObject instance into a mass limit map.

    2) collapse_2D_map_to_1D
    Collapses a 2D (x, y) / 3D (lambda, x, y) mass limit map into an azimuthally averaged 1D (sep) / 2D (lambda, sep) curve, respectively.
    
    """

    def __init__(self, file, file_type, stellar_parameters, **kwargs):

        if isinstance(file_type, str) == False:
            raise TypeError("Invalid input for keyword 'file_type'. Please provide a string named 'contrast_separation' or 'contrast_map'.")
        if file_type not in ['contrast_separation', 'contrast_map']:
            raise ValueError("Invalid input for keyword 'file_type'. Please provide a string named 'contrast_separation' or 'contrast_map'.")
        
        if isinstance(stellar_parameters,dict) == False:
            wrong_type = str(type(stellar_parameters)).replace('<class ','').replace('>','')
            raise TypeError('Invalid input of type {0}. Please provide a dictionary with stellar parameters.'.format(wrong_type))
        
        self.__input = copy.deepcopy(kwargs)
        self.file = file
        self.header = fits.getheader(file)
        self.stellar_parameters = stellar_parameters
        self.file_type = file_type
        
        try:
            self.platescale = fits.getheader(self.file)['PIXTOARC']
        except KeyError:
            self.platescale = float(input('Platescale not found. Please insert a value [mas/px]: '))
             
        necessary_parameters = ['parallax', 'parallax_error', 'app_mag', 'app_mag_error', 'band', 'age']

        for param in necessary_parameters:
            if param not in stellar_parameters.keys():
                raise KeyError("A valid value for '{0}' is required to be present in the 'stellar_parameter' dictionary.".format(param))
                
        if ('age_error' not in stellar_parameters.keys()) & (('age_min' not in stellar_parameters.keys()) | ('age_max' not in stellar_parameters.keys())):
            raise KeyError("One between 'age_error' and the 'age_min'/'age_max' doublet must be present in the 'stellar_parameter' dictionary.")
        elif ('age_error' in stellar_parameters.keys()) & (('age_min' in stellar_parameters.keys()) | ('age_max' in stellar_parameters.keys())):
            raise KeyError("Only one between 'age_error' and the 'age_min'/'age_max' doublet must be present in the 'stellar_parameter' dictionary.")
        elif ('age_error' not in stellar_parameters.keys()) & (('age_min' not in stellar_parameters.keys()) | ('age_max' not in stellar_parameters.keys())):
            raise KeyError("Both 'age_min'/'age_max' must be present in the 'stellar_parameter' dictionary.")
            
        if (('ebv' not in stellar_parameters.keys()) | ('ebv_error' not in stellar_parameters.keys())) & (('ra' not in stellar_parameters.keys()) | ('dec' not in stellar_parameters.keys())):
            raise KeyError("One doublet between 'ebv'/'ebv_error' or 'ra'/'dec' must be present in the 'stellar_parameter' dictionary.")
        
        
        star_app_phot, star_app_phot_err = stellar_parameters['app_mag'], stellar_parameters['app_mag_error']
        par, par_err = stellar_parameters['parallax'], stellar_parameters['parallax_error']
        band = stellar_parameters['band']
        
        if 'ebv' in stellar_parameters:
            ebv, ebv_err = stellar_parameters['ebv'], stellar_parameters['ebv_error']
        else:
            if 'ext_map' in stellar_parameters:
                if stellar_parameters['ext_map'] not in ['leike', 'stilism']:
                    raise ValueError("Invalid value provided for the keyword 'ext_map'. Choose a value between 'leike' or 'stilism'.")
                ext_map = stellar_parameters['ext_map']
            else: ext_map = 'leike'
            ebv = SampleObject.interstellar_ext(ra = stellar_parameters['ra'], dec = stellar_parameters['dec'], par = stellar_parameters['parallax'], ext_map = ext_map)
            
            if np.isnan(ebv):
                msg = """The E(B-V) automatically computed by madys is nan. This might be caused by two reasons:
                1) an invalid or wrong parallax was provided. Check carefully this value.
                2) the star is beyond the limits of the selected 3D map. Try to choose another 3D map.
                If the error still appears, please manually provide an E(B-V) value through the keyword 'ebv' in the input dictionary."""
                raise ValueError(msg)

            ebv_err = 0.
            self.stellar_parameters['ebv'] = ebv
            self.stellar_parameters['ebv_error'] = ebv_err
        
        
        self.data_unit = kwargs['data_unit'] if 'data_unit' in kwargs else 'flux'
        rescale_flux = kwargs['rescale_flux'] if 'rescale_flux' in kwargs else 1.
        
        if self.data_unit not in ['flux','magnitude']:
            raise ValueError("Please select a valid unit for the contrast curve: 'flux' if flux ratios, 'magnitude' if in mag units.")
        
        if band not in list(stored_data['filters'].keys()):
            msg = """Invalid filter provided: {0}.
            \nAvailable filters: {1}.
            \nPlease provide a valid filter.
            """.format(band,', '.join(list(stored_data['filters'].keys())))
            raise ValueError(msg)

        if file_type == 'contrast_separation':
            data = fits.getdata(file)
            contrasts, separations = data[:,0].ravel(), data[:,1].ravel()
            w, = np.where(contrasts != 0)
            contrasts, separations = contrasts[w], separations[w]
        if file_type == 'contrast_map':
            contrasts = fits.getdata(file)
            contrasts = np.where((contrasts>0) & (contrasts<1), contrasts, np.nan)        
            separations = np.zeros_like(contrasts)
                
                
        if self.data_unit == 'flux':
            contrasts = np.where((contrasts>0) & (contrasts<1), contrasts, np.nan)
            self.contrasts = contrasts*rescale_flux
            self.contrasts_mag = -2.5*np.log10(self.contrasts)
        else:
            contrasts = np.where(contrasts>0, contrasts, np.nan)
            self.contrasts = 10**(-0.4*contrasts)*rescale_flux
            self.contrasts_mag = contrasts-2.5*np.log10(rescale_flux)
            
        self.contrasts_mag = np.round(self.contrasts_mag, 2)
        
        self.star_abs_phot, self.star_abs_phot_err = SampleObject.app_to_abs_mag(star_app_phot,par,app_mag_error=star_app_phot_err,parallax_error=par_err,ebv=ebv,ebv_error=ebv_err,filters=[band])
        self.mag_limits = self.contrasts_mag + self.star_abs_phot
        self.mag_limits_err = self.star_abs_phot_err
        self.mag_limits_app = self.contrasts_mag + star_app_phot
        self.mag_limits_app_err = star_app_phot_err
        self.separations = separations
        self.band = band

    def __len__(self):
        return len(np.unique(self.contrasts_mag))


    def _mask_contrast_map(self, radius):
        
        if radius is None:
            return self.contrasts_mag
        elif radius < 0.01:
            return self.contrasts_mag

        if self.file_type == 'contrast_map':
            n_dim = len(self.contrasts_mag.shape)
            if n_dim == 2:
                len0, len1 = self.contrasts_mag.shape
            else:
                __, len0, len1 = self.contrasts_mag.shape
                
            center0, center1 = int(len0/2), int(len1/2)
            radius_px2 = (radius/self.platescale)**2

            cm = copy.deepcopy(self.contrasts_mag)

            f = lambda x, y: ((y-center1)**2+(x-center0)**2) < radius_px2
            x_arr = np.arange(len0)
            y_arr = np.arange(len1)
            YY, XX = np.meshgrid(y_arr, x_arr)
            index = f(XX,YY)
            if n_dim == 2:
                cm[index] = np.nan
            elif n_dim == 3:
                cm[:, index] = np.nan
        else:
            cm = copy.deepcopy(self.contrasts_mag)
            w, = np.where(self.separations < radius/1000)
            cm[w] = np.nan

        return cm

    def _get_mask_size_from_header(self):

        mask_sizes_sphere = {'N_ALC_YJH_S': 92.5, 'N_ALC_YJH_L': 120.,
                             'N_ALC_YJ_S': 72.5, 'N_ALC_YJ_L': 92.5,
                             'N_ALC_Ks': 120.}
        mask_size_gpi = {'Y': 78.1, 'J': 92.35, 'H': 123.35,
                         'K1': 123.35, 'K2': 123.35}

        instrument = self.header['INSTRUME']
        size, found = 0, False
        if instrument == 'SPHERE':
            try:
                size = mask_sizes_sphere[self.header['HIERARCH ESO INS COMB ICOR']]
                found = True
            except KeyError:
                pass
        elif instrument == 'GPI':
            try:
                mask_name = self.header['APODIZER']
                for key in mask_size_gpi.keys():
                    if '_'+key+'_' in mask_name:
                        size = mask_size_gpi[key]
                        found = True
                        break
            except KeyError:
                pass

        return size, found

    def compute_mass_limits(self, model_version, mask_radius=0, to_file=True, output_path=None, assume_resolved=False, **kwargs):

        """
        Function that turns a contrast limit map into a mass limit map.
        Additionally, if self.file_type='contrast_map' and to_file!=False, it also produces and saves an azimuthally averaged 1D (sep) curve / 2D (lambda, sep) curve.
            Input:
            - model_version: string, required. A valid model_version for MADYS. Use ModelHandler.available() to return a list of available models.
            - mask_radius: int or float or NoneType, optional. Coronagraphic mask radius (mas). Possible values are:
                - None: no mask is used;
                - 0: the program tries to guess the appropriate mask radius. If not found, no mask is used.
                - any value > 0: used as a custom mask radius.
              Default value: 0.
            - to_file: bool or string, optional. Use a bool to select whether to export the result to a file or not.
              Alternatively, one can provide a string with the full path to the output file. Default: True.
            - output_path: string, optional. Only used if to_file=True. Full output path for the output file. If not set, it uses the same path where self.file is located.
            - assume_resolved: bool, optional. Only used if output_path is set. Whether to consider 'output_path' as a full path (True), or the last directory of a larger path.
              If False, the output path will be os.file.basename(self.file)+output_path. Default: False.
            - feh: float, optional. See IsochroneGrid() docstrings for info.
            - afe: float, optional. See IsochroneGrid() docstrings for info.
            - he: float, optional. See IsochroneGrid() docstrings for info.
            - v_vcrit: float, optional. See IsochroneGrid() docstrings for info.
            - f_spot: float, optional. See IsochroneGrid() docstrings for info.
            - B: int, optional. See IsochroneGrid() docstrings for info.
            - m_unit: string, optional. Unit of measurement of the resulting mass. Choose either 'm_sun' or 'm_jup'. Default: 'm_jup'.
            - n_try: int, optional. Number of Monte Carlo draws for uncertainty estimation. Default: 10000.
            Output:
            - res: dict. Output mass curves/maps, shaped as the input curve/map. In particular, it stores the keywords:
                - mass: numpy array. Output curve, defined as the mean of the posterior mass distribution.
                - mass_min: numpy array. Upper mass curve, defined as exp(mean-std) of the posterior log-mass distribution.
                - mass_max: numpy array. Lower mass curve, defined as exp(mean+std) of the posterior log-mass distribution.

        """

        if (isinstance(mask_radius, float) == False) & (isinstance(mask_radius, int) == False) & (mask_radius is not None):
            raise TypeError("'mask_radius' must be either a float, an int, or None.")
            
        custom_mask = True
        if mask_radius is not None:
            if mask_radius < 0.01: 
                auto_mask, use_mask = self._get_mask_size_from_header()
                if use_mask:
                    mask_radius = auto_mask
                    custom_mask = False
                else:
                    print('Warning: the current mask was not recognized. Using mask_radius = 0 mas. If you want to use a mask, manually set mask_radius to a value > 0.')
        else: mask_radius = 0 

        contrasts_mag = self._mask_contrast_map(mask_radius)

        n_dim = len(contrasts_mag.shape)
        if n_dim>3:
            raise ValueError("The no. of dimensions of this dataset is {0} > 3. Are you sure about input file?".format(n_dim))

        t0 = time.perf_counter()
        p = np.array(['feh','he','afe','v_vcrit','fspot','B'])
        dic = {}
        for kw in p:
            if kw in kwargs: dic[kw] = kwargs[kw]
        model_params = ModelHandler._find_match(model_version, dic, list(stored_data['complete_model_list'].keys()))

        ModelHandler._find_model_grid(model_version, model_params)        

        try:
            model_p = ModelHandler._available_parameters(model_version)
        except ValueError as e:
            msg = """You decided not to download any grid for model_version """+model_version+""".
            However, the relative folder is empty, so MADYS does not have any model to compare data with.
            Re-run the program, downloading at least one model when prompted.
            Program ended."""
            e.args = (msg,)
            raise

        for kw in p:
            if kw in kwargs: 
                try:
                    kwargs[kw] = model_params[kw]
                except KeyError:
                    print('Input parameter {0} is not supported by model {1} and was not used.'.format(kw,model_version))


        kwargs['mass_range'] = IsochroneGrid._get_mass_range([1e-6,1e+6], model_version, dtype='mass', **kwargs)
        m_unit = kwargs['m_unit'] if 'm_unit' in kwargs else 'm_jup'
        n_try = kwargs['n_try'] if 'n_try' in kwargs else 10000

        phot, phot_err = self.mag_limits, self.mag_limits_err
        app_phot, app_phot_err = self.mag_limits_app, self.mag_limits_app_err
        star_app_phot, star_app_phot_err = self.stellar_parameters['app_mag'], self.stellar_parameters['app_mag_error']
        par, par_err, ebv, ebv_err = self.stellar_parameters['parallax'], self.stellar_parameters['parallax_error'], \
        self.stellar_parameters['ebv'], self.stellar_parameters['ebv_error']

        if self.file_type == 'contrast_map':
            used_contrasts_mag = np.unique(contrasts_mag)
        else:
            used_contrasts_mag = contrasts_mag

        if 'age_error' in self.stellar_parameters.keys():
            age_fit_type = 1
            no_age = 0
            while no_age < n_try:
                sampled_ages = self.stellar_parameters['age'] + np.random.randn(2*n_try)*self.stellar_parameters['age_error']
                no_age = np.sum(sampled_ages>=1)
            sampled_ages = np.sort(sampled_ages[sampled_ages>=1][:n_try])

            random_par, random_ebv = par + np.random.randn(n_try)*par_err, ebv + np.random.randn(n_try)*ebv_err

        elif 'age_min' in self.stellar_parameters.keys():
            age_fit_type = 2
            n_random_try = int(np.sqrt(n_try))
            sampled_ages = np.linspace(self.stellar_parameters['age_min'], self.stellar_parameters['age_max'], n_random_try-1)
            sampled_ages = np.sort(np.concatenate((sampled_ages, [self.stellar_parameters['age']])))
            i_age = np.where(sampled_ages == self.stellar_parameters['age'])[0]

            sampled_ages = np.tile(sampled_ages, n_random_try)

            random_par, random_ebv = par + np.random.randn(n_random_try)*par_err, ebv + np.random.randn(n_random_try)*ebv_err
            random_par[i_age], random_ebv[i_age] = par, ebv
            random_par, random_ebv = np.repeat(random_par, n_random_try), np.repeat(random_ebv, n_random_try)

        kwargs['age_range'] = sampled_ages
        th_model = IsochroneGrid(model_version, self.band, search_model=False, **kwargs)
        iso_mass, iso_age, iso_filt, iso_data = th_model.masses, th_model.ages, th_model.filters, th_model.data
        iso_mass_log = np.log10(iso_mass)
        l = iso_data.shape
        n_masses, n_ages, n_points = l[0], l[1], len(used_contrasts_mag)

        if age_fit_type == 1:

            altered_mag_limits = np.zeros([n_points,n_try])
            for j in range(n_try):
                star_abs_phot = SampleObject.app_to_abs_mag(star_app_phot,random_par[j],ebv=random_ebv[j],filters=[self.band])
                altered_mag_limits[:,j] = used_contrasts_mag + star_abs_phot
            mag_limits_err = self.mag_limits_err

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                reshuffled_masses = np.full([n_try,n_points], np.nan)
                iso_indices = np.argsort(iso_data[:,:,0], axis = 0)
                for j in range(n_try):
                    if np.isnan(np.nanmin(iso_data[:,j,0])): continue
                    iso_index = iso_indices[:,j]
                    iso_index = np.argsort(iso_data[:,j,0])
                    iso_row = iso_data[iso_index,j,0]
                    diff = np.searchsorted(iso_row, altered_mag_limits[:,j])-1
                    diff = np.where((altered_mag_limits[:,j]-iso_row[diff])>(iso_row[diff+1]-altered_mag_limits[:,j]),diff+1,diff)
                    is_mag_not_nan = (np.isnan(iso_row[diff])==False) & (np.isnan(altered_mag_limits[:,j])==False)

                    reshuffled_masses[j,is_mag_not_nan] = iso_mass_log[iso_index[diff]][is_mag_not_nan]
                    reshuffled_masses[j,is_mag_not_nan] = iso_mass_log[iso_index[diff]][is_mag_not_nan]

                log_m_fit = np.nanmean(reshuffled_masses, axis = 0)
                log_s_fit = np.nanstd(reshuffled_masses, axis = 0)
                m_fit, m_min, m_max = 10**log_m_fit, 10**(log_m_fit-log_s_fit), 10**(log_m_fit+log_s_fit)

        elif age_fit_type == 2:

            altered_mag_limits = np.zeros([n_points,n_try])
            for j in range(n_try):
                star_abs_phot = SampleObject.app_to_abs_mag(star_app_phot,random_par[j],ebv=random_ebv[j],filters=[self.band])
                altered_mag_limits[:,j] = used_contrasts_mag + star_abs_phot
            mag_limits_err = self.mag_limits_err

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                tt0 = time.perf_counter()
                reshuffled_masses = np.full([n_try,n_points], np.nan)
                iso_indices = np.tile(np.argsort(iso_data[:,:n_random_try,0], axis = 0), n_random_try)
                for j in range(n_try):
                    if np.isnan(np.nanmin(iso_data[:,j,0])): continue
                    iso_index = iso_indices[:,j]
                    iso_row = iso_data[iso_index,j,0]
                    diff = np.searchsorted(iso_row, altered_mag_limits[:,j])-1
                    diff = np.where((altered_mag_limits[:,j]-iso_row[diff])>(iso_row[diff+1]-altered_mag_limits[:,j]),diff+1,diff)
                    is_mag_not_nan = (np.isnan(iso_row[diff])==False) & (np.isnan(altered_mag_limits[:,j])==False)

                    reshuffled_masses[j,is_mag_not_nan] = iso_mass_log[iso_index[diff]][is_mag_not_nan]
                    reshuffled_masses[j,is_mag_not_nan] = iso_mass_log[iso_index[diff]][is_mag_not_nan]

                reshuffled_masses = reshuffled_masses.reshape([n_random_try, n_random_try, n_points])
                log_m_fit = reshuffled_masses[i_age, i_age, :].ravel()
                m_perc = np.nanpercentile(reshuffled_masses, [16, 84], axis = 0)
                m_fit, m_min, m_max = 10**log_m_fit, 10**np.nanmin(m_perc[0], axis = 0), 10**np.nanmax(m_perc[1], axis = 0)  

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            if self.file_type == 'contrast_map':
                m_fit = CurveObject._fill_array_like(contrasts_mag, m_fit)
                m_min = CurveObject._fill_array_like(contrasts_mag, m_min)
                m_max = CurveObject._fill_array_like(contrasts_mag, m_max)

            res={'mass':m_fit, 'mass_min':m_min, 'mass_max':m_max,
                 }

            if m_unit.lower()=='m_jup':
                res['mass']*=M_sun.value/M_jup.value
                res['mass_min']*=M_sun.value/M_jup.value
                res['mass_max']*=M_sun.value/M_jup.value
                if 'all_solutions' in dic.keys():
                    for i in range(len(res['all_solutions'])):
                        if 'mass' in res['all_solutions'][i].keys():
                            res['all_solutions'][i]['mass']*=M_sun.value/M_jup.value

        print('Execution ended. Elapsed time: '+'{:.0f}'.format(time.perf_counter()-t0)+' s.')

        output_file = None
        if isinstance(to_file, bool):
            if to_file:
                if output_path is None: 
                    output_path = os.path.dirname(self.file)
                else:
                    if assume_resolved == False:
                        output_path = os.path.join(os.path.dirname(os.path.dirname(self.file)), output_path)
                if os.path.isdir(output_path) == False: os.mkdir(output_path)
                file_split = os.path.basename(self.file).split('.')
                output_file = os.path.join(output_path, '.'.join(file_split[:-1])+'_mass_limits_2D_'+model_version+'.'+file_split[-1])
        elif isinstance(to_file,str):
            output_file = to_file

        if output_file is not None:

            if custom_mask:
                mask_header = 'Using custom coron. mask radius = {0:.2f} mas.'.format(mask_radius)
            else:
                mask_header = 'The coron. mask was recognized: using literature radius = {0:.2f} mas.'.format(mask_radius)

            if age_fit_type == 1:
                age_header = 'age = {0:.2f}+/-{1:.2f} Myr'.format(self.stellar_parameters['age'], self.stellar_parameters['age_error'])
            elif age_fit_type == 2:
                age_header = 'age = {0:.2f} Myr ({1:.2f} Myr, {2:.2f} Myr)'.format(self.stellar_parameters['age'], self.stellar_parameters['age_min'], self.stellar_parameters['age_max'])

            madys_header = """This file is a mass contrast curve [M_J].
            Generated using MADYS {0} on {1}.
            Photometric filter of the input contrast curve: {2}.
            Isochrone grids used for the conversion: {3}.
            Stellar parameters: parallax = {4:.2f}+/-{5:.2f} mas, {6},
            ebv = {7:.2f}+/-{8:.2f} mag.
            {9}""".format(MADYS_VERSION, 
                                              time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                                              self.stellar_parameters['band'], model_version,
                                              self.stellar_parameters['parallax'],
                                              self.stellar_parameters['parallax_error'],
                                              age_header,
                                              self.stellar_parameters['ebv'],
                                              self.stellar_parameters['ebv_error'],
                                              mask_header
                                              )


            if self.file_type == 'contrast_map':
                shape = res['mass'].shape
                new_shape = (3,) + shape
                final_array = np.zeros(new_shape)
                final_array[0] = res['mass']
                final_array[1] = res['mass_min']
                final_array[2] = res['mass_max']

                if n_dim == 2:
                    sep, array_1D = self.collapse_2D_map_to_1D(res['mass'])
                    __, array_1D_min = self.collapse_2D_map_to_1D(res['mass_min'])
                    __, array_1D_max = self.collapse_2D_map_to_1D(res['mass_max'])
                    averaged_array = np.zeros([3,len(sep),2])
                    averaged_array[0,:,0] = array_1D
                    averaged_array[1,:,0] = array_1D_min
                    averaged_array[2,:,0] = array_1D_max
                    averaged_array[:,:,1] = sep        
                elif n_dim == 3:
                    for i_wvl in range(shape[0]):
                        sep, array_1D = self.collapse_2D_map_to_1D(res['mass'][i_wvl])
                        if i_wvl == 0:
                            averaged_array = np.zeros([3,shape[0],len(sep),2])
                        __, array_1D_min = self.collapse_2D_map_to_1D(res['mass_min'][i_wvl])
                        __, array_1D_max = self.collapse_2D_map_to_1D(res['mass_max'][i_wvl])
                        averaged_array[0,i_wvl,:,0] = array_1D
                        averaged_array[1,i_wvl,:,0] = array_1D_min
                        averaged_array[2,i_wvl,:,0] = array_1D_max
                        averaged_array[:,i_wvl,:,1] = sep

                output_file1 = output_file.replace('_mass_limits_2D_','_mass_limits_1D_')
                hea = copy.deepcopy(self.header)
                for string in madys_header.replace('is a mass', 'is a 1D mass').split('\n'):
                    hea['HISTORY'] = string
                hdu = fits.PrimaryHDU(averaged_array, hea)
                hdu.writeto(output_file1, overwrite = True)

            else:
                shape = res['mass'].shape
                final_array = np.zeros([3,len(res['mass']),2])
                final_array[0,:,0] = res['mass']
                final_array[1,:,0] = res['mass_min']
                final_array[2,:,0] = res['mass_max']
                final_array[:,:,1] = self.separations        
                output_file = output_file.replace('_mass_limits_2D_','_mass_limits_1D_')

            hea = copy.deepcopy(self.header)
            madys_header = madys_header.replace('is a mass', 'is a 1D mass') if '_1D_' in output_file else madys_header.replace('is a mass', 'is a 2D mass')
            for string in madys_header.split('\n'):
                hea['HISTORY'] = string
            hdu = fits.PrimaryHDU(final_array, hea)
            hdu.writeto(output_file, overwrite = True)        

        return res
                    
    @staticmethod
    def _fill_array_like(x, array):

        shape = x.shape
        x = x.ravel()
        idx_sort = np.argsort(x)
        sorted_records_array = x[idx_sort]
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
        res = np.split(idx_sort, idx_start[1:])

        y = np.zeros(len(x.ravel()))
        for i in range(len(vals)):
            y[res[i]] = array[i]

        return y.reshape(shape)
    
    def collapse_2D_map_to_1D(self, data, center=None):

        """
        Function that collapses a 2D (x, y) / 3D (lambda, x, y) mass limit map into an azimuthally averaged 
        1D (sep) / 2D (lambda, sep) curve, respectively.
            Input:
            - data: numpy array, required. Output produced by compute_mass_limits().
            - center: list, optional. A two element list indicating the image center position, starting from 0.
              If None, it uses [len(x)/2, len(y)/2]. Default: None.
            Output:
            - sep: numpy array. Vector of separations [arcsec].
            - mass_curve: numpy array. Output 1D curve.
        """
        
        platescale = self.platescale

        shape = data.shape
        if center is None:
            center = [shape[0]/2, shape[1]/2]

        [X, Y] = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])
        R = np.sqrt(np.square(X) + np.square(Y))
        rad = np.arange(np.max(R))
        sep = rad*platescale/1000


        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            groups = np.searchsorted(rad-0.5,R.ravel())-1
            idx_sort = np.argsort(groups)
            sorted_records_array = groups[idx_sort]
            vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
            res = np.split(idx_sort, idx_start[1:])

            data_1d = data.ravel()
            mass_curve = np.zeros(len(sep))
            for i in range(len(sep)):
                mass_curve[vals[i]] = np.nanmedian(data_1d[res[i]])
            t05 = time.perf_counter()

        return sep, mass_curve
            
            
            
ModelHandler._check_updates()
ModelHandler._load_local_models()
ModelHandler._load_filters()
logo = make_logo()
