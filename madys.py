import sys
import copy
import warnings
from astropy.utils.exceptions import AstropyWarning
import logging
import numpy as np
from pathlib import Path
import os
from evolution import *
from scipy.interpolate import interp1d
from astropy.constants import M_jup,M_sun
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
from astroquery.gaia import Gaia
gaia = GaiaArchive()

"""
Class: MADYS

Tool for age and mass determination of young stellar and substellar objects. Given a list of stars:
- it retrieves and cross-matches photometry from Gaia and 2MASS
- corrects for interstellar extinction
- assesses the quality of each photometric measurement
- uses reliable photometric data to derive ages and masses of individual stars.
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
- max_tmass_q (1): worse 2MASS photometric flag ('ph_qual') still considered reliable. Possible values, ordered by decreasing quality: 'A','B','C','D','E','F','U','X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.
- max_wise_q (1): worse ALLWISE photometric flag ('ph_qual2') still considered reliable. Possible values, ordered by decreasing quality: 'A','B','C','U','Z','X'. For a given choice, excludes all measurements with a lower quality flag. Default: 'A'.

Attributes:
- file: corresponing to either file (1) or mock_file (2).
- path: working path, where all inputs and outputs are present.
- log_file: name of the log_file. Open it for details on the process outcome.
- phot_table: Table containing all retrieved data.
- abs_mag: absolute magnitudes in the required filters.
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
            - a 2D numpy array with 3 columns. The i-th row is used as fixed (mean_age,lower_age,upper_age) for the i-th star.
      Default: [1,1000]
    - n_steps: list, optional. Number of (mass, age) steps of the interpolated grid. Default: [1000,500].
    - verbose: bool, optional. Set to True to save the results in a file. Default: True.
    - feh: float, optional. Selects [Fe/H] of the isochrone set. Default: None (=0.00, solar metallicity).
    - afe: float, optional. Selects alpha enhancement [a/Fe] of the isochrone set. Default: None (=0.00).
    - v_vcrit: float, optional. Selects rotational velocity of the isochrone set. Default: None (=0.00, non-rotating).
    - fspot: float, optional. Selects fraction of stellar surface covered by star spots. Default: None (=0.00).
    - B: int, optional. Set to 1 to turn on the magnetic field (only for Dartmouth models). Default: 0.
    - ph_cut: float, optional. Maximum  allowed photometric uncertainty [mag]. Default: 0.2.
    - m_unit: string, optional. Unit of measurement of the resulting mass. Choose either 'm_sun' or 'm_jup'. Default: 'm_sun'.
    Output:
        (case 1: if age_range is a list or a 1D numpy array)
    - a_final: numpy array. Final age estimates, one element per star [Myr]
    - m_final: numpy array. Final mass estimates, one element per star [M_sun or M_jup]
    - a_err: numpy array. Error on age estimates, one element per star [Myr]
    - m_err: numpy array. Error on mass estimates, one element per star [M_sun or M_jup]
        (case 2: if age_range is a 2D numpy array)
    - a_final: numpy array. Final age estimates, one element per star [Myr]
    - m_final: numpy array. Final mass estimates, one element per star [M_sun or M_jup]
    - a_min: numpy array. Minimum age given by the user [Myr]
    - a_max: numpy array. Maximum age given by the user [Myr]
    - m_err_m: numpy array. Mass estimate corresponding to the a_min [M_sun or M_jup]
    - m_err_p: numpy array. Mass estimate corresponding to the a_max [M_sun or M_jup]
    
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
    - tofile: bool, optional. Saves the output to as .png image. Default: False.

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
    
"""

class MADYS(object):
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
            self.filters=np.array(['G','Gbp','Grp','G2','Gbp2','Grp2','J','H','K'])
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
                par=file['parallax']
                self.ebv=MADYS.interstellar_ext(ra=file['ra'],dec=file['dec'],par=par,ext_map=ext_map,logger=self.__logger)
                self.__logger.info('Extinction type: computed using '+ext_map+' extinction map')
            if 'parallax' in col0:
                par=file['parallax']
                par_err=file['parallax_err']
                self.abs_phot,self.abs_phot_err=MADYS.app_to_abs_mag(self.abs_phot,par,app_mag_error=self.abs_phot_err,parallax_error=par_err,ebv=self.ebv,filters=col)
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

            phot=np.full([nst,9],np.nan)
            phot_err=np.full([nst,9],np.nan)
            col=['edr3_gmag_corr','edr3_phot_bp_mean_mag','edr3_phot_rp_mean_mag','dr2_phot_g_mean_mag','dr2_phot_bp_mean_mag','dr2_phot_rp_mean_mag','j_m','h_m','ks_m']
            col_err=['edr3_phot_g_mean_mag_error','edr3_phot_bp_mean_mag_error','edr3_phot_rp_mean_mag_error','dr2_g_mag_error','dr2_bp_mag_error','dr2_rp_mag_error','j_msigcom','h_msigcom','ks_msigcom','w1mpro_error','w2mpro_error','w3mpro_error','w4mpro_error']

            for i in range(9):
                phot[:,i]=self.good_phot[col[i]].filled(np.nan)
                phot_err[:,i]=self.good_phot[col_err[i]].filled(np.nan)
            self.__app_phot=phot
            self.__app_phot_err=phot_err
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
                self.ebv=self.interstellar_ext(ra=ra,dec=dec,par=par,ext_map=ext_map,logger=self.__logger)
                self.__logger.info('Extinction type: computed using '+ext_map+' extinction map')
            self.abs_phot,self.abs_phot_err=self.app_to_abs_mag(self.__app_phot,par,app_mag_error=self.__app_phot_err,parallax_error=par_err,ebv=self.ebv,filters=self.filters)
            self.__logger.info('Input photometry: apparent, converted to absolute')

        logging.shutdown() 
        
    def sample_name_ext(self):
        sample_name=os.path.split(self.file)[1]
        i=0
        while sample_name[i]!='.': i=i+1
        self.__ext=sample_name[i:]
        self.__sample_name=sample_name[:i]
        
    def read_IDs(self):

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
                res=Simbad.query_objectids(self.ID[i])
                for rr in res:
                    if str(rr[0]).startswith('Gaia DR2'): self.GaiaID[i]=str(rr[0])#
                    

    def get_phot(self, save_phot, verbose=True):
        data=[]
        start = time.time()
        id_str = 'Gaia EDR3 ' if self.__id_type=='EDR3' else 'Gaia DR2 '
        id_sea = 'edr3.source_id' if self.__id_type=='EDR3' else 'dr2xmatch.dr2_source_id'
        for i in range(len(self.GaiaID)):            
            id=str(self.GaiaID[i]).split(id_str)[1]
            qstr="""
            select all
            edr3.designation as edr3_id, dr2.designation as dr2_id,
            tmass.designation as tmass_id, allwise.designation as allwise_id,
            apass.recno as apassdr9_id, sloan.objid as sloan_id,
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
            dr2.radial_velocity, dr2.radial_velocity_error,
            tmass.j_m, tmass.j_msigcom,
            tmass.h_m, tmass.h_msigcom,
            tmass.ks_m, tmass.ks_msigcom,
            tmass.ph_qual,
            tmass.ra as tmass_ra, tmass.dec as tmass_dec,
            allwise.w1mpro, allwise.w1mpro_error,
            allwise.w2mpro,allwise.w2mpro_error,
            allwise.w3mpro,allwise.w3mpro_error,
            allwise.w4mpro,allwise.w4mpro_error,
            allwise.cc_flags, allwise.ext_flag, allwise.var_flag, allwise.ph_qual as ph_qual_2, allwise.tmass_key,
            apass.b_v, apass.e_b_v, apass.vmag, apass.e_vmag, apass.bmag, apass.e_bmag, apass.r_mag, apass.e_r_mag, apass.i_mag, apass.e_i_mag,
            sloan.u, sloan.err_u, sloan.g, sloan.err_g, sloan.r, sloan.err_r, sloan.i, sloan.err_i, sloan.u, sloan.err_u
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
            LEFT OUTER JOIN
                gaiaedr3.allwise_best_neighbour AS allwisexmatch
                ON edr3.source_id = allwisexmatch.source_id
            LEFT OUTER JOIN
                gaiaedr3.tmass_psc_xsc_best_neighbour AS tmassxmatch
                ON edr3.source_id = tmassxmatch.source_id
            LEFT OUTER JOIN
                gaiadr1.tmass_original_valid AS tmass
                ON tmassxmatch.original_ext_source_id = tmass.designation
            LEFT OUTER JOIN
                gaiadr1.allwise_original_valid AS allwise
                ON allwisexmatch.original_ext_source_id = allwise.designation
            LEFT OUTER JOIN
                gaiaedr3.apassdr9_best_neighbour AS apassxmatch
                ON edr3.source_id = apassxmatch.source_id
            LEFT OUTER JOIN
                external.apassdr9 AS apass
                ON apassxmatch.clean_apassdr9_oid = apass.recno
            LEFT OUTER JOIN
                gaiaedr3.sdssdr13_best_neighbour AS sloanxmatch
                ON edr3.source_id = sloanxmatch.source_id
            LEFT OUTER JOIN
                external.sdssdr13_photoprimary as sloan
                ON sloanxmatch.clean_sdssdr13_oid = sloan.objid
            WHERE """ + id_sea + ' = ' + id
#            WHERE dr2xmatch.dr2_source_id = """ + id

            adql = QueryStr(qstr,verbose=False)
    #            t=timeit(gaia.query)(adql)

            try:
                t=gaia.query(adql)        
                t=MADYS.fix_double_entries(t,self.__id_type)
            except:
                qstr="""
                select all
                edr3.designation as edr3_id, dr2.designation as dr2_id,
                allwise.designation as allwise_id,
                apass.recno as apassdr9_id, sloan.objid as sloan_id,
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
                dr2.radial_velocity, dr2.radial_velocity_error,
                allwise.w1mpro, allwise.w1mpro_error,
                allwise.w2mpro,allwise.w2mpro_error,
                allwise.w3mpro,allwise.w3mpro_error,
                allwise.w4mpro,allwise.w4mpro_error,
                allwise.cc_flags, allwise.ext_flag, allwise.var_flag, allwise.ph_qual as ph_qual_2, allwise.tmass_key,
                apass.b_v, apass.e_b_v, apass.vmag, apass.e_vmag, apass.bmag, apass.e_bmag, apass.r_mag, apass.e_r_mag, apass.i_mag, apass.e_i_mag,
                sloan.u, sloan.err_u, sloan.g, sloan.err_g, sloan.r, sloan.err_r, sloan.i, sloan.err_i, sloan.u, sloan.err_u
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
                LEFT OUTER JOIN
                    gaiaedr3.allwise_best_neighbour AS allwisexmatch
                    ON edr3.source_id = allwisexmatch.source_id
                LEFT OUTER JOIN
                    gaiadr1.allwise_original_valid AS allwise
                    ON allwisexmatch.original_ext_source_id = allwise.designation
                LEFT OUTER JOIN
                    gaiaedr3.apassdr9_best_neighbour AS apassxmatch
                    ON edr3.source_id = apassxmatch.source_id
                LEFT OUTER JOIN
                    external.apassdr9 AS apass
                    ON apassxmatch.clean_apassdr9_oid = apass.recno
                LEFT OUTER JOIN
                    gaiaedr3.sdssdr13_best_neighbour AS sloanxmatch
                    ON edr3.source_id = sloanxmatch.source_id
                LEFT OUTER JOIN
                    external.sdssdr13_photoprimary as sloan
                    ON sloanxmatch.clean_sdssdr13_oid = sloan.objid
                WHERE """ + id_sea + ' = ' + id
    #            WHERE dr2xmatch.dr2_source_id = """ + id
                adql = QueryStr(qstr,verbose=False)
                t=gaia.query(adql)
                t=MADYS.fix_double_entries(t,self.__id_type)
                t=MADYS.fix_2mass(t)

            print(i,id)
            t=MADYS.fix_edr3(t)                
            
            with np.errstate(divide='ignore',invalid='ignore'):        
                edr3_gmag_corr, edr3_gflux_corr = self.correct_gband(t.field('edr3_bp_rp'), t.field('edr3_astrometric_params_solved'), t.field('edr3_phot_g_mean_mag'), t.field('edr3_phot_g_mean_flux'))
                edr3_bp_rp_excess_factor_corr = self.edr3_correct_flux_excess_factor(t.field('edr3_bp_rp'), t.field('edr3_phot_bp_rp_excess_factor'))
                edr3_g_mag_error, edr3_bp_mag_error, edr3_rp_mag_error = self.gaia_mag_errors(t.field('edr3_phot_g_mean_flux'), t.field('edr3_phot_g_mean_flux_error'), t.field('edr3_phot_bp_mean_flux'), t.field('edr3_phot_bp_mean_flux_error'), t.field('edr3_phot_rp_mean_flux'), t.field('edr3_phot_rp_mean_flux_error'))
                dr2_bp_rp_excess_factor_corr = self.dr2_correct_flux_excess_factor(t.field('dr2_phot_g_mean_mag'), t.field('dr2_bp_rp'), t.field('dr2_phot_bp_rp_excess_factor'))
                dr2_g_mag_error, dr2_bp_mag_error, dr2_rp_mag_error = self.gaia_mag_errors(t.field('dr2_phot_g_mean_flux'), t.field('dr2_phot_g_mean_flux_error'), t.field('dr2_phot_bp_mean_flux'), t.field('dr2_phot_bp_mean_flux_error'), t.field('dr2_phot_rp_mean_flux'), t.field('dr2_phot_rp_mean_flux_error'))
                t_ext=Table([edr3_gmag_corr, edr3_gflux_corr, edr3_bp_rp_excess_factor_corr, edr3_g_mag_error, edr3_bp_mag_error, edr3_rp_mag_error, dr2_bp_rp_excess_factor_corr, dr2_g_mag_error, dr2_bp_mag_error, dr2_rp_mag_error],
                    names=['edr3_gmag_corr', 'edr3_gflux_corr','edr3_phot_bp_rp_excess_factor_corr', 'edr3_phot_g_mean_mag_error', 'edr3_phot_bp_mean_mag_error', 'edr3_phot_rp_mean_mag_error', 'dr2_phot_bp_rp_excess_factor_corr', 'dr2_g_mag_error', 'dr2_bp_mag_error', 'dr2_rp_mag_error'],
                    units=["mag", "'electron'.s**-1", "", "mag", "mag", "mag", "", "mag", "mag", "mag"],
                    descriptions=['EDR3 G-band mean mag corrected as per Riello et al. (2020)', 'EDR3 G-band mean flux corrected as per Riello et al. (2020)', 'EDR3 BP/RP excess factor corrected as per Riello et al. (2020)','EDR3 Error on G-band mean mag', 'EDR3 Error on BP-band mean mag', 'EDR3 Error on RP-band mean mag', 'DR2 BP/RP excess factor corrected', 'DR2 Error on G-band mean mag', 'DR2 Error on BP-band mean mag', 'DR2 Error on RP-band mean mag'])
                data.append(hstack([self.ID[i], t, t_ext]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.phot_table=vstack(data)
        if save_phot == True:
            filename=os.path.join(self.path,(self.__sample_name+'_phot_table.csv'))
            ascii.write(self.phot_table, filename, format='csv', overwrite=True)
        if verbose == True:
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Total time needed to retrieve photometry for "+ np.str(len(self.GaiaID))+ " targets: - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))            

    def check_phot(self,**kwargs):
        if 'max_tmass_q' in kwargs:
            max_tmass_q=kwargs['max_tmass_q']
        else: max_tmass_q='A'
        if 'max_wise_q' in kwargs:
            max_wise_q=kwargs['max_wise_q']
        else: max_wise_q='A'

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

        tm_q = self.tmass_quality(t.field('ph_qual'),max_q=max_tmass_q)
        t['j_m'].mask[~tm_q[0]]=True
        t['h_m'].mask[~tm_q[1]]=True
        t['ks_m'].mask[~tm_q[2]]=True
        t['j_m'].fill_value = np.nan
        t['h_m'].fill_value = np.nan
        t['ks_m'].fill_value = np.nan

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
        relax=False
        self.mass_range = kwargs['mass_range'] if 'mass_range' in kwargs else [0.01,1.4]
        self.age_range = kwargs['age_range'] if 'age_range' in kwargs else [1,1000]
        self.n_steps = kwargs['n_steps'] if 'n_steps' in kwargs else [1000,500]
        verbose = kwargs['verbose'] if 'verbose' in kwargs else True
        self.feh = kwargs['feh'] if 'feh' in kwargs else None
        self.afe = kwargs['afe'] if 'afe' in kwargs else None
        self.v_vcrit = kwargs['v_vcrit'] if 'v_vcrit' in kwargs else None
        self.fspot = kwargs['fspot'] if 'fspot' in kwargs else None
        self.B = kwargs['B'] if 'B' in kwargs else 0
        self.ph_cut = kwargs['ph_cut'] if 'ph_cut' in kwargs else 0.2
        m_unit=kwargs['m_unit'] if 'm_unit' in kwargs else 'm_sun'
        
        self.__logger.info('Starting age determination...')
        iso_mass,iso_age,iso_filt,iso_data=MADYS.load_isochrones(model,self.filters,**kwargs)        
        self.__logger.info('Isochrones for model '+model+' correctly loaded')
        mass_range_str=["%.2f" % s for s in self.mass_range]
        try:
            age_range_str=["%s" % s for s in self.age_range]
        except TypeError: age_range_str=[str(self.age_range)]

        self.__logger.info('Input parameters for the model: mass range = ['+','.join(mass_range_str)+'] M_sun; age range = ['+','.join(age_range_str)+'] Myr')
        if self.feh==None: self.__logger.info('Metallicity: solar (use MADYS.info_models('+model+') for details).')
        else: self.__logger.info('Metallicity: [Fe/H]='+str(self.feh)+' (use MADYS.info_models('+model+') for details).')
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

        filt2=MADYS.where_v(MADYS.fix_filters(iso_filt,model),self.filters)

        phot=phot[:,filt2] #ordered columns. Cuts unnecessary columns
        phot_err=phot_err[:,filt2] #ordered columns. Cuts unnecessary columns
        
        mask=False
        if isinstance(self.age_range,np.ndarray):            
            if len(self.age_range.shape)==1:
                relax=True
                mask=True
                if len(self.age_range)!=l0[0]:
                    self.__logger.error('The number of stars is not equal to the number of input ages. Check the length of your input ages')
                    raise ValueError('The number of stars is not equal to the number of input ages.')
                a_final=self.age_range
                a_err=np.zeros(l0[0])
            elif len(self.age_range[0])==3:
                relax=True
                mask=True
                i_age=np.zeros(self.age_range.shape,dtype=int)
                for i in range(l0[0]):
                    i_age[i,:]=MADYS.closest(iso_age,self.age_range[i,:])
                if len(self.age_range)!=l0[0]:
                    self.__logger.error('The number of stars is not equal to the number of input ages. Check the length of your input ages')
                    raise ValueError('The number of stars is not equal to the number of input ages.')
                a_final=iso_age[i_age[:,0]]
                a_min=iso_age[i_age[:,1]]
                a_max=iso_age[i_age[:,2]]
#                a_min=np.full(xlen,np.nan)
#                a_max=np.full(xlen,np.nan)
                m_err_p=np.full(xlen,np.nan)
                m_err_m=np.full(xlen,np.nan)
        else:
            a_final=np.full(xlen,np.nan)
            a_err=np.full(xlen,np.nan)

        m_final=np.full(xlen,np.nan)
        m_err=np.full(xlen,np.nan)
        l=iso_data.shape
        sigma=np.full(([l[0],l[1],ylen]),np.nan)
        
        if l[1]==1: relax=True #if just one age is provided, no need for strict conditions on photometry
        
        if relax:
            sigma=np.full(([l[0],1,ylen]),np.nan)
            for i in range(xlen):
                w,=np.where(MADYS.is_phot_good(phot[i,:],phot_err[i,:],max_phot_err=self.ph_cut))
                if len(w)==0: continue
                if 'i_age' in locals(): i00=i_age[i,0]
                elif mask: i00=i
                else: i00=0
                b=np.zeros(len(w),dtype=bool)
                for h in range(len(w)):
                    ph=phot[i,w[h]]
                    sigma[:,0,w[h]]=((iso_data[:,i00,w[h]]-ph)/phot_err[i,w[h]])**2
                    ii=divmod(np.nanargmin(sigma[:,:,w[h]]), sigma.shape[1])+(w[h],) #builds indices (i1,i2,i3) of closest theor. point
                    if abs(iso_data[ii[0],i00,w[h]]-ph)<0.2: b[h]=True #if the best theor. match is more than 0.2 mag away, discards it
                if np.sum(b)<0.5: continue
                w2=w[b]
                cr=np.sum(sigma[:,:,w2],axis=2) 
                est,ind=MADYS.min_v(cr)
                m_final[i]=iso_mass[ind[0]]
                a_final[i]=iso_age[i00]
                if 'i_age' in locals():
                    for h in range(len(w2)):
                        ph=phot[i,w2[h]]
                        sigma[:,0,w2[h]]=((iso_data[:,i_age[i,1],w2[h]]-ph)/phot_err[i,w2[h]])**2
                    cr1=np.sum(sigma[:,:,w2],axis=2)
                    est1,ind1=MADYS.min_v(cr1)
                    m_f1=iso_mass[ind1[0]]
                    a_min[i]=iso_age[i_age[i,1]]
                    for h in range(len(w2)):
                        ph=phot[i,w2[h]]
                        sigma[:,0,w2[h]]=((iso_data[:,i_age[i,2],w2[h]]-ph)/phot_err[i,w2[h]])**2
                    cr1=np.sum(sigma[:,:,w2],axis=2)
                    est1,ind1=MADYS.min_v(cr1)
                    m_f2=iso_mass[ind1[0]]
                    a_max[i]=iso_age[i_age[i,2]]
                    m_err_m[i]=m_final[i]-m_f1
                    m_err_p[i]=m_f2-m_final[i]                    
                else:
                    m_f1=np.zeros(20)
                    a_f1=np.zeros(20)
                    for j in range(20):
                        phot1=phot+phot_err*np.random.normal(size=(xlen,ylen))
                        for h in range(len(w2)):
                            sigma[:,0,w2[h]]=((iso_data[:,i00,w2[h]]-phot1[i,w2[h]])/phot_err[i,w2[h]])**2
                        cr1=np.sum(sigma[:,:,w2],axis=2)
                        est1,ind1=MADYS.min_v(cr1)
                        m_f1[j]=iso_mass[ind1[0]]
                        a_f1[j]=iso_age[i00]
                    m_err[i]=np.std(m_f1,ddof=1)
                    a_err[i]=np.std(a_f1,ddof=1)            
        else:
            for i in range(xlen):
                w,=np.where(MADYS.is_phot_good(phot[i,:],phot_err[i,:],max_phot_err=self.ph_cut))
                if len(w)==0: continue
                b=np.zeros(len(w),dtype=bool)
                for h in range(len(w)):
                    ph=phot[i,w[h]]
                    sigma[:,:,w[h]]=((iso_data[:,:,w[h]]-ph)/phot_err[i,w[h]])**2
                    ii=divmod(np.nanargmin(sigma[:,:,w[h]]), sigma.shape[1])+(w[h],) #builds indices (i1,i2,i3) of closest theor. point
                    if abs(iso_data[ii]-ph)<0.2: b[h]=True #if the best theor. match is more than 0.2 mag away, discards it           
                w2=w[b]
                if len(w2)<3: continue #at least 3 filters needed for the fit
                cr=np.sum(sigma[:,:,w2],axis=2)
                est,ind=MADYS.min_v(cr)
                crit1=np.sort([sigma[ind+(w2[j],)] for j in range(len(w2))])
                crit2=np.sort([abs(iso_data[ind+(w2[j],)]-phot[i,w2[j]]) for j in range(len(w2))])
                if (crit1[2]<9) | (crit2[2]<0.1): #the 3rd best sigma < 3 and the 3rd best solution closer than 0.1 mag  
                    m_final[i]=iso_mass[ind[0]]
                    a_final[i]=iso_age[ind[1]]
                    m_f1=np.zeros(20)
                    a_f1=np.zeros(20)
                    for j in range(20):
                        phot1=phot+phot_err*np.random.normal(size=(xlen,ylen))
                        for h in range(len(w2)):
                            sigma[:,:,w2[h]]=((iso_data[:,:,w2[h]]-phot1[i,w2[h]])/phot_err[i,w2[h]])**2
                        cr1=np.sum(sigma[:,:,w2],axis=2)
                        est1,ind1=MADYS.min_v(cr1)
                        m_f1[j]=iso_mass[ind1[0]]
                        a_f1[j]=iso_age[ind1[1]]
                    m_err[i]=np.std(m_f1,ddof=1)
                    a_err[i]=np.std(a_f1,ddof=1)
                
        if m_unit=='m_jup':
            m_final*=M_sun.value/M_jup.value
            m_err*=M_sun.value/M_jup.value
            if 'i_age' in locals():
                m_err_p*=M_sun.value/M_jup.value
                m_err_m*=M_sun.value/M_jup.value

        if verbose==True:
            if type(self.GaiaID)==Table: star_names=self.GaiaID['ID'].value
            else: star_names=self.GaiaID.value
            filename=os.path.join(self.path,str(self.__sample_name+'_ages_'+model+'.txt'))
            f=open(filename, "w+")
            if 'i_age' in locals():
                f.write(tabulate(np.column_stack((star_names,m_final,m_err_m,m_err_p,a_final,a_min,a_max,self.ebv)),
                                 headers=['ID','MASS','MASS_ERROR_M','MASS_ERROR_P','AGE','AGE_MIN','AGE_MAX','E(B-V)'], tablefmt='plain', stralign='right', numalign='right', floatfmt=(None,".2f",".2f",".2f",".2f",".2f",".2f",".3f")))
            else:
                f.write(tabulate(np.column_stack((star_names,m_final,m_err,a_final,a_err,self.ebv)),
                                 headers=['ID','MASS','MASS_ERROR','AGE','AGE_ERROR','E(B-V)'], tablefmt='plain', stralign='right', numalign='right', floatfmt=(None,".2f",".2f",".2f",".2f",".3f")))
            f.close()
            self.__logger.info('Age determination ended. Results saved in '+filename)
        else:
            self.__logger.info('Age determination ended. Results not saved in any file because "verbose" is set to False.')

        logging.shutdown()

        if 'i_age' in locals(): return a_final,m_final,a_min,a_max,m_err_m,m_err_p
        else: return a_final,m_final,a_err,m_err
                
    @staticmethod
    def axis_range(col_name,col_phot,stick_to_points=False):
        try:
            len(col_phot)
            cmin=min(col_phot)-0.1
            cmax=min(70,max(col_phot))+0.1
        except TypeError:
            cmin=col_phot-0.1
            cmax=min(70,col_phot)+0.1
        
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
            if model in ['bt_settl','starevol','spots','dartmouth','ames_cond','ames_dusty','bt_nextgen','nextgen']:
                if col=='G': col2='G2'
                elif col=='Gbp': col2='Gbp2'
                elif col=='Grp': col2='Grp2'
                elif col=='G-Gbp': col2='G2-Gbp2'
                elif col=='Gbp-G': col2='Gbp2-G2'
                elif col=='G-Grp': col2='G2-Gbp2'
                elif col=='Grp-G': col2='Grp2-G2'
                elif col=='Gbp-Grp': col2='Gbp2-Grp2'
                elif col=='Grp-Gbp': col2='Grp2-Gbp2'
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
            
        iso=MADYS.load_isochrones(model,self.filters,**kwargs)

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
        plt.arrow(x_range[0]+0.2*(x_range[1]-x_range[0]),y_range[0]+0.1*(y_range[1]-y_range[0]),-np.median(x_ext),-np.median(y_ext),head_width=0.05, head_length=0.1, fc='k', ec='k', label='reddening')

        x1=x
        y1=y

        if type(plot_ages)!=bool:
            for i in range(len(plot_ages)):
                ii=MADYS.closest(iso_ages,plot_ages[i])
                plt.plot(col_th[:,ii],mag_th[:,ii],label=str(plot_ages[i])+' Myr')

        if type(plot_masses)!=bool:
            for i in range(len(plot_masses)):
                im=MADYS.closest(iso_masses,plot_masses[i])
                plt.plot(col_th[im,:],mag_th[im,:],linestyle='dashed',color='gray')
                c=0
                while (np.isfinite(col_th[im,c])==0) | (np.isfinite(mag_th[im,c])==0): 
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
            q1=np.where(abs(dr2_bp_rp_excess_factor)>3*s1(dr2_phot_g_mean_mag),0,1)
        q1[dr2_bp_rp_excess_factor.mask]=0
        return q1

    @staticmethod
    def edr3_quality(edr3_bp_rp_excess_factor,edr3_phot_g_mean_mag):
        s1=lambda x: 0.0059898+8.817481e-12*x**7.618399
        with np.errstate(invalid='ignore'):
            q1=np.where(abs(edr3_bp_rp_excess_factor)>3*s1(edr3_phot_g_mean_mag),0,1)
        q1[edr3_bp_rp_excess_factor.mask]=0
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
    
    @staticmethod    
    def fix_2mass(t):
        ra0=t['ra'].value[0]
        dec0=t['dec'].value[0]
        pmra0=t['edr3_pmra'].value[0]
        pmdec0=t['edr3_pmdec'].value[0]
        ep=t['edr3_epoch'].value[0]
        ra1=ra0-(ep-2000)*pmra0/3.6e+6
        dec1=dec0-(ep-2000)*pmdec0/3.6e+6

        v = Vizier(columns=["*", "+_r", "Cntr"], catalog="II/246")
        no_res=False
        try:
            res=v.query_region(SkyCoord(ra=ra1, dec=dec1,unit=(u.deg, u.deg),frame='icrs'),width="10s",catalog=["2MASS"])[0]
        except IndexError: no_res=True
        else:
            w,=np.where(res['Cntr']==t['tmass_key'])
            if len(w)==0: no_res=True
        finally:
            if no_res:
                t_ext=Table([[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],['XXX'],[ra1],[dec1]],
                    names=['j_m', 'h_m','ks_m','j_msigcom', 'h_msigcom','ks_msigcom','ph_qual','tmass_ra','tmass_dec'],
                    units=["mag","mag", "mag","mag","mag", "mag","","deg","deg"])
            else:
                t_ext=Table([res['Jmag'][w],res['Hmag'][w],res['Kmag'][w],res['e_Jmag'][w],res['e_Hmag'][w],res['e_Kmag'][w],res['Qflg'][w],res['RAJ2000'][w],res['DEJ2000'][w]],
                    names=['j_m', 'h_m','ks_m','j_msigcom','h_msigcom','ks_msigcom','ph_qual','tmass_ra','tmass_dec'],
                    units=["mag","mag","mag","mag","mag","mag","","deg","deg"])
            t_ext['ph_qual']=MaskedColumn(t_ext['ph_qual'],dtype=object)

        return hstack([t, t_ext])
    
    @staticmethod    
    def fix_edr3(t):
        if t['ra'].mask==False: return t

        print("Fix EDR3")
        ra0=t['dr2_ra'].value[0]
        dec0=t['dr2_dec'].value[0]
        pmra0=t['dr2_pmra'].value[0]
        pmdec0=t['dr2_pmdec'].value[0]
        ep=t['dr2_epoch'].value[0]
        ra1=ra0+(2016-ep)*pmra0/3.6e+6
        dec1=dec0+(2016-ep)*pmdec0/3.6e+6

        r=np.sqrt((pmra0/1000)**2+(pmdec0/1000)**2)*1.5*(2016-ep)
        print(pmra0,pmdec0,ep)
        if np.isnan(r): return t
        r=str(r)+'s'
        print(r)
        v = Vizier(columns=["*", "+_r"], catalog="I/350/gaiaedr3")
        no_res=False
        try:
            res=v.query_region(SkyCoord(ra=ra1, dec=dec1,unit=(u.deg, u.deg),frame='icrs'),width=r,catalog=["I/350/gaiaedr3"])[0]
            print('aaaaaa',res)
        except IndexError: 
            no_res=True
            l=0
        else:
            l=len(res)
            print(l)
            print(res)
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
                    print(res.colnames)
                    w=np.argmin(np.abs(res['Plx']-t['dr2_parallax'].value[0]))
                    res=res[w]
                    if np.abs(res['Gmag']-t['dr2_phot_g_mean_mag'].value[0])<0.2: no_res=False
        finally:
            if (no_res==False) & (l>0):
                id=str(res['Source'])
                print(id)
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
                
    @staticmethod
    def fix_double_entries(t,id_type='DR2'):
        if len(t)<2: return t
        else:
            if id_type=='EDR3':
                id3=t['edr3_id'][0].split('Gaia EDR3 ')[1]
                id2=np.array([t['dr2_id'][i].split('Gaia DR2 ')[1] for i in range(len(t))])
                cols=['dr2_id', 'dr2_epoch', 'dr2_ra', 'dr2_dec', 'dr2_parallax', 'dr2_parallax_error', 'dr2_parallax_over_error', 'dr2_pmra', 'dr2_pmra_error', 'dr2_pmdec', 'dr2_pmdec_error', 'dr2_ra_dec_corr', 'dr2_ra_parallax_corr', 'dr2_ra_pmra_corr', 'dr2_ra_pmdec_corr', 'dr2_dec_parallax_corr', 'dr2_dec_pmra_corr', 'dr2_dec_pmdec_corr', 'dr2_parallax_pmra_corr', 'dr2_parallax_pmdec_corr', 'dr2_pmra_pmdec_corr', 'dr2_phot_g_mean_mag', 'dr2_phot_g_mean_flux', 'dr2_phot_g_mean_flux_error', 'dr2_phot_bp_mean_flux', 'dr2_phot_bp_mean_flux_error', 'dr2_phot_bp_mean_mag', 'dr2_phot_rp_mean_flux', 'dr2_phot_rp_mean_flux_error', 'dr2_phot_rp_mean_mag', 'dr2_bp_rp', 'dr2_phot_bp_rp_excess_factor', 'dr2_ruwe', 'dr2_astrometric_params_solved', 'radial_velocity', 'radial_velocity_error']
            else:
                id2=t['dr2_id'][0].split('Gaia DR2 ')[1]
                id3=np.array([t['edr3_id'][i].split('Gaia EDR3 ')[1] for i in range(len(t))])
                cols=['edr3_id', 'ra', 'dec', 'edr3_epoch', 'edr3_parallax', 'edr3_parallax_error', 'edr3_parallax_over_error', 'edr3_pmra', 'edr3_pmra_error', 'edr3_pmdec', 'edr3_pmdec_error', 'edr3_ra_dec_corr', 'edr3_ra_parallax_corr', 'edr3_ra_pmra_corr', 'edr3_ra_pmdec_corr', 'edr3_dec_parallax_corr', 'edr3_dec_pmra_corr', 'edr3_dec_pmdec_corr', 'edr3_parallax_pmra_corr', 'edr3_parallax_pmdec_corr', 'edr3_pmra_pmdec_corr', 'edr3_phot_g_mean_mag', 'edr3_phot_g_mean_flux', 'edr3_phot_g_mean_flux_error', 'edr3_phot_bp_mean_flux', 'edr3_phot_bp_mean_flux_error', 'edr3_phot_bp_mean_mag', 'edr3_phot_rp_mean_flux', 'edr3_phot_rp_mean_flux_error', 'edr3_phot_rp_mean_mag', 'edr3_bp_rp', 'edr3_phot_bp_rp_excess_factor', 'edr3_ruwe', 'edr3_astrometric_params_solved']

            w,=np.where(id3==id2)
            if len(w)==0:
                for i in cols: t[i].mask=True
                w=np.array([0])

            return t[w]

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
        n=int(10*np.ceil(abs(max([x1-x0,y1-y0,z1-z0],key=abs))))    
        dim=f.shape    
        ndim=len(dim)

        x=np.floor(np.linspace(x0,x1,num=n)).astype(int)
        y=np.floor(np.linspace(y0,y1,num=n)).astype(int)
        I=0
        
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
                for i in range(len(w2)): I+=f[x[w2[i]],y[w2[i]]]*w_f[w[i]]
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
                for i in range(len(w2)): I+=f[x[w2[i]],y[w2[i]],z[w2[i]]]*w_f[w[i]]
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
                for i in range(len(w2)): I+=f[layer,x[w2[i]],y[w2[i]]]*w_f[w[i]]
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
                for i in range(len(w2)): I+=f[layer,x[w2[i]],y[w2[i]],z[w2[i]]]*w_f[w[i]]

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
            data = f[obj]
        elif ext_map=='stilism': 
            x=np.arange(-3000.,3005.,5)
            y=np.arange(-3000.,3005.,5)
            z=np.arange(-400.,405.,5)    
            data = f['stilism']['cube_datas']

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
        A_law={'U':1.531,'B':1.317,'V':1,'R':0.748,'I':0.482,'L':0.058,'M':0.023,
           'J':0.243,'H':0.131,'K':0.078,'G':0.789,'Gbp':1.002,'Grp':0.589,
               'G2':0.789,'Gbp2':1.002,'Grp2':0.589,
           'W1':0.039,'W2':0.026,'W3':0.040,'W4':0.020,
           'gmag':1.155,'rmag':0.843,'imag':0.628,'zmag':0.487,'ymag':0.395,
               'K1mag':0.078,'K2mag':0.078, #copied from 2MASS, should be checked
               'Hmag': 0.131, 'Jmag': 0.243, 'H2mag': 0.131 #copied from 2MASS, should be checked
          } #absorption coefficients

        if '-' in col:
            c1,c2=col.split('-')
            A=A_law[c1]-A_law[c2]
        else:
            A=A_law[col]
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
    def model_name(model,feh=None,afe=None,v_vcrit=None,fspot=None,B=0):
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
            feh_range=np.array([-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.00])
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
            else: model2=model+'_p0.00'
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
        elif model=='geneva':
            feh_range=np.array([-1.5,0.0])
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
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
#                param['v_vcrit']=v_vcrit0
                if v_vcrit0<0.01: s='norot'
                else: s='rot'
                model2+='_'+s
            else: model2+='_norot'
        else: model2=model
#        return model2,param
        return model2
    
    @staticmethod
    def get_isochrone_filter(model,filt):
        if model=='bt_settl':
            dic={'G':'G2018','Gbp':'G2018_BP','Grp':'G2018_RP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10','U':'U','B':'B',
                 'V':'V','R':'R','I':'i','gmag':'g_p1','rmag':'r_p1','imag':'i_p1',
                 'zmag':'z_p1','ymag':'y_p1','V_sl':'V','R_sl':'R','I_sl':'I','K_sl':'K',
                 'R_sl2':'Rsloan','Z_sl':'Zsloan','M_sl':'Msloan',
                 'Ymag':'B_Y','Jmag':'B_J','Hmag':'B_H','Kmag':'B_Ks','H2mag':'D_H2','H3mag':'D_H3',
                 'H4mag':'D_H4','J2mag':'D_J2','J3mag':'D_J3','K1mag':'D_K1','K2mag':'D_K2',
                 'Y2mag':'D_Y2','Y3mag':'D_Y3'}            
        elif model=='ames_cond':
            dic={'G':'G','Gbp':'G_BP','Grp':'G_BP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10','U':'U','B':'B',
                 'V':'V','R':'R','I':'i','gmag':'g_p1','rmag':'r_p1','imag':'i_p1',
                 'zmag':'z_p1','ymag':'y_p1','V_sl':'V','R_sl':'R','I_sl':'I','K_sl':'K',
                 'R_sl2':'Rsloan','Z_sl':'Zsloan','M_sl':'Msloan',
                 'Ymag':'B_Y','Jmag':'B_J','Hmag':'B_H','Kmag':'B_Ks','H2mag':'D_H2','H3mag':'D_H3',
                 'H4mag':'D_H4','J2mag':'D_J2','J3mag':'D_J3','K1mag':'D_K1','K2mag':'D_K2',
                 'Y2mag':'D_Y2','Y3mag':'D_Y3'}
        elif model=='ames_dusty':
            dic={'G':'G','Gbp':'G_BP','Grp':'G_BP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10','U':'U','B':'B',
                 'V':'V','R':'R','I':'i','gmag':'g_p1','rmag':'r_p1','imag':'i_p1',
                 'zmag':'z_p1','ymag':'y_p1','V_sl':'V','R_sl':'R','I_sl':'I','K_sl':'K',
                 'R_sl2':'Rsloan','Z_sl':'Zsloan','M_sl':'Msloan',
                 'Ymag':'B_Y','Jmag':'B_J','Hmag':'B_H','Kmag':'B_Ks','H2mag':'D_H2','H3mag':'D_H3',
                 'H4mag':'D_H4','J2mag':'D_J2','J3mag':'D_J3','K1mag':'D_K1','K2mag':'D_K2',
                 'Y2mag':'D_Y2','Y3mag':'D_Y3'}
        elif model=='nextgen':
            dic={'G':'G2018','Gbp':'G2018_BP','Grp':'G2018_RP','J':'J','H':'H','K':'K',
                 'W1':'W1_W10','W2':'W2_W10','W3':'W3_W10','W4':'W4_W10',
                 'gmag':'g_p1','rmag':'r_p1','imag':'i_p1','zmag':'z_p1','ymag':'y_p1',
                 'Ymag':'B_Y','Jmag':'B_J','Hmag':'B_H','Kmag':'B_Ks','H2mag':'D_H2','H3mag':'D_H3',
                 'H4mag':'D_H4','J2mag':'D_J2','J3mag':'D_J3','K1mag':'D_K1','K2mag':'D_K2',
                 'Y2mag':'D_Y2','Y3mag':'D_Y3'}            
        elif model=='mist':
            dic={'G':'Gaia_G_EDR3','Gbp':'Gaia_BP_EDR3','Grp':'Gaia_RP_EDR3',
                 'J':'2MASS_J','H':'2MASS_H','K':'2MASS_Ks',
                 'W1':'WISE_W1','W2':'WISE_W2','W3':'WISE_W3','W4':'WISE_W4',
                 'U':'Bessell_U','B':'Bessell_B','V':'Bessell_V','R':'Bessell_R','I':'Bessell_I',
                 'Kp':'Kepler_Kp','KD51':'Kepler_D51','Hp':'Hipparcos_Hp',
                 'B_tycho':'Tycho_B','V_tycho':'Tycho_V','TESS':'TESS'}            
        elif model=='parsec':
            dic={'G':'Gmag','Gbp':'G_BPmag','Grp':'G_RPmag',                 
                 'J':'Jmag','H':'Hmag','K':'Ksmag','Spitzer_3.6':'IRAC_3.6mag',
                 'Spitzer_4.5':'IRAC_4.5mag','Spitzer_5.8':'IRAC_5.8mag','Spitzer_8.0':'IRAC_8.0mag',
                 'Spitzer_24':'MIPS_24mag','Spitzer_70':'MIPS_70mag','Spitzer_160':'MIPS_160mag',
                 'W1':'W1mag','W2':'W2mag','W3':'W3mag','W4':'W4mag'} 
        elif model=='spots':
            dic={'G':'G_mag','Gbp':'BP_mag','Grp':'RP_mag',                 
                 'J':'J_mag','H':'H_mag','K':'K_mag',
                 'B':'B_mag','V':'V_mag','R':'Rc_mag','I':'Ic_mag',
                 'W1':'W1_mag', 'W2':'nan', 'W3':'nan', 'W4':'nan'} 
        elif model=='dartmouth':
            dic={'B': 'jc_B','V': 'jc_V','R': 'jc_R','I': 'jc_I',
                 'G':'gaia_G','Gbp':'gaia_BP','Grp':'gaia_RP',                 
                 'U':'U','B':'B','V':'V','R':'R','I':'I',
                 'J':'2mass_J','H':'2mass_H','K':'2mass_K'} 
        elif model=='starevol':
            dic={'U':'M_U','B':'M_B','V':'M_V','R':'M_R','I':'M_I',
                 'J':'M_J','H':'M_H','K':'M_K','G':'M_G','Gbp':'M_Gbp','Grp':'M_Grp'}
        elif model=='bhac15':
            dic={'G':'G','Gbp':'G_BP','Grp':'G_RP','J':'Mj','H':'Mh','K':'Mk',
                 'gmag':'g_p1','rmag':'r_p1','imag':'i_p1',
                 'zmag':'z_p1','ymag':'y_p1',
                 'Ymag':'B_Y','Jmag':'B_J','Hmag':'B_H','Kmag':'B_Ks','H2mag':'D_H2','H3mag':'D_H3',
                 'H4mag':'D_H4','J2mag':'D_J2','J3mag':'D_J3','K1mag':'D_K1','K2mag':'D_K2',
                 'Y2mag':'D_Y2','Y3mag':'D_Y3'}            
        elif model=='atmo2020_ceq':
            dic={'MKO_Y':'MKO_Y','MKO_J':'MKO_J','MKO_H':'MKO_H','MKO_K':'MKO_K','MKO_L':'MKO_Lp','MKO_M':'MKO_Mp',
                 'W1':'W1','W2':'W2','W3':'W3','W4':'W4',
                 'IRAC_CH1':'IRAC_CH1','IRAC_CH2':'IRAC_CH2'}
        elif model=='atmo2020_neq_s':
            dic={'MKO_Y':'MKO_Y','MKO_J':'MKO_J','MKO_H':'MKO_H','MKO_K':'MKO_K','MKO_L':'MKO_Lp','MKO_M':'MKO_Mp',
                 'W1':'W1','W2':'W2','W3':'W3','W4':'W4',
                 'IRAC_CH1':'IRAC_CH1','IRAC_CH2':'IRAC_CH2'}
        elif model=='atmo2020_neq_w':
            dic={'MKO_Y':'MKO_Y','MKO_J':'MKO_J','MKO_H':'MKO_H','MKO_K':'MKO_K','MKO_L':'MKO_Lp','MKO_M':'MKO_Mp',
                 'W1':'W1','W2':'W2','W3':'W3','W4':'W4',
                 'IRAC_CH1':'IRAC_CH1','IRAC_CH2':'IRAC_CH2'}
        elif model=='geneva':
            dic={'lum':'logL', 't_eff':'logTe','V':'Vmag','U-B':'U-B','B-V':'B-V',
                 'B':'Bmag','U':'Umag', 'R':'nan', 'i':'nan'}

        return dic[filt]
    
    @staticmethod
    def fix_filters(filters,model,mode='collapse'):
        mod2=['bt_settl','starevol','spots','dartmouth','ames_cond','ames_dusty','bt_nextgen','nextgen','bhac15']
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

        mass_range=kwargs['mass_range'] if 'mass_range' in kwargs else [0.01,1.4]
        age_range=kwargs['age_range'] if 'age_range' in kwargs else [1,1000]
        B=kwargs['B'] if 'B' in kwargs else 0
        feh=kwargs['feh'] if 'feh' in kwargs else None
        afe=kwargs['afe'] if 'afe' in kwargs else None
        v_vcrit=kwargs['v_vcrit'] if 'v_vcrit' in kwargs else None
        fspot=kwargs['fspot'] if 'fspot' in kwargs else None

        def filters_to_surveys(filters):    
            surveys=[]
            gaia=np.array(['G','Gbp','Grp','G2','Gbp2','Grp2'])
            tmass=np.array(['J','H','K'])
            wise=np.array(['W1','W2','W3','W4'])
            johnson=np.array(['U','B','V','R','i'])
            panstarrs=np.array(['gmag','rmag','imag','zmag','ymag'])
            sloan=np.array(['V_sl','R_sl','I_sl','K_sl','R_sl2','Z_sl','M_sl'])
            sphere=np.array(['Ymag','Jmag','Hmag','Kmag','H2mag','H3mag','H4mag','J2mag','J3mag','K1mag','K2mag','Y2mag','Y3mag'])
            if len(np.intersect1d(gaia,filters))>0: surveys.append('gaia')
            if len(np.intersect1d(tmass,filters))>0: surveys.append('2mass')
            if len(np.intersect1d(wise,filters))>0: surveys.append('wise')
            if len(np.intersect1d(johnson,filters))>0: surveys.append('johnson')
            if len(np.intersect1d(sloan,filters))>0: surveys.append('sloan')
            if len(np.intersect1d(sphere,filters))>0: surveys.append('sphere')

            return surveys


        model_code=MADYS.model_name(model,feh=feh,afe=afe,v_vcrit=v_vcrit,fspot=fspot,B=B)
    #    model_code,param=model_name(model,feh=feh,afe=afe,v_vcrit=v_vcrit,fspot=fspot,B=B)
    #    param['mass_range']=mass_range
    #    param['age_range']=age_range

        
        fnew=MADYS.fix_filters(filters,model,mode='collapse')
        nf=len(fnew)
        c=0

        surveys=filters_to_surveys(fnew)
        
        n1=n_steps[0]
        mnew=M_sun.value/M_jup.value*np.exp(np.log(0.999*mass_range[0])+(np.log(1.001*mass_range[1])-np.log(0.999*mass_range[0]))/(n1-1)*np.arange(n1))

        try: len(age_range) #NEWLINE
        except TypeError: #NEWLINE
            anew=age_range #NEWLINE
            n2=1 #NEWLINE
            case=1
        else:
            if isinstance(age_range,list): #NEWLINE
                n2=n_steps[1]
                anew=np.exp(np.log(1.0001*age_range[0])+(np.log(0.9999*age_range[1])-np.log(1.0001*age_range[0]))/(n2-1)*np.arange(n2)) #NEWLINE
                case=2
            elif isinstance(age_range,np.ndarray):  #NEWLINE
                if len(age_range.shape)==1:
                    anew=np.array(age_range)
                    n2=len(anew)
                    case=3
                elif len(age_range[0]==3):
                    anew=np.unique(age_range.ravel())
                    n2=len(anew)
                    case=4
            else: raise TypeError('Only scalar, list or numpy arrays are valid inputs for the keyword "age_range".')

        iso_f=np.full(([n1,n2,nf]), np.nan) #final matrix    
        found=np.zeros(nf,dtype=bool)
#        c=0
        if case>1:
            for i in range(len(surveys)):
#                if c==nf: break
                if np.sum(found)==nf: break
                try:
                    masses, ages, v0, data0 = model_data(surveys[i],model_code)
                except ValueError: #if the survey is not found for the isochrone set, its filters are set to NaN
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
        fnew=MADYS.fix_filters(fnew,model,mode='replace')

        return mnew,anew,fnew,iso_f

    @staticmethod
    def ang_deg(ang,form='hms'):    
        ang2=ang.split(' ')
        ang2=ang2[0]+form[0]+ang2[1]+form[1]+ang2[2]+form[2]
        return ang2

    
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
                         'bt_settl':'settl','bt-settl':'settl','bt settl':'settl'                         
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

                    print('--------------------------------------------------------------------------------------')
            if found==False: 
                mess='The inserted model does not exist. Check the spelling and try again. Available models: '+','.join(f_mods)
                raise ValueError(mess)
