#!/usr/bin/env python
# coding: utf-8

from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad
from pathlib import Path
import sys
from astroquery.xmatch import XMatch
import csv
import numpy as np
import os
from astropy.table import Table
from astropy.io import ascii


def import_phot(coord_file,output_file=None,mode=None):
    #working path selection
    folder=os.path.dirname(coord_file)

    #XMatch needs a .csv as input. If missing, this cycle converts the .txt to a .csv
    if coord_file.endswith('.txt'):
        with open(coord_file) as f:
            data = np.genfromtxt(f,dtype="str")
        new_file=coord_file.replace('.txt','.csv')
        with open(new_file, newline='', mode='w') as f:
            r_csv = csv.writer(f, delimiter=',')
            for i in range(len(data)):
                r_csv.writerow(data[i])
    else:
        new_file=coord_file
    #VizieR queries. Results saved as .txt

    if mode=='all':
        #2MASS
        data_2mass = XMatch.query(cat1=open(Path(new_file)),cat2='vizier:II/246/out',max_distance=2 * u.arcsec, colRA1='ra_v',colDec1='dec_v')
        name='2MASS'
        #no_input=0
        output_file=os.path.join(folder,str('py_cms_'+name+'.csv'))
        #if output_file==None:
        #    no_input=1
        #    output_file=os.path.join(folder,str('py_cms_'+name+'.csv'))
        data_2mass.write(output_file, overwrite=True)
        #file='py_cms_'+name+'.txt' #se volessi salvare come .txt
        #data_2mass.write(folder / file, overwrite=True,format='ascii.fixed_width',delimiter=None)

        #GAIA DR3
        data_gaiadr3 = XMatch.query(cat1=open(Path(new_file)),cat2='vizier:I/350/gaiaedr3',max_distance=2 * u.arcsec, colRA1='ra_v',colDec1='dec_v')
        name='GAIADR3'
        output_file=os.path.join(folder,str('py_cms_'+name+'.csv'))
        #if no_input==1: output_file=os.path.join(folder,str('py_cms_'+name+'.csv'))
        data1_gaiadr3=data_gaiadr3['angDist', 'source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',  'ruwe', 'phot_g_mean_flux_error', 'phot_g_mean_mag', 'phot_bp_mean_flux_error', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'dr2_radial_velocity', 'dr2_radial_velocity_error', 'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error', 'phot_g_mean_mag_corrected', 'phot_g_mean_mag_error_corrected', 'phot_g_mean_flux_corrected', 'phot_bp_rp_excess_factor_corrected', 'ra_epoch2000_error', 'dec_epoch2000_error', 'ra_dec_epoch2000_corr']
        data1_gaiadr3.write(output_file, overwrite=True)
    elif mode=='2MASS':
        data_2mass = XMatch.query(cat1=open(Path(new_file)),cat2='vizier:II/246/out',max_distance=2 * u.arcsec, colRA1='ra_v',colDec1='dec_v')
        name='2MASS'
        if output_file==None: output_file=os.path.join(folder,str('py_cms_'+name+'.csv'))
        data_2mass.write(output_file, overwrite=True)
    elif mode=='Gaia_EDR3':
        data_gaiadr3 = XMatch.query(cat1=open(Path(new_file)),cat2='vizier:I/350/gaiaedr3',max_distance=2 * u.arcsec, colRA1='ra_v',colDec1='dec_v')
        name='GAIADR3'
        if output_file==None: output_file=os.path.join(folder,str('py_cms_'+name+'.csv'))
        data1_gaiadr3=data_gaiadr3['angDist', 'source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',  'ruwe', 'phot_g_mean_flux_error', 'phot_g_mean_mag', 'phot_bp_mean_flux_error', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'dr2_radial_velocity', 'dr2_radial_velocity_error', 'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error', 'phot_g_mean_mag_corrected', 'phot_g_mean_mag_error_corrected', 'phot_g_mean_flux_corrected', 'phot_bp_rp_excess_factor_corrected', 'ra_epoch2000_error', 'dec_epoch2000_error', 'ra_dec_epoch2000_corr']
        data1_gaiadr3.write(output_file, overwrite=True)

    return None


def first_query_gaia(filename):
    with open(filename) as f:
        target_list = np.genfromtxt(f,dtype="str")

    n=len(target_list)
    ra=np.zeros(n)
    dec=np.zeros(n)
    for i in range(n):
        x=Simbad.query_object(target_list[i])
        ra0=Angle(x['RA'],'hour')
        dec0=Angle(x['DEC'],'degree')
        ra[i]=ra0.degree
        dec[i]=dec0.degree
    coo=set(zip(ra,dec))

    folder=os.path.dirname(filename)
    data = Table([ra, dec],names=['ra_v','dec_v'])
    ascii.write(data, os.path.join(folder,'coo.txt'), overwrite=True, format='fixed_width', delimiter=None)

    return None
