import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.coordinates import Angle
from astropy.table import Table, Column
from astroquery.skyview import SkyView
from astroquery.eso import Eso
eso=Eso()
import pylab as pl
import csv
from astropy.io import ascii
from tabulate import tabulate
import pdb
import time
import pickle

from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

Vizier.TIMEOUT = 100000000 # rise the timeout for Vizier

tab=ascii.read('csm_all.txt') # reads the table - no need to specify the format
# you can print(tab) to check the names of the columns
ra=tab.field('col1') # read a column into an array
dec=tab.field('col2')
c=SkyCoord(ra, dec, unit=(u.hourangle, u.deg)) # makes the array of coordinates with the specific units
radius=u.Quantity(50, u.arcsec) #sets the radius of the search to 50 arcsecs
ns=len(c) # checks the number of entries in the table
# define the table field and their format
tab_phot = Table(names=('ID', 'ra','dec','plx', 'eplx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE','Gmag','BPmag','RPmag','Kmag','Jmag','Hmag'),dtype=('S20','O','O','O','O','O','O','O','O','O','O','O','O','O','O'))

for i in range (ns): #loops over the first 100 elements
                     # note that range(n) generates n integers between 0 and n-1
    #remember to ALWAYS indent the loops.
    #set variables to 0 to have an entry in the table even if no source is found in Gaia or 2MASS
    plx=eplx=pmra=empra=pmdec=epmdec=Gmag=BPmag=RPmag=Jmag=Kmag=Hmag=0.0
    # search EDR3
    g2=Gaia.cone_search_async(c[i], radius, table_name='gaiaedr3.gaia_source').get_results()
    if g2: # only if there is an entry in gaia takes the values of the first entry
        plx = g2['parallax'][0]
        eplx = g2['parallax_error'][0]
        pmra = g2['pmra'][0]
        epmra = g2['pmra_error'][0]
        pmdec = g2['pmdec'][0]
        epmdec = g2['pmdec_error'][0]
        Gmag = g2['phot_g_mean_mag'][0]
        BPmag = g2['phot_bp_mean_mag'][0]
        RPmag = g2['phot_rp_mean_mag'][0]
    # search 2MASS
    d=Vizier.query_region(c[i], radius, catalog="II/246")
    if d: # only if there is an entry in "2MASS" takes the values of the first entry
          # note that since you've used Vizier the format of the output is different
        Jmag=d[0][0]['Jmag']
        Hmag=d[0][0]['Hmag']
        Kmag=d[0][0]['Kmag']
    #add a row to the table
    ID='Star #'+np.str(i)
    tab_phot.add_row((ID, ra[i], dec[i], plx, eplx, pmra, epmra, pmdec, epmdec, Gmag, BPmag, RPmag, Jmag, Hmag, Kmag))
    print(ID, ra[i], dec[i], plx, eplx, pmra, epmra, pmdec, epmdec, Gmag, BPmag, RPmag, Jmag, Hmag, Kmag)
# write the table in a comma separated file (can also be a tab separated file if format = 'tab')

f=open('csm_tab.txt', "w+")
f.write(tabulate(tab_phot,headers=['ID', 'ra','dec','plx', 'eplx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE','Gmag','BPmag','RPmag','Kmag','Jmag','Hmag'], tablefmt='tsv'))
f.close()
