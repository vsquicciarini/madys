# coding: utf-8

import numpy as np
from pathlib import Path
import sys
import os
from evolution import *
from scipy.interpolate import interp1d
from astropy.constants import M_jup,M_sun
import time
import pickle
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
import csv
from astropy.table import Table, vstack
from astropy.io import ascii
from tabulate import tabulate
import math
import shutil
import h5py


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    taken from:
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def n_elements(x):
    size = 1
    for dim in np.shape(x): size *= dim
    return size

def closest(array,value):
    '''Given an "array" and a (list of) "value"(s), finds the j(s) such that |array[j]-value|=min((array-value)).
    "array" must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    nv=n_elements(value)
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
                if (value[i] == array[0]):# edge cases at bottom
                    jn[i]=0
                elif (value[i] == array[n-1]):# and top
                    jn[i]=n-1
                else:
                    jn[i]=jl+np.argmin([value[i]-array[jl],array[jl+1]-value[i]])
        return jn

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

def is_phot_good(phot,phot_err,max_phot_err=0.1):
    if type(phot)==float: dim=0
    else:
        l=phot.shape
        dim=len(l)
    if dim<=1: gs=(np.isnan(phot)==False) & (phot_err < max_phot_err) & (abs(phot) < 70)
    else:
        gs=np.zeros([l[0],l[1]])
        for i in range(l[1]): gs[:,i]=(np.isnan(phot[:,i])==False) & (phot_err[:,i] < max_phot_err)
    return gs

def where_v(elements,array,approx=False):
    dim=n_dim(elements)
    if approx==True:
        if dim==0: 
            w=closest(array,elements)
            return w
        ind=np.zeros(len(elements),dtype=np.int16)
        for i in range(len(elements)):
            ind[i]=closest(array,elements[i])
        return ind
    else:
        if dim==0: 
            w,=np.where(array==elements)
            return w
        ind=np.zeros(len(elements),dtype=np.int16)
        for i in range(len(elements)):
            w,=np.where(array==elements[i])
            ind[i]=w
        return ind

def n_dim(x,shape=False):
    if isinstance(x,str): 
        dim=0
        return dim
    try:
        b=len(x)
        if shape: dim=x.shape
        else: dim=len(x.shape)
    except AttributeError:
        sh = []
        a = len(x)
        sh.append(a)
        b = x[0]
        while a > 0:
            try:
                if isinstance(b,str): break
                a = len(b)
                sh.append(a)
                b = b[0]

            except:
                break
        if shape: dim=sh
        else: dim=len(sh)
    except TypeError: dim=0
    return dim

def app_to_abs_mag(app_mag,parallax,app_mag_error=None,parallax_error=None):
    if isinstance(app_mag,list): app_mag=np.array(app_mag)
    if isinstance(parallax,list): parallax=np.array(parallax)
    dm=5*np.log10(100./parallax) #modulo di distanza
    dim=n_dim(app_mag)
    if dim <= 1:
        abs_mag=app_mag-dm
        if (type(app_mag_error)!=type(None)) & (type(parallax_error)!=type(None)): 
            if isinstance(app_mag_error,list): app_mag_error=np.array(app_mag_error)
            if isinstance(parallax_error,list): parallax_error=np.array(parallax_error)
            total_error=np.sqrt(app_mag_error**2+(5/np.log(10)/parallax)**2*parallax_error**2)
            result=(abs_mag,total_error)
        else: result=abs_mag
    else: #  se è 2D, bisogna capire se ci sono più filtri e se c'è anche l'errore fotometrico
        l=n_dim(app_mag,shape=True)
        abs_mag=np.empty([l[0],l[1]])
        for i in range(l[1]): abs_mag[:,i]=app_mag[:,i]-dm
        if type(parallax_error)!=type(None):
            if isinstance(app_mag_error,list): app_mag_error=np.array(app_mag_error)
            if isinstance(parallax_error,list): parallax_error=np.array(parallax_error)
            total_error=np.empty([l[0],l[1]])
            for i in range(l[1]): 
                total_error[:,i]=np.sqrt(app_mag_error[:,i]**2+(5/np.log(10)/parallax)**2*parallax_error**2)
            result=(abs_mag,total_error)
        else: result=abs_mag
            
    return result #se l'input è un array 1D, non c'è errore ed è un unico filtro



def load_isochrones(model,surveys=['gaia','2mass','wise'],mass_range=[0.01,1.4],age_range=[1,1000],n_steps=[1000,500],feh=None,afe=None,v_vcrit=None,fspot=None):

    #mass_range: massa minima e massima desiderata, in M_sun
    #age_range: età minima e massima desiderata, in Myr
    #n_steps: n. step desiderati in massa ed età
    
    #devo crearmi una funzione con un dizionario, che mi restituisca per il modello dato, per la survey di interesse e per
    #il filtro specificato, il nome del filtro nel modello dato. Es. f('bt_settl','wise','W1')='W1_W10'

    folder = os.path.dirname(os.path.realpath(__file__))
    add_search_path(folder)
    iso_y=False
    for x in os.walk(folder):
        add_search_path(x[0])
        if x[0].endswith('isochrones') and iso_y==False:
            PIK_path=x[0]
            iso_y=True
    if iso_y==False: PIK_path=folder


    def filter_code(model,f_model,filt):
        
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
                 'W1':'W1_mag'} 
        elif model=='dartmouth':
            dic={'G':'Gaia_G','Gbp':'Gaia_BP','Grp':'Gaia_RP',                 
                 'U':'U','B':'B','V':'V','R':'R','I':'I',
                 'J':'J','H':'H','K':'Ks',
                 'W1':'W1','W2':'W2','W3':'W3','W4':'W4',
                 'Kp':'Kp','KD51':'D51'} 
        elif model=='amard':
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
        elif model=='mamajek': #Mv    B-V  Bt-Vt    G-V  Bp-Rp   G-Rp    M_G     b-y    U-B   V-Rc   V-Ic   V-Ks    J-H   H-Ks   M_J    M_Ks  Ks-W1   W1-W2  W1-W3  W1-W4
            dic={}
        elif model=='ekstrom': #farlo
            dic={}
        w,=np.where(f_model==dic[filt])
        
        return w

    def model_name(model,feh=None,afe=None,v_vcrit=None,fspot=None):
        if model=='bt_settl': model2=model
        elif model=='mist':
            feh_range=np.array([-4.,-3.5,-3.,-2.5,-2,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5])
            afe_range=np.array([0.0])
            vcrit_range=np.array([0.0,0.4])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(afe)!=type(None):
                i=np.argmin(abs(afe_range-afe))
                afe0=afe_range[i]
                if afe0<0: s='m'
                else: s='p'
                afe1="{:.1f}".format(abs(afe0))            
                model2+='_'+s+afe1
            else: model2+='_p0.0'
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
                if v_vcrit0<0: s='m'
                else: s='p'
                v_vcrit1="{:.1f}".format(abs(v_vcrit0))            
                model2+='_'+s+v_vcrit1
            else: model2+='_p0.0'
        elif model=='parsec':
            feh_range=np.array([0.0])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
        elif model=='amard':
            feh_range=np.array([-0.813,-0.336,-0.211,-0.114,0.0,0.165,0.301])
            vcrit_range=np.array([0.0,0.2,0.4,0.6])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
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
                fspot1="{:.2f}".format(abs(fspot0))            
                model2=model+'_p'+fspot1
            else: model2=model+'_p0.00'
        else: model2=model
        return model2

    filter_vec={'gaia':['G','Gbp','Grp'],'2mass':['J','H','K'],
        'wise':['W1','W2','W3','W4'],'johnson':['U','B','V','R','i'],
         'panstarrs':['gmag','rmag','imag','zmag','ymag'],
         'sloan':['V_sl','R_sl','I_sl','K_sl','R_sl2','Z_sl','M_sl'],
         'sphere':['Ymag','Jmag','Hmag','Kmag','H2mag','H3mag','H4mag','J2mag','J3mag','K1mag','K2mag','Y2mag','Y3mag']}
    
    surveys=list(map(str.lower,surveys))    
    model=(str.lower(model)).replace('-','_')
    model_code=model_name(model,feh=feh,afe=afe,v_vcrit=v_vcrit,fspot=fspot)
    
    file=model_code
    for i in range(len(surveys)): file=file+'_'+sorted(surveys)[i]
    PIK=Path(PIK_path) / (file+'.pkl')

    try: #se c'è
        open(PIK,'r')
    except IOError:
        fnew=[]

        survey_list=['gaia','2mass','wise','johnson','panstarrs','sloan','sphere']
        survey_el=[3,3,4,5,5,7,13]
        survey_filt=[['G','Gbp','Grp'],['J','H','K'],['W1','W2','W3','W4'],['U','B','V','R','i'],['gmag','rmag','imag','zmag','ymag'],['V_sl','R_sl','I_sl','K_sl','R_sl2','Z_sl','M_sl'],['Ymag','Jmag','Hmag','Kmag','H2mag','H3mag','H4mag','J2mag','J3mag','K1mag','K2mag','Y2mag','Y3mag']]
        survey_col=[['G2018','G2018_BP','G2018_RP'],['J','H','K'],['W1_W10','W2_W10','W3_W10','W4_W10'],['U','B','V','R','i'],['g_p1','r_p1','i_p1','z_p1','y_p1'],['V','R','I','K','Rsloan','Zsloan','Msloan'],['B_Y','B_J','B_H','B_Ks','D_H2','D_H3','D_H4','D_J2','D_J3','D_K1','D_K2','D_Y2','D_Y3']]


        fnew=[]
        for i in range(len(surveys)):
            fnew.extend(filter_vec[surveys[i]])
        nf=len(fnew)
        c=0
        
        for i in range(len(surveys)):
            masses, ages, v0, data0 = model_data(surveys[i],model_code)
#            print('input:')
#            print(masses.shape)
#            print(ages.shape)
#            print(v0.shape)
#            print(data0.shape)
#            print(surveys[i],model)

#            print(masses)
#            print(ages)
#            print(v0)
#            print(data0)
#            sys.exit()
            
            if 'iso_f' not in locals():
                nm=len(masses)
                na=len(ages)
                n1=n_steps[0]
                n2=n_steps[1]
                mnew=M_sun/M_jup*mass_range[0]+M_sun/M_jup*(mass_range[1]-mass_range[0])/(n1-1)*np.arange(n1)
                anew=np.exp(np.log(age_range[0])+(np.log(age_range[1])-np.log(age_range[0]))/(n2-1)*np.arange(n2))
                iso_f=np.full(([n1,n2,nf]), np.nan) #matrice con spline in età, devo completarla
            iso=np.full(([n1,len(ages),len(filter_vec[surveys[i]])]),np.nan) #matrice con spline in massa, devo completarla        
        
            for j in range(len(filter_vec[surveys[i]])):
                w=filter_code(model,v0,filter_vec[surveys[i]][j])
                #print(v0,i,surveys[i],j,filter_vec[surveys[i]],w,type(v0))
                for k in range(len(ages)): #spline in massa. Per ogni età
                    nans, x= nan_helper(data0[:,k,w])
                    nans=nans.reshape(len(nans))
                    m0=masses[~nans]
                    if len(x(~nans))>1:
                        f=interp1d(masses[~nans],data0[~nans,k,w],kind='linear',fill_value=np.nan,bounds_error=False)
                        iso[:,k,j]=f(mnew)
                for k in range(n1): #spline in età. Per ogni massa
                    nans, x= nan_helper(iso[k,:,j])
                    nans=nans.reshape(n_elements(nans))
                    a0=ages[~nans]
                    if len(x(~nans))>1:
                        f=interp1d(ages[~nans],iso[k,~nans,j],kind='linear',fill_value=np.nan,bounds_error=False)
                        iso_f[k,:,j+c]=f(anew)
            c+=len(filter_vec[surveys[i]])
                        
        mnew=M_jup/M_sun*mnew
        fnew=np.array(fnew)
        with open(PIK,'wb') as f:
            pickle.dump(iso_f,f)
            pickle.dump(mnew,f)
            pickle.dump(anew,f)
            pickle.dump(fnew,f)
    finally:
        with open(PIK,'rb') as f:
            iso_f=pickle.load(f)
            mnew=pickle.load(f)
            anew=pickle.load(f)
            fnew=pickle.load(f)

    return mnew,anew,fnew,iso_f


def retrieve_parameters():
    with open(folder / 'options.txt') as f:
        opt = np.genfromtxt(f,dtype="str")
        parameters=opt[:,1]
        max_r=float(parameters[1])
        lit_file=parameters[2]
        ph_cut=float(parameters[5])
        max_flux_cont=float(parameters[6])
        if parameters[10]=='single': n_est=1
        elif parameters[10]=='multiple': n_est=4 #SNU, SRU, BNU, BRU
        surveys=parameters[12].split(',')
        ws0=['2MASS' in surveys,'WISE' in surveys,'Panstarrs' in surveys] #quali survey oltre a Gaia devo usare?
        print(max_r)
        print(lit_file)
        print(ph_cut)
        print(max_flux_cont)
        print(surveys)
        print(ws0)

def isochronal_age(phot_app,phot_err_app,par,par_err,iso,surveys,border_age=False,ebv=None):

    mnew=iso[0]
    anew=iso[1]
    fnew=iso[2]
    newMC=iso[3]
    

    #CONSTANTS
    ph_cut=0.2
    bin_frac=0.0
    
    #trasformo fotometria in assoluta
    phot,phot_err=app_to_abs_mag(phot_app,par,app_mag_error=phot_err_app,parallax_error=par_err)

    #raggi per peso della media

    #contaminazione in flusso
    cont=np.zeros(2) #contaminazione nei filtri 2MASS per le stelle, in questo caso nulla

    l0=phot.shape
    xlen=l0[0] #no. of stars: 85
    ylen=l0[1] #no. of filters: 6

    filt=where_v(['J','H','K','G','Gbp','Grp'],fnew)
    wc=np.array([[filt[2],filt[0],filt[1],filt[5]],[filt[3],filt[3],filt[3],filt[4]]]) #(G-K), (G-J), (G-H), (Gbp-Grp)
#    print(filt[wc])

    red=np.zeros([xlen,ylen]) #reddening da applicare
    if type(ebv)!=type(None):
        for i in range(ylen): red[:,i]=extinction(ebv,fnew[i])

    l=newMC.shape #(780,460,10) cioè masse, età e filtri
    sigma=100+np.zeros([l[0],l[1],ylen]) #matrice delle distanze fotometriche (780,460,3)
    loga=np.zeros([4,xlen]) #stime di log(età) in ognuno dei quattro canali (4,85)

    #calcolare reddening
    m_cmsf=np.full(([4,xlen]),np.nan) #stime di massa (4,85,1)
    a_cmsf=np.full(([4,xlen]),np.nan) #stime di età (4,85,1)

    n_val=np.zeros(4) #numero di stelle usate per la stima per canale (0) e tipologia (SRB ecc, 1) (4,1)
    tofit=np.zeros([4,xlen]) #contiene, per ogni CMS, 1 se fittato, 0 se non fittato (4,2,1)

    bin_corr=2.5*np.log10(2)*bin_frac #ossia, se le binarie sono identiche, la luminosità osservata è il doppio di quella della singola componente

    phot0=phot-red+bin_corr #(6,2) come phot

    fate=np.ones([4,xlen]) #(4,85)  ci dice se la stella i nella stima j e nel canale k è stata fittata, ha errori alti, contaminazione ecc. Di default è contaminata (1)


    sigma=np.full(([l[0],l[1],ylen]),np.nan) #(780,480,6) matrice delle distanze fotometriche
             
    for i in range(xlen): #devo escludere poi i punti con errore fotometrico non valido     
        w,=np.where(is_phot_good(phot[i,:],phot_err[i,:],max_phot_err=ph_cut))
     #   print('valid',i,w)
        if len(w)==0: continue
        go=0
        for j in range(4):
            go+=isnumber(phot0[i,wc[0,j]]+phot0[i,wc[1,j]],finite=True)
        if go==0: continue
        e_j=-10.**(-0.4*phot_err[i,w])+10.**(+0.4*phot_err[i,w])
        for h in range(len(w)):
    #        print(i,xlen,h,len(w),w[h],go,newMC[0,0,w[h]])
            sigma[:,:,w[h]]=(10.**(-0.4*(newMC[:,:,w[h]]-phot0[i,w[h]]))-1.)/e_j[h]
        cr=np.zeros([l[0],l[1],4]) #(780,480,4) #per il momento comprende le distanze in G-K, G-J, G-H, Gbp-Grp
        for j in range(4):
            if isnumber(phot0[i,wc[0,j]],finite=True)==0: continue
            cr[:,:,j]=(sigma[:,:,wc[0,j]])**2+(sigma[:,:,wc[1,j]])**2 #equivale alla matrice delle distanze in (G,K), (G,J), (G,H), (Gbp,Grp)
            colth=np.full(l[1],np.nan) #480 voglio verificare se la stella si trova "in mezzo" al set di isocrone oppure all'esterno; voglio fare un taglio a mag costante
            asa=np.zeros(l[1])
            for q in range(l[1]): #480
    #            print(i,j,q,anew[q],phot0[i,wc[0,j]],np.nanmin(newMC[:,q,wc[0,j]]))
                asa[q],im0=min_v(newMC[:,q,wc[0,j]]-phot0[i,wc[0,j]],absolute=True) #trova il punto teorico più vicino nel primo filtro per ogni isocrona
                if abs(asa[q])<0.1: colth[q]=newMC[im0,q,wc[1,j]] #trova la magnitudine corrispondente nel secondo filtro della coppia
            asb=min(asa,key=abs) #se la minima distanza nel primo filtro è maggiore della soglia, siamo al di fuori del range in massa delle isocrone
    #        print(i,j,phot0[i,wc[0,j]],phot_err[i,wc[0,j]],asb)
    #        print(cr[:,:,j])
            est,ind=min_v(cr[:,:,j])
            if (est <= 2.25 or (phot0[i,wc[1,j]] >= min(colth) and phot0[i,wc[1,j]] <= max(colth))) and np.isnan(est)==False and (np.isnan(min(colth))==False and np.isnan(max(colth))==False):  #condizioni per buon fit: la stella entro griglia isocrone o a non più di 3 sigma, a condizione che esista almeno un'isocrona al taglio in "colth"
                m_cmsf[j,i]=mnew[ind[0]] #massa del CMS i-esimo
                a_cmsf[j,i]=anew[ind[1]] #età del CMS i-esimo
                n_val[j]+=1
                tofit[j,i]=1

            if (is_phot_good(phot0[i,wc[0,j]],phot_err[i,wc[0,j]],max_phot_err=ph_cut)==0) or (is_phot_good(phot0[i,wc[1,j]],phot_err[i,wc[1,j]],max_phot_err=ph_cut)==0): pass #rimane 0
            elif est > 2.25 and phot0[i,wc[1,j]] < min(colth):  fate[j,i]=2
            elif est > 2.25 and phot0[i,wc[1,j]] > max(colth):  fate[j,i]=3
            elif est > 2.25 and abs(asb) >= 0.1: fate[j,i]=4
            else: fate[j,i]=5
            if (border_age==True and est>=2.25 and phot0[i,wc[1,j]]>max(colth)):
                a_cmsf[j,i]=anew[0]
                tofit[j,i]=1
                
        if anew[-1]<150: plot_ages=[1,3,5,10,20,30,100] #ossia l'ultimo elemento
        elif anew[-1]<250: plot_ages=[1,3,5,10,20,30,100,200]
        elif anew[-1]<550: plot_ages=[1,3,5,10,20,30,100,200,500]
        elif anew[-1]<1050: plot_ages=[1,3,5,10,20,30,100,200,500,1000]
        else: plot_ages=[1,3,5,10,20,30,100,200,500,1000]

#    if os.path.isfile(path / 'TestFile.txt')==0: print("Ora dovrebbe plottare una figura")
#    if file_search(path+'G-K_G_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[3,*]-phot0[2,*],phot0[3,*],newMC,'G-K','G',plot_ages,iso_ages=anew,xerr=phot_err[2,*]+phot_err[3,*],yerr=phot_err[3,*],tofile=path+'G-K_G_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[0,*,t],/show_errors,charsize=0.3
#    if file_search(path+'G-J_J_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[3,*]-phot0[0,*],phot0[0,*],newMC,'G-J','J',plot_ages,iso_ages=anew,xerr=phot_err[0,*]+phot_err[3,*],yerr=phot_err[0,*],tofile=path+'G-J_J_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[1,*,t],/show_errors,charsize=0.3
#    if file_search(path+'G-H_H_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[3,*]-phot0[1,*],phot0[1,*],newMC,'G-H','H',plot_ages,iso_ages=anew,xerr=phot_err[1,*]+phot_err[3,*],yerr=phot_err[1,*],tofile=path+'G-H_H_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[2,*,t],/show_errors,charsize=0.3
#    if file_search(path+'Gbp-Grp_G_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[4,*]-phot0[5,*],phot0[3,*],newMC,'Gbp-Grp','G',plot_ages,iso_ages=anew,xerr=phot_err[4,*]+phot_err[5,*],yerr=phot_err[3,*],tofile=path+'Gbp-Grp_G_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[3,*,t],/show_errors,charsize=0.3

    a_final=np.empty(xlen)
    m_final=np.empty(xlen)
    for i in range(xlen): 
        a_final[i]=np.nanmean(a_cmsf[:,i])
        m_final[i]=np.nanmean(m_cmsf[:,i])
 #       print(i,phot_app[i,:],par[i],phot0[i,:],a_final[i],a_cmsf[:,i])

    return a_final,m_final


def extinction(ebv,col):
    """
    computes extinction/color excess in a filter "col", given a certain E(B-V) "ebv",
    based on absorption coefficients from the literature

    input:
        ebv: E(B-V) of the source(s) (float or numpy array)
        col: name of the required filter(s) (string)
            if you want a color excess, it should be in the form 'c1-c2'
    usage:
        extinction(0.03,'G') gives the extinction in Gaia G band for E(B-V)=0.03 mag
        extinction(0.1,'G-K') gives the color excess in Gaia G - 2MASS K bands for E(B-V)=0.1 mag

    notes:
    assumes a total-to-selective absorption R=3.16 (Wang & Chen 2019)
    Alternative values of R=3.086 or R=3.1 are commonly used in the literature.

    sources:
    Johnson U: Rieke & Lebofsky (1985)
    Johnson B: Wang & Chen (2019)
        alternatives: A_B=1.36 (Indebetouw et al. 2005, SVO Filter Service)
        or A_B=1.324 (Rieke & Lebofsky 1985)
    Johnson V: *by definition*
    Johnson R: Rieke & Lebofsky (1985)
    Johnson I: Rieke & Lebofsky (1985)
    Johnson L: Rieke & Lebofsky (1985)
    Johnson M: Rieke & Lebofsky (1985)
    2MASS J: Wang & Chen (2019)
        alternatives: A_J=0.31 (Indebetouw et al. 2005, SVO Filter Service) 
        or A_J=0.282 (Rieke & Lebofsky 1985)
    2MASS H: Wang & Chen (2019)
        alternatives: A_H=0.19 (Indebetouw et al. 2005, SVO Filter Service) 
        or A_H=0.175 (Rieke & Lebofsky 1985)
    2MASS K: Wang & Chen (2019)
        alternatives: A_K=0.13 (Indebetouw et al. 2005, SVO Filter Service) 
        or A_K=0.112 (Rieke & Lebofsky 1985)
    GAIA G: Wang & Chen (2019)
        alternative: a variable A(ext)=[0.95,0.8929,0.8426] per ext=[1,3,5] magnitudes (Jordi et al 2010)
    GAIA Gbp: Wang & Chen (2019)
    GAIA Grp: Wang & Chen (2019)
    WISE W1: Wang & Chen (2019)
    WISE W2: Wang & Chen (2019)
    WISE W3: Wang & Chen (2019)
    WISE W4: Indebetouw et al. (2005), SVO Filter Service
    PANSTARRS g (gmag): Wang & Chen (2019)
        alternative: A_g=1.18 (Indebetouw et al. 2005, SVO Filter Service) 
    PANSTARRS r (rmag): Wang & Chen (2019)
        alternative: A_r=0.88 (Indebetouw et al. 2005, SVO Filter Service) 
    PANSTARRS i (imag): Wang & Chen (2019)
        alternative: A_i=0.67 (Indebetouw et al. 2005, SVO Filter Service) 
    PANSTARRS z (zmag): Wang & Chen (2019)
        alternative: A_z=0.53 (Indebetouw et al. 2005, SVO Filter Service) 
    PANSTARRS y (ymag): Wang & Chen (2019)
        alternative: A_y=0.46 (Indebetouw et al. 2005, SVO Filter Service) 
    """
    A_law={'U':1.531,'B':1.317,'V':1,'R':0.748,'I':0.482,'L':0.058,'M':0.023,
       'J':0.243,'H':0.131,'K':0.078,'G':0.789,'Gbp':1.002,'Grp':0.589,
       'W1':0.039,'W2':0.026,'W3':0.040,'W4':0.020,
       'gmag':1.155,'rmag':0.843,'imag':0.628,'zmag':0.487,'ymag':0.395
      } #absorption coefficients

    if '-' in col:
        c1,c2=col.split('-')
        A=A_law[c1]-A_law[c2]
    else:
        A=A_law[col]
    return 3.16*A*ebv


#definisce range per plot CMD
def axis_range(col_name,col_phot):
    cmin=min(col_phot)
    cmax=min(70,max(col_phot))
    dic1={'G':[max(15,cmax),min(1,cmin)], 'Gbp':[max(15,cmax),min(1,cmin)], 'Grp':[max(15,cmax),min(1,cmin)],
        'J':[max(10,cmax),min(0,cmin)], 'H':[max(10,cmax),min(0,cmin)], 'K':[max(10,cmax),min(0,cmin)],
        'W1':[max(10,cmax),min(0,cmin)], 'W2':[max(10,cmax),min(0,cmin)], 'W3':[max(10,cmax),min(0,cmin)],
        'W4':[max(10,cmax),min(0,cmin)], 'G-J':[min(0,cmin),max(5,cmax)],
        'G-H':[min(0,cmin),max(5,cmax)], 'G-K':[min(0,cmin),max(5,cmax)],
        'G-W1':[min(0,cmin),max(6,cmax)], 'G-W2':[min(0,cmin),max(6,cmax)],
        'G-W3':[min(0,cmin),max(10,cmax)], 'G-W4':[min(0,cmin),max(12,cmax)],
        'J-H':[min(0,cmin),max(1,cmax)], 'J-K':[min(0,cmin),max(1.5,cmax)],
        'H-K':[min(0,cmin),max(0.5,cmax)], 'Gbp-Grp':[min(0,cmin),max(5,cmax)]
        }

    try:
        xx=dic1[col_name]
    except KeyError:
        if '-' in col_name:
            xx=[cmin,cmax]
        else: xx=[cmax,cmin]
    
    return xx 

def ang_deg(ang,form='hms'):    
    ang2=ang.split(' ')
    ang2=ang2[0]+form[0]+ang2[1]+form[1]+ang2[2]+form[2]
    return ang2

def search_phot(filename,surveys,coordinates=True,verbose=False):
    """
    given a file of coordinates or star names and a list of surveys, returns
    a dictionary with astrometry, kinematics and photometry retrieved from the catalogs
    The returned tables can have different dimensions: they should be cross-matched with
    the function cross_match    
    
    input:
        filename: full path of the input file
        surveys: a list of surveys to be used. Available: 'GAIA_EDR3', '2MASS', 'ALLWISE'
        coordinates: if True, the file is read as a 2D matrix, each row specifying (ra,dec) or (l,b) of a star
            if False, it is a list of star names. Default: True
        verbose: set to True to create output files with the retrieved coordinates and data. Default: False

    usage:
        search_phot(filename,['GAIA_EDR3','2MASS','ALLWISE'],coordinates=False,verbose=True)
        will search for all the stars in the input file and return the results both as a Table object
        and as .txt files, each named filename+'_ithSURVEYNAME.txt'. It will create a file with equatorial coordinates
        named filename+'_coordinates.txt'.
        search_phot(filename,['GAIA_EDR3','2MASS','ALLWISE'],coordinates=False,verbose=False)
        will not create any file.
        
    notes:
        if coordinates=True (default mode):
            the input file must begin with a 1-row header like this:
            #coordinate system: 'equatorial'
            or this:
            #coordinate system: 'galactic'
            Coordinates are interpreted as J2000: be careful to it.
            This mode is faster, but might include some field stars.
        if coordinates=False:
            the input file must not have a header, and be a mere list of stars.
            This mode is slower, because every star will be searched for individually in Simbad and,
            if not found, tries on Vizier (sometimes Simbad does not find Gaia IDs).
            An alert is raised if some stars are missing, and the resulting row are filled with NaNs.
            The returned coordinate array has the same length as the input file,
            while the output Tables might not.
    """


    #stores path, file name, extension
    folder=os.path.dirname(filename)     #working path
    sample_name=os.path.split(filename)[1] #file name
    i=0
    while sample_name[i]!='.': i=i+1
    ext=sample_name[i:] #estension
    if ext=='.csv': delim=','
    else: delim=None
    sample_name=sample_name[0:i].replace('_coordinates','') #root name
    new_file=os.path.join(folder,sample_name+'_coordinates.csv')
    
    
    #survey properties
    index={'GAIA_EDR3':0, '2MASS':1, 'ALLWISE':2}
    surv_prop=[['vizier:I/350/gaiaedr3',['angDist', 'source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',  'ruwe', 'phot_g_mean_flux_error', 'phot_g_mean_mag', 'phot_bp_mean_flux_error', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'dr2_radial_velocity', 'dr2_radial_velocity_error', 'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error', 'phot_g_mean_mag_corrected', 'phot_g_mean_mag_error_corrected', 'phot_g_mean_flux_corrected', 'phot_bp_rp_excess_factor_corrected', 'ra_epoch2000_error', 'dec_epoch2000_error', 'ra_dec_epoch2000_corr'],['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','ruwe','phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag','phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error','dr2_radial_velocity','dr2_radial_velocity_error','phot_bp_rp_excess_factor_corrected']],
               ['vizier:II/246/out',['2MASS','RA','DEC','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag','Qfl']],
               ['vizier:II/328/allwise',['AllWISE','RAJ2000','DEJ2000','W1mag','e_W1mag','W2mag','e_W2mag','W3mag','e_W3mag','W4mag','e_W4mag','ccf','d2M']]
              ]
    short={'GAIA_EDR3':'Gaia', '2MASS':'2MASS', 'ALLWISE':'ALLWISE'}
    cat_code={'GAIA_EDR3':'I/350/gaiaedr3', '2MASS':'II/246/out', 'ALLWISE':'II/328/allwise'}
    if isinstance(surveys,str): surveys=[surveys]
    ns=len(surveys)
    
    #is the input file a coordinate file or a list of star names?
    if coordinates==True: #list of coordinates
        header = str(np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=1, comments='-'))
        old_coo = np.genfromtxt(filename,delimiter=delim,skip_header=1,comments='#')
        if "galactic" in header:
            gc = SkyCoord(l=old_coo[:,0]*u.degree, b=old_coo[:,1]*u.degree, frame='galactic')
            ec=gc.icrs
            coo_array=np.transpose([ec.ra.deg,ec.dec.deg])
        elif "equatorial" in header:
            coo_array=old_coo        
        n=len(coo_array)
    else: #list of star names
        with open(filename) as f:
            target_list = np.genfromtxt(f,dtype="str",delimiter='*@.,')
        n=len(target_list)
        gex=0
        coo_array=np.zeros([n,2])
        for i in range(n):
            x=Simbad.query_object(target_list[i])
            if type(x)==type(None):
                x=Vizier.query_object(target_list[i],catalog='I/350/gaiaedr3',radius=1*u.arcsec) #tries alternative resolver
                if len(x)>0:
                    if len(x[0]['RAJ2000'])>1:
                        if 'Gaia' in target_list[i]:
                            c=0
                            while str(x[0]['Source'][c]) not in target_list[i]:
                                c=c+1
                                if c==len(x[0]['RAJ2000']): break
                            if c==len(x[0]['RAJ2000']): c=np.argmin(x[0]['Gmag'])
                            coo_array[i,0]=x[0]['RAJ2000'][c]
                            coo_array[i,1]=x[0]['DEJ2000'][c]
                    else:
                        coo_array[i,0]=x[0]['RAJ2000']
                        coo_array[i,1]=x[0]['DEJ2000']
                else:
                    coo_array[i,0]=np.nan
                    coo_array[i,1]=np.nan                
                    print('Star',target_list[i],' not found. Perhaps misspelling? Setting row to NaN.')
                    gex=1
            else:
                coo_array[i,0]=Angle(ang_deg(x['RA'].data.data[0])).degree
                coo_array[i,1]=Angle(ang_deg(x['DEC'].data.data[0],form='dms')).degree                
                
        if gex:
            print('Some stars were not found. Would you like to end the program and check the spelling?')
            print('If not, these stars will be treated as missing data')
            key=str.lower(input('End program? [Y/N]'))
            while 1:
                if key=='yes' or key=='y':
                    print('Program ended.')
                    return
                elif key=='no' or key=='n':
                    break
                key=str.lower(input('Unvalid choice. Type Y or N.'))                
    
    if verbose==True:
        with open(new_file, newline='', mode='w') as f:
            r_csv = csv.writer(f, delimiter=',')
            r_csv.writerow(['ra','dec'])
            for i in range(len(coo_array)):
                r_csv.writerow([coo_array[i,0],coo_array[i,1]])
        
    #turns coo_array into a Table for XMatch
    coo_table = Table(coo_array, names=('RA', 'DEC'))

    #finds data on VizieR through a query on XMatch
    data_dic={}
    for i in range(len(surveys)): 
        data_s = XMatch.query(cat1=coo_table,cat2=surv_prop[index[surveys[i]]][0],max_distance=1.3 * u.arcsec, colRA1='RA',colDec1='DEC')
        data_dic[surveys[i]]=data_s
        if verbose==True:
            f=open(os.path.join(folder,str(sample_name+'_'+surveys[i]+'_data.txt')), "w+")
            if surveys[i]=='GAIA_EDR3':
                f.write(tabulate(data_s[surv_prop[0][2]],
                             headers=['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','ruwe','G','G_err','GBP','GBP_err','GRP','GRP_err','radial_velocity', 'radial_velocity_error','bp_rp_excess_factor'], tablefmt='plain', stralign='right', numalign='right', floatfmt=(".5f",".11f",".4f",".11f",".4f",".7f",".7f",".7f",".7f",".7f",".7f",".3f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f")))
                data_s=data_s[surv_prop[0][1]]
            elif surveys[i]=='2MASS':
                f.write(tabulate(data_s[surv_prop[1][1]],
                             headers=['ID','ra','dec','J','J_err','H','H_err','K','K_err','qfl'], tablefmt='plain', stralign='right', numalign='right', floatfmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f")))
            elif surveys[i]=='ALLWISE':
                f.write(tabulate(data_s[surv_prop[2][1]],
                             headers=['ID','ra','dec','W1','W1_err','W2','W2_err','W3','W3_err','W4','W4_err','ccf','d2M'], tablefmt='plain', stralign='right', numalign='right', floatfmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".4f")))
            f.close()

    #a Table per survey, retrievable through a keyword: e.g., data_dic['2MASS']        

    return coo_array,data_dic


def Wu_line_integrate(f,x0,x1,y0,y1,z0,z1):
    n=100*max(math.ceil(abs(max([x1-x0,y1-y0,z1-z0]))),500)
    I=0
    dim=f.shape    
    i=0
    
    x=np.linspace(x0,x1,num=n)
    if n_dim(f)==2:
        m=(y1-y0)/(x1-x0) #slope of the line
        d10=np.sqrt((x1-x0)**2+(y1-y0)**2) #distance
        
        y=y0+m*(x-x0)
        while (x[i]<dim[0]) & (y[i]<dim[1]) & (i<n):
            I+=f[math.floor(x[i]),math.floor(y[i])]
            i+=1
    elif n_dim(f)==3:
        m=(y1-y0)/(x1-x0) #slope of the line
        d10=np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2) #distance
        
        y=np.linspace(y0,y1,num=n)
        z=np.linspace(z0,z1,num=n)
        while (x[i]<dim[0]) & (y[i]<dim[1]) & (z[i]<dim[2]) & (i<n):
            I+=f[math.floor(x[i]),math.floor(y[i]),math.floor(z[i])]
            i+=1
    
    return I/n*d10

def interstellar_ext(ra=None,dec=None,l=None,b=None,par=None,d=None,test_time=False,ext_map='leike',color='B-V'):

    if ext_map=='leike': fname='leike_mean_std.h5'
    elif ext_map=='stilism': fname='STILISM_v.fits'
    else: fname='leike_mean_std.h5' #change for other maps
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
        if n_elements(ra)>1: ebv=np.zeros(n_elements(ra))
        else: ebv=0.
        return ebv

    fits_image_filename=os.path.join(map_path,fname)
    f = h5py.File(fits_image_filename,'r')
    data = f['mean']

    x=np.arange(-370.,370.)
    y=np.arange(-370.,370.)
    z=np.arange(-270.,540.)

    if type(ra)==type(None) and type(l)==type(None): raise NameError('At least one between RA and l must be supplied!') # ok=dialog_message(')
    if type(dec)==type(None) and type(b)==type(None): raise NameError('At least one between dec and b must be supplied!')
    if type(par)==type(None) and type(d)==type(None): raise NameError('At least one between parallax and distance must be supplied!')
    if type(ra)!=type(None) and type(l)!=type(None): raise NameError('Only one between RA and l must be supplied!')
    if type(dec)!=type(None) and type(b)!=type(None): raise NameError('Only one between dec and b must be supplied!')
    if type(par)!=type(None) and type(d)!=type(None): raise NameError('Only one between parallax and distance must be supplied!')

    sun=[closest(x,0),closest(z,0)]
    
    if type(ra)!=type(None): #computes galactic l and b, if missing
        c_eq = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        l=c_eq.galactic.l.degree
        b=c_eq.galactic.b.degree
    if type(d)==type(None): d=1000./par #computes heliocentric distance, if missing

    #  ;Sun-centered cartesian Galactic coordinates. X and Y are in the midplane
    x0=d*np.cos(l*np.pi/180)*np.cos(b*np.pi/180) #X is directed to the Galactic Center
    y0=d*np.sin(l*np.pi/180)*np.cos(b*np.pi/180) #Y is in the sense of rotation
    z0=d*np.sin(b*np.pi/180) #Z points to the north Galactic pole

    px=closest(x,x0)
    py=closest(y,y0)
    pz=closest(z,z0)

    dist=x[1]-x[0]


    if n_elements(px)==1:
        if px<len(x)-1: px2=(x0-x[px])/dist+px
        if py<len(y)-1: py2=(y0-y[py])/dist+py
        if pz<len(z)-1: pz2=(z0-z[pz])/dist+pz
        if ext_map=='stilism': ebv=dist*Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2)/3.16
        elif ext_map=='leike': ebv=dist*(2.5*Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2)*np.log10(np.exp(1)))/3.16/0.789
    else:
        wx,=np.where(px<len(x)-1)
        wy,=np.where(py<len(y)-1)
        wz,=np.where(pz<len(z)-1)
        px2=px.astype(float)
        py2=py.astype(float)    
        pz2=pz.astype(float)
        px2[wx]=(x0[wx]-x[px[wx]])/dist+px[wx]
        py2[wy]=(y0[wy]-y[py[wy]])/dist+py[wy]
        pz2[wz]=(z0[wz]-z[pz[wz]])/dist+pz[wz]    
        ebv=np.empty(n_elements(x0))
        ebv.fill(np.nan)
        if ext_map=='stilism':
            for i in range(n_elements(x0)):
                if np.isnan(px2[i])==0:
                    ebv[i]=dist*Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i])/3.16
        elif ext_map=='leike':
            for i in range(n_elements(x0)):
                if np.isnan(px2[i])==0:
                    ebv[i]=dist*(2.5*Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i])*np.log10(np.exp(1)))/3.16/0.789

    if color=='B-V': return ebv
    else: return extinction(ebv,color)

def cross_match(cat1,cat2,max_difference=0.01,other_column=None,rule=None,exact=False,parallax=None,min_parallax=2):
    """
    given two catalogues cat1 and cat2, returns the indices ind1 and ind2 such that:
    cat1[ind1]=cat2[ind2]
    if the input are 1D, or
    cat1[ind1,:]=cat2[ind2,:]
    if they are 2D.
    '=' must be thought as "for each i, |cat1[ind1[i]]-cat2[ind2[i]]|<max_difference",
    if exact=False (default mode). If exact=True, it is a strict equality.
    
    input:
        cat1: a 1D or 2D numpy array, specifying one row per star, in the first catalogue
        cat2: a 1D or 2D numpy array, specifying one row per star, in the second catalogue
            The number of columns of cat2 must be the same of cat1!
        max_difference: the threshold k to cross-match two entries. If |cat1[i]-cat2[j]|<k,
            the sources are considered the same and cross-matched
        other_column (optional): a 1D array with one entry per cat2 row. If a source of cat1 has more than
            1 source in cat2 meeting the cross-match criterion, it selects the one that has the lowest/highest
            value in other_column. If not set, the first occurrence will be returned.
        rule (optional): mandatory if other_column is set. rule='min' to select the lowest value,
            rule='max' to select the highest value
        exact: whether to look for an exact cross-match (e.g., for IDs or proper names) or not.

    usage:
        if exact=False (default mode):
            let cat1, for instance, be a 2D array [[ra1],[dec1]], with coordinates of n stars
            cat2 will be a similar [[ra2],[dec2]] in a second catalogue        
            cross_match(cat1,cat2,max_difference=0.03,other_column=radius,rule='min')
            tries to find, for each i, the star such that |ra1[i]-ra2|+|dec1[i]-dec2|<0.03.
            If two stars cat2[j,:] and cat2[k,:] are returned, it picks the j-th if radius[j]<radius[k], the k-th otherwise
        if exact=True:
            cross_match(cat1,cat2,exact=True)
            returns ind1, ind2 such that cat1[ind1]=cat2[ind2] (strict equality).
        
    notes:
    if exact=False:
        The number of columns of cat2 must be the same of cat1.
        The number of elements in other_column must equal the number of rows of cat2.
        rule must be set if other_column is set.
    if exact=True:
        no keyword other than cat1 and cat2 will be used.
        cat1 and cat2 must be 1D arrays.
        Be sure that cat2 does not contain any repeated entries, otherwise only one of them will be picked.

    """

    if exact==True:
        if (n_dim(cat1)!=1) or (n_dim(cat2)!=1):
            raise ValueError("cat1 and cat2 must be 1D!")
        c1=np.argsort(cat2)
        c=closest(cat2[c1],cat1)
        ind1,=np.where(cat2[c1[c]]==cat1)
        return ind1,c1[c[ind1]]

    n=len(cat1)
    ind1=np.zeros(n,dtype='int32')
    ind2=np.zeros(n,dtype='int32')
    c=0
    if type(parallax)==type(None): parallax=3.
    if n_dim(cat1)==1:
        if type(other_column)==type(None):
            for i in range(n):
                k,=np.where((abs(cat2-cat1[i])<max_difference) & (parallax>min_parallax))
                if len(k)==1:
                    ind1[c]=i
                    ind2[c]=k
                    c+=1
                elif len(k)>1:
                    ind1[c]=i
                    ind2[c]=k[0]
                    c+=1
        elif rule=='min':
            for i in range(n):
                k,=np.where((abs(cat2-cat1[i])<max_difference) & (parallax>min_parallax))
                if len(k)==1:
                    ind1[c]=i
                    ind2[c]=k
                    c+=1
                elif len(k)>1:
                    ind1[c]=i
                    k1=np.argmin(other_column[k])
                    ind2[c]=k[k1]
                    c+=1
        elif rule=='max':
            for i in range(n):
                k,=np.where((abs(cat2-cat1[i])<max_difference) & (parallax>min_parallax))
                if len(k)==1:
                    ind1[c]=i
                    ind2[c]=k
                    c+=1
                elif len(k)>1:
                    ind1[c]=i
                    k1=np.argmax(other_column[k])
                    ind2[c]=k[k1]
                    c+=1
        else: raise NameError("Keyword 'rule' not set! Specify if rule='min' or rule='max'")                    
    else:
        if type(cat1)==Table: 
            cat1=np.lib.recfunctions.structured_to_unstructured(np.array(cat1))        
        if type(cat2)==Table: 
            cat2=np.lib.recfunctions.structured_to_unstructured(np.array(cat2))
        
        if len(cat1[0])!=len(cat2[0]): 
            raise ValueError("The number of columns of cat1 must equal that of cat2.")
        if type(other_column)==type(None):
            for i in range(n):
                d=0
                for j in range(len(cat1[0])): d+=abs(cat2[:,j]-cat1[i,j])
                k,=np.where((d<max_difference) & (parallax>min_parallax))
                if len(k)==1:
                    ind1[c]=i
                    ind2[c]=k
                    c+=1
                elif len(k)>1:
                    ind1[c]=i
                    ind2[c]=k[0]
                    c+=1
        elif rule=='min':
            if len(other_column)!=len(cat2): 
                raise ValueError("The length of other_column must equal the no. of rows of cat2.")
            for i in range(n):
                d=0
                for j in range(len(cat1[0])): d+=abs(cat2[:,j]-cat1[i,j])
                k,=np.where((d<max_difference) & (parallax>min_parallax))
                if len(k)==1:
                    ind1[c]=i
                    ind2[c]=k
                    c+=1
                elif len(k)>1:
                    ind1[c]=i
                    k1=np.argmin(other_column[k])
                    ind2[c]=k[k1]
                    c+=1
        elif rule=='max':
            if len(other_column)!=len(cat2): 
                raise ValueError("The length of other_column must equal the no. of rows of cat2.")
            for i in range(n):
                d=0
                for j in range(len(cat1[0])): d+=abs(cat2[:,j]-cat1[i,j])
                k,=np.where((d<max_difference) & (parallax>min_parallax))
                if len(k)==1:
                    ind1[c]=i
                    ind2[c]=k
                    c+=1
                elif len(k)>1:
                    ind1[c]=i
                    k1=np.argmax(other_column[k])
                    ind2[c]=k[k1]
                    c+=1
        else: raise NameError("Keyword 'rule' not set! Specify if rule='min' or rule='max'")                
    ind1=ind1[0:c]
    ind2=ind2[0:c]
    return ind1,ind2

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
    if isinstance(files,str):
        try:
            open(files,'r')
        except IOError:
            return 0
    else:
        for i in range(n_elements(files)):
            try:
                open(files[i],'r')
            except IOError:
                return 0
    return 1

def load_phot(filename,surveys):

    if n_elements(surveys)==1: surveys=[surveys]
    is_g=0
    while 'gaia' not in str.lower(surveys[is_g]):
        is_g+=1
        if is_g==len(surveys): break
    if is_g==len(surveys):
        print('Gaia is missing! Perhaps you should consider using it to have reliable parallaxes.')

    path=os.path.dirname(filename)
    sample_name=os.path.split(filename)[1]
    i=0
    while sample_name[i]!='.': i=i+1
    sample_name=sample_name[0:i]
    if sample_name.endswith('_coordinates'): sample_name=sample_name[0:-12]

    file=''+sample_name
    for i in range(len(surveys)): file+='_'+surveys[i]
    PIK=os.path.join(path,('photometry_'+file+'.pkl'))

    try: #se c'è
        open(PIK,'r')
        with open(PIK,'rb') as f:
            phot=pickle.load(f)
            phot_err=pickle.load(f)
            filt=pickle.load(f)
            kin=pickle.load(f)
    except IOError:
        survey_files = [os.path.join(path,sample_name+'_'+x+'_data.txt') for x in surveys]
        if file_search(survey_files)==0:
            search_phot(os.path.join(path,(sample_name+'.txt')),coordinates=False,surveys=['GAIA_EDR3','2MASS','ALLWISE'])
        coo_file=os.path.join(path,(sample_name+'_coordinates.csv'))
        coo = np.genfromtxt(coo_file, names=True, delimiter=',') #coo_h=coo.dtype.names
        nst=len(coo)
        cat1=np.zeros([nst,2])
        cat1[:,0]=coo['ra']
        cat1[:,1]=coo['dec']    

        crit={'GAIA_EDR3':'G','2MASS':'J','ALLWISE':'W1'}
        n_filters={'GAIA_EDR3':3, '2MASS':3, 'ALLWISE':4}
        f_list={'GAIA_EDR3':['G','GBP','GRP'],'2MASS':['J','H','K'],'ALLWISE':['W1','W2','W3','W4']}
        kin_list=['ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','radial_velocity','radial_velocity_error']

        nf=0
        for i in range(len(surveys)): nf+=n_filters[surveys[i]]

        phot=np.empty([nst,nf])
        phot.fill(np.nan)
        phot_err=np.empty([nst,nf])
        phot_err.fill(np.nan)
        kin=np.empty([nst,12])
        kin.fill(np.nan)

        p=0
        filt=[]
        for i in range(len(surveys)):
            cat2_data = np.genfromtxt(survey_files[i], names=True) #header=cat2_data.dtype.names
            n_cat2=len(cat2_data)
            cat2=np.zeros([n_cat2,2])
            cat2[:,0]=cat2_data['ra']
            cat2[:,1]=cat2_data['dec']
            try:
                para=cat2_data['parallax']
            except ValueError: para=np.full(n_cat2,1000)
            mag=cat2_data[crit[surveys[i]]]
            indG1,indG2=cross_match(cat1,cat2,max_difference=0.001,other_column=mag,rule='min',parallax=para)
            for j in range(len(f_list[surveys[i]])):
                f_i=f_list[surveys[i]][j]
                mag_i=cat2_data[f_i]
                phot[indG1,j+p]=mag_i[indG2]
                mag_err=cat2_data[f_i+'_err']
                phot_err[indG1,j+p]=mag_err[indG2]
            p+=len(f_list[surveys[i]])
            filt.extend(f_list[surveys[i]])
            if i==is_g:
                for j in range(len(kin_list)):
                    kin_i=cat2_data[kin_list[j]] 
                    kin[indG1,j]=kin_i[indG2]

        filt2=[s+'_err' for s in filt]

        fff=[]
        fff.extend(filt)
        fff.extend(filt2)

        np.concatenate((phot,phot_err),axis=1)
        f=open(os.path.join(path,(sample_name+'_photometry.txt')), "w+")
        f.write(tabulate(np.concatenate((phot,phot_err),axis=1),
                     headers=fff, tablefmt='plain', stralign='right', numalign='right'))
        f.close()

        f=open(os.path.join(path,(sample_name+'_kinematics.txt')), "w+")
        f.write(tabulate(kin,
                     headers=kin_list, tablefmt='plain', stralign='right', numalign='right'))
        f.close()

    with open(PIK,'wb') as f:
        pickle.dump(phot,f)
        pickle.dump(phot_err,f)
        pickle.dump(filt,f)
        pickle.dump(kin,f)
    
    return phot,phot_err,filt,kin


def plot_CMD(x,y,isochrones,iso_filters,iso_ages,x_axis,y_axis,plot_ages=[1,3,5,10,20,30,100],ebv=None,tofile=False,x_error=None,y_error=None,groups=None,group_names=None,label_points=False):

    """
    plots the CMD of a given set of stars, with theoretical isochrones overimposed

    input:
        x: data to be plotted on the x-axis
        y: data to be plotted on the y-axis
        isochrones: a 3D isochrone grid M(masses,ages,filters)
        iso_filters: isochrone grid filter list
        iso_ages: isochrone grid ages
        x_axis: name of the x axis, e.g. 'G-K'
        y_axis: name of the y axis
        plot_ages: ages (in Myr) of the isochrones to be plotted. Default: [1,3,5,10,20,30,100] 
        ebv (optional): color excess E(B-V) of the sources
        tofile: specify a filename if you want to save the output as a figure. Default: plots on the
            command line
        x_error (optional): errors on x
        y_error (optional): errors on y
        groups (optional): a vector with same size as x, indicating the group which the corresponding
            star belongs to
        group_names (mandatory if groups is set): labels of 'groups'
        label_points (optional): an array with labels for each star. Default=False.
        
    usage:
        let G, K be magnitudes of a set of stars with measured errors G_err, K_err;
        isochrones the theoretical matrix, with 'filters' and 'ages' vectors
        the stars are divided into two groups: label=['group0','group1'] through the array 'classes'
        plot_CMD(G-K,G,isochrones,filters,ages,'G-K','G',x_error=col_err,y_error=mag_err,groups=classes,group_names=label,tofile=file)
        draws the isochrones, plots the stars in two different colors and saves the CMD on the file 'file'
        An array with the (median) extinction/color excess is plotted, pointing towards dereddening.
        If label_points==True, sequential numbering 0,1,...,n-1 is used. If given an array, it uses them as labels

    """

    #axes ranges
    x_range=axis_range(x_axis,x)
    y_range=axis_range(y_axis,y)
    
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
    tot_iso=len(isochrones[0]) #no. of grid ages
    npo=n_elements(x) #no. of stars
    nis=len(plot_ages) #no. of isochrones to be plotted

        
#    fig=plt.figure(figsize=(16,12))
    fig, ax = plt.subplots(figsize=(16,12))
    
    if type(ebv)!=type(None): #subtracts extinction, if E(B-V) is provided
        x_ext=extinction(ebv,x_axis)
        y_ext=extinction(ebv,y_axis)
        x1=x-x_ext
        y1=y-y_ext
    else:
        x1=x
        y1=y
    plt.arrow(x_range[0]+0.2*(x_range[1]-x_range[0]),y_range[0]+0.1*(y_range[1]-y_range[0]),-np.median(x_ext),-np.median(y_ext),head_width=0.05, head_length=0.1, fc='k', ec='k', label='reddening')
    
    for i in range(len(plot_ages)):
        ii=closest(iso_ages,plot_ages[i])
        plt.plot(col_th[:,ii],mag_th[:,ii],label=str(plot_ages[i])+' Myr')

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
        n=(np.linspace(0,npo-1,num=npo,dtype=int)).astype('str')
        for i, txt in enumerate(n):
            print(i,txt)
            ax.annotate(txt, (x1[i], y1[i]))
    elif label_points!=False:
        if isinstance(label_points[0],str): label_points=label_points.astype('str')
        for i, txt in enumerate(label_points):
            ax.annotate(txt, (x1[i], y1[i]))
        
    
    plt.ylim(y_range)
    plt.xlim(x_range)
    plt.xlabel(x_axis, fontsize=18)
    plt.ylabel(y_axis, fontsize=18)
    plt.legend()
    if tofile==False:
        plt.show()
    else:
        plt.savefig(tofile)
        plt.close(fig)    
    
    return None

def ang_dist(ra1,dec1,ra2,dec2):  
    
    dist=2*np.arcsin(np.sqrt(np.sin((dec2-dec1)/2.*u.degree)**2+np.cos(dec2*u.degree)*np.cos(dec1*u.degree)*np.sin((ra2-ra1)/2.*u.degree)**2)).to(u.deg)

    return dist.value
