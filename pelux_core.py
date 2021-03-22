# coding: utf-8

import numpy as np
from pathlib import Path
import sys
sys.path.append("C:\\Users\\Vito\\Downloads\\Python-utils-master\\vigan\\astro")
from evolution import *
from scipy.interpolate import interp1d
from astropy.constants import M_jup,M_sun
import time
import pickle


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
    '''Given an "array" and a "value", finds the j such that |array[j]-value|=min((array-value)).
    "array" must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
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

def load_isochrones(model,surveys=None,folder=Path('C:/Users/Vito/Desktop/PhD/Progetti/BEAST/Stellar_ages/CMS/BT-Settl_stilism_cs0_GaiaDR3')):
    if surveys==None: surveys=['gaia','2mass','wise']
    else: surveys=list(map(str.lower,surveys))

    model=(str.lower(model)).replace('-','_')

    file=model
    for i in range(len(surveys)): file=file+'_'+surveys[i]
    PIK=folder / (file+'.pkl')

    try: #se c'è
        with open(PIK,'rb') as f:
            iso_f=pickle.load(f)
            mnew=pickle.load(f)
            anew=pickle.load(f)
            fnew=pickle.load(f)
    except IOError:
        fnew=[]
        nf=0

        survey_list=['gaia','2mass','wise','johnson','panstarrs','sloan','sphere']
        survey_el=[3,3,4,5,5,7,13]
        survey_filt=[['G','Gbp','Grp'],['J','H','K'],['W1','W2','W3','W4'],['U','B','V','R','i'],['gmag','rmag','imag','zmag','ymag'],['V_sl','R_sl','I_sl','K_sl','R_sl2','Z_sl','M_sl'],['Ymag','Jmag','Hmag','Kmag','H2mag','H3mag','H4mag','J2mag','J3mag','K1mag','K2mag','Y2mag','Y3mag']]
        survey_col=[['G2018','G2018_BP','G2018_RP'],['J','H','K'],['W1_W10','W2_W10','W3_W10','W4_W10'],['U','B','V','R','i'],['g_p1','r_p1','i_p1','z_p1','y_p1'],['V','R','I','K','Rsloan','Zsloan','Msloan'],['B_Y','B_J','B_H','B_Ks','D_H2','D_H3','D_H4','D_J2','D_J3','D_K1','D_K2','D_Y2','D_Y3']]


        nf=0
        for i in range(len(survey_list)):
            if survey_list[i] in surveys: nf+=survey_el[i]

        c=0
        for i in range(len(survey_list)):
            if survey_list[i] in surveys:
                fnew.extend(survey_filt[i])
                masses, ages, v0, data0 = model_data(survey_list[i],model)
                if 'data' not in locals():
                    nm=len(masses)
                    na=len(ages)
                    data=np.zeros([nm,na,nf])
                    n1=30*nm
                    n2=10*na
                    mnew=masses[0]+(masses[-1]-masses[0])/(n1-1)*np.arange(n1)
                    anew=np.exp(np.log(ages[0])+(np.log(ages[-1])-np.log(ages[0]))/(n2-1)*np.arange(n2))
                    iso_f=np.empty([n1,n2,nf]) #matrice con spline in età, devo completarla
                    iso_f.fill(np.nan)
                iso=np.empty([n1,len(ages),survey_el[i]]) #matrice con spline in massa, devo completarla
                iso.fill(np.nan)
                for j in range(len(survey_filt[i])):
                    w,=np.where(v0==survey_col[i][j])
                    for k in range(len(ages)): #spline in massa. Per ogni età
                        nans, x= nan_helper(data0[:,k,w])
                        nans=nans.reshape(len(nans))
                        m0=masses[~nans]
                        f=interp1d(masses[~nans],data0[~nans,k,w],kind='linear')
                        c1=0
                        while mnew[c1]<m0[0]: c1=c1+1
                        c2=c1
                        while mnew[c2]<m0[-1]: c2=c2+1 #sarà il primo elemento maggiore, ma quando farò l'indexing, a[c1,c2] equivarrà ad a[c1,...,c2-1]
                        iso[c1:c2,k,j]=f(mnew[c1:c2])
                    for k in range(n1): #spline in età. Per ogni massa
                        nans, x= nan_helper(iso[k,:,j])
                        nans=nans.reshape(n_elements(nans))
                        a0=ages[~nans]
                        if n_elements(ages[~nans])==0: continue
                        f=interp1d(ages[~nans],iso[k,~nans,j],kind='linear')
                        c1=0
                        while anew[c1]<a0[0]: c1=c1+1
                        c2=c1
                        while anew[c2]<a0[-1]: c2=c2+1 #sarà il primo elemento maggiore, ma quando farò l'indexing, a[c1,c2] equivarrà ad a[c1,...,c2-1]
                        iso_f[k,c1:c2,j+c]=f(anew[c1:c2])
                c+=survey_el[i]
        mnew=M_jup/M_sun*mnew
        fnew=np.array(fnew)
        with open(PIK,'wb') as f:
            pickle.dump(iso_f,f)
            pickle.dump(mnew,f)
            pickle.dump(anew,f)
            pickle.dump(fnew,f)

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
