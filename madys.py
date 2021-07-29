import os
import sys
import numpy as np
import time
import pickle
import h5py
from astropy.coordinates import Angle, SkyCoord, Galactocentric, galactocentric_frame_defaults
from astropy import units as u
from pathlib import Path
from tabulate import tabulate
from astropy.table import Table
from astroquery.xmatch import XMatch
import warnings
from astropy.utils.exceptions import AstropyWarning
import logging
from astropy.constants import M_jup,M_sun
from scipy.interpolate import interp1d
working_path=os.getcwd()
sys.path.append(working_path)
from evolution import *


#################################################################
# UTILITY FUNCTIONS
def n_elements(x):
    size = 1
    for dim in np.shape(x): size *= dim
    return size

def where_v(elements,array,approx=False):
    
    if isinstance(array,list): array=np.array(array)
    try:
        dd=len(elements)
        if isinstance(elements,list): elements=np.array(elements)
        dim=len(elements.shape)
    except TypeError: dim=0
    
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
            if len(w)==0: ind[i]=len(array) #so it will raise an error
            else: ind[i]=w[0]
        return ind

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
                if (value[i] == array[0]):# edge cases at bottomlog
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
        for i in range(n_elements(files)):
            try:
                open(files[i],'r')
            except IOError:
                return 0
    return 1

def cross_match(cat1,cat2,max_difference=0.01,other_column=None,rule=None,exact=False,parallax=None,min_parallax=2):

    def n_dim(x,shape=False):
        if isinstance(x,str): return 0
        try:
            b=len(x)
            try:
                x.shape
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

def split_if_nan(a):
    ind=[]
    res=[]
    for s in np.ma.clump_unmasked(np.ma.masked_invalid(a)):
        ind.append(s)
        res.append(a[s])    
    return res,ind


#################################################################
# ASTRONOMICAL FUNCTIONS

def Wu_line_integrate(f,x0,x1,y0,y1,z0,z1,layer=None,star_id=None,to_log=False):
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
            if (len(w)<len(w_g)) & (to_log==True):
                logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')                
            w2=w_g[w]
            for i in range(len(w2)): I+=f[x[w2[i]],y[w2[i]]]*w_f[w[i]]
        elif ndim==3:
            d10=np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2) #distance        
            z=np.floor(np.linspace(z0,z1,num=n)).astype(int)           
            w_g,=np.where((x[1:]!=x[:-1]) | (y[1:]!=y[:-1]) | (z[1:]!=z[:-1]))
            w_g=np.insert(w_g+1,0,0)
            w_f=np.insert(w_g[1:]-w_g[:-1],len(w_g)-1,len(x)-w_g[-1])            
            w,=np.where((x[w_g]<dim[0]) & (y[w_g]<dim[1]) & (z[w_g]<dim[2]))
            if (len(w)<len(w_g)) & (to_log==True):
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
            if (len(w)<len(w_g)) & (to_log==True):
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
            if (len(w)<len(w_g)) & (to_log==True):
                logger.info('Star '+str(star_id)+' outside the extinction map. Its extinction is an underestimate.')                
            w2=w_g[w]
            for i in range(len(w2)): I+=f[layer,x[w2[i]],y[w2[i]],z[w2[i]]]*w_f[w[i]]
                
    return I/n*d10

def interstellar_ext(ra=None,dec=None,l=None,b=None,par=None,d=None,test_time=False,ext_map='leike',color='B-V',error=False):

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
        if n_elements(ra)>1: ebv=np.zeros(n_elements(ra))
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

    if type(ra)==type(None) and type(l)==type(None): raise NameError('At least one between RA and l must be supplied!')
    if type(dec)==type(None) and type(b)==type(None): raise NameError('At least one between dec and b must be supplied!')
    if type(par)==type(None) and type(d)==type(None): raise NameError('At least one between parallax and distance must be supplied!')
    if type(ra)!=type(None) and type(l)!=type(None): raise NameError('Only one between RA and l must be supplied!')
    if type(dec)!=type(None) and type(b)!=type(None): raise NameError('Only one between dec and b must be supplied!')
    if type(par)!=type(None) and type(d)!=type(None): raise NameError('Only one between parallax and distance must be supplied!')

    sun=[closest(x,0),closest(z,0)]
    
    #  ;Sun-centered Cartesian Galactic coordinates (right-handed frame)
    if type(d)==type(None): d=1000./par #computes heliocentric distance, if missing
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

    px=closest(x,x0)
    py=closest(y,y0)
    pz=closest(z,z0)

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
            for i in range(n_elements(x0)):
                if np.isnan(px2[i])==0:
                    ebv[i]=dist*Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i],star_id=i,to_log=True)/3.16
        elif ext_map=='leike':
            if error==False: 
                for i in range(n_elements(x0)):
                    if np.isnan(px2[i])==0:
                        ebv[i]=dist*(2.5*Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i],star_id=i,to_log=True)*np.log10(np.exp(1)))/3.16/0.789
            else:
                dim=data.shape
                ebv0=np.full([len(x0),dim[0]],np.nan)
                ebv_s=np.full(len(x0),np.nan)
                for i in range(n_elements(x0)):
                    if np.isnan(px2[i])==0:
                        for k in range(dim[0]):
                            ebv0[i,k]=dist*(2.5*Wu_line_integrate(data,sun[0],px2[i],sun[0],py2[i],sun[1],pz2[i],layer=k,star_id=i,to_log=True)*np.log10(np.exp(1)))/3.16/0.789
                    ebv[i]=np.mean(ebv0[i,:])
                    ebv_s[i]=np.std(ebv0[i,:],ddof=1) #sample std dev                
    except TypeError:
        if px<len(x)-1: px2=(x0-x[px])/dist+px
        if py<len(y)-1: py2=(y0-y[py])/dist+py
        if pz<len(z)-1: pz2=(z0-z[pz])/dist+pz
        if ext_map=='stilism': ebv=dist*Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2,star_id=0,to_log=True)/3.16
        elif ext_map=='leike': 
            if error==False:
                ebv=dist*(2.5*Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2,star_id=0,to_log=True)*np.log10(np.exp(1)))/3.16/0.789
            else:
                dim=data.shape
                ebv0=np.zeros(dim[0])
                for k in range(dim[0]):
                    ebv0[k]=dist*(2.5*Wu_line_integrate(data,sun[0],px2,sun[0],py2,sun[1],pz2,layer=k)*np.log10(np.exp(1)))/3.16/0.789
                ebv=np.mean(ebv0)
                ebv_s=np.std(ebv0,ddof=1) #sample std dev
                    
    if color=='B-V': 
        if error==False:
            return ebv
        else: return ebv,ebv_s
    else: 
        if error==False:
            return extinction(ebv,color)
        else:
            return extinction(ebv,color),extinction(ebv_s,color)

def app_to_abs_mag(app_mag,parallax,app_mag_error=None,parallax_error=None):
    if isinstance(app_mag,list): app_mag=np.array(app_mag)
    if isinstance(parallax,list): parallax=np.array(parallax)
    dm=5*np.log10(100./parallax) #modulo di distanza
    
    try:
        dd=len(app_mag)
        dim=len(app_mag.shape)
    except TypeError: dim=0

    if dim <= 1:
        abs_mag=app_mag-dm
        if (type(app_mag_error)!=type(None)) & (type(parallax_error)!=type(None)): 
            if isinstance(app_mag_error,list): app_mag_error=np.array(app_mag_error)
            if isinstance(parallax_error,list): parallax_error=np.array(parallax_error)
            total_error=np.sqrt(app_mag_error**2+(5/np.log(10)/parallax)**2*parallax_error**2)
            result=(abs_mag,total_error)
        else: result=abs_mag
    else: #  se è 2D, bisogna capire se ci sono più filtri e se c'è anche l'errore fotometrico
        l=app_mag.shape
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

def extinction(ebv,col):
    A_law={'U':1.531,'B':1.317,'V':1,'R':0.748,'I':0.482,'L':0.058,'M':0.023,
       'J':0.243,'H':0.131,'K':0.078,'G':0.789,'Gbp':1.002,'Grp':0.589,
           'G2':0.789,'Gbp2':1.002,'Grp2':0.589,
       'W1':0.039,'W2':0.026,'W3':0.040,'W4':0.020,
       'gmag':1.155,'rmag':0.843,'imag':0.628,'zmag':0.487,'ymag':0.395
      } #absorption coefficients

    if '-' in col:
        c1,c2=col.split('-')
        A=A_law[c1]-A_law[c2]
    else:
        A=A_law[col]
    return 3.16*A*ebv

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

def load_isochrones(model,surveys=['gaia','2mass','wise'],mass_range=[0.01,1.4],age_range=[1,1000],n_steps=[1000,500],feh=None,afe=None,v_vcrit=None,fspot=None,B=0, **kwargs):

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
                 'W1':'W1_mag', 'W2':'nan', 'W3':'nan', 'W4':'nan'} 
        elif model=='dartmouth':
            dic={'B': 'jc_B','V': 'jc_V','R': 'jc_R','I': 'jc_I',
                 'G':'gaia_G','Gbp':'gaia_BP','Grp':'gaia_RP',                 
                 'U':'U','B':'B','V':'V','R':'R','I':'I',
                 'J':'2mass_J','H':'2mass_H','K':'2mass_K'} 
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
        elif model=='ekstroem':
            dic={'lum':'logL', 't_eff':'logTe','V':'Vmag','U-B':'U-B','B-V':'B-V',
                 'B':'Bmag','U':'Umag', 'R':'nan', 'i':'nan'}
        w,=np.where(f_model==dic[filt])
        
        return w

    def model_name(model,feh=None,afe=None,v_vcrit=None,fspot=None,B=0):
        param={'model':model,'feh':0.0,'afe':0.0,'v_vcrit':0.0,'fspot':0.0,'B':0}
        if model=='bt_settl': model2=model
        elif model=='mist':
            feh_range=np.array([-4.,-3.5,-3.,-2.5,-2,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5])
            afe_range=np.array([0.0])
            vcrit_range=np.array([0.0,0.4])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(afe)!=type(None):
                i=np.argmin(abs(afe_range-afe))
                afe0=afe_range[i]
                param['afe']=afe0
                if afe0<0: s='m'
                else: s='p'
                afe1="{:.1f}".format(abs(afe0))            
                model2+='_'+s+afe1
            else: model2+='_p0.0'
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
                param['v_vcrit']=v_vcrit0
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
                param['feh']=feh0
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
                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
                param['v_vcrit']=v_vcrit0
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
                param['fspot']=fspot0
                fspot1="{:.2f}".format(abs(fspot0))            
                model2=model+'_p'+fspot1
            else: model2=model+'_p0.00'
        elif model=='dartmouth':
            feh_range=np.array([0.0])
            afe_range=np.array([0.0])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(afe)!=type(None):
                i=np.argmin(abs(afe_range-afe))
                afe0=afe_range[i]
                param['afe']=afe0
                if afe0<0: s='m'
                else: s='p'
                afe1="{:.1f}".format(abs(afe0))            
                model2+='_'+s+afe1
            else: model2+='_p0.0'
            if B==0: 
                model2+='_nomag'
            else: 
                model2+='_mag'
                param['B']=1  
        elif model=='ekstroem':
            feh_range=np.array([-1.5,0.0])
            vcrit_range=np.array([0.0,0.4])
            if type(feh)!=type(None):
                i=np.argmin(abs(feh_range-feh))
                feh0=feh_range[i]
                param['feh']=feh0
                if feh0<0: s='m'
                else: s='p'
                feh1="{:.2f}".format(abs(feh0))            
                model2=model+'_'+s+feh1
            else: model2=model+'_p0.00'
            if type(v_vcrit)!=type(None):
                i=np.argmin(abs(vcrit_range-v_vcrit))
                v_vcrit0=vcrit_range[i]
                param['v_vcrit']=v_vcrit0
                if v_vcrit0<0.01: s='norot'
                else: s='rot'
                model2+='_'+s
            else: model2+='_norot'
        else: model2=model
        return model2,param

    def filter_model(survey,model):
        if survey!='gaia': return survey
        if model=='bt_settl': return 'gaia_dr2'
        elif model=='mist': return 'gaia_edr3'
        elif model=='parsec': return 'gaia_edr3'
        elif model=='amard': return 'gaia_dr2'    
        elif model=='spots': return 'gaia_dr2'
        elif model=='dartmouth': return 'gaia_dr2'
        elif model=='ames_cond': return 'gaia_dr2'
        elif model=='ames_dusty': return 'gaia_dr2'
        elif model=='bt_nextgen': return 'gaia_dr2'
        elif model=='nextgen': return 'gaia_dr2'
        elif model=='bhac15': return 'gaia_dr2'
        
    filter_vec={'gaia':['G','Gbp','Grp'],'2mass':['J','H','K'],
        'wise':['W1','W2','W3','W4'],'johnson':['U','B','V','R','i'],
         'panstarrs':['gmag','rmag','imag','zmag','ymag'],
         'sloan':['V_sl','R_sl','I_sl','K_sl','R_sl2','Z_sl','M_sl'],
         'sphere':['Ymag','Jmag','Hmag','Kmag','H2mag','H3mag','H4mag','J2mag','J3mag','K1mag','K2mag','Y2mag','Y3mag'],
                'gaia_dr2':['G2','Gbp2','Grp2'],'gaia_edr3':['G','Gbp','Grp']}
    
    surveys=list(map(str.lower,surveys))    
    model=(str.lower(model)).replace('-','_')
    model_code,param=model_name(model,feh=feh,afe=afe,v_vcrit=v_vcrit,fspot=fspot,B=B)
    param['mass_range']=mass_range
    param['age_range']=age_range
    
    file=model_code
    for i in range(len(surveys)): file=file+'_'+sorted(surveys)[i]
    PIK=Path(PIK_path) / (file+'.pkl')

    do_it=0
    if file_search(PIK)==0: do_it=1
    else:
        with open(PIK,'rb') as f:
            iso_f=pickle.load(f)
            mnew=pickle.load(f)
            anew=pickle.load(f)
            fnew=pickle.load(f)
            param0=pickle.load(f)
        if ((param0['mass_range'][0] > mass_range[0]) | (param0['mass_range'][1] < mass_range[1]) |
            (param0['age_range'][0] > age_range[0]) | (param0['age_range'][1] < age_range[1]) |
            (param0['feh']!=param['feh']) | (param0['afe']!=param['afe']) | (param0['v_vcrit']!=param['v_vcrit'])
            | (param0['fspot']!=param['fspot']) | (param0['B']!=param['B'])): 
            del iso_f,mnew,anew
            do_it=1
        elif (param0['age_range'][0] < age_range[0]) | (param0['age_range'][1] > age_range[1]):
            w,=np.where((anew>=age_range[0]) & (anew<=age_range[1]))
            anew=anew[w]
            iso_f=iso_f[:,w,:]

    if do_it:
        fnew=[]

        survey_list=['gaia','2mass','wise','johnson','panstarrs','sloan','sphere']
        survey_el=[3,3,4,5,5,7,13]
        survey_filt=[['G','Gbp','Grp'],['J','H','K'],['W1','W2','W3','W4'],['U','B','V','R','i'],['gmag','rmag','imag','zmag','ymag'],['V_sl','R_sl','I_sl','K_sl','R_sl2','Z_sl','M_sl'],['Ymag','Jmag','Hmag','Kmag','H2mag','H3mag','H4mag','J2mag','J3mag','K1mag','K2mag','Y2mag','Y3mag']]
        survey_col=[['G2018','G2018_BP','G2018_RP'],['J','H','K'],['W1_W10','W2_W10','W3_W10','W4_W10'],['U','B','V','R','i'],['g_p1','r_p1','i_p1','z_p1','y_p1'],['V','R','I','K','Rsloan','Zsloan','Msloan'],['B_Y','B_J','B_H','B_Ks','D_H2','D_H3','D_H4','D_J2','D_J3','D_K1','D_K2','D_Y2','D_Y3']]


        fnew=[]
        for i in range(len(surveys)):
            fnew.extend(filter_vec[filter_model(surveys[i],model)])
        nf=len(fnew)
        c=0
        
        n1=n_steps[0]
        n2=n_steps[1]
        for i in range(len(surveys)):
            masses, ages, v0, data0 = model_data(surveys[i],model_code,**kwargs)
            if 'iso_f' not in locals():
                nm=len(masses)
                na=len(ages)
                mnew=M_sun.value/M_jup.value*np.exp(np.log(0.999*mass_range[0])+(np.log(1.001*mass_range[1])-np.log(0.999*mass_range[0]))/(n1-1)*np.arange(n1))
                anew=np.exp(np.log(1.0001*age_range[0])+(np.log(0.9999*age_range[1])-np.log(1.0001*age_range[0]))/(n2-1)*np.arange(n2))
                iso_f=np.full(([n1,n2,nf]), np.nan) #final matrix
            iso=np.full(([n1,len(ages),len(filter_vec[surveys[i]])]),np.nan)
        
            for j in range(len(filter_vec[surveys[i]])):
                w=filter_code(model,v0,filter_vec[surveys[i]][j])
                if len(w)==0: continue #leaves NaN if the filter is not found
                for k in range(len(ages)): #interpolates along mass
                    gv = np.isfinite(data0[:,k,w]).ravel()
                    m0=masses[gv]
                    if len(m0)>1:
                        f=interp1d(m0,data0[gv,k,w],kind='linear',fill_value=np.nan,bounds_error=False)
                        iso[:,k,j]=f(mnew)
                for k in range(n1):  #interpolates along age
                    gv, igv = split_if_nan((iso[k,:,j]).ravel())
                    for l in range(len(gv)):
                        a0=ages[igv[l]]
                        an,=np.where((anew>0.95*a0[0]) & (anew<1.05*a0[-1]))
                        if len(an)==0: continue
                        if len(a0)>2:
                            f=interp1d(a0,iso[k,igv[l],j],kind='linear',fill_value='extrapolate',bounds_error=False)
                            iso_f[k,an,j+c]=f(anew[an])
                        elif len(a0)==2:
                            f=lambda x: iso[k,igv[l].start,j]+(iso[k,igv[l].stop,j]-iso[k,igv[l].start,j])/(a0[1]-a0[0])*x
                            iso_f[k,an,j+c]=f(anew[an])
                        elif len(a0)==1: iso_f[k,an,j+c]=iso[k,igv[l],j]
            c+=len(filter_vec[surveys[i]])
                        
        mnew=M_jup.value/M_sun.value*mnew
        fnew=np.array(fnew)
        with open(PIK,'wb') as f:
            pickle.dump(iso_f,f)
            pickle.dump(mnew,f)
            pickle.dump(anew,f)
            pickle.dump(fnew,f)
            pickle.dump(param,f)

    return mnew,anew,fnew,iso_f

#################################################################
# TO CREATE LOG FILES

def setup_custom_logger(name,file,mode='a'):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(file, mode=mode)
    handler.setFormatter(formatter)
#    screen_handler = logging.StreamHandler(stream=sys.stdout)
#    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
#    logger.addHandler(screen_handler)
    return logger

#################################################################
# THE CLASS

class MADYS:    
    def __init__(self, file, **kwargs):
        self.file = file
        self.path = os.path.dirname(file)     #working path        
        sample_name=os.path.split(self.file)[1] #file name
        i=0
        while sample_name[i]!='.': i=i+1
        ext=sample_name[i:] #estension
        sample_name=sample_name[:i]        
        self.log_file = Path(self.path) / (sample_name+'_log.txt')
        if 'logger' not in locals():
            logger = setup_custom_logger('madys',self.log_file)
        self.surveys = ['2MASS','WISE','ALLWISE']
        coord_type = 'equatorial'
        verbose = True
        if len(kwargs)>0:
            if 'surveys' in kwargs: self.surveys = kwargs['surveys']
            if 'coord_type' in kwargs: coord_type = kwargs['coord_type']
            if 'model' in kwargs: self.model = kwargs['model']
            if 'mass_range' in kwargs: self.mass_range = kwargs['mass_range']
            if 'verbose' in kwargs: verbose = kwargs['verbose']

        logger.info('Program started')
        logger.info('Input file:')
        logger.info('Coordinate type:'+str(coord_type))
        logger.info('Looking for photometry in the surveys:')

        self.phot,self.phot_err,self.kin,self.flags,self.headers=self.search_phot(verbose=verbose,coordinates=coord_type)
        logger.info('Query ended. Results saved to the file...')
        #logger.error('Error')
        par=self.kin[:,4]
        par_err=self.kin[:,5]
        coo=self.kin[:,[0,2]]
        self.ebv=interstellar_ext(ra=coo[:,0],dec=coo[:,1],par=par)
        self.abs_phot,self.abs_phot_err=app_to_abs_mag(self.phot,par,app_mag_error=self.phot_err,parallax_error=par_err)
        logging.shutdown()
        
    def search_phot(self,coordinates='equatorial',verbose=True,overwrite=False,merge=False):

        surveys=self.surveys
        
        def survey_properties(survey):
            if survey=='GAIA_EDR3':
                code='vizier:I/350/gaiaedr3'
                col1=['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','ruwe','phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag','phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error','dr2_radial_velocity','dr2_radial_velocity_error','phot_bp_rp_excess_factor_corrected'] #last from Riello et al. 2020
                hea=['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','ruwe','G','G_err','Gbp','Gbp_err','Grp','Grp_err','radial_velocity', 'radial_velocity_error','edr3_bp_rp_excess_factor_corr']
                fmt=(".5f",".11f",".4f",".11f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".3f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f")
                f_list=['G','Gbp','Grp']
                q_flags=['ruwe','edr3_bp_rp_excess_factor_corr']
                fill_value=np.nan
            elif survey=='2MASS':
                code='vizier:II/246/out'
                col1=['2MASS','RA','DEC','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag','Qfl']
                hea=['ID','ra','dec','J','J_err','H','H_err','K','K_err','qfl']
                fmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f")
                f_list=['J','H','K']
                q_flags=['qfl']
                fill_value='ZZZ'
            elif survey=='ALLWISE':
                code='vizier:II/328/allwise'
                col1=['AllWISE','RAJ2000','DEJ2000','W1mag','e_W1mag','W2mag','e_W2mag','W3mag','e_W3mag','W4mag','e_W4mag','ccf','d2M']
                hea=['ID','ra','dec','W1','W1_err','W2','W2_err','W3','W3_err','W4','W4_err','ccf','d2M']
                fmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".4f")
                f_list=['W1','W2','W3','W4']
                q_flags=['ccf']
                fill_value='ZZZZ'
            elif survey=='GAIA_DR2':
                code='vizier:I/345/gaia2'
                col1=['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','phot_g_mean_flux','phot_g_mean_flux_error','phot_g_mean_mag','phot_bp_mean_flux','phot_bp_mean_flux_error','phot_bp_mean_mag','phot_rp_mean_flux','phot_rp_mean_flux_error','phot_rp_mean_mag','radial_velocity','radial_velocity_error','phot_bp_rp_excess_factor','phot_bp_rp_excess_factor_corrected']
                hea=['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','G2_flux','G2_flux_err','G2','Gbp2_flux','Gbp2_flux_err','Gbp2','Grp2_flux','Grp2_flux_err','Grp2','radial_velocity','radial_velocity_error','dr2_bp_rp_excess_factor','dr2_bp_rp_excess_factor_corr']
                fmt=(".5f",".11f",".4f",".11f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f")
                f_list=['G2','Gbp2','Grp2']            
                q_flags=['dr2_bp_rp_excess_factor','dr2_bp_rp_excess_factor_corr']
                fill_value=np.nan
            elif survey=='WISE':
                code='vizier:II/311/wise'
                col1=['JNAME','ra','dec','W1mag','e_W1mag','W2mag','e_W2mag','W3mag','e_W3mag','W4mag','e_W4mag','cc_flags']
                hea=['ID','ra','dec','W1_w','W1_w_err','W2_w','W2_w_err','W3_w','W3_w_err','W4_w','W4_w_err','ccf_w']
                fmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f")
                f_list=['W1_w','W2_w','W3_w','W4_w']
                q_flags=['ccf_w']
                fill_value='ZZZZ'

            return code,col1,hea,fmt,f_list,q_flags,fill_value

        def survey_radec(survey):
            if survey=='GAIA_EDR3':
                ra_name='ra_epoch2000'
                dec_name='dec_epoch2000'
            elif survey=='GAIA_DR2':
                ra_name='ra_epoch2000'
                dec_name='dec_epoch2000'
            elif survey=='2MASS':
                ra_name='RAJ2000'
                dec_name='DEJ2000'
            if survey=='ALLWISE':
                ra_name='RAJ2000'
                dec_name='DEJ2000'
            if survey=='WISE':
                ra_name='ra'
                dec_name='dec'        
            return ra_name,dec_name

        #stores path, file name, extension
        sample_name=os.path.split(self.file)[1] #file name
        i=0
        while sample_name[i]!='.': i=i+1
        ext=sample_name[i:] #estension
        sample_name=sample_name[:i]
        if ext=='.csv': delim=','
        else: delim=None


        surveys0=['GAIA_EDR3','GAIA_DR2']
        if isinstance(surveys,str): surveys=[surveys]
        surveys=[x.upper() for x in surveys]
        if merge=='WISE':
            if 'WISE' not in surveys: surveys.append('WISE')
            if 'ALLWISE' not in surveys: surveys.append('ALLWISE')        
        surveys0.extend(surveys)
        surveys=surveys0
        ns=len(surveys)

        file=''+sample_name
        for i in range(len(surveys)): file+='_'+surveys[i]
        PIK=os.path.join(self.path,(file+'.pkl'))

        nf=0 #total no. of filters
        nq=0 #total no. of quality flags
        for i in range(len(surveys)): 
            nf+=len(survey_properties(surveys[i])[4])
            nq+=len(survey_properties(surveys[i])[5])

        if (file_search(PIK)) & (overwrite==0) : #see if the search result is already present
            with open(PIK,'rb') as f:
                phot=pickle.load(f)
                phot_err=pickle.load(f)
                kin=pickle.load(f)
                flags2=pickle.load(f)
                headers=pickle.load(f)
        else:
            #is the input file a coordinate file or a list of star names?
            if coordinates=='equatorial': #list of equatorial coordinates
                coo_array = np.genfromtxt(self.file,delimiter=delim)
                n=len(coo_array)
            elif coordinates=='galactic': #list of galactic coordinates
                old_coo = np.genfromtxt(self.file,delimiter=delim)
                gc = SkyCoord(l=old_coo[:,0]*u.degree, b=old_coo[:,1]*u.degree, frame='galactic')
                ec=gc.icrs
                coo_array=np.transpose([ec.ra.deg,ec.dec.deg])
                n=len(coo_array)
            else: #list of star names
                if ext!='.csv':            
                    with open(self.file) as f:
                        target_list = np.genfromtxt(f,dtype="str",delimiter='*@.,')
                else:
                    target_list = (pd.read_csv(self.file, sep=',', header=0, comment='#', usecols=['Name'])).to_numpy()
                    target_list = target_list.ravel()
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

            kin_list=np.array(['ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','radial_velocity','radial_velocity_error'])

            headers=[]
            phot=np.full([n,nf],np.nan)
            phot_err=np.full([n,nf],np.nan)
            kin=np.full([n,12],np.nan)
            flags=np.full([n,nq],'',dtype='<U10')
            flags2={}        

            #turns coo_array into a Table for XMatch
            coo_table = Table(coo_array, names=('RA', 'DEC'))

            #finds data on VizieR through a query on XMatch
            p=0
            p1=0
            filt=[]
            flag_h=[]
            for i in range(len(surveys)):
                cat_code,col2,hea,fmt,f_list,q_flags,fill_value=survey_properties(surveys[i])
                n_f=len(f_list)
                n_q=len(q_flags)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyWarning)
                    data_s = XMatch.query(cat1=coo_table,cat2=cat_code,max_distance=1.3 * u.arcsec, colRA1='RA',colDec1='DEC')
                if i==1: #aggiunge colonna per BP-RP excess factor
                    C0=(data_s['phot_bp_mean_flux']+data_s['phot_rp_mean_flux'])/data_s['phot_g_mean_flux']
                    data_s['phot_bp_rp_excess_factor']=C0
                    data_G=np.array(np.ma.filled(data_s['phot_g_mean_mag'],fill_value=np.nan))
                    data_dG=np.array(np.ma.filled(data_s['phot_bp_mean_mag']-data_s['phot_rp_mean_mag'],fill_value=np.nan))
                    a0=lambda x: -1.121221*np.heaviside(-(x-0.5),0)-1.1244509*np.heaviside(-(x-3.5),0)-(-1.1244509*np.heaviside(-(x-0.5),0))-0.9288966*np.heaviside(x-3.5,1)
                    a1=lambda x: 0.0505276*np.heaviside(-(x-0.5),0)+0.0288725*np.heaviside(-(x-3.5),0)-(0.0288725*np.heaviside(-(x-0.5),0))-0.168552*np.heaviside(x-3.5,1)
                    a2=lambda x: -0.120531*np.heaviside(-(x-0.5),0)-0.0682774*np.heaviside(-(x-3.5),0)-(-0.0682774*np.heaviside(-(x-0.5),0))
                    a3=lambda x: 0.00795258*np.heaviside(-(x-3.5),0)-(0.00795258*np.heaviside(-(x-0.5),0))
                    a4=lambda x: -0.00555279*np.heaviside(-(x-0.5),0)-0.00555279*np.heaviside(-(x-3.5),0)-(-0.00555279*np.heaviside(-(x-0.5),0))-0.00555279*np.heaviside(x-3.5,1)
                    C1 = C0 + a0(data_dG)+a1(data_dG)*data_dG+a2(data_dG)*data_dG**2+a3(data_dG)*data_dG**3+a4(data_dG)*data_G #final corrected factor
                    data_s['phot_bp_rp_excess_factor_corrected']=C1
                n_cat2=len(data_s)
                cat2=np.zeros([n_cat2,2])
                cat2[:,0]=data_s[survey_radec(surveys[i])[0]]
                cat2[:,1]=data_s[survey_radec(surveys[i])[1]]
                if verbose==True:                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")    
                        f=open(os.path.join(self.path,str(sample_name+'_'+surveys[i]+'_data.txt')), "w+")
                        f.write(tabulate(data_s[col2], 
                                         headers=hea, tablefmt='plain', stralign='right', numalign='right', floatfmt=fmt))
                        data_s=data_s[col2]
                        data_s.rename_columns(col2,hea)        
                        f.close()                
                try:
                    para=data_s['parallax']
                except KeyError: para=np.full(n_cat2,1000)
                mag=data_s[f_list[0]]
                indG1,indG2=cross_match(coo_array,cat2,max_difference=0.001,other_column=mag,rule='min',parallax=para)
                for j in range(n_f):
                    mag_i=data_s[f_list[j]]
                    phot[indG1,j+p]=np.ma.filled(mag_i[indG2],fill_value=np.nan) #missing values replaced by NaN
                    try:
                        mag_err=data_s[f_list[j]+'_err']
                    except KeyError:
                        mag_eofl=data_s[f_list[j]+'_flux_err']/data_s[f_list[j]+'_flux']
                        mag_err=0.5*(2.5*np.log10(1+mag_eofl)-2.5*np.log10(1-mag_eofl))                    
                    phot_err[indG1,j+p]=np.ma.filled(mag_err[indG2],fill_value=np.nan) #missing values replaced by NaN
                p+=n_f
                filt.extend(f_list)
                if i==0:
                    for j in range(len(kin_list)):
                        kin_i=data_s[kin_list[j]] 
                        kin[indG1,j]=np.ma.filled(kin_i[indG2],fill_value=np.nan) #missing values replaced by NaN

                flags2[surveys[i]]={}
                for j in range(n_q):
                    flag=data_s[q_flags[j]]
                    flags[indG1,j+p1]=np.ma.filled(flag[indG2],fill_value=np.nan) #missing values replaced by NaN
                    temp_flag=np.array(np.ma.filled(flag[indG2],fill_value=np.nan))
                    dd=np.full_like(temp_flag,fill_value,shape=n)
                    dd[indG1]=temp_flag
                    flags2[surveys[i]][q_flags[j]]=dd
                p1+=n_q
                flag_h.extend(q_flags)

            filt2=[s+'_err' for s in filt]

            filt=np.array(filt)
            filt2=np.array(filt2)
            flag_h=np.array(flag_h)

            if merge=='WISE':
                w_w=where_v(['W1_w','W2_w','W3_w','W4_w'],filt)
                w_a=where_v(['W1','W2','W3','W4'],filt)
                nw_w=complement_v(w_w,len(filt))
                for j in range(4): 
                    phot[:,w_a[j]]=np.where(isnumber(phot_err[:,w_a[j]],finite=True), phot[:,w_a[j]], phot[:,w_w[j]])
                    phot_err[:,w_a[j]]=np.where(isnumber(phot_err[:,w_a[j]],finite=True), phot_err[:,w_a[j]], phot_err[:,w_w[j]])
                filt=filt[nw_w]
                filt2=filt2[nw_w]
                cc=where_v(['ccf','ccf_w'],flag_h)
                flags[:,cc[0]]=np.where(isnumber(phot_err[:,w_a[j]],finite=True), flags[:,cc[0]], flags[:,cc[1]])
                flags2['WISE']['ccf']=flags[:,cc[0]]
                phot=phot[:,nw_w] #deletes WISE magn
                phot_err=phot_err[:,nw_w]
                ncc=complement_v(cc[1],len(flag_h))
                flags=flags[:,ncc]
                flag_h=flag_h[ncc]

            headers.append(filt)
            headers.append(kin_list)

            fff=[]
            fff.extend(filt)
            fff.extend(filt2)

            if verbose==True:         
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    f=open(os.path.join(self.path,(sample_name+'_photometry.txt')), "w+")
                    f.write(tabulate(np.concatenate((phot,phot_err),axis=1),headers=fff, tablefmt='plain', stralign='right',
                                     numalign='right', floatfmt=".4f"))
                    f.close()

                    f=open(os.path.join(self.path,(sample_name+'_kinematics.txt')), "w+")
                    f.write(tabulate(kin,headers=kin_list, tablefmt='plain', stralign='right', numalign='right', 
                                     floatfmt=(".11f",".4f",".11f",".4f",".4f",".4f",".3f",".3f",".3f",".3f",".3f",".3f")))
                    f.close()

                    f=open(os.path.join(self.path,(sample_name+'_properties.txt')), "w+")
                    f.write(tabulate(flags,headers=flag_h, tablefmt='plain', stralign='right', numalign='right'))
                    f.close()

            with open(PIK,'wb') as f:
                pickle.dump(phot,f)
                pickle.dump(phot_err,f)
                pickle.dump(kin,f)
                pickle.dump(flags2,f)
                pickle.dump(headers,f)

        return phot,phot_err,kin,flags2,headers
        
    def get_age(self, model, **kwargs):
        logger = setup_custom_logger('madys',self.log_file,mode='a')
        self.mass_range =[0.01,1.4]
        self.age_range =[1,1000]
        verbose=True
        if len(kwargs)>0:
            if 'mass_range' in kwargs: self.mass_range = kwargs['mass_range']
            if 'age_range' in kwargs: self.age_range = kwargs['age_range']
            if 'verbose' in kwargs: verbose = kwargs['verbose']
        logger.info('Starting age determination')
        iso=load_isochrones(model,mass_range=self.mass_range,age_range=self.age_range)
        logger.info('Isochrones for model '+model+' correctly loaded.')               
   
        phot_filters=self.headers[0]
        flags=self.flags
        ebv=self.ebv
        output=[self.file,model]
        border_age=False
        ph_cut=0.2
        phot=self.abs_phot
        phot_err=self.abs_phot_err        

        mnew=iso[0]
        anew=iso[1]
        fnew=iso[2]
        newMC=iso[3]

        #selects Gaia DR2 photometry if the isochrones have DR2 filters, EDR3 otherwise
        if 'G2' in fnew: f_right=['J','H','K','G2','Gbp2','Grp2'] #right order
        else: f_right=['J','H','K','G','Gbp','Grp']

        l0=phot.shape
        xlen=l0[0] #no. of stars: 85
        ylen=len(f_right) #no. of filters: 6

        filt=where_v(f_right,fnew)
        filt2=where_v(f_right,phot_filters)

        newMC=newMC[:,:,filt] #ordered columns. Cuts unnecessary columns    
        phot=phot[:,filt2] #ordered columns. Cuts unnecessary columns
        phot_err=phot_err[:,filt2] #ordered columns. Cuts unnecessary columns

        if 'G2' in fnew:
            qfl=flags['GAIA_DR2']['dr2_bp_rp_excess_factor_corr']
            s1=0.004+8e-12*phot[:,3]**7.55
            with np.errstate(invalid='ignore'):           
                q1,=np.where(abs(qfl)>3*s1)
        else:
            qfl=flags['GAIA_EDR3']['edr3_bp_rp_excess_factor_corr']
            s1=0.0059898+8.817481e-12*phot[:,3]**7.618399
            with np.errstate(invalid='ignore'):           
                q1,=np.where(abs(qfl)>3*s1) #excluded
        if len(q1)>0:       
            phot[q1,4]=np.nan
            phot_err[q1,4]=np.nan
            phot[q1,5]=np.nan
            phot_err[q1,5]=np.nan

        qfl=flags['2MASS']['qfl']
        qJ=[]
        qH=[]
        qK=[]
        for i in range(len(qfl)):
            if qfl[i][0]!='A': qJ.append(i)
            if qfl[i][1]!='A': qH.append(i)
            if qfl[i][2]!='A': qK.append(i)
        if len(qJ)>0:
            qJ=np.array(qJ)
            phot[qJ,0]=np.nan
            phot_err[qJ,0]=np.nan
        if len(qH)>0:
            qH=np.array(qH)
            phot[qH,1]=np.nan
            phot_err[qH,1]=np.nan
        if len(qK)>0:
            qK=np.array(qK)
            phot[qK,2]=np.nan
            phot_err[qK,2]=np.nan

        red=np.zeros([xlen,ylen]) #reddening
        if type(ebv)!=type(None):
            for i in range(ylen): red[:,i]=extinction(ebv,f_right[i])
        phot=phot-red

        l=newMC.shape

        a_final=np.full(xlen,np.nan)
        m_final=np.full(xlen,np.nan)
        a_err=np.full(xlen,np.nan)
        m_err=np.full(xlen,np.nan)


        sigma=np.full(([l[0],l[1],ylen]),np.nan)

        for i in range(xlen):
            w,=np.where(is_phot_good(phot[i,:],phot_err[i,:],max_phot_err=ph_cut))
            if len(w)==0: continue
            b=np.zeros(len(w),dtype=bool)
            for h in range(len(w)):
                ph=phot[i,w[h]]
                sigma[:,:,w[h]]=((newMC[:,:,w[h]]-ph)/phot_err[i,w[h]])**2
                ii=divmod(np.nanargmin(sigma[:,:,w[h]]), sigma.shape[1])+(w[h],)
                if abs(newMC[ii]-ph)<0.2: b[h]=True            
            w2=w[b]
            if len(w2)<3: continue
            cr=np.sum(sigma[:,:,w2],axis=2)
            est,ind=min_v(cr)
            crit1=np.sort([sigma[ind+(w2[j],)] for j in range(len(w2))])
            crit2=np.sort([abs(newMC[ind+(w2[j],)]-phot[i,w2[j]]) for j in range(len(w2))])
            if (crit1[2]<9) | (crit2[2]<0.1):
                m_final[i]=mnew[ind[0]]
                a_final[i]=anew[ind[1]]
                m_f1=np.zeros(20)
                a_f1=np.zeros(20)
                for j in range(20):
                    phot1=phot+phot_err*np.random.normal(size=(xlen,ylen))
                    for h in range(len(w2)):
                        sigma[:,:,w2[h]]=((newMC[:,:,w2[h]]-phot1[i,w2[h]])/phot_err[i,w2[h]])**2
                    cr1=np.sum(sigma[:,:,w2],axis=2)
                    est1,ind1=min_v(cr1)
                    m_f1[j]=mnew[ind1[0]]
                    a_f1[j]=anew[ind1[1]]
                m_err[i]=np.std(m_f1,ddof=1)
                a_err[i]=np.std(a_f1,ddof=1)

        if verbose==True:
            filename=output[0]
            model=output[1]
            sample_name=os.path.split(filename)[1] #file name
            i=0
            while sample_name[i]!='.': i=i+1
            ext=sample_name[i:] #estension
            sample_name=sample_name[:i]

            f=open(os.path.join(self.path,str(sample_name+'_ages_'+model+'.txt')), "w+")
            f.write(tabulate(np.column_stack((m_final,m_err,a_final,a_err)),
                             headers=['MASS','MASS_ERROR','AGE','AGE_ERROR'], tablefmt='plain', stralign='right', numalign='right', floatfmt=".2f"))
            f.close()

        logger.info('Age determination ended. Results saved in ... ')
        logging.shutdown()
        return a_final,m_final,a_err,m_err
                
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
                'W4':[cmax,cmin], 'G-J':[cmin,cmax],
                'G-H':[cmin,cmax], 'G-K':[cmin,cmax],
                'G-W1':[cmin,cmax], 'G-W2':[cmin,cmax],
                'G-W3':[cmin,cmax], 'G-W4':[cmin,cmax],
                'J-H':[cmin,cmax], 'J-K':[cmin,cmax],
                'H-K':[cmin,cmax], 'Gbp-Grp':[cmin,cmax]
                }
        else:
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
    
    def CMD(self,col,mag,model,ids=None,**kwargs):

        def filter_model(model,col):
            if model in ['bt_settl','amard','spots','dartmouth','ames_cond','ames_dusty','bt_nextgen','nextgen']:
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
            c1,=np.where(self.headers[0]==col_n[0])
            c2,=np.where(self.headers[0]==col_n[1])
            col1,col1_err=app_to_abs_mag(self.phot[:,c1],self.kin[:,4],app_mag_error=self.phot_err[:,c1],parallax_error=self.kin[:,5])
            col2,col2_err=app_to_abs_mag(self.phot[:,c2],self.kin[:,4],app_mag_error=self.phot_err[:,c2],parallax_error=self.kin[:,5])
            col_data=col1-col2
            col_err=col1_err+col2_err
        else:
            c1,=np.where(self.headers[0]==filter_model(model,col))
            col_data,col_err=app_to_abs_mag(self.phot[:,c1],self.kin[:,4],app_mag_error=self.phot_err[:,c1],parallax_error=self.kin[:,5])
        if '-' in mag:
            mag_n=filter_model(model,mag).split('-')
            m1,=np.where(self.headers[0]==mag_n[0])
            m2,=np.where(self.headers[0]==mag_n[1])
            mag1,mag1_err=app_to_abs_mag(self.phot[:,m1],self.kin[:,4],app_mag_error=self.phot_err[:,m1],parallax_error=self.kin[:,5])
            mag2,mag2_err=app_to_abs_mag(self.phot[:,m2],self.kin[:,4],app_mag_error=self.phot_err[:,m2],parallax_error=self.kin[:,5])
            mag_data=mag1-mag2
            mag_err=mag1_err+mag2_err
        else:
            m1,=np.where(self.headers[0]==filter_model(model,mag))
            mag_data,mag_err=app_to_abs_mag(self.phot[:,m1],self.kin[:,4],app_mag_error=self.phot_err[:,m1],parallax_error=self.kin[:,5])
            
        plot_ages=np.array([1,3,5,10,20,30,100,200,500,1000])
        iso=load_isochrones(model)

        col_data=col_data.reshape(len(col_data))
        mag_data=mag_data.reshape(len(col_data))
        col_err=col_err.reshape(len(col_data))
        mag_err=mag_err.reshape(len(col_data))

        if type(ids)!=type(None):
            col_data=col_data[ids]
            mag_data=mag_data[ids]
            col_err=col_err[ids]
            mag_err=mag_err[ids]
            ebv1=self.ebv[ids]
        else:
            ebv1=self.ebv

        if 'stick_to_points' in kwargs:
            stick_to_points=kwargs['stick_to_points']
        else: stick_to_points=False

        if 'tofile' in kwargs:
            tofile=kwargs['tofile']
        else: tofile=False

        if 'plot_masses' in kwargs:
            plot_masses=kwargs['plot_masses']
        else: plot_masses=None

        try:
            len(col_err)
            col_err=col_err.ravel()
            mag_err=mag_err.ravel()
            po=np.arange(len(mag_data))
        except TypeError:
            po=np.array([0])
        
        x=col_data
        y=mag_data
        x_axis=col
        y_axis=mag
        ebv=ebv1
        x_error=col_err
        y_error=mag_err
        label_points=po        
        groups=None
        group_names=None

        
        isochrones=iso[3]
        iso_ages=iso[1]
        iso_filters=iso[2]
        iso_masses=iso[0]

        #changes names of Gaia_DR2 filters to EDR3
        if 'G2' in iso_filters: 
            w=where_v(['G2','Gbp2','Grp2'],iso_filters)
            iso_filters[w]=['G','Gbp','Grp']

        #axes ranges
        if 'stick_to_points' in kwargs:
            stick_to_points=kwargs['stick_to_points']
        else: stick_to_points=False
        x_range=self.axis_range(x_axis,x,stick_to_points)
        y_range=self.axis_range(y_axis,y,stick_to_points)

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

        fig, ax = plt.subplots(figsize=(16,12))

        if type(ebv)!=type(None): #subtracts extinction, if E(B-V) is provided
            x_ext=extinction(ebv,x_axis)
            y_ext=extinction(ebv,y_axis)
            x1=x-x_ext
            y1=y-y_ext
            plt.arrow(x_range[0]+0.2*(x_range[1]-x_range[0]),y_range[0]+0.1*(y_range[1]-y_range[0]),-np.median(x_ext),-np.median(y_ext),head_width=0.05, head_length=0.1, fc='k', ec='k', label='reddening')
        else:
            x1=x
            y1=y

        for i in range(len(plot_ages)):
            ii=closest(iso_ages,plot_ages[i])
            plt.plot(col_th[:,ii],mag_th[:,ii],label=str(plot_ages[i])+' Myr')

        if 'plot_masses' in kwargs:
            plot_masses=kwargs['plot_masses']
            if type(plot_masses)!=type(None):
                for i in range(len(plot_masses)):
                    im=closest(iso_masses,plot_masses[i])
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

        if n_elements(label_points)==1:
            if label_points==True:
                n=(np.linspace(0,npo-1,num=npo,dtype=int)).astype('str')
                for i, txt in enumerate(n):
                    ax.annotate(txt, (x1[i], y1[i]))
        else:
            if isinstance(label_points[0],str)==0:
                if isinstance(label_points,list): label_points=np.array(label_points,dtype=str)
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
