import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import pandas as pd
import re
import io
import os
import astropy.constants as cst
import astropy.units as units
import scipy.interpolate as interp

from collections import deque
from pathlib import Path 
import sys

#######################################
# model loading functions
#
def _read_model_BHAC2015(path, fname, instrument):
    '''
    (Private) Read the BHAC2015 models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array
    '''
    
    # read general data
    data = pd.read_csv(path / fname, sep='\s+', header=None, comment='!')

    # add ages
    data.insert(0, 'age', 0)

    # read column headers and age values
    p_cols = re.compile('!\s+(mass)\s+(Teff)')
    p_ages = re.compile('!\s+t\s+\(Gyr\)\s+=\s+([0-9]+\.[0-9]+)')
    p_vals = re.compile('\s+([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)')

    cols = ['age']
    ages = []
    cage = 0

    file = open(path / fname, 'r')
    for line in file:
        # skip non-comment lines
        if (line[0] != '!'):
            m = p_vals.match(line)
            if (m is not None):
                ages.append(cage)
            continue

        # column names
        if (len(cols) == 1):
            m = p_cols.match(line)
            if (m is not None):
                cols.extend(line[1:].split())

        # age value
        m = p_ages.match(line)
        if (m is not None):
            cage = float(m.group(1))

    file.close()

    # rename columns and add age values
    data.columns = cols    
    data.age = ages

    # unit conversion
    data.age    *= 1000
    data.mass   *= cst.M_sun / cst.M_jup
    data.radius *= cst.R_sun / cst.R_jup
    
    # reshape in final format
    masses, ages, values, dat = _reshape_data(data)

    return masses, ages, values, dat


def _read_model_PHOENIX_websim(path, fname, instrument):
    '''
    (Private) Read models from the PHOENIX web simulator

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array
    '''

    # read column headers and number of values
    p_cols = re.compile('\s+M/Ms\s*Teff.K.\s+L/Ls\s+lg\(g\)\s+R.(\w+).\s+D\s+Li\s+([A-Za-z0-9\\s_.\']+)')
    p_ages = re.compile('\s+t\s+\(Gyr\)\s+=\s+([0-9]+\.[0-9]+)')
    p_vals = re.compile('(\s+[+-]*[0-9]+\.[0-9]*){3}')

    cols  = ['age', 'mass', 'Teff', 'logL', 'logg', 'radius', 'D', 'Li']
    cage  = 0
    ages  = []
    unit  = None
    lines = []
    
    # get column names
    file = open(path / fname, 'r')
    for line in file:
        # age value
        m = p_ages.match(line)
        if (m is not None):            
            cage = float(m.group(1))
            continue
        
        # column names
        if (len(cols) == 8):
            m = p_cols.match(line)
            if (m is not None):
                unit = m.group(1)

                names = m.group(2)
                names = names.replace("'", "p")
                
                cols.extend(names.split())
                
                continue
            
        # model values
        m = p_vals.match(line)
        if (m is not None):
            lines.append(line)
            ages.append(cage)
            
    file.close()
                
    # create data frame
    lines = ''.join(lines)
    data = pd.read_csv(io.StringIO(lines), sep='\s+', header=None)

    # add ages
    data.insert(0, 'age', 0)
    data.age = ages

    # rename columns
    data.columns = cols    
    
    # unit conversion
    data.age  *= 1000
    data.mass *= cst.M_sun / cst.M_jup
    if unit == 'Gm':
        # data.radius /= cst.R_jup.to(units.Gm)
        pass
    elif unit == 'Gcm':
        #data.radius /= cst.R_jup.to(units.Gm*100)
        pass
    else:
        pass
        
    # reshape in final format
    masses, ages, values, dat = _reshape_data(data)
        
    return masses, ages, values, dat


def _read_model_sonora(path, fname, instrument):
    '''
    (Private) Read the SONORA models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array
    '''

    df = pd.read_csv(path / fname, index_col=(0, 1))

    masses  = np.sort(np.unique(df.index.get_level_values(0)))  # MJup
    ages    = np.sort(np.unique(df.index.get_level_values(1)))  # yr
    values  = df.columns

    data = np.zeros((len(masses), len(ages), len(values)))
    for iv, val in enumerate(values):
        for im, mass in enumerate(masses):
            tmp = df.loc[(mass, slice(None)), val]
            data[im, :, iv] = tmp
            
    # converts ages in Myr
    ages = ages / 1e6
    
    return masses, ages, values, data


def _read_model_bex(path, fname, instrument):
    '''
    (Private) Read the BEX models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array
    '''

    df = pd.read_csv(path / fname, index_col=(0, 1))

    masses  = np.sort(np.unique(df.index.get_level_values(0)))  # MJup
    ages    = np.sort(np.unique(df.index.get_level_values(1)))  # yr
    values  = df.columns

    data = np.zeros((len(masses), len(ages), len(values)))
    for iv, val in enumerate(values):
        for im, mass in enumerate(masses):
            tmp = df.loc[(mass, slice(None)), val]
            data[im, :, iv] = tmp
            
    # converts ages in Myr
    ages = ages / 1e6
    
    return masses, ages, values, data

    
def _read_model_atmo(path, fname, instrument):
    '''
    (Private) Read the ATMO models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array
    '''

    df = pd.read_csv(path / fname, index_col=(0, 1))

    masses  = np.sort(np.unique(df.index.get_level_values(0)))  # MJup
    ages    = np.sort(np.unique(df.index.get_level_values(1)))  # yr
    values  = df.columns

    data = np.zeros((len(masses), len(ages), len(values)))
    for iv, val in enumerate(values):
        for im, mass in enumerate(masses):
            tmp = df.loc[(mass, slice(None)), val]
            data[im, :, iv] = tmp
            
    # converts ages in Myr
    ages = ages / 1e6
    
    return masses, ages, values, data



def _read_model_MIST(path, fname, instrument, max_phase=2): #VS21
    '''
    (Private) Read the MIST models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array, interpolated along the mass axis to have consistent masses
        for all age steps.
    '''

    # read column headers and number of values
    p_cols = re.compile('\s*#*\s*EEP\s+(log10_isochrone_age_yr\s+initial_mass\s+star_mass\s+.+)')
    p_vals = re.compile('\s+[0-9]+\s+([0-9]+.+)')    
    
    # get column names
    cols  = []
    file = open(path / fname, 'r')    
    line = file.readline()
    while len(cols) == 0:
        m = p_cols.match(line)
        if (m is not None):
            names = m.group(1)
            cols.extend(names.split())
        line = file.readline()         #reads next line
    file.close()    

    #create data frame and selects only numeric rows
    data=pd.read_fwf(path / fname,header=None,comment='#',infer_nrows=50000)
    w,=np.where((isnumber(data[0],finite=True)) & (isnumber(data[1],finite=True)) & (isnumber(data[2],finite=True)) & ((data[1]!='2') & (data[2]!='3')))
    w_ch,=np.where(w[1:]-w[:-1]>1)
    w_cut=np.insert(w_ch+1,[0,len(w_ch)],[0,len(w)]) #row indices of the isochrones
    data=data.iloc[w,1:] #slicing

    #convert pandas to numpy
    data2=data.to_numpy(dtype=float)
    w_m,=np.where(np.array(cols)=='initial_mass')
    w_a,=np.where(np.array(cols)=='log10_isochrone_age_yr')    
    w_p,=np.where(np.array(cols)=='phase')    
    mass_range=[np.min(data2[:,w_m]),np.max(data2[:,w_m])]
    n_m=int(1.1*np.max(w_cut[1:]-w_cut[:-1]))
    
    #output values
    values=np.array(cols[3:]) #exclude ages, initial mass and star_mass
    ages=data2[w_cut[1:]-1,w_a]        
    masses=np.logspace(np.log10(mass_range[0]),np.log10(mass_range[1]),n_m)    
    dat=np.full((n_m, len(ages), len(values)), np.nan)

    #interpolate across the grid to fill dat
    for i in range(len(ages)):
        c=w_cut[i]
        while data2[c,w_p]<=max_phase:
            c+=1
            if c==w_cut[i+1]: break
        ma=data2[w_cut[i]:c,w_m].ravel()
        for j in range(len(values)):
            y=data2[w_cut[i]:c,j+3]
            f = interp.interp1d(ma, y, bounds_error=False, fill_value=np.nan)
            dat[:,i,j]=f(masses)
        
    masses=masses*cst.M_sun.value / cst.M_jup.value #M_sun -> M_Jup
    ages=10**(ages-6) #log10(age) -> Myr

                
    return masses, ages, values, dat

def _read_model_PARSEC(path, fname, instrument, max_phase=3): #VS21
    '''
    (Private) Read the PARSEC models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array, interpolated along the mass axis to have consistent masses
        for all age steps.
    '''

    # read column headers and number of values
    p_cols = re.compile('\s*#*\s*Zini\s+.+(logAge\s+Mini\s+.+)')
    p_vals = re.compile('\s+[0-9]+\s+([0-9]+.+)')    
    
    # get column names
    cols  = []
    file = open(path / fname, 'r')    
    line = file.readline()
    while len(cols) == 0:
        m = p_cols.match(line)
        if (m is not None):
            names = m.group(1)
            cols.extend(names.split())
        line = file.readline()         #reads next line
    file.close()    

    #create data frame
    data=pd.read_csv(path / fname,delim_whitespace=True,header=None,comment='#')
    data=data.iloc[:,2:] #slicing
    data2=data.to_numpy(dtype=float)    
    w_ch,=np.where(data2[1:,0]-data2[:-1,0]>0)
    w_cut=np.insert(w_ch+1,[0,len(w_ch)],[0,len(data2)]) #row indices of the isochrones

    #converts pandas to numpy
    w_m,=np.where(np.array(cols)=='Mini')
    w_a,=np.where(np.array(cols)=='logAge')
    w_p,=np.where(np.array(cols)=='label')    
    mass_range=[np.min(data2[:,w_m]),np.max(data2[:,w_m])]
    n_m=int(1.1*np.max(w_cut[1:]-w_cut[:-1]))

    #output values
    values=np.array(cols[2:]) #exclude ages and metallicity
    ages=data2[w_cut[1:]-1,w_a]        
    masses=np.logspace(np.log10(mass_range[0]),np.log10(mass_range[1]),n_m)    
    dat=np.full((n_m, len(ages), len(values)), np.nan)
    
    #interpolate across the grid to fill dat
    for i in range(len(ages)):
        c=w_cut[i]
        while data2[c,w_p]<=max_phase:
            c+=1
            if c==w_cut[i+1]: break
        ma=data2[w_cut[i]:c,w_m].ravel()
        for j in range(len(values)):
            y=data2[w_cut[i]:c,j+2]
            f = interp.interp1d(ma, y, bounds_error=False, fill_value=np.nan)
            dat[:,i,j]=f(masses)
        
    masses=masses*cst.M_sun.value / cst.M_jup.value #converts into M_Jup
    ages=10**(ages-6) #converts into Myr

    return masses, ages, values, dat

def _read_model_BHAC15(path, fname, instrument): #VS21
    '''
    (Private) Read the BHAC15 models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array, interpolated to have consistent masses for all ages
    '''
        # read general data
    data = pd.read_csv(path / fname, sep='\s+', header=None, comment='!')
    
    w,=np.where((isnumber(data[0],finite=True)) & (isnumber(data[1],finite=True)) & (isnumber(data[2],finite=True)))
    data=data.iloc[w,:] #slicing    
    
#    data=pd.read_csv(path / fname,delim_whitespace=True,header=None,comment='!')
    
    # add ages
    data.insert(0, 'age', 0)

    # read column headers and age values
    p_cols = re.compile('!\s+(M/Ms)\s+(Teff)')
    p_ages = re.compile('!\s+t\s+\(Gyr\)\s+=\s+([0-9]+\.[0-9]+)')
    p_vals = re.compile('\s+([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]*)')

    cols = ['age']
    ages = []
    cage = 0

    i1=0
    i2=0
    file = open(path / fname, 'r')
    for line in file:
        # skip non-comment lines
        i2+=1
        if (line[0] != '!'):
            m = p_vals.match(line)
            if (m is not None):
                ages.append(cage)
                i1+=1
            continue

        # column names
        if (len(cols) == 1):
            m = p_cols.match(line)
            if (m is not None):
                cols.extend(line[1:].split())

        # age value
        m = p_ages.match(line)
        if (m is not None):
            cage = float(m.group(1))

    file.close()

    # rename columns and add age values
    c=0
    while c<len(cols):
        if cols[c]=='M/Ms':
            cols[c]=cols[c].replace('M/Ms','mass')
            break
        c+=1
    c=0
    while c<len(cols):
        if cols[c]=='R/Rs':
            cols[c]=cols[c].replace('R/Rs','radius')
            break
        c+=1
    data.columns = cols    
    data.age = ages

    # unit conversion
    data.age    *= 1000
    data.mass   *= cst.M_sun / cst.M_jup
    data.radius *= cst.R_sun / cst.R_jup
    
    # reshape in final format
    masses, ages, values, dat = _reshape_data(data)

    return masses, ages, values, dat

def _read_model_starevol(path, fname, instrument): #VS21
    '''
    (Private) Read the starevol models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array, interpolated along the mass axis to have consistent masses
        for all age steps.
    '''
    #each file is an age

    c=len(fname)-1
    while (fname[c:c+2]!='_t') & (c>0): c-=1
    files=os.listdir(path)
    iso_list=[] #list of all the isochrones with given rotation
    for i in range(len(files)):
        if fname[:c+1] in files[i]: iso_list.append(files[i])
    
    # read general data for a given age
    data0 = pd.read_csv(path / iso_list[0], sep='\s+', comment='*',header=0)
    cols = ['#M_ini','logTeff','logL','logLgrav','M','R','logg','rho_phot','Mbol','BC','M_U','M_B','M_V','M_R','M_I','M_H','M_J','M_K','M_G','M_Gbp','M_Grp']
    data = data0[cols]

    data2=data.to_numpy(dtype=float)
    w,=np.where((data2[1:,0]-data2[:-1,0]>0)) #to avoid repeated mass entries
    w=np.insert(w+1,0,0)
    mass_range=[np.min(data['#M_ini']),np.max(data['#M_ini'])]
    data2=data2[w,:]

    values=np.array(cols[1:]) #exclude masses
    
            
    
    #riprendere da qua: fare un ciclo con tutti i file
    #output values
    ages=[]
    n_m=int(1.1*len(w))
    masses=np.logspace(np.log10(mass_range[0]),np.log10(mass_range[1]),n_m)    
    dat=np.full((n_m, len(iso_list), len(values)), np.nan)
    
    
    #interpolates across the grid to fill dat
    for i in range(len(iso_list)):
        c=len(iso_list[i])-1
        while (iso_list[i][c:c+2]!='_t') & (c>0): c-=1
        age=iso_list[i][c+2:]
        c=len(age)-1
        while (age[c]!='.') & (c>0): c-=1
        ages.append(float(age[0:c]))
        if i>0:
            data0 = pd.read_csv(path / iso_list[i], sep='\s+', comment='*',header=0)
            data = data0[cols]
            data2=data.to_numpy(dtype=float)
            w,=np.where((data2[1:,0]-data2[:-1,0]>0)) #to avoid repeated mass entries
            w=np.insert(w+1,0,0)
            data2=data2[w,:]            
        ma=data2[:,0].reshape(len(data2))
        for j in range(len(values)):
            y=data2[:,j+1]
            f = interp.interp1d(ma, y, bounds_error=False, fill_value=np.nan)
            dat[:,i,j]=f(masses)
        
    masses=masses*cst.M_sun.value / cst.M_jup.value #converts into M_Jup
    ages=10**(np.array(ages)-6) #converts into Myr


    return masses, ages, values, dat

def _read_model_atmo2020(path, fname, instrument): #VS21
    '''
    (Private) Read the atmo2020 models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array
    '''
    #each file is an age

    c=len(fname)-1
    while (fname[c:c+2]!='_t') & (c>0): c-=1
    track_list=os.listdir(path)

    # read column headers and number of values
    p_cols = re.compile('\s*#*\s*(Mass\s+Age\s+.+)')
    p_vals = re.compile('\s+[0-9]+\s+([0-9]+.+)')    
    
    # get column names
    cols  = []
    file = open(path / fname, 'r')    
    line = file.readline()
    while len(cols) == 0:
        m = p_cols.match(line)
        if (m is not None):
            names = m.group(1)
            cols.extend(names.split())
        line = file.readline()         #reads next line
    file.close()    
    
    all_data=[]
    
    for i in range(len(track_list)):      
        data=pd.read_fwf(path / track_list[i],header=None,comment='#',infer_nrows=10000)
        w,=np.where((isnumber(data[0],finite=True)) & (isnumber(data[1],finite=True)) & (isnumber(data[2],finite=True)) & ((data[1]!='2') & (data[2]!='3')))
        data=data.iloc[w,:]
        all_data.append(data.replace(0, np.nan)) #0 is used as missing value in these files but we want nan
                
    data=pd.concat(all_data)
    cols[0]='mass'
    cols[1]='age'
    data.columns=cols
    data.mass = data.mass.astype(float) * cst.M_sun / cst.M_jup
    data.age = data.age.astype(float) * 1000
    masses, ages, values, dat = _reshape_data(data)
              
    return masses, ages, values, dat

def _read_model_SPOTS(path, fname, instrument): #VS21
    '''
    (Private) Read the SPOTS models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array, interpolated to have consistent masses for all ages
    '''

    # read column headers and number of values
    p_cols = re.compile('\s*#*\s*logAge\s+Mass\s+Fspot\s+Xspot\s+(log\s*.+)')
    p_ages = re.compile('\s*#+\s+log10 Age\(yr\)\s+=\s+([0-9]+\.[0-9]+)')
    p_vals = re.compile('\s+[0-9]+.*([0-9]+.+)')        
    
    cols  = ['age','mass','Fspot','Xspot']
    lines = []
    
    # get column names
    file = open(path / fname, 'r')
    for line in file:
        # age value
        m = p_ages.match(line)
        if (m is not None):            
            continue
        
        # column names
        if (len(cols) == 4):
            m = p_cols.match(line)
            if (m is not None):
                names = m.group(1)     
                cols.extend(names.split())                
                continue
            
        # model values
        m = p_vals.match(line)
        if (m is not None):
            lines.append(line)
            
    file.close()
    
    # create data frame
    lines = ''.join(lines)
    data = pd.read_csv(io.StringIO(lines), sep='\s+', header=None)
    data=data.replace(-99,np.nan)

    c=0
    while c<len(cols):
        if cols[c]=='log(Teff)':
            cols[c]=cols[c].replace('log(Teff)','Teff')
            break
        c+=1

    # rename columns
    data.columns = cols    
    data=df_column_switch(data,'Fspot','Teff')
    
    # unit conversion
    data.age  = 10**(data.age-6)
    data.mass *= cst.M_sun / cst.M_jup

    # reshape in final format
    masses, ages, values, dat = _reshape_data(data)
        
    return masses, ages, values, dat

def _read_model_Dartmouth(path, fname, instrument): #VS21
    '''
    (Private) Read the Dartmouth models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array, interpolated to have consistent masses for all ages
    '''

    # read column headers and number of values
    p_cols = re.compile('\s*#*\s*Mass\s+(log\(Teff\)\s+log\(g\)\s+log\(L\).+)')
    p_vals = re.compile('\s+[0-9]+\s+([0-9]+.+)')
    p_age = re.compile('\s*#*\s*Age\s+=\s+([0-9]+.+)+(\s*.yr\s*)+(\[Fe/H\]+.+)')

    
    # get column names
    cols  = ['age','mass']
    file = open(path / fname, 'r')
    line = file.readline()
    while len(cols) == 2:
        m = p_cols.match(line)
        if (m is not None):
            names = m.group(1)
            cols.extend(names.split())
        line = file.readline()         #reads next line
    file.close()    

    c2=len(fname)-1
    while (fname[c2:c2+3]!='myr') & (c2>0): c2-=1
    files=os.listdir(path)
    iso_list=[] #list of all the isochrones with given Fe/H and alpha
    for i in range(len(files)):
        if fname[c2:] in files[i]: iso_list.append(files[i])
    
    all_data=[] #stores all data    
    
    for i in range(len(iso_list)):
        #recovers age
        file = open(path / iso_list[i], 'r')
        line = file.readline()
        found = 0
        while found==0:
            a = p_age.match(line)
            if (a is not None):
                age = float(a.group(1))
                unit = (a.group(2)).strip()
                if unit=='Myr': pass
                elif unit=='Gyr': age*=1000
                elif unit=='yr': age*=10**-6
                found=1
            line = file.readline()
        file.close()    
        data=pd.read_fwf(path / iso_list[i],header=None,comment='#',infer_nrows=10000)
        w,=np.where((isnumber(data[0],finite=True)) & (isnumber(data[1],finite=True)) & (isnumber(data[2],finite=True)))
        data=data.iloc[w,:] #slicing
        data.insert(0, 'age', 0)
        data.age=np.full(len(w),age)
        all_data.append(data)
        
    data=pd.concat(all_data)
    data.columns=cols
    data.mass = data.mass.astype(float) * cst.M_sun / cst.M_jup
    masses, ages, values, dat = _reshape_data(data)
        
    return masses, ages, values, dat    

def _read_model_Ekstroem(path, fname, instrument, v_vcrit=0.0):
    '''
    (Private) Read the Dartmouth models

    Parameters
    ----------
    path : str
        Full path to the directory containing the model files

    fname : str
        Full model file name

    instrument : str
        Name of the instrument (or observatory) for the file

    v_vcrit : rotational velocity. Available: [0.0,0.4]. Default: 0.0.
    
    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array, interpolated to have consistent masses for all ages
    '''

    if v_vcrit!=0: sel='r'
    else: sel='n'
        
    p_cols1 = re.compile('\s*(logt\s+R\s+Mini\s+.+)')
    p_cols2 = re.compile('\s*(Mini\s+.+)')
    
    # get column names
    cols  = []
    file = open(path / fname, 'r')    
    line = file.readline()
    while len(cols) == 0:
        m = p_cols1.match(line)
        if (m is not None):
            names = m.group(1)
            cols.extend(names.split())
            feh=0
        m = p_cols2.match(line)
        if (m is not None):
            names = m.group(1)
            cols.extend(names.split())
            feh=1
        line = file.readline()         #reads next line
    file.close()

    if feh==0: mass_col = 2
    else: mass_col = 0
    
    # create data frame
    data=pd.read_fwf(path / fname,header=0,comment='#',infer_nrows=50000)
    w,=np.where((np.array(data.iloc[1:,mass_col])-np.array(data.iloc[:-1,mass_col])>0) & (data.iloc[:-1,1]==sel)) #to avoid repeated mass entries
    data=data.iloc[w,:]
    
    if data.columns[0]=='logt':
        data.logt = 10**(data.logt.astype(float)-6)
        cols[0]='age'
        data.insert(len(data.columns), 'Bmag', 0)
        data.Bmag = data['B-V'].astype(float)+data['Vmag'].astype(float)
        data.insert(len(data.columns), 'Umag', 0)
        data.Umag = data['U-B'].astype(float)+data['Bmag'].astype(float)
        cols.extend(['Bmag','Umag'])
    cols[2]='mass'
        
    data.columns=cols
    data.mass = data.mass.astype(float) * cst.M_sun / cst.M_jup
       
    # reshape in final format
    masses, ages, values, dat = _reshape_data(data,start_col=5)
        
    return masses, ages, values, dat


def _reshape_data(dataframe,start_col=2):
    '''
    Reshape the data frame in a regular grid that can be used as input in scipy functions.

    Parameters
    ----------
    dataframe : pandas DataFrame
        Data frame with all the data

    Returns
    -------
    masses : vector
        Numpy vector with unique masses, in MJup

    ages : vector
        Numpy vector with unique ages, in Myr

    values : array
        Array with names of parameters

    data : array
        Numpy data array
    '''

    # unique ages and masses
    masses = dataframe.mass.unique()
    ages = dataframe.age.unique()

    # values
    values = dataframe.columns.values[start_col:]
    col = dataframe.columns[start_col]

    # fill array
    data = np.full((masses.size, ages.size, values.size), np.nan)
    for m, mass in enumerate(masses):
        for a, age in enumerate(ages):
            mask = (dataframe.mass == mass) & (dataframe.age == age)
            
            if mask.any():
                data[m, a, :] = dataframe.loc[mask, col:].values.squeeze()

    return masses, ages, values, data


#######################################
# utility functions
#

def n_elements(x): #VS21
    size = 1
    for dim in np.shape(x): size *= dim
    return size

def isnumber(s,finite=False): #VS21
    '''
    Checks if a given string, or each element of a list of strings, is a
    valid number
    
    If finite=True, values like 'nan' or 'inf' will return 'False'.
    By default, they return 'True'.

    Parameters
    ----------
    s : string, list or array

    Returns
    -------
    res : boolean value or array, specifying if (any of) the (i-th) element(s)
    is a valid number or not.
    '''
    n=n_elements(s)
    if n==0: return False
    elif n==1:
        try:
            s=float(s)
            if finite==True:
                if np.isfinite(s): return True
                else: return False
            else: return True
        except ValueError:
            return False
    else:
        res=np.zeros(n,dtype=bool)
        for i in range(n):            
            try:
                x=float(s[i])
                if finite==True:
                    if np.isfinite(x): res[i]=1
                else: res[i]=1
            except ValueError:
                continue
        return res

def df_column_switch(df, column1, column2): #VS21
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

def _monotonic_sublists(lst):
    '''
    (Private) Extract monotonic sublists from a list of values
    
    Given a list of values that is not sorted (such that for some valid
    indices i,j, i<j, sometimes lst[i] > lst[j]), produce a new
    list-of-lists, such that in the new list, each sublist *is*
    sorted: for all sublist \elem returnval: assert_is_sorted(sublist)
    and furthermore this is the minimal set of sublists required to
    achieve the condition.

    Thus, if the input list lst is actually sorted, this returns
    [list(lst)].

    Parameters
    ----------
    lst : list or array
        List of values

    Returns
    -------
    ret_i : list
        List of indices of monotonic sublists
    
    ret_v : list
        List of values of monotonic sublists
    '''

    # Make a copy of lst before modifying it; use a deque so that
    # we can pull entries off it cheaply.
    idx = deque(range(len(lst)))
    deq = deque(lst)
    ret_i = []
    ret_v = []
    while deq:
        sub_i = [idx.popleft()]
        sub_v = [deq.popleft()]

        if len(deq) > 1:
            if deq[0] <= sub_v[-1]:
                while deq and deq[0] <= sub_v[-1]:
                    sub_i.append(idx.popleft())
                    sub_v.append(deq.popleft())
            else:
                while deq and deq[0] >= sub_v[-1]:
                    sub_i.append(idx.popleft())
                    sub_v.append(deq.popleft())
                    
        ret_i.append(sub_i)
        ret_v.append(sub_v)
        
    return ret_i, ret_v


def _interpolate_model(masses, ages, values, data, age, filt, param, Mabs, fill):
    '''
    (Private) Interpolate model grid

    Parameters
    ----------
    masses : array
        Mass values in the model grid

    ages : array
        Age values in the model grid
    
    values : array
        Name of filters (and other parameters) in the model grid

    data : array
        Data of the model grid

    age : float
        Age at which interpolation is needed

    filt : str
        Filter in which the magnitude is provided

    param : str
        Parameter to be interpolated

    Mabs : array
        Absolute magnitude

    fill : bool
        Fill interpolated values with min/max values in the models when
        trying to interpolate outside the values in the models
    
    Returns
    -------
    values : array
        Interpolated values
    '''

    # age indices
    ii = np.abs(ages-age).argmin()
    if age <= ages[ii]:
        imin = ii-1
        imax = ii
    elif age > ages[ii]:
        imin = ii
        imax = ii+1

    agemin = ages[imin]
    agemax = ages[imax]
        
    # parameter value
    if param == 'Mass':
        ifilt = np.where(values == filt)[0]

        Zmin = data[:, imin, ifilt].squeeze()
        Zmax = data[:, imax, ifilt].squeeze()

        Znew = (Zmin - Zmax) / (agemin - agemax) * (age - agemin) + Zmin

        # remove missing values
        masses = masses[np.isfinite(Znew)]
        Znew   = Znew[np.isfinite(Znew)]
        
        # find monotonic parts of the signal
        mono_i, mono_v = _monotonic_sublists(Znew)
        
        nsub = len(mono_i)
        sub_idx = np.zeros((2*nsub-1, 2), dtype=np.int)
        for s in range(nsub):
            sub_idx[s, 0] = mono_i[s][0]
            sub_idx[s, 1] = mono_i[s][-1]
        for s in range(nsub-1):
            sub_idx[s+nsub, 0] = mono_i[s][-1]
            sub_idx[s+nsub, 1] = mono_i[s+1][0]

        sub_idx = np.sort(sub_idx, axis=0)

        # interpolate over each part
        values = np.zeros((2*nsub-1, Mabs.size))
        for i, s in enumerate(sub_idx):
            sub_Znew   = Znew[s[0]:s[1]+1]
            sub_masses = masses[s[0]:s[1]+1]

            if len(sub_Znew) < 2:
                continue
            
            interp_func = interp.interp1d(sub_Znew, sub_masses, bounds_error=False, fill_value=np.nan)
            values[i] = interp_func(Mabs)

            # fill if outside of available values
            if fill:
                values[i, Mabs < sub_Znew.min()] = masses.max()
                values[i, Mabs > sub_Znew.max()] = masses.min()
        
        # combine
        values = np.nanmax(values, axis=0)
    else:
        raise ValueError('Interpolation for parameter {0} is not implemented yet.'.format(param))

    return values


def _read_model_data(paths, models, instrument, model):
    '''
    Return the data from a model and instrument

    Parameters
    ----------
    paths : list
        List of paths where to find the models

    models : dict
        Dictionary containing all the models information and data

    instrument : str
        Instrument name

    model : str
        Model name

    Returns
    -------
    path : str
        The complete path to the model file
    '''

    # lower case
    model = model.lower()
    instrument = instrument.lower()
    
    # model key
    key = instrument.lower()+'_'+model.lower()

    # find proper model
    data = None
    for mod in models['properties']:
        if (mod['name'] == model) and (mod['instrument'] == instrument):
            fname = mod['file']
            
            # search for path
            found = False
            for path in paths:
                if (path / fname).exists():
                    mod['path'] = path
                    found = True
                    break

            if not found:
                raise ValueError('File {0} for model {1} and instrument {2} does not exists. Are you sure it is in your search path?'.format(path, model, instrument))
            
            # get data in format (masses, ages, values, data)
            data = mod['function'](path, fname, instrument)

    # not found
    if data is None:
        raise ValueError('Could not find model {0} for instrument {1}'.format(model, instrument))

    # save data
    models['data'][key] = data

    
#######################################
# models definitions
#
search_path = [(Path(__file__) / '../../data/evolution/').resolve()]
models = {
    'properties': [
        {'instrument': 'nicmos',    'name': 'dusty2000',             'file': 'model.AMES-dusty-2000.M-0.0.HST',           'function': _read_model_PHOENIX_websim},
        {'instrument': 'naco',      'name': 'dusty2000',             'file': 'model.AMES-dusty-2000.M-0.0.NaCo',          'function': _read_model_PHOENIX_websim},
        {'instrument': 'irdis',     'name': 'dusty2000',             'file': 'model.AMES-dusty-2000.M-0.0.SPHERE.Vega',   'function': _read_model_PHOENIX_websim},
        {'instrument': 'nicmos',    'name': 'cond2003',              'file': 'model.AMES-Cond-2003.M-0.0.HST',            'function': _read_model_PHOENIX_websim},
        {'instrument': 'naco',      'name': 'cond2003',              'file': 'model.AMES-Cond-2003.M-0.0.NaCo',           'function': _read_model_PHOENIX_websim},
        {'instrument': 'irdis',     'name': 'cond2003',              'file': 'model.AMES-Cond-2003.M-0.0.SPHERE.Vega',    'function': _read_model_PHOENIX_websim},    
        {'instrument': 'irdis',     'name': 'bhac2015+dusty2000',    'file': 'BHAC15_DUSTY00_iso_t10_10.SPHERE.txt',      'function': _read_model_BHAC2015},
        {'instrument': 'irdis',     'name': 'bhac2015+cond2003',     'file': 'BHAC15_COND03_iso_t10_10.SPHERE.txt',       'function': _read_model_BHAC2015},
        
        {'instrument': 'mko',       'name': 'sonora',                'file': 'sonora_mko.csv.gz',                         'function': _read_model_sonora},
        {'instrument': '2mass',     'name': 'sonora',                'file': 'sonora_2mass.csv.gz',                       'function': _read_model_sonora},
        {'instrument': 'keck',      'name': 'sonora',                'file': 'sonora_keck.csv.gz',                        'function': _read_model_sonora},
        {'instrument': 'sdss',      'name': 'sonora',                'file': 'sonora_sdss.csv.gz',                        'function': _read_model_sonora},
        {'instrument': 'irac',      'name': 'sonora',                'file': 'sonora_irac.csv.gz',                        'function': _read_model_sonora},
        {'instrument': 'wise',      'name': 'sonora',                'file': 'sonora_wise.csv.gz',                        'function': _read_model_sonora},        
        
        {'instrument': 'irdis',     'name': 'bex_cond_coldest',      'file': 'bex_ames-cond_coldest.csv.gz',              'function': _read_model_bex},
        {'instrument': 'irdis',     'name': 'bex_cond_warm',         'file': 'bex_ames-cond_warm.csv.gz',                 'function': _read_model_bex},
        {'instrument': 'irdis',     'name': 'bex_cond_hot',          'file': 'bex_ames-cond_hot.csv.gz',                  'function': _read_model_bex},
        {'instrument': 'irdis',     'name': 'bex_cond_hottest',      'file': 'bex_ames-cond_hottest.csv.gz',              'function': _read_model_bex},
        {'instrument': 'irdis',     'name': 'bex_dusty_coldest',     'file': 'bex_ames-dusty_coldest.csv.gz',             'function': _read_model_bex},
        {'instrument': 'irdis',     'name': 'bex_dusty_warm',        'file': 'bex_ames-dusty_warm.csv.gz',                'function': _read_model_bex},
        {'instrument': 'irdis',     'name': 'bex_dusty_hot',         'file': 'bex_ames-dusty_hot.csv.gz',                 'function': _read_model_bex},
        {'instrument': 'irdis',     'name': 'bex_dusty_hottest',     'file': 'bex_ames-dusty_hottest.csv.gz',             'function': _read_model_bex},

        {'instrument': 'mko',       'name': 'atmo_ceq',              'file': 'ATMO_CEQ_MKO.csv.gz',                       'function': _read_model_atmo},
        {'instrument': 'mko',       'name': 'atmo_neq_strong',       'file': 'ATMO_NEQ_strong_MKO.csv.gz',                'function': _read_model_atmo},
        {'instrument': 'mko',       'name': 'atmo_neq_weak',         'file': 'ATMO_NEQ_weak_MKO.csv.gz',                  'function': _read_model_atmo},
        {'instrument': 'irac',      'name': 'atmo_ceq',              'file': 'ATMO_CEQ_MKO.csv.gz',                       'function': _read_model_atmo},
        {'instrument': 'irac',      'name': 'atmo_neq_strong',       'file': 'ATMO_NEQ_strong_MKO.csv.gz',                'function': _read_model_atmo},
        {'instrument': 'irac',      'name': 'atmo_neq_weak',         'file': 'ATMO_NEQ_weak_MKO.csv.gz',                  'function': _read_model_atmo},
        {'instrument': 'wise',      'name': 'atmo_ceq',              'file': 'ATMO_CEQ_MKO.csv.gz',                       'function': _read_model_atmo},
        {'instrument': 'wise',      'name': 'atmo_neq_strong',       'file': 'ATMO_NEQ_strong_MKO.csv.gz',                'function': _read_model_atmo},
        {'instrument': 'wise',      'name': 'atmo_neq_weak',         'file': 'ATMO_NEQ_weak_MKO.csv.gz',                  'function': _read_model_atmo},

        #added by VS21
        {'instrument': 'gaia',      'name': 'ames_cond',                  'file': 'model.AMES-Cond-2000.M-0.0.GAIA.Vega.txt',  'function': _read_model_PHOENIX_websim},
        {'instrument': 'gaia',      'name': 'ames_dusty',                 'file': 'model.AMES-dusty.M-0.0.GAIA.Vega.txt',      'function': _read_model_PHOENIX_websim},
        {'instrument': 'gaia',      'name': 'bt_nextgen',            'file': 'model.BT-NextGen.M-0.0.GAIA.Vega.txt',      'function': _read_model_PHOENIX_websim},
        {'instrument': 'gaia',      'name': 'bt_settl',              'file': 'model.BT-Settl.M-0.0.GAIA.Vega.txt',        'function': _read_model_PHOENIX_websim},
        {'instrument': 'gaia',      'name': 'nextgen',               'file': 'model.NextGen.M-0.0.GAIA.Vega.txt',         'function': _read_model_PHOENIX_websim},

        {'instrument': '2mass',     'name': 'bt_settl',              'file': 'model.BT-Settl.M-0.0.2MASS.Vega.txt',       'function': _read_model_PHOENIX_websim},
        {'instrument': '2mass',     'name': 'ames_cond',                  'file': 'model.AMES-Cond-2000.M-0.0.2MASS.Vega.txt', 'function': _read_model_PHOENIX_websim},
        {'instrument': '2mass',     'name': 'bt_nextgen',            'file': 'model.BT-NextGen.M-0.0.2MASS.Vega.txt',     'function': _read_model_PHOENIX_websim},
        {'instrument': '2mass',     'name': 'ames_dusty',                 'file': 'model.AMES-dusty.M-0.0.2MASS.Vega.txt',     'function': _read_model_PHOENIX_websim},
        {'instrument': '2mass',     'name': 'nextgen',               'file': 'model.NextGen.M-0.0.2MASS.Vega.txt',        'function': _read_model_PHOENIX_websim},

        {'instrument': 'panstarrs', 'name': 'bt_settl',              'file': 'model.BT-Settl.M-0.0.PANSTARRS.Vega.txt',   'function': _read_model_PHOENIX_websim},
        {'instrument': 'panstarrs', 'name': 'ames_cond',                  'file': 'model.AMES-Cond-2000.M-0.0.PS1.Vega.txt',   'function': _read_model_PHOENIX_websim},
        {'instrument': 'panstarrs', 'name': 'bt_nextgen',            'file': 'model.BT-NextGen.M-0.0.PS1.Vega.txt',       'function': _read_model_PHOENIX_websim},
        {'instrument': 'panstarrs', 'name': 'ames_dusty',                 'file': 'model.AMES-dusty.M-0.0.PS1.Vega.txt',       'function': _read_model_PHOENIX_websim},
        {'instrument': 'panstarrs', 'name': 'nextgen',               'file': 'model.NextGen.M-0.0.PS1.Vega.txt',          'function': _read_model_PHOENIX_websim},

        {'instrument': 'wise',      'name': 'bt_settl',              'file': 'model.BT-Settl.M-0.0.WISE.Vega.txt',        'function': _read_model_PHOENIX_websim},
        {'instrument': 'wise',      'name': 'ames_cond',                  'file': 'model.AMES-Cond-2000.M-0.0.WISE.Vega.txt',  'function': _read_model_PHOENIX_websim},
        {'instrument': 'wise',      'name': 'bt_nextgen',            'file': 'model.BT-NextGen.M-0.0.WISE.Vega.txt',      'function': _read_model_PHOENIX_websim},
        {'instrument': 'wise',      'name': 'ames_dusty',                 'file': 'model.AMES-dusty.M-0.0.WISE.Vega.txt',      'function': _read_model_PHOENIX_websim},
        {'instrument': 'wise',      'name': 'nextgen',               'file': 'model.NextGen.M-0.0.WISE.Vega.txt',         'function': _read_model_PHOENIX_websim},

        {'instrument': 'sphere',    'name': 'bt_settl',              'file': 'model.BT-Settl.M-0.0.SPHERE.Vega.txt',      'function': _read_model_PHOENIX_websim},
        {'instrument': 'sphere',    'name': 'bt_nextgen',            'file': 'model.BT-NextGen.M-0.0.SPHERE.Vega.txt',    'function': _read_model_PHOENIX_websim},
        {'instrument': 'sphere',    'name': 'ames_dusty',                 'file': 'model.AMES-dusty.M-0.0.SPHERE.Vega.txt',    'function': _read_model_PHOENIX_websim},
        {'instrument': 'sphere',    'name': 'nextgen',               'file': 'model.NextGen.M-0.0.SPHERE.Vega.txt',       'function': _read_model_PHOENIX_websim},

        {'instrument': 'sloan',     'name': 'bt_settl',              'file': 'model.BT-Settl.M-0.0.SLOAN.Vega.txt',       'function': _read_model_PHOENIX_websim},
        {'instrument': 'johnson',   'name': 'bt_settl',              'file': 'model.BT-Settl.M-0.0.JOHNSON.Vega.txt',     'function': _read_model_PHOENIX_websim},
        
        {'instrument': 'bessell',   'name': 'mist_m0.25_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m0.25_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m0.25_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m0.25_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m0.25_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m0.25_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_m0.25_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m0.25_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m0.25_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m0.25_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m0.25_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m0.25_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_m0.50_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m0.50_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m0.50_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m0.50_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m0.50_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m0.50_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_m0.50_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m0.50_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m0.50_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m0.50_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m0.50_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m0.50_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_m0.75_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m0.75_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m0.75_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m0.75_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m0.75_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m0.75_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_m0.75_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m0.75_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m0.75_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m0.75_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m0.75_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m0.75_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m0.75_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_m1.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m1.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m1.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m1.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m1.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m1.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_m1.00_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_m1.00_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_m1.00_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_m1.00_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_m1.00_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_m1.00_p0.0_p0.4',  'file': 'MIST_v1.2_feh_m1.00_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.00_p0.0_p0.0',  'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.25_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.25_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.25_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.25_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.25_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.25_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.25_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.25_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.25_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.25_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.25_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.25_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.50_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.50_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.50_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.50_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.50_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.50_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.50_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.50_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.50_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.50_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.50_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.50_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.75_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.75_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.75_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.75_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.75_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.75_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p0.75_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p0.75_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p0.75_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p0.75_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p0.75_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p0.75_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p0.75_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p1.00_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p1.00_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p1.00_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p1.00_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p1.00_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p1.00_p0.0_p0.0',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.0_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': 'bessell',   'name': 'mist_p1.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'gaia',      'name': 'mist_p1.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': '2mass',     'name': 'mist_p1.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'hipparcos', 'name': 'mist_p1.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'tycho',     'name': 'mist_p1.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd.txt',  'function': _read_model_MIST},
        {'instrument': 'wise',      'name': 'mist_p1.00_p0.0_p0.4',   'file': 'MIST_v1.2_feh_p1.00_afe_p0.0_vvcrit0.4_WISE.iso.cmd.txt',       'function': _read_model_MIST},

        {'instrument': '2mass',     'name': 'parsec_p0.00',           'file': '2MASS_WISE_feh_p0.00.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_p0.00',           'file': '2MASS_WISE_feh_p0.00.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_p0.00',           'file': 'GAIA_EDR3_feh_p0.00.txt',                                       'function': _read_model_PARSEC},
        {'instrument': 'bessell',   'name': 'parsec_p0.00',           'file': 'Bessell_feh_p0.00.txt',                                         'function': _read_model_PARSEC},
        {'instrument': 'panstarrs', 'name': 'parsec_p0.00',           'file': 'PANSTARRS_feh_p0.00.txt',                                       'function': _read_model_PARSEC},
        {'instrument': 'skymapper', 'name': 'parsec_p0.00',           'file': 'SkyMapper_feh_p0.00.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_p0.25',           'file': '2MASS_WISE_feh_p0.25.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_p0.25',           'file': '2MASS_WISE_feh_p0.25.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_p0.25',           'file': 'GAIA_EDR3_feh_p0.25.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_p0.50',           'file': '2MASS_WISE_feh_p0.50.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_p0.50',           'file': '2MASS_WISE_feh_p0.50.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_p0.50',           'file': 'GAIA_EDR3_feh_p0.50.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_p0.75',           'file': '2MASS_WISE_feh_p0.75.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_p0.75',           'file': '2MASS_WISE_feh_p0.75.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_p0.75',           'file': 'GAIA_EDR3_feh_p0.75.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_p1.00',           'file': '2MASS_WISE_feh_p1.00.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_p1.00',           'file': '2MASS_WISE_feh_p1.00.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_p1.00',           'file': 'GAIA_EDR3_feh_p1.00.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_m0.25',           'file': '2MASS_WISE_feh_m0.25.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_m0.25',           'file': '2MASS_WISE_feh_m0.25.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_m0.25',           'file': 'GAIA_EDR3_feh_m0.25.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_m0.50',           'file': '2MASS_WISE_feh_m0.50.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_m0.50',           'file': '2MASS_WISE_feh_m0.50.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_m0.50',           'file': 'GAIA_EDR3_feh_m0.50.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_m0.75',           'file': '2MASS_WISE_feh_m0.75.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_m0.75',           'file': '2MASS_WISE_feh_m0.75.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_m0.75',           'file': 'GAIA_EDR3_feh_m0.75.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'parsec_m1.00',           'file': '2MASS_WISE_feh_m1.00.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'wise',      'name': 'parsec_m1.00',           'file': '2MASS_WISE_feh_m1.00.txt',                                      'function': _read_model_PARSEC},
        {'instrument': 'gaia',      'name': 'parsec_m1.00',           'file': 'GAIA_EDR3_feh_m1.00.txt',                                       'function': _read_model_PARSEC},

        {'instrument': '2mass',     'name': 'bhac15',                 'file': 'BHAC15_iso.2mass.txt',                                          'function': _read_model_BHAC15},
        {'instrument': 'gaia',      'name': 'bhac15',                 'file': 'BHAC15_iso.GAIA.txt',                                           'function': _read_model_BHAC15},
        {'instrument': 'sphere',    'name': 'bhac15',                 'file': 'BHAC15_iso.SPHERE.txt',                                         'function': _read_model_BHAC15},
        {'instrument': 'panstarrs', 'name': 'bhac15',                 'file': 'BHAC15_iso.panstar.txt',                                        'function': _read_model_BHAC15},

        {'instrument': 'gaia',      'name': 'starevol_m0.83_p0.0',       'file': 'Isochr_Z0.0020_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.83_p0.0',       'file': 'Isochr_Z0.0020_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.83_p0.0',       'file': 'Isochr_Z0.0020_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.83_p0.2',       'file': 'Isochr_Z0.0020_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.83_p0.2',       'file': 'Isochr_Z0.0020_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.83_p0.2',       'file': 'Isochr_Z0.0020_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.83_p0.4',       'file': 'Isochr_Z0.0020_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.83_p0.4',       'file': 'Isochr_Z0.0020_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.83_p0.4',       'file': 'Isochr_Z0.0020_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.83_p0.6',       'file': 'Isochr_Z0.0020_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.83_p0.6',       'file': 'Isochr_Z0.0020_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.83_p0.6',       'file': 'Isochr_Z0.0020_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},

        {'instrument': 'gaia',      'name': 'starevol_m0.35_p0.0',       'file': 'Isochr_Z0.0060_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.35_p0.0',       'file': 'Isochr_Z0.0060_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.35_p0.0',       'file': 'Isochr_Z0.0060_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.35_p0.2',       'file': 'Isochr_Z0.0060_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.35_p0.2',       'file': 'Isochr_Z0.0060_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.35_p0.2',       'file': 'Isochr_Z0.0060_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.35_p0.4',       'file': 'Isochr_Z0.0060_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.35_p0.4',       'file': 'Isochr_Z0.0060_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.35_p0.4',       'file': 'Isochr_Z0.0060_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.35_p0.6',       'file': 'Isochr_Z0.0060_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.35_p0.6',       'file': 'Isochr_Z0.0060_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.35_p0.6',       'file': 'Isochr_Z0.0060_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},

        {'instrument': 'gaia',      'name': 'starevol_m0.22_p0.0',       'file': 'Isochr_Z0.0080_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.22_p0.0',       'file': 'Isochr_Z0.0080_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.22_p0.0',       'file': 'Isochr_Z0.0080_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.22_p0.2',       'file': 'Isochr_Z0.0080_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.22_p0.2',       'file': 'Isochr_Z0.0080_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.22_p0.2',       'file': 'Isochr_Z0.0080_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.22_p0.4',       'file': 'Isochr_Z0.0080_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.22_p0.4',       'file': 'Isochr_Z0.0080_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.22_p0.4',       'file': 'Isochr_Z0.0080_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.22_p0.6',       'file': 'Isochr_Z0.0080_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.22_p0.6',       'file': 'Isochr_Z0.0080_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.22_p0.6',       'file': 'Isochr_Z0.0080_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},

        {'instrument': 'gaia',      'name': 'starevol_m0.13_p0.0',       'file': 'Isochr_Z0.0100_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.13_p0.0',       'file': 'Isochr_Z0.0100_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.13_p0.0',       'file': 'Isochr_Z0.0100_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.13_p0.2',       'file': 'Isochr_Z0.0100_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.13_p0.2',       'file': 'Isochr_Z0.0100_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.13_p0.2',       'file': 'Isochr_Z0.0100_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.13_p0.4',       'file': 'Isochr_Z0.0100_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.13_p0.4',       'file': 'Isochr_Z0.0100_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.13_p0.4',       'file': 'Isochr_Z0.0100_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.13_p0.6',       'file': 'Isochr_Z0.0100_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.13_p0.6',       'file': 'Isochr_Z0.0100_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.13_p0.6',       'file': 'Isochr_Z0.0100_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},

        {'instrument': 'gaia',      'name': 'starevol_m0.01_p0.0',       'file': 'Isochr_Z0.0130_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.01_p0.0',       'file': 'Isochr_Z0.0130_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.01_p0.0',       'file': 'Isochr_Z0.0130_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.01_p0.2',       'file': 'Isochr_Z0.0130_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.01_p0.2',       'file': 'Isochr_Z0.0130_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.01_p0.2',       'file': 'Isochr_Z0.0130_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.01_p0.4',       'file': 'Isochr_Z0.0130_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.01_p0.4',       'file': 'Isochr_Z0.0130_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.01_p0.4',       'file': 'Isochr_Z0.0130_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_m0.01_p0.6',       'file': 'Isochr_Z0.0130_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_m0.01_p0.6',       'file': 'Isochr_Z0.0130_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_m0.01_p0.6',       'file': 'Isochr_Z0.0130_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},

        {'instrument': 'gaia',      'name': 'starevol_p0.15_p0.0',       'file': 'Isochr_Z0.0190_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.15_p0.0',       'file': 'Isochr_Z0.0190_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.15_p0.0',       'file': 'Isochr_Z0.0190_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_p0.15_p0.2',       'file': 'Isochr_Z0.0190_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.15_p0.2',       'file': 'Isochr_Z0.0190_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.15_p0.2',       'file': 'Isochr_Z0.0190_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_p0.15_p0.4',       'file': 'Isochr_Z0.0190_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.15_p0.4',       'file': 'Isochr_Z0.0190_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.15_p0.4',       'file': 'Isochr_Z0.0190_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_p0.15_p0.6',       'file': 'Isochr_Z0.0190_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.15_p0.6',       'file': 'Isochr_Z0.0190_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.15_p0.6',       'file': 'Isochr_Z0.0190_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},

        {'instrument': 'gaia',      'name': 'starevol_p0.29_p0.0',       'file': 'Isochr_Z0.0260_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.29_p0.0',       'file': 'Isochr_Z0.0260_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.29_p0.0',       'file': 'Isochr_Z0.0260_Vini0.00_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_p0.29_p0.2',       'file': 'Isochr_Z0.0260_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.29_p0.2',       'file': 'Isochr_Z0.0260_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.29_p0.2',       'file': 'Isochr_Z0.0260_Vini0.20_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_p0.29_p0.4',       'file': 'Isochr_Z0.0260_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.29_p0.4',       'file': 'Isochr_Z0.0260_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.29_p0.4',       'file': 'Isochr_Z0.0260_Vini0.40_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'gaia',      'name': 'starevol_p0.29_p0.6',       'file': 'Isochr_Z0.0260_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': '2mass',     'name': 'starevol_p0.29_p0.6',       'file': 'Isochr_Z0.0260_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},
        {'instrument': 'bessell',   'name': 'starevol_p0.29_p0.6',       'file': 'Isochr_Z0.0260_Vini0.60_t06.000.dat',                           'function': _read_model_starevol},

        {'instrument': 'wise',      'name': 'atmo2020_neq_s',         'file': '0.0005_ATMO_NEQ_strong_vega.txt',                               'function': _read_model_atmo2020},
        {'instrument': 'wise',      'name': 'atmo2020_neq_w',         'file': '0.0005_ATMO_NEQ_weak_vega.txt',                                 'function': _read_model_atmo2020},
        {'instrument': 'wise',      'name': 'atmo2020_ceq',           'file': '0.0005_ATMO_CEQ_vega.txt',                                      'function': _read_model_atmo2020},

        {'instrument': 'gaia',      'name': 'spots_p0.00',            'file': 'f000_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': '2mass',     'name': 'spots_p0.00',            'file': 'f000_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': 'gaia',      'name': 'spots_p0.17',            'file': 'f017_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': '2mass',     'name': 'spots_p0.17',            'file': 'f017_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': 'gaia',      'name': 'spots_p0.34',            'file': 'f034_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': '2mass',     'name': 'spots_p0.34',            'file': 'f034_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': 'gaia',      'name': 'spots_p0.51',            'file': 'f051_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': '2mass',     'name': 'spots_p0.51',            'file': 'f051_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': 'gaia',      'name': 'spots_p0.68',            'file': 'f068_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': '2mass',     'name': 'spots_p0.68',            'file': 'f068_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': 'gaia',      'name': 'spots_p0.85',            'file': 'f085_all_filters.isoc',                                         'function': _read_model_SPOTS},
        {'instrument': '2mass',     'name': 'spots_p0.85',            'file': 'f085_all_filters.isoc',                                         'function': _read_model_SPOTS},

        {'instrument': 'johnson',   'name': 'dartmouth_p0.00_p0.0_nomag',  'file': 'dmestar_00001.0myr_z+0.00_a+0.00_gas07_mrc.JC2MASSGaia',   'function': _read_model_Dartmouth},
        {'instrument': 'gaia',      'name': 'dartmouth_p0.00_p0.0_nomag',  'file': 'dmestar_00001.0myr_z+0.00_a+0.00_gas07_mrc.JC2MASSGaia',   'function': _read_model_Dartmouth},
        {'instrument': '2mass',     'name': 'dartmouth_p0.00_p0.0_nomag',  'file': 'dmestar_00001.0myr_z+0.00_a+0.00_gas07_mrc.JC2MASSGaia',   'function': _read_model_Dartmouth},

        {'instrument': 'johnson',   'name': 'dartmouth_p0.00_p0.0_mag',  'file': 'dmestar_00001.0myr_z+0.00_a+0.00_gas07_mrc_magBeq.JC2MASSGaia',   'function': _read_model_Dartmouth},
        {'instrument': 'gaia',      'name': 'dartmouth_p0.00_p0.0_mag',  'file': 'dmestar_00001.0myr_z+0.00_a+0.00_gas07_mrc_magBeq.JC2MASSGaia',   'function': _read_model_Dartmouth},
        {'instrument': '2mass',     'name': 'dartmouth_p0.00_p0.0_mag',  'file': 'dmestar_00001.0myr_z+0.00_a+0.00_gas07_mrc_magBeq.JC2MASSGaia',   'function': _read_model_Dartmouth},

        {'instrument': 'mko',       'name': 'atmo2020_ceq',              'file': '0.001_ATMO_CEQ_vega.txt',                       'function': _read_model_atmo2020},
        {'instrument': 'mko',       'name': 'atmo2020_neq_strong',       'file': '0.001_ATMO_NEQ_strong_vega.txt',                'function': _read_model_atmo2020},
        {'instrument': 'mko',       'name': 'atmo2020_neq_weak',         'file': '0.001_ATMO_NEQ_weak_vega.txt',                  'function': _read_model_atmo2020},
        {'instrument': 'irac',      'name': 'atmo2020_ceq',              'file': '0.001_ATMO_CEQ_vega.txt',                       'function': _read_model_atmo2020},
        {'instrument': 'irac',      'name': 'atmo2020_neq_strong',       'file': '0.001_ATMO_NEQ_strong_vega.txt',                'function': _read_model_atmo2020},
        {'instrument': 'irac',      'name': 'atmo2020_neq_weak',         'file': '0.001_ATMO_NEQ_weak_vega.txt',                  'function': _read_model_atmo2020},
        {'instrument': 'wise',      'name': 'atmo2020_ceq',              'file': '0.001_ATMO_CEQ_vega.txt',                       'function': _read_model_atmo2020},
        {'instrument': 'wise',      'name': 'atmo2020_neq_strong',       'file': '0.001_ATMO_NEQ_strong_vega.txt',                'function': _read_model_atmo2020},
        {'instrument': 'wise',      'name': 'atmo2020_neq_weak',         'file': '0.001_ATMO_NEQ_weak_vega.txt',                  'function': _read_model_atmo2020},

        {'instrument': 'johnson',   'name': 'ekstroem_p0.00_rot',        'file': 'iso_p0.00.dat',                                 'function': _read_model_Ekstroem},
        {'instrument': 'johnson',   'name': 'ekstroem_p0.00_norot',      'file': 'iso_p0.00.dat',                                 'function': _read_model_Ekstroem},
        {'instrument': 'hr',        'name': 'ekstroem_p0.00_rot',        'file': 'iso_p0.00.dat',                                 'function': _read_model_Ekstroem},
        {'instrument': 'hr',        'name': 'ekstroem_p0.00_norot',      'file': 'iso_p0.00.dat',                                 'function': _read_model_Ekstroem},
        {'instrument': 'hr',        'name': 'ekstroem_m1.50_rot',        'file': 'iso_m1.50.dat',                                 'function': _read_model_Ekstroem},
        {'instrument': 'hr',        'name': 'ekstroem_m1.50_norot',      'file': 'iso_m1.50.dat',                                 'function': _read_model_Ekstroem}

    ],
    'data': {}
}


#######################################
# public functions
#
def mag_to_mass(age, distance, mag, Dmag, filt,
                instrument='IRDIS', model='bhac2015+cond2003', fill=False,
                age_range=None, distance_range=None, mag_err=None, Dmag_range=None):
    '''
    Convert a contrast value into mass

    Parameters
    ----------
    age : float
        Age of the target in Myr

    distance : float
        Distance of the target in pc

    mag : float
        Magnitude of the target in the filter

    Dmag : array
        Contrast value(s) in the filter

    filt : str
        Name of the filter

    instrument : str
        Name of the instrument. The default is IRDIS

    model : str
        Name of the evolutionary model. The default is bhac2015+cond2003

    fill : bool
        Fill interpolated values with min/max values in the models when
        trying to interpolate outside the values in the models
    
    age_range : list
        [min, max] age estimations for the target

    distance_range : list
        [min, max] distance estimations for the target

    mag_err : float
        Error on the target magnitude

    Dmag_range : array
        [min, max] contrast estimations

    Returns
    -------
    mass, mass_min, mass_max : array
        Values of the mass interpolated into the model
    '''    
    
    # -------------------------------
    # get model data
    # -------------------------------
    masses, ages, values, data = model_data(instrument, model)

    # check ages
    if (age < ages.min()) or (age > ages.max()):
        raise ValueError('Age {0} Myr outside of model range [{1}, {2}]'.format(age, ages.min(), ages.max()))

    # check filter
    if filt not in values:
        raise ValueError('Filter {0} not available in list: {1}'.format(filt, values))
    
    # -------------------------------
    # explicit variable names
    # -------------------------------
    
    # age range
    if age_range is not None:
        if not isinstance(age_range, list):
            raise ValueError('Age range must be a 2-elements array')

        age_min = np.min(age_range)
        age_max = np.max(age_range)
    else:
        age_min = age
        age_max = age
                
    # dist range
    if distance_range is not None:
        if not isinstance(distance_range, list):
            raise ValueError('Dist range must be a 2-elements array')

        dist_min = np.min(distance_range)
        dist_max = np.max(distance_range)
    else:
        dist_min = distance
        dist_max = distance

    # Stellar mag range
    if mag_err is not None:
        if not isinstance(mag_err, (int, float)):
            raise ValueError('Stellar mag error must be a float')

        mag_min = mag - mag_err
        mag_max = mag + mag_err
    else:
        mag_min = mag
        mag_max = mag

    # delta mag range
    if Dmag_range is not None:
        raise ValueError('Dmag error not implemented')
    else:
        Dmag_faint  = Dmag
        Dmag_bright = Dmag

    # -------------------------------
    # absolute magnitude conversion
    # -------------------------------

    # nominal values
    Mabs_nom = mag - 5*np.log10(distance) + 5 + Dmag

    # taking errors into account
    Mabs_faint  = mag_min - 5*np.log10(dist_min) + 5 + Dmag_faint
    Mabs_bright = mag_max - 5*np.log10(dist_max) + 5 + Dmag_bright

    # -------------------------------
    # interpolate models
    # -------------------------------
    param = 'Mass'   # only parameter currently available
    values_nom = _interpolate_model(masses, ages, values, data, age, filt, param, Mabs_nom, fill)
    values_min = _interpolate_model(masses, ages, values, data, age_min, filt, param, Mabs_faint, fill)
    values_max = _interpolate_model(masses, ages, values, data, age_max, filt, param, Mabs_bright, fill)
    
    values_all = np.vstack((values_min, values_nom, values_max))
    values_min = np.nanmin(values_all, axis=0)
    values_max = np.nanmax(values_all, axis=0)
        
    return values_nom, values_min, values_max


def list_models():
    '''
    Print the list of available models
    '''
    print()
    print('Search paths:')
    for p in search_path:
        print(' * {}'.format(p))
    print()

    for i in range(len(models['properties'])):
        prop = models['properties'][i]
        
        print(prop['file'])
        print(' * instrument: {0}'.format(prop['instrument']))
        print(' * name:       {0}'.format(prop['name']))
        print(' * function:   {0}'.format(prop['function'].__name__))
        try:
            print(' * path:       {0}'.format(prop['path']))
        except KeyError:
            pass
        print()


def model_data(instrument, model):
    '''
    Return the model data for a given instrument

    Directly returns the data if it has been read and stored
    already. Otherwise read and store it before returning.
    
    Parameters
    ----------
    instrument : str
        Instrument name

    model : str
        Model name

    Returns
    -------
    data : tuple 
        Tuple (masses, ages, values, data)
    '''

    key = instrument.lower()+'_'+model.lower()

    if key not in models['data'].keys():
    #    print('Loading model {0} for {1}'.format(model, instrument))
        
        _read_model_data(search_path, models, instrument, model)

    return models['data'][key]


def add_search_path(path):
    '''
    Add a new location in the search path

    Useful to easily handle "private" models that are not provided
    with the public distribution of the package.

    Parameters
    ----------
    path : str
        Path to the additional directory
    '''
    
    path = Path(path).expanduser().resolve()
    
    # add only if necessary
    if path not in search_path:
        search_path.append(path)


def plot_model(instrument, model, param, age_list=None, mass_list=None):
    '''
    Plot parameter evolution as a function of age for a model and instrument

    Parameters
    ----------
    instrument : str
        Instrument name

    model : str
        Model name

    param : str
        Parameter of the model to be plotted
    
    age_list : array
        List of ages to use for the plots. Default is None, so it will 
        use all available ages

    mass_list : array
        List of masses to use for the plots. Default is None, so it will 
        use all available masses

    Returns
    -------
    path : str
        The complete path to the model file
    '''
    
    masses, ages, values, data = model_data(instrument, model)

    if not mass_list:
        mass_list = masses
        
    if not age_list:
        age_list = ages

    cmap = cm.plasma
    norm = colors.LogNorm(vmin=ages.min(), vmax=ages.max())
    
    #
    # param vs. age
    #
    fig = plt.figure(0, figsize=(12, 9))
    plt.clf()
    ax = fig.add_subplot(111)
    
    for mass in mass_list:
        if (mass <= 75):
            ax.plot(ages, data[masses == mass, :, values == param].squeeze(), 
                    label=r'{0:.1f} MJup'.format(mass), color=cmap(mass/75.))

    ax.set_xscale('log')
    ax.set_yscale('linear')
    
    ax.set_xlabel('Age [Myr]')
    ax.set_ylabel(param)

    ax.set_title('{0}, {1}'.format(model, instrument))
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()

    #
    # param vs. mass
    #
    fig = plt.figure(1, figsize=(12, 9))
    plt.clf()
    ax = fig.add_subplot(111)
    
    for age in age_list:
        ax.plot(masses, data[:, ages == age, values == param].squeeze(), 
                label=r'{0:.4f} Myr'.format(age), color=cmap(norm(age)))

    ax.set_xlim(0, 75)
        
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    
    ax.set_xlabel(r'Mass [$M_{Jup}$]')
    ax.set_ylabel(param)

    ax.set_title('{0}, {1}'.format(model, instrument))
    
    # ax.legend(loc='upper right')
    
    plt.tight_layout()
