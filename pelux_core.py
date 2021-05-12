# coding: utf-8

import numpy as np
from pathlib import Path
import sys
sys.path.append("C:\\Users\\Vito\\Downloads\\Python-utils-master\\vigan\\astro")
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
        if type(app_mag_error)!=type(None) & type(parallax_error)!=type(None): 
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



def load_isochrones(model,surveys=None,folder=Path('C:/Users/Vito/Desktop/PhD/Progetti/BEAST/Stellar_ages/CMS/BT-Settl_stilism_cs0_GaiaDR3')):
    if surveys==None: surveys=['gaia','2mass','wise']
    else: surveys=list(map(str.lower,surveys))

    model=(str.lower(model)).replace('-','_')

    file=model
    for i in range(len(surveys)): file=file+'_'+surveys[i]
    PIK=folder / (file+'.pkl')

    try: #se c'è
        open(PIK,'r')
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
                        f=interp1d(masses[~nans],data0[~nans,k,w],kind='linear',fill_value='extrapolate')
                        c1=0
                        while mnew[c1]<m0[0]: c1=c1+1
                        c2=c1
                        while mnew[c2]<=m0[-1]: 
                            c2=c2+1 #sarà il primo elemento maggiore, ma quando farò l'indexing, a[c1,c2] equivarrà ad a[c1,...,c2-1]
                            if c2==n1: break
                        iso[c1:c2,k,j]=f(mnew[c1:c2])
                    for k in range(n1): #spline in età. Per ogni massa
                        nans, x= nan_helper(iso[k,:,j])
                        nans=nans.reshape(n_elements(nans))
                        a0=ages[~nans]
                        if n_elements(ages[~nans])==0: continue
                        f=interp1d(ages[~nans],iso[k,~nans,j],kind='linear',fill_value='extrapolate')
                        c1=0
                        while anew[c1]<a0[0]: c1=c1+1
                        c2=c1
                        while anew[c2]<1.0001*a0[-1]:
                            c2=c2+1 #sarà il primo elemento maggiore, ma quando farò l'indexing, a[c1,c2] equivarrà ad a[c1,...,c2-1]
                            if c2==n2: break
                        iso_f[k,c1:c2,j+c]=f(anew[c1:c2])
                c+=survey_el[i]
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



def isochronal_age(phot_app,phot_err_app,par,par_err,border_age=False):

    path=Path('C:/Users/Vito/Desktop/PhD/Progetti/BEAST/Stellar_ages/CMS/BT-Settl_stilism_cs0_GaiaDR3')

    #carica parametri da file
    with open(path / 'options.txt') as f:
        opt = np.genfromtxt(f,dtype="str")
        parameters=opt[:,1]
        max_r=float(parameters[1])
        lit_file=parameters[2]
        ph_cut=float(parameters[5])
        max_flux_cont=float(parameters[6])
        bin_frac=float(parameters[7])
        fitting_method=parameters[9].strip()
        if parameters[10]=='single': n_est=1
        elif parameters[10]=='multiple': n_est=4 #SNU, SRU, BNU, BRU
        surveys=parameters[12].split(',')
        ws0=['2MASS' in surveys,'WISE' in surveys,'Panstarrs' in surveys] #quali survey oltre a Gaia devo usare?

    #carica isocrone
    upperpath=path.parent.absolute()
    isochrones=None
    if isochrones==None:
        max_iso_age=float(parameters[3])
        model=parameters[0]
        iso_path=parameters[4]
        mnew,anew,fnew,newMC=load_isochrones(model)

    #una qualche lista di fotometria di stelle con errori
    #in questo esempio, ho 6 filtri e 2 stelle
#    phot_app=np.array([[16.08196,15.29399],[17.18979,16.95128],[15.01009,14.04471],[13.51800,11.99600],[12.82700,11.35400],[12.60300,11.01900]]) #una riga per filtro, una colonna per stella
#    phot_err_app=np.array([[0.00093,0.00636],[0.00664,0.02289],[0.00170,0.02044],[0.02600,0.02200],[0.02600,0.02800],[0.03000,0.02300]])
#    par=np.array([5.70005477,5.70101925])
#    par_err=np.array([0.12052761,0.03417909])

    #le trasformo in assolute
    phot,phot_err=app_to_abs_mag(phot_app,par,app_mag_error=phot_err_app,parallax_error=par_err)


    #raggi per peso della media

    #contaminazione in flusso
    cont=np.zeros(2) #contaminazione nei filtri 2MASS per le stelle, in questo caso nulla

    l0=phot.shape #(6,2)
    xlen=l0[0] #n. filtri: 6
    ylen=l0[1] #n. stelle: 2

    filt=where_v(['J','H','K','G','Gbp','Grp'],fnew)
    wc=np.array([[filt[2],filt[0],filt[1],filt[5]],[filt[3],filt[3],filt[3],filt[4]]]) #(G-K), (G-J), (G-H), (Gbp-Grp)

    red=np.zeros([xlen,ylen]) #reddening da applicare
    #for i in range(xlen): red[i,:]=extinction(redd,filt[i])

    l=newMC.shape #(780,460,10) cioè masse, età e filtri
    sigma=100+np.zeros([l[0],l[1],xlen]) #matrice delle distanze fotometriche (780,460,3)
    loga=np.zeros([4,ylen]) #stime di log(età) in ognuno dei quattro canali (4,10)

    #calcolare reddening
    m_cmsf=np.empty([4,ylen,n_est]) #stime di massa (4,2,1)
    m_cmsf.fill(np.nan)
    a_cmsf=np.empty([4,ylen,n_est]) #stime di età (4,2,1)
    a_cmsf.fill(np.nan)

    n_val=np.zeros([4,n_est]) #numero di stelle usate per la stima per canale (0) e tipologia (SRB ecc, 1) (4,1)
    tofit=np.zeros([4,ylen,n_est]) #contiene, per ogni CMS, 1 se fittato, 0 se non fittato (4,2,1)

    bin_corr=2.5*np.log10(2)*bin_frac #ossia, se le binarie sono identiche, la luminosità osservata è il doppio di quella della singola componente

    if n_est==4:
        phot_srb=phot-red #stelle singole (S), con reddening (R), biased. Se stima singola, include correzione binaria
        phot_brb=phot-red+bin_corr #applico la correzione binaria (B)
        phot_snb=phot #reddening nullo (N)
        phot_bnb=phot+bin_corr    
    else: phot_srb=phot-red+bin_corr #(6,2) come phot

    fate=np.ones([4,ylen,n_est]) #(4,2,1)  ci dice se la stella i nella stima j e nel canale k è stata fittata, ha errori alti, contaminazione ecc. Di default è contaminata (1)

    for t in range(n_est):
        if t==0:
            phot0=phot_srb
            wh='SRB'
        elif t==1:
            phot0=phot_brb
            wh='BRB'
        elif t==2:
            phot0=phot_snb
            wh='SNB'
        elif t==3:
            phot0=phot_bnb
            wh='BNB'

        sigma=np.empty([l[0],l[1],xlen]) #(780,480,3) matrice delle distanze fotometriche
        sigma.fill(np.nan) #inizializzata a NaN
         
        for i in range(ylen): #devo escludere poi i punti con errore fotometrico non valido     
            w,=np.where(is_phot_good(phot[:,i],phot_err[:,i],max_phot_err=ph_cut))
            if len(w)>0:
                e_j=-10.**(-0.4*phot_err[w,i])+10.**(+0.4*phot_err[w,i])
                for h in range(len(w)): sigma[:,:,w[h]]=(10.**(-0.4*(newMC[:,:,w[h]]-phot0[w[h],i]))-1.)/e_j[h]
            cr=np.zeros([l[0],l[1],4]) #(780,480,4) #per il momento comprende le distanze in G-K, G-J, G-H, Gbp-Grp
    #        wc=np.array([[2,3],[0,3],[1,3],[4,5]]) #(G-K), (G-J), (G-H), (Gbp-Grp)
            for j in range(4):
                cr[:,:,j]=(sigma[:,:,wc[0,j]])**2+(sigma[:,:,wc[1,j]])**2 #equivale alla matrice delle distanze in (G,K), (G,J), (G,H), (Gbp,Grp)
                colth=np.empty(l[1]) #480 voglio verificare se la stella si trova "in mezzo" al set di isocrone oppure all'esterno; voglio fare un taglio a mag costante
                colth.fill(np.nan)
                asa=np.zeros(l[1])
                for q in range(l[1]): #480
                    asa[q],im0=min_v(newMC[:,q,wc[0,j]]-phot0[wc[0,j],i],absolute=True) #trova il punto teorico più vicino nel primo filtro per ogni isocrona
                    if abs(asa[q])<0.1: colth[q]=newMC[im0,q,wc[1,j]] #trova la magnitudine corrispondente nel secondo filtro della coppia
                asb=min(asa,key=abs) #se la minima distanza nel primo filtro è maggiore della soglia, siamo al di fuori del range in massa delle isocrone
                est,ind=min_v(cr[:,:,j])
                if (est <= 2.25 or (phot0[wc[1,j],i] >= min(colth) and phot0[wc[1,j],i] <= max(colth))) and np.isnan(est)==False and (np.isnan(min(colth))==False and np.isnan(max(colth))==False):  #condizioni per buon fit: la stella entro griglia isocrone o a non più di 3 sigma, a condizione che esista almeno un'isocrona al taglio in "colth"
                    m_cmsf[j,i,t]=mnew[ind[0]] #massa del CMS i-esimo
                    a_cmsf[j,i,t]=anew[ind[1]] #età del CMS i-esimo
                    n_val[j,t]=n_val[j,t]+1
                    tofit[j,i,t]=1

                if (is_phot_good(phot0[wc[0,j],i],phot_err[wc[0,j],i],max_phot_err=ph_cut)==0) or (is_phot_good(phot0[wc[1,j],i],phot_err[wc[1,j],i],max_phot_err=ph_cut)==0): pass #rimane 0
                elif est > 2.25 and phot0[wc[1,j],i] < min(colth):  fate[j,i,t]=2
                elif est > 2.25 and phot0[wc[1,j],i] > max(colth):  fate[j,i,t]=3
                elif est > 2.25 and abs(asb) >= 0.1: fate[j,i,t]=4
                else: fate[j,i,t]=5
                if (border_age==True and est>=2.25 and phot0[wc[1,j],i]>max(colth)):
                    a_cmsf[j,i,t]=anew[0]
                    tofit[j,i,t]=1
                
        if anew[-1]<150: plot_ages=[1,3,5,10,20,30,100] #ossia l'ultimo elemento
        elif anew[-1]<250: plot_ages=[1,3,5,10,20,30,100,200]
        elif anew[-1]<550: plot_ages=[1,3,5,10,20,30,100,200,500]
        elif anew[-1]<1050: plot_ages=[1,3,5,10,20,30,100,200,500,1000]
        else: plot_ages=[1,3,5,10,20,30,100,200,500,1000]

    if os.path.isfile(path / 'TestFile.txt')==0: print("Ora dovrebbe plottare una figura")
#    if file_search(path+'G-K_G_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[3,*]-phot0[2,*],phot0[3,*],newMC,'G-K','G',plot_ages,iso_ages=anew,xerr=phot_err[2,*]+phot_err[3,*],yerr=phot_err[3,*],tofile=path+'G-K_G_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[0,*,t],/show_errors,charsize=0.3
#    if file_search(path+'G-J_J_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[3,*]-phot0[0,*],phot0[0,*],newMC,'G-J','J',plot_ages,iso_ages=anew,xerr=phot_err[0,*]+phot_err[3,*],yerr=phot_err[0,*],tofile=path+'G-J_J_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[1,*,t],/show_errors,charsize=0.3
#    if file_search(path+'G-H_H_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[3,*]-phot0[1,*],phot0[1,*],newMC,'G-H','H',plot_ages,iso_ages=anew,xerr=phot_err[1,*]+phot_err[3,*],yerr=phot_err[1,*],tofile=path+'G-H_H_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[2,*,t],/show_errors,charsize=0.3
#    if file_search(path+'Gbp-Grp_G_'+wh+'.*') eq '' and keyword_set(no_img) eq 0 and keyword_set(silent) eq 0 then plot_stars2,phot0[4,*]-phot0[5,*],phot0[3,*],newMC,'Gbp-Grp','G',plot_ages,iso_ages=anew,xerr=phot_err[4,*]+phot_err[5,*],yerr=phot_err[3,*],tofile=path+'Gbp-Grp_G_'+wh+'.eps',label_points=1+indgen(ylen),sym_size=radius,highlight=tofit[3,*,t],/show_errors,charsize=0.3

    a_final=np.empty(ylen)
    m_final=np.empty(ylen)
    for i in range(ylen): 
        a_final[i]=np.nanmean(a_cmsf[:,i])
        m_final[i]=np.nanmean(m_cmsf[:,i])

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
    return(3.16*A*ebv)


#definisce range per plot CMD
def axis_range(col_name,col_phot):
    cmin=min(col_phot)
    cmax=max(col_phot)
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
    return(ang2)

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
            target_list = np.genfromtxt(f,dtype="str")
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
                            while str(x[0]['Source'][c]) not in obj1:
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
        data_s = XMatch.query(cat1=coo_table,cat2=surv_prop[index[surveys[i]]][0],max_distance=2 * u.arcsec, colRA1='RA',colDec1='DEC')
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
    
    x=np.linspace(x0,x1,num=n)
    if n_dim(f)==2:
        m=(y1-y0)/(x1-x0) #slope of the line
        d10=np.sqrt((x1-x0)**2+(y1-y0)**2) #distance
        
        y=y0+m*(x-x0)
        for i in range(n): I+=f[math.floor(x[i]),math.floor(y[i])]
    elif n_dim(f)==3:
        m=(y1-y0)/(x1-x0) #slope of the line
        d10=np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2) #distance
        
        y=np.linspace(y0,y1,num=n)
        z=np.linspace(z0,z1,num=n)
        for i in range(n): I+=f[math.floor(x[i]),math.floor(y[i]),math.floor(z[i])]
    
    return I/n*d10

def interstellar_ext(ra=None,dec=None,l=None,b=None,par=None,d=None,map_path=r'C:\Users\Vito\Desktop\PhD\Modelli\Extinction_maps',test_time=False,ext_map='leike',color='B-V'):
    fits_image_filename=os.path.join(map_path,'leike_mean_std.h5')
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

    if color=='B-V': return(ebv)
    else: return(extinction(ebv,color))

def cross_match(cat1,cat2,max_difference=0.01,other_column=None,rule=None,exact=False):
    """
    given two catalogues cat1 and cat2, returns the indices ind1 and ind2 such that:
    cat1[ind1]=cat2[ind2]
    if the input are 1D, or
    cat1[ind1,:]=cat2[ind2,:]
    if they are 2D.
    '=' must be thought as "for each i, |cat1[ind1[i]]-cat2[ind2[i]]|<max_difference",
    if exact=False (default mode). If exact=True, it is a strict equality.
    
    input:
        cat1: a 1D or 2D numpy array, specifying one raw per star, in the first catalogue
        cat2: a 1D or 2D numpy array, specifying one raw per star, in the second catalogue
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
    if n_dim(cat1)==1:
        if type(other_column)==type(None):
            for i in range(n):
                k,=np.where(abs(cat2-cat1[i])<max_difference)
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
                k,=np.where(abs(cat2-cat1[i])<max_difference)
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
                k,=np.where(abs(cat2-cat1[i])<max_difference)
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
                k,=np.where(d<max_difference)
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
                k,=np.where(d<max_difference)
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
                k,=np.where(d<max_difference)
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
        cat1[:,0]=coo['ra_v']
        cat1[:,1]=coo['dec_v']    

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
            mag=cat2_data[crit[surveys[i]]]
            indG1,indG2=cross_match(cat1,cat2,max_difference=0.001,other_column=mag,rule='min')
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
