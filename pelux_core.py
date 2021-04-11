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
from astroquery.xmatch import XMatch
import csv
from astropy.table import Table
from astropy.io import ascii
from tabulate import tabulate


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
        shape = []
        a = len(x)
        shape.append(a)
        b = x[0]
        while a > 0:
            try:
                if isinstance(b,str): break
                a = len(b)
                shape.append(a)
                b = b[0]

            except:
                break
        if shape: dim=shape
        else: dim=len(shape)
    except TypeError: dim=0
    return dim

def app_to_abs_mag(app_mag,parallax,app_mag_error=False,parallax_error=False):
    if isinstance(app_mag,list): app_mag=np.array(app_mag)
    if isinstance(parallax,list): parallax=np.array(parallax)
    dm=5*np.log10(100./parallax) #modulo di distanza
    dim=n_dim(app_mag)
    if dim <= 1:
        result=app_mag-dm
        if (app_mag_error!=False) & (parallax_error!=False): 
            if isinstance(app_mag_error,list): app_mag_error=np.array(app_mag_error)
            if isinstance(parallax_error,list): parallax_error=np.array(parallax_error)
            total_error=np.sqrt(app_mag_error**2+(5/np.log(10)/parallax)**2*parallax_error**2)
            result=[result,total_error]
    else: #  se è 2D, bisogna capire se ci sono più filtri e se c'è anche l'errore fotometrico
        l=n_dim(app_mag,shape=True)
        abs_mag=np.empty([l[0],l[1]])
        for i in range(l[0]): abs_mag[i,:]=app_mag[i,:]-dm
        result=abs_mag
        if (parallax_error.any()!=False):
            if isinstance(app_mag_error,list): app_mag_error=np.array(app_mag_error)
            if isinstance(parallax_error,list): parallax_error=np.array(parallax_error)
            total_error=np.empty([l[0],l[1]])
            for i in range(l[0]): 
                total_error[i,:]=np.sqrt(app_mag_error[i,:]**2+(5/np.log(10)/parallax)**2*parallax_error**2)
            result=[result,total_error]        
            
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
         
        for i in range(y_len): #devo escludere poi i punti con errore fotometrico non valido     
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


def import_phot(filename,coordinates=False,surveys=['2MASS','GAIA_EDR3']): #coord_file

    folder=os.path.dirname(filename)     #working path selection

    #creates a .csv file with coordinates
    if coordinates==False: 
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
        data = Table([ra, dec],names=['ra_v','dec_v'])
        new_file=filename.replace('.txt','_coordinates.csv')
        ascii.write(data, new_file, overwrite=True, format='csv', delimiter=',')
    else:
        if filename.endswith('.txt'):
            with open(filename) as f:
                data = np.genfromtxt(f,dtype="str")
            new_file=filename.replace('.txt','.csv')
            with open(new_file, newline='', mode='w') as f:
                r_csv = csv.writer(f, delimiter=',')
                for i in range(len(data)):
                    r_csv.writerow(data[i])
        else: new_file=filename

    #VizieR queries. Results saved as .txt
    index={'GAIA_EDR3':0, '2MASS':1, 'ALLWISE':2}

    surv_prop=[['vizier:I/350/gaiaedr3',['angDist', 'source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',  'ruwe', 'phot_g_mean_flux_error', 'phot_g_mean_mag', 'phot_bp_mean_flux_error', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'dr2_radial_velocity', 'dr2_radial_velocity_error', 'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error', 'phot_g_mean_mag_corrected', 'phot_g_mean_mag_error_corrected', 'phot_g_mean_flux_corrected', 'phot_bp_rp_excess_factor_corrected', 'ra_epoch2000_error', 'dec_epoch2000_error', 'ra_dec_epoch2000_corr'],['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','ruwe','phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag','phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error','dr2_radial_velocity','dr2_radial_velocity_error','phot_bp_rp_excess_factor_corrected']],
               ['vizier:II/246/out',['2MASS','ra_v','dec_v','Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag','Qfl']],
               ['vizier:II/328/allwise',['AllWISE','RAJ2000','DEJ2000','W1mag','e_W1mag','W2mag','e_W2mag','W3mag','e_W3mag','W4mag','e_W4mag','ccf','d2M']]
              ]

    
#    angDist,ra_v,dec_v,AllWISE,RAJ2000,DEJ2000,eeMaj,eeMin,eePA,W1mag,W2mag,W3mag,W4mag,Jmag,Hmag,Kmag,e_W1mag,e_W2mag,e_W3mag,e_W4mag,e_Jmag,e_Hmag,e_Kmag,ID,ccf,ex,var,qph,pmRA,e_pmRA,pmDE,e_pmDE,d2M

    
    if isinstance(surveys,str):        
        data_s = XMatch.query(cat1=open(Path(new_file)),cat2=surv_prop[index[surveys]][0],max_distance=2 * u.arcsec, colRA1='ra_v',colDec1='dec_v')
        output_file=os.path.join(folder,str('py_cms_'+surveys+'.csv'))
        f=open(os.path.join(folder,str(surveys+'_tab.txt')), "w+")
        if surveys=='GAIA_EDR3':
            f.write(tabulate(data_s[surv_prop[0][2]],
                         headers=['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','ruwe','G','err_G','Gbp','err_GBP','GRP','err_GRP','radial_velocity', 'radial_velocity_error','bp_rp_excess_factor'], tablefmt='plain', stralign='right',floatfmt=(".5f",".11f",".4f",".11f",".4f",".7f",".7f",".7f",".7f",".7f",".7f",".3f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f")))
            data_s=data_s[surv_prop[0][1]]
        elif surveys=='2MASS':
            f.write(tabulate(data_s[surv_prop[1][1]],
                         headers=['ID','ra','dec','J','err_J','H','err_H','K','err_K','qfl'], tablefmt='plain', stralign='right',floatfmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f")))
        elif surveys=='ALLWISE':
            f.write(tabulate(data_s[surv_prop[2][1]],
                         headers=['ID','ra','dec','W1','err_W1','W2','err_W2','W3','err_W3','W4','err_W4','ccf','d2M'], tablefmt='plain', stralign='right',floatfmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".4f")))
        f.close()
        data_s.write(output_file, overwrite=True)        
    else:
        for i in range(len(surveys)): 
            if isinstance(surveys[i],str):        
                data_s = XMatch.query(cat1=open(Path(new_file)),cat2=surv_prop[index[surveys[i]]][0],max_distance=2 * u.arcsec, colRA1='ra_v',colDec1='dec_v')
                output_file=os.path.join(folder,str('py_cms_'+surveys[i]+'.csv'))
                f=open(os.path.join(folder,str(surveys[i]+'_tab.txt')), "w+")
                if surveys[i]=='GAIA_EDR3':
                    f.write(tabulate(data_s[surv_prop[0][2]],
                                 headers=['source_id','ra','ra_error','dec','dec_error','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error','ruwe','G','err_G','Gbp','err_GBP','GRP','err_GRP','radial_velocity', 'radial_velocity_error','bp_rp_excess_factor'], tablefmt='plain', stralign='right',floatfmt=(".5f",".11f",".4f",".11f",".4f",".7f",".7f",".7f",".7f",".7f",".7f",".3f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f",".4f")))
                    data_s=data_s[surv_prop[0][1]]
                elif surveys[i]=='2MASS':
                    f.write(tabulate(data_s[surv_prop[1][1]],
                                 headers=['ID','ra','dec','J','err_J','H','err_H','K','err_K','qfl'], tablefmt='plain', stralign='right',floatfmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f")))
                elif surveys[i]=='ALLWISE':
                    f.write(tabulate(data_s[surv_prop[2][1]],
                                 headers=['ID','ra','dec','W1','err_W1','W2','err_W2','W3','err_W3','W4','err_W4','ccf','d2M'], tablefmt='plain', stralign='right',floatfmt=(".5f",".8f",".8f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".3f",".4f")))
                f.close()
                data_s.write(output_file, overwrite=True)        

    return None

