import sys
sys.path.append(r'C:\Users\Vito\Desktop\PhD\Progetti\MADYS')
from pelux_core import *

class MADYS:
    def __init__(self, name, file, **kwargs):
        self.name = name
        self.file = file
        self.surveys = ['GAIA_EDR3','2MASS']
        self.coord = True
        self.model = 'bt_settl'           
        if len(kwargs)>0:
            if 'surveys' in kwargs: self.surveys = kwargs['surveys']
            if 'coord' in kwargs: self.coord = kwargs['coord']
            if 'model' in kwargs: self.model = kwargs['model']
        self.coo,data=search_phot(self.file,self.surveys,verbose=True,coordinates=self.coord)
        self.phot,self.phot_err,self.filters,self.kin=load_phot(self.file,self.surveys)
        par=self.kin[:,4]
        par_err=self.kin[:,5]
        self.iso=load_isochrones(self.model)
        self.ebv=interstellar_ext(ra=self.coo[:,0],dec=self.coo[:,1],par=par)
        self.ages,self.masses=isochronal_age(self.phot,self.phot_err,par,par_err,self.iso,self.surveys,ebv=self.ebv)
    def CMD(self,filt):
        plot_ages=np.array([1,3,5,10,20,30,100,200,500,1000])
        g_abs,g_err=app_to_abs_mag(self.phot[:,0],self.kin[:,4],app_mag_error=self.phot_err[:,0],parallax_error=self.kin[:,5])
        gbp_abs,gbp_err=app_to_abs_mag(self.phot[:,1],self.kin[:,4],app_mag_error=self.phot_err[:,1],parallax_error=self.kin[:,5])
        grp_abs,grp_err=app_to_abs_mag(self.phot[:,2],self.kin[:,4],app_mag_error=self.phot_err[:,2],parallax_error=self.kin[:,5])
        plot_CMD(gbp_abs-grp_abs,g_abs,self.iso[3],self.iso[2],self.iso[1],'Gbp-Grp','G',plot_ages=plot_ages,ebv=self.ebv,x_error=gbp_err+grp_err,y_error=g_err)
