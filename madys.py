import sys
sys.path.append(r'C:\Users\Vito\Desktop\PhD\Progetti\MADYS')
from pelux_core import *

class MADYS:
    def __init__(self, file, **kwargs):
#        self.name = name
        self.file = file
        self.surveys = ['2MASS','WISE','ALLWISE']
        self.coord = 'equatorial'
        if len(kwargs)>0:
            if 'surveys' in kwargs: self.surveys = kwargs['surveys']
            if 'coord' in kwargs: self.coord = kwargs['coord']
            if 'model' in kwargs: self.model = kwargs['model']
        self.phot,self.phot_err,self.kin,self.flags,self.headers=search_phot(self.file,self.surveys,verbose=True,coordinates=self.coord)
    def get_age(self,model):
        par=self.kin[:,4]
        par_err=self.kin[:,5]
        coo=self.kin[:,[0,2]]
        iso=load_isochrones(model)
        self.ebv=interstellar_ext(ra=coo[:,0],dec=coo[:,1],par=par)
        ages,masses=isochronal_age(self.phot,self.phot_err,self.headers[0],par,par_err,self.flags,iso,self.surveys,ebv=self.ebv,verbose=True,filename=self.file)
        return self.ebv,ages,masses
    def CMD(self,col,mag,model):
        
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

        plot_CMD(col_data,mag_data,iso,col,mag,plot_ages=plot_ages,ebv=self.ebv,x_error=col_err.reshape(len(col_err)),y_error=mag_err.reshape(len(col_err)),label_points=np.arange(len(mag_data)))        
