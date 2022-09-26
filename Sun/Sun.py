import os
from dateutil import parser
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
from matplotlib import cm
from datetime import datetime,time, tzinfo, timedelta

cwd = os.getcwd()
path = os.path.join(cwd, "lightModel", "Sun")
path_setting = 'new_result'

sun_model_df = pd.read_csv(os.path.join(path,'Sun_weight.csv'),index_col=0)
Input = pd.read_csv(os.path.join(path,'Window.csv'),index_col=0)
Output = pd.read_csv(os.path.join(path,'Work_surf.csv'),index_col=0)
area_pos = sun_model_df[['x','y']]

class Sun_model:
    def __init__ (self,delta=0.5,mn = [-1,-2],mx = [10,11]):
        self.sun_model_df = sun_model_df
        self.area_pos = area_pos
        self.Input = Input
        self.Output = Output
        self.delta = delta
        self.mn = mn
        self.mx = mx
        self.new_X,self.new_Y = np.meshgrid(np.arange(mn[0], mx[0]+delta, delta),
                                            np.arange(mn[1], mx[1]+delta, delta))
        self.xflat = np.dstack((self.new_X.flatten(),self.new_Y.flatten()))[0]
        self.med_win = '10s'
        self.mean_win = '60s'
        self.feature_col = sun_model_df.columns[:-2]
        self.sun_arr = sun_model_df[self.feature_col].values
    
    def Time_point(self,start=7,end=17,minute_sec=20,time_point=None,Win=True):
        start_point = datetime(2022,5,7,start,0,0)
        end_point = datetime(2022,5,7,end,0,0)
        nums = (end_point - start_point)/timedelta(minutes=minute_sec)
        if time_point==None:
            time_ps2 = [ (start_point+timedelta(minutes=minute_sec*i)).strftime('%Y/%m/%d %H:%M:%S') for i in range(int(nums))]
        else:
            time_ps2 = [datetime(2022,5,7).strftime('%Y/%m/%d ')+ time_point]
        if Win:
            df = self.Input.copy()
        else:
            df = self.Output.copy()
            
        ffs2 = df.dropna()
        time_df2 = pd.DataFrame([])
        for tp in time_ps2:
            if time_df2.empty:
                # print(tp)
                time_df2 = ffs2.loc[tp:].iloc[0:1]
                # print(time_df2)
            else:
                time_df2 = pd.concat([time_df2,ffs2.loc[tp:].iloc[0:1]])
        time_df2 = time_df2[~time_df2.index.duplicated(keep='first')]
        return time_df2
    
    def Work_lx_meas(self,input_work_lx,name='',draw=True):
        input_work_lx = pd.merge(input_work_lx.T,
                                 self.area_pos,left_index=True,right_index=True,how='left')#.sort_index()
        input_work_lx.columns = ['lux','x','y']
        ##xflat = np.dstack((self.new_X.flatten(),self.new_Y.flatten()))[0]
        model_rbf = RBFInterpolator(input_work_lx[['x','y']].values,
                                    input_work_lx["lux"].values.reshape(-1,1),
                                    smoothing=0.2)      
        
        yflat = model_rbf(self.xflat)
        y_grid = yflat.reshape(self.new_X.shape)
        
        if draw:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_title(name)
            ax.plot_surface(self.new_Y,self.new_X, y_grid, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5)
            p = ax.scatter(input_work_lx['y'],input_work_lx['x'],input_work_lx['lux'],c=input_work_lx['lux'], 
                           s=50,ec='k',cmap=cm.coolwarm)
            for dd in input_work_lx[['y','x','lux']].values:
                label = '{0:.0f}'.format(dd[2])
                ax.text(dd[0]+0.1, dd[1]+0.1, dd[2], label,color='black')
            fig.colorbar(p,shrink=0.5)
            # ax.view_init(elev=90, azim=-90)
            ax.view_init(elev=30, azim=-90)
            plt.ylim( self.mx[0], self.mn[0])
            plt.xlim( self.mn[1], self.mx[1])#mn[1], 
            ax = fig.add_subplot(1, 2, 2)
            contours = ax.contour(self.new_Y,self.new_X,  y_grid, 40, cmap=cm.coolwarm)
            ax.set_aspect('equal', 'box')
            plt.ylim(self.mx[0],self.mn[0])
            plt.xlim(self.mn[1],self.mx[1])
            plt.show()
        return y_grid  
    
    def Work_lx_pred(self,right=0,left=0,Real=False,sample=None,name='',draw=True):
        model = self.sun_model_df.copy()
        if not Real:
            in_ = [right]*3+[left]*2
            upper_left,center_left,lower_left,upper_right,center_right = in_
            sensor_mapping_dict = {
                '2W': center_right,
                '2R_B': lower_left,
                '2L_B': upper_left,
                '2L_F': upper_right,
                '1test': center_left
            }
            input_vector = np.array([sensor_mapping_dict[cc] for cc in self.feature_col])
        else:
            input_vector = np.array([sample[cc] for cc in self.feature_col])
        model['lux'] = self.sun_arr.dot(input_vector)
        model_rbf = RBFInterpolator(model[['x','y']].values,model["lux"].values.reshape(-1,1),smoothing=0.2)
        yflat = model_rbf(self.xflat)
        y_grid = yflat.reshape(self.new_X.shape)
        if draw:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_title(name)
            ax.plot_surface(self.new_Y,self.new_X, y_grid, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5)
            p = ax.scatter(model['y'],model['x'],model['lux'],c=model['lux'], s=50,ec='k',cmap=cm.coolwarm)
            for dd in model[['y','x','lux']].values:
                label = '{0:.0f}'.format(dd[2])
                ax.text(dd[0]+0.1, dd[1]+0.1, dd[2], label,color='black')
            fig.colorbar(p,shrink=0.5)
            # ax.view_init(elev=90, azim=-90)
            ax.view_init(elev=30, azim=-90)
            plt.ylim( self.mx[0],self.mn[0])
            plt.xlim(self.mn[1],self.mx[1])#mn[1], 
            ax = fig.add_subplot(1, 2, 2)
            contours = ax.contour(self.new_Y,self.new_X,  y_grid, 40, cmap=cm.coolwarm)
            ax.set_aspect('equal', 'box')
            plt.ylim( self.mx[0],self.mn[0])
            plt.xlim(self.mn[1],self.mx[1])
            plt.show()
        return y_grid