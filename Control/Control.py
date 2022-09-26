import gurobipy
import cvxpy as cp
print(cp.installed_solvers())
x = cp.Variable(2)
obj = cp.Minimize(x[0] + cp.norm(x, 1))
constraints = [x >= 2]
prob = cp.Problem(obj, constraints)
# Solve with GUROBI.
prob.solve(solver=cp.GUROBI)
print("optimal value with GUROBI:", prob.value)
import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
# import hdbscan
from datetime import datetime,time, tzinfo, timedelta
from matplotlib import cm,patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import cv2
from scipy.optimize import lsq_linear
from lightModel.Dark import Dark as dk
from lightModel.Sun import Sun as sn
from lightModel.Test import Functions as fc

newcolors =  cm.get_cmap('summer', 1024)(np.linspace(0, 1, 1024))
newcolors[:10, :] = np.array([0, 0, 0, 1])
my_map = ListedColormap(newcolors)

def linear_prog(dark,sun,L_m,input_temp,my_map=my_map,draw=False,):
    s_b = sun.Work_lx_meas(input_temp,draw=False)
    A = dark.conv_arr
    A,b = A.T[:,:-1], L_m -s_b.flatten()-A.T[:,-1]
    res = lsq_linear(A, b, bounds=(0, 1), 
                     lsmr_tol='exact',
                     verbose=0)
    
    cont_res = np.append(np.array([float('{:.2f}'.format(x)) for x in (res.x).tolist()]),1)
    L_r = dark.conv_arr.T.dot(cont_res)
    L_c = dk.empty_status.copy()
    L_c['status'] = cont_res
    res_dict ={'ME':np.mean(L_r+s_b.flatten()),
               'SD':np.std(res.fun),
               'MAE':np.mean(abs(res.fun)),
               'RMAE':np.mean(abs(L_r+s_b.flatten()-L_m))/L_m*100,
               'RMSE':np.mean(res.fun**2)**0.5,
              }
    print('Iter: ',res.nit,', Success: ',res.success)
    print('ME:{:.2f} +/-{:.3f}'.format( res_dict['ME'],res_dict['SD'])+\
          ' MAE: {:.3f}'.format(res_dict['MAE'])+\
          ' RMAE: {:.3f}'.format(res_dict['RMAE'])+\
          ' RMSE: {:.3f}'.format(res_dict['RMSE']))
    if draw:
        plot_control(dark,s_b,b,L_c,L_r,L_m,my_map,vmin=0,vmax=L_m+100,shrink=1,square_size=600,cmap='cividis')
    return L_r,s_b,res_dict

def zone_frame2(L_r,s_b,sun,v_range=25,plt_size = 1,outlier_thres=0.05,draw=False):
    daylighting = pd.DataFrame({'x':sun.xflat[:,0],'y':sun.xflat[:,1],'z':L_r+s_b.flatten()})
    zone_between = daylighting[(daylighting['z']>L_m-v_range)&(daylighting['z']<L_m+v_range)]
    min_pos = zone_between.quantile(outlier_thres,interpolation='lower')[['x','y']].values
    max_pos = zone_between.quantile(1-outlier_thres,interpolation='higher')[['x','y']].values

    if draw:
        zone_all = daylighting.copy()
        zone_under = daylighting[(daylighting['z']<L_m-v_range)]
        zone_over = daylighting[(daylighting['z']>L_m+v_range)]
        fig = plt.figure(figsize= (int(20*plt_size),int(4*plt_size)))
        for idx,zone in enumerate([zone_all,zone_under,zone_over,zone_between]):
            ax=fig.add_subplot(1,4,idx+1)
            sc = plt.scatter(zone['y'], zone['x'], c= zone['z'], vmin=L_m-v_range*2, vmax=L_m+v_range*2
                        ,cmap='coolwarm', alpha=alpha)    
            if idx==3:
                rect = patches.Rectangle((min_pos[1],min_pos[0]),
                                     (max_pos[1]-min_pos[1]),
                                     (max_pos[0]-min_pos[0]),
                                     linewidth=2,
                                     edgecolor='black',
                                     fill = False)
                ax.add_patch(rect)#range(L_m-v_range,  L_m+v_range,5),
            plt.ylim(12,-2)
            plt.xlim(-2,11)
        plt.show()
        loc_list = ['upper left','upper right']
        if (daylighting['z'].max()-daylighting['z'].mean())>=(daylighting['z'].mean()-daylighting['z'].min()):
            loc = loc_list[1]
        else:
            loc = loc_list[0]
            
        fig = plt.figure(figsize=[int(20*plt_size),int(4*plt_size)])
        ax = fig.add_subplot(1,2,1)
        sns.histplot(daylighting['z'],binwidth=10)
        ax_in = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height=1.3, # height : 1 inch
                    loc=loc)
        plt.scatter(daylighting['y'], daylighting['x'], c= daylighting['z'], vmin=L_m-v_range*2, vmax=L_m+v_range*2
                        ,cmap='coolwarm', alpha=alpha) 
        plt.ylim(12,-2)
        plt.xlim(-2,11)
        plt.axis('off')
        ax = fig.add_subplot(1,2,2)
        new_zone = daylighting[
        (daylighting['y']<=max_pos[1])&
        (daylighting['y']>=min_pos[1])&
        (daylighting['x']<=max_pos[0])&
        (daylighting['x']>=min_pos[0])]
        sns.histplot(new_zone['z'],binwidth=10)


        ax.set_xlim(daylighting['z'].min(),daylighting['z'].max())
        ax_in = inset_axes(ax,
                    width="30%", # width = 30% of parent_bbox
                    height=1.3, # height : 1 inch
                    loc=loc)
        plt.scatter(new_zone['y'], new_zone['x'], c= new_zone['z'], vmin=L_m-v_range*2, vmax=L_m+v_range*2
                        ,cmap='coolwarm', alpha=alpha) 
        plt.ylim(12,-2)
        plt.xlim(-2,11)
        plt.axis('off')
        plt.show()
    return min_pos,max_pos

def integer_prog(dark1,sun1,dark,sun,L_m,time_point):

    s_b1 = sun1.Work_lx_meas(sun1.Time_point(Win=False,time_point=time_point),draw=False)
    s_b = sun.Work_lx_meas(sun.Time_point(Win=False,time_point=time_point),draw=False)
    
    A = dark1.conv_arr
    new_A,new_b = A.T[:,:-1], L_m -s_b1.flatten()-A.T[:,-1]
    
    res_dict = {}
    A1 = new_A/2
    # Generate a random problem
    res_x = cp.Variable(48, integer=True)
    constraints = [0<= res_x, res_x <= 2]
    objective = cp.Minimize(cp.sum_squares(A1 @ res_x - new_b))
    try:
        prob = cp.Problem(objective,constraints)
        prob.solve(solver='GUROBI',verbose=0)# mosek GUROBI GLOP CPLEX GLPK_MI SCIP XPRESS ,verbose=True
    except:
        print("Status: ", prob.status)
        
    Int_L_c = dk.empty_status.copy()
    Int_L_c['status'] = np.append(res_x.value.astype('int')/2,1)

    # Gradient
    init_st= Int_L_c.copy()
    Int_L_r = dark.conv_arr.T.dot(Int_L_c['status'])
    lighting = cv2.GaussianBlur(Int_L_r.reshape(dark.new_X.shape)+s_b, (9,9),0)
    Gx= np.gradient(lighting,axis=1)
    grad_st = pd.merge(
        pd.merge(dark.pos_df.astype('float'),init_st[init_st['status']==0.5]
                 ,how='right',left_index=True,right_index=True).reset_index(),
        pd.DataFrame({'x':dark.new_X.flatten(), 'y':dark.new_Y.flatten(),'grad':Gx.flatten()}),
        on=('x','y'),how='left')
    init_st.loc[grad_st[grad_st['grad']>=0]['index'],'status'] = 2
    init_st.loc[grad_st[grad_st['grad']<0]['index'],'status'] = 3

    Int_L_r1 = dark1.bulb_conv(init_st.squeeze())['z'].values
    result = Int_L_r1+s_b1.flatten()
    err = result-L_m
    res_dict ={'ME':np.mean(result),
               'SD':np.std(result),
               'MAE':np.mean(abs(err)),
               'RMAE':np.mean(abs(err))/L_m*100,
               'RMSE':np.mean(err**2)**0.5,
              }
    print('ME:{:.2f} +/-{:.3f}'.format( res_dict['ME'],res_dict['SD'])+\
          ' MAE: {:.3f}'.format(res_dict['MAE'])+\
          ' RMAE: {:.3f}'.format(res_dict['RMAE'])+\
          ' RMSE: {:.3f}'.format(res_dict['RMSE']))
    plot_control(dark1,s_b1,new_b,init_st,Int_L_r1,L_m,my_map,vmin=0,mode=True,
                 vmax=L_m+100,shrink=1,square_size=600,cmap='cividis')
   
    return Int_L_r1,s_b1,res_dict



class Control_model:
    def __init__ (self,delta=0.5,mn = [-1,-2],mx = [10,11],kernel_size=10):
        self.init_mn = [-1, -2]
        self.init_mx = [11, 11]
        self.dark = dk.Dark_model(delta=delta,kernel_size=kernel_size,mn=init_mn,mx=init_mx)
        self.sun = sn.Sun_model(delta=delta,mn=init_mn,mx=init_mx)
        self.L_m = 1000 #照度需求 light mantaining level

        
    def control_test(self,L_m,draw=True):
        input_temp = sun.Time_point(Win=False,time_point=time_point)
        L_r,s_b,res_dict1 = linear_prog(self.dark,self.sun,L_m,input_temp,draw=draw)
        new_mn,new_mx = zone_frame2(L_r,s_b,sun,draw=draw)