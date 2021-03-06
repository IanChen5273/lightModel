from dateutil import parser
from statistics import median,mean
import cv2
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
import hdbscan
from datetime import datetime,time, tzinfo, timedelta
import requests
import math
import sympy
from itertools import combinations
from scipy.interpolate import BSpline, splrep, splev,make_lsq_spline,griddata,RBFInterpolator
import seaborn as sns
from tqdm import tqdm
from lightModel.Dark import *
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import os
cwd = os.getcwd()
path = os.path.join(cwd, "lightModel", "Test")
pd.options.display.float_format = '{:,.5f}'.format
pos_df = pd.read_csv( os.path.join(path,'Area.csv'),index_col=0)
sec = [ chr(aa+ord('A')) for aa in range(ord('L')-ord('A')+1)]
bulb_name = [ x for x in pos_df.index.to_list() if x not in sec]+['BB1']
exclude1 = ['1test','2L_B','2R_F','2R_B','2W']
exclude2 = ['2W','2R_B','2R_F','2L_B','2L_F','1test']
calib_df = pd.read_csv(os.path.join(path,'calibration_real.csv'),index_col = 0)
lux_dist_df = pd.read_csv(os.path.join(path,'Lux_to_distance.csv'),index_col =0)
sensor_all = sorted(calib_df.index.tolist())
sensor_all.remove('test_lid')
Global_lux_dict = {}
def Get_raw(Date_start,Date_end):
    global calib_df,sensor_all
    Get_lux_url = "http://led.incku.com:8000/lux_all_download"
    Get_bulb_url = "http://led.incku.com:8000/bulb_sec_download"
    lux_query = {'time_start':Date_start, 'time_end':Date_end,'data_req':1}
    lux_raw = requests.get(Get_lux_url,params=lux_query).json()
    bulb_query = {'time_start':Date_start, 'time_end':Date_end,'data_req':1}
    bulb_raw = requests.get(Get_bulb_url,params=bulb_query).json()
    lux_df = pd.DataFrame.from_dict(lux_raw['data'],orient='index').sort_index().fillna(method="ffill")
    lux_df = lux_df.apply(lambda x: x/calib_df.loc[x.name,'slope'] if x.name in sensor_all else x)
    bulb_df = pd.DataFrame.from_dict(bulb_raw['data']['1'],orient='index').sort_index()#.drop(columns='template')
    return lux_raw,lux_df,bulb_raw,bulb_df

def Position_data(bulb_df,lux_df,seconds,exclude,draw=False):
    global sec,bulb_name,pos_df,sensor_all
    area_all = [x for x in sensor_all if x not in exclude]
    # bulb_name = pos_df.index.to_list()
    bulb_df['status'] = bulb_df[bulb_name].apply(lambda x: x.sum(), axis=1)
    bulb_df['code'] = bulb_df[bulb_name+['status']].apply(lambda x: list(x.keys()[np.where(x==x['status'])[0]])[0], axis=1)
    bulb_df = bulb_df[['code','status']]#.drop(columns=bulb_name+['BB1','date','time']+sec)
    
    pos_end = bulb_df[bulb_df['code']=='status'].iloc[0].name
    print(pos_end)
    pos_end = bulb_df.loc[:pos_end].iloc[-2].name
    data = pd.merge(lux_df.loc[:pos_end],bulb_df.loc[:pos_end],left_index=True, 
                         right_index=True,how='outer',suffixes=('_left', '_right')).sort_index()
    data['status'] = data['status'].fillna(method="ffill")
    data['code'] = data['code'].fillna(method="ffill")
    data.dropna(subset=['code','status'],inplace=True)
    data.loc[data['code']=='A','code'] = 'ALL'

    ## Drop data after control 8 seconds
    index_list = []
    for date in bulb_df.index:
        tt = datetime.strptime(date, '%Y/%m/%d %H:%M:%S')
        start_time = (tt-timedelta(seconds=0)).strftime("%Y/%m/%d %H:%M:%S")
        end_time = (tt+timedelta(seconds=seconds)).strftime("%Y/%m/%d %H:%M:%S")
        index_list.extend(data[start_time:end_time].index)
    data.drop(index_list, axis=0, inplace=True)
    data = data.dropna()
    data_prepare = Data_mean(data,exclude,draw=draw)#pd.DataFrame(data_array)
    comb_st = {}
    for bb in bulb_name:
        tt = data_prepare[data_prepare['code']==bb].copy()
        tt.dropna(axis='columns',inplace=True)
        comb_st[bb] = tt.drop(columns=['code','status']).sum(axis=0)/2#len(tt)
    comb_st = pd.DataFrame.from_dict(comb_st,orient='index')
    return data_prepare,comb_st.drop('BB1')

def Data_mean(data,exclude,draw=True):
    global pos_df,sensor_all,bulb_name
    area_all = [aa for aa in sensor_all if aa not in exclude]
    data_array = []
    for bb in bulb_name:
        if draw:
            plt.figure(figsize=(16, 6))
            plt.clf()
        for status in [1,2,3]:
            if draw:
                plt.subplot(1,3,status)
                plt.ylim(0,250)
                ax = plt.gca()
                ax.set_title(bb+" status:"+str(status))
            temp_bulb = data[(data['code']==bb)&(data['status']==status)]
            # print(temp_bulb)
            temp_dict = {}
            for aa in area_all:
                tt = temp_bulb[aa]
                if status==2 or status==3:
                    vv = tt[(tt>0)&(tt<130)]
                else:
                    vv = tt[tt>0]
                # if bb=='E2' and status ==2 and aa == '2c': 
                #     vv=vv.iloc[-2:-1]
                if draw:
                    color = next(ax._get_lines.prop_cycler)['color']
                    tt[tt>0].plot(rot=10,style='o',color=color,label=aa)
                    plt.plot(np.arange(len(tt)),[vv.median()]*len(tt),'--',color=color, linewidth=1.5)
                if not vv.empty:
    #                 if vv.std()>1:
    #                     print('area',aa,vv.std())
                    temp_dict[aa] = vv.median()
            if temp_dict:
                temp_dict['status'] = status
                temp_dict['code'] = bb
                data_array.append(temp_dict)
        if draw:
            plt.show()
    return pd.DataFrame(data_array)#.drop('BB1')

def Spline_fit(lux_max,n_interior_knots,head,lux_dist_df=None,draw=False):
    #     print(lux_dist_df)
    if lux_dist_df is None:
        df = pd.read_csv(os.path.join(path,'lux_dist_final.csv'))
        df = pd.DataFrame({'x':[ float("{:0.4f}".format(x)) for x in df['lux']],
                           'y':(df['x']**2 +df['y']**2)**0.5})\
        .sort_values(by='x').reset_index(drop=True)
    else:
        df = pd.DataFrame({'x':[ float("{:0.4f}".format(x)) for x in lux_dist_df['lux']],
                           'y':(lux_dist_df['x']**2 +lux_dist_df['y']**2)**0.5})\
        .sort_values(by='x').reset_index(drop=True)
    for _, g in df.groupby(['x']):
        if len(g) > 1:
            index_list = g.index.tolist()
            df.loc[index_list[0]] = g.mean()
            df.drop(index_list[1:] , inplace=True)
    df = df.drop(df[df['x']>lux_max].index)
    df = pd.concat([df,pd.DataFrame([[lux_max,0]],columns=['x', 'y'])], ignore_index=True)
    
    df = pd.concat([pd.DataFrame([[0,15]],columns=['x', 'y']),df], ignore_index=True)
    #     print(df.head(5))
    ts = df['x']
    ys = df['y']
    qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
    knots = np.quantile(ts, qs)
    t = np.r_[(ts.values[0],)*(head+1),
              knots,
              (ts.values[-1],)*(head+1)]
    spl = make_lsq_spline(ts, ys, t, head)
    if draw:
        plt.figure(figsize=(12, 6))
        plt.plot(ys,ts,  '.c')
        plt.plot(spl(np.linspace(0, lux_max+10, 50)),np.linspace(0, lux_max+10, 50),  'g-', label='LSQ spline')
        plt.show()
    return spl
def triposition(data): 
    xa,ya,da,xb,yb,db,xc,yc,dc = data[0][0],data[0][1],data[0][2],\
                                    data[1][0],data[1][1],data[1][2],\
                                    data[2][0],data[2][1],data[2][2]
    x,y = sympy.symbols('x y')
    f1 = 2*x*(xa-xc)+np.square(xc)-np.square(xa)+2*y*(ya-yc)+np.square(yc)-np.square(ya)-(np.square(dc)-np.square(da))
    f2 = 2*x*(xb-xc)+np.square(xc)-np.square(xb)+2*y*(yb-yc)+np.square(yc)-np.square(yb)-(np.square(dc)-np.square(db))
    result = sympy.solve([f1,f2],[x,y])
    if len(result)==0:
        return []
    else:
        return [result[x],result[y]]
def plot_area(area_center,area_pos,alpha=0.5,style='default'):
    global pos_df
    color={0:'grey',0.5:'lightyellow',1:'yellow'}
    labels = {'grey': 'OFF','yellow':'ON','lightyellow':'HALF'}
    di = {0:0,1:1, 2: 0.5,3:0.5}
    fig = plt.figure(figsize=(9, 9))
    # plt.style.use('dark_background')
    plt.style.use(style)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('The Position of Sensor',fontsize=20,weight='bold')

    for index,row in pos_df.iterrows():
        st = 0
        ax.scatter(row['y'],row['x'],c=color[st],s=800,marker='s')
    #     ax.text(pos_dict[dd][1],pos_dict[dd][0], dd,color='white',ha="center", va="center")
    sns.scatterplot(data=area_pos, x="y", y="x", style="area", hue="area")
    for index,row in area_center.iterrows():
        ax.text(row['y'],row['x'], index,color='white',ha="center", va="center",weight='bold',fontsize=14)
    sns.scatterplot(data=area_center, x="y", y="x",s=800,alpha=alpha,hue="area",legend=False)#,style="area")

    legend = plt.legend(title="Sensors",loc='center right',fontsize = 12)
    title = legend.get_title()
    title.set_color("white")
    title.set_weight("bold")
    title.set_size(16)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    plt.xlim(-2,14)
    plt.ylim(12,-2)
    plt.show()
def area_center_cal(comb_st,spl,thres,outlier_thres,exclude=[]):
    global pos_df,sensor_all
    area_all = [aa for aa in sensor_all if aa not in exclude]
    area_data = {}
    for aa in area_all:   
        temp = comb_st[comb_st[aa]>thres][aa].to_frame()
        temp[['x','y']] = pos_df.loc[temp.index]
        dd = spl(temp[aa])
        dd[dd<0] = 0
        temp['dist'] = dd #if dd >0 else 0 #spl(temp[aa])
        temp.drop(columns=aa,inplace=True)
        area_data[aa]= temp.values
   
    area_pos = []
    for key in tqdm(area_data):
        vals = area_data[key]
        comb = combinations(np.arange(vals.shape[0]), 3)
        for i in list(comb):
            res = triposition(vals[i,:])
            if len(res)>0:
                area_pos.append({'area':key,'x':res[0],'y':res[1]})
    area_pos = pd.DataFrame(area_pos)
    area_pos = area_pos.drop(area_pos[(area_pos['x']>11.5) | (area_pos['x']<-1.5)
                           |(area_pos['y']>10.5) | (area_pos['y']<-1.5)].index,axis=0)
    area_center=[]
    outliers_list = []
    for aa in area_all:
        data = area_pos[area_pos['area']==aa]
        data = np.vstack((data.values[:,2],data.values[:,1])).T
        if len(data)>2:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2).fit(data)
            threshold = pd.Series(clusterer.outlier_scores_).quantile(outlier_thres)
            outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
            data = np.delete(data, outliers,axis=0)
        center = np.mean(data,axis=0)
        area_center.append({'area':aa,'x':float(center[1]),'y':float(center[0])})
    area_dict = pd.DataFrame(area_center).set_index('area')
    return area_dict,area_pos
def lux_dis_combined(data_prepare1,data_prepare2,area_pos1,area_pos2):
    lux_ = {}
    lux_table_st1 = lux_table_status(data_prepare1)
    lux_table_st2 = lux_table_status(data_prepare2)
    for st_ in range(3):
        lux_t1 = lux_to_distance(lux_table_st1[st_+1],area_pos1)
        lux_t2 = lux_to_distance(lux_table_st2[st_+1],area_pos2)
        lux_[st_+1] = pd.concat([lux_t1,lux_t2])
    return lux_[1],lux_[2],lux_[3]
def lux_table_status(data_prepare):
    global bulb_name
    lux_table_st = {}
    for st in [1,2,3]:
        temp_st ={}
        for bb in [x for x in bulb_name if x != 'BB1']:
            tt = data_prepare[(data_prepare['code']==bb)&(data_prepare['status']==st)].copy()
            tt.dropna(axis='columns',inplace=True)
            temp_st[bb] = tt.drop(columns=['code','status']).squeeze()
        temp_st = pd.DataFrame.from_dict(temp_st,orient='index')
        lux_table_st[st] = temp_st
    return lux_table_st

def lux_to_distance(comb_st,area_dict):
    global pos_df
    comb_st = comb_st[area_dict.index.tolist()]
    lux_dist = []
    for index,row in comb_st.iterrows():
        raw = pd.merge(row.dropna().to_frame(),area_dict,left_index=True,right_index=True,how='left')
        raw[['x','y']] = raw[['x','y']] - pos_df.loc[index].values#[index]
        lux_dist.extend(raw.values)
    lux_dist = np.array(lux_dist)
    lux_dist_df = pd.DataFrame({'x':lux_dist[:,1],'y':lux_dist[:,2],'lux':lux_dist[:,0]})
    return lux_dist_df


def kernel_lux(data_prepare1,data_prepare2,area_pos1,area_pos2,kernel_size = 10,filter_size = 3,delta=0.5,save=False):
    global Global_lux_dict
    lux1,lux2,lux3 = lux_dis_combined(data_prepare1,data_prepare2,area_pos1,area_pos2)
    #     lux_table_st = lux_table_status(data_prepare)
    #     lux1 = lux_to_distance(lux_table_st[1],area_dict)
    #     lux2 = lux_to_distance(lux_table_st[2],area_dict)
    #     lux3 = lux_to_distance(lux_table_st[3],area_dict)
    kernel_X, kernel_Y = np.meshgrid(np.arange(-kernel_size, kernel_size+delta, delta),
                         np.arange(-kernel_size, kernel_size+delta, delta))   
    delta_nn=0.01
    nn_X, nn_Y = np.meshgrid(np.arange(-kernel_size, kernel_size+delta_nn, delta_nn),
                             np.arange(-kernel_size, kernel_size+delta_nn, delta_nn))    
    lux_ = {}
    lux2.loc[:,'y'] = lux2['y'].abs()
    lux3.loc[:,'y'] = lux3['y'].abs()
    lux3.loc[:,'x'] = -lux3['x']
    lux_2 = pd.concat([lux2,lux3])
    lux_temp = lux_2.copy()
    lux_2.loc[:,'y'] = -lux_2['y']
    lux_2 = pd.concat([lux_temp,lux_2])
    lux_[2] = lux_2.copy()   
    lux_2.loc[:,'x'] = -lux_2['x']       
    lux_[3] = lux_2.copy()    
    lux1['x'] = lux1['x'].abs()
    lux1['y'] = lux1['y'].abs()
    temp_arr = lux1.values
    data_arr = np.r_[
        temp_arr, # ???????????? (x,y,lux)
        np.c_[-temp_arr[:,:2],temp_arr[:,2]] , # ???????????? (-x,-y,lux)
        np.c_[temp_arr[:,0],-temp_arr[:,1],temp_arr[:,2]] ,# ???????????? (x,-y,lux)
        np.c_[-temp_arr[:,0],temp_arr[:,1],temp_arr[:,2]] # ???????????? (-x,y,lux)
    ]
    lux_[1] = pd.DataFrame(data_arr,columns=['x','y','lux'])
    kernel_dict = {}
    for ii in range(3):
        lux_t = lux_[ii+1] 
        if save:
            lux_t.to_csv( os.path.join(cwd, 'lux_'+str(ii+1)+'_final.csv') ,index=False)
        Global_lux_dict[ii+1] = lux_t
        nn = griddata(lux_t[['x','y']].values,lux_t['lux'].values, (nn_X, nn_Y), method='nearest')
        nn = cv2.resize(nn, kernel_X.shape, interpolation=cv2.INTER_AREA)
        grid_ = cv2.GaussianBlur(nn, (filter_size,filter_size),0)
        kernel_dict[ii+1] = grid_
    return kernel_dict

def draw_kernel(Kernel,kernel_size = 10,delta=0.5,size=8,style='default'):
    kernel_X, kernel_Y = np.meshgrid(np.arange(-kernel_size, kernel_size+delta, delta),
                     np.arange(-kernel_size, kernel_size+delta, delta))    
    # plt.style.use('dark_background')
    plt.style.use(style)
    row_len = len( Kernel)
    fig = plt.figure(figsize=(size+4,size*row_len-4))
    
    for kk in Kernel.keys():
        # plt.gca().set_aspect('equal', adjustable='box')
        ax = fig.add_subplot(row_len, 2, 1+int((kk-1)*2), projection='3d')
        ax.set_title('Nearest interpolation + Gaussian Filter',fontsize=14,weight='bold')
        p=ax.plot_surface(kernel_Y, kernel_X, Kernel[kk], cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.8)
        # p=ax.plot_surface(X, Y, grid_1, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=1)
        # p=ax.plot_surface(X, Y, grid_z1, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.2)
        # p=ax.scatter(data_arr[:,0], data_arr[:,1], data_arr[:,2], c=data_arr[:,2], s=20,cmap=cm.coolwarm)
        ax.view_init(elev=30, azim=90)
        fig.colorbar(p,shrink=0.4,ax=ax)

        
        ax = fig.add_subplot(row_len, 2, 2+int((kk-1)*2))
        ax.set_title('status: '+str(kk),fontsize=14,weight='bold')
        contours = ax.contour(kernel_Y, kernel_X,  Kernel[kk], 40, cmap=cm.coolwarm)
        ax.set_aspect('equal', 'box')
        fig.colorbar(contours,shrink=0.4,ax=ax)
        ax.set_xlim(-10,10)
        ax.set_ylim(10,-10)
    plt.tight_layout()
    plt.show()
def Board_light(data,area_dict,sym_y =4.4,sym=True):
    temp_bulb = data[(data['code']=='BB1')&(data['status']==1)].reset_index(drop=True)

    bk_board = pd.merge(area_dict,temp_bulb.T.rename(columns={0:'lux'})
                    ,left_index=True,right_index=True,how='left')
    if sym:
        bk_board['y'] = (bk_board['y']-sym_y).abs()#+4
        bk_board2 = bk_board.copy()
        bk_board2['y'] = -bk_board2['y']+sym_y
        bk_board['y']  = bk_board['y'] +sym_y
        bk_board = pd.concat([bk_board,bk_board2])

    return bk_board
def board_to_grid(bk_board,smoothing=0.1,delta = 0.5,mn = [-1,-2],mx = [10,11]):
    new_X,new_Y = np.meshgrid(np.arange(mn[0], mx[0]+delta, delta), np.arange(mn[1], mx[1]+delta, delta))
    one_data = bk_board.values
    xflat = np.dstack((new_X.flatten(),new_Y.flatten()))[0]
    yflat = RBFInterpolator(one_data[:,:2], one_data[:,2:3],smoothing=smoothing)(xflat)
    ygrid = yflat.reshape(new_X.shape)
    BB1_df = pd.DataFrame({'x':new_X.flatten(),'y':new_Y.flatten(),
                             'lux':ygrid.flatten()})
    return BB1_df
def plot_board_light(bk_board,board_df,delta = 0.5,mn = [-1,-2],mx = [10,11] ,style='default'):
    new_X,new_Y = np.meshgrid(np.arange(mn[0], mx[0]+delta, delta), np.arange(mn[1], mx[1]+delta, delta))
    one_data = bk_board.values
    ygrid = board_df['lux'].values.reshape(new_X.shape)
    # plt.style.use('dark_background')
    plt.style.use(style)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(new_Y,new_X, ygrid, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5)
    p = ax.scatter(one_data[:,1],one_data[:,0],one_data[:,2],c=one_data[:,2], s=50,ec='k',cmap=cm.coolwarm)
    for dd in one_data:
        label = '%d' % (dd[2])
        ax.text(dd[1]+0.1, dd[0]+0.1, dd[2]+0.1, label,color='black')
    fig.colorbar(p,shrink=0.5)
    ax.view_init(elev=90, azim=-90)
    plt.ylim( mx[0],mn[0])
    plt.xlim(mn[1],mx[1])#mn[1], 
    ax = fig.add_subplot(1, 2, 2)
    contours = ax.contour(new_Y,new_X,  ygrid, 40, cmap=cm.coolwarm)
    fig.colorbar(contours,shrink=0.5)
    ax.set_aspect('equal', 'box')
    plt.ylim( mx[0],mn[0])
    plt.xlim(mn[1],mx[1])
    plt.show()
def f(sym_y,sym,smoothing):
    bk_board1 = Board_light(dark1,area_pos1[0],sym_y =sym_y,sym=sym)
    bk_board2 = Board_light(dark2,area_pos2[0],sym_y =sym_y,sym=sym)
    bk_board = pd.concat([bk_board1,bk_board2])
    #     bk_board[bk_board.index!='2c':,'lux'] = bk_board.drop('2c').apply(lambda x: x['lux']/calibration.loc[x.name,'slope'] ,axis=1)
    #     temp_df = pd.merge(bk_board,calibration,left_index=True,right_index=True,how = 'inner')
    #     temp_df['lux'] = temp_df['lux']/temp_df['slope']
    #     temp_df[['x','y','lux']]
    board_df = board_to_grid(bk_board[['x','y','lux']],smoothing)
    plot_board_light(bk_board,board_df)
    return sym_y
def Board_field(dark1,dark2,area_pos1,area_pos2,sym_y,sym,smoothing):
    bk_board1 = Board_light(dark1,area_pos1,sym_y =sym_y,sym=sym)
    bk_board2 = Board_light(dark2,area_pos2,sym_y =sym_y,sym=sym)
    bk_board = pd.concat([bk_board1,bk_board2])
    #     bk_board[bk_board.index!='2c':,'lux'] = bk_board.drop('2c').apply(lambda x: x['lux']/calibration.loc[x.name,'slope'] ,axis=1)
    #     temp_df = pd.merge(bk_board,calibration,left_index=True,right_index=True,how = 'inner')
    #     temp_df['lux'] = temp_df['lux']/temp_df['slope']
    #     temp_df[['x','y','lux']]
    board_df = board_to_grid(bk_board[['x','y','lux']],smoothing)
    plot_board_light(bk_board,board_df)
    return bk_board
def generate_lux(template,modes,exclude,std = 5,data_len=3,draw=False):
    area_table = [aa for aa in sensor_all if aa not in exclude]
    plt.style.use('dark_background')
    outliers = []
    lux_mode = []
    for mm in modes:
        if draw:
            plt.figure(figsize=(8, 4))
            plt.clf()
            ax = plt.gca()
            ax.set_title("mode: "+str(mm))
        temp_bulb = template[template['template']==mm]
        temp_dict = {}
        for aa in area_table:
            tt = temp_bulb[aa]
            vv = tt[tt>0]
            if vv.max()-vv.min()>10:
                vv_score = abs((vv - vv.mean())/vv.std(ddof=0))
                vv = vv[vv_score[vv_score<1].index]
    #         plt.plot(np.arange(len(tt)+2),[vv.median()]*(len(tt)+2),'--',color=color, linewidth=1.5)
            if not vv.empty:
                if vv.std()<std and vv.shape[0]>=data_len:                
                    temp_dict[aa] = vv.median()
                    if draw:
                        color = next(ax._get_lines.prop_cycler)['color']
                        vv.plot(rot=10,style='o-',color=color,label=aa,ax=ax)
                        plt.plot(np.arange(len(tt)+2),[vv.median()]*(len(tt)+2),'--',color=color, linewidth=1.5)
                else:
                    outliers.append(tt[tt>0])
                    if draw:
                        color = next(ax._get_lines.prop_cycler)['color']
                        tt[tt>0].plot(rot=10,style='o',color=color)
    #         if not vv.empty:
    #             temp_dict[aa] = vv.median()
        if temp_dict:
            temp_dict['mode'] = mm
            lux_mode.append(temp_dict)
            
            if draw:
                plt.legend(loc='center right')
                plt.show()
            else:
                plt.close()
    return pd.DataFrame(lux_mode).drop(0),outliers
def convolution_evl2(lux_mode,template_bb,area_,delta=0.125,exclude_mode=None):
    global pos_rev
    error_dict = {}
    result_eval = []
    dark = Dark.Dark_model(delta=delta,kernel_size=10,mn=[-1,-1],mx=[11,11])
    if exclude_mode is not None:
        lux_mode = lux_mode.drop(lux_mode[lux_mode['mode'].isin(exclude_mode)].index)
    for mm in  tqdm(lux_mode['mode']):
        test_tmp = template_bb[template_bb['template']==mm].iloc[0]
        test_lux = lux_mode[lux_mode['mode']==mm].iloc[0].rename('lux').drop('mode').dropna()
        result = dark.bulb_conv(test_tmp,calibrate=False)
        empty_df2 = pd.DataFrame({'x':np.round( area_['x']/delta )*delta,
                                  'y':np.round( area_['y']/delta )*delta,
                                  'area':area_.index})
        # take the conv result on sample point
        sensors = pd.merge(empty_df2,result, on=['x','y'],how='left',suffixes=('_left', '_right'))
        error = pd.merge(test_lux,sensors.set_index('area')[['z']],how='left',left_index=True,right_index=True)
        error_dict[int(mm)] = error['lux']-error['z']
        error_df = pd.merge(error,area_.reset_index(),left_index=True,right_on='area', how='left')
        error_df['diff'] = error['lux'].values - error_df['z'].values
        error_df['abs_diff'] = abs(error['lux'].values - error_df['z'].values)
        error_df['mode'] = mm
        result_eval.extend(error_df.rename(columns={'z':'pred'}).to_dict('records'))
        
    error_comb = pd.merge( pd.DataFrame.from_dict(result_eval) , dark.test_modes,left_on = 'mode',right_on = 'id',how='left')
    error_df = pd.DataFrame.from_dict(error_dict,orient='index')
    print('MAE: {:.2f},'.format(mean(error_df.dropna(how='all', axis=1).abs().mean())),
      'Std: {:.2f},'.format(mean(error_df.dropna(how='all', axis=1).std())),
      'Max: {:.2f},'.format(max(error_df.dropna(how='all', axis=1).max())))
    return error_comb,error_df
def mode_outlier(error_comb,g_thresh = 50,max_thresh = 200):
    group_err = {}
    for index,group in error_comb.groupby(by='mode'):
        group_err[index] = {'MAE':group['abs_diff'].sum(),'ME':group['diff'].sum(),'Max':group['abs_diff'].max(),
                             'Std':group['diff'].std(),'Lux':group['lux'].mean(),'Pred':group['pred'].mean()}
    group_err = pd.DataFrame(group_err).T.sort_values(by='MAE',ascending=False)
    x = group_err[group_err['Max']<max_thresh]['Lux'].values.reshape((-1, 1))
    y = group_err[group_err['Max']<max_thresh]['MAE'].values
    model = LinearRegression(fit_intercept=False).fit(x, y) 
    x = group_err['Lux'].values.reshape((-1, 1))
    y = group_err['MAE'].values
    group_err['Outlier'] = abs(y-model.predict(x))
    group_err['Pred'] = model.predict(x)
    group_err['Tresh'] = np.ones(len(x))*g_thresh
    mode_exclude = sorted([int(mm) for mm in group_err[group_err['Outlier']>g_thresh].index])
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16,8))
    sns.lineplot(data= group_err,x='Lux',y='Outlier',color='green')
    #     sns.lineplot(data= group_err,x='Lux',y='Tresh',color='black')
    sns.lineplot(data= group_err,x='Lux',y='Pred',color='red')
    sns.lineplot(data= group_err,x='Lux',y='MAE',color='black')
    # sns.lineplot(data= group_err,x='Lux',y='Std',color='black')
    # sns.lineplot(data= group_err,x='Lux',y='Outlier',color='green')
    sns.scatterplot(data= group_err.loc[mode_exclude],x='Lux',y='Outlier',color='red',s=100)
    #     sns.scatterplot(data= group_err.loc[mode_exclude],x='Lux',y='MAE',color='grey',s=200)
    plt.show()
    return mode_exclude
def space_err(error_df,error_comb,area_pos):
    err = pd.DataFrame({'MAE':error_df.abs().mean().sort_values(ascending=False),
     'ME':error_df.mean().sort_values(ascending=False),
     'STD':error_df.abs().std().sort_values(ascending=False)})
    err = pd.merge(err,area_pos,left_index=True,right_index=True,how='left')
    rev_dict = {}
    for aa in error_df.columns:
        x = error_comb[error_comb['area']==aa]['pred'].values.reshape((-1, 1))
        y = error_comb[error_comb['area']==aa]['lux'].values
        model = LinearRegression(fit_intercept=False).fit(x, y) 
        #     r_sq = model.score(x, y)
        x = error_comb[error_comb['area']==aa]['lux'].values.reshape((-1, 1))
        y = error_comb[error_comb['area']==aa]['diff'].values
        r_sq = LinearRegression(fit_intercept=False).fit(x, y).score(x, y)
        rev_dict[aa] = {'coeff':abs(r_sq),'slope':1+(model.coef_[0]-1)*0.5}
        rev_dict[aa] = {'coeff':abs(r_sq),'slope':model.coef_[0]}
    rev_df = pd.DataFrame(rev_dict).T.sort_values(by='coeff')

    spec = pd.merge(rev_df,area_pos,left_index=True,right_index=True,how='left')

    spec['dist'] = ((spec['x']-5)**2 + (spec['y']-4.5)**2)**0.5

    return pd.merge(spec, err,on=['x','y'],how='left')
def plot_space_err(data,weight=True):
    #     data =  pd.merge(pd.concat([spec1,spec2]), pd.concat([err1,err2]),on=['x','y'],how='left')
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    p = ax.scatter(data['y'],data['x'], data['slope'],c=( data['weight']),
                   s=data['weight']*100,ec='k',cmap='hot')   
    # ax.scatter(spec1['y'],spec1['x'], spec1['slope'],c=( spec1['MAE']), s=200,ec='k',cmap=cm.coolwarm)
    # sns.scatterplot(data=pd.concat([spec1,spec2]),x="x", y="y", hue="slope",size='slope',s=400)
    fig.colorbar(p,shrink=0.5)
    plt.ylim(12,-2)
    plt.xlim(-2,11)
    ax.view_init(elev=80, azim=-90)
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    p = ax.scatter(data['y'],data['x'], data['slope'],c=( data['slope']),
                       s=data['slope']*100,cmap='hot')
    # ax.scatter(spec1['y'],spec1['x'], spec1['slope'],c=( spec1['MAE']), s=200,ec='k',cmap=cm.coolwarm)
    # sns.scatterplot(data=pd.concat([spec1,spec2]),x="x", y="y", hue="slope",size='slope',s=400)
    fig.colorbar(p,shrink=0.5)
    plt.ylim(12,-2)
    plt.xlim(-2,11)
    ax.view_init(elev=80, azim=-90)
    plt.show()
def field_calibration(data_test,weight=True,filter_size = 5,smoothing=1, delta = 0.5, mn = [-1,-2], mx = [10,11]):
    data2 = []
    if weight:        
        for xx in data_test[['x','y','slope','weight']].values:
            if int(xx[3]):
                temp = np.array(xx[:3].tolist()*int(xx[3])).reshape(int(xx[3]),-1)
                data2.extend(temp)

        data2 = pd.DataFrame(np.array(data2),columns=['x','y','slope'])
    else:
        data2 = data_test[['x','y','slope']]
    new_X,new_Y = np.meshgrid(np.arange(mn[0], mx[0]+delta, delta), np.arange(mn[1], mx[1]+delta, delta))
    y_grid,model_rbf = build_space_calibration(new_X,new_Y,smoothing=smoothing,filter_size=filter_size,data2=data2)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(new_Y,new_X, y_grid, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5)
    p = ax.scatter(data2['y'],data2['x'],data2['slope'],c=data2['slope'], s=50,ec='k',cmap=cm.coolwarm)
    for dd in data2[['y','x','slope']].values:
        label = '{0:.2f}'.format(dd[2])
        ax.text(dd[0]+0.1, dd[1]+0.1, dd[2], label,color='black')
    fig.colorbar(p,shrink=0.5)
    ax.view_init(elev=90, azim=-90)
    plt.ylim( mx[0],mn[0])
    plt.xlim(mn[1],mx[1])#mn[1], 
    ax = fig.add_subplot(1, 2, 2)
    contours = ax.contour(new_Y,new_X,  y_grid, 40, cmap=cm.coolwarm)
    fig.colorbar(contours,shrink=0.5)
    ax.set_aspect('equal', 'box')
    plt.ylim( mx[0],mn[0])
    plt.xlim(mn[1],mx[1])
    plt.show()
    return model_rbf
def build_space_calibration(new_X,new_Y,smoothing=0.6,filter_size=5,data2=None):
    if data2 is None:
        data2 = pd.read_csv(os.path.join(path,'Space_calibration.csv'))
    #     new_X,new_Y = np.meshgrid(np.arange(mn[0], mx[0]+delta, delta), np.arange(mn[1], mx[1]+delta, delta))
    xflat = np.dstack((new_X.flatten(),new_Y.flatten()))[0]
    model_rbf = RBFInterpolator(data2[['x','y']].values,data2['slope'].values.reshape(-1,1),smoothing=smoothing)
    yflat = model_rbf(xflat)
    y_grid = yflat.reshape(new_X.shape)
    y_grid = cv2.GaussianBlur(y_grid, (filter_size,filter_size),0)
    return y_grid,model_rbf

def plot_calibrate(error_comb):
    size = 4
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16,8*size))
    ax = fig.add_subplot(size, 1, 1)
    title = "(AE: {0:.2f}".format(error_comb['diff'].mean()) +" MAE: {0:.2f}".format(error_comb['abs_diff'].mean())+\
    " Std: {0:.2f}".format(error_comb['diff'].std())+" Max: {0:.2f})".format(error_comb['abs_diff'].max())
    ax.set_title('Error before calibration '+title,fontsize=14)
    table = pd.plotting.table(ax=ax,
    data=  error_comb[['area','diff','abs_diff']].rename(columns={'diff':'Avg Error','abs_diff':'Mean Abs Error'})\
    .set_index('area').reset_index().groupby('area').mean()\
    .applymap(lambda x: float('{:,.2f}'.format(x))).sort_values(by='Mean Abs Error',ascending=False).head(5),
                     cellLoc = 'center', rowLoc = 'center',colWidths = [0.08]*2,loc='upper left')
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # may help
    scatter = sns.scatterplot(data= error_comb,x='lux',y='diff',style='area',hue="area")#,markers=True, dashes=False)
    scatter.set_ylim(bottom=-150, top=150)
    ax = fig.add_subplot(size, 1, 2)
    title = "(AE: {0:.2f}".format(error_comb['diff2'].mean()) +" MAE: {0:.2f}".format(error_comb['abs_diff2'].mean())+\
    " Std: {0:.2f}".format(error_comb['diff2'].std())+" Max: {0:.2f})".format(error_comb['abs_diff2'].max())
    ax.set_title('Error after calibration ' +title,fontsize=14)
    table = pd.plotting.table(ax=ax,
    data=  error_comb[['area','diff2','abs_diff2']].rename(columns={'diff2':'Avg Error','abs_diff2':'Mean Abs Error'})\
    .set_index('area').reset_index().groupby('area').mean().applymap(lambda x: float('{:,.2f}'.format(x)))\
    .sort_values(by='Mean Abs Error',ascending=False).head(5),
                     cellLoc = 'center', rowLoc = 'center',colWidths = [0.08]*2,loc='lower center')
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # may help
    scatter = sns.scatterplot(data= error_comb,x='lux',y='diff2',style='area',hue="area")#,markers=True, dashes=False)
    scatter.set_ylim(bottom=-100, top=100)
    
    ax = fig.add_subplot(size, 1, 3)
    ax.set_title('Prediction vs Sensor lux values',fontsize=14)
    sns.lineplot(data=error_comb,x="lux", y="pred2", style="area", hue="area",markers=True, dashes=False)
    ax = fig.add_subplot(size, 1, 4)
    ax.set_title('Prediction Error count',fontsize=14)
    his = sns.histplot(data=error_comb, x="abs_diff2", log_scale=False)
    his.set_xlim(left=0, right=100)
    ax.set_xlabel('Prediction Error [lux]')
    plt.show()
def plot_calibrate2(error_comb):
    size = 3
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16,8*size))
    ax = fig.add_subplot(size, 1, 1)
    title = "(AE: {0:.2f}".format(error_comb['diff'].mean()) +" MAE: {0:.2f}".format(error_comb['abs_diff'].mean())+\
    " Std: {0:.2f}".format(error_comb['diff'].std())+" Max: {0:.2f})".format(error_comb['abs_diff'].max())
    ax.set_title('Error'+title,fontsize=14)
    table = pd.plotting.table(ax=ax,
    data=  error_comb[['area','diff','abs_diff']].rename(columns={'diff':'Avg Error','abs_diff':'Mean Abs Error'})\
    .set_index('area').reset_index().groupby('area').mean()\
    .applymap(lambda x: float('{:,.2f}'.format(x))).sort_values(by='Mean Abs Error',ascending=False).head(5),
                     cellLoc = 'center', rowLoc = 'center',colWidths = [0.08]*2,loc='upper left')
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # may help
    scatter = sns.scatterplot(data= error_comb,x='lux',y='diff',style='area',hue="area")#,markers=True, dashes=False)
    scatter.set_ylim(bottom=-150, top=150)
    
    #     ax = fig.add_subplot(size, 1, 2)
    #     title = "(AE: {0:.2f}".format(error_comb['diff2'].mean()) +" MAE: {0:.2f}".format(error_comb['abs_diff2'].mean())+\
    #     " Std: {0:.2f}".format(error_comb['diff2'].std())+" Max: {0:.2f})".format(error_comb['abs_diff2'].max())
    #     ax.set_title('Error after calibration ' +title,fontsize=14)
    #     table = pd.plotting.table(ax=ax,
    #     data=  error_comb[['area','diff2','abs_diff2']].rename(columns={'diff2':'Avg Error','abs_diff2':'Mean Abs Error'})\
    #     .set_index('area').reset_index().groupby('area').mean().applymap(lambda x: float('{:,.2f}'.format(x)))\
    #     .sort_values(by='Mean Abs Error',ascending=False).head(5),
    #                      cellLoc = 'center', rowLoc = 'center',colWidths = [0.08]*2,loc='lower center')
    #     table.set_fontsize(12)
    #     table.scale(1.5, 1.5)  # may help
    #     scatter = sns.scatterplot(data= error_comb,x='lux',y='diff2',style='area',hue="area")#,markers=True, dashes=False)
    #     scatter.set_ylim(bottom=-100, top=100)
        
    ax = fig.add_subplot(size, 1, 2)
    ax.set_title('Prediction vs Sensor lux values',fontsize=14)
    sns.lineplot(data=error_comb,x="lux", y="pred", style="area", hue="area",markers=True, dashes=False)
    ax = fig.add_subplot(size, 1, 3)
    ax.set_title('Prediction Error count',fontsize=14)
    his = sns.histplot(data=error_comb, x="abs_diff", log_scale=False)
    his.set_xlim(left=0, right=100)
    ax.set_xlabel('Prediction Error [lux]')
    plt.show()
def convolution_final(lux_mode,template_bb,area_,samples= 5,delta=0.25,modes_list=None,draw=False,accurate=False):
    error_dict = {}
    result_eval = []
    global pos_df
    dark = Dark.Dark_model(delta=delta,kernel_size=10,mn=[-1,-1],mx=[11,12])
    if samples=='ALL':
        modes_list= [int(ii) for ii in set(lux_mode['mode'].values)]
    #         print(modes_list)
    elif modes_list is None:
        modes_list = np.random.choice(lux_mode['mode'], samples,replace=False)
    for mm in  modes_list:#
        test_tmp = template_bb[template_bb['template']==mm].iloc[0]
        test_lux = lux_mode[lux_mode['mode']==mm].iloc[0].rename('lux').drop('mode').dropna()
        if accurate:
            result = dark.bulb_conv(test_tmp)
        else:
            result = dark.fast_conv(test_tmp.drop('template'))
        sensors = dark.result_in_points(result,area_)
        
        error = pd.merge(test_lux,sensors.set_index('area')[['z']],how='left',left_index=True,right_index=True)
    #         print(error)
        error_dict[int(mm)] = error['lux']-error['z']
        error_df = pd.merge(error,area_.reset_index(),left_index=True,right_on='area', how='left')
        error_df['diff'] = error['lux'].values - error_df['z'].values
        error_df['abs_diff'] = abs(error['lux'].values - error_df['z'].values)
        error_df['mode'] = mm
        result_eval.extend(error_df.rename(columns={'z':'pred'}).to_dict('records'))
        if draw:
            plot_template(sensors,result,pd.merge(pos_df,test_tmp.rename('lux'),
                     how='inner',left_index=True,right_index=True),error_df,dark.new_Y, dark.new_X)
    error_comb = pd.merge( pd.DataFrame.from_dict(result_eval) , dark.test_modes,left_on = 'mode',right_on = 'id',how='left')
    error_df = pd.DataFrame.from_dict(error_dict,orient='index')
    print('MAE: {:.2f},'.format(mean(error_df.abs().mean())),
      'Std: {:.2f},'.format(mean(error_df.std())),
      'Max: {:.2f},'.format(max(error_df.abs().max())))
    return error_comb,error_df

def plot_template(sensors,lighting,bulb_lux,error_df,new_Y, new_X,back_alpha=0.5,style='default'):
    color={0:'grey',0.5:'lightyellow',1:'yellow'}
    labels = {'grey': 'OFF','yellow':'ON','lightyellow':'HALF'}
    st_dict = {1:1,2:0.5,3:0.5,0:0}
    plt.style.use(style)
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(sensors['y'],sensors['x'], sensors['z'],c=( sensors['z']), s=200,ec='k',cmap=cm.coolwarm)
    for index,row in sensors.iterrows():
        ax.text(row['y'],row['x'], row['z']+50,'{:.1f}'.format(row['z']),color='black',fontsize=12,alpha=back_alpha)#row['area']+

    p=ax.plot_surface(new_Y, new_X, lighting['z'].values.reshape(new_X.shape),
                      cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.6)
    fig.colorbar(p,shrink=0.5)
    plt.ylim(10,-1)
    plt.xlim(-2,11)
    ax.view_init(elev=80, azim=-90)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_facecolor('black')
    ax.set_title('Error',fontsize=20,weight='bold')
    for vv in bulb_lux.values:
        ax.scatter(vv[1],vv[0],c=color[st_dict[int(vv[2])]],s=800,marker='s',alpha=back_alpha)
    for vv in pos_df.loc[sec].values:
        ax.scatter(vv[1],vv[0],c=color[0],s=800,marker='s',alpha=back_alpha)

    sns.scatterplot(data=error_df, x="y", y="x",color='green',s=800,legend=False,palette="icefire",alpha=0.5)
    for index,row in error_df.iterrows():
        ax.text(row['y'],row['x'], row['area'],color='white',ha="center", va="center"
                ,weight='bold',fontsize=14)
        ax.text(row['y'],row['x']-0.6, '{:.1f}'.format(row['z']),color='white',ha="center", va="center"
                ,weight='bold',fontsize=14)
        ax.text(row['y'],row['x']+0.6, '{:.1f}'.format(row['lux']),color='white',ha="center", va="center"
                ,weight='bold',fontsize=14)
        ax.text(row['y']+1,row['x'], '{:.1f}'.format(row['diff']),
                color='red',ha="center", va="center",weight='bold',fontsize=16)
    ax.grid(False)
    ax.set_aspect('equal', 'box')
    plt.ylim(12,-2)
    plt.xlim(-2,11)
    plt.show()