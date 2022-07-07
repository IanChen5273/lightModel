import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.interpolate import RBFInterpolator
# test_modes = pd.read_csv('Test modes.csv', index_col=0)
# template_bb = pd.read_csv('Test Template.csv', index_col=0)
# pos_df = pd.read_csv('Area.csv', index_col=0)
import os
cwd = os.getcwd()
path = os.path.join(cwd, "lightModel", "Dark")

def build_kernel(kernel_X,kernel_Y,kernel_size,filter_size = 3):
    delta_nn=0.05
    nn_X, nn_Y = np.meshgrid(np.arange(-kernel_size, kernel_size+delta_nn, delta_nn),
                             np.arange(-kernel_size, kernel_size+delta_nn, delta_nn))
    kernel_dict = {}
    for ii in range(3):
#         lux_ = pd.read_csv('lux_'+str(ii+1)+'.csv') 
        lux_ = pd.read_csv(os.path.join(path,'lux_'+str(ii+1)+'_final.csv') ) 
        nn = griddata(lux_[['x','y']].values,lux_['lux'].values, (nn_X, nn_Y), method='nearest')
        nn = cv2.resize(nn, kernel_X.shape, interpolation=cv2.INTER_AREA)
        grid_ = cv2.GaussianBlur(nn, (filter_size,filter_size),0)
        kernel_dict[ii+1] = grid_
    # kernel_dict= {1:grid_1,2:grid_2,3:grid_3}
    return kernel_dict


# def build_field(new_X,new_Y):
#     one_data = pd.read_csv('field.csv').values
#     xflat = np.dstack((new_X.flatten(),new_Y.flatten()))[0]
#     yflat = RBFInterpolator(one_data[:,:2], one_data[:,2:3])(xflat)
#     ygrid = yflat.reshape(new_X.shape)
#     BB1_df = pd.DataFrame({'x':new_X.flatten(),'y':new_Y.flatten(),
#                              'z':ygrid.flatten()})
#     return BB1_df

def build_field(new_X,new_Y,smoothing=1):
#     new_X,new_Y = np.meshgrid(np.arange(mn[0], mx[0]+delta, delta), np.arange(mn[1], mx[1]+delta, delta))
    one_data = pd.read_csv(os.path.join(path,'field_final.csv')).values
    xflat = np.dstack((new_X.flatten(),new_Y.flatten()))[0]
    yflat = RBFInterpolator(one_data[:,:2], one_data[:,2:3],smoothing=smoothing)(xflat)
    ygrid = yflat.reshape(new_X.shape)
    BB1_df = pd.DataFrame({'x':new_X.flatten(),'y':new_Y.flatten(),
                             'z':ygrid.flatten()})
    return BB1_df

def build_space_calibration(new_X,new_Y,smoothing=0.1,filter_size=5):
    data2 = pd.read_csv(os.path.join(path,'Space_calibration2.csv'))
    xflat = np.dstack((new_X.flatten(),new_Y.flatten()))[0]
    model_rbf = RBFInterpolator(data2[['x','y']].values,data2['slope'].values.reshape(-1,1),smoothing=smoothing)
    yflat = model_rbf(xflat)
    y_grid = yflat.reshape(new_X.shape)
    y_grid = cv2.GaussianBlur(y_grid, (filter_size,filter_size),0)
    return y_grid
    

def build_conv_arr(init_st,pos_df,new_X,new_Y,kernel_X,kernel_Y,kernel_1,BB1_df,calibrate):
    df_ = pd.merge(init_st,pos_df,how='left',left_index=True,right_index=True).drop(columns='status')
    conv_arr = []
    for index,row in df_.iterrows():
        if index !='BB1':
            empty_df = pd.DataFrame({'x':new_X.flatten(),'y':new_Y.flatten()})
            new_df = pd.DataFrame({
                'x':(kernel_X+row['x']).flatten(),
                'y':(kernel_Y+row['y']).flatten(),
                'z':kernel_1.flatten()
            })
            temp_df = pd.merge(empty_df,new_df, on=['x','y'],how='left').fillna(0)
        else:
            temp_df = BB1_df
        temp_df['z'] = temp_df['z']*calibrate.flatten()
        conv_arr.append(temp_df['z'].values)
    conv_arr = np.array(conv_arr)
    return conv_arr

class Dark_model:
    def __init__ (self,delta,kernel_size,filter_size = 3,mn = [-1,-2],mx = [10,11]):
        self.pos_df =  pd.read_csv(os.path.join(path,'Area.csv'), index_col=0)
        self.template_bb = pd.read_csv(os.path.join(path,'Test Template.csv'), index_col=0)
        self.test_modes = pd.read_csv(os.path.join(path,'Test modes.csv'), index_col=0)
        self.delta = delta
        self.kernel_size = kernel_size
        self.kernel_X, self.kernel_Y = np.meshgrid(np.arange(-kernel_size, kernel_size+delta, delta),
                                 np.arange(-kernel_size, kernel_size+delta, delta))
        self.mn = mn
        self.mx = mx
        self.new_X,self.new_Y = np.meshgrid(np.arange(mn[0], mx[0]+delta, delta), np.arange(mn[1], mx[1]+delta, delta))
        self.BB1_df = build_field(self.new_X,self.new_Y)
        self.calibrate = build_space_calibration(self.new_X,self.new_Y,smoothing=0.6,filter_size=7)
        self.kernel_lux = build_kernel(self.kernel_X,self.kernel_Y,kernel_size,filter_size)
        self.filter_size = filter_size
        self.empty_status = pd.read_csv(os.path.join(path,'initial_status.csv'),index_col=0)
        bulb_sector = []
        sec_name = ['1', '2', '3', '4']
        self.bulb_name = []
        for i in range(ord('A'), ord('L')+1):
            bulb_sector.append(chr(i))
        self.bulb_name.extend(bulb_sector)
        for i in range(12):
            for k in range(4):
                self.bulb_name.append(bulb_sector[i]+sec_name[k])
        self.conv_arr = build_conv_arr(self.empty_status ,self.pos_df ,self.new_X,self.new_Y,
                                       self.kernel_X,self.kernel_Y,self.kernel_lux[1],self.BB1_df,self.calibrate)
    def draw_kernel(self,size=8):
        plt.style.use('seaborn')
        row_len = len(self.kernel_lux)
        fig = plt.figure(figsize=(size+4,size*row_len-4))
        for kk in self.kernel_lux.keys():
            ax = fig.add_subplot(row_len, 2, 1+int((kk-1)*2), projection='3d')
            ax.set_title('Linear Interpolation',fontsize=12,weight='bold')
            p=ax.plot_surface(self.kernel_Y,self.kernel_X, self.kernel_lux[kk], cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.8)
            ax.view_init(elev=30, azim=90)
            fig.colorbar(p,shrink=0.4,ax=ax)
            ax = fig.add_subplot(row_len, 2, 2+int((kk-1)*2))
            ax.set_title('status: '+str(kk),fontsize=12)
            contours = ax.contour(self.kernel_Y, self.kernel_X,  self.kernel_lux[kk], 20, cmap=cm.coolwarm)
            ax.set_aspect('equal', 'box')
            fig.colorbar(contours,shrink=0.4,ax=ax)
            ax.set_xlim(-self.kernel_size,self.kernel_size)
            ax.set_ylim(self.kernel_size,-self.kernel_size)
        plt.tight_layout()
        plt.show()

    def draw_field(self,size=8,dense = 15):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(int(size*2),size))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title('Field with board light on',fontsize=12,weight='bold')
        p=ax.plot_surface(self.new_Y,self.new_X, self.BB1_df['z'].values.reshape(self.new_X.shape),
                          cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.4)
        fig.colorbar(p,shrink=0.5)
        ax.view_init(elev=90, azim=-90)
        plt.ylim(self.mx[0],self.mn[0])
        plt.xlim(self.mn[1],self.mx[1])#mn[1], 
        ax = fig.add_subplot(1, 2, 2)
        contours = ax.contour(self.new_Y,self.new_X, self.BB1_df['z'].values.reshape(self.new_X.shape)
                              , dense, cmap=cm.coolwarm,linewidths = 1)
        ax.clabel(contours, inline=1, fontsize=12)
        fig.colorbar(contours,shrink=0.5)
        plt.ylim(self.mx[0],self.mn[0])
        plt.xlim(self.mn[1],self.mx[1])
        ax.set_aspect('equal', 'box')
        plt.show()

    def plot_ligting(self,test_tmp,lighting,back_alpha=0.5,shrink = 1,target_lux = 1000,slicing=16):
        # print(lighting)
        title = '  Max: {:.2f}  '.format(lighting.max()['z'])+\
        'min: {:.2f}  '.format(lighting.min()['z'])+\
        'std: {:.2f}'.format(lighting.std()['z'])
        bulb_lux= pd.merge(self.pos_df,test_tmp.map({1:1,2:2,3:3,0:0,0.5:2}).rename('lux'),
             how='inner',left_index=True,right_index=True)
        color={0:'grey',0.5:'lightyellow',1:'yellow'}
        labels = {'grey': 'OFF','yellow':'ON','lightyellow':'HALF'}
        st_dict = {1:1,2:0.5,3:0.5,0:0}
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(int(16*shrink),int(8*shrink)))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title(title,fontsize=12,weight='bold')    
        p=ax.plot_surface(self.new_Y, self.new_X, lighting['z'].values.reshape(self.new_X.shape),
                          cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5)
        ax.plot_wireframe(self.new_Y[::slicing,::slicing], self.new_X[::slicing,::slicing], 
                          np.ones(self.new_X.shape)[::slicing,::slicing]*target_lux,color='black', antialiased=False,alpha=1)
        fig.colorbar(p,shrink=0.5)
        # plt.ylim(10,-1)
        # plt.xlim(-2,11)
        plt.ylim(self.mx[0],self.mn[0])
        plt.xlim(self.mn[1],self.mx[1])
        ax.view_init(elev=30, azim=-90)
        ax = fig.add_subplot(1, 2, 2)
        ax.set_facecolor('xkcd:black')
        ax.set_title('Result',fontsize=12,weight='bold')
        for vv in bulb_lux.values:
            ax.scatter(vv[1],vv[0],c=color[st_dict[int(vv[2])]],s=int(795*shrink**2),marker='s',alpha=back_alpha)
        for vv in self.pos_df.loc[self.bulb_name[:12]].values:
            ax.scatter(vv[1],vv[0],c=color[0],s=int(795*shrink**2),marker='s',alpha=back_alpha)
        contours = ax.contour(self.new_Y, self.new_X, lighting['z'].values.reshape(self.new_X.shape), 10, cmap=cm.coolwarm,linewidths = 1)
        ax.clabel(contours, inline=1, fontsize=10)
        ax.grid(False)
        ax.set_aspect('equal', 'box')
        plt.ylim(12,-2)
        plt.xlim(-2,11)
        # plt.ylim(self.mx[0],self.mn[0])
        # plt.xlim(self.mn[1],self.mx[1])
        plt.show()

    def bulb_conv(self,test_tmp,calibrate=True):
        bulb_lux= pd.merge(self.pos_df,test_tmp.rename('lux'),
                 how='inner',left_index=True,right_index=True)
        result_df = pd.DataFrame([])
        for bb in bulb_lux.values:
            if bb[2]!=0:
                new_df = pd.DataFrame({
                    'x':(self.kernel_X+bb[0]).flatten(),
                    'y':(self.kernel_Y+bb[1]).flatten(),
                    'z':self.kernel_lux[bb[2]].flatten()
                })
                if result_df.empty:
                    result_df = new_df
                else:
                    temp_df = pd.merge(result_df,new_df, on=['x','y'],how='outer',suffixes=('_left', '_right')).fillna(0)
                    temp_df['z'] = temp_df['z_left'] + temp_df['z_right']
                    result_df = temp_df.drop(['z_left','z_right'],axis=1)
        if not test_tmp['BB1']:       
            empty_df = pd.DataFrame({'x':self.new_X.flatten(),'y':self.new_Y.flatten(),
                                 'z':np.zeros(self.new_X.flatten().shape)})
        else:
            empty_df = self.BB1_df

        if not result_df.empty:
            temp_df = pd.merge(empty_df,result_df, on=['x','y'],how='left',suffixes=('_left', '_right')).fillna(0)
            temp_df['z'] = temp_df['z_left'] + temp_df['z_right']
            temp_df = temp_df.drop(['z_left','z_right'],axis=1)
        else:
            temp_df = empty_df
        # lighting = temp_df.copy()
        if calibrate:
            temp_df['z'] = temp_df['z']*self.calibrate.flatten()
#         temp_df*build_space_calibration(new_X,new_Y,smoothing=0.6,filter_size=7).flatten()
        return temp_df#lighting
    
    def fast_conv(self,test_tmp,out_type='DataFrame'):
#         print(test_tmp.shape)
        st_dict = {2:0.5,3:0.5,1:1,0:0,0.5:0.5}
        result = None
        if isinstance(test_tmp, pd.Series):
            result =  test_tmp.map(st_dict).values.reshape(1,49).dot(self.conv_arr)
            if out_type=='DataFrame':
                return pd.DataFrame({'x':self.new_X.flatten(),'y':self.new_Y.flatten(),'z':result.flatten()})
                
        elif isinstance(test_tmp, pd.DataFrame):
            result =  test_tmp.applymap(lambda a: st_dict[a]).values.reshape(-1,49).dot(self.conv_arr)
            if out_type=='DataFrame':
                return pd.concat([pd.DataFrame({'x':self.new_X.flatten(),'y':self.new_Y.flatten()}),pd.DataFrame(result.T)],axis=1)

        return result
    
    def result_in_points(self,result,area_):
        empty_df2 = pd.DataFrame({'x':np.round( area_['x']/self.delta )*self.delta,
                                      'y':np.round( area_['y']/self.delta )*self.delta,
                                      'area':area_.index})
        sensors = pd.merge(empty_df2,result, on=['x','y'],how='left',suffixes=('_left', '_right'))
        return sensors
    def test(self,plt_shrink=0.8):
        tmp_choice = np.random.choice(self.test_modes['id'].values, 1)[0]
        mode_info = self.test_modes.set_index('id').loc[tmp_choice,].to_frame().T
        # plot_table(mode_info)
        test_tmp = self.template_bb[self.template_bb['template']==tmp_choice].iloc[0]
        lighting = self.bulb_conv(test_tmp)
        self.plot_ligting(test_tmp,lighting,shrink=plt_shrink)
