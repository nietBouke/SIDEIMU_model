# =============================================================================
# Visualisation, salinity and river discharge 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def plot_new_compHO(self):
    # =============================================================================
    # Function to plot salt, discharge, stratification and the different processes 
    # in the network. Transport by tides not included. 
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,self.soc_sca)

    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(12,9))
    gs = fig.add_gridspec(11,14)

    ax_cen = fig.add_subplot(gs[3:8,3:9])
    ax_leg = fig.add_subplot(gs[0,-1])

    ax0 = fig.add_subplot(gs[0:2,3:6])
    ax1 = fig.add_subplot(gs[0:2,8:11])
    
    ax2 = fig.add_subplot(gs[3:5,0:3])
    ax3 = fig.add_subplot(gs[3:5,11:14])

    ax4 = fig.add_subplot(gs[6:8,0:3])
    ax5 = fig.add_subplot(gs[6:8,11:14])

    ax6 = fig.add_subplot(gs[9:11,3:6])
    ax7 = fig.add_subplot(gs[9:11,8:11])

    #plot salinity
    for key in self.ch_keys:
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(self.ch_outp[key]['sb'])
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
        
    #layout salinity plot
    ax_cen.axis('scaled')
    ax_cen.set_xlim(87.8,88.3) , ax_cen.set_ylim(21.6,22.3) #zoom in on mouth RM

    ax_cen.set_ylabel('degrees N ')
    ax_cen.set_xlabel('degrees E ')
    ax_cen.set_facecolor('grey')

    cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
    cb1.set_label(label=r'$\bar{s}$ [g/kg]')    

    #plot the salt intrusion limit
    for key in self.ch_keys:
        i0 = np.where(self.ch_outp[key]['sb']>2)[0]
        i1 = np.where(self.ch_outp[key]['sb']<2)[0]
        if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)


    #Plot the transports for every channel
    keys_now = ['Haldia' , 'Hooghly 4', 'Haldi Main', 'Hooghly 3', 'Haldi seaward', 'Mooriganga', 'Hooghly 1', 'Hooghly 2']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    for k in range(len(axes)):
        key = keys_now[k]
        axes[k].set_title(self.ch_gegs[key]['Name'])
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ'],label='$T_Q$',lw = 2,c='m')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TE'],label='$T_E$',lw = 2,c='brown')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TT'],label='$T_T$',lw = 2,c='red')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TD'],label='$T_D$',lw = 2,c='c')
        axes[k].plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ']+self.ch_outp[key]['TD']+self.ch_outp[key]['TT']+self.ch_outp[key]['TE'],label='$T_o$',lw = 2,c='navy')
                
        axes[k].grid()
        axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$T$ [kg/s]')
        axes[k].set_xlim(self.ch_outp[key]['px'][0]/1000,self.ch_outp[key]['px'][-1]/1000)
        

    #plt.xlim(-50,5)
    #ax0.legend()#,loc=2)
    #add the arrows
    ax_cen.arrow(87.88,22.12,-0.25,0.02,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(88.02,21.95,-0.43,-0.15,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(88.2,22.17,0.15,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(88,21.75,-0.07,-0.27,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(88.2,22,0.35,0.13,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(88.22,21.75,0.3,-0.02,clip_on = False,head_width=0.02,color='g',width=0.005)

    ax_cen.arrow(88.12,22.06,-0.22,0.14,clip_on = False,head_width=0,color='g',width=0.005)
    ax_cen.arrow(87.9,22.2,-0.02,0.15,clip_on = False,head_width=0.02,color='g',width=0.005)

    ax_cen.arrow(88.1,21.9,-0.02,-0.25,clip_on = False,head_width=0,color='g',width=0.005)
    ax_cen.arrow(88.08,21.65,0.2,-0.15,clip_on = False,head_width=0.02,color='g',width=0.005)
    

    clr,lab  = [] , []
    legs = ['$T_Q$','$T_E$','$T_T$','$T_D$','$T_O$']
    colors = ['m','brown','red','c', 'navy']
    for k in range(len(legs)):
        clr.append(Line2D([0], [0], color=colors[k], lw=2)) 
        lab.append(legs[k])
    
    ax_leg.legend(clr ,lab)
    ax_leg.axis('off')#,ax_ti.axis('off')


    plt.show()
    
'''
plot_new_compHO(delta)


#%%
plt.contourf(delta.ch_outp['Haldi seaward']['ss'].T,cmap='RdBu_r')
plt.colorbar()


#%%
plt.plot(delta.ch_outp['Haldi seaward']['ss'][0])
print(delta.ch_outp['Haldi seaward']['ss'].shape)

'''

