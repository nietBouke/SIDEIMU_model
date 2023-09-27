# =============================================================================
# Visualisation, salinity and river discharge 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.animation as ani       #make animations

def plot_new_compRM(self):
    # =============================================================================
    # Function to plot salt, discharge, stratification and the different processes 
    # in the network. Transport by tides not included. 
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,self.soc_sca)

    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(16,9))
    gs = fig.add_gridspec(11,12+2)

    ax_cen = fig.add_subplot(gs[4:7,4:9])
    ax_leg = fig.add_subplot(gs[0:3,-2])
    
    ax0 = fig.add_subplot(gs[0:3,1:4])
    ax1 = fig.add_subplot(gs[0:3,5:8])
    ax2 = fig.add_subplot(gs[0:3,9:12])
    
    ax3 = fig.add_subplot(gs[4:7,0:3])
    ax4 = fig.add_subplot(gs[4:7,10:13])

    ax5 = fig.add_subplot(gs[8:11,1:4])
    ax6 = fig.add_subplot(gs[8:11,5:8])
    ax7 = fig.add_subplot(gs[8:11,9:12])
    
    
    #plot salinity
    for key in self.ch_keys:
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(self.ch_outp[key]['sb'])
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
        
    #layout salinity plot
    ax_cen.axis('scaled')
    ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM

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
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
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
        
                        
        if keys_now[k] != 'Spui':     axes[k].invert_xaxis()
        #    print('hi')
    #plt.xlim(-50,5)
    #ax0.legend()#,loc=2)

    #add the arrows
    ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='g',width=0.005)
    
    ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='g',width=0.005)

    ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='g',width=0.005)

    ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='g',width=0.005)


    clr,lab  = [] , []
    legs = ['$T_Q$','$T_E$','$T_T$','$T_D$','$T_O$']
    colors = ['m','brown','red','c', 'navy']
    for k in range(len(legs)):
        clr.append(Line2D([0], [0], color=colors[k], lw=2)) 
        lab.append(legs[k])
    
    ax_leg.legend(clr ,lab)
    ax_leg.axis('off')#,ax_ti.axis('off')


    plt.show()
 
def plot_salt_compRM(self,plot_t):
    # =============================================================================
    # Function to plot salt, in the tidal cycle, at different locations
    # in the network. 
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,35)

    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(16,9))
    gs = fig.add_gridspec(11,12+2)
    

    ax_cen = fig.add_subplot(gs[4:7,4:9])
    #ax_leg = fig.add_subplot(gs[0:3,-2])
    
    ax0 = fig.add_subplot(gs[0:3,1:4])
    ax1 = fig.add_subplot(gs[0:3,5:8])
    ax2 = fig.add_subplot(gs[0:3,9:12])
    
    ax3 = fig.add_subplot(gs[4:7,0:3])
    ax4 = fig.add_subplot(gs[4:7,10:13])

    ax5 = fig.add_subplot(gs[8:11,1:4])
    ax6 = fig.add_subplot(gs[8:11,5:8])
    ax7 = fig.add_subplot(gs[8:11,9:12])
    
    
    #plot salinity
    for key in self.ch_keys:
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1))
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
        
    #layout salinity plot
    ax_cen.axis('scaled')
    ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM

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
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    for k in range(len(axes)):
        key = keys_now[k]
        s_here = self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t]
        
        axes[k].set_title(self.ch_gegs[key]['Name'])
        axes[k].contourf(self.ch_outp[key]['px']/1000 , self.z_nd, s_here.T,cmap = 'RdBu_r',levels = np.linspace(0,35,15))

        axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')

    #plt.xlim(-50,5)
    #ax0.legend()#,loc=2)

    #add the arrows
    ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='g',width=0.005)
    
    ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='g',width=0.005)

    ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='g',width=0.005)

    ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
    plt.show()
        
import matplotlib as mpl
       
def anim_new_compRM(self,savename):
    print('Warning: making of this animation is probably quite slow, find something to do in the meantime')
    
    # =============================================================================
    # Function to plot salt, discharge, stratification and the different processes 
    # in the network. Transport by tides not included. 
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,35)

    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(16,9))
    gs = fig.add_gridspec(11,12+2)
    ax_cen = fig.add_subplot(gs[4:7,4:9])
    #ax_leg = fig.add_subplot(gs[0:3,-2])
    
    ax0 = fig.add_subplot(gs[0:3,1:4])
    ax1 = fig.add_subplot(gs[0:3,5:8])
    ax2 = fig.add_subplot(gs[0:3,9:12])
    
    ax3 = fig.add_subplot(gs[4:7,0:3])
    ax4 = fig.add_subplot(gs[4:7,10:13])

    ax5 = fig.add_subplot(gs[8:11,1:4])
    ax6 = fig.add_subplot(gs[8:11,5:8])
    ax7 = fig.add_subplot(gs[8:11,9:12])
    
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    
    
    #make water level varying for tides
    coords_tides = {}
    
    for k in range(len(axes)):
        key = keys_now[k]

        #take water level into account for htis
        px_tide = np.tile(self.ch_outp[key]['px'],self.nz).reshape(self.nz,len(self.ch_outp[key]['px']))
        pz_tide = np.zeros((self.nz,len(self.ch_outp[key]['px']),self.nt)) + np.nan
        for t in range(self.nt):
            pz_1t = np.zeros((self.nz,len(self.ch_outp[key]['px']))) + np.nan
            for x in range(len(self.ch_outp[key]['px'])): pz_1t[:,x] = np.linspace(-self.ch_gegs[key]['H'],self.ch_outp[key]['eta_r'][x,t],self.nz)
            pz_tide[:,:,t] = pz_1t
                
        coords_tides[key] = (px_tide , pz_tide)

    def init():
        plot_t = 0
        ax_cen.cla(), ax0.cla(), ax1.cla(), ax2.cla(), ax3.cla(), ax4.cla(), ax5.cla(), ax6.cla(), ax7.cla()
        
        #overview plot
        for key in self.ch_keys:
            #plot salinity
            self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
            self.ch_outp[key]['lc1'].set_array(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1))
            self.ch_outp[key]['lc1'].set_linewidth(5)
            line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
            
            #plot the salt intrusion limit
            #subtidal 
            i0 = np.where(self.ch_outp[key]['sb']>2)[0]
            i1 = np.where(self.ch_outp[key]['sb']<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100, alpha = 0.5)
            #tidal
            i0 = np.where(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1)>2)[0]
            i1 = np.where(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1)<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)
        
            
        #layout overview salinity plot
        ax_cen.axis('scaled')
        ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM
        
        ax_cen.set_ylabel('degrees N ')
        ax_cen.set_xlabel('degrees E ')
        ax_cen.set_facecolor('grey')
        
        cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
        cb1.set_label(label=r'$\bar{s}$ [g/kg]')    
        
        #Plot the salinity in the tidal cycle in 2DV                
        for k in range(len(axes)):
            key = keys_now[k]
            s_here = self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t]
               
            cm = mpl.cm.get_cmap("RdBu_r").copy()
            a = axes[k].contourf(coords_tides[key][0]*1e-3,coords_tides[key][1][:,:,plot_t], s_here.T,cmap = cm,levels = np.linspace(0,35,15),extend='both')
            
            axes[k].set_title(self.ch_gegs[key]['Name'])
            axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')
            axes[k].set_ylim(-self.ch_gegs[key]['H'],np.max(self.ch_outp[key]['eta_r'])*1.25*2)
            axes[k].set_facecolor('lightgrey')
            if keys_now[k] != 'Spui': axes[k].invert_xaxis()
            a.cmap.set_under('white'), a.cmap.set_over('white')
            a.set_clim(0,35)

        #add the arrows
        ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='g',width=0.005)
        
        ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='g',width=0.005)


    def animate(plot_t):        
        ax_cen.cla(), ax0.cla(), ax1.cla(), ax2.cla(), ax3.cla(), ax4.cla(), ax5.cla(), ax6.cla(), ax7.cla()
        
        #overview plot
        for key in self.ch_keys:
            #plot salinity
            self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
            self.ch_outp[key]['lc1'].set_array(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1))
            self.ch_outp[key]['lc1'].set_linewidth(5)
            line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])
            
            #plot the salt intrusion limit
            #subtidal 
            i0 = np.where(self.ch_outp[key]['sb']>2)[0]
            i1 = np.where(self.ch_outp[key]['sb']<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100, alpha = 0.5)
            #tidal
            i0 = np.where(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1)>2)[0]
            i1 = np.where(np.mean(self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t] , 1)<2)[0]
            if len(i0)>1 and len(i1)>0: ax_cen.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)

            
        #layout overview salinity plot
        ax_cen.axis('scaled')
        ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM
    
        ax_cen.set_ylabel('degrees N ')
        ax_cen.set_xlabel('degrees E ')
        ax_cen.set_facecolor('grey')
    
        #cb1=plt.colorbar(line1, ax=ax_cen,orientation='vertical')
        #cb1.set_label(label=r'$\bar{s}$ [g/kg]')    

        #Plot the salinity in the tidal cycle in 2DV
        for k in range(len(axes)):
            key = keys_now[k]
            s_here = self.ch_outp[key]['ss']+self.ch_outp[key]['st_cor_r'][:,:,plot_t]

            cm = mpl.cm.get_cmap("RdBu_r").copy()
            a = axes[k].contourf(coords_tides[key][0]*1e-3,coords_tides[key][1][:,:,plot_t], s_here.T,cmap = cm,levels = np.linspace(0,35,15), extend="both")
            axes[k].set_title(self.ch_gegs[key]['Name'])
            axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')
            axes[k].set_ylim(-self.ch_gegs[key]['H'],np.max(self.ch_outp[key]['eta_r'])*1.25*2)
            axes[k].set_facecolor('lightgrey')
            if keys_now[k] != 'Spui': axes[k].invert_xaxis()
            a.cmap.set_under('white'), a.cmap.set_over('white')
            a.set_clim(0,35)

            
        #add the arrows
        ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='g',width=0.005)
        ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='g',width=0.005)
        
        ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
        ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='g',width=0.005)
        ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='g',width=0.005)


    anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
    #frames per second is now nt/11. should be determined from input but ja geen zin 
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=int(self.nt/11), extra_args=['-vcodec', 'libx264'])
    
    plt.show()

def plot_salt_compRM(self,var,title):
    # =============================================================================
    # Function to plot salt, in the tidal cycle, at different locations
    # in the network. 
    # =============================================================================
    keys_here = self.ch_keys    
    norm1 = plt.Normalize(0,30)

    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(16,9))
    gs = fig.add_gridspec(11,12+2)
    

    ax_cen = fig.add_subplot(gs[4:7,4:9])
    #ax_leg = fig.add_subplot(gs[0:3,-2])
    
    ax0 = fig.add_subplot(gs[0:3,1:4])
    ax1 = fig.add_subplot(gs[0:3,5:8])
    ax2 = fig.add_subplot(gs[0:3,9:12])
    
    ax3 = fig.add_subplot(gs[4:7,0:3])
    ax4 = fig.add_subplot(gs[4:7,10:13])

    ax5 = fig.add_subplot(gs[8:11,1:4])
    ax6 = fig.add_subplot(gs[8:11,5:8])
    ax7 = fig.add_subplot(gs[8:11,9:12])
    
    
    #plot salinity
    for key in self.ch_keys:
        if var == 'ss':  varplot = np.mean(self.ch_outp[key][var],1)
        elif var == 'st': varplot = np.abs(np.mean(self.ch_pars[key][var],1))
        elif var == 'stc': varplot = np.abs(np.mean(self.ch_outp[key]['st_cor']-self.ch_pars[key]['st'],1))
        else: print('Unkwon var')
        
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(varplot)
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax_cen.add_collection(self.ch_outp[key]['lc1'])

    #layout salinity plot
    ax_cen.axis('scaled')
    ax_cen.set_xlim(4,4.6) , ax_cen.set_ylim(51.75,52.02) #zoom in on mouth RM

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
    keys_now = ['Breeddiep' , 'Nieuwe Waterweg v2', 'Nieuwe Maas 1 old', 'Hartelkanaal v2', 'Hollandse IJssel', 'Oude Maas 2', 'Spui', 'Oude Maas 3']
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]
    for k in range(len(axes)):
        key = keys_now[k]
        if var == 'ss':  varplot = self.ch_outp[key][var]
        elif var == 'st': varplot = np.abs(self.ch_pars[key][var])
        elif var == 'stc': varplot = np.abs(self.ch_outp[key]['st_cor']-self.ch_pars[key]['st'])
        else: print('Unkwon var')
        
        axes[k].set_title(self.ch_gegs[key]['Name'])
        axes[k].contourf(self.ch_outp[key]['px']/1000 , self.z_nd, varplot.T,cmap = 'RdBu_r',levels = np.linspace(0,30,15))
        if keys_now[k] != 'Spui': axes[k].invert_xaxis()

        axes[k].set_xlabel('$x$ [km]'),axes[k].set_ylabel('$z/H$ [ ]')

    #plt.xlim(-50,5)
    #ax0.legend()#,loc=2)

    #add the arrows
    ax_cen.arrow(4.06,52,-0.1,0.1,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.25,51.94,0.01,0.14,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.47,51.93,0.2,0.22,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.15,51.91,-0.27,0.01,clip_on = False,head_width=0.02,color='g',width=0.005)
    ax_cen.arrow(4.35,51.8,0.03,-0.12,clip_on = False,head_width=0.02,color='g',width=0.005)
    
    ax_cen.arrow(4.34,51.84,-0.2,0,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.14,51.84,-0.15,-0.2,clip_on = False,head_width=0.02,color ='g',width=0.005)

    ax_cen.arrow(4.55,51.81,0,-0.15,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.55,51.66,0.2,-0.04,clip_on = False,head_width=0.02,color ='g',width=0.005)

    ax_cen.arrow(4.58,51.94,0,0.07,clip_on = False,head_width=0,color ='g',width=0.005)
    ax_cen.arrow(4.58,52.01,0.25,0.01,clip_on = False,head_width=0.02,color ='g',width=0.005)
    
    fig.suptitle(title,fontsize=20)
    plt.show()

#plot_salt_compRM(delta,'ss','Subtidal salinity')
#plot_salt_compRM(delta,'st','Tidal salinity')
#plot_salt_compRM(delta,'stc','Tidal correction salinity')


#for key in delta.ch_keys: print(np.max(np.abs(delta.ch_outp[key]['st_cor']-delta.ch_pars[key]['st'])))

#plot_new_compRM(delta)