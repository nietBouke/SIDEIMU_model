# =============================================================================
# make animations in tidal cycle of salinity, water level and currents in a general network  
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
import matplotlib.animation as ani       #make animations
from matplotlib.lines import Line2D

def anim_tide(self,savename):   
    # =============================================================================
    # Make an animation of water level, tidal currents and salinity in the tidal cycle
    # =============================================================================
    #normalize
    ext_eta, ext_utb, ext_stb = np.zeros((len(self.ch_keys),2)) , np.zeros((len(self.ch_keys),2)) , np.zeros((len(self.ch_keys),2))
    count = 0
    for key in self.ch_keys:
        ext_eta[count] = [np.nanmin(self.ch_outp[key]['eta_r']),np.nanmax(self.ch_outp[key]['eta_r'])]
        ext_utb[count] = [np.nanmin(self.ch_outp[key]['utb_r']),np.nanmax(self.ch_outp[key]['utb_r'])]
        #ext_stb[count] = [np.min(self.ch_outp[key]['stb_r']),np.max(self.ch_outp[key]['stb_r'])]
        ext_stb[count] = [np.min(self.ch_outp[key]['stb_cor_r']),np.max(self.ch_outp[key]['stb_cor_r'])]
        count +=1
    norm_p1 = plt.Normalize(np.round(np.min(ext_eta[:,0]),1),np.round(np.max(ext_eta[:,1]),1))
    norm_p2 = plt.Normalize(np.round(np.min(ext_utb[:,0]),1),np.round(np.max(ext_utb[:,1]),1))
    norm_p3 = plt.Normalize(0,20)#np.round(np.min(ext_stb[:,0]),1),np.round(np.max(ext_stb[:,1]),1))

    def init():
        plot_t = 0
        #ax_ti.text(0,0,'Time = '+str(plot_t)+' timesteps after start of the simulation',fontsize=12)    
        for key in self.ch_keys:
            self.ch_outp[key]['p1'] = LineCollection(self.ch_outp[key]['segments'], cmap='ocean', norm=norm_p1)
            self.ch_outp[key]['p1'].set_array(self.ch_outp[key]['eta_r'][:,plot_t])
            self.ch_outp[key]['p1'].set_linewidth(5)
            
            self.ch_outp[key]['p2'] = LineCollection(self.ch_outp[key]['segments'], cmap='Spectral', norm=norm_p2)
            self.ch_outp[key]['p2'].set_array(self.ch_outp[key]['utb_r'][:,plot_t])
            self.ch_outp[key]['p2'].set_linewidth(5)

            self.ch_outp[key]['p3'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm_p3)
            self.ch_outp[key]['p3'].set_array(self.ch_outp[key]['stb_cor_r'][:,plot_t]+self.ch_outp[key]['sb'])
            #self.ch_outp[key]['p3'].set_array(self.ch_outp[key]['stb_r'][:,plot_t]+self.ch_outp[key]['sb'])
            self.ch_outp[key]['p3'].set_linewidth(5)

            line0=ax[0].add_collection(self.ch_outp[key]['p1'])
            line1=ax[1].add_collection(self.ch_outp[key]['p2'])
            line2=ax[2].add_collection(self.ch_outp[key]['p3'])
            
        ax[0].axis('scaled'),ax[1].axis('scaled'),ax[2].axis('scaled')
        ax[1].set_xlabel('degrees E '),ax[0].set_ylabel('degrees N '),ax[1].set_ylabel('degrees N ')
        ax[0].set_facecolor('lightgrey'),ax[1].set_facecolor('lightgrey'),ax[2].set_facecolor('lightgrey')

        #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
        cb0=fig.colorbar(line0, ax=ax[0],orientation='vertical')
        cb0.set_label(label='Water level [m]') 
        cb1=fig.colorbar(line1, ax=ax[1],orientation='vertical')
        cb1.set_label(label='Tidal current [m/s]')    
        cb2=fig.colorbar(line2, ax=ax[2],orientation='vertical')
        cb2.set_label(label='Salinity [g/kg]')    
           
    def animate(plot_t):
        for i in range(3):   ax[i].cla() #, cb0.cla(),cb1.cla(),cb2.cla()
        
        for key in self.ch_keys:
            self.ch_outp[key]['p1'] = LineCollection(self.ch_outp[key]['segments'], cmap='ocean', norm=norm_p1)
            self.ch_outp[key]['p1'].set_array(self.ch_outp[key]['eta_r'][:,plot_t])
            self.ch_outp[key]['p1'].set_linewidth(5)
            
            self.ch_outp[key]['p2'] = LineCollection(self.ch_outp[key]['segments'], cmap='Spectral', norm=norm_p2)
            self.ch_outp[key]['p2'].set_array(self.ch_outp[key]['utb_r'][:,plot_t])
            self.ch_outp[key]['p2'].set_linewidth(5)

            self.ch_outp[key]['p3'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm_p3)
            self.ch_outp[key]['p3'].set_array(self.ch_outp[key]['stb_cor_r'][:,plot_t]+self.ch_outp[key]['sb'])
            #self.ch_outp[key]['p3'].set_array(self.ch_outp[key]['stb_r'][:,plot_t]+self.ch_outp[key]['sb'])
            self.ch_outp[key]['p3'].set_linewidth(5)

            line0=ax[0].add_collection(self.ch_outp[key]['p1'])
            line1=ax[1].add_collection(self.ch_outp[key]['p2'])
            line2=ax[2].add_collection(self.ch_outp[key]['p3'])
            
        ax[0].axis('scaled'),ax[1].axis('scaled'),ax[2].axis('scaled')
        ax[-1].set_xlabel('degrees E '),ax[0].set_ylabel('degrees N '),ax[1].set_ylabel('degrees N '),ax[2].set_ylabel('degrees N ')
        ax[0].set_facecolor('lightgrey'),ax[1].set_facecolor('lightgrey'),ax[2].set_facecolor('lightgrey')
        
        #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
        return ax 

    fig,ax = plt.subplots(3,1,figsize=(10,10))

    anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
    #frames per second is now nt/11. should be determined from input but ja geen zin 
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=int(self.nt/11), extra_args=['-vcodec', 'libx264'])
    
    plt.show()


    return

def anim_tide_wl(self,savename):   
    # =============================================================================
    # Make an animation with only the tidal water level in the entire network
    # =============================================================================
    #normalize
    ext_eta, ext_utb, ext_stb = np.zeros((len(self.ch_keys),2)) , np.zeros((len(self.ch_keys),2)) , np.zeros((len(self.ch_keys),2))
    count = 0
    for key in self.ch_keys:
        ext_eta[count] = [np.nanmin(self.ch_outp[key]['eta_r']),np.nanmax(self.ch_outp[key]['eta_r'])]
        count +=1
    norm_p1 = plt.Normalize(np.round(np.min(ext_eta[:,0]),1),np.round(np.max(ext_eta[:,1]),1))

    def init():
        plot_t = 0
        #ax_ti.text(0,0,'Time = '+str(plot_t)+' timesteps after start of the simulation',fontsize=12)    
        for key in self.ch_keys:
            self.ch_outp[key]['p1'] = LineCollection(self.ch_outp[key]['segments'], cmap='cool', norm=norm_p1)
            self.ch_outp[key]['p1'].set_array(self.ch_outp[key]['eta_r'][:,plot_t])
            self.ch_outp[key]['p1'].set_linewidth(5)

            line0=ax.add_collection(self.ch_outp[key]['p1'])

            
        ax.axis('scaled')
        ax.set_xlabel('degrees E '),ax.set_ylabel('degrees N ')
        ax.set_facecolor('lightgrey')
        
        #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
        cb0=fig.colorbar(line0, ax=ax,orientation='vertical')
        cb0.set_label(label='Water level [m]') 
           
    def animate(plot_t):
        ax.cla() #, cb0.cla(),cb1.cla(),cb2.cla()
        
        for key in self.ch_keys:
            self.ch_outp[key]['p1'] = LineCollection(self.ch_outp[key]['segments'], cmap='cool', norm=norm_p1)
            self.ch_outp[key]['p1'].set_array(self.ch_outp[key]['eta_r'][:,plot_t])
            self.ch_outp[key]['p1'].set_linewidth(5)


            line0=ax.add_collection(self.ch_outp[key]['p1'])
            
        ax.axis('scaled')
        ax.set_xlabel('degrees E '),ax.set_ylabel('degrees N ')
        ax.set_facecolor('lightgrey')

        return ax 

    fig,ax = plt.subplots(1,1,figsize=(8,4))

    anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
    anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=21,extra_args=['-vcodec', 'libx264'])
    
    plt.show()


    return




