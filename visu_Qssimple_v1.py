# =============================================================================
# Visualisation, salinity and river discharge 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt         
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def plot_Qs_simple(self,show_inds=False,arrow_scale =0.01,arc = 'black'):
    # =============================================================================
    # function to plot the different channels, with the discharge through them, and 
    # the subtidal salinity in the complete network
    # =============================================================================
    
    #normalize
    norm = plt.Normalize(0,self.soc_sca)
    fig,ax = plt.subplots(2,1,figsize=(10,6))
    
    for key in self.ch_keys:
        self.ch_outp[key]['lc'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm)
        self.ch_outp[key]['lc'].set_array(self.ch_outp[key]['sb'])
        self.ch_outp[key]['lc'].set_linewidth(5)
        
        line=ax[1].add_collection(self.ch_outp[key]['lc'])
    
    count=0
    used_jun = []
    for key in self.ch_keys:
        ax[0].plot(self.ch_gegs[key]['plot x'],self.ch_gegs[key]['plot y'],c=self.ch_gegs[key]['plot color'],
                 label = self.ch_gegs[key]['Name']+': $Q$ = '+str(int(np.abs(self.ch_pars[key]['Q'])))+' m$^3$/s')

        if show_inds==True:
            if self.ch_gegs[key]['loc x=-L'] not in used_jun:
                ax[0].text(self.ch_gegs[key]['plot x'][0],self.ch_gegs[key]['plot y'][0],self.ch_gegs[key]['loc x=-L'])
                used_jun.append(self.ch_gegs[key]['loc x=-L'])
            if self.ch_gegs[key]['loc x=0'] not in used_jun:
                ax[0].text(self.ch_gegs[key]['plot x'][-1],self.ch_gegs[key]['plot y'][-1],self.ch_gegs[key]['loc x=0'])
                used_jun.append(self.ch_gegs[key]['loc x=0'])
        
        if self.ch_pars[key]['Q']>0:
            ax[0].arrow(self.ch_gegs[key]['plot x'].mean()+1*arrow_scale,
                      self.ch_gegs[key]['plot y'].mean()+0.5*arrow_scale,
                      (self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      (self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      head_width = 2*arrow_scale, width = 0.1*arrow_scale,color=arc)# ch_gegs[key]['plot color'])
        else:
            ax[0].arrow(self.ch_gegs[key]['plot x'].mean()+1*arrow_scale+(self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      self.ch_gegs[key]['plot y'].mean()+0.5*arrow_scale+(self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      -(self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      -(self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      head_width = 2*arrow_scale, width = 0.1*arrow_scale,color=arc)# ch_gegs[key]['plot color'])
        count = count+1
    
    ax[0].legend(loc='center left', bbox_to_anchor=(1, -0.44))#,ncol =np.max([1, int(len(ch_gegs)/7)]))
    
    ax[0].axis('scaled'),ax[1].axis('scaled')
    ax[1].set_xlabel('degrees E '),ax[0].set_ylabel('degrees N '),ax[1].set_ylabel('degrees N ')
    ax[0].set_facecolor('lightgrey'),ax[1].set_facecolor('lightgrey')
    
    #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
    
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.033])
    cb=fig.colorbar(line, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label='Depth-averaged salinity [g/kg]')    
    #cb.ax.tick_params(labelsize=15)
    plt.show()
    
    return 
    

def plot_Qs_comp(self, show_inds=False, arrow_scale =0.01, arc = 'black',zoom = False, channels = None):
    # =============================================================================
    # Function to plot salt, discharge, stratification and the different processes 
    # in the network. Transport by tides not included. 
    # =============================================================================
    if channels == None:keys_here = self.ch_keys
    else:  keys_here = channels
        
    #normalization factors
    TQmin,TQmax = [],[]
    TEmin,TEmax = [],[]
    TDmin,TDmax = [],[]
    TTmin,TTmax = [],[]
    for key in self.ch_keys:
        TQmin.append(np.nanmin(self.ch_outp[key]['TQ']))
        TQmax.append(np.nanmax(self.ch_outp[key]['TQ']))
        TEmin.append(np.nanmin(self.ch_outp[key]['TE']))
        TEmax.append(np.nanmax(self.ch_outp[key]['TE']))
        TDmin.append(np.nanmin(self.ch_outp[key]['TD']))
        TDmax.append(np.nanmax(self.ch_outp[key]['TD']))
        TTmin.append(np.nanmin(self.ch_outp[key]['TT']+self.ch_outp[key]['TE']+self.ch_outp[key]['TD']+self.ch_outp[key]['TQ']))
        TTmax.append(np.nanmax(self.ch_outp[key]['TT']+self.ch_outp[key]['TE']+self.ch_outp[key]['TD']+self.ch_outp[key]['TQ']))
        
        
    norm1 = plt.Normalize(0,self.soc_sca)
    norm2 = plt.Normalize(0,1)
    norm3 = plt.Normalize(np.min(TQmin),np.max(TQmin))
    norm4 = plt.Normalize(np.min(TEmin),np.max(TEmin))
    norm5 = plt.Normalize(np.min(TDmin),np.max(TDmin))
    norm6 = plt.Normalize(np.min(TTmin),np.max(TTmin))
    # =============================================================================
    # Plot
    # =============================================================================
    fig = plt.figure(figsize=(12,7))
    gs = fig.add_gridspec(100,50)
    
    ax_leg = fig.add_subplot(gs[:,:7])
    ax0 = fig.add_subplot(gs[3:25,10:30])
    ax1 = fig.add_subplot(gs[28:50,10:30])
    ax2 = fig.add_subplot(gs[53:75,10:30])
    ax3 = fig.add_subplot(gs[3:25,30:50])
    ax4 = fig.add_subplot(gs[28:50,30:50])
    ax5 = fig.add_subplot(gs[53:75,30:50])
    ax6 = fig.add_subplot(gs[78:100,30:50])

    for key in self.ch_keys:
        #plot salinity
        self.ch_outp[key]['lc1'] = LineCollection(self.ch_outp[key]['segments'], cmap='RdBu_r', norm=norm1)
        self.ch_outp[key]['lc1'].set_array(self.ch_outp[key]['sb'])
        self.ch_outp[key]['lc1'].set_linewidth(5)
        line1=ax1.add_collection(self.ch_outp[key]['lc1'])
        
        #plot stratification
        self.ch_outp[key]['lc2'] = LineCollection(self.ch_outp[key]['segments'], cmap='Reds', norm=norm2)
        self.ch_outp[key]['lc2'].set_array(self.ch_outp[key]['ds'])
        self.ch_outp[key]['lc2'].set_linewidth(5)
        line2=ax2.add_collection(self.ch_outp[key]['lc2'])
    
        #plot TQ
        self.ch_outp[key]['lc3'] = LineCollection(self.ch_outp[key]['segments'], cmap='Purples', norm=norm3)
        self.ch_outp[key]['lc3'].set_array(self.ch_outp[key]['TQ'])
        self.ch_outp[key]['lc3'].set_linewidth(5)
        line3=ax3.add_collection(self.ch_outp[key]['lc3'])    
        
        #plot TE
        self.ch_outp[key]['lc4'] = LineCollection(self.ch_outp[key]['segments'], cmap='Reds_r', norm=norm4)
        self.ch_outp[key]['lc4'].set_array(self.ch_outp[key]['TE'])
        self.ch_outp[key]['lc4'].set_linewidth(5)
        line4=ax4.add_collection(self.ch_outp[key]['lc4'])    
        
        #plot TD
        self.ch_outp[key]['lc5'] = LineCollection(self.ch_outp[key]['segments'], cmap='Reds_r', norm=norm5)
        self.ch_outp[key]['lc5'].set_array(self.ch_outp[key]['TD'])
        self.ch_outp[key]['lc5'].set_linewidth(5)
        line5=ax5.add_collection(self.ch_outp[key]['lc5'])    
        
        #plot TT or overspill
        self.ch_outp[key]['lc6'] = LineCollection(self.ch_outp[key]['segments'], cmap='Spectral', norm=norm6)
        #self.ch_outp[key]['lc6'].set_array(self.ch_outp[key]['TT'])
        self.ch_outp[key]['lc6'].set_array(self.ch_outp[key]['TT']+self.ch_outp[key]['TE']+self.ch_outp[key]['TD']+self.ch_outp[key]['TQ'])
        self.ch_outp[key]['lc6'].set_linewidth(5)
        line6=ax6.add_collection(self.ch_outp[key]['lc6'])    
           
    #cb=fig.colorbar(line, ax=ax,orientation='horizontal')
    #cb.set_label(label='Depth-averaged salinity  [g kg$^{-1}$]')    
    #cb.ax.tick_params(labelsize=11)
    count=0
    used_jun = []
    for key in self.ch_keys:
        ax0.plot(self.ch_gegs[key]['plot x'],self.ch_gegs[key]['plot y'],c=self.ch_gegs[key]['plot color'],
                 label = self.ch_gegs[key]['Name']+': $Q$ = '+str(np.abs(np.round(self.ch_pars[key]['Q'])))+' m$^3$/s')
        
        #plt.text(self.ch_gegs[keys[i]]['plot x'].mean(),self.ch_gegs[keys[i]]['plot y'].mean(),self.ch_gegs[keys[i]]['Name']+': $Q$= '+str(int(Qdist[i]))+'m3/s')
        if show_inds==True:
            if self.ch_gegs[key]['loc x=-L'] not in used_jun:
                ax0.text(self.ch_gegs[key]['plot x'][0],self.ch_gegs[key]['plot y'][0],self.ch_gegs[key]['loc x=-L'])
                used_jun.append(self.ch_gegs[key]['loc x=-L'])
            if self.ch_gegs[key]['loc x=0'] not in used_jun:
                ax0.text(self.ch_gegs[key]['plot x'][-1],self.ch_gegs[key]['plot y'][-1],self.ch_gegs[key]['loc x=0'])
                used_jun.append(self.ch_gegs[key]['loc x=0'])
        
        if self.ch_pars[key]['Q']>0:
            ax0.arrow(self.ch_gegs[key]['plot x'].mean()+1*arrow_scale,
                      self.ch_gegs[key]['plot y'].mean()+0.5*arrow_scale,
                      (self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      (self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      head_width = 2*arrow_scale, width = 0.1*arrow_scale,color=arc)# self.ch_gegs[key]['plot color'])
        else:
            ax0.arrow(self.ch_gegs[key]['plot x'].mean()+1*arrow_scale+(self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      self.ch_gegs[key]['plot y'].mean()+0.5*arrow_scale+(self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      -(self.ch_gegs[key]['plot x'][-1]-self.ch_gegs[key]['plot x'][0])/4,
                      -(self.ch_gegs[key]['plot y'][-1]-self.ch_gegs[key]['plot y'][0])/4,
                      head_width = 2*arrow_scale, width = 0.1*arrow_scale,color=arc)# self.ch_gegs[key]['plot color'])
        count = count+1
        
    #add the discharge
    clr,lab  = [] , []
    for key in keys_here:
        clr.append(Line2D([0], [0], color=self.ch_gegs[key]['plot color'], lw=2)) 
        lab.append(self.ch_gegs[key]['Name']+': $Q$ = '+str(np.abs(np.round(self.ch_pars[key]['Q'])))+' m$^3$/s')
    
    ax_leg.legend(clr ,lab)
    ax_leg.axis('off')#,ax_ti.axis('off')
    
    #build the axis labels and ticks
    ax0.axis('scaled'),ax1.axis('scaled'),ax2.axis('scaled'),ax3.axis('scaled'),ax4.axis('scaled'),ax5.axis('scaled'),ax6.axis('scaled')
    if zoom==True:
        for ax in [ax0,ax1,ax2,ax3,ax4,ax5,ax6]: ax.set_xlim(4,4.6) , ax.set_ylim(51.75,52.02) #zoom in on mouth RM

    ax0.set_ylabel('degrees N '),ax1.set_ylabel('degrees N '),ax2.set_ylabel('degrees N ') #does not always work properly
    ax2.set_xlabel('degrees E '),ax6.set_xlabel('degrees E ')
    ax0.xaxis.set_ticklabels([])   , ax1.xaxis.set_ticklabels([]), ax3.xaxis.set_ticklabels([]), ax4.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])   , ax4.yaxis.set_ticklabels([]), ax5.yaxis.set_ticklabels([]), ax5.xaxis.set_ticklabels([]), ax6.yaxis.set_ticklabels([])
    ax0.set_facecolor('lightgrey'),ax1.set_facecolor('lightgrey'),ax2.set_facecolor('lightgrey'),ax3.set_facecolor('lightgrey')
    ax4.set_facecolor('lightgrey'),ax5.set_facecolor('lightgrey'),ax6.set_facecolor('lightgrey')

    #ax0.text(0.93,0.9, 'Q' ,transform=ax0.transAxes,fontsize=12)
    #ax1.text(0.93,0.9, r'$\bar{s}$' ,transform=ax1.transAxes,fontsize=12)
    #ax2.text(0.93,0.9, r'$\Delta s$' ,transform=ax2.transAxes,fontsize=12)
    #ax3.text(0.93,0.9, r'$T_Q$' ,transform=ax3.transAxes,fontsize=12)
    #ax4.text(0.93,0.9, r'$T_E$' ,transform=ax4.transAxes,fontsize=12)
    #ax5.text(0.93,0.9, r'$T_D$' ,transform=ax5.transAxes,fontsize=12)
    #ax5.text(0.93,0.9, r'$T_D$' ,transform=ax5.transAxes,fontsize=12)

    #add colorbars
    cb0=plt.colorbar(line1, ax=ax0,orientation='vertical')
    cb0.remove()
    
    cb1=plt.colorbar(line1, ax=ax1,orientation='vertical')
    cb1.set_label(label='s [g/kg]')    
    
    cb2=plt.colorbar(line2, ax=ax2,orientation='vertical')
    cb2.set_label(label='$\Delta s$ []')    
    
    cb3=plt.colorbar(line3, ax=ax3,orientation='vertical')
    cb3.set_label(label='$T_Q$ [kg/s]')    
    
    cb4=plt.colorbar(line4, ax=ax4,orientation='vertical')
    cb4.set_label(label='$T_E$ [kg/s]')    
    
    cb5=plt.colorbar(line5, ax=ax5,orientation='vertical')
    cb5.set_label(label='$T_D$ [kg/s]')  
    
    cb6=plt.colorbar(line6, ax=ax6,orientation='vertical')
    #cb6.set_label(label='$T_T$ [kg/s]')    
    cb6.set_label(label='$T_o$ [kg/s]')    
    
 
    #plot the salt intrusion limit
    for key in self.ch_keys:
        i0 = np.where(self.ch_outp[key]['sb']>2)[0]
        i1 = np.where(self.ch_outp[key]['sb']<2)[0]
        if len(i0)>1 and len(i1)>0: ax1.scatter(*self.ch_outp[key]['segments'][i0[0]][0],marker='|',color='red',linewidth=3,zorder=100)

    plt.show()
    

def plot_proc_ch(self,channels = None):
    # =============================================================================
    # function to plot the salt transport processes in each channel seperately. 
    # =============================================================================
    if channels == None:
        keys_here = self.ch_keys
    else: 
        keys_here = channels

    for key in keys_here:
        plt.figure(figsize=(7,4))
        plt.title(self.ch_gegs[key]['Name'],fontsize=14)
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ'],label='$T_Q$',lw = 2,c='m')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TE'],label='$T_E$',lw = 2,c='brown')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TT'],label='$T_T$',lw = 2,c='red')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TD'],label='$T_D$',lw = 2,c='c')
        plt.plot(self.ch_outp[key]['px']/1000,self.ch_outp[key]['TQ']+self.ch_outp[key]['TD']+self.ch_outp[key]['TT']+self.ch_outp[key]['TE'],label='$T_o$',lw = 2,c='navy')
        
        if self.ch_gegs[key]['loc x=0'][0] == 's': #plot sea domain
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TQs'],lw = 2,c='m')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TEs'],lw = 2,c='brown')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TTs'],lw = 2,c='red')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TDs'],lw = 2,c='c')
            plt.plot(self.ch_outp[key]['pxs']/1000,self.ch_outp[key]['TQs']+self.ch_outp[key]['TDs']+self.ch_outp[key]['TTs']+self.ch_outp[key]['TEs'],lw = 2,c='navy')

        plt.grid()
        plt.xlabel('$x$ [km]',fontsize=14),plt.ylabel('$T$ [kg/s]',fontsize=14)
        plt.xlim(self.ch_outp[key]['px'][0]/1000,self.ch_outp[key]['px'][-1]/1000)
        #plt.xlim(-50,5)
        plt.legend(fontsize=14)#,loc=2)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        
    
        
def plot_contours(self,channels = None):
    # =============================================================================
    # function to plot the salt transport processes in each channel seperately. 
    # =============================================================================
    if channels == None: keys_here = self.ch_keys
    else:   keys_here = channels

    for key in keys_here:
        fig,ax =plt.subplots(1,3,figsize=(12,3))
        
        a0=ax[0].contourf(self.ch_outp[key]['px']/1000,self.z_nd,self.ch_outp[key]['ss'].T,cmap='RdBu_r')
        a1=ax[1].contourf(self.ch_outp[key]['px']/1000,self.z_nd,np.abs(self.ch_pars[key]['st'].T),cmap='RdBu_r')
        a2=ax[2].contourf(self.ch_outp[key]['px']/1000,self.z_nd,np.abs(self.ch_outp[key]['st_cor']-self.ch_pars[key]['st']).T,cmap='RdBu_r')
        
        ax[1].set_title(self.ch_gegs[key]['Name'])
        
        cb0=plt.colorbar(a0, ax=ax[0],orientation='horizontal')
        cb0.set_label(label=r'$s_{st}$ [g/kg]')  
        cb1=plt.colorbar(a1, ax=ax[1],orientation='horizontal')
        cb1.set_label(label=r'$s_{ti}$ [g/kg]')  
        cb2=plt.colorbar(a2, ax=ax[2],orientation='horizontal')
        cb2.set_label(label=r'$s_{bl}$ [g/kg]')          
        
        ax[0].text(.01, .99, 'subtidal', ha='left', va='top', transform=ax[0].transAxes,c='yellow')
        ax[1].text(.01, .99, 'tidal', ha='left', va='top', transform=ax[1].transAxes,c='yellow')
        ax[2].text(.01, .99, 'boundary', ha='left', va='top', transform=ax[2].transAxes,c='yellow')
        
        plt.show()
        
#plot_contours(delta,['Nieuwe Waterweg v2'])
#plot_contours(delta,['Hartelkanaal v2'])
#plot_contours(delta,['Oude Maas 3'])




def plot_junc_tid(self,savename=False):
    # =============================================================================
    # Plot the salinity and velocities in the tidal cycle around a junction, including the boundary layer correction. 
    # Als an animation is possible.  
    # =============================================================================
    sigma = np.linspace(-1,0,self.nz)
    t= np.linspace(0,2*np.pi/self.omega,self.nt)
    
    colors = ['b','r','m']
    
    #calculate boundary layer correction
    for j in range(self.n_j):
        Pct_list = []
        Pct_dom_list = []
        Pt_list  = []
        key_list = []
        px_list = []
        utb_list = []
        for key in self.ch_keys: 
            
            if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1): 
                px_list.append(self.ch_outp[key]['px']-self.ch_outp[key]['px'][0])
                
                P = self.ch_pars[key]['st'][0]/self.soc_sca
                cor = np.sum(self.bbb_n[key][0][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0)
                cor_dom = np.real((np.exp(-px_list[-1]/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(self.bbb_n[key][0][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0))[:,:,np.newaxis]* np.exp(1j*self.omega*t))
                
                Pc = cor + P
                Pct_dom = self.ch_outp[key]['st_r'] + self.soc_sca * cor_dom
                
                Pt_list.append(self.soc_sca*np.real(P[:,np.newaxis] * np.exp(1j*self.omega*t)))
                Pct_list.append(self.soc_sca*np.real(Pc[:,np.newaxis] * np.exp(1j*self.omega*t)))
                Pct_dom_list.append(Pct_dom)
                key_list.append(key)
                
                utb_list.append(np.real(np.mean(self.ch_pars[key]['ut'][0],1)[0,np.newaxis] * np.exp(1j*self.omega*t)))

    
            elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): 
                px_list.append(self.ch_outp[key]['px'])
                
                P = self.ch_pars[key]['st'][-1]/self.soc_sca
                cor = np.sum(self.bbb_n[key][1][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0)
                cor_dom = np.real((np.exp(px_list[-1]/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(self.bbb_n[key][1][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0))[:,:,np.newaxis]* np.exp(1j*self.omega*t))
                
                Pc = cor + P
                Pct_dom = self.ch_outp[key]['st_r'] + self.soc_sca * cor_dom

                Pt_list.append(self.soc_sca*np.real(P[:,np.newaxis] * np.exp(1j*self.omega*t)))
                Pct_list.append(self.soc_sca*np.real(Pc[:,np.newaxis] * np.exp(1j*self.omega*t)))
                Pct_dom_list.append(Pct_dom)
                key_list.append(key)
                
                utb_list.append(np.real(np.mean(self.ch_pars[key]['ut'][0],1)[-1,np.newaxis] * np.exp(1j*self.omega*t)))

        #plot salinity timeseries at a certain depth. 
        plot_depth = 0

        for i in range(3):
            plt.plot(Pct_list[i][plot_depth],c=colors[i],label=key_list[i])
            plt.plot(Pt_list[i][plot_depth],c=colors[i],alpha=0.5)

        plt.legend()
        plt.grid()
        plt.xlabel('Time [tidal cycle]') , plt.ylabel(r'$s_{sur}$ [psu]')
        plt.show()
        
        #plot velocity
        for i in range(3):
            plt.plot(utb_list[i],c=colors[i],label=key_list[i])

        plt.legend()
        plt.grid()
        plt.xlabel('Time [tidal cycle]') , plt.ylabel(r'$\bar{u_t}$ [psu]')
        plt.show()
        '''
        #plot profiles at a certain point of time
        tplt = 30
        plt.title('T='+str(tplt))
        
        for i in range(3):
            plt.plot(px_list[i],self.ch_outp[key_list[i]]['st_r'][:,plot_depth,tplt] ,c=colors[i],alpha=0.5)
            plt.plot(px_list[i],Pct_dom_list[i][:,plot_depth,tplt] ,c=colors[i],label=key_list[i])

        plt.xlim(-5000,5000)
        plt.legend()
        plt.show()
        '''
        
        #make animation
        import matplotlib.animation as ani       #make animations
        if savename != False:

            def init():
                tplt = 0
                ax.set_title('T='+str(tplt))
                for i in range(3):
                    plt.plot(px_list[i]/1000,self.ch_outp[key_list[i]]['st_r'][:,plot_depth,tplt] ,c=colors[i],alpha=0.5)
                    plt.plot(px_list[i]/1000,Pct_dom_list[i][:,plot_depth,tplt] ,c=colors[i],label=key_list[i])
                ax.set_ylim(-5,5)
                ax.set_xlim(-5,5)
                
                
            def animate(tplt):
                ax.cla() #, cb0.cla(),cb1.cla(),cb2.cla()
                ax.set_title('T='+str(tplt))
    
                for i in range(3):
                    plt.plot(px_list[i]/1000,self.ch_outp[key_list[i]]['st_r'][:,plot_depth,tplt] ,c=colors[i],alpha=0.5)
                    plt.plot(px_list[i]/1000,Pct_dom_list[i][:,plot_depth,tplt] ,c=colors[i],label=key_list[i])
    
                ax.legend()
                ax.set_ylim(-5,5)
                ax.set_xlim(-5,5)
                ax.set_xlabel('$x$ [km]') , ax.set_ylabel('$s_{ti}$ [psu]')
                return ax 
    
            fig,ax = plt.subplots(1,1,figsize=(6,4))
            anim = ani.FuncAnimation(fig,animate,self.nt,init_func=init,blit=False)
            #frames per second is now nt/11. should be determined from input but ja geen zin 
            anim.save("/Users/biemo004/Documents/UU phd Saltisolutions/Output/Animations all/"+savename+".mp4", fps=int(self.nt/6), extra_args=['-vcodec', 'libx264'])
    
            plt.show()


def plot_jinyang(self):
    # =============================================================================
    # Make Jinyang style plots of depth-averaged salinity. Not guaranteed to work in every circumstance
    # =============================================================================
    for key in self.ch_keys: 

        if self.ch_gegs[key]['loc x=-L'] == 'j'+str(1):  px_here = self.ch_outp[key]['px']-self.ch_outp[key]['px'][0]
        elif self.ch_gegs[key]['loc x=0'] == 'j'+str(1):  px_here = self.ch_outp[key]['px']
    
        plt.plot(px_here/1000,self.ch_outp[key]['sb'],label=key)
        
    plt.legend()
    plt.grid()
    plt.xlabel('$x$ [km]') , plt.ylabel(r'$\bar{s}$ [psu]')
    #plt.xlim(-10,10) , plt.ylim(0,12)
    plt.show()
    
  

def plot_fases(self):
    # =============================================================================
    # Plot the phases of the tidal velocity and the tidal salinity around the junctions
    # =============================================================================
    for key in self.ch_keys: 

        if self.ch_gegs[key]['loc x=-L'] == 'j'+str(1): 
            px_here = self.ch_outp[key]['px']-self.ch_outp[key]['px'][0]
          
        elif self.ch_gegs[key]['loc x=0'] == 'j'+str(1): 
            px_here = self.ch_outp[key]['px']

    
        plt.plot(px_here,np.angle(np.mean(self.ch_pars[key]['ut'][0],1))/np.pi*180,label=key)
            
        #print(len(np.angle(np.mean(self.ch_pars[key]['ut'][0],1))) , len(px_here))
        
    plt.legend()
    plt.grid()
    plt.xlim(-1e4,1e4)
    plt.xlabel('$x$ [km]') , plt.ylabel(r'phase $\bar{u}$ [deg]')
    plt.show()
    
    # =============================================================================
    # Plot the salinity in the tidal cycle around a junction, including the boundary layer correction. 
    # Has to be upgraded to an animation in the future. 
    # =============================================================================
    sigma = np.linspace(-1,0,self.nz)
    t= np.linspace(0,2*np.pi/self.omega,self.nt)
    
    colors = ['b','r','m']
    
    for j in range(self.n_j):
        Pct_list = []
        Pct_dom_list = []
        Pt_list  = []
        key_list = []
        px_list = []
        for key in self.ch_keys: 
            #print(self.ch_outp[key]['px'], )
            
            
            if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1): 
                px_list.append(self.ch_outp[key]['px']-self.ch_outp[key]['px'][0])
                cor_dom = (np.exp(-px_list[-1]/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(self.bbb_n[key][0][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0))
                Pct_dom = self.ch_pars[key]['st'] + self.soc_sca * cor_dom
                Pct_dom_list.append(Pct_dom)
                key_list.append(key)
                

            elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): 
                px_list.append(self.ch_outp[key]['px'])
                cor_dom = (np.exp(px_list[-1]/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(self.bbb_n[key][1][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0))
                Pct_dom = self.ch_pars[key]['st'] + self.soc_sca * cor_dom
                Pct_dom_list.append(Pct_dom)
                key_list.append(key)


        #plot timeseries at a certain depth. 
        plot_depth = 0
        #'''
        for i in range(3):
            plt.plot(px_list[i],np.angle(np.mean(Pct_dom_list[i],1))/np.pi*180,c=colors[i],label=key_list[i])
            #plt.plot(Pt_list[i][plot_depth],c=colors[i],alpha=0.5)
            
        plt.xlabel('$x$ [km]') , plt.ylabel(r'phase $\bar{s}$ [deg]')
        plt.grid()
        plt.xlim(-1e4,1e4)
        plt.legend()
        plt.show()
        #'''

    
    
    
    
    
    
    
    


