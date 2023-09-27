# =============================================================================
# Visualisation, plot the output from the tides 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt      
import pandas as pd   
from matplotlib.collections import LineCollection

def plot_tide(self,nor_amp=1,nor_pha = 180):
    # =============================================================================
    # plot the tidal water level amplitude and phase of the water level in the entire 
    # network of channels
    # =============================================================================
        
    #normalize
    norm_amp = plt.Normalize(0,nor_amp)
    norm_pha = plt.Normalize(0,nor_pha)

    fig,ax = plt.subplots(2,1,figsize=(10,6))
    
    for key in self.ch_keys:
        #amplitude
        self.ch_outp[key]['amp'] = LineCollection(self.ch_outp[key]['segments'], cmap='Spectral_r', norm=norm_amp)
        self.ch_outp[key]['amp'].set_array(np.abs(self.ch_pars[key]['eta']))
        self.ch_outp[key]['amp'].set_linewidth(5)

        #phase
        pha = -np.angle(self.ch_pars[key]['eta'])/np.pi*180
        #reduce the phase
        pha[np.where(pha<0)[0]] = pha[np.where(pha<0)[0]] + 360
        pha[np.where(pha>180)[0]] = 360 - pha[np.where(pha>180)[0]]
        
        self.ch_outp[key]['pha'] = LineCollection(self.ch_outp[key]['segments'], cmap='Spectral', norm=norm_pha)
        self.ch_outp[key]['pha'].set_array(pha)
        self.ch_outp[key]['pha'].set_linewidth(5)

        line0=ax[0].add_collection(self.ch_outp[key]['amp'])
        line1=ax[1].add_collection(self.ch_outp[key]['pha'])

    

    used_jun = []
    show_inds=False
    arrow_scale =0.01
    arc = 'black'
    
    ax[0].axis('scaled'),ax[1].axis('scaled')
    ax[1].set_xlabel('degrees E '),ax[0].set_ylabel('degrees N '),ax[1].set_ylabel('degrees N ')
    ax[0].set_facecolor('lightgrey'),ax[1].set_facecolor('lightgrey')
    
    #ax[1].set_xlim(4,4.6) , ax[1].set_ylim(51.75,52.05) #zoom in on mouth RM
    cb0=fig.colorbar(line0, ax=ax[0],orientation='vertical')
    cb0.set_label(label='Tidal  amplitude [m]') 
    cb1=fig.colorbar(line1, ax=ax[1],orientation='vertical')
    cb1.set_label(label='Tidal phase [deg]')    
    #cb.ax.tick_params(labelsize=15)
    plt.show()
    
    
    '''    
    for key in self.ch_keys:
        #if key == 'C2' or key == 'C3': plt.plot(self.ch_outp[key]['px']-self.ch_gegs['C1']['L'][0],np.abs(self.ch_pars[key]['eta']))
        #else: plt.plot(self.ch_outp[key]['px'],np.abs(self.ch_pars[key]['eta']))
        plt.plot(self.ch_outp[key]['px'],np.abs(self.ch_pars[key]['eta']))
        #plt.plot(np.abs(self.ch_pars[key]['detadx']))
        #plt.title(key)
    plt.xlim(-200000,50000)
    plt.show()#'''
    
    
    return 


def plot_salt_ti_point(self):
    # =============================================================================
    # Plot salinity in the tidal cylce at a certain point. Mostly useful for RM?
    # =============================================================================
    
    # =============================================================================
    # load the observations
    # =============================================================================
    lats=  [1,100,-100]
    lons =  [-100,110,100]
    
    # =============================================================================
    #     subtract the model output at the location of the observations
    # =============================================================================

    for pt in range(len(lats)):
        locE,locN = lons[pt],lats[pt]
        #find closest point. Note that the difference is in degrees and not in meters. This should not be too much of a problem. 
        close1,close2 = [] , []
        for key in self.ch_keys:
            temp = ((self.ch_outp[key]['plot xs']-locE)**2+(self.ch_outp[key]['plot ys']-locN)**2)**0.5
            close1.append(np.min(temp))
            close2.append(np.argmin(temp))
        if np.min(close1)>1000: 
            print('WARNING: point '+' too far away from estuary, not plotted')
           
            ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
            ind_co = close2[np.argmin(close1)]  #index of the point in this channel
            continue 
        
        ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
        ind_co = close2[np.argmin(close1)]  #index of the point in this channel

        #load salinity at this point
        s_here = self.ch_outp[ind_ch]['sb'][ind_co] + np.nanmean(self.ch_outp[ind_ch]['st_r'][ind_co],0)
        plt.plot(s_here,label=ind_ch)
    plt.legend(),plt.grid()
    plt.xlabel('time')
    plt.ylabel('salinity')
    plt.show()

    

def plot_tide_pointRM(self,dat):
    # =============================================================================
    # function to plot the tidal water level amplitude and phase at the observations
    # points in the Rhine-Meuse estuary, together with the observations. 
    # =============================================================================
        
    # =============================================================================
    # load the observations
    # =============================================================================
    punten = np.array(dat['Punt'])
    amp_M2 = np.array(dat['amplitude M2'])/100
    amp_M2n= np.array(dat['error van harmo'])/100

    pha_M2 = np.array(dat['fase M2'])
    pha_M2n= np.array(dat['error van harmo.1'])

    lats=  np.array(dat['lat'])
    lons =  np.array(dat['lon'])
    
    # =============================================================================
    # subtract the model output at the location of the observations
    # =============================================================================
    aM2_mod = np.zeros(len(lats))+np.nan
    pM2_mod = np.zeros(len(lats))+np.nan
    
    for pt in range(len(lats)):
        locE,locN = lons[pt],lats[pt]
        #find closest point. Note that the difference is in degrees and not in meters. This should not be too much of a problem. 
        close1,close2 = [] , []
        for key in self.ch_keys:
            temp = ((self.ch_outp[key]['plot xs']-locE)**2+(self.ch_outp[key]['plot ys']-locN)**2)**0.5
            close1.append(np.min(temp))
            close2.append(np.argmin(temp))
        if np.min(close1)>0.02: 
            print('WARNING: point '+punten[pt]+' too far away from estuary, not plotted')
           
            ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
            ind_co = close2[np.argmin(close1)]  #index of the point in this channel
            print(self.ch_outp[ind_ch]['plot ys'][ind_co],self.ch_outp[ind_ch]['plot xs'][ind_co])
            continue 
        
        ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
        ind_co = close2[np.argmin(close1)]  #index of the point in this channel
        
        aM2_mod[pt] = np.abs(self.ch_pars[ind_ch]['eta'][ind_co])
        pM2_mod[pt] = -np.angle(self.ch_pars[ind_ch]['eta'][ind_co])/np.pi*180
        
        #reduce the phase
        if pM2_mod[pt]<0:
            pM2_mod[pt] = pM2_mod[pt]+360
        #tricks
        if pM2_mod[pt]>180:
            pM2_mod[pt] = 360-pM2_mod[pt]
        if pha_M2[pt]>180:
            pha_M2[pt] = 360-pha_M2[pt]
            
    # =============================================================================
    # Plot this
    # =============================================================================
    plt.scatter(punten[np.where(~np.isnan(aM2_mod))[0]],amp_M2[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='obs',color='blue')
    plt.scatter(punten[np.where(~np.isnan(aM2_mod))[0]],aM2_mod[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='mod',color='red')
    #plt.errorbar(punten[np.where(~np.isnan(aM2_mod))[0]],amp_M2[np.where(~np.isnan(aM2_mod))[0]], yerr=amp_M2n[np.where(~np.isnan(aM2_mod))[0]], fmt="o")
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.grid(),plt.legend()
    plt.ylim(0,1)
    plt.ylabel('Amplitude M$_2$ tide [m]')
    plt.show()
    
    '''#phase 
    plt.scatter(punten[np.where(~np.isnan(aM2_mod))[0]],pha_M2[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='obs',color='blue')
    plt.scatter(punten[np.where(~np.isnan(aM2_mod))[0]],pM2_mod[np.where(~np.isnan(aM2_mod))[0]],marker='D',label='mod',color='red')
    #plt.errorbar(punten[np.where(~np.isnan(aM2_mod))[0]],pha_M2[np.where(~np.isnan(aM2_mod))[0]], yerr=pha_M2n[np.where(~np.isnan(aM2_mod))[0]], fmt="o")
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.grid(),plt.legend(),plt.ylim(0,180)
    plt.ylabel('Phase M$_2$ tide [deg]')
    plt.show()
    #'''
    

def plot_tide_timeseries(self):
    print('WARNING: plot_tide_timeseries function is not general and does not guarantuee to give correct number')
    # =============================================================================
    # function to plot timeseries of salinity, currents and transport at the junctions 
    # =============================================================================
    fig, ax = plt.subplots(5,1,figsize = (8,13))
    
    #water level     
    ax[0].plot(self.ch_outp['C1']['eta_r'][0], label = 'C1')
    ax[0].plot(self.ch_outp['C2']['eta_r'][-1], label = 'C2')
    ax[0].plot(self.ch_outp['C3']['eta_r'][-1], label = 'C3'   )


    #discharges
    ax[1].plot(np.nanmean(self.ch_outp['C1']['ut_r'][0],0)*self.ch_gegs['C1']['H']*self.ch_pars['C1']['b'][0], label = 'C1')
    ax[1].plot(np.nanmean(self.ch_outp['C2']['ut_r'][-1],0)*self.ch_gegs['C2']['H']*self.ch_pars['C2']['b'][-1], label = 'C2')
    ax[1].plot(np.nanmean(self.ch_outp['C3']['ut_r'][-1],0)*self.ch_gegs['C3']['H']*self.ch_pars['C3']['b'][-1], label = 'C3')
    
    ax[1].plot(np.nanmean(self.ch_outp['C2']['ut_r'][-1],0)*self.ch_gegs['C2']['H']*self.ch_pars['C2']['b'][-1]\
               +np.nanmean(self.ch_outp['C3']['ut_r'][-1],0)*self.ch_gegs['C3']['H']*self.ch_pars['C3']['b'][-1],label='C2+C3')
    
    #salinity da
    ax[2].plot(self.ch_outp['C1']['sb'][0] + np.nanmean(self.ch_outp['C1']['st_r'][0],0), label = 'C1')
    ax[2].plot(self.ch_outp['C2']['sb'][-1] + np.nanmean(self.ch_outp['C2']['st_r'][-1],0), label = 'C2')
    ax[2].plot(self.ch_outp['C3']['sb'][-1] + np.nanmean(self.ch_outp['C3']['st_r'][-1],0), label = 'C3')
    
    #salinity sur
    ax[3].plot(self.ch_outp['C1']['sb'][0] + self.ch_outp['C1']['st_r'][0,-1], label = 'C1')
    ax[3].plot(self.ch_outp['C2']['sb'][-1] + self.ch_outp['C2']['st_r'][-1,-1], label = 'C2')
    ax[3].plot(self.ch_outp['C3']['sb'][-1] + self.ch_outp['C3']['st_r'][-1,-1], label = 'C3')

    #transport
    T1 = np.mean(self.ch_outp['C1']['ut_r'][0 ] * self.ch_outp['C1']['st_r'][0 ] , 0)*self.ch_gegs['C1']['H']*self.ch_pars['C1']['b'][0]
    T2 = np.mean(self.ch_outp['C2']['ut_r'][-1] * self.ch_outp['C2']['st_r'][-1] , 0)*self.ch_gegs['C2']['H']*self.ch_pars['C2']['b'][-1]
    T3 = np.mean(self.ch_outp['C3']['ut_r'][-1] * self.ch_outp['C3']['st_r'][-1] , 0)*self.ch_gegs['C3']['H']*self.ch_pars['C3']['b'][-1]
    ax[4].plot(T1, label = 'C1')
    ax[4].plot(T2, label = 'C2')
    ax[4].plot(T3, label = 'C3')
    ax[4].plot(T2 + T3, label = 'C2 + C3')


    ax[0].set_ylabel('water level [m]')    
    ax[1].set_ylabel('tidal discharge [m3/s]')
    ax[2].set_ylabel('da salinity [psu]')
    ax[3].set_ylabel('sur salinity [psu]')
    ax[4].set_ylabel('Transport [kg/s]')
    ax[-1].set_xlabel('Time [tidal cycle]')
    
    for a in range(5):
        ax[a].grid()
        ax[a].legend()
    
    plt.tight_layout()
    plt.show()

    #check of er eigenlijk inderdaad wel gematchet wordt 
    #print(np.nanmean(self.ch_outp['C1']['sb'][0] + self.ch_outp['C1']['st_r'][0]))
    #print(np.nanmean(self.ch_outp['C2']['sb'][-1] + self.ch_outp['C2']['st_r'][0-1]))
    #print(np.nanmean(self.ch_outp['C3']['sb'][-1] + self.ch_outp['C3']['st_r'][-1]))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
