# =============================================================================
# calculate root mean squared error for M2 amplitudes 
# =============================================================================

import numpy as np

def calc_tide_pointRM(self, dat):
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
    #     subtract the model output at the location of the observations
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
            #print('WARNING: point '+punten[pt]+' too far away from estuary, not plotted')
           
            ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
            ind_co = close2[np.argmin(close1)]  #index of the point in this channel
            #print(self.ch_outp[ind_ch]['plot ys'][ind_co],self.ch_outp[ind_ch]['plot xs'][ind_co])
            continue 
        
        ind_ch = self.ch_keys[np.argmin(close1)] #name of the channel where the point is
        ind_co = close2[np.argmin(close1)]  #index of the point in this channel

        aM2_mod[pt] = np.abs(self.ch_pars[ind_ch]['eta'][ind_co])
        pM2_mod[pt] = -np.angle(self.ch_pars[ind_ch]['eta'][ind_co])/np.pi*180
        
    # =============================================================================
    #  calculate error
    # =============================================================================
    
    return amp_M2[np.where(~np.isnan(aM2_mod))[0]],aM2_mod[np.where(~np.isnan(aM2_mod))[0]]


def max_min_salts(self):
    
    print('##################################################')
    print('Calculation of maximum and minimum salinity in the tidal cycle')
    print('The maximum prescribed subtidal salinity is ', np.max(self.soc), ' and the minimal physical possible salinity is 0.')
    print('##################################################')

    for key in self.ch_keys:
        #print maximum and minimum salinity in the tidal cycle
        stot = self.ch_outp[key]['st_r'] + self.ch_outp[key]['ss'][:,:,np.newaxis]
        print('Channel = '+self.ch_gegs[key]['Name'])
        print('Maximum salinity in tidal cycle: ', np.max(stot),', Minimum salinity in tidal cycle: ',np.min(stot))
        print()
        
    print('##################################################')
