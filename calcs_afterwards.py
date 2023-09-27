import numpy 

def max_min_salts(self):
    #print maximum and minimum salinity in the tidal cycle
    stot = self.ch_outp[key]['st_r'] + self.ch_outp[key]['ss'][:,:,np.newaxis]
    print('Channel = '+self.ch_gegs[key]['Name'])
    print('The maximum prescribed subtidal salinity is ', self.soc, ' and the minimal physical possible salinity is 0.')
    print('Maximum salinity in tidal cycle: ', np.max(stot),', Minimum salinity in tidal cycle: ',np.min(stot))
    print()