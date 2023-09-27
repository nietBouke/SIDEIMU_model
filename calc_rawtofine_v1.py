# =============================================================================
# functions to convert output of the saltmodule to plottable stuff, ie.e salinity and velocity values
# =============================================================================

import numpy as np

def conv_ans(self, key): 
    # =============================================================================
    # function to calcualte salinity and derivatives from raw output from the model
    # =============================================================================
    #indices
    inds = {}
    inds['di_l'] = np.zeros((len(self.ch_pars[key]['di'])-2)*2)
    for i in range(1,len(self.ch_pars[key]['di'])-1):
        inds['di_l'][i*2-2] = self.ch_pars[key]['di'][i]-1
        inds['di_l'][i*2-1] = self.ch_pars[key]['di'][i]
    inds['di_l'] = np.array(inds['di_l'], dtype=int)
    inds['x'] = np.delete(np.arange(self.ch_pars[key]['di'][-1]),inds['di_l'])[1:-1] # x coordinates for the points which are not on a aboundary
    inds['xr'] = inds['x'].repeat(self.N) # x for N values, mostly i in old code
    inds['xr_m'] = inds['xr']*self.M #M*i in old coe
    inds['xrm_m'] = (inds['xr']-1)*self.M
    inds['xrp_m'] = (inds['xr']+1)*self.M
    inds['xr_mj'] = inds['xr_m']+np.tile(np.arange(1,self.M),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))
    inds['xrm_mj'] = inds['xrm_m']+np.tile(np.arange(1,self.M),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))
    inds['xrp_mj'] = inds['xrp_m']+np.tile(np.arange(1,self.M),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))

    inds['x_m'] = inds['x']*self.M
    inds['xp_m'] = (inds['x']+1)*self.M
    inds['xm_m'] = (inds['x']-1)*self.M
        
    #prepare 
    ss = self.ch_outp[key]['sss']*self.soc_sca
    inds = inds.copy()
    dx = self.ch_pars[key]['dl']*self.Lsc
    di_here = self.ch_pars[key]['di']
    #make empty vectors.
    dsbdx,dsbdx2 = np.zeros(di_here[-1])+np.nan , np.zeros(di_here[-1])+np.nan
    dsndx = np.zeros((di_here[-1]*self.M)) +np.nan
  
    # calculate derivatives in interior
    dsbdx[inds['x']] = (ss[inds['xp_m']]-ss[inds['xm_m']])/(2*dx[inds['x']]) 
    dsbdx2[inds['x']] = (ss[inds['xp_m']]-2*ss[inds['x_m']]+ss[inds['xm_m']])/(dx[inds['x']]**2)         
    dsndx[inds['xr_mj']] = (ss[inds['xrp_mj']]-ss[inds['xrm_mj']])/(2*dx[inds['xr']])

    #calculate derivatives at the boundaries
    #left boundary
    for d in di_here[:-1]:
        dsbdx[d] = (-3*ss[(d+0)*self.M] + 4*ss[(d+1)*self.M] - ss[(d+2)*self.M]) / (2*dx[d])
        dsbdx2[d] = (2*ss[(d+0)*self.M] - 5*ss[(d+1)*self.M] + 4*ss[(d+2)*self.M] - ss[(d+3)*self.M]) / (dx[d]**2)
        dsndx[d*self.M+1:(d+1)*self.M] = (-3*ss[(d+0)*self.M+1:(d+1)*self.M] + 4*ss[(d+1)*self.M+1:(d+2)*self.M] - ss[(d+2)*self.M+1:(d+3)*self.M]) / (2*dx[d])
    #right boundary
    for d in di_here[1:]:
        dsbdx[d-1] = (3*ss[(d-1)*self.M] - 4*ss[(d-2)*self.M] + ss[(d-3)*self.M]) / (2*dx[d-1])
        dsbdx2[d-1] = (2*ss[(d-1)*self.M] - 5*ss[(d-2)*self.M] + 4*ss[(d-3)*self.M] - ss[(d-4)*self.M]) / (dx[d-1]**2)
        dsndx[(d-1)*self.M+1:d*self.M] = (3*ss[(d-1)*self.M+1:(d-0)*self.M] - 4*ss[(d-2)*self.M+1:(d-1)*self.M] + ss[(d-3)*self.M+1:(d-2)*self.M]) / (2*dx[d-1])

    #put it in the right format. 
    sn = ss.reshape((di_here[-1],self.M))[:,1:].T
    dsndx = dsndx.reshape((di_here[-1],self.M))[:,1:].T
    d_ans = (sn[:,:,np.newaxis],dsndx[:,:,np.newaxis],dsbdx[np.newaxis,:,np.newaxis],dsbdx2[np.newaxis,:,np.newaxis])
    
    return d_ans

def calc_output(self):
    # =============================================================================
    #     function to convert output of model to other quantities
    # =============================================================================
    
    #starting indices of the channels in the big matrix, beetje omslachtig hier. 
    ind = [0]
    for key in self.ch_keys: ind.append(ind[-1]+self.ch_pars[key]['di'][-1]*self.M) 
    nnp = np.arange(1,self.M)*np.pi #n*pi


    #do the operatoins for every channel
    count = 1
    for key in self.ch_keys:
        #salinity
        try: #if the salinity module did not run, this cannot be calculated
            self.ch_outp[key]['sss'] = self.sss_n[ind[count-1]:ind[count]]
        except:
            self.ch_outp[key]['sss'] = np.zeros(ind[count]-ind[count-1])
            
        self.ch_outp[key]['sb'] = np.reshape(self.ch_outp[key]['sss'],(self.ch_pars[key]['di'][-1],self.M))[:,0 ] * self.soc_sca
        self.ch_outp[key]['sn'] = np.reshape(self.ch_outp[key]['sss'],(self.ch_pars[key]['di'][-1],self.M))[:,1:] * self.soc_sca
        self.ch_outp[key]['ss'] = self.ch_outp[key]['sb'][:,np.newaxis] + np.sum(self.ch_outp[key]['sn'][:,:,np.newaxis]*np.cos(np.arange(1,self.M)[:,np.newaxis]*np.pi*self.z_nd),1)
        self.ch_outp[key]['ds'] = np.sum(self.ch_outp[key]['sn']*(np.cos(nnp)-1),1)/self.soc_sca #stratification
       
        #check negative salinity
        if np.min(self.ch_outp[key]['ss']) < -1e-10: #actually salinity should not be smaller than the river slainity
            print('WARNING: negative salinity simulated: ', np.min(self.ch_outp[key]['ss']) ,' in canal', key, '. The solution is probably to decrease the spatial step')
            import matplotlib.pyplot as plt
            plt.contourf(self.ch_outp[key]['ss'].T,levels=np.linspace(0,35,11),cmap='RdBu_r')
            plt.colorbar()
            plt.title('Negative salinity! channel = '+str(key))
            plt.show()
            
        #check inverse stratification
        #print(np.where(np.argmax(self.ch_outp[key]['ss'],axis=1)!=0)[0])
        #print(self.ch_outp[key]['sb'].shape)
        
        arg_is = np.where(np.argmax(self.ch_outp[key]['ss'],axis=1)!=0)[0]
        arg_ishs = np.where(self.ch_outp[key]['sb'][arg_is] > 0.01)[0]
                
        if len(arg_ishs) >0: 
            print('WARNING: Inverse stratification simulated in channel ',key)
            import matplotlib.pyplot as plt
            plt.contourf(self.ch_outp[key]['ss'].T,cmap='RdBu_r')
            plt.colorbar()
            plt.title('Inverse stratification!')
            plt.show()
        
        #calculate derivative of depth-averaged subtidal salinity 
        self.ch_outp[key]['sb_x'] = np.zeros(self.ch_pars[key]['di'][-1]) + np.nan #
        for dom in range(len(self.ch_pars[key]['di'])-1):
            self.ch_outp[key]['sb_x'][self.ch_pars[key]['di'][dom]+1:self.ch_pars[key]['di'][dom+1]-1] = (self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom]+2:self.ch_pars[key]['di'][dom+1]]\
                 -self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom]:self.ch_pars[key]['di'][dom+1]-2])/(2*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom]+1:self.ch_pars[key]['di'][dom+1]-1]*self.Lsc)   
            #also do the boundaries
            self.ch_outp[key]['sb_x'][self.ch_pars[key]['di'][dom]] = (-3*self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom]] +4*self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom]+1] -self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom]+2]) / (2*self.Lsc*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom]])
            self.ch_outp[key]['sb_x'][self.ch_pars[key]['di'][dom+1]-1] =(3*self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom+1]-1] -4*self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom+1]-2] +self.ch_outp[key]['sb'][self.ch_pars[key]['di'][dom+1]-3]) / (2*self.Lsc*self.ch_pars[key]['dl'][self.ch_pars[key]['di'][dom+1]-1])
            
        #x coordinate for plotting
        self.ch_outp[key]['px'] = np.zeros(np.sum(self.ch_pars[key]['nxn']))
        self.ch_outp[key]['px'][0:self.ch_pars[key]['nxn'][0]] = -np.linspace(np.sum(self.ch_gegs[key]['L'][0:]), np.sum(self.ch_gegs[key]['L'][0+1:]), self.ch_pars[key]['nxn'][0])
        for i in range(1,len(self.ch_pars[key]['nxn'])): self.ch_outp[key]['px'][np.sum(self.ch_pars[key]['nxn'][:i]):np.sum(self.ch_pars[key]['nxn'][:i+1])] =\
            -np.linspace(np.sum(self.ch_gegs[key]['L'][i:]), np.sum(self.ch_gegs[key]['L'][i+1:]), self.ch_pars[key]['nxn'][i])
        tot_L = np.sum(self.ch_gegs[key]['L'])
        
        #calcualte subtidal velocities. 
        self.ch_outp[key]['u_st'] = self.ch_pars[key]['Q'] * self.ch_pars[key]['bH_1'][:,np.newaxis] * (self.ch_pars[key]['g1'] + 1 + self.ch_pars[key]['g2'] * self.z_nd**2) \
            + self.ch_pars[key]['alf'] * self.ch_outp[key]['sb_x'][:,np.newaxis] * (self.ch_pars[key]['g3'] + self.ch_pars[key]['g4'] * self.z_nd**2 + self.ch_pars[key]['g5'] * self.z_nd**3)
        self.ch_outp[key]['w_st'] = np.zeros(self.ch_outp[key]['u_st'].shape) #TODO
            
        #Calculate transports      
        self.ch_outp[key]['TQ'] = self.ch_pars[key]['Q']*self.ch_outp[key]['sb'] #transport by mean current
        self.ch_outp[key]['TE'] = np.sum(self.ch_outp[key]['sn']*(2*self.ch_pars[key]['Q']*self.ch_pars[key]['g2']*np.cos(nnp)/nnp**2 \
                              + self.ch_gegs[key]['H']*self.ch_pars[key]['b'][:,np.newaxis]*self.ch_pars[key]['alf']*self.ch_outp[key]['sb_x'][:,np.newaxis]*(2*self.ch_pars[key]['g4']\
                              * np.cos(nnp)/nnp**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp)-6)/nnp**4 - 3*np.cos(nnp)/nnp**2))),1) #transport by vertically sheared current
        self.ch_outp[key]['TD'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b']*- self.ch_pars[key]['Kh']*self.ch_outp[key]['sb_x'] #transport by horizontal diffusion
        #try:
        self.ch_outp[key]['TT'] , self.ch_pars[key]['st'] = self.ch_tide['flux'](key,*conv_ans(self, key),self.bbb_n[key]) #transport by tides 
        self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'] * self.ch_gegs[key]['H']*self.ch_pars[key]['b']
        #except: 
        #    self.ch_outp[key]['TT'] = np.zeros(self.ch_outp[key]['TD'].shape)
        #    self.ch_pars[key]['st'] = np.zeros(self.ch_outp[key]['ss'].shape)
        #remove edge points -  not nessecary
        #self.ch_outp[key]['TQ'][[0,-1]],self.ch_outp[key]['TE'][[0,-1]],self.ch_outp[key]['TD'][[0,-1]] = None , None , None
        #self.ch_outp[key]['TQ'][self.ch_pars[key]['di'][1:-1]],self.ch_outp[key]['TE'][self.ch_pars[key]['di'][1:-1]],self.ch_outp[key]['TD'][self.ch_pars[key]['di'][1:-1]] = None, None, None
        #self.ch_outp[key]['TQ'][self.ch_pars[key]['di'][1:-1]-1],self.ch_outp[key]['TE'][self.ch_pars[key]['di'][1:-1]-1],self.ch_outp[key]['TD'][self.ch_pars[key]['di'][1:-1]-1] = None, None, None


        #for animation on tidal timescale:   
        t= np.linspace(0,2*np.pi/self.omega,self.nt)
        self.ch_outp[key]['eta'] = self.ch_pars[key]['eta'][0,:,0]
        self.ch_outp[key]['eta_r'] = np.real(self.ch_outp[key]['eta'][:,np.newaxis]* np.exp(1j*self.omega*t))
        self.ch_outp[key]['ut_r'] = np.real(self.ch_pars[key]['ut'][0,:,:,np.newaxis] * np.exp(1j*self.omega*t))
        self.ch_outp[key]['wt_r'] = np.real(self.ch_pars[key]['wt'][0,:,:,np.newaxis] * np.exp(1j*self.omega*t))
        self.ch_outp[key]['st_r'] = np.real(self.ch_pars[key]['st'][:,:,np.newaxis] * np.exp(1j*self.omega*t))
        
        #remove sea domain
        if self.ch_gegs[key]['loc x=0'][0] == 's':
            self.ch_outp[key]['pxs'] = self.ch_outp[key]['px'][-self.nx_sea:]+self.length_sea
            self.ch_outp[key]['TQs'] = self.ch_outp[key]['TQ'][-self.nx_sea:]
            self.ch_outp[key]['TEs'] = self.ch_outp[key]['TE'][-self.nx_sea:]
            self.ch_outp[key]['TDs'] = self.ch_outp[key]['TD'][-self.nx_sea:]
            self.ch_outp[key]['TTs'] = self.ch_outp[key]['TT'][-self.nx_sea:]
            
            
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_sea]+self.length_sea
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_sea]
            self.ch_pars[key]['ut'] = self.ch_pars[key]['ut'][:,:-self.nx_sea]
            self.ch_outp[key]['ss'] = self.ch_outp[key]['ss'][:-self.nx_sea]
            self.ch_pars[key]['st'] = self.ch_pars[key]['st'][:-self.nx_sea]
            self.ch_outp[key]['sb'] = self.ch_outp[key]['sb'][:-self.nx_sea] 
            self.ch_outp[key]['sn'] = self.ch_outp[key]['sn'][:-self.nx_sea]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_sea]
            self.ch_outp[key]['w_st'] = self.ch_outp[key]['w_st'][:-self.nx_sea]
            self.ch_outp[key]['sb_x'] = self.ch_outp[key]['sb_x'][:-self.nx_sea]
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_sea]
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_sea]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_sea]
            self.ch_outp[key]['st_r'] = self.ch_outp[key]['st_r'][:-self.nx_sea]

            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:-self.nx_sea]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:-self.nx_sea]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:-self.nx_sea]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:-self.nx_sea]

            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])

        elif self.ch_gegs[key]['loc x=-L'][0] == 's':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_sea:]
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_sea:]
            self.ch_outp[key]['ss'] = self.ch_outp[key]['ss'][self.nx_sea:]
            self.ch_pars[key]['st'] = self.ch_pars[key]['st'][self.nx_sea:]
            self.ch_pars[key]['ut'] = self.ch_pars[key]['ut'][:,self.nx_sea:]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_sea:]
            self.ch_outp[key]['w_st'] = self.ch_outp[key]['w_st'][self.nx_sea:]
            self.ch_outp[key]['sb'] = self.ch_outp[key]['sb'][self.nx_sea:] 
            self.ch_outp[key]['sn'] = self.ch_outp[key]['sn'][self.nx_sea:]            
            self.ch_outp[key]['sb_x'] = self.ch_outp[key]['sb_x'][self.nx_sea:]            
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_sea:]            
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_sea:]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_sea:]
            self.ch_outp[key]['st_r'] = self.ch_outp[key]['st_r'][self.nx_sea:]
            
            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][self.nx_sea:]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][self.nx_sea:]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][self.nx_sea:]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][self.nx_sea:]
            
            tot_L = np.sum(self.ch_gegs[key]['L'][1:])
                    
        #remove river domain
        if self.ch_gegs[key]['loc x=0'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][:-self.nx_riv]+self.length_riv
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][:-self.nx_riv]
            self.ch_outp[key]['ss'] = self.ch_outp[key]['ss'][:-self.nx_riv]
            self.ch_pars[key]['st'] = self.ch_pars[key]['st'][:-self.nx_riv]
            self.ch_pars[key]['ut'] = self.ch_pars[key]['ut'][:,:-self.nx_riv]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][:-self.nx_riv]
            self.ch_outp[key]['w_st'] = self.ch_outp[key]['w_st'][:-self.nx_riv]
            self.ch_outp[key]['sb'] = self.ch_outp[key]['sb'][:-self.nx_riv] 
            self.ch_outp[key]['sn'] = self.ch_outp[key]['sn'][:-self.nx_riv]
            self.ch_outp[key]['sb_x'] = self.ch_outp[key]['sb_x'][:-self.nx_riv]
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][:-self.nx_riv]
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][:-self.nx_riv]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][:-self.nx_riv]
            self.ch_outp[key]['st_r'] = self.ch_outp[key]['st_r'][:-self.nx_riv]
            
            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][:-self.nx_riv]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][:-self.nx_riv]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][:-self.nx_riv]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][:-self.nx_riv]

            tot_L = np.sum(self.ch_gegs[key]['L'][:-1])

        elif self.ch_gegs[key]['loc x=-L'][0] == 'r':
            self.ch_outp[key]['px'] = self.ch_outp[key]['px'][self.nx_riv:]
            self.ch_outp[key]['eta'] = self.ch_outp[key]['eta'][self.nx_riv:]
            self.ch_outp[key]['ss'] = self.ch_outp[key]['ss'][self.nx_riv:]
            self.ch_pars[key]['st'] = self.ch_pars[key]['st'][self.nx_riv:]
            self.ch_pars[key]['ut'] = self.ch_pars[key]['ut'][:,self.nx_riv:]
            self.ch_outp[key]['u_st'] = self.ch_outp[key]['u_st'][self.nx_riv:]
            self.ch_outp[key]['w_st'] = self.ch_outp[key]['w_st'][self.nx_riv:]
            self.ch_outp[key]['sb'] = self.ch_outp[key]['sb'][self.nx_riv:] 
            self.ch_outp[key]['sn'] = self.ch_outp[key]['sn'][self.nx_riv:]            
            self.ch_outp[key]['sb_x'] = self.ch_outp[key]['sb_x'][self.nx_riv:]            
            self.ch_outp[key]['eta_r'] = self.ch_outp[key]['eta_r'][self.nx_riv:]            
            self.ch_outp[key]['ut_r'] = self.ch_outp[key]['ut_r'][self.nx_riv:]
            self.ch_outp[key]['wt_r'] = self.ch_outp[key]['wt_r'][self.nx_riv:]
            self.ch_outp[key]['st_r'] = self.ch_outp[key]['st_r'][self.nx_riv:]
            
            self.ch_outp[key]['TQ'] = self.ch_outp[key]['TQ'][self.nx_riv:]
            self.ch_outp[key]['TE'] = self.ch_outp[key]['TE'][self.nx_riv:]
            self.ch_outp[key]['TD'] = self.ch_outp[key]['TD'][self.nx_riv:]
            self.ch_outp[key]['TT'] = self.ch_outp[key]['TT'][self.nx_riv:]
            
            tot_L = np.sum(self.ch_gegs[key]['L'][1:])
            
        #prepare some plotting quantities, x and y coordinates in map plots
        self.ch_outp[key]['plot d'] = np.zeros(len(self.ch_gegs[key]['plot x']))
        for i in range(1,len(self.ch_gegs[key]['plot x'])): self.ch_outp[key]['plot d'][i] = \
            self.ch_outp[key]['plot d'][i-1]+ ((self.ch_gegs[key]['plot x'][i]-self.ch_gegs[key]['plot x'][i-1])**2 + (self.ch_gegs[key]['plot y'][i]-self.ch_gegs[key]['plot y'][i-1])**2)**0.5
        self.ch_outp[key]['plot d'] = (self.ch_outp[key]['plot d']-self.ch_outp[key]['plot d'][-1])/self.ch_outp[key]['plot d'][-1]*tot_L
        self.ch_outp[key]['plot xs'] = np.interp(self.ch_outp[key]['px'],self.ch_outp[key]['plot d'],self.ch_gegs[key]['plot x'])
        self.ch_outp[key]['plot ys'] = np.interp(self.ch_outp[key]['px'],self.ch_outp[key]['plot d'],self.ch_gegs[key]['plot y'])
            
        self.ch_outp[key]['points'] = np.array([self.ch_outp[key]['plot xs'],self.ch_outp[key]['plot ys']]).T.reshape(-1, 1, 2)
        self.ch_outp[key]['segments'] = np.concatenate([self.ch_outp[key]['points'][:-1], self.ch_outp[key]['points'][1:]], axis=1)

        #depth averaged tidal salinity and current
        self.ch_outp[key]['utb_r'] = np.mean(self.ch_outp[key]['ut_r'],1)
        self.ch_outp[key]['stb_r'] = np.mean(self.ch_outp[key]['st_r'],1)

        #prepare boundary layer correction to tidal salinity
        self.ch_outp[key]['st_cor'] = self.ch_pars[key]['st'].copy()
        
        count+=1
               

    #boundary layer correction to tidal salinity
    for j in range(self.n_j):
        sigma = np.linspace(-1,0,self.nz)
        eps = self.Kh_tide/(self.omega)
        for key in self.ch_keys: 
            if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1): 
                #calculate x
                x_all = np.arange(self.ch_pars[key]['di'][1])*self.ch_gegs[key]['dx'][0] 
                x_bnd = x_all[np.where(x_all<(-np.sqrt(eps)*np.log(self.tol)))[0][1:]]
                #correction due to boundary layer
                st_cor = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(self.bbb_n[key][0] * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca                 
                self.ch_outp[key]['st_cor'][:len(st_cor)] = self.ch_outp[key]['st_cor'][:len(st_cor)] + st_cor

            if self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): 
                #calculate x
                x_all = np.arange(self.ch_pars[key]['di'][-2]-self.ch_pars[key]['di'][-1]+1,1)*self.ch_gegs[key]['dx'][-1] 
                x_bnd = x_all[np.where(x_all>(np.sqrt(eps)*np.log(self.tol)))[0][:-1]]
                if -x_bnd[0] > self.ch_gegs[key]['L'][-1]: print('ERROR: boundary layer too large. Can be solved...')
                #correction due to boundary layer
                st_cor = np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(self.bbb_n[key][1] * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
                self.ch_outp[key]['st_cor'][-len(st_cor):] = self.ch_outp[key]['st_cor'][-len(st_cor):] + st_cor
                
                    
    #quantities as time series in tidal cycle
    for key in self.ch_keys: 
        self.ch_outp[key]['st_cor_r'] = np.real(self.ch_outp[key]['st_cor'][:,:,np.newaxis] * np.exp(1j*self.omega*t))
        self.ch_outp[key]['stb_cor_r']= np.mean(self.ch_outp[key]['st_cor_r'],1)





