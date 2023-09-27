# =============================================================================
# module to solve the subtidal salinity balance in a general estuarine network
# model includes tides, with vertical advection, those are all taken into account 
# in the subtidal depth-averaged balance, but not in the the depth-perturbed balance
# at the junctions a boundary layer correction is applied, and in that manner salinity
# matches also at the tidal timescale. 
# =============================================================================

#import libraries
import numpy as np
import scipy as sp          #do fits
from scipy import optimize   , interpolate 
import time                              #measure time of operation
import matplotlib.pyplot as plt         


def saltmodel_ti(self,init, init_method = 'from_notide'):
    # =============================================================================
    # The function to calculate the salinity in the network.      
    # =============================================================================

    #create helpful dictionaries
    ch_parja = {} # parameters for solution vector/jacobian
    ch_inds = {} # indices for solution vector/jacobian

    #lists of ks and ns and pis
    kkp = np.linspace(1,self.N,self.N)*np.pi #k*pi
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    pkn = np.array([kkp]*self.N) + np.transpose([nnp]*self.N) #pi*(n+k)
    pnk = np.array([nnp]*self.N) - np.transpose([kkp]*self.N) #pi*(n-k)
    np.fill_diagonal(pkn,None),np.fill_diagonal(pnk,None)
     
    # =============================================================================
    # parameters for solution vector/jacobian
    # =============================================================================
    for key in self.ch_keys:
        ch_parja[key] = {}
        # =============================================================================
        # Vertical subtidal salt balance 
        # =============================================================================
        #term 1
        ch_parja[key]['C1a'] = self.ch_pars[key]['bH_1']/2
        
        #term 2
        ch_parja[key]['C2a'] = self.ch_pars[key]['bH_1'][:,np.newaxis] * (self.ch_pars[key]['g1']/2 + self.ch_pars[key]['g2']/6 + self.ch_pars[key]['g2']/(4*kkp**2))
        ch_parja[key]['C2b'] = self.ch_pars[key]['alf']*self.soc_sca/self.Lsc * (self.ch_pars[key]['g3']/2 + self.ch_pars[key]['g4']*(1/6 + 1/(4*kkp**2)) -self.ch_pars[key]['g5']*(1/8 + 3/(8*kkp**2)) )
        ch_parja[key]['C2c'] = self.ch_pars[key]['bH_1'][:,np.newaxis,np.newaxis]*self.ch_pars[key]['g2']* ( np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2 ) 
        ch_parja[key]['C2d'] = self.soc_sca/self.Lsc*self.ch_pars[key]['alf'] * (self.ch_pars[key]['g4']*(np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2) \
                                                + self.ch_pars[key]['g5']* ((3*np.cos(pkn)-3)/pkn**4 + (3*np.cos(pnk)-3)/pnk**4 - 3/2*np.cos(pkn)/pkn**2 - 3/2*np.cos(pnk)/pnk**2) )
        ch_parja[key]['C2c'][np.where(np.isnan(ch_parja[key]['C2c']))] = 0 
        ch_parja[key]['C2d'][np.where(np.isnan(ch_parja[key]['C2d']))] = 0 
            
        #term 3
        ch_parja[key]['C3a'] = 2*self.ch_pars[key]['bH_1'][:,np.newaxis]*self.ch_pars[key]['g2']/(kkp**2)*np.cos(kkp) 
        ch_parja[key]['C3b'] = self.ch_pars[key]['alf']*self.soc_sca/self.Lsc * ( 2*self.ch_pars[key]['g4']/kkp**2 * np.cos(kkp) - self.ch_pars[key]['g5']/kkp**4 *(6-6*np.cos(kkp) +3*kkp**2*np.cos(kkp)) )
        
        #term 4 does not exist 
        
        #term 5
        ch_parja[key]['C5a'] = self.ch_pars[key]['alf']*self.soc_sca/self.Lsc *kkp* ( -9*self.ch_pars[key]['g5']+6*self.ch_pars[key]['g4']+kkp**2*(-12*self.ch_pars[key]['g3']-4*self.ch_pars[key]['g4']+3*self.ch_pars[key]['g5']) ) / (48*kkp**3)
        ch_parja[key]['C5b'] = self.ch_pars[key]['alf']*self.soc_sca*self.ch_pars[key]['bex'][:,np.newaxis]**(-1)*kkp * ( -9*self.ch_pars[key]['g5']+6*self.ch_pars[key]['g4']+kkp**2*(-12*self.ch_pars[key]['g3']-4*self.ch_pars[key]['g4']+3*self.ch_pars[key]['g5']) ) / (48*kkp**3)    
        ch_parja[key]['C5c'] = self.ch_pars[key]['alf']*self.soc_sca/self.Lsc*nnp* ( (3*self.ch_pars[key]['g5']*np.cos(pkn)-3*self.ch_pars[key]['g5'])/pkn**5 + np.cos(pkn)/pkn**3 * \
                                                                     (self.ch_pars[key]['g4']-1.5*self.ch_pars[key]['g5']) + np.cos(pkn)/pkn * (self.ch_pars[key]['g5']/8 - self.ch_pars[key]['g4']/6 - self.ch_pars[key]['g3']/2) 
                                +(3*self.ch_pars[key]['g5']*np.cos(pnk)-3*self.ch_pars[key]['g5'])/pnk**5 + np.cos(pnk)/pnk**3 * (self.ch_pars[key]['g4']-1.5*self.ch_pars[key]['g5']) \
                                    + np.cos(pnk)/pnk * (self.ch_pars[key]['g5']/8 - self.ch_pars[key]['g4']/6 - self.ch_pars[key]['g3']/2))
        ch_parja[key]['C5d'] = self.ch_pars[key]['alf']*self.soc_sca*nnp*self.ch_pars[key]['bex'][:,np.newaxis,np.newaxis]**(-1)*( (3*self.ch_pars[key]['g5']*np.cos(pkn)-3*self.ch_pars[key]['g5'])/pkn**5 + np.cos(pkn)/pkn**3 * (self.ch_pars[key]['g4']-1.5*self.ch_pars[key]['g5']) 
                                    + np.cos(pkn)/pkn * (self.ch_pars[key]['g5']/8 - self.ch_pars[key]['g4']/6 - self.ch_pars[key]['g3']/2)
                                    +(3*self.ch_pars[key]['g5']*np.cos(pnk)-3*self.ch_pars[key]['g5'])/pnk**5 + np.cos(pnk)/pnk**3 * (self.ch_pars[key]['g4']-1.5*self.ch_pars[key]['g5'])\
                                        + np.cos(pnk)/pnk * (self.ch_pars[key]['g5']/8 - self.ch_pars[key]['g4']/6 - self.ch_pars[key]['g3']/2))
        #no wind
        ch_parja[key]['C5e'] = np.zeros((np.sum(self.ch_pars[key]['nxn']),self.M))
        ch_parja[key]['C5f'] = np.zeros((np.sum(self.ch_pars[key]['nxn']),self.M,self.M))
        
        ch_parja[key]['C5c'][np.where(np.isnan(ch_parja[key]['C5c']))] = 0 
        ch_parja[key]['C5d'][np.where(np.isnan(ch_parja[key]['C5d']))] = 0 
        #ch_parja[key]['C5f'][np.where(np.isnan(ch_parja[key]['C5f']))] = 0  #no wind
        
        #term 6
        ch_parja[key]['C6a'] = self.Lsc*self.ch_pars[key]['Kv']*kkp**2/(2*self.ch_gegs[key]['H']**2)        
        
        #term 7
        ch_parja[key]['C7a'] = -self.ch_pars[key]['bex']**-1 * self.ch_pars[key]['Kh'] / 2 - self.ch_pars[key]['Kh_x']/2
        ch_parja[key]['C7b'] = -self.ch_pars[key]['Kh']/(2*self.Lsc)
        

        #term 8 and 9 do not exist
        
        # =============================================================================
        # Horizontal subtidal salt balance
        # =============================================================================
        ch_parja[key]['C10a'] = (-self.ch_pars[key]['bex']**(-1)*self.ch_pars[key]['Kh'] - self.ch_pars[key]['Kh_x'])
        ch_parja[key]['C10b'] = -self.ch_pars[key]['Kh']/self.Lsc
        ch_parja[key]['C10c'] = self.ch_pars[key]['bex'][:,np.newaxis]**(-1)*self.ch_pars[key]['alf']*self.soc_sca * ( 2*self.ch_pars[key]['g4']/nnp**2*np.cos(nnp) - self.ch_pars[key]['g5']/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        ch_parja[key]['C10d'] = self.ch_pars[key]['alf']*self.soc_sca/self.Lsc * ( 2*self.ch_pars[key]['g4']/nnp**2*np.cos(nnp) - self.ch_pars[key]['g5']/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        ch_parja[key]['C10e'] = (2*self.ch_pars[key]['bH_1'][:,np.newaxis]*self.ch_pars[key]['g2']) / nnp**2 * np.cos(nnp) 
        ch_parja[key]['C10f'] = self.ch_pars[key]['alf']*self.soc_sca/self.Lsc * ( 2*self.ch_pars[key]['g4']/nnp**2*np.cos(nnp) - self.ch_pars[key]['g5']/nnp**4 * (6-6*np.cos(nnp)+3*nnp**2*np.cos(nnp)) )
        ch_parja[key]['C10g'] = np.zeros((np.sum(self.ch_pars[key]['nxn']),self.M)) #no wind
        ch_parja[key]['C10h'] = self.ch_pars[key]['bH_1']
        
        # =============================================================================
        # Boundaries: transport calculations
        # =============================================================================
        #vertical
        ch_parja[key]['C12a_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*0.5*self.ch_pars[key]['Kh'][0]
        ch_parja[key]['C12b_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.ch_pars[key]['alf']*self.soc_sca*kkp* ( self.ch_pars[key]['g5']*(kkp**2-3)/(16*kkp**3) + self.ch_pars[key]['g4']*(3-2*kkp**2)/(24*kkp**3) - self.ch_pars[key]['g3']/(4*kkp) )
        ch_parja[key]['C12c_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.ch_pars[key]['alf']*self.soc_sca*nnp* ( self.ch_pars[key]['g5']/8*(np.cos(pkn)/pkn**5*(24-12*pkn**2+pkn**4) - 24/pkn**5 + np.cos(pnk)/pnk**5*(24-12*pnk**2+pnk**4) - 24/pnk**5 )
                                          +self.ch_pars[key]['g4']/6*((6-pkn**2)*np.cos(pkn)/pkn**3  + (6-pnk**2)*np.cos(pnk)/pnk**3 ) + self.ch_pars[key]['g3']/2*(np.cos(pkn)/pkn + np.cos(pnk)/pnk) )
        ch_parja[key]['C12d_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.Lsc*self.ch_pars[key]['bH_1'][0]*(0.5+0.5*self.ch_pars[key]['g1']+self.ch_pars[key]['g2']*(1/6+1/(4*kkp**2)))
        ch_parja[key]['C12e_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.Lsc*self.ch_pars[key]['bH_1'][0]* (np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2)
        ch_parja[key]['C12f_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.Lsc*self.ch_pars[key]['bH_1'][0]* 2*self.ch_pars[key]['g2']*np.cos(kkp)/kkp**2 
        ch_parja[key]['C12g_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.ch_pars[key]['alf']*self.soc_sca * (self.ch_pars[key]['g4']*2*np.cos(kkp)/kkp**2 + self.ch_pars[key]['g5']*((6*np.cos(kkp)-1)/kkp**4 - 3*np.cos(kkp)/kkp**2))
        
    
        ch_parja[key]['C12a_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*0.5*self.ch_pars[key]['Kh'][-1]
        ch_parja[key]['C12b_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['alf']*self.soc_sca*kkp* ( self.ch_pars[key]['g5']*(kkp**2-3)/(16*kkp**3) + self.ch_pars[key]['g4']*(3-2*kkp**2)/(24*kkp**3) - self.ch_pars[key]['g3']/(4*kkp) )
        ch_parja[key]['C12c_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['alf']*self.soc_sca*nnp* ( self.ch_pars[key]['g5']/8*(np.cos(pkn)/pkn**5*(24-12*pkn**2+pkn**4) - 24/pkn**5 + np.cos(pnk)/pnk**5*(24-12*pnk**2+pnk**4) - 24/pnk**5 )
                                          +self.ch_pars[key]['g4']/6*((6-pkn**2)*np.cos(pkn)/pkn**3  + (6-pnk**2)*np.cos(pnk)/pnk**3 ) + self.ch_pars[key]['g3']/2*(np.cos(pkn)/pkn + np.cos(pnk)/pnk) )
        ch_parja[key]['C12d_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.Lsc*self.ch_pars[key]['bH_1'][-1]*(0.5+0.5*self.ch_pars[key]['g1']+self.ch_pars[key]['g2']*(1/6+1/(4*kkp**2)))
        ch_parja[key]['C12e_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.Lsc*self.ch_pars[key]['bH_1'][-1]* (np.cos(pkn)/pkn**2 + np.cos(pnk)/pnk**2)
        ch_parja[key]['C12f_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.Lsc*self.ch_pars[key]['bH_1'][-1]* 2*self.ch_pars[key]['g2']*np.cos(kkp)/kkp**2 
        ch_parja[key]['C12g_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['alf']*self.soc_sca * (self.ch_pars[key]['g4']*2*np.cos(kkp)/kkp**2 + self.ch_pars[key]['g5']*((6*np.cos(kkp)-1)/kkp**4 - 3*np.cos(kkp)/kkp**2))

        ch_parja[key]['C12c_x=-L'][np.where(np.isnan(ch_parja[key]['C12c_x=-L']))] = 0 
        ch_parja[key]['C12c_x=0'][np.where(np.isnan(ch_parja[key]['C12c_x=0']))] = 0 
        ch_parja[key]['C12e_x=-L'][np.where(np.isnan(ch_parja[key]['C12e_x=-L']))] = 0 
        ch_parja[key]['C12e_x=0'][np.where(np.isnan(ch_parja[key]['C12e_x=0']))] = 0    
        
        # depth-averaged 
        ch_parja[key]['C13a_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.ch_pars[key]['bH_1'][0]
        ch_parja[key]['C13b_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.ch_pars[key]['bH_1'][0]*2*self.ch_pars[key]['g2']*np.cos(nnp)/nnp**2
        ch_parja[key]['C13c_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*self.ch_pars[key]['alf']*self.soc_sca/self.Lsc * (2*self.ch_pars[key]['g4']*np.cos(nnp)/nnp**2 + self.ch_pars[key]['g5']*(6*np.cos(nnp)-6)/nnp**4 -self.ch_pars[key]['g5']*3*np.cos(nnp)/nnp**2 )
        ch_parja[key]['C13d_x=-L'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]*-self.ch_pars[key]['Kh'][0]/self.Lsc
    
        ch_parja[key]['C13a_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['bH_1'][-1]
        ch_parja[key]['C13b_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['bH_1'][-1]*2*self.ch_pars[key]['g2']*np.cos(nnp)/nnp**2
        ch_parja[key]['C13c_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*self.ch_pars[key]['alf']*self.soc_sca/self.Lsc * (2*self.ch_pars[key]['g4']*np.cos(nnp)/nnp**2 + self.ch_pars[key]['g5']*(6*np.cos(nnp)-6)/nnp**4 -self.ch_pars[key]['g5']*3*np.cos(nnp)/nnp**2 )
        ch_parja[key]['C13d_x=0'] = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1]*-self.ch_pars[key]['Kh'][-1]/self.Lsc
    
        # =============================================================================
        # indices for solution vector/jacobian
        # =============================================================================
        ch_inds[key] = {}
        #how this exactly works I find it very hard to write down. It is the allocation of the different terms in the solution vector and jacobian. 
        
        ch_inds[key]['di_l'] = np.zeros((len(self.ch_pars[key]['di'])-2)*2)
        for i in range(1,len(self.ch_pars[key]['di'])-1):
            ch_inds[key]['di_l'][i*2-2] = self.ch_pars[key]['di'][i]-1
            ch_inds[key]['di_l'][i*2-1] = self.ch_pars[key]['di'][i]
        ch_inds[key]['di_l'] = np.array(ch_inds[key]['di_l'], dtype=int)
    
        ch_inds[key]['x'] = np.delete(np.arange(self.ch_pars[key]['di'][-1]),ch_inds[key]['di_l'])[1:-1] # x coordinates for the points which are not on a aboundary
        ch_inds[key]['xr'] = ch_inds[key]['x'].repeat(self.N) # x for N values, mostly i in old code
        ch_inds[key]['xr_m'] = ch_inds[key]['xr']*self.M #M*i in old coe
        ch_inds[key]['xrm_m'] = (ch_inds[key]['xr']-1)*self.M
        ch_inds[key]['xrp_m'] = (ch_inds[key]['xr']+1)*self.M
        ch_inds[key]['xr_mj'] = ch_inds[key]['xr_m']+np.tile(np.arange(1,self.M),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))
        ch_inds[key]['xrm_mj'] = ch_inds[key]['xrm_m']+np.tile(np.arange(1,self.M),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))
        ch_inds[key]['xrp_mj'] = ch_inds[key]['xrp_m']+np.tile(np.arange(1,self.M),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))
        ch_inds[key]['jl'] = np.tile(np.arange(self.N),self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))
    
        #for the k things we need to repeat some arrays - mysterious comment
        ch_inds[key]['xr_mjr'] = np.repeat(ch_inds[key]['xr_m'],self.N)+np.tile(np.arange(1,self.M),(self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))).repeat(self.N)
        ch_inds[key]['xrr'] = ch_inds[key]['xr'].repeat(self.N)
        ch_inds[key]['jlr'] = ch_inds[key]['jl'].repeat(self.N)
        ch_inds[key]['xr_mr'] = ch_inds[key]['xr_m'].repeat(self.N)
        ch_inds[key]['xrp_mr'] = ch_inds[key]['xrp_m'].repeat(self.N)
        ch_inds[key]['xrm_mr'] = ch_inds[key]['xrm_m'].repeat(self.N)
    
        ch_inds[key]['kl'] = np.tile(ch_inds[key]['jl'],self.N)
        ch_inds[key]['xr_mk'] = np.repeat(ch_inds[key]['xr_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),(self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))),self.N)
        ch_inds[key]['xrp_mk'] = np.repeat(ch_inds[key]['xrp_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),(self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))),self.N)
        ch_inds[key]['xrm_mk'] = np.repeat(ch_inds[key]['xrm_m'],self.N)+np.tile(np.tile(np.arange(1,self.M),(self.ch_pars[key]['di'][-1]-2-(len(self.ch_pars[key]['di'][1:-1])*2))),self.N)
    
        ch_inds[key]['x_m'] = ch_inds[key]['x']*self.M
        ch_inds[key]['xp_m'] = (ch_inds[key]['x']+1)*self.M
        ch_inds[key]['xm_m'] = (ch_inds[key]['x']-1)*self.M
        
        #boundaries of segments
        ch_inds[key]['in_bnd'] = (self.ch_pars[key]['di'][np.arange(1,len(self.ch_pars[key]['nxn']))]*self.M+np.arange(self.M)[:,np.newaxis]).flatten()
    
        #boundaries
        ch_inds[key]['bnd1_x=-L'] = 0*self.M+np.arange(self.M) 
        ch_inds[key]['bnd2_x=-L'] = 1*self.M+np.arange(self.M) 
        ch_inds[key]['bnd3_x=-L'] = 2*self.M+np.arange(self.M) 
        ch_inds[key]['bnd4_x=-L'] = 3*self.M+np.arange(self.M) 
    
        ch_inds[key]['bnd1_x=0'] = (self.ch_pars[key]['di'][-1]-1)*self.M+np.arange(self.M)
        ch_inds[key]['bnd2_x=0'] = (self.ch_pars[key]['di'][-1]-2)*self.M+np.arange(self.M)
        ch_inds[key]['bnd3_x=0'] = (self.ch_pars[key]['di'][-1]-3)*self.M+np.arange(self.M)
        ch_inds[key]['bnd4_x=0'] = (self.ch_pars[key]['di'][-1]-4)*self.M+np.arange(self.M)
                
        #for the sea domain
        ch_inds[key]['xs'] = np.arange(self.ch_pars[key]['di'][-2],self.ch_pars[key]['di'][-1])[1:-1] # x coordinates for the points which are not on a aboundary
        ch_inds[key]['x0'] = np.array([self.ch_pars[key]['di'][-2]-3,self.ch_pars[key]['di'][-2]-2,self.ch_pars[key]['di'][-2]-1])
        ch_inds[key]['xs_m'] = ch_inds[key]['xs']*self.M
        ch_inds[key]['x0_m'] = ch_inds[key]['x0']*self.M
        ch_inds[key]['x0n'] = self.ch_pars[key]['di'][-2]*self.M+np.arange(1,self.M)

    # =============================================================================
    # extra indices, required because of the addtion of the matching at the tidal timescale
    # Misschien had dit toch handiger gekund haha. Nouja kan altijd nog later. 
    # =============================================================================
    ind,ind2 = [0] , [0] 
    count = 0

    for key in self.ch_keys: 
        #number of unknows
        n_unk_ch = ind[-1]+self.ch_pars[key]['di'][-1]*self.M
        begin,end = 0,0
        if self.ch_gegs[key]['loc x=-L'][0] =='j': begin = self.M*2
        if self.ch_gegs[key]['loc x=0'][0] =='j': end = self.M*2
        
        n_unk_ch = n_unk_ch + begin+end
        ind.append(n_unk_ch)
        ind2.append(ind2[-1]+self.ch_pars[key]['di'][-1]*self.M) 
        
        #de locaties van het zout in de totale oplossingsvector
        ch_inds[key]['s_intot'] = [ind[count]+begin,ind[count+1]-end]
        #de allocatie van de Bs bij de grenzen, alleen als ze bestaan dus
        ch_inds[key]['B_intot'] = {}
        ch_inds[key]['B_intot']['loc x=-L'] = [ind[count  ],ind[count  ]+self.M*2] if self.ch_gegs[key]['loc x=-L'][0] =='j' else None
        ch_inds[key]['B_intot']['loc x=0'] = [ind[count+1]-self.M*2,ind[count+1]] if self.ch_gegs[key]['loc x=0'][0] =='j' else None
        #de grenzen van de verschillende kanalen, met zowel zout als Bs erin
        ch_inds[key]['sB_intot'] = [ind[count],ind[count+1]]
        #de allocatie alsof er geen Bs zouden zijn, de ouderwetse
        ch_inds[key]['s_inold'] = [ind2[count],ind2[count+1]]
        
        count+=1
        
    del ind,ind2

    def conv_ans(ans, key): 
        # =============================================================================
        # function to convert the answer function to salinities: sn, dsndx, dsbdx, dsbdx2
        # =============================================================================
        #prepare 
        ss = ans*self.soc_sca
        inds = ch_inds[key].copy()
        dx = self.ch_pars[key]['dl']*self.Lsc
        di_here = self.ch_pars[key]['di']
        #make empty vectors.
        dsbdx,dsbdx2 = np.zeros(di_here[-1])+np.nan , np.zeros(di_here[-1])+np.nan
        dsndx = np.zeros((di_here[-1]*self.M)) +np.nan
      
        #derivatives in interior
        dsbdx[inds['x']] = (ss[inds['xp_m']]-ss[inds['xm_m']])/(2*dx[inds['x']]) 
        dsbdx2[inds['x']] = (ss[inds['xp_m']]-2*ss[inds['x_m']]+ss[inds['xm_m']])/(dx[inds['x']]**2)         
        dsndx[inds['xr_mj']] = (ss[inds['xrp_mj']]-ss[inds['xrm_mj']])/(2*dx[inds['xr']])

        #derivatives left boundary
        for d in di_here[:-1]:
            dsbdx[d] = (-3*ss[(d+0)*self.M] + 4*ss[(d+1)*self.M] - ss[(d+2)*self.M]) / (2*dx[d])
            dsbdx2[d] = (2*ss[(d+0)*self.M] - 5*ss[(d+1)*self.M] + 4*ss[(d+2)*self.M] - ss[(d+3)*self.M]) / (dx[d]**2)
            dsndx[d*self.M+1:(d+1)*self.M] = (-3*ss[(d+0)*self.M+1:(d+1)*self.M] + 4*ss[(d+1)*self.M+1:(d+2)*self.M] - ss[(d+2)*self.M+1:(d+3)*self.M]) / (2*dx[d])
        #derivatives right boundary
        for d in di_here[1:]:
            dsbdx[d-1] = (3*ss[(d-1)*self.M] - 4*ss[(d-2)*self.M] + ss[(d-3)*self.M]) / (2*dx[d-1])
            dsbdx2[d-1] = (2*ss[(d-1)*self.M] - 5*ss[(d-2)*self.M] + 4*ss[(d-3)*self.M] - ss[(d-4)*self.M]) / (dx[d-1]**2)
            dsndx[(d-1)*self.M+1:d*self.M] = (3*ss[(d-1)*self.M+1:(d-0)*self.M] - 4*ss[(d-2)*self.M+1:(d-1)*self.M] + ss[(d-3)*self.M+1:(d-2)*self.M]) / (2*dx[d-1])
            
        #put in right shape
        sn = ss.reshape((di_here[-1],self.M))[:,1:].T
        dsndx = dsndx.reshape((di_here[-1],self.M))[:,1:].T
        d_ans = (sn[:,:,np.newaxis],dsndx[:,:,np.newaxis],dsbdx[np.newaxis,:,np.newaxis],dsbdx2[np.newaxis,:,np.newaxis])
        
        return d_ans
    
    # =============================================================================
    # define functions to solve in the Newton-Raphson algoritm
    # =============================================================================
        
    def sol_inner(ans, bbs, key):
        # =============================================================================
        # function to build the internal part of the solution vector 
        # =============================================================================
        
        #create empty vector
        so = np.zeros(self.ch_pars[key]['di'][-1]*self.M)
        #local variables, for shorter notation
        inds = ch_inds[key].copy()
        dl = self.ch_pars[key]['dl'].copy()
        pars = ch_parja[key].copy()
              
        # =============================================================================
        # subtidal part    
        # =============================================================================
        #vertical
        #term 1
        so[inds['xr_mj']] = so[inds['xr_mj']] + pars['C1a'][inds['xr']] * self.ch_pars[key]['Q'] * (ans[inds['xrp_mj']]-ans[inds['xrm_mj']])/(2*dl[inds['xr']])

        #term 2
        so[inds['xr_mj']] = so[inds['xr_mj']] + (pars['C2a'][inds['xr'],inds['jl']]*self.ch_pars[key]['Q']*(ans[inds['xrp_mj']] - 
                            ans[inds['xrm_mj']])/(2*dl[inds['xr']]) + pars['C2b'][inds['jl']] * 
                            ((ans[inds['xrp_mj']]-ans[inds['xrm_mj']])/(2*dl[inds['xr']]))*((ans[inds['xrp_m']]-ans[inds['xrm_m']]) /
                            (2*dl[inds['xr']])) + np.sum([pars['C2c'][inds['xr'],inds['jl'],n-1] * self.ch_pars[key]['Q'] * 
                            (ans[inds['xrp_m']+n]-ans[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(2*dl[inds['xr']])  
                            +np.sum([pars['C2d'][inds['jl'],n-1] * (ans[inds['xrp_m']+n]-ans[inds['xrm_m']+n]) 
                            for n in range(1,self.M)],0)/(2*dl[inds['xr']]) * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']])
                             )
        
        #term 3
        so[inds['xr_mj']] = so[inds['xr_mj']] + pars['C3a'][inds['xr'],inds['jl']] * self.ch_pars[key]['Q'] * (ans[inds['xrp_m']]-ans[inds['xrm_m']]) /  \
                    (2*dl[inds['xr']]) + pars['C3b'][inds['jl']] * ((ans[inds['xrp_m']]-ans[inds['xrm_m']])/ (2*dl[inds['xr']]))**2
        
        #term 4 is absent
        
        #term 5
        so[inds['xr_mj']] = so[inds['xr_mj']] + (pars['C5a'][inds['jl']]*(ans[inds['xrp_m']] - 2*ans[inds['xr_m']] 
                                     + ans[inds['xrm_m']])/(dl[inds['xr']]**2)*ans[inds['xr_mj']] 
                                     + pars['C5b'][inds['xr'],inds['jl']]*(ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']]) * ans[inds['xr_mj']]  
                                       +np.sum([pars['C5c'][inds['jl'],n-1]*ans[inds['xr_m']+n] for n in range(1,self.M)],0) * (ans[inds['xrp_m']] - 2*ans[inds['xr_m']] 
                                        + ans[inds['xrm_m']])/(dl[inds['xr']]**2) 
                                       +np.sum([pars['C5d'][inds['xr'],inds['jl'],n-1]*ans[inds['xr_m']+n] for n in range(1,self.M)],0) * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']])
                                       + pars['C5e'][inds['xr'],inds['jl']]*ans[inds['xr_mj']] + np.sum([pars['C5f'][inds['xr'],inds['jl'],n-1]*ans[inds['xr_m']+n] for n in range(1,self.M)],0)  )
        
        #term 6
        so[inds['xr_mj']] = so[inds['xr_mj']] + pars['C6a'][inds['jl']]*ans[inds['xr_mj']]

        #term 7    
        so[inds['xr_mj']] = (so[inds['xr_mj']] + pars['C7a'][inds['xr']]*(ans[inds['xrp_mj']]-ans[inds['xrm_mj']])/(2*dl[inds['xr']])  
                                     + pars['C7b'][inds['xr']]*(ans[inds['xrp_mj']] - 2*ans[inds['xr_mj']] + ans[inds['xrm_mj']])/(dl[inds['xr']]**2) )

        #depth - averaged
        so[inds['x_m']] = so[inds['x_m']] + (pars['C10a'][inds['x']]*(ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) 
                            + pars['C10b'][inds['x']]*(ans[inds['xp_m']] - 2*ans[inds['x_m']] + ans[inds['xm_m']])/(dl[inds['x']]**2) 
                            + (ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) * np.sum([pars['C10c'][inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                            + (ans[inds['xp_m']] - 2*ans[inds['x_m']] + ans[inds['xm_m']])/(dl[inds['x']]**2) * np.sum([pars['C10d'][n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                            + np.sum([pars['C10e'][inds['x'],n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*dl[inds['x']]) for n in range(1,self.M)],0) * self.ch_pars[key]['Q'] 
                            + (ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) 
                            * np.sum([pars['C10f'][n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*dl[inds['x']]) for n in range(1,self.M)],0)
                            + np.sum([pars['C10g'][inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) + pars['C10h'][inds['x']]*self.ch_pars[key]['Q']*(ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) )

        #boundaries of segments - this for loop may live on as a living fossil. 
        for d in range(1,len(self.ch_pars[key]['nxn'])):
            s_3, s_2, s_1 = ans[(self.ch_pars[key]['di'][d]-3)*self.M:(self.ch_pars[key]['di'][d]-2)*self.M], ans[(self.ch_pars[key]['di'][d]-2)*self.M:(self.ch_pars[key]['di'][d]-1)*self.M], ans[(self.ch_pars[key]['di'][d]-1)*self.M:self.ch_pars[key]['di'][d]*self.M]
            s0, s1, s2 = ans[self.ch_pars[key]['di'][d]*self.M:(self.ch_pars[key]['di'][d]+1)*self.M], ans[(self.ch_pars[key]['di'][d]+1)*self.M:(self.ch_pars[key]['di'][d]+2)*self.M], ans[(self.ch_pars[key]['di'][d]+2)*self.M:(self.ch_pars[key]['di'][d]+3)*self.M]
            so[(self.ch_pars[key]['di'][d]-1)*self.M:(self.ch_pars[key]['di'][d]-0)*self.M] = (3*s_1-4*s_2+s_3)/(2*dl[self.ch_pars[key]['di'][d]-1]) - (-3*s0+4*s1-s2)/(2*dl[self.ch_pars[key]['di'][d]])
            so[(self.ch_pars[key]['di'][d]-0)*self.M:(self.ch_pars[key]['di'][d]+1)*self.M] = s_1 - s0
        
        # =============================================================================
        # tidal part
        # =============================================================================

        #create input for tidal module
        tid_inp = conv_ans(ans, key)
        #run tidal module
        tid_otp = self.ch_tide['sol_i'](key,*tid_inp)
        #add the results of the tidal module to the solution vector - depth-averaged
        so[inds['x_m']] = so[inds['x_m']] + tid_otp[inds['x']]    

        #add tides in sea part
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : 
            tid_otp_sea = self.ch_tide['sol_s'](key,*tid_inp)
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : 
            tid_otp_sea = self.ch_tide['sol_s'](key,*tid_inp)
            so[inds['x_m']] = so[inds['x_m']] + tid_otp_sea[inds['x']]    
        

        #add the correction due to the boundary layer in the internal part. 
        if self.ch_gegs[key]['loc x=-L'][0] =='j': 
            #Depth-averaged
            Tcor = self.ch_tide['sol_i_cor'](key,bbs[0],'loc x=-L','da')
            ind_cor = inds['x_m'][:len(Tcor)]
            so[ind_cor] = so[ind_cor] + Tcor
            #At depth levels
            Tcor = self.ch_tide['sol_i_cor'](key,bbs[0],'loc x=-L','dv')
            ind_cor = inds['xr_mj'][:len(Tcor.flatten())]
            i1 = np.tile(np.arange(self.N),len(Tcor[0]))
            i2 = np.repeat(np.arange(len(Tcor[0])),self.N)
            so[ind_cor] += Tcor[i1,i2] 

                
                
        if self.ch_gegs[key]['loc x=0' ][0] =='j': 
            #Depth-averaged
            Tcor = self.ch_tide['sol_i_cor'](key,bbs[1],'loc x=0','da')
            ind_cor = inds['x_m'][-len(Tcor):]
            so[ind_cor] = so[ind_cor] + Tcor           

            #At depth levels
            Tcor = self.ch_tide['sol_i_cor'](key,bbs[1],'loc x=0','dv')
            ind_cor = inds['xr_mj'][-len(Tcor.flatten()):]
            i1 = np.tile(np.arange(self.N),len(Tcor[0]))
            i2 = np.repeat(np.arange(len(Tcor[0])),self.N)
            so[ind_cor] += Tcor[i1,i2]    
                    
        #tides in vertical balance.
        t1_sp, t2_sp, t3_sp = self.ch_tide['solz_i'](key,*tid_inp)
        so[inds['xr_mj']] = so[inds['xr_mj']] + t1_sp[inds['jl'],inds['xr']] + t2_sp[inds['jl'],inds['xr']] + t3_sp[inds['jl'],inds['xr']]
        
        return so
    
    
    def sol_complete(ans_all):
        # =============================================================================
        # function to build complete solution vector
        # =============================================================================
            
        #first: break down ans_all to ch_ans
        ch_ans = {}
        for key in self.ch_keys: ch_ans[key] = ans_all[ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1]]
        
        #run the internal solution vectors and add to dictionary
        ch_sols = {}
        for key in self.ch_keys: 
            #select boundary layer correction
            bbs = [None, None]
            if self.ch_gegs[key]['loc x=-L'][0] =='j': bbs[0] = ans_all[ch_inds[key]['B_intot']['loc x=-L'][0] : ch_inds[key]['B_intot']['loc x=-L'][1]]
            if self.ch_gegs[key]['loc x=0'][0] =='j':  bbs[1] = ans_all[ch_inds[key]['B_intot']['loc x=0'][0] : ch_inds[key]['B_intot']['loc x=0'][1]]
            #run inner solution
            ch_sols[key] = sol_inner(ch_ans[key], bbs, key)
            
        #Load the tides, i.e. calculate the tidal transport at the boundaries for every channel. 
        tid_otp = {}
        for key in self.ch_keys: 
            tid_inp = conv_ans(ch_ans[key], key)
            tid_otp[key] = self.ch_tide['sol_b'](key,*tid_inp)
        
        # =============================================================================
        # Add the boundary conditions to the solution vector
        # =============================================================================
        for key in self.ch_keys:
            
            #river boundaries _ sriv prescribed
            if self.ch_gegs[key]['loc x=-L'][0] == 'r':
                ch_sols[key][0] = ch_ans[key][0] - self.sri[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca
                ch_sols[key][1:self.M] = ch_ans[key][1:self.M] 
            elif self.ch_gegs[key]['loc x=0'][0] == 'r':
                ch_sols[key][-self.M] = ch_ans[key][-self.M] - self.sri[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca
                ch_sols[key][-self.N:] = ch_ans[key][-self.N:] 
               
            #weir boundaries: flux kind of prescribed
            if self.ch_gegs[key]['loc x=-L'][0] == 'w':
                if self.ch_pars[key]['Q']>0: #flux is equal to the advective flux through weir
                    ch_sols[key][0] = self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*ch_ans[key][0]*self.Lsc - self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*self.swe[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca*self.Lsc \
                        - self.ch_pars[key]['Kh'][0] * (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)]) + self.ch_pars[key]['bH_1'][0]*tid_otp[key][0][0]            
                elif self.ch_pars[key]['Q']<=0: #flux is equal to the advective flux through weir, set by river salinity
                    ch_sols[key][0] =  - self.ch_pars[key]['Kh'][0] * (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)]) + self.ch_pars[key]['bH_1'][0]*tid_otp[key][0][0]   
                #no flux at depth - dit is een beetje slordig...
                ch_sols[key][1:self.M] = (-3*ch_ans[key][1:self.M]+4*ch_ans[key][self.M+1:2*self.M]-ch_ans[key][2*self.M+1:3*self.M])/(2*self.ch_pars[key]['dl'][0]) 
                
            elif self.ch_gegs[key]['loc x=0'][0] == 'w':
                if self.ch_pars[key]['Q']>=0: #flux is equal to the advective flux through weir
                    ch_sols[key][-self.M] = - self.ch_pars[key]['Kh'][-1] * (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][-self.M+n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)]) + self.ch_pars[key]['bH_1'][-1]*tid_otp[key][0][1]   
                elif self.ch_pars[key]['Q']<0: #flux is equal to the advective flux through weir, set by weir salinity
                    ch_sols[key][-self.M] = self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*ch_ans[key][-self.M]*self.Lsc - self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*self.swe[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca*self.Lsc \
                        - self.ch_pars[key]['Kh'][-1] * (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][-self.M+n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)]) + self.ch_pars[key]['bH_1'][-1]*tid_otp[key][0][1]  
                 #no flux at depth - dit is een beetje slordig...
                ch_sols[key][-self.N:] =  (3*ch_ans[key][-self.N:]-4*ch_ans[key][-2*self.M+1:-self.M]+ch_ans[key][-3*self.M+1:-2*self.M])/(2*self.ch_pars[key]['dl'][-1]) 
                    
               
            #har boundaries: only seaward flux 
            if self.ch_gegs[key]['loc x=-L'][0] == 'h':
                #only seaward flux
                ch_sols[key][0] = - self.ch_pars[key]['Kh'][0] * (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                    + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                    + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                    + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])  + self.ch_pars[key]['bH_1'][0]*tid_otp[key][0][0]  
                #no flux at depth 
                ch_sols[key][1:self.M] = (-3*ch_ans[key][1:self.M]+4*ch_ans[key][self.M+1:2*self.M]-ch_ans[key][2*self.M+1:3*self.M])/(2*self.ch_pars[key]['dl'][0]) 

            elif self.ch_gegs[key]['loc x=0'][0] == 'h':
                #only seaward flux
                ch_sols[key][-self.M] = - self.ch_pars[key]['Kh'][-1] * (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                    + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][self.M+n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                    + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                    + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])  + self.ch_pars[key]['bH_1'][-1]*tid_otp[key][0][1]  
                #no diffusive flux at depth
                ch_sols[key][-self.N:] =  (3*ch_ans[key][-self.N:]-4*ch_ans[key][-2*self.M+1:-self.M]+ch_ans[key][-3*self.M+1:-2*self.M])/(2*self.ch_pars[key]['dl'][-1]) 
  
            #sea boundaries - soc prescribed
            if self.ch_gegs[key]['loc x=-L'][0] == 's':
                ch_sols[key][0] = ch_ans[key][0] - self.soc[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca
                ch_sols[key][1:self.M] = ch_ans[key][1:self.M] 
            elif self.ch_gegs[key]['loc x=0'][0] == 's':
                ch_sols[key][-self.M] = ch_ans[key][-self.M] - self.soc[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca
                ch_sols[key][-self.N:] = ch_ans[key][-self.N:] 

        # =============================================================================
        # Conditions at the junction
        # =============================================================================
        for j in range(self.n_j):
            #find the connections
            ju_geg = []
            for key in self.ch_keys: 
                if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1):  ju_geg.append([key,'loc x=-L',ch_inds[key]['bnd1_x=-L'],ch_inds[key]['bnd2_x=-L'],ch_inds[key]['bnd3_x=-L'],ch_inds[key]['bnd4_x=-L']])
                elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): ju_geg.append([key,'loc x=0',ch_inds[key]['bnd1_x=0'],ch_inds[key]['bnd2_x=0'],ch_inds[key]['bnd3_x=0'],ch_inds[key]['bnd4_x=0']])
     
            #the coupling is: 0: 0-1 equal. 1: 1-2 equal. 2: transport conserved
            
            #eerst: 0-1 gelijk
            ch_sols[ju_geg[0][0]][ju_geg[0][2]] = ch_ans[ju_geg[0][0]][ju_geg[0][2]] - ch_ans[ju_geg[1][0]][ju_geg[1][2]]
            
            #second: 1-2 equal
            ch_sols[ju_geg[1][0]][ju_geg[1][2]] = ch_ans[ju_geg[1][0]][ju_geg[1][2]] - ch_ans[ju_geg[2][0]][ju_geg[2][2]]
            
            #third: transport
            #for depth-averaged transport
            temp = 0
            for i in range(3): #calculate contributions from channels seperately 
                key_here = ju_geg[i][0]
                #tidal contribution with boundary layer correction - implementation is a little different from before. not consistent... maybe change at some point
                bb = ans_all[ch_inds[key_here]['B_intot'][ju_geg[i][1]][0] : ch_inds[key_here]['B_intot'][ju_geg[i][1]][1]].copy()
                tid_cor = self.ch_tide['sol_b_cor'](key_here, bb, ju_geg[i][1], 'da')
                
                if ju_geg[i][1] == 'loc x=-L':
                    temp = temp - (ch_parja[key_here]['C13a_x=-L']*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]] + np.sum(ch_ans[key_here][ju_geg[i][2][1:]]*(ch_parja[key_here]['C13b_x=-L']*self.ch_pars[key_here]['Q'] + ch_parja[key_here]['C13c_x=-L'] * 
                              (-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0]) )) 
                              +  ch_parja[key_here]['C13d_x=-L']*(-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0])  + tid_otp[key_here][0][0] + tid_cor)
                
                elif ju_geg[i][1] == 'loc x=0':
                    temp = temp + (ch_parja[key_here]['C13a_x=0']*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]] + np.sum(ch_ans[key_here][ju_geg[i][2][1:]]*(ch_parja[key_here]['C13b_x=0']*self.ch_pars[key_here]['Q']+ch_parja[key_here]['C13c_x=0'] * 
                             (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1]) )) + ch_parja[key_here]['C13d_x=0'] * 
                             (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1]) + tid_otp[key_here][0][1] + tid_cor)
                else: print('ERROR')
            ch_sols[ju_geg[2][0]][ju_geg[2][2][0]] = temp #add to the solution vector
    
            #calculate for transport at depth, with tidal contribution   
            for k in range(1,self.M):
                temp = 0
                for i in range(3): #calculate contributions from channels seperately 
                    key_here = ju_geg[i][0]
                    bb = ans_all[ch_inds[key_here]['B_intot'][ju_geg[i][1]][0] : ch_inds[key_here]['B_intot'][ju_geg[i][1]][1]].copy()
                    tid_cor = self.ch_tide['sol_b_cor'](key_here, bb, ju_geg[i][1], 'dv')
                    
                    if ju_geg[i][1] == 'loc x=-L':
                        temp = temp - (ch_parja[key_here]['C12a_x=-L'] * (-3*ch_ans[key_here][ju_geg[i][2][k]]+4*ch_ans[key_here][ju_geg[i][3][k]]-ch_ans[key_here][ju_geg[i][4][k]])/(2*self.ch_pars[key_here]['dl'][0])
                                       + ch_parja[key_here]['C12b_x=-L'][k-1] * ch_ans[key_here][ju_geg[i][2][k]] * (-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0])  
                                       + np.sum([ch_parja[key_here]['C12c_x=-L'][k-1,n]*ch_ans[key_here][ju_geg[i][2]][n+1] for n in range(self.N)])*(-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0]) 
                                       + ch_parja[key_here]['C12d_x=-L'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][k]]
                                       + np.sum([ch_parja[key_here]['C12e_x=-L'][k-1,n]*ch_ans[key_here][ju_geg[i][2][1+n]] for n in range(self.N)])*self.ch_pars[key_here]['Q']
                                       + ch_parja[key_here]['C12f_x=-L'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]]
                                       + ch_parja[key_here]['C12g_x=-L'][k-1]*ch_ans[key_here][ju_geg[i][2][0]] * (-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0])
                                       + tid_otp[key_here][1][k-1,0] + tid_cor[k-1]
                                       )

                    elif ju_geg[i][1] == 'loc x=0' :
                        temp = temp + (ch_parja[key_here]['C12a_x=0'] * (ch_ans[key_here][ju_geg[i][4][k]]-4*ch_ans[key_here][ju_geg[i][3][k]]+3*ch_ans[key_here][ju_geg[i][2][k]])/(2*self.ch_pars[key_here]['dl'][-1])  
                                       + ch_parja[key_here]['C12b_x=0'][k-1] * ch_ans[key_here][ju_geg[i][2][k]] * (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1])  
                                       + np.sum([ch_parja[key_here]['C12c_x=0'][k-1,n]*ch_ans[key_here][ju_geg[i][2]][n+1] for n in range(self.N)]) * (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1]) 
                                       + ch_parja[key_here]['C12d_x=0'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][k]]
                                       + np.sum([ch_parja[key_here]['C12e_x=0'][k-1,n]*ch_ans[key_here][ju_geg[i][2]][n+1] for n in range(self.N)])*self.ch_pars[key_here]['Q']
                                       + ch_parja[key_here]['C12f_x=0'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]]
                                       + ch_parja[key_here]['C12g_x=0'][k-1]*ch_ans[key_here][ju_geg[i][2][0]]*(ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1])
                                       + tid_otp[key_here][1][k-1,1] + tid_cor[k-1]
                                       )
        
                    else: print("ERROR")
                    
                ch_sols[ju_geg[2][0]][ju_geg[2][2][k]] = temp #add to the solution vector

        # =============================================================================
        # complete to one big solution vector
        # =============================================================================
        sol_all = np.zeros(ans_all.shape)
        for key in self.ch_keys: sol_all[ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1]] = ch_sols[key]
                        
        # =============================================================================
        # Add the conditions regarding continuity of salt in the tidal cylce
        # thus the boundary layer correction
        # =============================================================================
        for j in range(self.n_j):
            #find the connections
            ju_geg = []
            for key in self.ch_keys: 
                if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1):  ju_geg.append([key,'loc x=-L',ch_inds[key]['bnd1_x=-L'],ch_inds[key]['bnd2_x=-L'],ch_inds[key]['bnd3_x=-L'],ch_inds[key]['bnd4_x=-L']])
                elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): ju_geg.append([key,'loc x=0',ch_inds[key]['bnd1_x=0'],ch_inds[key]['bnd2_x=0'],ch_inds[key]['bnd3_x=0'],ch_inds[key]['bnd4_x=0']])

            #prepare the relevant information for the associated points
            ju_geg_fortid = []
            for k in range(3):
                ke = ju_geg[k][0]
                temp = [ke,ju_geg[k][1]] #add channel name, begin/end
                temp.append(ch_ans[ke][ju_geg[k][2]][1:]*self.soc_sca) #add sn
                
                #calculate derivatives of the subtidal salinity
                if ju_geg[k][1] == 'loc x=-L':
                    temp2 = (-3*ch_ans[ke][ju_geg[k][2]] + 4*ch_ans[ke][ju_geg[k][3]] - ch_ans[ke][ju_geg[k][4]]) / (2*self.ch_pars[ke]['dl'][0])  * (self.soc_sca/self.Lsc)
                    temp3 = ( 2*ch_ans[ke][ju_geg[k][2]] - 5*ch_ans[ke][ju_geg[k][3]] + 4*ch_ans[ke][ju_geg[k][4]] - ch_ans[ke][ju_geg[k][5]]) / (self.ch_pars[ke]['dl'][0]**2) * (self.soc_sca/self.Lsc**2)
                elif ju_geg[k][1] == 'loc x=0':
                    temp2 = ( 3*ch_ans[ke][ju_geg[k][2]] - 4*ch_ans[ke][ju_geg[k][3]] + ch_ans[ke][ju_geg[k][4]]) / (2*self.ch_pars[ke]['dl'][-1])  * (self.soc_sca/self.Lsc)
                    temp3 = ( 2*ch_ans[ke][ju_geg[k][2]] - 5*ch_ans[ke][ju_geg[k][3]] + 4*ch_ans[ke][ju_geg[k][4]] - ch_ans[ke][ju_geg[k][5]]) / (self.ch_pars[ke]['dl'][-1]**2) * (self.soc_sca/self.Lsc**2)
    
                temp.append(temp2[1:]) #add dsndx
                temp.append(temp2[0])  #add dsbdx
                temp.append(temp3[0])  #add dsb2dx2
                
                bbn = ans_all[ch_inds[ke]['B_intot'][ju_geg[k][1]][0] : ch_inds[ke]['B_intot'][ju_geg[k][1]][1]]
                temp.append(bbn) #append bbb
                
                ju_geg_fortid.append(temp)
                            
            #calculate the boundary conditions
            sol_pt1,sol_pt2,sol_pt3,sol_pt4,sol_pt5,sol_pt6 = self.ch_tide['sol_bound'](ju_geg_fortid)
            
            #now add these parts to the solution vector 
            sol_all[ch_inds[ju_geg_fortid[0][0]]['B_intot'][ju_geg_fortid[0][1]][0] : ch_inds[ju_geg_fortid[0][0]]['B_intot'][ju_geg_fortid[0][1]][1]] = np.concatenate([sol_pt1,sol_pt2])
            sol_all[ch_inds[ju_geg_fortid[1][0]]['B_intot'][ju_geg_fortid[1][1]][0] : ch_inds[ju_geg_fortid[1][0]]['B_intot'][ju_geg_fortid[1][1]][1]] = np.concatenate([sol_pt3,sol_pt4])
            sol_all[ch_inds[ju_geg_fortid[2][0]]['B_intot'][ju_geg_fortid[2][1]][0] : ch_inds[ju_geg_fortid[2][0]]['B_intot'][ju_geg_fortid[2][1]][1]] = np.concatenate([sol_pt5,sol_pt6])

        return sol_all
    
    
    
    def jac_inner(ans, bbs, key):
        # =============================================================================
        # function to build the Jacobian for the internal part
        # =============================================================================
        
        #create empty matrix
        jac = np.zeros((self.ch_pars[key]['di'][-1]*self.M,self.ch_pars[key]['di'][-1]*self.M))

        #local variables, for short notation
        inds = ch_inds[key].copy()
        dl = self.ch_pars[key]['dl'].copy()
        pars = ch_parja[key].copy()

        # =============================================================================
        # Subtidal part
        # =============================================================================
        #vertical
        
        #term 1
        jac[inds['xr_mj'],inds['xrm_mj']] = jac[inds['xr_mj'],inds['xrm_mj']] - pars['C1a'][inds['xr']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) 
        jac[inds['xr_mj'],inds['xrp_mj']] = jac[inds['xr_mj'],inds['xrp_mj']] + pars['C1a'][inds['xr']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) 
        
        #term 2
        jac[inds['xr_mj'],inds['xrm_mj']] = (jac[inds['xr_mj'],inds['xrm_mj']] - pars['C2a'][inds['xr'],inds['jl']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) 
                                                               - pars['C2b'][inds['jl']]/(2*dl[inds['xr']]) * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']]) )
        jac[inds['xr_mj'],inds['xrp_mj']] = (jac[inds['xr_mj'],inds['xrp_mj']] + pars['C2a'][inds['xr'],inds['jl']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) 
                                                               + pars['C2b'][inds['jl']]/(2*dl[inds['xr']]) * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']]) )
        jac[inds['xr_mj'],inds['xrm_m']] = ( jac[inds['xr_mj'],inds['xrm_m']] - pars['C2b'][inds['jl']]/(2*dl[inds['xr']]) * (ans[inds['xrp_mj']]-ans[inds['xrm_mj']])/(2*dl[inds['xr']])
                              - np.sum([pars['C2d'][inds['jl'],n-1] * (ans[inds['xrp_m']+n]-ans[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(4*dl[inds['xr']]**2))
        jac[inds['xr_mj'],inds['xrp_m']] = (jac[inds['xr_mj'],inds['xrp_m']]  + pars['C2b'][inds['jl']]/(2*dl[inds['xr']]) * (ans[inds['xrp_mj']]-ans[inds['xrm_mj']])/(2*dl[inds['xr']]) 
                              + np.sum([pars['C2d'][inds['jl'],n-1] * (ans[inds['xrp_m']+n]-ans[inds['xrm_m']+n]) for n in range(1,self.M)],0)/(4*dl[inds['xr']]**2))
        jac[inds['xr_mjr'],inds['xrm_mk']] = jac[inds['xr_mjr'],inds['xrm_mk']] - pars['C2c'][inds['xrr'],inds['jlr'],inds['kl']]*self.ch_pars[key]['Q']/(2*dl[inds['xrr']]) - pars['C2d'][inds['jlr'],inds['kl']]*(ans[inds['xrp_mr']]-ans[inds['xrm_mr']])/(4*dl[inds['xrr']]**2)
        jac[inds['xr_mjr'],inds['xrp_mk']] = jac[inds['xr_mjr'],inds['xrp_mk']] + pars['C2c'][inds['xrr'],inds['jlr'],inds['kl']]*self.ch_pars[key]['Q']/(2*dl[inds['xrr']]) + pars['C2d'][inds['jlr'],inds['kl']]*(ans[inds['xrp_mr']]-ans[inds['xrm_mr']])/(4*dl[inds['xrr']]**2)
        
        #term 3
        jac[inds['xr_mj'],inds['xrm_m']] = jac[inds['xr_mj'],inds['xrm_m']] - pars['C3a'][inds['xr'],inds['jl']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) - pars['C3b'][inds['jl']]/dl[inds['xr']] * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']])
        jac[inds['xr_mj'],inds['xrp_m']] = jac[inds['xr_mj'],inds['xrp_m']] + pars['C3a'][inds['xr'],inds['jl']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) + pars['C3b'][inds['jl']]/dl[inds['xr']] * (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']])

        #term 4 does not exist
        
        #term 5
        jac[inds['xr_mj'], inds['xrm_m']] = (jac[inds['xr_mj'], inds['xrm_m']] + pars['C5a'][inds['jl']]*ans[inds['xr_mj']]/(dl[inds['xr']]**2) - pars['C5b'][inds['xr'],inds['jl']]/(2*dl[inds['xr']])*ans[inds['xr_mj']]
                                       + np.sum([ans[inds['xr_m']+n]*pars['C5c'][inds['jl'],n-1]/(dl[inds['xr']]**2) for n in range(1,self.M)],0) 
                                       - np.sum([ans[inds['xr_m']+n]*pars['C5d'][inds['xr'],inds['jl'],n-1]/(2*dl[inds['xr']]) for n in range(1,self.M)],0) )
        jac[inds['xr_mj'], inds['xr_m']] = jac[inds['xr_mj'], inds['xr_m']] - 2*pars['C5a'][inds['jl']]*ans[inds['xr_mj']]/(dl[inds['xr']]**2) -2* np.sum([ans[inds['xr_m']+n]*pars['C5c'][inds['jl'],n-1]/(dl[inds['xr']]**2) for n in range(1,self.M)],0)
        jac[inds['xr_mj'], inds['xrp_m']] = (jac[inds['xr_mj'], inds['xrp_m']] + pars['C5a'][inds['jl']]*ans[inds['xr_mj']]/(dl[inds['xr']]**2) + pars['C5b'][inds['xr'],inds['jl']]/(2*dl[inds['xr']])*ans[inds['xr_mj']]
                                       + np.sum([ans[inds['xr_m']+n]*pars['C5c'][inds['jl'],n-1]/(dl[inds['xr']]**2) for n in range(1,self.M)],0)
                                       + np.sum([ans[inds['xr_m']+n]*pars['C5d'][inds['xr'],inds['jl'],n-1]/(2*dl[inds['xr']]) for n in range(1,self.M)],0) )
        jac[inds['xr_mj'],inds['xr_mj']] = (jac[inds['xr_mj'], inds['xr_mj']] + pars['C5a'][inds['jl']]*(ans[inds['xrp_m']]-2*ans[inds['xr_m']]+ans[inds['xrm_m']])/(dl[inds['xr']]**2) 
                                         + pars['C5b'][inds['xr'],inds['jl']]*(ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']]) + pars['C5e'][inds['xr'],inds['jl']] )
        jac[inds['xr_mjr'], inds['xr_mk']] = (jac[inds['xr_mjr'], inds['xr_mk']] + pars['C5c'][inds['jlr'],inds['kl']]*(ans[inds['xrp_mr']]-2*ans[inds['xr_mr']]+ans[inds['xrm_mr']])/(dl[inds['xrr']]**2)
                                             + pars['C5d'][inds['xrr'],inds['jlr'],inds['kl']]*(ans[inds['xrp_mr']]-ans[inds['xrm_mr']])/(2*dl[inds['xrr']]) +pars['C5f'][inds['xrr'],inds['jlr'],inds['kl']] )
        
        #term 6
        jac[inds['xr_mj'],inds['xr_mj']] = jac[inds['xr_mj'],inds['xr_mj']] + pars['C6a'][inds['jl']]

        #term 7 
        jac[inds['xr_mj'],inds['xrm_mj']] = jac[inds['xr_mj'],inds['xrm_mj']] - pars['C7a'][inds['xr']]/(2*dl[inds['xr']]) + pars['C7b'][inds['xr']] / (dl[inds['xr']]**2)
        jac[inds['xr_mj'],inds['xr_mj']] = jac[inds['xr_mj'],inds['xr_mj']] - 2*pars['C7b'][inds['xr']]/(dl[inds['xr']]**2)
        jac[inds['xr_mj'],inds['xrp_mj']] = jac[inds['xr_mj'],inds['xrp_mj']] + pars['C7a'][inds['xr']]/(2*dl[inds['xr']]) + pars['C7b'][inds['xr']] / (dl[inds['xr']]**2)

        #depth-averaged salt
        jac[inds['x_m'], inds['xm_m']] = ( jac[inds['x_m'], inds['xm_m']] - pars['C10a'][inds['x']]/(2*dl[inds['x']]) + pars['C10b'][inds['x']]/(dl[inds['x']]**2) - 1/(2*dl[inds['x']])*np.sum([pars['C10c'][inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                       + 1/(dl[inds['x']]**2)*np.sum([pars['C10d'][n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                       - 1/(2*dl[inds['x']])*np.sum([pars['C10f'][n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*dl[inds['x']]) for n in range(1,self.M)],0) - pars['C10h'][inds['x']]*self.ch_pars[key]['Q']/(2*dl[inds['x']]) )
        jac[inds['xr_m'], inds['xrm_mj']] = jac[inds['xr_m'], inds['xrm_mj']] - pars['C10e'][inds['xr'],inds['jl']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) - pars['C10f'][inds['jl']]/(2*dl[inds['xr']])*(ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']])
        jac[inds['x_m'],inds['x_m']] = jac[inds['x_m'],inds['x_m']] - 2/(dl[inds['x']]**2)*pars['C10b'][inds['x']]  -2/(dl[inds['x']]**2)*np.sum([pars['C10d'][n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
        jac[inds['xr_m'], inds['xr_mj']] = (jac[inds['xr_m'], inds['xr_mj']] + (ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']]) * pars['C10c'][inds['xr'],inds['jl']] 
                                            + (ans[inds['xrp_m']]-2*ans[inds['xr_m']]+ans[inds['xrm_m']])/(dl[inds['xr']]**2) * pars['C10d'][inds['jl']] + pars['C10g'][inds['xr'],inds['jl']] )
        jac[inds['x_m'], inds['xp_m']] = (jac[inds['x_m'], inds['xp_m']] + pars['C10a'][inds['x']]/(2*dl[inds['x']]) + pars['C10b'][inds['x']]/(dl[inds['x']]**2) + 1/(2*dl[inds['x']])*np.sum([pars['C10c'][inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                       + 1/(dl[inds['x']]**2)*np.sum([pars['C10d'][n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                                       + 1/(2*dl[inds['x']])*np.sum([pars['C10f'][n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*dl[inds['x']]) for n in range(1,self.M)],0) + pars['C10h'][inds['x']]*self.ch_pars[key]['Q']/(2*dl[inds['x']]) )
        jac[inds['xr_m'], inds['xrp_mj']] = jac[inds['xr_m'], inds['xrp_mj']] + pars['C10e'][inds['xr'],inds['jl']]*self.ch_pars[key]['Q']/(2*dl[inds['xr']]) + pars['C10f'][inds['jl']]/(2*dl[inds['xr']])*(ans[inds['xrp_m']]-ans[inds['xrm_m']])/(2*dl[inds['xr']])

        #boundaries of segments
        #points equal
        jac[inds['in_bnd'],inds['in_bnd']-self.M] = 1
        jac[inds['in_bnd'],inds['in_bnd']] = -1
        #derivatives equal
        for d in range(1,len(self.ch_pars[key]['nxn'])):
            jac[(self.ch_pars[key]['di'][d]-1)*self.M+np.arange(self.M), (self.ch_pars[key]['di'][d]-3)*self.M+np.arange(self.M)] =  1/(2*dl[self.ch_pars[key]['di'][d]-1])
            jac[(self.ch_pars[key]['di'][d]-1)*self.M+np.arange(self.M), (self.ch_pars[key]['di'][d]-2)*self.M+np.arange(self.M)] = - 4/(2*dl[self.ch_pars[key]['di'][d]-1])
            jac[(self.ch_pars[key]['di'][d]-1)*self.M+np.arange(self.M), (self.ch_pars[key]['di'][d]-1)*self.M+np.arange(self.M)] =  3/(2*dl[self.ch_pars[key]['di'][d]-1]) 
            
            jac[(self.ch_pars[key]['di'][d]-1)*self.M+np.arange(self.M), self.ch_pars[key]['di'][d]*self.M+np.arange(self.M)] =  3/(2*dl[self.ch_pars[key]['di'][d]]) 
            jac[(self.ch_pars[key]['di'][d]-1)*self.M+np.arange(self.M), (self.ch_pars[key]['di'][d]+1)*self.M+np.arange(self.M)] =  - 4/(2*dl[self.ch_pars[key]['di'][d]])
            jac[(self.ch_pars[key]['di'][d]-1)*self.M+np.arange(self.M), (self.ch_pars[key]['di'][d]+2)*self.M+np.arange(self.M)] =  1/(2*dl[self.ch_pars[key]['di'][d]])
        
        # =============================================================================
        # Tidal part
        # =============================================================================

        #tides in inner domain
        tid_inp = conv_ans(ans, key)
        tid_otp = self.ch_tide['jac_i'](key,*tid_inp)
        
        jac[inds['x_m'], inds['xm_m']] = jac[inds['x_m'], inds['xm_m']] + tid_otp[0][inds['x']]
        jac[inds['x_m'], inds['x_m']] = jac[inds['x_m'], inds['x_m']] + tid_otp[1][inds['x']]
        jac[inds['x_m'], inds['xp_m']] = jac[inds['x_m'], inds['xp_m']] + tid_otp[2][inds['x']]

        jac[inds['xr_m'], inds['xrm_mj']] = jac[inds['xr_m'], inds['xrm_mj']] + tid_otp[3][inds['xr'],inds['jl']]
        jac[inds['xr_m'], inds['xr_mj']] = jac[inds['xr_m'], inds['xr_mj']] + tid_otp[4][inds['xr'],inds['jl']]
        jac[inds['xr_m'], inds['xrp_mj']] = jac[inds['xr_m'], inds['xrp_mj']] + tid_otp[5][inds['xr'],inds['jl']]

        #tides in sea part
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : 
            tid_otp_sea = self.ch_tide['jac_s'](key,*tid_inp)
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : 
            tid_otp_sea = self.ch_tide['jac_s'](key,*tid_inp)
            d_2,d_1 = self.ch_pars[key]['di'][-2]  ,  self.ch_pars[key]['di'][-1]
            
            jac[ch_inds[key]['x_m'], ch_inds[key]['xm_m']] = jac[ch_inds[key]['x_m'], ch_inds[key]['xm_m']] + tid_otp_sea[4][inds['x']]
            jac[ch_inds[key]['x_m'], ch_inds[key]['x_m'] ] = jac[ch_inds[key]['x_m'], ch_inds[key]['x_m'] ] + tid_otp_sea[5][inds['x']]
            jac[ch_inds[key]['x_m'], ch_inds[key]['xp_m']] = jac[ch_inds[key]['x_m'], ch_inds[key]['xp_m']] + tid_otp_sea[6][inds['x']]
            
            jac[ch_inds[key]['xs_m'], (d_2-1)*self.M+1:(d_2)*self.M] = jac[ch_inds[key]['xs_m'], (d_2-1)*self.M+1:(d_2)*self.M] + tid_otp_sea[0][1:-1]
            jac[ch_inds[key]['xs_m'], (d_2-3)*self.M] = jac[ch_inds[key]['xs_m'], (d_2-3)*self.M] + tid_otp_sea[1][1:-1]
            jac[ch_inds[key]['xs_m'], (d_2-2)*self.M] = jac[ch_inds[key]['xs_m'], (d_2-2)*self.M] + tid_otp_sea[2][1:-1]
            jac[ch_inds[key]['xs_m'], (d_2-1)*self.M] = jac[ch_inds[key]['xs_m'], (d_2-1)*self.M] + tid_otp_sea[3][1:-1]        
            
        
        NRv = self.ch_tide['jacz_i'](key,*tid_inp)
        t1a,t1b,t1c,t1d,t1e,t1f = NRv[0]
        t2a,t2b,t2c,t2d,t2e,t2f = NRv[1]
        t3a,t3c,t3e = NRv[2]
        
        jac[inds['xr_mj'],inds['xrm_m']] = jac[inds['xr_mj'],inds['xrm_m']] + (t1a[inds['jl'],inds['xr']]+t2a[inds['jl'],inds['xr']]+t3a[inds['jl'],inds['xr']])*self.soc_sca
        jac[inds['xr_mj'],inds['xr_m']]  = jac[inds['xr_mj'], inds['xr_m']] + (t1b[inds['jl'],inds['xr']]+t2b[inds['jl'],inds['xr']])*self.soc_sca
        jac[inds['xr_mj'],inds['xrp_m']] = jac[inds['xr_mj'],inds['xrp_m']] + (t1c[inds['jl'],inds['xr']]+t2c[inds['jl'],inds['xr']]+t3c[inds['jl'],inds['xr']])*self.soc_sca 
        
        jac[inds['xr_mjr'],inds['xrm_mk']] = jac[inds['xr_mjr'],inds['xrm_mk']] + (t1d[inds['kl'],inds['jlr'],inds['xrr']]+t2d[inds['kl'],inds['jlr'],inds['xrr']])*self.soc_sca
        jac[inds['xr_mjr'], inds['xr_mk']] = jac[inds['xr_mjr'], inds['xr_mk']] + (t1e[inds['kl'],inds['jlr'],inds['xrr']]+t2e[inds['kl'],inds['jlr'],inds['xrr']]+t3e[inds['kl'],inds['jlr'],inds['xrr']])*self.soc_sca
        jac[inds['xr_mjr'],inds['xrp_mk']] = jac[inds['xr_mjr'],inds['xrp_mk']] + (t1f[inds['kl'],inds['jlr'],inds['xrr']]+t2f[inds['kl'],inds['jlr'],inds['xrr']])*self.soc_sca

        return jac
    
    
    def jac_complete(ans_all):        
        # =============================================================================
        # Function to calculate the Jacobian for the full system of equations
        # =============================================================================
        #prepare
        
        #starting indices of the channels int he big matrix
        ind = [0]
        for key in self.ch_keys:ind.append(ind[-1]+self.ch_pars[key]['di'][-1]*self.M)

        #first: break down ans_all to ch_ans
        ch_ans = {}
        for key in self.ch_keys: ch_ans[key] = ans_all[ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1]]

        #build the internal parts of the domains
        ch_jacs = {}
        for key in self.ch_keys:
            #select boundary layer correction
            bbs = [None, None]
            if self.ch_gegs[key]['loc x=-L'][0] =='j': bbs[0] = ans_all[ch_inds[key]['B_intot']['loc x=-L'][0] : ch_inds[key]['B_intot']['loc x=-L'][1]]
            if self.ch_gegs[key]['loc x=0'][0] =='j':  bbs[1] = ans_all[ch_inds[key]['B_intot']['loc x=0'][0] : ch_inds[key]['B_intot']['loc x=0'][1]]
            #build internal parts
            ch_jacs[key] = jac_inner(ch_ans[key], bbs, key)

        #prepare the tides
        tid_otp = {}
        for key in self.ch_keys: 
            tid_inp = conv_ans(ch_ans[key], key)
            tid_otp[key] = self.ch_tide['jac_b'](key,*tid_inp)
            
        # =============================================================================
        # add boundary conditions to Jacobian
        # =============================================================================
        for key in self.ch_keys:
            #river, sea boundaries: the same
            if self.ch_gegs[key]['loc x=-L'][0] == 'r' or self.ch_gegs[key]['loc x=-L'][0] == 's': ch_jacs[key][np.arange(self.M),np.arange(self.M)] = 1
            elif self.ch_gegs[key]['loc x=0'][0] == 'r' or self.ch_gegs[key]['loc x=0'][0] == 's': ch_jacs[key][np.arange(-self.M,0),np.arange(-self.M,0)] = 1
                
            # weir boundaries: another cook, zou Meinte zeggen. 
            if self.ch_gegs[key]['loc x=-L'][0] == 'w':
                if self.ch_pars[key]['Q']>0: #flux is equal to the advective flux through weir
                    ch_jacs[key][0,0] =  self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*self.Lsc +3*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) -3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) \
                        * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + self.ch_pars[key]['bH_1'][0]*tid_otp[key][0][0]
                    ch_jacs[key][0,self.M] =  -4*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) + 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][1]   /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0])
                    ch_jacs[key][0,2*self.M] = self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) - self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][2]   /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0])
                    for j in range(1,self.M):
                        ch_jacs[key][0,j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2)) + tid_otp[key][0][3][j-1]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0])
                                
                elif self.ch_pars[key]['Q']<=0: #flux is equal to the advective flux through weir, set by river salinity
                    ch_jacs[key][0,0] = 3*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) -3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) \
                        * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][0]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    ch_jacs[key][0,self.M] =  -4*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) + 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][1]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    ch_jacs[key][0,2*self.M] = self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) - self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][2]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    for j in range(1,self.M):
                        ch_jacs[key][0,j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  + tid_otp[key][0][3][j-1]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0])
 
                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][j, 2*self.M+j] = -1/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, self.M+j] = 4/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, j] = -3/(2*self.ch_pars[key]['dl'][0])  

            elif self.ch_gegs[key]['loc x=0'][0] == 'w':
                if self.ch_pars[key]['Q']>0: #flux is equal to the advective flux through weir
                    #no total flux
                    ch_jacs[key][-self.M,-self.M] = - 3*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + 3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) \
                        * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][4]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    ch_jacs[key][-self.M,-2*self.M] =  4*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) - 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][5]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    ch_jacs[key][-self.M,-3*self.M] = - self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][6]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    for j in range(1,self.M):
                        ch_jacs[key][-self.M,-self.M+j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  + tid_otp[key][0][7][j-1]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
 
                elif self.ch_pars[key]['Q']<=0: #flux is equal to the advective flux through weir, set by river salinity
                    #no total flux
                    ch_jacs[key][-self.M,-self.M] =  self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*self.Lsc - 3*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + 3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) \
                        * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][4]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    ch_jacs[key][-self.M,-2*self.M] =  4*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) - 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][5]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    ch_jacs[key][-self.M,-3*self.M] = - self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][6]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                    for j in range(1,self.M):
                        ch_jacs[key][-self.M,-self.M+j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2)) + tid_otp[key][0][7][j-1]  /(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
 
                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][-self.M+j, -3*self.M+j] = 1/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -2*self.M+j] = -4/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -self.M+j] = 3/(2*self.ch_pars[key]['dl'][-1])      
                
        
            #har boundaries: 
            if self.ch_gegs[key]['loc x=-L'][0] == 'h':
                #no total flux
                ch_jacs[key][0,0] = 3*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) -3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) \
                    * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][0]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]) 
                ch_jacs[key][0,self.M] =  -4*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) + 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                      + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][1]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0])
                ch_jacs[key][0,2*self.M] = self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) - self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                       + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][2]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0]) 
                for j in range(1,self.M):
                    ch_jacs[key][0,j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                            * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))+ tid_otp[key][0][3][j-1]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0])

                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][j, 2*self.M+j] = -1/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, self.M+j] = 4/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, j] = -3/(2*self.ch_pars[key]['dl'][0])  

            elif self.ch_gegs[key]['loc x=0'][0] == 'h':
                #no total flux
                ch_jacs[key][-self.M,-self.M] =  - 3*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + 3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) \
                    * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][4]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                ch_jacs[key][-self.M,-2*self.M] =  4*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) - 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                      + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][5]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                ch_jacs[key][-self.M,-3*self.M] = - self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                       + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)]) + tid_otp[key][0][6]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])
                for j in range(1,self.M):
                    ch_jacs[key][-self.M,-self.M+j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                            * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  + tid_otp[key][0][7][j-1]/(self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1])

                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][-self.M+j, -3*self.M+j] = 1/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -2*self.M+j] = -4/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -self.M+j] = 3/(2*self.ch_pars[key]['dl'][-1])  
                    
        # =============================================================================
        # build large matrix
        # =============================================================================
        jac_all = np.zeros((ch_inds[self.ch_keys[-1]]['sB_intot'][1],ch_inds[self.ch_keys[-1]]['sB_intot'][1]))
        for key in self.ch_keys:
            ch_inds[key]['start'] = ch_inds[key]['s_intot'][0]
            jac_all[ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1],ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1]] = ch_jacs[key] #this should be correct
      
        # =============================================================================
        # boundary layer in inner domain - placed here because the derivatives are outside local jac
        # =============================================================================
        for key in self.ch_keys:        
            if self.ch_gegs[key]['loc x=-L'][0] =='j': 
                #derivatives 
                dT_dBR, dT_dBI  = self.ch_tide['jac_i_cor'](key,'loc x=-L','da')
                dTz_dBR,dTz_dBI = self.ch_tide['jac_i_cor'](key,'loc x=-L','dv')

                #indices in jacobian
                BR0 = ch_inds[key]['B_intot']['loc x=-L'][0]
                BI0 = ch_inds[key]['B_intot']['loc x=-L'][0] + self.M
                SB0 = ch_inds[key]['s_intot'][0]
                
                #TODO: do this with indices instead of the ugly for loops                
                for x3 in range(len(dT_dBR[0])):
                    for ba in range(self.M):
                        #depth-averaged
                        jac_all[SB0 + self.M*(x3+1) , BR0 + ba] = dT_dBR[ba,x3]
                        jac_all[SB0 + self.M*(x3+1) , BI0 + ba] = dT_dBI[ba,x3]
                        for k in range(1,self.M):     #depth levels
                            jac_all[SB0 + self.M*(x3+1) + k , BR0 + ba] = dTz_dBR[ba,k-1,x3]
                            jac_all[SB0 + self.M*(x3+1) + k , BI0 + ba] = dTz_dBI[ba,k-1,x3]

            if self.ch_gegs[key]['loc x=0' ][0] =='j':
                #derivatives
                dT_dBR, dT_dBI  = self.ch_tide['jac_i_cor'](key,'loc x=0','da')
                dTz_dBR,dTz_dBI = self.ch_tide['jac_i_cor'](key,'loc x=0','dv')

                #indices in jacobian
                BR0 = ch_inds[key]['B_intot']['loc x=0'][0]
                BI0 = ch_inds[key]['B_intot']['loc x=0'][0] + self.M
                SB0 = ch_inds[key]['s_intot'][1] - self.M*len(dT_dBR[0])
                                
                #TODO: do this with indices instead of the ugly for loops                
                for x3 in range(len(dT_dBR[0])):
                    for ba in range(self.M):
                        #depth-averaged
                        jac_all[SB0 + self.M*(x3-1) , BR0 + ba] = dT_dBR[ba,x3]
                        jac_all[SB0 + self.M*(x3-1) , BI0 + ba] = dT_dBI[ba,x3] 
                        for k in range(1,self.M):     #depth levels
                            jac_all[SB0 + self.M*(x3-1) + k, BR0 + ba] = dTz_dBR[ba,k-1,x3]
                            jac_all[SB0 + self.M*(x3-1) + k, BI0 + ba] = dTz_dBI[ba,k-1,x3]

        # =============================================================================
        # conditions at junctions 
        # =============================================================================
        for j in range(self.n_j):
            #find the connections
            ju_geg = []
            for key in self.ch_keys: 
                if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1):  ju_geg.append([key,'loc x=-L',ch_inds[key]['bnd1_x=-L'],ch_inds[key]['bnd2_x=-L'],ch_inds[key]['bnd3_x=-L'],ch_inds[key]['bnd4_x=-L']])
                elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): ju_geg.append([key,'loc x=0',ch_inds[key]['bnd1_x=0'],ch_inds[key]['bnd2_x=0'],ch_inds[key]['bnd3_x=0'],ch_inds[key]['bnd4_x=0']])
            
            #indices 
            ind_bnd = ([ju_geg[0][2]+ch_inds[ju_geg[0][0]]['start'] , ju_geg[0][3]+ch_inds[ju_geg[0][0]]['start'] , ju_geg[0][4]+ch_inds[ju_geg[0][0]]['start']],
                [ju_geg[1][2]+ch_inds[ju_geg[1][0]]['start'] , ju_geg[1][3]+ch_inds[ju_geg[1][0]]['start'] , ju_geg[1][4]+ch_inds[ju_geg[1][0]]['start']],
                [ju_geg[2][2]+ch_inds[ju_geg[2][0]]['start'] , ju_geg[2][3]+ch_inds[ju_geg[2][0]]['start'] , ju_geg[2][4]+ch_inds[ju_geg[2][0]]['start']] )
                        
            #first: 0-1 equal
            jac_all[ind_bnd[0][0],ind_bnd[0][0]] = 1
            jac_all[ind_bnd[0][0],ind_bnd[1][0]] = -1
    
            #second: 1-2 equal
            jac_all[ind_bnd[1][0],ind_bnd[1][0]] = 1
            jac_all[ind_bnd[1][0],ind_bnd[2][0]] = -1
            
            
            #third: transport    
            #depth-averaged transport
            for i in range(3): #calculate contributions from channels seperately 
                key_here = ju_geg[i][0]

                #tidal contribution with boundary layer correction - implementation is a little different from before
                tid_cor = self.ch_tide['jac_b_cor'](key_here, ju_geg[i][1], 'da')
                #associated indices
                tid_cor_i = np.arange(ch_inds[key_here]['B_intot'][ju_geg[i][1]][0] , ch_inds[key_here]['B_intot'][ju_geg[i][1]][1])
                                
                if ju_geg[i][1] == 'loc x=-L':
                    jac_all[ind_bnd[2][0][0],ind_bnd[i][0][0]] = (- ch_parja[key_here]['C13a_x=-L'] * self.ch_pars[key_here]['Q'] +  3/(2*self.ch_pars[key_here]['dl'][0]) *  np.sum(ch_parja[key_here]['C13c_x=-L']*ans_all[ind_bnd[i][0][1:]])  
                                                          + ch_parja[key_here]['C13d_x=-L'] * 3/(2*self.ch_pars[key_here]['dl'][0]) - tid_otp[key_here][0][0]   )
                    jac_all[ind_bnd[2][0][0],ind_bnd[i][1][0]] = (ch_parja[key_here]['C13d_x=-L'] * -4/(2*self.ch_pars[key_here]['dl'][0]) 
                                                          - 4/(2*self.ch_pars[key_here]['dl'][0]) *  np.sum(ch_parja[key_here]['C13c_x=-L']*ans_all[ind_bnd[i][0][1:]]) - tid_otp[key_here][0][1]   )
                    jac_all[ind_bnd[2][0][0],ind_bnd[i][2][0]] = (ch_parja[key_here]['C13d_x=-L'] * 1/(2*self.ch_pars[key_here]['dl'][0]) 
                                                          + 1/(2*self.ch_pars[key_here]['dl'][0]) *  np.sum(ch_parja[key_here]['C13c_x=-L']*ans_all[ind_bnd[i][0][1:]]) - tid_otp[key_here][0][2]   )
                    for k in range(1,self.M): jac_all[ind_bnd[2][0][0],ind_bnd[i][0][k]]  = jac_all[ind_bnd[2][0][0],ind_bnd[i][0][k]] - ch_parja[key_here]['C13b_x=-L'][k-1] * self.ch_pars[key_here]['Q'] \
                        + ch_parja[key_here]['C13c_x=-L'][k-1] * (ans_all[ind_bnd[i][2][0]] - 4*ans_all[ind_bnd[i][1][0]] + 3*ans_all[ind_bnd[i][0][0]] )/(2*self.ch_pars[key_here]['dl'][0]) - tid_otp[key_here][0][3][k-1]   
                    #boundary layer correction
                    jac_all[ind_bnd[2][0][0],tid_cor_i] = -tid_cor
                    
                    
                elif ju_geg[i][1] == 'loc x=0': 
                    jac_all[ind_bnd[2][0][0],ind_bnd[i][0][0]] = (ch_parja[key_here]['C13a_x=0'] * self.ch_pars[key_here]['Q'] + ch_parja[key_here]['C13d_x=0'] * 3/(2*self.ch_pars[key_here]['dl'][-1]) 
                                                          + 3/(2*self.ch_pars[key_here]['dl'][-1]) *  np.sum(ch_parja[key_here]['C13c_x=0']*ans_all[ind_bnd[i][0][1:]]) + tid_otp[key_here][0][4]   )
                    jac_all[ind_bnd[2][0][0],ind_bnd[i][1][0]] = (ch_parja[key_here]['C13d_x=0'] * -4/(2*self.ch_pars[key_here]['dl'][-1]) 
                                                          - 4/(2*self.ch_pars[key_here]['dl'][-1]) *  np.sum(ch_parja[key_here]['C13c_x=0']*ans_all[ind_bnd[i][0][1:]]) + tid_otp[key_here][0][5]   )
                    jac_all[ind_bnd[2][0][0],ind_bnd[i][2][0]] = (ch_parja[key_here]['C13d_x=0'] * 1/(2*self.ch_pars[key_here]['dl'][-1]) 
                                                          + 1/(2*self.ch_pars[key_here]['dl'][-1]) *  np.sum(ch_parja[key_here]['C13c_x=0']*ans_all[ind_bnd[i][0][1:]]) + tid_otp[key_here][0][6]   )
                    for k in range(1,self.M): jac_all[ind_bnd[2][0][0],ind_bnd[i][0][k]] = jac_all[ind_bnd[2][0][0],ind_bnd[i][0][k]] + ch_parja[key_here]['C13b_x=0'][k-1] * self.ch_pars[key_here]['Q'] \
                        + ch_parja[key_here]['C13c_x=0'][k-1] * (ans_all[ind_bnd[i][2][0]]-4*ans_all[ind_bnd[i][1][0]]+3*ans_all[ind_bnd[i][0][0]])/(2*self.ch_pars[key_here]['dl'][-1]) + tid_otp[key_here][0][7][k-1]   
                    #boundary layer correction
                    jac_all[ind_bnd[2][0][0],tid_cor_i] = tid_cor
                    
                else: print('ERROR')
    
            
            #transport at every vertical level , with tidal contriubtion
            for k in range(1,self.M):
                for i in range(3): #calculate contributions from channels seperately 
                    key_here = ju_geg[i][0]
                    
                    #tidal contribution with boundary layer correction - implementation is a little different from before
                    tid_cor = self.ch_tide['jac_b_cor'](key_here, ju_geg[i][1], 'dv')


                    #associated indices
                    tid_cor_i = np.arange(ch_inds[key_here]['B_intot'][ju_geg[i][1]][0] , ch_inds[key_here]['B_intot'][ju_geg[i][1]][1])                
                
                    if ju_geg[i][1] == 'loc x=-L':
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][0][0]] = ch_parja[key_here]['C12b_x=-L'][k-1] * ans_all[ind_bnd[i][0][k]] * 3/(2*self.ch_pars[key_here]['dl'][0]) \
                            + np.sum([ch_parja[key_here]['C12c_x=-L'][k-1,n-1] * ans_all[ind_bnd[i][0][n]] for n in range(1,self.M)]) * 3/(2*self.ch_pars[key_here]['dl'][0]) \
                            - ch_parja[key_here]['C12f_x=-L'][k-1]*self.ch_pars[key_here]['Q'] + ch_parja[key_here]['C12g_x=-L'][k-1] * ( ans_all[ind_bnd[i][2][0]]-4*ans_all[ind_bnd[i][1][0]]+3*ans_all[ind_bnd[i][0][0]] )/(2*self.ch_pars[key_here]['dl'][0]) \
                            + ch_parja[key_here]['C12g_x=-L'][k-1] * ans_all[ind_bnd[i][0][0]] * 3/(2*self.ch_pars[key_here]['dl'][0]) - tid_otp[key_here][1][0][k-1]
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][1][0]] = ch_parja[key_here]['C12b_x=-L'][k-1] * ans_all[ind_bnd[i][0][k]] *-4/(2*self.ch_pars[key_here]['dl'][0]) + np.sum([ch_parja[key_here]['C12c_x=-L'][k-1,n-1] * ans_all[ind_bnd[i][0][n]] for n in range(1,self.M)]) *-4/(2*self.ch_pars[key_here]['dl'][0]) \
                            + ch_parja[key_here]['C12g_x=-L'][k-1] * ans_all[ind_bnd[i][0][0]] *-4/(2*self.ch_pars[key_here]['dl'][0]) - tid_otp[key_here][1][1][k-1]
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][2][0]] = ch_parja[key_here]['C12b_x=-L'][k-1] * ans_all[ind_bnd[i][0][k]] * 1/(2*self.ch_pars[key_here]['dl'][0]) + np.sum([ch_parja[key_here]['C12c_x=-L'][k-1,n-1] * ans_all[ind_bnd[i][0][n]] for n in range(1,self.M)]) * 1/(2*self.ch_pars[key_here]['dl'][0]) \
                            + ch_parja[key_here]['C12g_x=-L'][k-1] * ans_all[ind_bnd[i][0][0]] * 1/(2*self.ch_pars[key_here]['dl'][0]) - tid_otp[key_here][1][2][k-1]
                        
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][0][k]] = ch_parja[key_here]['C12a_x=-L'] * 3/(2*self.ch_pars[key_here]['dl'][0]) + ch_parja[key_here]['C12b_x=-L'][k-1] * ( ans_all[ind_bnd[i][2][0]]-4*ans_all[ind_bnd[i][1][0]]+3*ans_all[ind_bnd[i][0][0]] )/(2*self.ch_pars[key_here]['dl'][0]) \
                            - ch_parja[key_here]['C12d_x=-L'][k-1]*self.ch_pars[key_here]['Q']
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][1][k]] = ch_parja[key_here]['C12a_x=-L'] *-4/(2*self.ch_pars[key_here]['dl'][0])
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][2][k]] = ch_parja[key_here]['C12a_x=-L'] * 1/(2*self.ch_pars[key_here]['dl'][0])
                        
                        for k2 in range(1,self.M): jac_all[ind_bnd[2][0][k],ind_bnd[i][0][k2]] = jac_all[ind_bnd[2][0][k],ind_bnd[i][0][k2]] + ch_parja[key_here]['C12c_x=-L'][k-1,k2-1] * ( ans_all[ind_bnd[i][2][0]]-4*ans_all[ind_bnd[i][1][0]]+3*ans_all[ind_bnd[i][0][0]] )/(2*self.ch_pars[key_here]['dl'][0]) \
                            - ch_parja[key_here]['C12e_x=-L'][k-1,k2-1]*self.ch_pars[key_here]['Q']  - tid_otp[key_here][1][3][k2-1,k-1]
                        
                        #boundary layer correction
                        jac_all[ind_bnd[2][0][k],tid_cor_i] = -tid_cor[k-1]
                        
                        
                    elif ju_geg[i][1] == 'loc x=0' :
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][0][0]] = ch_parja[key_here]['C12b_x=0'][k-1] * ans_all[ind_bnd[i][0][k]] * 3/(2*self.ch_pars[key_here]['dl'][-1]) \
                            + np.sum([ch_parja[key_here]['C12c_x=0'][k-1,n-1] * ans_all[ind_bnd[i][0][n]] for n in range(1,self.M)]) * 3/(2*self.ch_pars[key_here]['dl'][-1]) \
                            + ch_parja[key_here]['C12f_x=0'][k-1]*self.ch_pars[key_here]['Q'] + ch_parja[key_here]['C12g_x=0'][k-1] * ( ans_all[ind_bnd[i][2][0]]-4*ans_all[ind_bnd[i][1][0]]+3*ans_all[ind_bnd[i][0][0]] )/(2*self.ch_pars[key_here]['dl'][-1]) \
                            + ch_parja[key_here]['C12g_x=0'][k-1] * ans_all[ind_bnd[i][0][0]] * 3/(2*self.ch_pars[key_here]['dl'][-1]) + tid_otp[key_here][1][4][k-1]
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][1][0]] = ch_parja[key_here]['C12b_x=0'][k-1] * ans_all[ind_bnd[i][0][k]] *-4/(2*self.ch_pars[key_here]['dl'][-1]) + np.sum([ch_parja[key_here]['C12c_x=0'][k-1,n-1] * ans_all[ind_bnd[i][0][n]] for n in range(1,self.M)]) *-4/(2*self.ch_pars[key_here]['dl'][-1]) \
                            + ch_parja[key_here]['C12g_x=0'][k-1] * ans_all[ind_bnd[i][0][0]] *-4/(2*self.ch_pars[key_here]['dl'][-1]) + tid_otp[key_here][1][5][k-1]
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][2][0]] = ch_parja[key_here]['C12b_x=0'][k-1] * ans_all[ind_bnd[i][0][k]] * 1/(2*self.ch_pars[key_here]['dl'][-1]) + np.sum([ch_parja[key_here]['C12c_x=0'][k-1,n-1] * ans_all[ind_bnd[i][0][n]] for n in range(1,self.M)]) * 1/(2*self.ch_pars[key_here]['dl'][-1]) \
                            + ch_parja[key_here]['C12g_x=0'][k-1] * ans_all[ind_bnd[i][0][0]] * 1/(2*self.ch_pars[key_here]['dl'][-1]) + tid_otp[key_here][1][6][k-1]

                        jac_all[ind_bnd[2][0][k],ind_bnd[i][0][k]] = ch_parja[key_here]['C12a_x=0'] * 3/(2*self.ch_pars[key_here]['dl'][-1]) + ch_parja[key_here]['C12b_x=0'][k-1] * ( ans_all[ind_bnd[i][2][0]]-4*ans_all[ind_bnd[i][1][0]]+3*ans_all[ind_bnd[i][0][0]] )/(2*self.ch_pars[key_here]['dl'][-1]) \
                            + ch_parja[key_here]['C12d_x=0'][k-1]*self.ch_pars[key_here]['Q']
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][1][k]] = ch_parja[key_here]['C12a_x=0'] *-4/(2*self.ch_pars[key_here]['dl'][-1]) 
                        jac_all[ind_bnd[2][0][k],ind_bnd[i][2][k]] = ch_parja[key_here]['C12a_x=0'] * 1/(2*self.ch_pars[key_here]['dl'][-1]) 
                        
                        for k2 in range(1,self.M): jac_all[ind_bnd[2][0][k],ind_bnd[i][0][k2]] = jac_all[ind_bnd[2][0][k],ind_bnd[i][0][k2]] + ch_parja[key_here]['C12c_x=0'][k-1,k2-1] * ( ans_all[ind_bnd[i][2][0]]-4*ans_all[ind_bnd[i][1][0]]+3*ans_all[ind_bnd[i][0][0]] )/(2*self.ch_pars[key_here]['dl'][-1]) \
                            + ch_parja[key_here]['C12e_x=0'][k-1,k2-1]*self.ch_pars[key_here]['Q'] + tid_otp[key_here][1][7][k2-1,k-1]
                        
                        #boundary layer correction
                        jac_all[ind_bnd[2][0][k],tid_cor_i] = tid_cor[k-1]
                        
                    else: print("ERROR")
                
            # =============================================================================
            # add here the matching on the tidal timescale
            # =============================================================================
            #prepare the relevant information for the associated points
            ju_geg_fortid = []
            for k in range(3):
                ke = ju_geg[k][0] #key
                temp = [ke,ju_geg[k][1]] #add channel name, begin/end
                temp.append(ch_ans[ke][ju_geg[k][2]][1:]*self.soc_sca) #add sn
                
                #derivatives at boundaries
                if ju_geg[k][1] == 'loc x=-L':
                    temp2 = (-3*ch_ans[ke][ju_geg[k][2]] + 4*ch_ans[ke][ju_geg[k][3]] - ch_ans[ke][ju_geg[k][4]]) / (2*self.ch_pars[ke]['dl'][0])  * (self.soc_sca/self.Lsc)
                    temp3 = ( 2*ch_ans[ke][ju_geg[k][2]] - 5*ch_ans[ke][ju_geg[k][3]] + 4*ch_ans[ke][ju_geg[k][4]] - ch_ans[ke][ju_geg[k][5]]) / (self.ch_pars[ke]['dl'][0]**2) * (self.soc_sca/self.Lsc**2)
                elif ju_geg[k][1] == 'loc x=0':
                    temp2 = ( 3*ch_ans[ke][ju_geg[k][2]] - 4*ch_ans[ke][ju_geg[k][3]] + ch_ans[ke][ju_geg[k][4]]) / (2*self.ch_pars[ke]['dl'][-1])  * (self.soc_sca/self.Lsc)
                    temp3 = ( 2*ch_ans[ke][ju_geg[k][2]] - 5*ch_ans[ke][ju_geg[k][3]] + 4*ch_ans[ke][ju_geg[k][4]] - ch_ans[ke][ju_geg[k][5]]) / (self.ch_pars[ke]['dl'][-1]**2) * (self.soc_sca/self.Lsc**2)
          
                temp.append(temp2[1:]) #add dsndx
                temp.append(temp2[0])  #add dsbdx
                temp.append(temp3[0])  #add dsb2dx2
            
                bbn = ans_all[ch_inds[ke]['B_intot'][ju_geg[k][1]][0] : ch_inds[ke]['B_intot'][ju_geg[k][1]][1]]
                temp.append(bbn) #append bbb
                    
                ju_geg_fortid.append(temp)
                
            
            #calculate the contribution to the jacobian with the tidal equations        
            derv_C1 = self.ch_tide['jac_bound'](ju_geg_fortid[0])
            derv_C2 = self.ch_tide['jac_bound'](ju_geg_fortid[1])
            derv_C3 = self.ch_tide['jac_bound'](ju_geg_fortid[2])
            
            sigma = np.linspace(-1,0,self.nz)
            iis = np.arange(self.M)

            z_inds = np.linspace(0,self.nz-1,self.M,dtype=int) # at which vertical levels we evaluate the expression
            #short for indices
            iC1a = ch_inds[ju_geg_fortid[0][0]]['B_intot'][ju_geg_fortid[0][1]]
            iC2a = ch_inds[ju_geg_fortid[1][0]]['B_intot'][ju_geg_fortid[1][1]]
            iC3a = ch_inds[ju_geg_fortid[2][0]]['B_intot'][ju_geg_fortid[2][1]]
            iC1b = ch_inds[ju_geg_fortid[0][0]]['s_intot']
            iC2b = ch_inds[ju_geg_fortid[1][0]]['s_intot']
            iC3b = ch_inds[ju_geg_fortid[2][0]]['s_intot']
                       
            # =============================================================================
            #  first condition: 1-2=0
            # =============================================================================
            for m in range(self.M): #TODO: for now a loop, later replace with indices allocation. but that is less easy to check. 
                #real part
                #afgeleides naar Bs van hetzelfde kanaal
                jac_all[iC1a[0]+m,iC1a[0]+iis] = np.cos(iis*np.pi*sigma[z_inds[m]])
                #afgeleiders naar Bs van het andere kanaal
                jac_all[iC1a[0]+m,iC2a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]])
                #afgeleides naar st van hetzelfde kanaal
                if ju_geg_fortid[0][1] == 'loc x=-L':
                    jac_all[iC1a[0]+m,iC1b[0]+0*self.M] = derv_C1[0][0][z_inds[m]]
                    jac_all[iC1a[0]+m,iC1b[0]+1*self.M] = derv_C1[0][1][z_inds[m]]
                    jac_all[iC1a[0]+m,iC1b[0]+2*self.M] = derv_C1[0][2][z_inds[m]]
                    jac_all[iC1a[0]+m,iC1b[0]+0*self.M+1+iis[:-1]] = derv_C1[0][3][:,z_inds[m]]
                elif ju_geg_fortid[0][1] == 'loc x=0':
                    jac_all[iC1a[0]+m,iC1b[1]-1*self.M] = derv_C1[0][0][z_inds[m]]
                    jac_all[iC1a[0]+m,iC1b[1]-2*self.M] = derv_C1[0][1][z_inds[m]]
                    jac_all[iC1a[0]+m,iC1b[1]-3*self.M] = derv_C1[0][2][z_inds[m]]
                    jac_all[iC1a[0]+m,iC1b[1]-1*self.M+1+iis[:-1]] = derv_C1[0][3][:,z_inds[m]]
                #afgeleides naar st van het andere kanaal
                if ju_geg_fortid[1][1] == 'loc x=-L':
                    jac_all[iC1a[0]+m,iC2b[0]+0*self.M] = -derv_C2[0][0][z_inds[m]]
                    jac_all[iC1a[0]+m,iC2b[0]+1*self.M] = -derv_C2[0][1][z_inds[m]]
                    jac_all[iC1a[0]+m,iC2b[0]+2*self.M] = -derv_C2[0][2][z_inds[m]]
                    jac_all[iC1a[0]+m,iC2b[0]+0*self.M+1+iis[:-1]] = -derv_C2[0][3][:,z_inds[m]]
                elif ju_geg_fortid[1][1] == 'loc x=0':
                    jac_all[iC1a[0]+m,iC2b[1]-1*self.M] = -derv_C2[0][0][z_inds[m]]
                    jac_all[iC1a[0]+m,iC2b[1]-2*self.M] = -derv_C2[0][1][z_inds[m]]
                    jac_all[iC1a[0]+m,iC2b[1]-3*self.M] = -derv_C2[0][2][z_inds[m]]
                    jac_all[iC1a[0]+m,iC2b[1]-1*self.M+1+iis[:-1]] = -derv_C2[0][3][:,z_inds[m]]
                #imaginairy part 
                #afgeleides naar Bs van hetzelfde kanaal
                jac_all[iC1a[0]+self.M+m,iC1a[0]+self.M+iis] = np.cos(iis*np.pi*sigma[z_inds[m]]) 
                #afgeleiders naar Bs van het andere kanaal
                jac_all[iC1a[0]+self.M+m,iC2a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]])
                #afgeleides naar st van hetzelfde kanaal
                if ju_geg_fortid[0][1] == 'loc x=-L':
                    jac_all[iC1a[0]+self.M+m,iC1b[0]+0*self.M] = derv_C1[1][0][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC1b[0]+1*self.M] = derv_C1[1][1][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC1b[0]+2*self.M] = derv_C1[1][2][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC1b[0]+0*self.M+1+iis[:-1]] = derv_C1[1][3][:,z_inds[m]]
                elif ju_geg_fortid[0][1] == 'loc x=0':
                    jac_all[iC1a[0]+self.M+m,iC1b[1]-1*self.M] = derv_C1[1][0][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC1b[1]-2*self.M] = derv_C1[1][1][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC1b[1]-3*self.M] = derv_C1[1][2][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC1b[1]-1*self.M+1+iis[:-1]] = derv_C1[1][3][:,z_inds[m]]
                #afgeleides naar st van het andere kanaal
                if ju_geg_fortid[1][1] == 'loc x=-L':
                    jac_all[iC1a[0]+self.M+m,iC2b[0]+0*self.M] = -derv_C2[1][0][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC2b[0]+1*self.M] = -derv_C2[1][1][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC2b[0]+2*self.M] = -derv_C2[1][2][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC2b[0]+0*self.M+1+iis[:-1]] = -derv_C2[1][3][:,z_inds[m]]
                elif ju_geg_fortid[1][1] == 'loc x=0':
                    jac_all[iC1a[0]+self.M+m,iC2b[1]-1*self.M] = -derv_C2[1][0][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC2b[1]-2*self.M] = -derv_C2[1][1][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC2b[1]-3*self.M] = -derv_C2[1][2][z_inds[m]]
                    jac_all[iC1a[0]+self.M+m,iC2b[1]-1*self.M+1+iis[:-1]] = -derv_C2[1][3][:,z_inds[m]]
                    
            # =============================================================================
            #  second condition: 2-3=0
            # =============================================================================
            for m in range(self.M): #for now a loop, later replace with indices allocation. but that is less easy to check. 
                #real part
                #afgeleides naar Bs van hetzelfde kanaal
                jac_all[iC2a[0]+m,iC2a[0]+iis] = np.cos(iis*np.pi*sigma[z_inds[m]])
                #afgeleiders naar Bs van het andere kanaal
                jac_all[iC2a[0]+m,iC3a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]])
                #afgeleides naar st van hetzelfde kanaal
                if ju_geg_fortid[1][1] == 'loc x=-L':
                    jac_all[iC2a[0]+m,iC2b[0]+0*self.M] = derv_C2[0][0][z_inds[m]]
                    jac_all[iC2a[0]+m,iC2b[0]+1*self.M] = derv_C2[0][1][z_inds[m]]
                    jac_all[iC2a[0]+m,iC2b[0]+2*self.M] = derv_C2[0][2][z_inds[m]]
                    jac_all[iC2a[0]+m,iC2b[0]+0*self.M+1+iis[:-1]] = derv_C2[0][3][:,z_inds[m]]
                elif ju_geg_fortid[1][1] == 'loc x=0':
                    jac_all[iC2a[0]+m,iC2b[1]-1*self.M] = derv_C2[0][0][z_inds[m]]
                    jac_all[iC2a[0]+m,iC2b[1]-2*self.M] = derv_C2[0][1][z_inds[m]]
                    jac_all[iC2a[0]+m,iC2b[1]-3*self.M] = derv_C2[0][2][z_inds[m]]
                    jac_all[iC2a[0]+m,iC2b[1]-1*self.M+1+iis[:-1]] = derv_C2[0][3][:,z_inds[m]]
                #afgeleides naar st van het andere kanaal
                if ju_geg_fortid[2][1] == 'loc x=-L':
                    jac_all[iC2a[0]+m,iC3b[0]+0*self.M] = -derv_C3[0][0][z_inds[m]]
                    jac_all[iC2a[0]+m,iC3b[0]+1*self.M] = -derv_C3[0][1][z_inds[m]]
                    jac_all[iC2a[0]+m,iC3b[0]+2*self.M] = -derv_C3[0][2][z_inds[m]]
                    jac_all[iC2a[0]+m,iC3b[0]+0*self.M+1+iis[:-1]] = -derv_C3[0][3][:,z_inds[m]]
                elif ju_geg_fortid[2][1] == 'loc x=0':
                    jac_all[iC2a[0]+m,iC3b[1]-1*self.M] = -derv_C3[0][0][z_inds[m]]
                    jac_all[iC2a[0]+m,iC3b[1]-2*self.M] = -derv_C3[0][1][z_inds[m]]
                    jac_all[iC2a[0]+m,iC3b[1]-3*self.M] = -derv_C3[0][2][z_inds[m]]
                    jac_all[iC2a[0]+m,iC3b[1]-1*self.M+1+iis[:-1]] = -derv_C3[0][3][:,z_inds[m]]
    
                #imaginairy part
                #afgeleides naar Bs van hetzelfde kanaal
                jac_all[iC2a[0]+self.M+m,iC2a[0]+self.M+iis] = np.cos(iis*np.pi*sigma[z_inds[m]])
                #afgeleiders naar Bs van het andere kanaal
                jac_all[iC2a[0]+self.M+m,iC3a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]])
                #afgeleides naar st van hetzelfde kanaal
                if ju_geg_fortid[1][1] == 'loc x=-L':
                    jac_all[iC2a[0]+self.M+m,iC2b[0]+0*self.M] = derv_C2[1][0][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC2b[0]+1*self.M] = derv_C2[1][1][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC2b[0]+2*self.M] = derv_C2[1][2][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC2b[0]+0*self.M+1+iis[:-1]] = derv_C2[1][3][:,z_inds[m]]
                elif ju_geg_fortid[1][1] == 'loc x=0':
                    jac_all[iC2a[0]+self.M+m,iC2b[1]-1*self.M] = derv_C2[1][0][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC2b[1]-2*self.M] = derv_C2[1][1][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC2b[1]-3*self.M] = derv_C2[1][2][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC2b[1]-1*self.M+1+iis[:-1]] = derv_C2[1][3][:,z_inds[m]]
                #afgeleides naar st van het andere kanaal
                if ju_geg_fortid[2][1] == 'loc x=-L':
                    jac_all[iC2a[0]+self.M+m,iC3b[0]+0*self.M] = -derv_C3[1][0][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC3b[0]+1*self.M] = -derv_C3[1][1][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC3b[0]+2*self.M] = -derv_C3[1][2][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC3b[0]+0*self.M+1+iis[:-1]] = -derv_C3[1][3][:,z_inds[m]]
                elif ju_geg_fortid[2][1] == 'loc x=0':
                    jac_all[iC2a[0]+self.M+m,iC3b[1]-1*self.M] = -derv_C3[1][0][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC3b[1]-2*self.M] = -derv_C3[1][1][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC3b[1]-3*self.M] = -derv_C3[1][2][z_inds[m]]
                    jac_all[iC2a[0]+self.M+m,iC3b[1]-1*self.M+1+iis[:-1]] = -derv_C3[1][3][:,z_inds[m]]
    
            # =============================================================================
            # third condition: diffusive flux in tidal cycle conserved
            # =============================================================================
            for m in range(self.M): #for now a loop, later replace with indices allocation. but that is less easy to check. 
                #real part
                #channel 1
                if ju_geg_fortid[0][1] == 'loc x=-L': 
                    #afgeleides naar Bs van C1
                    jac_all[iC3a[0]+m,iC1a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2]**0.5
                    #afgeleides naar st van C1
                    jac_all[iC3a[0]+m,iC1b[0]+0*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[0][4][z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[0]+1*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[0][5][z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[0]+2*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[0][6][z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[0]+3*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[0][7][z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[0]+0*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[0][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[0]+1*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[0][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[0]+2*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[0][10][:,z_inds[m]]
    
                elif ju_geg_fortid[0][1] == 'loc x=0':
                    #afgeleides naar Bs van C1
                    jac_all[iC3a[0]+m,iC1a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2]**0.5
                    #afgeleides naar st van C1
                    jac_all[iC3a[0]+m,iC1b[1]-1*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[0][4][z_inds[m]] 
                    jac_all[iC3a[0]+m,iC1b[1]-2*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[0][5][z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[1]-3*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[0][6][z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[1]-4*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[0][7][z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[1]-1*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[0][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[1]-2*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[0][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC1b[1]-3*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[0][10][:,z_inds[m]]
    
                #channel 2
                if ju_geg_fortid[1][1] == 'loc x=-L': 
                    #afgeleides naar Bs van C2
                    jac_all[iC3a[0]+m,iC2a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2]**0.5
                    #afgeleides naar st van C2
                    jac_all[iC3a[0]+m,iC2b[0]+0*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[0][4][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[0]+1*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[0][5][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[0]+2*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[0][6][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[0]+3*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[0][7][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[0]+0*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[0][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[0]+1*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[0][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[0]+2*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[0][10][:,z_inds[m]]
                
                elif ju_geg_fortid[1][1] == 'loc x=0':    
                    #afgeleides naar Bs van C2
                    jac_all[iC3a[0]+m,iC2a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2]**0.5
                    #afgeleides naar st van C2
                    jac_all[iC3a[0]+m,iC2b[1]-1*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[0][4][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[1]-2*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[0][5][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[1]-3*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[0][6][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[1]-4*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[0][7][z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[1]-1*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[0][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[1]-2*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[0][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC2b[1]-3*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[0][10][:,z_inds[m]]
    
                #channel 3
                if ju_geg_fortid[2][1] == 'loc x=-L': 
                    #afgeleides naar Bs van C3
                    jac_all[iC3a[0]+m,iC3a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2]**0.5
                    #afgeleides naar st van C3
                    jac_all[iC3a[0]+m,iC3b[0]+0*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[0][4][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[0]+1*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[0][5][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[0]+2*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[0][6][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[0]+3*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[0][7][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[0]+0*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[0][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[0]+1*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[0][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[0]+2*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[0][10][:,z_inds[m]]
    
                elif ju_geg_fortid[2][1] == 'loc x=0': 
                    #afgeleides naar Bs van C3
                    jac_all[iC3a[0]+m,iC3a[0]+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2]**0.5
                    #afgeleides naar st van C3
                    jac_all[iC3a[0]+m,iC3b[1]-1*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[0][4][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[1]-2*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[0][5][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[1]-3*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[0][6][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[1]-4*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[0][7][z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[1]-1*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[0][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[1]-2*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[0][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+m,iC3b[1]-3*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[0][10][:,z_inds[m]]
    
                #imaginairy part follows here
                #channel 1
                if ju_geg_fortid[0][1] == 'loc x=-L': 
                    #afgeleides naar Bs van C1
                    jac_all[iC3a[0]+self.M+m,iC1a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2]**0.5
                    #afgeleides naar st van C1
                    jac_all[iC3a[0]+self.M+m,iC1b[0]+0*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[1][4][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[0]+1*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[1][5][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[0]+2*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[1][6][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[0]+3*self.M] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[1][7][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[0]+0*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[1][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[0]+1*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[1][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[0]+2*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][0 ] * derv_C1[2] * derv_C1[1][10][:,z_inds[m]]
    
                elif ju_geg_fortid[0][1] == 'loc x=0':
                    #afgeleides naar Bs van C1
                    jac_all[iC3a[0]+self.M+m,iC1a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2]**0.5
                    #afgeleides naar st van C1
                    jac_all[iC3a[0]+self.M+m,iC1b[1]-1*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[1][4][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[1]-2*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[1][5][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[1]-3*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[1][6][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[1]-4*self.M] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[1][7][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[1]-1*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[1][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[1]-2*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[1][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC1b[1]-3*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[0][0]]['H']*self.ch_pars[ju_geg_fortid[0][0]]['b'][-1] * derv_C1[2] * derv_C1[1][10][:,z_inds[m]]
    
                #channel 2
                if ju_geg_fortid[1][1] == 'loc x=-L': 
                    #afgeleides naar Bs van C2
                    jac_all[iC3a[0]+self.M+m,iC2a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2]**0.5
                    #afgeleides naar st van C2
                    jac_all[iC3a[0]+self.M+m,iC2b[0]+0*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[1][4][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[0]+1*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[1][5][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[0]+2*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[1][6][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[0]+3*self.M] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[1][7][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[0]+0*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[1][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[0]+1*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[1][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[0]+2*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][0 ] * derv_C2[2] * derv_C2[1][10][:,z_inds[m]]
                
                elif ju_geg_fortid[1][1] == 'loc x=0':    
                    #afgeleides naar Bs van C2
                    jac_all[iC3a[0]+self.M+m,iC2a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2]**0.5
                    #afgeleides naar st van C2
                    jac_all[iC3a[0]+self.M+m,iC2b[1]-1*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[1][4][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[1]-2*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[1][5][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[1]-3*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[1][6][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[1]-4*self.M] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[1][7][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[1]-1*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[1][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[1]-2*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[1][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC2b[1]-3*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[1][0]]['H']*self.ch_pars[ju_geg_fortid[1][0]]['b'][-1] * derv_C2[2] * derv_C2[1][10][:,z_inds[m]]
    
                #channel 3
                if ju_geg_fortid[2][1] == 'loc x=-L': 
                    #afgeleides naar Bs van C3
                    jac_all[iC3a[0]+self.M+m,iC3a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2]**0.5
                    #afgeleides naar st van C3
                    jac_all[iC3a[0]+self.M+m,iC3b[0]+0*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[1][4][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[0]+1*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[1][5][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[0]+2*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[1][6][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[0]+3*self.M] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[1][7][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[0]+0*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[1][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[0]+1*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[1][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[0]+2*self.M+1+iis[:-1]] = - self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][0 ] * derv_C3[2] * derv_C3[1][10][:,z_inds[m]]
    
                elif ju_geg_fortid[2][1] == 'loc x=0': 
                    #afgeleides naar Bs van C3
                    jac_all[iC3a[0]+self.M+m,iC3a[0]+self.M+iis] = - np.cos(iis*np.pi*sigma[z_inds[m]]) * self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2]**0.5
                    #afgeleides naar st van C3
                    jac_all[iC3a[0]+self.M+m,iC3b[1]-1*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[1][4][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[1]-2*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[1][5][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[1]-3*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[1][6][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[1]-4*self.M] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[1][7][z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[1]-1*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[1][8 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[1]-2*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[1][9 ][:,z_inds[m]]
                    jac_all[iC3a[0]+self.M+m,iC3b[1]-3*self.M+1+iis[:-1]] = self.ch_gegs[ju_geg_fortid[2][0]]['H']*self.ch_pars[ju_geg_fortid[2][0]]['b'][-1] * derv_C3[2] * derv_C3[1][10][:,z_inds[m]]
                            
        return jac_all
    

    # =============================================================================
    # initialisation of the answer vector
    # =============================================================================

    if init_method == 'from_notide': #normal initialisation
        try: init_all = np.zeros(ch_inds[self.ch_keys[-1]]['B_intot']['loc x=0'][1])
        except: init_all = np.zeros(ch_inds[self.ch_keys[-1]]['s_intot'][-1])
        #do the correct allocation, taking into account the boundary layer stuff
        for key in self.ch_keys: 
            temp = init[ch_inds[key]['s_inold'][0]:ch_inds[key]['s_inold'][1]]
            init_all[ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1]] = init[ch_inds[key]['s_inold'][0]:ch_inds[key]['s_inold'][1]]
    
    elif init_method == 'from_Khiter': #if we initialize from the Kh initalisation
        init_all = init.copy()
    
    else: #if there is nothing provided , start with zeros. 
        ch_init = {}
        init = np.array([])
        for key in self.ch_keys: ch_init[key] = np.zeros(self.ch_pars[key]['di'][-1]*self.M) #
        for key in self.ch_keys:init= np.concatenate([init,ch_init[key]])

    #plot initialisation
    #plt.plot(init_all.reshape((int(len(init_all)/self.M),self.M))[:,0],c='r')#-sss_n.reshape((int(len(sss)/self.M),self.M))[:,0])
    #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,1])
    #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,2])
    #plt.show()
    
  
    
    

    # =============================================================================
    # run the model
    # =============================================================================
    #do the first iteration step
    solu = sol_complete(init_all)
    jaco = jac_complete(init_all)
    sss_n =init_all - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix) 
    
    t=1
    print('That was iteration step ', t)
    sss=init_all
        
    
    # do the rest of the iterations
    while np.max(np.abs(sss-sss_n))>10e-6: #check whether the algoritm has converged
        sss = sss_n.copy() #update
        solu = sol_complete(sss)
        jaco = jac_complete(sss)

        sss_n =sss - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)
        
        #plotting
        #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,0]-sss_n.reshape((int(len(sss)/self.M),self.M))[:,0])
        #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,1])
        #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,2])
        #plt.show()
        
        t=1+t
        print('That was iteration step ', t)
    
        if t>=20: break
    
    if t<20:
        print('The algoritm has converged \n')  
    else:
        print('ERROR: no convergence')
        return [[None]]
    
    # =============================================================================
    # return output
    # a bit more complicated here because we have the boundary layer terms in the ans vector
    # =============================================================================
    out_ss = np.array([]) 
    out_Bs = {}
    
    for key in self.ch_keys:
        #subtract normal output
        out_ss = np.concatenate([out_ss, sss[ch_inds[key]['s_intot'][0]:ch_inds[key]['s_intot'][1]]])    
        #subtract output for tidal matching at junctions. bit ugly with the try statements but it seems the best for now. 
        res = []
        try:  
            temp = sss[ch_inds[key]['B_intot']['loc x=-L'][0]:ch_inds[key]['B_intot']['loc x=-L'][1]]
            res.append(temp[:self.M]+1j*temp[self.M:])        
        except:  res.append(np.zeros(self.M)+np.nan)
        try: 
            temp = sss[ch_inds[key]['B_intot']['loc x=0' ][0]:ch_inds[key]['B_intot']['loc x=0' ][1]]
            res.append(temp[:self.M]+1j*temp[self.M:])
        except: res.append(np.zeros(self.M)+np.nan)
        
        out_Bs[key] = np.array(res)

    return out_ss, out_Bs, sss


'''
def run_model(self, init_all):
    #do the run
    print('Start the salinity calculation with tides included')
    tijd = time.time()
    #spin up from subtidal model. Lets hope that does the job. 
    out = saltmodel_ti(self, init_all)
    self.sss_n, self.bbb_n = out
    print('The salinity calculation with tides takes  '+ str(time.time()-tijd)+' seconds')

    return 

'''







def run_model(self, init_all):
    # =============================================================================
    # code to run the subtidal salintiy model
    # =============================================================================
    print('Start the salinity calculation with tides')
    tijd = time.time()

    #spin up from subtidal model. Lets hope that does the job. 
    out = saltmodel_ti(self, init_all)

    
    if out[0][0] == None: #check if this calculation worked. If not, we will try again with a higher horizontal diffusivity
        #build Kfac trajectory
        n_guess = 5 #the number of iteration steps
        Kf_start = 5 #the factor with what to multiply the mixing 
        Kfac = np.linspace(Kf_start,1,n_guess)

        #do the procedure again, with a beter approximation
        for sim in range(n_guess): 
            #choose a higher value for Kh
            count=0
            for key in self.ch_keys: 
                #formulation depends on how Kh is formulated. 
                if self.ch2 == None: 
                    self.ch_pars[key]['Kh'] = self.ch*self.ch_gegs[key]['Ut']*self.ch_pars[key]['b']*Kfac[sim]
                elif self.ch == None:
                    self.ch_pars[key]['Kh'] = (self.ch2+np.zeros(self.ch_pars[key]['di'][-1]))*Kfac[sim] #horizontal mixing coefficient
                    #add the increase of Kh in the adjacent sea domain
                    if self.ch_gegs[key]['loc x=-L'][0] == 's' : self.ch_pars[key]['Kh'][:self.ch_pars[key]['di'][1]] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][1]] \
                                                                    * self.ch_pars[key]['b'][:self.ch_pars[key]['di'][1]]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][1]]
                    if self.ch_gegs[key]['loc x=0'][0] == 's': self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]:] = self.ch_pars[key]['Kh'][self.ch_pars[key]['di'][-2]] \
                                                                    * self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]:]/self.ch_pars[key]['b'][self.ch_pars[key]['di'][-2]]
                count+=1
            #do the simulation
            if sim ==0 : out = saltmodel_ti(self, init_all)
            else: out = saltmodel_ti(self,out[2],init_method = 'from_Khiter')
                
            if out[0][0] == None: #if this also not works, stop the calculation
                import sys
                sys.exit("ABORT CALCULATION: Also with increased Kh no answer has been found. Check your input and think about \
                         if the model has a physical solution. If you think it has, you might wanna try increasing Kf_start or n_guess")   
                         
            print('Step ', sim, ' of ',n_guess-1,' is finished')

    self.sss_n, self.bbb_n, troep = out
    print('The salinity calculation with tides takes  '+ str(time.time()-tijd)+' seconds')

    return 



















