# =============================================================================
# solve time-independent salinity in a random network of channels
# subtidal code, no tides included at all. 
# horizontal diffusion is thus a bit low here
# =============================================================================

#import libraries
import numpy as np
import scipy as sp          #do fits
from scipy import optimize   , interpolate 
import time                              #measure time of operation
import matplotlib.pyplot as plt         
import sys


def saltmodel_ti_nt(self,init_all):
    # =============================== =============================================
    # the function to calculate the subtidal salinity
    # =============================================================================
    
    #create dicionaries
    ch_parja = {} # parameters for solution vector/jacobian
    ch_inds = {} # indices for solution vector/jacobian

    #lists of ks and ns and pis
    kkp = np.linspace(1,self.N,self.N)*np.pi #k*pi
    nnp = np.linspace(1,self.N,self.N)*np.pi #n*pi
    pkn = np.array([kkp]*self.N) + np.transpose([nnp]*self.N) #pi*(n+k)
    pnk = np.array([nnp]*self.N) - np.transpose([kkp]*self.N) #pi*(n-k)
    np.fill_diagonal(pkn,None),np.fill_diagonal(pnk,None)
     
    # =============================================================================
    # parameters for solution vector/jacobian, like in the single channel code. 
    # for meaning of the symbols, most of it is written in freshwater pulses
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
        ch_parja[key]['C7a'] = (-self.ch_pars[key]['bex']**-1 * self.ch_pars[key]['Kh'] / 2 - self.ch_pars[key]['Kh_x']/2)
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
        
        #depth-averaged
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
        
        #inner boundaries
        ch_inds[key]['in_bnd'] = (self.ch_pars[key]['di'][np.arange(1,len(self.ch_pars[key]['nxn']))]*self.M+np.arange(self.M)[:,np.newaxis]).flatten()
    
        #boundaries
        ch_inds[key]['bnd1_x=-L'] = 0*self.M+np.arange(self.M) 
        ch_inds[key]['bnd2_x=-L'] = 1*self.M+np.arange(self.M) 
        ch_inds[key]['bnd3_x=-L'] = 2*self.M+np.arange(self.M) 
    
        ch_inds[key]['bnd1_x=0'] = (self.ch_pars[key]['di'][-1]-1)*self.M+np.arange(self.M)
        ch_inds[key]['bnd2_x=0'] = (self.ch_pars[key]['di'][-1]-2)*self.M+np.arange(self.M)
        ch_inds[key]['bnd3_x=0'] = (self.ch_pars[key]['di'][-1]-3)*self.M+np.arange(self.M)
                
    
    # =============================================================================
    # from here functions for the solution vector and associated Jacobian are defined
    # =============================================================================
    def sol_inner(ans, key):
        # =============================================================================
        # build the internal part of the solution vector (for every channel)
        # =============================================================================

        #create empty vector
        so = np.zeros(self.ch_pars[key]['di'][-1]*self.M)
        
        #local variables
        inds = ch_inds[key].copy()
        dl = self.ch_pars[key]['dl'].copy()
        pars = ch_parja[key].copy()
        
        #add parts to the solution vector
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
        
        #depth-averaged salt balance
        so[inds['x_m']] = so[inds['x_m']] + (pars['C10a'][inds['x']]*(ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) 
                            + pars['C10b'][inds['x']]*(ans[inds['xp_m']] - 2*ans[inds['x_m']] + ans[inds['xm_m']])/(dl[inds['x']]**2) 
                            + (ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) * np.sum([pars['C10c'][inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                            + (ans[inds['xp_m']] - 2*ans[inds['x_m']] + ans[inds['xm_m']])/(dl[inds['x']]**2) * np.sum([pars['C10d'][n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) 
                            + np.sum([pars['C10e'][inds['x'],n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*dl[inds['x']]) for n in range(1,self.M)],0) * self.ch_pars[key]['Q'] 
                            + (ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) 
                            * np.sum([pars['C10f'][n-1]*(ans[inds['xp_m']+n]-ans[inds['xm_m']+n])/(2*dl[inds['x']]) for n in range(1,self.M)],0)
                            + np.sum([pars['C10g'][inds['x'],n-1]*ans[inds['x_m']+n] for n in range(1,self.M)],0) + pars['C10h'][inds['x']]*self.ch_pars[key]['Q']*(ans[inds['xp_m']]-ans[inds['xm_m']])/(2*dl[inds['x']]) )

        #boundaries of the segments - this for loop may live on as a living fossil.
        for d in range(1,len(self.ch_pars[key]['nxn'])):
            s_3, s_2, s_1 = ans[(self.ch_pars[key]['di'][d]-3)*self.M:(self.ch_pars[key]['di'][d]-2)*self.M], ans[(self.ch_pars[key]['di'][d]-2)*self.M:(self.ch_pars[key]['di'][d]-1)*self.M], ans[(self.ch_pars[key]['di'][d]-1)*self.M:self.ch_pars[key]['di'][d]*self.M]
            s0, s1, s2 = ans[self.ch_pars[key]['di'][d]*self.M:(self.ch_pars[key]['di'][d]+1)*self.M], ans[(self.ch_pars[key]['di'][d]+1)*self.M:(self.ch_pars[key]['di'][d]+2)*self.M], ans[(self.ch_pars[key]['di'][d]+2)*self.M:(self.ch_pars[key]['di'][d]+3)*self.M]
            so[(self.ch_pars[key]['di'][d]-1)*self.M:(self.ch_pars[key]['di'][d]-0)*self.M] = (3*s_1-4*s_2+s_3)/(2*dl[self.ch_pars[key]['di'][d]-1]) - (-3*s0+4*s1-s2)/(2*dl[self.ch_pars[key]['di'][d]])
            so[(self.ch_pars[key]['di'][d]-0)*self.M:(self.ch_pars[key]['di'][d]+1)*self.M] = s_1 - s0
        
        return so
    
    def sol_complete(ans_all):
        # =============================================================================
        # build solution vector for the full network of channels
        # =============================================================================

        #starting indices of the channels int he big matrix
        ind = [0]
        for key in self.ch_keys: ind.append(ind[-1]+self.ch_pars[key]['di'][-1]*self.M)
        
        #first: break the full answer vector down to answer vectors per channel
        ch_ans = {}
        count = 0
        for key in self.ch_keys:
            ch_ans[key] = ans_all[ind[count]:ind[count+1]]
            count +=1                             
    
        #run the solution vectors for every channel and add to dictionary
        ch_sols = {}
        for key in self.ch_keys: ch_sols[key] = sol_inner(ch_ans[key], key)
     
        # =============================================================================
        # Add the boundaries to the solution vector
        # =============================================================================
        for key in self.ch_keys:
            #river boundaries _ sriv prescribed
            if self.ch_gegs[key]['loc x=-L'][0] == 'r':
                ch_sols[key][0] = ch_ans[key][0] - self.sri[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca
                ch_sols[key][1:self.M] = ch_ans[key][1:self.M] 
                
            elif self.ch_gegs[key]['loc x=0'][0] == 'r':
                ch_sols[key][-self.M] = ch_ans[key][-self.M] - self.sri[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca
                ch_sols[key][-self.N:] = ch_ans[key][-self.N:] 
                
            #weir boundaries: flux prescribed (kind of)
            if self.ch_gegs[key]['loc x=-L'][0] == 'w':
                if self.ch_pars[key]['Q']>0: #flux is equal to the advective flux through weir
                    ch_sols[key][0] = self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*ch_ans[key][0]*self.Lsc - self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*self.swe[int(self.ch_gegs[key]['loc x=-L'][1])-1]/self.soc_sca*self.Lsc \
                        - self.ch_pars[key]['Kh'][0] * (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])                
                elif self.ch_pars[key]['Q']<=0: #flux is equal to the advective flux through weir, set by river salinity
                    ch_sols[key][0] =  - self.ch_pars[key]['Kh'][0] * (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])
                #no flux at depth 
                ch_sols[key][1:self.M] = (-3*ch_ans[key][1:self.M]+4*ch_ans[key][self.M+1:2*self.M]-ch_ans[key][2*self.M+1:3*self.M])/(2*self.ch_pars[key]['dl'][0]) 
                
            elif self.ch_gegs[key]['loc x=0'][0] == 'w':
                if self.ch_pars[key]['Q']>=0: #flux is equal to the advective flux through weir
                    ch_sols[key][-self.M] = - self.ch_pars[key]['Kh'][-1] * (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][-self.M+n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])
                elif self.ch_pars[key]['Q']<0: #flux is equal to the advective flux through weir, set by weir salinity
                    ch_sols[key][-self.M] = self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*ch_ans[key][-self.M]*self.Lsc - self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*self.swe[int(self.ch_gegs[key]['loc x=0'][1])-1]/self.soc_sca*self.Lsc \
                        - self.ch_pars[key]['Kh'][-1] * (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                        + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][-self.M+n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                        + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])  
                ch_sols[key][-self.N:] =  (3*ch_ans[key][-self.N:]-4*ch_ans[key][-2*self.M+1:-self.M]+ch_ans[key][-3*self.M+1:-2*self.M])/(2*self.ch_pars[key]['dl'][-1]) 
                
                
            #har boundaries: only seaward flux 
            if self.ch_gegs[key]['loc x=-L'][0] == 'h':
                #only seaward flux
                ch_sols[key][0] = - self.ch_pars[key]['Kh'][0] * (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                    + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                    + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                    + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])
                #no flux at depth 
                ch_sols[key][1:self.M] = (-3*ch_ans[key][1:self.M]+4*ch_ans[key][self.M+1:2*self.M]-ch_ans[key][2*self.M+1:3*self.M])/(2*self.ch_pars[key]['dl'][0]) 

            elif self.ch_gegs[key]['loc x=0'][0] == 'h':
                #only seaward flux
                ch_sols[key][-self.M] = - self.ch_pars[key]['Kh'][-1] * (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                    + 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.sum([np.cos(nnp[n])*ch_ans[key][-self.M+n+1]/nnp[n]**2 for n in range(self.N)])*self.Lsc \
                    + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                    + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2)) for n in range(self.N)])
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
        # add the conditions at the junctions to the solution vector
        # =============================================================================
        for j in range(self.n_j):
            #find the connections
            ju_geg = []
            for key in self.ch_keys: 
                if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1):  ju_geg.append([key,'x=-L',ch_inds[key]['bnd1_x=-L'],ch_inds[key]['bnd2_x=-L'],ch_inds[key]['bnd3_x=-L']])
                elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): ju_geg.append([key,'x=0',ch_inds[key]['bnd1_x=0'],ch_inds[key]['bnd2_x=0'],ch_inds[key]['bnd3_x=0']])
           
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
                if ju_geg[i][1] == 'x=-L':
                    temp = temp - (ch_parja[key_here]['C13a_x=-L']*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]] + np.sum(ch_ans[key_here][ju_geg[i][2][1:]]*(ch_parja[key_here]['C13b_x=-L']*self.ch_pars[key_here]['Q'] + ch_parja[key_here]['C13c_x=-L'] * 
                              (-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0]) )) 
                              +  ch_parja[key_here]['C13d_x=-L']*(-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0])  )
                   
                elif ju_geg[i][1] == 'x=0':
                    temp = temp + (ch_parja[key_here]['C13a_x=0']*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]] + np.sum(ch_ans[key_here][ju_geg[i][2][1:]]*(ch_parja[key_here]['C13b_x=0']*self.ch_pars[key_here]['Q']+ch_parja[key_here]['C13c_x=0'] * 
                             (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1]) )) + ch_parja[key_here]['C13d_x=0'] * 
                             (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1]) )
                              
                else: print('ERROR')
            ch_sols[ju_geg[2][0]][ju_geg[2][2][0]] = temp #add to the solution fector
    
            #calculate for transport at depth
            for k in range(1,self.M):
                temp = 0
                for i in range(3): #calculate contributions from channels seperately 
                    key_here = ju_geg[i][0]
                    if ju_geg[i][1] == 'x=-L':
                        temp = temp - (ch_parja[key_here]['C12a_x=-L'] * (-3*ch_ans[key_here][ju_geg[i][2][k]]+4*ch_ans[key_here][ju_geg[i][3][k]]-ch_ans[key_here][ju_geg[i][4][k]])/(2*self.ch_pars[key_here]['dl'][0])
                                       + ch_parja[key_here]['C12b_x=-L'][k-1] * ch_ans[key_here][ju_geg[i][2][k]] * (-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0])  
                                       + np.sum([ch_parja[key_here]['C12c_x=-L'][k-1,n]*ch_ans[key_here][ju_geg[i][2]][n+1] for n in range(self.N)])*(-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0]) 
                                       + ch_parja[key_here]['C12d_x=-L'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][k]]
                                       + np.sum([ch_parja[key_here]['C12e_x=-L'][k-1,n]*ch_ans[key_here][ju_geg[i][2][1+n]] for n in range(self.N)])*self.ch_pars[key_here]['Q']
                                       + ch_parja[key_here]['C12f_x=-L'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]]
                                       + ch_parja[key_here]['C12g_x=-L'][k-1]*ch_ans[key_here][ju_geg[i][2][0]] * (-3*ch_ans[key_here][ju_geg[i][2][0]]+4*ch_ans[key_here][ju_geg[i][3][0]]-ch_ans[key_here][ju_geg[i][4][0]])/(2*self.ch_pars[key_here]['dl'][0]) 
                                       )

                    elif ju_geg[i][1] == 'x=0' :
                        temp = temp + (ch_parja[key_here]['C12a_x=0'] * (ch_ans[key_here][ju_geg[i][4][k]]-4*ch_ans[key_here][ju_geg[i][3][k]]+3*ch_ans[key_here][ju_geg[i][2][k]])/(2*self.ch_pars[key_here]['dl'][-1])  
                                       + ch_parja[key_here]['C12b_x=0'][k-1] * ch_ans[key_here][ju_geg[i][2][k]] * (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1])  
                                       + np.sum([ch_parja[key_here]['C12c_x=0'][k-1,n]*ch_ans[key_here][ju_geg[i][2]][n+1] for n in range(self.N)]) * (ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1]) 
                                       + ch_parja[key_here]['C12d_x=0'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][k]]
                                       + np.sum([ch_parja[key_here]['C12e_x=0'][k-1,n]*ch_ans[key_here][ju_geg[i][2]][n+1] for n in range(self.N)])*self.ch_pars[key_here]['Q']
                                       + ch_parja[key_here]['C12f_x=0'][k-1]*self.ch_pars[key_here]['Q']*ch_ans[key_here][ju_geg[i][2][0]]
                                       + ch_parja[key_here]['C12g_x=0'][k-1]*ch_ans[key_here][ju_geg[i][2][0]]*(ch_ans[key_here][ju_geg[i][4][0]]-4*ch_ans[key_here][ju_geg[i][3][0]]+3*ch_ans[key_here][ju_geg[i][2][0]])/(2*self.ch_pars[key_here]['dl'][-1])
                                       )
        
                    else: print("ERROR")
                ch_sols[ju_geg[2][0]][ju_geg[2][2][k]] = temp #add to the solution vector
        
        # =============================================================================
        # complete to one big solution vector
        # =============================================================================
        sol_all = np.array([])
        for key in self.ch_keys: sol_all = np.concatenate([sol_all,ch_sols[key]])
            
        return sol_all
    
    def jac_inner(ans, key):
        # =============================================================================
        # build the associated internal part of the Jacobian
        # =============================================================================
        
        #create empty matrix
        jac = np.zeros((self.ch_pars[key]['di'][-1]*self.M,self.ch_pars[key]['di'][-1]*self.M))
        
        #local variables
        inds = ch_inds[key].copy()
        dl = self.ch_pars[key]['dl'].copy()
        pars = ch_parja[key].copy()
        
        # =============================================================================
        # vertical balance
        # =============================================================================
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
    
        # =============================================================================
        # Depth-averaged balance
        # =============================================================================
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
        
        
        # =============================================================================
        # boundaries of segments
        # =============================================================================
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
    
        return jac
    
    def jac_complete(ans_all):
        # =============================================================================
        # build the jacobian for the full network
        # =============================================================================
        #starting indices of the channels int he big matrix
        ind = [0]
        for key in self.ch_keys: ind.append(ind[-1]+self.ch_pars[key]['di'][-1]*self.M)
    
        #first: break down ans_all to ch_ans
        ch_ans = {}
        count = 0
        for key in self.ch_keys:
            ch_ans[key] = ans_all[ind[count]:ind[count+1]]
            count +=1   
            
        #build the jacobians of the internal parts of the domains
        ch_jacs = {}
        for key in self.ch_keys: ch_jacs[key] = jac_inner(ch_ans[key], key)
    
        # =============================================================================
        # add boundaries to Jacobian
        # =============================================================================
        for key in self.ch_keys:
            #river,sea boundaries: same conditions
            if self.ch_gegs[key]['loc x=-L'][0] == 'r' or self.ch_gegs[key]['loc x=-L'][0] == 's':  ch_jacs[key][np.arange(self.M),np.arange(self.M)] = 1
            elif self.ch_gegs[key]['loc x=0'][0] == 'r' or self.ch_gegs[key]['loc x=0'][0] == 's':  ch_jacs[key][np.arange(-self.M,0),np.arange(-self.M,0)] = 1
            
            # weir boundaries: bit complicated jacobian. Condition is on the transport
            if self.ch_gegs[key]['loc x=-L'][0] == 'w':
                if self.ch_pars[key]['Q']>0: #flux is equal to the advective flux through weir
                    ch_jacs[key][0,0] =  self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*self.Lsc +3*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) -3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) \
                        * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][0,self.M] =  -4*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) + 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][0,2*self.M] = self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) - self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    for j in range(1,self.M):
                        ch_jacs[key][0,j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  
               
                elif self.ch_pars[key]['Q']<=0: #flux is equal to the advective flux through weir, set by river salinity
                    ch_jacs[key][0,0] = 3*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) -3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) \
                        * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][0,self.M] =  -4*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) + 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][0,2*self.M] = self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) - self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    for j in range(1,self.M):
                        ch_jacs[key][0,j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  
 
                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][j, 2*self.M+j] = -1/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, self.M+j] = 4/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, j] = -3/(2*self.ch_pars[key]['dl'][0])  

            elif self.ch_gegs[key]['loc x=0'][0] == 'w':
                if self.ch_pars[key]['Q']>0: #flux is equal to the advective flux through weir
                    #no total flux
                    ch_jacs[key][-self.M,-self.M] = - 3*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + 3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) \
                        * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][-self.M,-2*self.M] =  4*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) - 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][-self.M,-3*self.M] = - self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    for j in range(1,self.M):
                        ch_jacs[key][-self.M,-self.M+j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  
 
                elif self.ch_pars[key]['Q']<=0: #flux is equal to the advective flux through weir, set by river salinity
                    #no total flux
                    ch_jacs[key][-self.M,-self.M] =  self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*self.Lsc - 3*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + 3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) \
                        * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][-self.M,-2*self.M] =  4*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) - 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                          + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    ch_jacs[key][-self.M,-3*self.M] = - self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                           + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                    for j in range(1,self.M):
                        ch_jacs[key][-self.M,-self.M+j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                            + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                                * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  
 
                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][-self.M+j, -3*self.M+j] = 1/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -2*self.M+j] = -4/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -self.M+j] = 3/(2*self.ch_pars[key]['dl'][-1])        
                    
            #har boundaries: 
            if self.ch_gegs[key]['loc x=-L'][0] == 'h':
                #no total flux
                ch_jacs[key][0,0] = 3*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) -3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) \
                    * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                ch_jacs[key][0,self.M] =  -4*self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) + 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                      + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                ch_jacs[key][0,2*self.M] = self.ch_pars[key]['Kh'][0]/(2*self.ch_pars[key]['dl'][0]) - self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][0]) * np.sum([ch_ans[key][n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                       + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                for j in range(1,self.M):
                    ch_jacs[key][0,j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][0]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (-3*ch_ans[key][0] + 4*ch_ans[key][self.M] - ch_ans[key][2*self.M])/(2*self.ch_pars[key]['dl'][0]) \
                            * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  

                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][j, 2*self.M+j] = -1/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, self.M+j] = 4/(2*self.ch_pars[key]['dl'][0]) 
                    ch_jacs[key][j, j] = -3/(2*self.ch_pars[key]['dl'][0])  

            elif self.ch_gegs[key]['loc x=0'][0] == 'h':
                #no total flux
                ch_jacs[key][-self.M,-self.M] =  - 3*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + 3*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) \
                    * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                ch_jacs[key][-self.M,-2*self.M] =  4*self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) - 4*self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                      + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                ch_jacs[key][-self.M,-3*self.M] = - self.ch_pars[key]['Kh'][-1]/(2*self.ch_pars[key]['dl'][-1]) + self.ch_pars[key]['alf']*self.soc_sca/(2*self.ch_pars[key]['dl'][-1]) * np.sum([ch_ans[key][-self.M+n+1]*(2*self.ch_pars[key]['g4']*np.cos(nnp[n])/nnp[n]**2 \
                       + self.ch_pars[key]['g5']*((6*np.cos(nnp[n])-6)/nnp[n]**4 - 3*np.cos(nnp[n])/nnp[n]**2))  for n in range(self.N)])
                for j in range(1,self.M):
                    ch_jacs[key][-self.M,-self.M+j] = 2*self.ch_pars[key]['g2']*self.ch_pars[key]['bH_1'][-1]*self.ch_pars[key]['Q']*np.cos(nnp[j-1])/nnp[j-1]**2*self.Lsc \
                        + self.ch_pars[key]['alf']*self.soc_sca* (3*ch_ans[key][-self.M] - 4*ch_ans[key][-2*self.M] + ch_ans[key][-3*self.M])/(2*self.ch_pars[key]['dl'][-1]) \
                            * (2*self.ch_pars[key]['g4']*np.cos(nnp[j-1])/nnp[j-1]**2 + self.ch_pars[key]['g5']*((6*np.cos(nnp[j-1])-6)/nnp[j-1]**4 - 3*np.cos(nnp[j-1])/nnp[j-1]**2))  

                #no diffusive flux at depth
                for j in range(1,self.M):
                    ch_jacs[key][-self.M+j, -3*self.M+j] = 1/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -2*self.M+j] = -4/(2*self.ch_pars[key]['dl'][-1]) 
                    ch_jacs[key][-self.M+j, -self.M+j] = 3/(2*self.ch_pars[key]['dl'][-1])  

        
        # =============================================================================
        # build complete matrix
        # =============================================================================
        di_all = [0]
        for key in self.ch_keys:
            ch_inds[key]['start'] = di_all[-1]
            di_all.append(di_all[-1]+len(ch_jacs[key]))
        
        jac_all = np.zeros((di_all[-1],di_all[-1]))
        count = 0
        for key in self.ch_keys:
            jac_all[di_all[count]:di_all[count+1],di_all[count]:di_all[count+1]] = ch_jacs[key]
            count += 1    
        
        # =============================================================================
        # add junctions to the large matrix
        # =============================================================================
        for j in range(self.n_j):
            #find the connections
            ju_geg = []
            for key in self.ch_keys: 
                if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1): 
                    ju_geg.append([key,'x=-L',ch_inds[key]['bnd1_x=-L'],ch_inds[key]['bnd2_x=-L'],ch_inds[key]['bnd3_x=-L']])
                elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): 
                    ju_geg.append([key,'x=0',ch_inds[key]['bnd1_x=0'],ch_inds[key]['bnd2_x=0'],ch_inds[key]['bnd3_x=0']])
            
            #indices  
            ind = ([ju_geg[0][2]+ch_inds[ju_geg[0][0]]['start'] , ju_geg[0][3]+ch_inds[ju_geg[0][0]]['start'] , ju_geg[0][4]+ch_inds[ju_geg[0][0]]['start']],
                [ju_geg[1][2]+ch_inds[ju_geg[1][0]]['start'] , ju_geg[1][3]+ch_inds[ju_geg[1][0]]['start'] , ju_geg[1][4]+ch_inds[ju_geg[1][0]]['start']],
                [ju_geg[2][2]+ch_inds[ju_geg[2][0]]['start'] , ju_geg[2][3]+ch_inds[ju_geg[2][0]]['start'] , ju_geg[2][4]+ch_inds[ju_geg[2][0]]['start']] )
            
            #first: 0-1 equal
            jac_all[ind[0][0],ind[0][0]] = 1
            jac_all[ind[0][0],ind[1][0]] = -1
    
            #second: 1-2 equal
            jac_all[ind[1][0],ind[1][0]] = 1
            jac_all[ind[1][0],ind[2][0]] = -1
            
            
            #third: transport
            #depth-averaged transport
            for i in range(3): #calculate contributions from channels seperately 
                if ju_geg[i][1] == 'x=-L':
                   jac_all[ind[2][0][0],ind[i][0][0]] = (- ch_parja[ju_geg[i][0]]['C13a_x=-L'] * self.ch_pars[ju_geg[i][0]]['Q'] +  3/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) *  np.sum(ch_parja[ju_geg[i][0]]['C13c_x=-L']*ans_all[ind[i][0][1:]])  
                                                         + ch_parja[ju_geg[i][0]]['C13d_x=-L'] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) )
                   jac_all[ind[2][0][0],ind[i][1][0]] = (ch_parja[ju_geg[i][0]]['C13d_x=-L'] * -4/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) 
                                                         - 4/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) *  np.sum(ch_parja[ju_geg[i][0]]['C13c_x=-L']*ans_all[ind[i][0][1:]]) )
                   jac_all[ind[2][0][0],ind[i][2][0]] = (ch_parja[ju_geg[i][0]]['C13d_x=-L'] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) 
                                                         + 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) *  np.sum(ch_parja[ju_geg[i][0]]['C13c_x=-L']*ans_all[ind[i][0][1:]]) )
                   for k in range(1,self.M): jac_all[ind[2][0][0],ind[i][0][k]]  = jac_all[ind[2][0][0],ind[i][0][k]] - ch_parja[ju_geg[i][0]]['C13b_x=-L'][k-1] * self.ch_pars[ju_geg[i][0]]['Q'] + ch_parja[ju_geg[i][0]]['C13c_x=-L'][k-1] * (ans_all[ind[i][2][0]] - 4*ans_all[ind[i][1][0]] + 3*ans_all[ind[i][0][0]] )/(2*self.ch_pars[ju_geg[i][0]]['dl'][0])
             
                elif ju_geg[i][1] == 'x=0': 
                    jac_all[ind[2][0][0],ind[i][0][0]] = (ch_parja[ju_geg[i][0]]['C13a_x=0'] * self.ch_pars[ju_geg[i][0]]['Q'] + ch_parja[ju_geg[i][0]]['C13d_x=0'] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) 
                                                          + 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) *  np.sum(ch_parja[ju_geg[i][0]]['C13c_x=0']*ans_all[ind[i][0][1:]]) )
                    jac_all[ind[2][0][0],ind[i][1][0]] = (ch_parja[ju_geg[i][0]]['C13d_x=0'] * -4/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) 
                                                          - 4/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) *  np.sum(ch_parja[ju_geg[i][0]]['C13c_x=0']*ans_all[ind[i][0][1:]]) )
                    jac_all[ind[2][0][0],ind[i][2][0]] = (ch_parja[ju_geg[i][0]]['C13d_x=0'] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) 
                                                          + 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) *  np.sum(ch_parja[ju_geg[i][0]]['C13c_x=0']*ans_all[ind[i][0][1:]]) )
                    for k in range(1,self.M): jac_all[ind[2][0][0],ind[i][0][k]] = jac_all[ind[2][0][0],ind[i][0][k]] + ch_parja[ju_geg[i][0]]['C13b_x=0'][k-1] * self.ch_pars[ju_geg[i][0]]['Q'] + ch_parja[ju_geg[i][0]]['C13c_x=0'][k-1] * (ans_all[ind[i][2][0]]-4*ans_all[ind[i][1][0]]+3*ans_all[ind[i][0][0]])/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1])
    
                else: print('ERROR')
    
            
            #transport at every vertical level
            for k in range(1,self.M):
                for i in range(3): #calculate contributions from channels seperately 
    
                    if ju_geg[i][1] == 'x=-L':
                        jac_all[ind[2][0][k],ind[i][0][0]] = ch_parja[ju_geg[i][0]]['C12b_x=-L'][k-1] * ans_all[ind[i][0][k]] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) + np.sum([ch_parja[ju_geg[i][0]]['C12c_x=-L'][k-1,n-1] * ans_all[ind[i][0][n]] for n in range(1,self.M)]) * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) \
                            - ch_parja[ju_geg[i][0]]['C12f_x=-L'][k-1]*self.ch_pars[ju_geg[i][0]]['Q'] + ch_parja[ju_geg[i][0]]['C12g_x=-L'][k-1] * ( ans_all[ind[i][2][0]]-4*ans_all[ind[i][1][0]]+3*ans_all[ind[i][0][0]] )/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) + ch_parja[ju_geg[i][0]]['C12g_x=-L'][k-1] * ans_all[ind[i][0][0]] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][0])
                        jac_all[ind[2][0][k],ind[i][1][0]] = ch_parja[ju_geg[i][0]]['C12b_x=-L'][k-1] * ans_all[ind[i][0][k]] *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) + np.sum([ch_parja[ju_geg[i][0]]['C12c_x=-L'][k-1,n-1] * ans_all[ind[i][0][n]] for n in range(1,self.M)]) *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) + ch_parja[ju_geg[i][0]]['C12g_x=-L'][k-1] * ans_all[ind[i][0][0]] *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][0])
                        jac_all[ind[2][0][k],ind[i][2][0]] = ch_parja[ju_geg[i][0]]['C12b_x=-L'][k-1] * ans_all[ind[i][0][k]] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) + np.sum([ch_parja[ju_geg[i][0]]['C12c_x=-L'][k-1,n-1] * ans_all[ind[i][0][n]] for n in range(1,self.M)]) * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) + ch_parja[ju_geg[i][0]]['C12g_x=-L'][k-1] * ans_all[ind[i][0][0]] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][0])
                        
                        jac_all[ind[2][0][k],ind[i][0][k]] = ch_parja[ju_geg[i][0]]['C12a_x=-L'] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) + ch_parja[ju_geg[i][0]]['C12b_x=-L'][k-1] * ( ans_all[ind[i][2][0]]-4*ans_all[ind[i][1][0]]+3*ans_all[ind[i][0][0]] )/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) - ch_parja[ju_geg[i][0]]['C12d_x=-L'][k-1]*self.ch_pars[ju_geg[i][0]]['Q']
                        jac_all[ind[2][0][k],ind[i][1][k]] = ch_parja[ju_geg[i][0]]['C12a_x=-L'] *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][0])
                        jac_all[ind[2][0][k],ind[i][2][k]] = ch_parja[ju_geg[i][0]]['C12a_x=-L'] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][0])
                        
                        for k2 in range(1,self.M): jac_all[ind[2][0][k],ind[i][0][k2]] = jac_all[ind[2][0][k],ind[i][0][k2]] + ch_parja[ju_geg[i][0]]['C12c_x=-L'][k-1,k2-1] * ( ans_all[ind[i][2][0]]-4*ans_all[ind[i][1][0]]+3*ans_all[ind[i][0][0]] )/(2*self.ch_pars[ju_geg[i][0]]['dl'][0]) - ch_parja[ju_geg[i][0]]['C12e_x=-L'][k-1,k2-1]*self.ch_pars[ju_geg[i][0]]['Q']
    
                    elif ju_geg[i][1] == 'x=0' :
                        jac_all[ind[2][0][k],ind[i][0][0]] = ch_parja[ju_geg[i][0]]['C12b_x=0'][k-1] * ans_all[ind[i][0][k]] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + np.sum([ch_parja[ju_geg[i][0]]['C12c_x=0'][k-1,n-1] * ans_all[ind[i][0][n]] for n in range(1,self.M)]) * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) \
                            + ch_parja[ju_geg[i][0]]['C12f_x=0'][k-1]*self.ch_pars[ju_geg[i][0]]['Q'] + ch_parja[ju_geg[i][0]]['C12g_x=0'][k-1] * ( ans_all[ind[i][2][0]]-4*ans_all[ind[i][1][0]]+3*ans_all[ind[i][0][0]] )/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + ch_parja[ju_geg[i][0]]['C12g_x=0'][k-1] * ans_all[ind[i][0][0]] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1])
                        jac_all[ind[2][0][k],ind[i][1][0]] = ch_parja[ju_geg[i][0]]['C12b_x=0'][k-1] * ans_all[ind[i][0][k]] *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + np.sum([ch_parja[ju_geg[i][0]]['C12c_x=0'][k-1,n-1] * ans_all[ind[i][0][n]] for n in range(1,self.M)]) *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + ch_parja[ju_geg[i][0]]['C12g_x=0'][k-1] * ans_all[ind[i][0][0]] *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1])
                        jac_all[ind[2][0][k],ind[i][2][0]] = ch_parja[ju_geg[i][0]]['C12b_x=0'][k-1] * ans_all[ind[i][0][k]] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + np.sum([ch_parja[ju_geg[i][0]]['C12c_x=0'][k-1,n-1] * ans_all[ind[i][0][n]] for n in range(1,self.M)]) * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + ch_parja[ju_geg[i][0]]['C12g_x=0'][k-1] * ans_all[ind[i][0][0]] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1])
                        
                        jac_all[ind[2][0][k],ind[i][0][k]] = ch_parja[ju_geg[i][0]]['C12a_x=0'] * 3/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + ch_parja[ju_geg[i][0]]['C12b_x=0'][k-1] * ( ans_all[ind[i][2][0]]-4*ans_all[ind[i][1][0]]+3*ans_all[ind[i][0][0]] )/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + ch_parja[ju_geg[i][0]]['C12d_x=0'][k-1]*self.ch_pars[ju_geg[i][0]]['Q']
                        jac_all[ind[2][0][k],ind[i][1][k]] = ch_parja[ju_geg[i][0]]['C12a_x=0'] *-4/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) 
                        jac_all[ind[2][0][k],ind[i][2][k]] = ch_parja[ju_geg[i][0]]['C12a_x=0'] * 1/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) 
                        
                        for k2 in range(1,self.M): jac_all[ind[2][0][k],ind[i][0][k2]] = jac_all[ind[2][0][k],ind[i][0][k2]] + ch_parja[ju_geg[i][0]]['C12c_x=0'][k-1,k2-1] * ( ans_all[ind[i][2][0]]-4*ans_all[ind[i][1][0]]+3*ans_all[ind[i][0][0]] )/(2*self.ch_pars[ju_geg[i][0]]['dl'][-1]) + ch_parja[ju_geg[i][0]]['C12e_x=0'][k-1,k2-1]*self.ch_pars[ju_geg[i][0]]['Q']
                        
                    else: print("ERROR")
        
        return jac_all
    
    # =============================================================================
    # initialisation - create a vector with all zeros. 
    # =============================================================================

    if init_all[0] == None:
        ch_init = {}
        init_all = np.array([])
        for key in self.ch_keys: ch_init[key] = np.zeros(self.ch_pars[key]['di'][-1]*self.M) 
        for key in self.ch_keys: init_all= np.concatenate([init_all,ch_init[key]])
            
    # =============================================================================
    # solve the set of equations with the Newton-Raphson alghoritm
    # =============================================================================
    solu = sol_complete(init_all)
    jaco = jac_complete(init_all)
    
    #do the first time step
    sss_n =init_all - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)
    
    t=1
    print('That was iteration step ', t)
    sss=init_all
    # do the rest of the iterations
    while np.max(np.abs(sss-sss_n))>10e-6: #check whether the algoritm has converged
        sss = sss_n.copy() #update
        solu = sol_complete(sss)
        jaco = jac_complete(sss)
        
        #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,0])
        #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,1])
        #plt.plot(sss.reshape((int(len(sss)/self.M),self.M))[:,2])
        #plt.show()

        sss_n =sss - sp.sparse.linalg.spsolve(sp.sparse.csc_matrix(jaco),solu)  # this is faster then np.linalg.solve (at least for a sufficiently large matrix)
        
        t=1+t
        print('That was iteration step ', t)
        
        #if more than 20 iteration steps, we will not find a solution in this manner probably
        if t>=20: break
    
    if t<20:
        print('The algoritm has converged \n')  
    else:
        print('No convergence! Do the tricks \n')
        return [None]
    
    # =============================================================================
    #     return output
    # =============================================================================
    
    return sss 



def run_model_nt(self, init_all):
    # =============================================================================
    # code to run the subtidal salintiy model
    # =============================================================================
    print('Start the subtidal salinity calculation')
    tijd = time.time()

    #lets try first to directly spin up from scratch. 
    sss_u = saltmodel_ti_nt(self, [None])
    
    if sss_u[0] == None: #check if this calculation worked. If not, we will try again with a higher horizontal diffusivity
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
            sss_u = saltmodel_ti_nt(self, sss_u)
            
            if sss_u[0] == None: #if this also not works, stop the calculation
                import sys
                sys.exit("ABORT CALCULATION: Also with increased Kh no answer has been found. Check your input and think about \
                         if the model has a physical solution. If you think it has, you might wanna try increasing Kf_start or n_guess")   
       
            print('Step ', sim, ' of ',n_guess-1,' is finished')

    print('Subtidal salinty calculation completed, this took  '+ str(time.time()-tijd)+' seconds \n')


    return sss_u



