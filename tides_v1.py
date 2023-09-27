# =============================================================================
# Code to calculate everything about the tides in a general channel network. 
# So water level, currents, salinity, contribution to subtidal balance, fluxes, boundary layer
# A bit slow, if we have a good idea how to solve that we should implement it. 
# There should be options to make this faster, because some calculations are repeated quite a few times. 
# =============================================================================
#libraries
import numpy as np
import matplotlib.pyplot as plt  
import time                              #measure time of operation

#the functions to calculate the tidal salinity
from tide_funcs1 import sti,dsti_dc2,dsti_dc3,dsti_dc4 ,dsti_dz,dsti_dz_dc2,dsti_dz_dc3,dsti_dz_dc4

def tide_calc(self):
    # =============================================================================
    # This function calculates tidal properties, and, based on that, defines functions
    # to calculate tidal salinty and such, given a subtidal salinty
    # =============================================================================
    
    #add properties which will  be used for the tides
    for key in self.ch_keys:  
        # =============================================================================
        #   add physical quantities
        # =============================================================================
        #We have two possibilities for Av and bottom fricition:
        #Av depends on depth or does not depend on depth
        #we specify sf or we specify r 
        
        #calculate Av       
        if type(self.cv_t) == float or type(self.cv_t) == np.float64: self.ch_pars[key]['Av_t'] = self.cv_t*self.ch_gegs[key]['H']
        elif type(self.Av_t) == float or type(self.Av_t) == np.float64: self.ch_pars[key]['Av_t'] = self.Av_t
        else: print('ERROR, Av_t not well specified')        
        
        #Calcualte Kv
        self.ch_pars[key]['Kv_t'] = self.Kv_t #cv_t*ch_gegs[key]['Ut']*ch_gegs[key]['H']/Sc
        
        #Calculate r
        if type(self.sf_t) == float or type(self.sf_t) == np.float64: self.ch_pars[key]['r_t'] = (self.ch_pars[key]['Av_t']/(self.sf_t*self.ch_gegs[key]['H']))
        elif type(self.r_t) == float or type(self.r_t) == np.float64: self.ch_pars[key]['r_t'] = self.r_t
        else: print('ERROR, sf_t well specified')        

        #parameters for equations - see Wang et al. (2021) for an explanation what they are
        self.ch_pars[key]['deA'] = (1+1j)*self.ch_gegs[key]['H']/np.sqrt(2*self.ch_pars[key]['Av_t']/self.omega)
        self.ch_pars[key]['deK'] = (1+1j)*self.ch_gegs[key]['H']/np.sqrt(2*self.ch_pars[key]['Kv_t']/self.omega)
        
        self.ch_pars[key]['B'] = (np.cosh(self.ch_pars[key]['deA']) +  self.ch_pars[key]['r_t'] * self.ch_pars[key]['deA'] * np.sinh(self.ch_pars[key]['deA']))**-1
        self.ch_pars[key]['ka'] = np.sqrt(1/4*self.ch_pars[key]['bn']**-2 + self.omega**2/(self.g*self.ch_gegs[key]['H']) * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1)**-1)

    # =============================================================================
    # build matrix equation to calculate tdal water levels in each channel
    # basically solving analytical system of equations Ax = b
    # equations look a bit weird because of the change from x to x' late in the process. 
    # but it should be correct 
    # =============================================================================

    # =============================================================================
    # internal parts of channels, solution vectors are empty here, but the matrix A has values, due to segments
    # =============================================================================
    ch_matr  = {} 
    for key in self.ch_keys:  
        #create empty vector
        ch_matr[key] = np.zeros((2*(len(self.ch_pars[key]['nxn'])),2*(len(self.ch_pars[key]['nxn']))),dtype=complex)
        
        for dom in range(len(self.ch_pars[key]['nxn'])-1): #for every segment
            #water level equal
            ch_matr[key][dom*2+1, dom*2+0] = 1
            ch_matr[key][dom*2+1, dom*2+1] = 1
            ch_matr[key][dom*2+1, dom*2+2] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp(self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1])
            ch_matr[key][dom*2+1, dom*2+3] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp(-self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1])
            #discharge equal, i.e. water level gradient
            ch_matr[key][dom*2+2, dom*2+0] = -1/(2*self.ch_pars[key]['bn'][dom]) - self.ch_pars[key]['ka'][dom]
            ch_matr[key][dom*2+2, dom*2+1] = -1/(2*self.ch_pars[key]['bn'][dom]) + self.ch_pars[key]['ka'][dom]
            ch_matr[key][dom*2+2, dom*2+2] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp( self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1]) \
                * (-1/(2*self.ch_pars[key]['bn'][dom+1]) - self.ch_pars[key]['ka'][dom+1])
            ch_matr[key][dom*2+2, dom*2+3] = -np.exp(self.ch_gegs[key]['L'][dom+1]/(2*self.ch_pars[key]['bn'][dom+1])) * np.exp(-self.ch_gegs[key]['L'][dom+1]*self.ch_pars[key]['ka'][dom+1]) \
                * (-1/(2*self.ch_pars[key]['bn'][dom+1]) + self.ch_pars[key]['ka'][dom+1])

        #remove sea domain
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : 
            ch_matr[key] = ch_matr[key][2:,2:]
            ch_matr[key][0] = 0
        if self.ch_gegs[key]['loc x=0'][0] == 's' : 
            ch_matr[key] = ch_matr[key][:-2,:-2]
            ch_matr[key][-1] = 0
        
    #now build complete matrices
    sol_tot = np.zeros(np.sum([len(ch_matr[key]) for key in self.ch_keys]),dtype=complex)
    matr_tot = np.zeros((np.sum([len(ch_matr[key]) for key in self.ch_keys]),np.sum([len(ch_matr[key]) for key in self.ch_keys])),dtype=complex)
    
    ind = 0
    count = 0
    ind_jun = np.zeros((self.n_j,1),dtype=int)
    
    for key in self.ch_keys:
        #first add matrix for inner part
        matr_tot[ind:ind+len(ch_matr[key]),ind:ind+len(ch_matr[key])] = ch_matr[key]

        #river boundaries
        if self.ch_gegs[key]['loc x=-L'][0] == 'r' :
            matr_tot[ind,ind] =   np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp( self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) - self.ch_pars[key]['ka'][0])
            matr_tot[ind,ind+1] = np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp(-self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) + self.ch_pars[key]['ka'][0])        
        
        #weir and har boundaries
        if self.ch_gegs[key]['loc x=-L'][0] == 'w' or self.ch_gegs[key]['loc x=-L'][0] == 'h': 
            matr_tot[ind,ind] =   np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp( self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) - self.ch_pars[key]['ka'][0])
            matr_tot[ind,ind+1] = np.exp(self.ch_gegs[key]['L'][0]/(2*self.ch_pars[key]['bn'][0])) * np.exp(-self.ch_gegs[key]['L'][0]*self.ch_pars[key]['ka'][0]) * (-1/(2*self.ch_pars[key]['bn'][0]) + self.ch_pars[key]['ka'][0])        
        
        #sea boundary
        if self.ch_gegs[key]['loc x=-L'][0] == 's': 
            matr_tot[ind,ind] = np.exp(self.ch_gegs[key]['L'][1]/(2*self.ch_pars[key]['bn'][1])) * np.exp( self.ch_gegs[key]['L'][1]*self.ch_pars[key]['ka'][1])
            matr_tot[ind,ind+1] = np.exp(self.ch_gegs[key]['L'][1]/(2*self.ch_pars[key]['bn'][1])) * np.exp(-self.ch_gegs[key]['L'][1]*self.ch_pars[key]['ka'][1]) 

            sol_tot[ind] = self.a_tide[count] * np.exp(-1j*self.p_tide[count])
            count+=1
        
        #update ind, and go to the x=0 boundary
        ind += len(ch_matr[key])
        
        #river
        if self.ch_gegs[key]['loc x=0'][0] == 'r' :
            matr_tot[ind-1,ind-2] = -1/(2*self.ch_pars[key]['bn'][-1]) - self.ch_pars[key]['ka'][-1]
            matr_tot[ind-1,ind-1] = -1/(2*self.ch_pars[key]['bn'][-1]) + self.ch_pars[key]['ka'][-1]
        
        #weir and har boundaries
        if self.ch_gegs[key]['loc x=0'][0] == 'w' or self.ch_gegs[key]['loc x=0'][0] == 'h': 
            matr_tot[ind-1,ind-2] = -1/(2*self.ch_pars[key]['bn'][-1]) - self.ch_pars[key]['ka'][-1]  
            matr_tot[ind-1,ind-1] = -1/(2*self.ch_pars[key]['bn'][-1]) + self.ch_pars[key]['ka'][-1]    
        
        #sea boundaries
        elif self.ch_gegs[key]['loc x=0'][0] == 's': 
            matr_tot[ind-1,ind-2] = 1
            matr_tot[ind-1,ind-1] = 1
            
            sol_tot[ind-1] = self.a_tide[count] * np.exp(-1j*self.p_tide[count])
            count+=1
    
    #finally add the junctions 
    #create a new index array
    ind = 0
    for key in self.ch_keys:
        ind2 = len(ch_matr[key])+ind
        self.ch_pars[key]['ind_wl'] = [ind,ind2-1]
        ind =ind2
           
    for j in range(self.n_j):
        #find the connections
        ju_geg = []
        for key in self.ch_keys: 
            if self.ch_gegs[key]['loc x=-L'] == 'j'+str(j+1): 
                ju_geg.append([key,'x=-L',self.ch_pars[key]['ind_wl'][0]])
            elif self.ch_gegs[key]['loc x=0'] == 'j'+str(j+1): 
                ju_geg.append([key,'x=0',self.ch_pars[key]['ind_wl'][1]])
        
        #three conditions: 
        #first:n1=n2
        if ju_geg[0][1] == 'x=-L':
            matr_tot[ju_geg[0][2],ju_geg[0][2]] = np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[0][0]]['L'][0]*self.ch_pars[ju_geg[0][0]]['ka'][0])
            matr_tot[ju_geg[0][2],ju_geg[0][2]+1] = np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[0][0]]['L'][0]*self.ch_pars[ju_geg[0][0]]['ka'][0]) 
        elif ju_geg[0][1] == 'x=0':
            matr_tot[ju_geg[0][2],ju_geg[0][2]-1] = 1
            matr_tot[ju_geg[0][2],ju_geg[0][2]] = 1
            
        if ju_geg[1][1] == 'x=-L':
            matr_tot[ju_geg[0][2],ju_geg[1][2]] = -np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0])
            matr_tot[ju_geg[0][2],ju_geg[1][2]+1] = -np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0]) 
        elif ju_geg[1][1] == 'x=0':
            matr_tot[ju_geg[0][2],ju_geg[1][2]-1] = -1
            matr_tot[ju_geg[0][2],ju_geg[1][2]] = -1
            
        
        #second: n2=n3
        if ju_geg[1][1] == 'x=-L':
            matr_tot[ju_geg[1][2],ju_geg[1][2]] = np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0])
            matr_tot[ju_geg[1][2],ju_geg[1][2]+1] = np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[1][0]]['L'][0]*self.ch_pars[ju_geg[1][0]]['ka'][0]) 
        elif ju_geg[1][1] == 'x=0':
            matr_tot[ju_geg[1][2],ju_geg[1][2]-1] = 1
            matr_tot[ju_geg[1][2],ju_geg[1][2]] = 1
            
        if ju_geg[2][1] == 'x=-L':
            matr_tot[ju_geg[1][2],ju_geg[2][2]] =  -np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0])) * np.exp( self.ch_gegs[ju_geg[2][0]]['L'][0]*self.ch_pars[ju_geg[2][0]]['ka'][0])
            matr_tot[ju_geg[1][2],ju_geg[2][2]+1] = -np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0])) * np.exp(-self.ch_gegs[ju_geg[2][0]]['L'][0]*self.ch_pars[ju_geg[2][0]]['ka'][0]) 
        elif ju_geg[2][1] == 'x=0':
            matr_tot[ju_geg[1][2],ju_geg[2][2]-1] = -1
            matr_tot[ju_geg[1][2],ju_geg[2][2]] = -1
            
        #third: sum Q = 0 - updated wrt previous version! now also valid for H not constant. 
        if ju_geg[0][1] == 'x=-L':
            matr_tot[ju_geg[2][2],ju_geg[0][2]] = self.ch_gegs[ju_geg[0][0]]['b'][0]*self.ch_gegs[ju_geg[0][0]]['H']*(-self.ch_pars[ju_geg[0][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][0]))\
                * np.exp(self.ch_pars[ju_geg[0][0]]['ka'][0]*self.ch_gegs[ju_geg[0][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[0][0]]['B']/self.ch_pars[ju_geg[0][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA']) - 1)
            matr_tot[ju_geg[2][2],ju_geg[0][2]+1] = self.ch_gegs[ju_geg[0][0]]['b'][0]*self.ch_gegs[ju_geg[0][0]]['H']*(self.ch_pars[ju_geg[0][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][0]))\
                * np.exp(-self.ch_pars[ju_geg[0][0]]['ka'][0]*self.ch_gegs[ju_geg[0][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[0][0]]['L'][0]/(2*self.ch_pars[ju_geg[0][0]]['bn'][0]))      \
                * (self.ch_pars[ju_geg[0][0]]['B']/self.ch_pars[ju_geg[0][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA']) - 1)     
        elif ju_geg[0][1] == 'x=0':
            matr_tot[ju_geg[2][2],ju_geg[0][2]-1] =-self.ch_gegs[ju_geg[0][0]]['b'][-1]*self.ch_gegs[ju_geg[0][0]]['H']*(-self.ch_pars[ju_geg[0][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][-1])) \
                * (self.ch_pars[ju_geg[0][0]]['B']/self.ch_pars[ju_geg[0][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA']) - 1)
            matr_tot[ju_geg[2][2],ju_geg[0][2]] = - self.ch_gegs[ju_geg[0][0]]['b'][-1]*self.ch_gegs[ju_geg[0][0]]['H']*(self.ch_pars[ju_geg[0][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[0][0]]['bn'][-1]))  \
                * (self.ch_pars[ju_geg[0][0]]['B']/self.ch_pars[ju_geg[0][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[0][0]]['deA']) - 1)
        
        if ju_geg[1][1] == 'x=-L':
            matr_tot[ju_geg[2][2],ju_geg[1][2]] = self.ch_gegs[ju_geg[1][0]]['b'][0]*self.ch_gegs[ju_geg[1][0]]['H']*(-self.ch_pars[ju_geg[1][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][0]))\
                * np.exp(self.ch_pars[ju_geg[1][0]]['ka'][0]*self.ch_gegs[ju_geg[1][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[1][0]]['B']/self.ch_pars[ju_geg[1][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA']) - 1)
            matr_tot[ju_geg[2][2],ju_geg[1][2]+1] = self.ch_gegs[ju_geg[1][0]]['b'][0]*self.ch_gegs[ju_geg[1][0]]['H']*(self.ch_pars[ju_geg[1][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][0]))\
                * np.exp(-self.ch_pars[ju_geg[1][0]]['ka'][0]*self.ch_gegs[ju_geg[1][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[1][0]]['L'][0]/(2*self.ch_pars[ju_geg[1][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[1][0]]['B']/self.ch_pars[ju_geg[1][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA']) - 1)  
        elif ju_geg[1][1] == 'x=0':
            matr_tot[ju_geg[2][2],ju_geg[1][2]-1] = -self.ch_gegs[ju_geg[1][0]]['b'][-1]*self.ch_gegs[ju_geg[1][0]]['H']*(-self.ch_pars[ju_geg[1][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][-1])) \
                * (self.ch_pars[ju_geg[1][0]]['B']/self.ch_pars[ju_geg[1][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA']) - 1)
            matr_tot[ju_geg[2][2],ju_geg[1][2]] = - self.ch_gegs[ju_geg[1][0]]['b'][-1]*self.ch_gegs[ju_geg[1][0]]['H']*(self.ch_pars[ju_geg[1][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[1][0]]['bn'][-1])) \
                * (self.ch_pars[ju_geg[1][0]]['B']/self.ch_pars[ju_geg[1][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[1][0]]['deA']) - 1)
            
        if ju_geg[2][1] == 'x=-L':
            matr_tot[ju_geg[2][2],ju_geg[2][2]] = self.ch_gegs[ju_geg[2][0]]['b'][0]*self.ch_gegs[ju_geg[2][0]]['H']*(-self.ch_pars[ju_geg[2][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][0]))\
                * np.exp(self.ch_pars[ju_geg[2][0]]['ka'][0]*self.ch_gegs[ju_geg[2][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0])) \
                * (self.ch_pars[ju_geg[2][0]]['B']/self.ch_pars[ju_geg[2][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA']) - 1)
            matr_tot[ju_geg[2][2],ju_geg[2][2]+1] = self.ch_gegs[ju_geg[2][0]]['b'][0]*self.ch_gegs[ju_geg[2][0]]['H']*(self.ch_pars[ju_geg[2][0]]['ka'][0]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][0]))\
                * np.exp(-self.ch_pars[ju_geg[2][0]]['ka'][0]*self.ch_gegs[ju_geg[2][0]]['L'][0])*np.exp(self.ch_gegs[ju_geg[2][0]]['L'][0]/(2*self.ch_pars[ju_geg[2][0]]['bn'][0]))  \
                * (self.ch_pars[ju_geg[2][0]]['B']/self.ch_pars[ju_geg[2][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA']) - 1)
        elif ju_geg[2][1] == 'x=0':
            matr_tot[ju_geg[2][2],ju_geg[2][2]-1] = -self.ch_gegs[ju_geg[2][0]]['b'][-1]*self.ch_gegs[ju_geg[2][0]]['H']*(-self.ch_pars[ju_geg[2][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][-1]))\
                * (self.ch_pars[ju_geg[2][0]]['B']/self.ch_pars[ju_geg[2][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA']) - 1)
            matr_tot[ju_geg[2][2],ju_geg[2][2]] = - self.ch_gegs[ju_geg[2][0]]['b'][-1]*self.ch_gegs[ju_geg[2][0]]['H']*(self.ch_pars[ju_geg[2][0]]['ka'][-1]-1/(2*self.ch_pars[ju_geg[2][0]]['bn'][-1]))\
                * (self.ch_pars[ju_geg[2][0]]['B']/self.ch_pars[ju_geg[2][0]]['deA'] * np.sinh(self.ch_pars[ju_geg[2][0]]['deA']) - 1)



    #solve the matrix equation
    oplossing = np.linalg.solve(matr_tot,sol_tot)

    # =============================================================================
    # Build eta from the raw output 
    # =============================================================================
    nn = np.arange(1,self.N+1)[:,np.newaxis,np.newaxis] 
    z_nd = np.linspace(-1,0,self.nz)[np.newaxis,np.newaxis,:]
    
    ind = 0
    for key in self.ch_keys:
        #prepare some indices
        di_here = self.ch_pars[key]['di']
        ind2 = len(ch_matr[key])+ind
        coef_eta= oplossing[ind:ind2].reshape((int(len(ch_matr[key])/2),2))
        ind =ind2 #not sure what goes on here

        #save coefficients for single channel code
        self.ch_pars[key]['coef_eta'] = coef_eta

        #create empty vectors 
        eta = np.zeros(di_here[-1],dtype=complex)
        etar = np.zeros(di_here[-1],dtype=complex)
        detadx = np.zeros(di_here[-1],dtype=complex)
        detadx2 = np.zeros(di_here[-1],dtype=complex)
        detadx3 = np.zeros(di_here[-1],dtype=complex)

        i0,i1 = 0,0
        #remove sea domain
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : i0=1
        if self.ch_gegs[key]['loc x=0'][0] == 's' : i1=1
        
        #calculate eta and its derivatives. 
        for dom in range(i0,len(self.ch_pars[key]['nxn'])-i1):
            x_here = np.linspace(-self.ch_gegs[key]['L'][dom],0,self.ch_pars[key]['nxn'][dom])
            eta[di_here[dom]:di_here[dom+1]]  = np.exp(-x_here/(2*self.ch_pars[key]['bn'][dom])) * ( coef_eta[dom-i0,0]*np.exp(-x_here*self.ch_pars[key]['ka'][dom]) + coef_eta[dom-i0,1]*np.exp(x_here*self.ch_pars[key]['ka'][dom]) )
            etar[di_here[dom]:di_here[dom+1]] = np.exp(-x_here/(2*self.ch_pars[key]['bn'][dom])) * (-coef_eta[dom-i0,0]*np.exp(-x_here*self.ch_pars[key]['ka'][dom]) + coef_eta[dom-i0,1]*np.exp(x_here*self.ch_pars[key]['ka'][dom]) )
            
            detadx[di_here[dom]:di_here[dom+1]] = - eta[di_here[dom]:di_here[dom+1]]/(2*self.ch_pars[key]['bn'][dom]) \
                + self.ch_pars[key]['ka'][dom] * etar[di_here[dom]:di_here[dom+1]]
            
            detadx2[di_here[dom]:di_here[dom+1]] = eta[di_here[dom]:di_here[dom+1]]/(4*self.ch_pars[key]['bn'][dom]**2) \
                - self.ch_pars[key]['ka'][dom]/self.ch_pars[key]['bn'][dom] * etar[di_here[dom]:di_here[dom+1]] \
                + self.ch_pars[key]['ka'][dom]**2 * eta[di_here[dom]:di_here[dom+1]]
            
            detadx3[di_here[dom]:di_here[dom+1]] = etar[di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['ka'][dom]**3 \
                                                    - eta[di_here[dom]:di_here[dom+1]]/(8*self.ch_pars[key]['bn'][dom]**3) \
                                                    + 3*etar[di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['ka'][dom]/(4*self.ch_pars[key]['bn'][dom]**2) \
                                                    - 3*eta[di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['ka'][dom]**2/(2*self.ch_pars[key]['bn'][dom])
       
        #fill sea domain with nans
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : 
            eta[:di_here[1]] = np.nan
            detadx[:di_here[1]] = np.nan
            detadx2[:di_here[1]] = np.nan
            detadx3[:di_here[1]] = np.nan
        if self.ch_gegs[key]['loc x=0'][0] == 's': 
            eta[di_here[-2]:] = np.nan
            detadx[di_here[-2]:] = np.nan
            detadx2[di_here[-2]:] = np.nan
            detadx3[di_here[-2]:] = np.nan
        
        #save in the right format
        self.ch_pars[key]['eta'] = eta[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['detadx'] = detadx[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['detadx2'] = detadx2[np.newaxis,:,np.newaxis]
        self.ch_pars[key]['detadx3'] = detadx3[np.newaxis,:,np.newaxis]
        
        
        #calculate also long-channel velocity
        self.ch_pars[key]['ut'] = self.g/(1j*self.omega) * self.ch_pars[key]['detadx']  * (self.ch_pars[key]['B']*np.cosh(self.ch_pars[key]['deA']*z_nd) - 1)
        self.ch_pars[key]['dutdx'] = self.g/(1j*self.omega) * self.ch_pars[key]['detadx2']  * (self.ch_pars[key]['B']*np.cosh(self.ch_pars[key]['deA']*z_nd) - 1)
        self.ch_pars[key]['wt'] = 1j*self.omega*self.ch_pars[key]['eta'] - self.g/(1j*self.omega) * (self.ch_pars[key]['detadx2'] + self.ch_pars[key]['detadx']/self.ch_pars[key]['bex'][np.newaxis,:,np.newaxis]) \
            * (self.ch_pars[key]['B']*self.ch_gegs[key]['H']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA']*z_nd) - z_nd*self.ch_gegs[key]['H'])        
        
    # =============================================================================
    # From here, I will define functions, to be used in the subtidal module, to calculate the effect of the tides
    # =============================================================================
    
    def func_sol_int(key,sn,dsndx,dsbdx,dsbdx2):   
        # =============================================================================
        # Function to calculate the contribution of the tides to the subtidal depth-averaged salinity solution vector
        # =============================================================================
        #prepare: indices
        di_here = self.ch_pars[key]['di']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        
        # ===========================================================================
        # calculate salinity and salt derivative
        # ===========================================================================        
        #coefficients for salinity calculation
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            c2[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
            c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.g/(self.omega**2) \
                * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
        #coefficients for derivative
        dc2dx, dc3dx, dc4dx = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            dc2dx[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]*dsbdx[:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]*dsbdx2[:,di_here[dom]:di_here[dom+1]])
            dc3dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * (self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]*dsndx[:,di_here[dom]:di_here[dom+1]] + sn[:,di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]])
            dc4dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * self.g/(self.omega**2) * (dsndx[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]) \
                                                                                                                 + sn[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]))
        #tidal salinity    
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))         
        #derivative of tidal salinity
        dstidx = (dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc2dx 
            + (dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc3dx).sum(0) 
            + (dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc4dx).sum(0))[0]
        # =============================================================================
        # Terms for in equation
        # =============================================================================
        dfdx = np.mean(-1/4 * np.real(self.ch_pars[key]['ut'][0]*np.conj(dstidx) + np.conj(self.ch_pars[key]['ut'][0])*dstidx + self.ch_pars[key]['dutdx'][0]*np.conj(st) + np.conj(self.ch_pars[key]['dutdx'][0])*st) , axis=1)
        flux = np.mean(-1/4 * np.real(self.ch_pars[key]['ut'][0]*np.conj(st) + np.conj(self.ch_pars[key]['ut'][0])*st) , axis=1) #part for dbdx
        tot = (dfdx + flux /self.ch_pars[key]['bex']) * self.Lsc/self.soc_sca
        
        #remove sea parts 
        if self.ch_gegs[key]['loc x=-L'][0] == 's':  tot[:di_here[1]] = 0
        if self.ch_gegs[key]['loc x=0'][0] == 's':   tot[di_here[-2]:] = 0
        
        return -tot
    
    
    def func_jac_int(key,sn,dsndx,dsbdx,dsbdx2):
        # =============================================================================
        # Contributions to Jacobian for the tidal salt in the internal part of the channels
        # =============================================================================
        #preparations
        di_here = self.ch_pars[key]['di']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        
        #calculate tidal salinity
        #coefficients
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            c2[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
            c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.g/(self.omega**2) \
                * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
        #tidal salinity
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))     

        # =============================================================================
        # Jacobian terms
        # 6 terms
        # =============================================================================
        
        #derivatives
        dsdc2 = dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc3 = dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc4 = dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        
        #create empty vectors
        dT_dsb1,dT_dsb0,dT_dsb_1 = np.zeros(di_here[-1]) , np.zeros(di_here[-1]) , np.zeros(di_here[-1])
        dT_dsn1,dT_dsn0,dT_dsn_1 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1]))
        
        #for every domain a calculation
        for dom in range(len(self.ch_pars[key]['nxn'])):
            #select parameters
            dx_here = self.ch_gegs[key]['dx'][dom]
            bn_here = self.ch_pars[key]['bn'][dom]
            nph = nn*np.pi/self.ch_gegs[key]['H']
            ut_here = self.ch_pars[key]['ut'][:,di_here[dom]:di_here[dom+1]]
            dutdx_here = self.ch_pars[key]['dutdx'][:,di_here[dom]:di_here[dom+1]]
            eta_here= self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            detadx_here= self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]
            detadx2_here= self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]
            detadx3_here= self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]]
            
            #calculate contributions
            dT_dsb1[di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real(dutdx_here * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here * 1/(2*dx_here) ) 
                                   + np.conj(dutdx_here) * dsdc2 * self.g/self.omega**2 * detadx_here * 1/(2*dx_here)
                                   + ut_here * np.conj(dsdc2 * self.g/self.omega**2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                                   + np.conj(ut_here)*(dsdc2 * self.g/self.omega**2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) 
                                   + ut_here/bn_here * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here * 1/(2*dx_here) ) 
                                   + np.conj(ut_here/bn_here) * dsdc2 * self.g/self.omega**2 * detadx_here * 1/(2*dx_here)
                                   ).mean(2)
            
            dT_dsb0[di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc2 * -2 * self.g/self.omega**2 * detadx_here/(dx_here**2)) 
                                   +np.conj(ut_here)*(dsdc2 * -2 * self.g/self.omega**2 * detadx_here/(dx_here**2)) ).mean(2) 
            
            dT_dsb_1[di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real(dutdx_here * np.conj( dsdc2 *-self.g/self.omega**2 * detadx_here * 1/(2*dx_here) ) 
                                                                + np.conj(dutdx_here)* dsdc2 *-self.g/self.omega**2 * detadx_here * 1/(2*dx_here)
                                                                + ut_here * np.conj(dsdc2 * self.g/self.omega**2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                                                                + np.conj(ut_here)*(dsdc2 * self.g/self.omega**2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) 
                                                                + ut_here/bn_here * np.conj( dsdc2 *-self.g/self.omega**2 * detadx_here * 1/(2*dx_here) ) 
                                                                + np.conj(ut_here/bn_here) * dsdc2 *-self.g/self.omega**2 * detadx_here * 1/(2*dx_here)
                                                                ).mean(2)
    
    
            dT_dsn1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc3 * -nph * eta_here/(2*dx_here) + dsdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) 
                                                          + np.conj(ut_here)*(dsdc3 * -nph * eta_here/(2*dx_here) + dsdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) ).mean(2)
            
            dT_dsn0[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real(  dutdx_here * np.conj( dsdc3 * - nph*eta_here + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here + detadx_here/bn_here) ) 
                                                        + np.conj(dutdx_here)*( dsdc3 * - nph*eta_here + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here + detadx_here/bn_here) ) 
                                                        + ut_here * np.conj(dsdc3 * (- nph * detadx_here) + dsdc4 * (-nph * self.g/self.omega**2 * (detadx3_here + detadx2_here/bn_here)))
                                                        + np.conj(ut_here)*(dsdc3 * (- nph * detadx_here) + dsdc4 * (-nph * self.g/self.omega**2 * (detadx3_here + detadx2_here/bn_here)))
                                                        + ut_here/bn_here * np.conj( dsdc3 * - nph*eta_here + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here + detadx_here/bn_here) ) 
                                                        + np.conj(ut_here)/bn_here*( dsdc3 * - nph*eta_here + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here + detadx_here/bn_here) ) 
                                                        ).mean(2)
            
            dT_dsn_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc * np.real( ut_here * np.conj(dsdc3 * nph * eta_here/(2*dx_here) + dsdc4 * nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) 
                                     + np.conj(ut_here)*(dsdc3 * nph * eta_here/(2*dx_here) + dsdc4 * nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/bn_here)) ).mean(2)
        
        
        #remove sea parts 
        if self.ch_gegs[key]['loc x=-L'][0] == 's': 
            dT_dsb_1[:di_here[1]] = 0
            dT_dsb0[:di_here[1]] = 0
            dT_dsb1[:di_here[1]] = 0
            dT_dsn_1[:,:di_here[1]] = 0
            dT_dsn0[:,:di_here[1]] = 0
            dT_dsn1[:,:di_here[1]] = 0
            
        if self.ch_gegs[key]['loc x=0'][0] == 's': 
            dT_dsb_1[di_here[-2]:] = 0
            dT_dsb0[di_here[-2]:] = 0
            dT_dsb1[di_here[-2]:] = 0
            dT_dsn_1[:,di_here[-2]:] = 0
            dT_dsn0[:,di_here[-2]:] = 0
            dT_dsn1[:,di_here[-2]:] = 0

        #I added here thwat we divide by soc. this is different as the single channel case. Needs a check maybe....
        #the sb are checked and correct, the sn are also checked but not understood why the minus sign change 

        return -dT_dsb_1,-dT_dsb0,-dT_dsb1 , -dT_dsn_1.T,-dT_dsn0.T,-dT_dsn1.T


    def func_solz_int(key,sn,dsndx,dsbdx,dsbdx2): 
        # =============================================================================
        # Function to calculate the contribution of the tide to the subtidal stratification
        # =============================================================================
        di_here = self.ch_pars[key]['di']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        # ===========================================================================
        # calculate salinity and salt derivative
        # ===========================================================================        
        #coefficients for salinity calculation
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            c2[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
            c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.g/(self.omega**2) \
                * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
        #coefficients for derivative
        dc2dx, dc3dx, dc4dx = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            dc2dx[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]*dsbdx[:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]*dsbdx2[:,di_here[dom]:di_here[dom+1]])
            dc3dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * (self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]*dsndx[:,di_here[dom]:di_here[dom+1]] + sn[:,di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]])
            dc4dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * self.g/(self.omega**2) * (dsndx[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]) \
                                                                                                                 + sn[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]))
        #tidal salinity    
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))         
        #derivative of tidal salinity
        dstidx = (dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc2dx 
            + (dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc3dx).sum(0) 
            + (dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc4dx).sum(0))[0]

        utb = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0]
        utp = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0]
        stb = np.mean(st,1)[:,np.newaxis] #kan in theorie analytisch
        stp = st-stb #kan in theorie analytisch
        
        dstibdx = np.mean(dstidx,1)[:,np.newaxis] #kan in theorie analytisch
        dstipdx = dstidx-dstibdx #kan in theorie analytisch
        dstdz = dsti_dz(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))

        term1 = self.Lsc/self.soc_sca * (-1/4*np.real(utp*np.conj(dstidx) + np.conj(utp)*dstidx)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)).mean(2)
        term2 = self.Lsc/self.soc_sca * (-1/4*np.real(utb*np.conj(dstipdx) + np.conj(utb)*dstipdx)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)).mean(2)
        term3 = self.Lsc/self.soc_sca * (-1/4*np.real(self.ch_pars[key]['wt'] * np.conj(dstdz) + np.conj(self.ch_pars[key]['wt']) * dstdz) * np.cos(nn*np.pi*z_nd)).mean(2)

        #remove sea parts 
        if self.ch_gegs[key]['loc x=-L'][0] == 's':  
            term1[:,:di_here[1]] = 0
            term2[:,:di_here[1]] = 0
            term3[:,:di_here[1]] = 0
        if self.ch_gegs[key]['loc x=0'][0] == 's':   
            term1[:,di_here[-2]:] = 0
            term2[:,di_here[-2]:] = 0
            term3[:,di_here[-2]:] = 0

        #return term1*0, term2, term3
        return term1, term2, term3

    def func_jacz_int(key,sn,dsndx,dsbdx,dsbdx2): 
        # =============================================================================
        # Associated Jacobian for the contribution of the tide to the subtidal stratification
        # =============================================================================
        nph = nn*np.pi/self.ch_gegs[key]['H']
        di_here = self.ch_pars[key]['di']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        
        # ===========================================================================
        # calculate salinity and salt derivative
        # ===========================================================================        
        #coefficients for salinity calculation
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            c2[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
            c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.g/(self.omega**2) \
                * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
        #coefficients for derivative
        dc2dx, dc3dx, dc4dx = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            dc2dx[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]*dsbdx[:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]*dsbdx2[:,di_here[dom]:di_here[dom+1]])
            dc3dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * (self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]*dsndx[:,di_here[dom]:di_here[dom+1]] + sn[:,di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]])
            dc4dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * self.g/(self.omega**2) * (dsndx[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]) \
                                                                                                                 + sn[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]))
        #tidal salinity    
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))         
        #derivative of tidal salinity
        dstidx = (dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc2dx 
            + (dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc3dx).sum(0) 
            + (dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc4dx).sum(0))[0]
    

        utb = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0]
        utp = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0]
        stb = np.mean(st,1)[:,np.newaxis] #kan in theorie analytisch
        stp = st-stb #kan in theorie analytisch

        dstibdx = np.mean(dstidx,1)[:,np.newaxis] #kan in theorie analytisch
        dstipdx = dstidx-dstibdx #kan in theorie analytisch
    
        dstdz = dsti_dz(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        
        # =============================================================================
        # the terms in the NR algoritm
        # =============================================================================        
        dsdc2 = dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc3 = dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc4 = dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        
        dspdc2 = dsdc2 - dsdc2.mean(2)[:,:,np.newaxis]
        dspdc3 = dsdc3 - dsdc3.mean(2)[:,:,np.newaxis]
        dspdc4 = dsdc4 - dsdc4.mean(2)[:,:,np.newaxis]
        
        dstdz_dc2 = dsti_dz_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dstdz_dc3 = dsti_dz_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dstdz_dc4 = dsti_dz_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        
        dt1_dsb1,dt1_dsb0,dt1_dsb_1 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) 
        dt1_dsn1,dt1_dsn0,dt1_dsn_1 = np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1]))
        dt2_dsb1,dt2_dsb0,dt2_dsb_1 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) 
        dt2_dsn1,dt2_dsn0,dt2_dsn_1 = np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1]))
        dt3_dsb1,dt3_dsb_1,dt3_dsn0 = np.zeros((self.N,di_here[-1])) , np.zeros((self.N,di_here[-1])) , np.zeros((self.N,self.N,di_here[-1])) 
                        
        for dom in range(len(self.ch_pars[key]['nxn'])-1):     
            utp_here     = utp[np.newaxis,di_here[dom]:di_here[dom+1]]
            utb_here     = utb[np.newaxis,di_here[dom]:di_here[dom+1]]
            wt_here      = self.ch_pars[key]['wt'][:,di_here[dom]:di_here[dom+1]]
            eta_here     = self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            detadx_here  = self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]
            detadx2_here = self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]
            detadx3_here = self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]]
            dx_here      = self.ch_gegs[key]['dx'][dom]

            #term 1
            dt1_dsb1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real(utp_here * np.conj( dsdc2 * self.g/self.omega**2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)) ) 
                                   + np.conj(utp_here) * dsdc2 * self.g/self.omega**2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
            
            dt1_dsb0[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real( utp_here * np.conj(dsdc2 * -2 * self.g/self.omega**2 * detadx_here/(dx_here**2)) 
                                   + np.conj(utp_here) * (dsdc2 * -2 * self.g/self.omega**2 * detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
        
            dt1_dsb_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real(utp_here * np.conj(dsdc2 * self.g/self.omega**2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                                    + np.conj(utp_here) * dsdc2 * self.g/self.omega**2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
                                                               
            dt1_dsn1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real( utp_here * np.conj(dsdc3 * -nph * eta_here/(2*dx_here) + dsdc4 * - nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                    + np.conj(utp_here)*(dsdc3 * - nph * eta_here/(2*dx_here) + dsdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])))[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1)
        
            dt1_dsn0[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real(    utp_here * np.conj(dsdc3 * (- nph * detadx_here) + dsdc4 * (-nph * self.g/self.omega**2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom])))
                                                                      + np.conj(utp_here) * (dsdc3 * (- nph * detadx_here) + dsdc4 * (-nph * self.g/self.omega**2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom]))))[:,np.newaxis,:,:] * np.cos(nph*zlist)).mean(-1)
        
            dt1_dsn_1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real( utp_here * np.conj(dsdc3 * -nph * eta_here/(2*dx_here) + dsdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                    + np.conj(utp_here)*(dsdc3 * - nph * eta_here/(2*dx_here) + dsdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) )[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1) *-1 
            #term 2
            dt2_dsb1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real(utb_here * np.conj( dspdc2 * self.g/self.omega**2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)) ) 
                                   + np.conj(utb_here) * dspdc2 * self.g/self.omega**2 * (detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
            
            dt2_dsb0[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real( utb_here * np.conj(dspdc2 * -2 * self.g/self.omega**2 * detadx_here/(dx_here**2)) 
                                   + np.conj(utb_here) * (dspdc2 * -2 * self.g/self.omega**2 * detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
        
            dt2_dsb_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real(utb_here * np.conj(dspdc2 * self.g/self.omega**2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2)))
                                    + np.conj(utb_here) * dspdc2 * self.g/self.omega**2 * (-detadx2_here/(2*dx_here) + detadx_here/(dx_here**2))) * np.cos(nph*zlist)).mean(2)
                                                               
            dt2_dsn1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real( utb_here * np.conj(dspdc3 * -nph * eta_here/(2*dx_here) + dspdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                    + np.conj(utb_here)*(dspdc3 * -nph * eta_here/(2*dx_here) + dspdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])))[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1)
        
            dt2_dsn0[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real(    utb_here * np.conj(dspdc3 * (- nph * detadx_here) + dspdc4 * (-nph * self.g/self.omega**2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom])))
                                                                      + np.conj(utb_here) * (dspdc3 * (- nph * detadx_here) + dspdc4 * (-nph * self.g/self.omega**2 * (detadx3_here + detadx2_here/self.ch_pars[key]['bn'][dom]))))[:,np.newaxis,:,:] * np.cos(nph*zlist)).mean(-1)
        
            dt2_dsn_1[:,:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real( utb_here * np.conj(dspdc3 * -nph * eta_here/(2*dx_here) + dspdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) 
                                    + np.conj(utb_here)*(dspdc3 * -nph * eta_here/(2*dx_here) + dspdc4 * -nph * self.g/self.omega**2 /(2*dx_here) * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom])) )[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1) *-1 
            
            #term 3
            dt3_dsb1[:,di_here[dom]:di_here[dom+1]]  = -1/4*self.Lsc/self.soc_sca * (np.real(wt_here * np.conj(dstdz_dc2 * self.g/self.omega**2 * detadx_here * 1/(2*dx_here)) + np.conj(wt_here) * (dstdz_dc2 * self.g/self.omega**2 * detadx_here * 1/(2*dx_here)))* np.cos(nph*zlist) ).mean(-1)
            dt3_dsb_1[:,di_here[dom]:di_here[dom+1]] = -1/4*self.Lsc/self.soc_sca * (np.real(wt_here * np.conj(dstdz_dc2 * self.g/self.omega**2 * detadx_here *-1/(2*dx_here)) + np.conj(wt_here) * (dstdz_dc2 * self.g/self.omega**2 * detadx_here *-1/(2*dx_here)))* np.cos(nph*zlist) ).mean(-1)
            dt3_dsn0[:,:,di_here[dom]:di_here[dom+1]]= -1/4*self.Lsc/self.soc_sca * (np.real(wt_here * np.conj(- nph * eta_here * dstdz_dc3 - nph * self.g/self.omega**2 * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom]) * dstdz_dc4 ) \
                                            + np.conj(wt_here) * (- nph * eta_here * dstdz_dc3 - nph * self.g/self.omega**2 * (detadx2_here + detadx_here/self.ch_pars[key]['bn'][dom]) * dstdz_dc4   ))[:,np.newaxis,:,:] * np.cos(nph*zlist) ).mean(-1)
            

        
        return (dt1_dsb_1,dt1_dsb0,dt1_dsb1 , dt1_dsn_1,dt1_dsn0,dt1_dsn1) , (dt2_dsb_1,dt2_dsb0,dt2_dsb1 , dt2_dsn_1,dt2_dsn0,dt2_dsn1) , (dt3_dsb_1,dt3_dsb1,dt3_dsn0)
        #return (dt1_dsb_1*0,dt1_dsb0*0,dt1_dsb1*0 , dt1_dsn_1*0,dt1_dsn0*0,dt1_dsn1*0) , (dt2_dsb_1,dt2_dsb0,dt2_dsb1 , dt2_dsn_1,dt2_dsn0,dt2_dsn1) , (dt3_dsb_1,dt3_dsb1,dt3_dsn0)


    def func_sol_int_cor(key,bb,loc,dep): 
        # =============================================================================
        # Function to calculate the contribution of the boundary layer correction to the inner domain
        # =============================================================================
        #prepare
        Bm = bb[:self.M] + 1j*bb[self.M:]
        eps = self.Kh_tide/(self.omega)

        if loc == 'loc x=-L':
            #calculate x 
            x_all = np.arange(self.ch_pars[key]['di'][1])*self.ch_gegs[key]['dx'][0] 
            x_bnd = x_all[np.where(x_all<(-np.sqrt(eps)*np.log(self.tol)))[0][1:]]
            #print(key, x_all, x_bnd)
            if x_bnd[-1] >= self.ch_gegs[key]['L'][0]: print('ERROR: boundary layer too large. Can be solved...')
            #calculate the salinity correctoin
            st_cor = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            st_cor_x = st_cor * -1/np.sqrt(eps)
            
            #calculate the associated flux. 
            if dep == 'da' : #depth - averaged
                #calculate tidal velocity
                ut_here   = self.ch_pars[key]['ut'][0,1:len(x_bnd)+1]
                ut_here_x = self.ch_pars[key]['dutdx'][0,1:len(x_bnd)+1]
                
                tot = -np.mean(-1/4 * np.real(ut_here*np.conj(st_cor_x) + np.conj(ut_here) * st_cor_x \
                                             + ut_here_x*np.conj(st_cor) + np.conj(ut_here_x)*st_cor \
                                             + (ut_here*np.conj(st_cor) + np.conj(ut_here) * st_cor)/self.ch_pars[key]['bex'][1:len(x_bnd)+1,np.newaxis])  , -1)* self.Lsc/self.soc_sca
            
            elif dep == 'dv': # at depth levels
                utb_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0][1:len(x_bnd)+1]
                utp_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0][1:len(x_bnd)+1]
                wt_here   = self.ch_pars[key]['wt'][0,1:len(x_bnd)+1]
                stp_cor_x = -1/np.sqrt(eps) * np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm[1:] * np.cos(np.arange(1,self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
                st_cor_z= np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * -np.arange(self.M)*np.pi/self.ch_gegs[key]['H'] * np.sin(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
                
                t1 = np.mean(-1/4 * np.real(utp_here*np.conj( st_cor_x) + np.conj(utp_here)* st_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
                t2 = np.mean(-1/4 * np.real(utb_here*np.conj(stp_cor_x) + np.conj(utb_here)*stp_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
                t3 = np.mean(-1/4 * np.real(wt_here*np.conj(st_cor_z  ) + np.conj(wt_here)*st_cor_z  )[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca

                tot= t1+t2+t3
            
            else: print('ERROR: invalid specification of dep')

        elif loc == 'loc x=0':
            #calculate x
            x_all = np.arange(self.ch_pars[key]['di'][-2]-self.ch_pars[key]['di'][-1]+1,1)*self.ch_gegs[key]['dx'][-1] 
            x_bnd = x_all[np.where(x_all>(np.sqrt(eps)*np.log(self.tol)))[0][:-1]]
            if -x_bnd[0] >= self.ch_gegs[key]['L'][-1]: print('ERROR: boundary layer too large. Can be solved...')
            #calculate the salinity correction
            st_cor = np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            st_cor_x = st_cor * 1/np.sqrt(eps)

            #calculate the associated flux. 
            if dep == 'da': #depth-averaged 
                #calculate the tidal velocity here
                ut_here = self.ch_pars[key]['ut'][0,-len(x_bnd)-1:-1]
                ut_here_x = self.ch_pars[key]['dutdx'][0,-len(x_bnd)-1:-1]
                
                tot = -np.mean(-1/4 * np.real(ut_here*np.conj(st_cor_x) + np.conj(ut_here) * st_cor_x \
                                             + ut_here_x*np.conj(st_cor) + np.conj(ut_here_x)*st_cor \
                                             + (ut_here*np.conj(st_cor) + np.conj(ut_here) * st_cor)/self.ch_pars[key]['bex'][-len(x_bnd)-1:-1,np.newaxis])  , -1)* self.Lsc/self.soc_sca
                    
            elif dep == 'dv':# at depth levels 
                utb_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0,-len(x_bnd)-1:-1]
                utp_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0,-len(x_bnd)-1:-1]
                wt_here   = self.ch_pars[key]['wt'][0,-len(x_bnd)-1:-1]
                stp_cor_x = 1/np.sqrt(eps) * np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm[1:] * np.cos(np.arange(1,self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
                st_cor_z= np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * -np.arange(self.M)*np.pi/self.ch_gegs[key]['H'] * np.sin(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
    
                t1 = np.mean(-1/4 * np.real(utp_here*np.conj( st_cor_x) + np.conj(utp_here)* st_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
                t2 = np.mean(-1/4 * np.real(utb_here*np.conj(stp_cor_x) + np.conj(utb_here)*stp_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
                t3 = np.mean(-1/4 * np.real(wt_here*np.conj(st_cor_z  ) + np.conj(wt_here)*st_cor_z  )[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
                
                tot = t1+t2+t3
            
            else: print('ERROR: invalid specification of dep')
                
        else:  print('ERROR!')
            
        return tot

    def func_jac_int_cor(key,loc, dep):   
        # =============================================================================
        # Function to calculate Jacobian associated with the contribution of the boundary layer correction to the inner domain
        # =============================================================================
        #prepare
        eps = self.Kh_tide/(self.omega)
        
        if loc == 'loc x=-L':
            #calculate x
            x_all = np.arange(self.ch_pars[key]['di'][1])*self.ch_gegs[key]['dx'][0] 
            x_bnd = x_all[np.where(x_all<(-np.sqrt(eps)*np.log(self.tol)))[0][1:]]
            if x_bnd[-1] > self.ch_gegs[key]['L'][0]: print('ERROR: boundary layer too large. Can be solved...')
            #calculate the derivative of st
            st_cor_der = np.exp(-x_bnd/np.sqrt(eps))[np.newaxis,:,np.newaxis] *  np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)[np.newaxis,np.newaxis,:]) #* self.soc_sca 
            #calculate local tidal velocity
            ut_here = self.ch_pars[key]['ut'][0][1:len(x_bnd)+1]
            ut_here_x = self.ch_pars[key]['dutdx'][0][1:len(x_bnd)+1]
            
            #terms for Jacobian
            if dep == 'da':#depth-averaged            
                dT_dBR = np.mean(-1/4 * (2*np.real(ut_here) * st_cor_der * -1/np.sqrt(eps) + 2*np.real(ut_here_x) * st_cor_der \
                                         + (2*np.real(ut_here) * st_cor_der)/self.ch_pars[key]['bex'][1:len(x_bnd)+1,np.newaxis])  , -1)* self.Lsc
                dT_dBI = np.mean(-1/4 * (2*np.imag(ut_here) * st_cor_der * -1/np.sqrt(eps) + 2*np.imag(ut_here_x) * st_cor_der \
                                         + (2*np.imag(ut_here) * st_cor_der)/self.ch_pars[key]['bex'][1:len(x_bnd)+1,np.newaxis])  , -1)* self.Lsc
            elif dep == 'dv':#depth levels 
                utb_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0][1:len(x_bnd)+1]
                utp_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0][1:len(x_bnd)+1]
                wt_here   = self.ch_pars[key]['wt'][0,1:len(x_bnd)+1]
                
                stp_cor_der = np.zeros(st_cor_der.shape)
                stp_cor_der[1:] = st_cor_der[1:]
                st_cor_z_der = np.exp(-x_bnd/np.sqrt(eps))[np.newaxis,:,np.newaxis] * -(np.arange(self.M)*np.pi/self.ch_gegs[key]['H'])[:,np.newaxis,np.newaxis] * np.sin(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)[np.newaxis,np.newaxis,:]) 
                       
                dt1_dBR = np.mean(-1/4 * (2*np.real(utp_here) * -1/np.sqrt(eps) *  st_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt2_dBR = np.mean(-1/4 * (2*np.real(utb_here) * -1/np.sqrt(eps) * stp_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt3_dBR = np.mean(-1/4 * (2*np.real(wt_here)  * st_cor_z_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc

                dt1_dBI = np.mean(-1/4 * (2*np.imag(utp_here) * -1/np.sqrt(eps) *  st_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt2_dBI = np.mean(-1/4 * (2*np.imag(utb_here) * -1/np.sqrt(eps) * stp_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt3_dBI = np.mean(-1/4 * (2*np.imag(wt_here)  * st_cor_z_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc

                dT_dBR = -dt1_dBR-dt2_dBR-dt3_dBR
                dT_dBI = -dt1_dBI-dt2_dBI-dt3_dBI
                                   
            else: print('ERROR: invalid specification of dep')
            
                    

        elif loc == 'loc x=0':
            #calculate x
            x_all = np.arange(self.ch_pars[key]['di'][-2]-self.ch_pars[key]['di'][-1]+1,1)*self.ch_gegs[key]['dx'][-1] 
            x_bnd = x_all[np.where(x_all>(np.sqrt(eps)*np.log(self.tol)))[0][:-1]]
            if -x_bnd[0] > self.ch_gegs[key]['L'][-1]: print('ERROR: boundary layer too large. Can be solved...')
            #calcualte the derivative of st
            st_cor_der = np.exp(x_bnd/np.sqrt(eps))[np.newaxis,:,np.newaxis] * np.cos(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)[np.newaxis,np.newaxis,:]) #* self.soc_sca 
            #calculate local tidal velocity
            ut_here = self.ch_pars[key]['ut'][0][-len(x_bnd)-1:-1]
            ut_here_x = self.ch_pars[key]['dutdx'][0][-len(x_bnd)-1:-1]
            
            #terms for Jacobian
            if dep == 'da': #depth-averaged 
                dT_dBR = np.mean(-1/4 * (2*np.real(ut_here) * st_cor_der * 1/np.sqrt(eps) + 2*np.real(ut_here_x) * st_cor_der \
                                         + (2*np.real(ut_here) * st_cor_der)/self.ch_pars[key]['bex'][-len(x_bnd)-1:-1,np.newaxis])  , -1)* self.Lsc
                dT_dBI = np.mean(-1/4 * (2*np.imag(ut_here) * st_cor_der * 1/np.sqrt(eps) + 2*np.imag(ut_here_x) * st_cor_der \
                                         + (2*np.imag(ut_here) * st_cor_der)/self.ch_pars[key]['bex'][-len(x_bnd)-1:-1,np.newaxis])  , -1)* self.Lsc
            elif dep == 'dv':#depth levels
                utb_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0,-len(x_bnd)-1:-1]
                utp_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0,-len(x_bnd)-1:-1]
                wt_here   = self.ch_pars[key]['wt'][0,-len(x_bnd)-1:-1]

                stp_cor_der = np.zeros(st_cor_der.shape)
                stp_cor_der[1:] = st_cor_der[1:]
                st_cor_z_der = np.exp(x_bnd/np.sqrt(eps))[np.newaxis,:,np.newaxis] * -(np.arange(self.M)*np.pi/self.ch_gegs[key]['H'])[:,np.newaxis,np.newaxis] * np.sin(np.arange(self.M)[:,np.newaxis,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)[np.newaxis,np.newaxis,:]) 

                dt1_dBR = np.mean(-1/4 * (2*np.real(utp_here) * 1/np.sqrt(eps) *  st_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt2_dBR = np.mean(-1/4 * (2*np.real(utb_here) * 1/np.sqrt(eps) * stp_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt3_dBR = np.mean(-1/4 * (2*np.real(wt_here)  * st_cor_z_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc

                dt1_dBI = np.mean(-1/4 * (2*np.imag(utp_here) * 1/np.sqrt(eps) *  st_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt2_dBI = np.mean(-1/4 * (2*np.imag(utb_here) * 1/np.sqrt(eps) * stp_cor_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc
                dt3_dBI = np.mean(-1/4 * (2*np.imag(wt_here)  * st_cor_z_der)[:,np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc

                dT_dBR = -dt1_dBR-dt2_dBR-dt3_dBR
                dT_dBI = -dt1_dBI-dt2_dBI-dt3_dBI
                                
            else: print('ERROR: invalid specification of dep')
                    
        return  -dT_dBR, -dT_dBI
         
    def func_sol_sea(key,sn,dsndx,dsbdx,dsbdx2):
        # =============================================================================
        # contribution to solution vector by tides in the sea domain, i.e. with the decaying dispersion
        # =============================================================================
        di_here = self.ch_pars[key]['di']
 
        #select local parameters
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : 
           eta_h = self.ch_pars[key]['eta'][0,0,0]
           detadx_h = self.ch_pars[key]['detadx'][0,0,0]
           detadx2_h = self.ch_pars[key]['detadx2'][0,0,0]
           detadx3_h = self.ch_pars[key]['detadx3'][0,0,0]
            
           dsbdx_h = dsbdx[0,self.ch_pars[key]['di'][1],0]
           sn_h = sn[:,self.ch_pars[key]['di'][1],:]       
            
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : 
           eta_h = self.ch_pars[key]['eta'][:,[di_here[-2]-1],:]
           detadx_h = self.ch_pars[key]['detadx'][:,[di_here[-2]-1],:]
           detadx2_h = self.ch_pars[key]['detadx2'][:,[di_here[-2]-1],:]
           detadx3_h = self.ch_pars[key]['detadx3'][:,[di_here[-2]-1],:]
             
           dsbdx_h = dsbdx[:,[di_here[-2]-1],:]
           sn_h = sn[:,[di_here[-2]-1],:]
             
        else: print('no sea boundary detected!')
 
        # =============================================================================
        #  calculate tidal velocity and salinity at the estuary-sea transition
        # =============================================================================
        nph = nn*np.pi/self.ch_gegs[key]['H']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        #tidal velocity
        ut = (self.g/(1j*self.omega) * detadx_h * (self.ch_pars[key]['B']*np.cosh(self.ch_pars[key]['deA']*z_nd[0]) - 1))[0]
        dutdx = (self.g/(1j*self.omega) * detadx2_h * (self.ch_pars[key]['B']*np.cosh(self.ch_pars[key]['deA']*z_nd[0]) - 1))[0]
        
        #coefficients for tidal salinity
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,1,self.nz),dtype=complex) , np.zeros((self.N,1,self.nz),dtype=complex) , np.zeros((self.N,1,self.nz),dtype=complex)
        c2[:,0] = self.g/(self.omega**2) * detadx_h * dsbdx_h
        c3[:,[0]] = - nph * sn_h * eta_h
        if self.ch_gegs[key]['loc x=-L'][0] == 's' :  c4[:,[0]] = - nph * sn_h * self.g/(self.omega**2) * (detadx2_h + detadx_h/self.ch_pars[key]['bn'][1])
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : c4[:,[0]] = - nph * sn_h * self.g/(self.omega**2) * (detadx2_h + detadx_h/self.ch_pars[key]['bn'][-2])                
            
        #tidal salinity
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))     
        
        # ===========================================================================
        # calculate salt flux and associated effective dispersion
        # =============================================================================
        flux = -1/4 * np.real(ut*np.conj(st) + np.conj(ut)*st).mean() 
        
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : print('not coded actually!')
            
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : 
            x_sea = np.linspace(0,self.ch_gegs[key]['L'][-1],self.ch_pars[key]['nxn'][-1])  #x coordinates in sea      
            Kh_tidesea, diffu = np.zeros(di_here[-1]) , np.zeros(di_here[-1]) #empty vectors
            Kh_tidesea[di_here[-2]:] = flux/dsbdx_h * np.exp(-x_sea/self.ch_pars[key]['bn'][-1]) #Kh in the sea domain due to tides
            diffu[di_here[-2]:] = flux/dsbdx_h * np.exp(-x_sea/self.ch_pars[key]['bn'][-1]) * dsbdx2[0,di_here[-2]:,0] * self.Lsc/self.soc_sca #contribution to solution vector
 
        return -diffu
    
    def func_jac_sea(key,sn,dsndx,dsbdx,dsbdx2):
        # =============================================================================
        # jacobin associated with contribution by tides in the sea domain, i.e. with the decaying dispersion
        # =============================================================================
        di_here = self.ch_pars[key]['di']

        #select local parameters
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : 
           eta_h = self.ch_pars[key]['eta'][0,0,0]
           detadx_h = self.ch_pars[key]['detadx'][0,0,0]
           detadx2_h = self.ch_pars[key]['detadx2'][0,0,0]
           detadx3_h = self.ch_pars[key]['detadx3'][0,0,0]
            
           dsbdx_h = dsbdx[0,self.ch_pars[key]['di'][1],0]
           sn_h = sn[:,self.ch_pars[key]['di'][1],:]       
            
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : 
           eta_h = self.ch_pars[key]['eta'][:,[di_here[-2]-1],:]
           detadx_h = self.ch_pars[key]['detadx'][:,[di_here[-2]-1],:]
           detadx2_h = self.ch_pars[key]['detadx2'][:,[di_here[-2]-1],:]
           detadx3_h = self.ch_pars[key]['detadx3'][:,[di_here[-2]-1],:]
             
           dsbdx_h = dsbdx[:,[di_here[-2]-1],:]
           sn_h = sn[:,[di_here[-2]-1],:]
             
        else: print('no sea boundary detected!')

        # =============================================================================
        # tidal velocity and salinity
        # =============================================================================
        nph = nn*np.pi/self.ch_gegs[key]['H']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        #tidal velocity
        ut = (self.g/(1j*self.omega) * detadx_h * (self.ch_pars[key]['B']*np.cosh(self.ch_pars[key]['deA']*z_nd[0]) - 1))
        dutdx = (self.g/(1j*self.omega) * detadx2_h * (self.ch_pars[key]['B']*np.cosh(self.ch_pars[key]['deA']*z_nd[0]) - 1))

        #coefficients
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,1,self.nz),dtype=complex) , np.zeros((self.N,1,self.nz),dtype=complex) , np.zeros((self.N,1,self.nz),dtype=complex)
        c2[:,0] = self.g/(self.omega**2) * detadx_h * dsbdx_h
        c3[:,[0]] = - nph * sn_h * eta_h
        if self.ch_gegs[key]['loc x=-L'][0] == 's' : 
            c4[:,[0]] = - nph * sn_h * self.g/(self.omega**2) * (detadx2_h + detadx_h/self.ch_pars[key]['bn'][1])
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : 
            c4[:,[0]] = - nph * sn_h * self.g/(self.omega**2) * (detadx2_h + detadx_h/self.ch_pars[key]['bn'][-2])                

        dsdc2 = dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc3 = dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc4 = dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        #tidal salinity
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))     
        
        # ===========================================================================
        # calculate salt flux and effective dispersion
        # =============================================================================
        flux = -1/4 * np.real(ut*np.conj(st) + np.conj(ut)*st).mean() 

        if self.ch_gegs[key]['loc x=-L'][0] == 's' :  print('not coded actually!')
    
        elif self.ch_gegs[key]['loc x=0'][0] == 's' : 
            #local x coordinate, in sea domain
            x_sea = np.linspace(0,self.ch_gegs[key]['L'][-1],self.ch_pars[key]['nxn'][-1])
            
            # =============================================================================
            # Terms for Jacobian
            # =============================================================================
            #derivatives to variables on the edge
            dKh_dsn = ( dsbdx_h**-1 * -1/4 * np.real(ut * np.conj(dsdc3 * - nph*eta_h + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_h + detadx_h/self.ch_pars[key]['bn'][-2]) )
                      + np.conj(ut)*(dsdc3 * - nph*eta_h + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_h + detadx_h/self.ch_pars[key]['bn'][-2]) )) * np.exp(-x_sea/self.ch_pars[key]['bn'][-1])[np.newaxis,:,np.newaxis]  ).mean(2)

            dKh_dsb_3 = ( (np.exp(-x_sea/self.ch_pars[key]['bn'][-1])[np.newaxis,:,np.newaxis] / dsbdx_h *  1/(2*self.ch_gegs[key]['dx'][-2])) 
                         * (- flux/dsbdx_h - 1/4 * np.real(ut * np.conj(dsdc2 * self.g/self.omega**2 * detadx_h) + np.conj(ut)*(dsdc2 * self.g/self.omega**2 * detadx_h)))).mean(2)[0]
            dKh_dsb_2 = ( (np.exp(-x_sea/self.ch_pars[key]['bn'][-1])[np.newaxis,:,np.newaxis] / dsbdx_h * -4/(2*self.ch_gegs[key]['dx'][-2])) 
                         * (- flux/dsbdx_h - 1/4 * np.real(ut * np.conj(dsdc2 * self.g/self.omega**2 * detadx_h) + np.conj(ut)*(dsdc2 * self.g/self.omega**2 * detadx_h)))).mean(2)[0]
            dKh_dsb_1 = ( (np.exp(-x_sea/self.ch_pars[key]['bn'][-1])[np.newaxis,:,np.newaxis] / dsbdx_h *  3/(2*self.ch_gegs[key]['dx'][-2])) 
                         * (- flux/dsbdx_h - 1/4 * np.real(ut * np.conj(dsdc2 * self.g/self.omega**2 * detadx_h) + np.conj(ut)*(dsdc2 * self.g/self.omega**2 * detadx_h)))).mean(2)[0]

            ddif_dsn   = dKh_dsn   * dsbdx2[0,di_here[-2]:,0] *-self.Lsc
            ddif_dsb_3 = dKh_dsb_3 * dsbdx2[0,di_here[-2]:,0] *-self.Lsc
            ddif_dsb_2 = dKh_dsb_2 * dsbdx2[0,di_here[-2]:,0] *-self.Lsc
            ddif_dsb_1 = dKh_dsb_1 * dsbdx2[0,di_here[-2]:,0] *-self.Lsc

            #derivatives to variables in the sea domain             
            ddif_dsb_s_1,ddif_dsb_s0,ddif_dsb_s1 = np.zeros(di_here[-1]),np.zeros(di_here[-1]),np.zeros(di_here[-1])            #this is easier with indices
            ddif_dsb_s_1[di_here[-2]:] = flux/dsbdx_h * np.exp(-x_sea/self.ch_pars[key]['bn'][-1]) * (1/(self.ch_gegs[key]['dx'][-1]**2)) *-self.Lsc
            ddif_dsb_s0[di_here[-2]:] = flux/dsbdx_h * np.exp(-x_sea/self.ch_pars[key]['bn'][-1]) * (-2/(self.ch_gegs[key]['dx'][-1]**2)) *-self.Lsc
            ddif_dsb_s1[di_here[-2]:] = flux/dsbdx_h * np.exp(-x_sea/self.ch_pars[key]['bn'][-1]) * (1/(self.ch_gegs[key]['dx'][-1]**2))  *-self.Lsc

            return ddif_dsn.T,ddif_dsb_3,ddif_dsb_2,ddif_dsb_1 , ddif_dsb_s_1,ddif_dsb_s0,ddif_dsb_s1
            

    def func_sol_bnd(key,sn,dsndx,dsbdx,dsbdx2):
        # =============================================================================
        # Transport at boundaries, to add to solution vector 
        # =============================================================================
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        nph = nn*np.pi/self.ch_gegs[key]['H']

        #coefficients
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2 = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,[0,-1]] * dsbdx[:,[0,-1]]
        c3 = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,[0,-1]] * self.ch_pars[key]['eta'][:,[0,-1]]
        c4 = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,[0,-1]] * self.g/(self.omega**2) * (self.ch_pars[key]['detadx2'][:,[0,-1]] + self.ch_pars[key]['detadx'][:,[0,-1]]/self.ch_pars[key]['bn'][np.newaxis,[0,-1],np.newaxis])
        #tidal salinity
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))  
        
        #transport
        flux = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * np.real(self.ch_pars[key]['ut'][:,[0,-1]]*np.conj(st) + np.conj(self.ch_pars[key]['ut'][:,[0,-1]])*st) , axis=2)[0] / self.soc_sca #part for dbdx       
        #transport at vertical levels
        flux_z = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][[0,-1]] * np.mean(1/4 * np.real(self.ch_pars[key]['ut'][:,[0,-1]]*np.conj(st) + np.conj(self.ch_pars[key]['ut'][:,[0,-1]])*st) * np.cos(nph*zlist) , axis=2) / self.soc_sca   

        return flux , flux_z 


    def func_jac_bnd(key,sn,dsndx,dsbdx,dsbdx2):
        # =============================================================================
        # Transport at boundaries, to add to jacobian 
        # =============================================================================
        #prepare
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        nph = nn*np.pi/self.ch_gegs[key]['H']
        #coefficients
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)

        dsdc2 = dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc3 = dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
        dsdc4 = dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))

        
        # =============================================================================
        # jacobian, derivatives, 8 terms
        # =============================================================================
        #local variables
        dx_here = self.ch_gegs[key]['dx'][[0,-1]]
        bn_here = self.ch_pars[key]['bn'][[0,-1]]
        ut_here = self.ch_pars[key]['ut'][:,[0,-1]]
        dutdx_here = self.ch_pars[key]['dutdx'][:,[0,-1]]
        eta_here= self.ch_pars[key]['eta'][:,[0,-1]]
        detadx_here= self.ch_pars[key]['detadx'][:,[0,-1]]
        detadx2_here= self.ch_pars[key]['detadx2'][:,[0,-1]]
        
        #depth-averaged
        #derivatives for x=-L
        dT0_dsb0 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -3/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -3/(2*dx_here[0]) ).mean(2)
        dT0_dsb1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,0] *  4/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,0] *  4/(2*dx_here[0]) ).mean(2)
        dT0_dsb2 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -1/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -1/(2*dx_here[0]) ).mean(2)
        dT0_dsn0 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.real( ut_here[:,0] * np.conj( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) 
                                                               + np.conj(ut_here[:,0]) * ( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) ).mean(2) 
        
        #derivatives for x=0
        dT_1_dsb_1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  3/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  3/(2*dx_here[1]) ).mean(2)
        dT_1_dsb_2 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,1] * -4/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,1] * -4/(2*dx_here[1]) ).mean(2)
        dT_1_dsb_3 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  1/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  1/(2*dx_here[1]) ).mean(2)
        dT_1_dsn_1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.real( ut_here[:,1] * np.conj( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) 
                                                               + np.conj(ut_here[:,1]) * ( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) ).mean(2) 
    
        #z levels
        #derivatives for x=-L
        dTz0_dsb0 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -3/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -3/(2*dx_here[0]) ) * np.cos(nph*zlist), axis=-1)
        dTz0_dsb1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,0] *  4/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,0] *  4/(2*dx_here[0]) ) * np.cos(nph*zlist), axis=-1)
        dTz0_dsb2 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -1/(2*dx_here[0]) ) + np.conj(ut_here[:,0]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -1/(2*dx_here[0]) ) * np.cos(nph*zlist), axis=-1)
        dTz0_dsn0 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][0] * np.mean(np.real( ut_here[:,0] * np.conj( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) 
                                                               + np.conj(ut_here[:,0]) * ( dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) )[:,np.newaxis,:,:] * np.cos(nph*zlist) , axis = -1)

        #derivatives for x=0
        dTz_1_dsb_1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  3/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  3/(2*dx_here[1]) ) * np.cos(nph*zlist) , axis=-1)
        dTz_1_dsb_2 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,1] * -4/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,1] * -4/(2*dx_here[1]) ) * np.cos(nph*zlist) , axis=-1)
        dTz_1_dsb_3 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  1/(2*dx_here[1]) ) + np.conj(ut_here[:,1]) * dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  1/(2*dx_here[1]) ) * np.cos(nph*zlist) , axis=-1)
        dTz_1_dsn_1 = 1/4* self.ch_gegs[key]['H']*self.ch_pars[key]['b'][-1] * np.mean(np.real( ut_here[:,1] * np.conj( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) 
                                                               + np.conj(ut_here[:,1]) * ( dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) )[:,np.newaxis,:,:] * np.cos(nph*zlist) , axis=-1)
            
        #again a different scaling, do not know where it comes from. 
        return (dT0_dsb0[0,0],dT0_dsb1[0,0],dT0_dsb2[0,0],dT0_dsn0[:,0] , dT_1_dsb_1[0,0],dT_1_dsb_2[0,0],dT_1_dsb_3[0,0],dT_1_dsn_1[:,0]) , (dTz0_dsb0[:,0],dTz0_dsb1[:,0],dTz0_dsb2[:,0],dTz0_dsn0[:,:,0] , dTz_1_dsb_1[:,0],dTz_1_dsb_2[:,0],dTz_1_dsb_3[:,0],dTz_1_dsn_1[:,:,0]) 
            
    
    
    def func_sol_bnd_cor(key,bb,loc,dep):       
        # =============================================================================
        # Solution vector for contribution to transport at boundaries due to boundary correction
        # =============================================================================
        if loc == 'loc x=0':xx = -1
        elif loc == 'loc x=-L': xx = 0    
        else: print('ERROR: no correct loc')
        
        #calculate salinity correction
        Bm = bb[:self.M] + 1j*bb[self.M:]
        st_cor = np.sum(Bm * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca # I think this is the correct scaling

        #select tidal velocity
        ut_here = self.ch_pars[key]['ut'][:,xx]
        
        if dep == 'da': #calculate associated depth-averaged transport
            flux = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][xx] * np.mean(-1/4 * np.real(ut_here*np.conj(st_cor) + np.conj(ut_here)*st_cor) , axis=1)[0] / self.soc_sca #waarschijnlijk kan dit analytisch
            return flux
        
        elif dep == 'dv': #calculate associated transport at depth
            fluxz = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][xx] * np.mean(-1/4 * np.real(ut_here*np.conj(st_cor) + np.conj(ut_here)*st_cor)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd) , axis=-1)[:,0] / self.soc_sca #waarschijnlijk kan dit analytisch
            return fluxz
        
        else: print('ERROR: invalid dep')
        

    def func_jac_bnd_cor(key,loc,dep): 
        # =============================================================================
        # Jacobian for contribution to transport at boundaries due to boundary correction
        # =============================================================================
        if loc == 'loc x=0':xx = -1
        elif loc == 'loc x=-L': xx = 0    
        else: print('ERROR: no correct loc')
        #selected tidal velocity
        ut_here = self.ch_pars[key]['ut'][:,xx]
                
        if dep == 'da': #derivatives for st_cor, depth-averaged
            dT_dBR = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][xx] * np.mean(-1/4 * 2*np.real(ut_here)*np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)) , axis=1) #waarschijnlijk kan dit analytisch
            dT_dBI = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][xx] * np.mean(-1/4 * 2*np.imag(ut_here)*np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)) , axis=1) #waarschijnlijk kan dit analytisch
            return np.concatenate([dT_dBR, dT_dBI])

        elif dep == 'dv':#derivatives for st_cor, at depth levels
            dTz_dBR = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][xx] * np.mean(-1/4 * 2*np.real(ut_here)*np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)) * np.cos(nn*np.pi*z_nd) , axis=-1) #waarschijnlijk kan dit analytisch
            dTz_dBI = self.ch_gegs[key]['H']*self.ch_pars[key]['b'][xx] * np.mean(-1/4 * 2*np.imag(ut_here)*np.cos(np.arange(self.M)[:,np.newaxis]*np.pi*np.linspace(-1,0,self.nz)) * np.cos(nn*np.pi*z_nd) , axis=-1) #waarschijnlijk kan dit analytisch
            return np.concatenate([dTz_dBR.T, dTz_dBI.T]).T

        else: print('ERROR: invalid dep')
        #scaling is a mystery to me - ok boomer
    
    
  
    def func_sol_boundarylayer(dat_C):
        # =============================================================================
        # solution vector for the matching conditions for the boundary layer correction 
        # =============================================================================
        #prepare
        sigma = np.linspace(-1,0,self.nz)
        z_inds = np.linspace(0,self.nz-1,self.M,dtype=int) # at which vertical levels we evaluate the expression

        res_3ch = []
        for ch in range(3): 
            #prepare
            xi_C=-1 if dat_C[ch][1] == 'loc x=0' else 0
            key_here = dat_C[ch][0] #key
                        
            # =============================================================================
            # calculate (not corrected) tidal salinity
            # =============================================================================
            #coefficients
            zlist = np.linspace(-self.ch_gegs[key_here]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
            c1 = - self.ch_pars[key_here]['Kv_t']/(1j*self.omega)
            c2 = self.g/(self.omega**2) * self.ch_pars[key_here]['detadx'][:,xi_C] * dat_C[ch][4]
            c3 = - nn*np.pi/self.ch_gegs[key_here]['H'] * dat_C[ch][2][:,np.newaxis,np.newaxis] * self.ch_pars[key_here]['eta'][:,xi_C]
            c4 = - nn*np.pi/self.ch_gegs[key_here]['H'] * dat_C[ch][2][:,np.newaxis,np.newaxis] * self.g/(self.omega**2) * (self.ch_pars[key_here]['detadx2'][:,xi_C] + self.ch_pars[key_here]['detadx'][:,xi_C]/self.ch_pars[key_here]['bn'][xi_C])
            
            dc2dx = self.g/(self.omega**2) * (self.ch_pars[key_here]['detadx2'][:,xi_C] * dat_C[ch][4] + self.ch_pars[key_here]['detadx'][:,xi_C]*dat_C[ch][5]) 
            dc3dx = - nn*np.pi/self.ch_gegs[key_here]['H'] * (self.ch_pars[key_here]['eta'][:,xi_C]*dat_C[ch][3][:,np.newaxis,np.newaxis] + dat_C[ch][2][:,np.newaxis,np.newaxis]*self.ch_pars[key_here]['detadx'][:,xi_C])
            dc4dx = - nn*np.pi/self.ch_gegs[key_here]['H'] * self.g/(self.omega**2) * (dat_C[ch][3][:,np.newaxis,np.newaxis] * (self.ch_pars[key_here]['detadx2'][:,xi_C] + self.ch_pars[key_here]['detadx'][:,xi_C]/self.ch_pars[key_here]['bn'][xi_C]) \
                    + dat_C[ch][2][:,np.newaxis,np.newaxis] * (self.ch_pars[key_here]['detadx3'][:,xi_C] + self.ch_pars[key_here]['detadx2'][:,xi_C]/self.ch_pars[key_here]['bn'][xi_C]))
            
            #tidal salinity and gradient
            st_C = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key_here]['deA'],self.ch_gegs[key_here]['H'],self.ch_pars[key_here]['B'],nn))    
            dstdx_C = (dsti_dc2(zlist,c1,(self.ch_pars[key_here]['deA'],self.ch_gegs[key_here]['H'],self.ch_pars[key_here]['B'],nn)) * dc2dx 
                + (dsti_dc3(zlist,c1,(self.ch_pars[key_here]['deA'],self.ch_gegs[key_here]['H'],self.ch_pars[key_here]['B'],nn)) * dc3dx).sum(0) 
                + (dsti_dc4(zlist,c1,(self.ch_pars[key_here]['deA'],self.ch_gegs[key_here]['H'],self.ch_pars[key_here]['B'],nn)) * dc4dx).sum(0))[0]
            #epsilon, the normalised horizontal diffusion
            eps = self.Kh_tide/(self.omega*self.Lsc**2)
            #the normalized tidal salinity, complex number
            P = st_C[0]/self.soc_sca
            #the normalized tidal salinity gradient, complex number
            dPdx = dstdx_C[0] /self.soc_sca*self.Lsc

            # =============================================================================
            # calculate salinity correction
            # =============================================================================
            Bm_Re = dat_C[ch][6][:self.M]
            Bm_Im = dat_C[ch][6][self.M:]
            Bm_Re_sum = np.sum(Bm_Re[:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0) #real part of salinity correction
            Bm_Im_sum = np.sum(Bm_Im[:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0) #imaginary part of salinity correction
 
            res_3ch.append([eps,np.real(P),np.imag(P),np.real(dPdx),np.imag(dPdx),Bm_Re_sum,Bm_Im_sum])

        # =============================================================================
        # calculate the matching conditions for the solution vector
        # =============================================================================
        #real parts C1 and C2 equal
        sol_pt1 = ((res_3ch[0][1]+res_3ch[0][5]) - (res_3ch[1][1]+res_3ch[1][5]))[z_inds]
        #imaginairy parts C1 and C2 equal
        sol_pt2 = ((res_3ch[0][2]+res_3ch[0][6]) - (res_3ch[1][2]+res_3ch[1][6]))[z_inds]
        #real parts C2 and C3 equal
        sol_pt3 = ((res_3ch[1][1]+res_3ch[1][5]) - (res_3ch[2][1]+res_3ch[2][5]))[z_inds]
        #imaginairy parts C2 and C3 equal
        sol_pt4 = ((res_3ch[1][2]+res_3ch[1][6]) - (res_3ch[2][2]+res_3ch[2][6]))[z_inds]
        
        #(diffusive) transport in tidal cycle conserved
        sol_pt5 = 0
        sol_pt6 = 0
        for ch in range(3):#transport for the channels
            if dat_C[ch][1] == 'loc x=-L':
                sol_pt5 += -self.ch_gegs[dat_C[ch][0]]['H']*self.ch_pars[dat_C[ch][0]]['b'][0]*res_3ch[ch][0] * ( res_3ch[ch][3] + res_3ch[ch][5]/np.sqrt(res_3ch[ch][0]) )[z_inds]
                sol_pt6 += -self.ch_gegs[dat_C[ch][0]]['H']*self.ch_pars[dat_C[ch][0]]['b'][0]*res_3ch[ch][0] * ( res_3ch[ch][4] + res_3ch[ch][6]/np.sqrt(res_3ch[ch][0]) )[z_inds]
            elif dat_C[ch][1] == 'loc x=0':
                sol_pt5 += self.ch_gegs[dat_C[ch][0]]['H']*self.ch_pars[dat_C[ch][0]]['b'][-1]*res_3ch[ch][0]* ( res_3ch[ch][3] - res_3ch[ch][5]/np.sqrt(res_3ch[ch][0]) )[z_inds]
                sol_pt6 += self.ch_gegs[dat_C[ch][0]]['H']*self.ch_pars[dat_C[ch][0]]['b'][-1]*res_3ch[ch][0]* ( res_3ch[ch][4] - res_3ch[ch][6]/np.sqrt(res_3ch[ch][0]) )[z_inds]
            else: print('ERROR')

        #return the results 
        return sol_pt1,sol_pt2,sol_pt3,sol_pt4,sol_pt5,sol_pt6
    

    def func_jac_boundarylayer(dat_C):
        # =============================================================================
        # jacobian associated with the matching conditions for the boundary layer correction 
        # =============================================================================
        #prepare
        eps = self.Kh_tide/(self.omega*self.Lsc**2)        #epsilon, the normalised horizontal diffusion
        key_here = dat_C[0] #key        
        zlist = np.linspace(-self.ch_gegs[key_here]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        nph = nn*np.pi/self.ch_gegs[key_here]['H']
        
        #coefficients
        c1 = - self.ch_pars[key_here]['Kv_t']/(1j*self.omega)
        dsdc2 = dsti_dc2(zlist,c1,(self.ch_pars[key_here]['deA'],self.ch_gegs[key_here]['H'],self.ch_pars[key_here]['B'],nn))
        dsdc3 = dsti_dc3(zlist,c1,(self.ch_pars[key_here]['deA'],self.ch_gegs[key_here]['H'],self.ch_pars[key_here]['B'],nn))
        dsdc4 = dsti_dc4(zlist,c1,(self.ch_pars[key_here]['deA'],self.ch_gegs[key_here]['H'],self.ch_pars[key_here]['B'],nn))
        
        #local parameters
        dx_here = self.ch_gegs[key_here]['dx'][[0,-1]]
        bn_here = self.ch_pars[key_here]['bn'][[0,-1]]
        ut_here = self.ch_pars[key_here]['ut'][:,[0,-1]]
        dutdx_here = self.ch_pars[key_here]['dutdx'][:,[0,-1]]
        eta_here= self.ch_pars[key_here]['eta'][:,[0,-1]]
        detadx_here= self.ch_pars[key_here]['detadx'][:,[0,-1]]
        detadx2_here= self.ch_pars[key_here]['detadx2'][:,[0,-1]]
        detadx3_here= self.ch_pars[key_here]['detadx3'][:,[0,-1]]
                
        # =============================================================================
        # derivatives for st and dstdx
        # =============================================================================
        if dat_C[1] == 'loc x=-L':          
            dst0_dsb0 = dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -3/(2*dx_here[0])
            dst0_dsb1 = dsdc2 * self.g/self.omega**2 * detadx_here[:,0] *  4/(2*dx_here[0]) 
            dst0_dsb2 = dsdc2 * self.g/self.omega**2 * detadx_here[:,0] * -1/(2*dx_here[0]) 
            dst0_dsn0 = (dsdc3 * - nph*eta_here[:,0] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]))
        
            dst0_x_dsb0 = self.Lsc*dsdc2 * self.g/(self.omega**2) * (detadx2_here[:,0] * -3/(2*dx_here[0]) + detadx_here[:,0] *  2/(dx_here[0]**2) )
            dst0_x_dsb1 = self.Lsc*dsdc2 * self.g/(self.omega**2) * (detadx2_here[:,0] *  4/(2*dx_here[0]) + detadx_here[:,0] * -5/(dx_here[0]**2) )
            dst0_x_dsb2 = self.Lsc*dsdc2 * self.g/(self.omega**2) * (detadx2_here[:,0] * -1/(2*dx_here[0]) + detadx_here[:,0] *  4/(dx_here[0]**2) )
            dst0_x_dsb3 = self.Lsc*dsdc2 * self.g/(self.omega**2) *  detadx_here[:,0] * -1/(dx_here[0]**2)
            dst0_x_dsn0 = self.Lsc*(dsdc3 * - nph * detadx_here[:,0] + dsdc4 * -nph * self.g/self.omega**2 * (detadx3_here[:,0] + detadx2_here[:,0]/bn_here[0]) \
                        + (dsdc3 * -nph *eta_here[:,0] + dsdc4 * -nph *  self.g/(self.omega**2) * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) * -3/(2*dx_here[0]))
            dst0_x_dsn1 = self.Lsc*(dsdc3 * -nph *eta_here[:,0] + dsdc4 * -nph *  self.g/(self.omega**2) * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) *  4/(2*dx_here[0])
            dst0_x_dsn2 = self.Lsc*(dsdc3 * -nph *eta_here[:,0] + dsdc4 * -nph *  self.g/(self.omega**2) * (detadx2_here[:,0] + detadx_here[:,0]/bn_here[0]) ) * -1/(2*dx_here[0])
            
            #real and imaginairy
            return (np.real(dst0_dsb0[0,0]), np.real(dst0_dsb1[0,0]), np.real(dst0_dsb2[0,0]),
                    np.real(dst0_dsn0[:,0]),
                    np.real(dst0_x_dsb0[0,0]),np.real(dst0_x_dsb1[0,0]),np.real(dst0_x_dsb2[0,0]),np.real(dst0_x_dsb3[0,0]),
                    np.real(dst0_x_dsn0[:,0]),np.real(dst0_x_dsn1[:,0]),np.real(dst0_x_dsn2[:,0])) , \
                   (np.imag(dst0_dsb0[0,0]), np.imag(dst0_dsb1[0,0]), np.imag(dst0_dsb2[0,0]),
                    np.imag(dst0_dsn0[:,0]),
                    np.imag(dst0_x_dsb0[0,0]),np.imag(dst0_x_dsb1[0,0]),np.imag(dst0_x_dsb2[0,0]),np.imag(dst0_x_dsb3[0,0]),
                    np.imag(dst0_x_dsn0[:,0]),np.imag(dst0_x_dsn1[:,0]),np.imag(dst0_x_dsn2[:,0])) , eps

        
        elif dat_C[1] == 'loc x=0':
            dst_1_dsb_1 = dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  3/(2*dx_here[1]) 
            dst_1_dsb_2 = dsdc2 * self.g/self.omega**2 * detadx_here[:,1] * -4/(2*dx_here[1]) 
            dst_1_dsb_3 = dsdc2 * self.g/self.omega**2 * detadx_here[:,1] *  1/(2*dx_here[1]) 
            dst_1_dsn_1 = (dsdc3 * - nph*eta_here[:,1] + dsdc4 * - nph * self.g/self.omega**2 * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1])) 
            
            dst_1_x_dsb_1 = self.Lsc*dsdc2 * self.g/(self.omega**2) * (detadx2_here[:,1] *  3/(2*dx_here[1]) + detadx_here[:,1] *  2/(dx_here[1]**2) )
            dst_1_x_dsb_2 = self.Lsc*dsdc2 * self.g/(self.omega**2) * (detadx2_here[:,1] * -4/(2*dx_here[1]) + detadx_here[:,1] * -5/(dx_here[1]**2) )
            dst_1_x_dsb_3 = self.Lsc*dsdc2 * self.g/(self.omega**2) * (detadx2_here[:,1] *  1/(2*dx_here[1]) + detadx_here[:,1] *  4/(dx_here[1]**2) )
            dst_1_x_dsb_4 = self.Lsc*dsdc2 * self.g/(self.omega**2) *  detadx_here[:,1] * -1/(dx_here[1]**2)
            dst_1_x_dsn_1 = self.Lsc*(dsdc3 * - nph * detadx_here[:,1] + dsdc4 * -nph * self.g/self.omega**2 * (detadx3_here[:,1] + detadx2_here[:,1]/bn_here[1]) \
                          + (dsdc3 * -nph *eta_here[:,1] + dsdc4 * -nph *  self.g/(self.omega**2) * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) *  3/(2*dx_here[1]))
            dst_1_x_dsn_2 = self.Lsc*(dsdc3 * -nph *eta_here[:,1] + dsdc4 * -nph *  self.g/(self.omega**2) * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) * -4/(2*dx_here[1])
            dst_1_x_dsn_3 = self.Lsc*(dsdc3 * -nph *eta_here[:,1] + dsdc4 * -nph *  self.g/(self.omega**2) * (detadx2_here[:,1] + detadx_here[:,1]/bn_here[1]) ) *  1/(2*dx_here[1])
            
            #real and imaginairy
            return (np.real(dst_1_dsb_1[0,0]), np.real(dst_1_dsb_2[0,0]), np.real(dst_1_dsb_3[0,0]),
                    np.real(dst_1_dsn_1[:,0]),
                    np.real(dst_1_x_dsb_1[0,0]), np.real(dst_1_x_dsb_2[0,0]), np.real(dst_1_x_dsb_3[0,0]), np.real(dst_1_x_dsb_4[0,0]),
                    np.real(dst_1_x_dsn_1[:,0]), np.real(dst_1_x_dsn_2[:,0]), np.real(dst_1_x_dsn_3[:,0])) ,\
                   (np.imag(dst_1_dsb_1[0,0]), np.imag(dst_1_dsb_2[0,0]), np.imag(dst_1_dsb_3[0,0]),
                    np.imag(dst_1_dsn_1[:,0]),
                    np.imag(dst_1_x_dsb_1[0,0]), np.imag(dst_1_x_dsb_2[0,0]), np.imag(dst_1_x_dsb_3[0,0]), np.imag(dst_1_x_dsb_4[0,0]),
                    np.imag(dst_1_x_dsn_1[:,0]), np.imag(dst_1_x_dsn_2[:,0]), np.imag(dst_1_x_dsn_3[:,0])) , eps
        
        

    def func_flux(key,sn,dsndx,dsbdx,dsbdx2,bb):       
        # =============================================================================
        # Function to calculate total transport and salinity
        # for plotting purposes afterwards
        # =============================================================================
        di_here = self.ch_pars[key]['di']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
    
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            c2[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
            c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.g/(self.omega**2) \
                * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
       
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))    
        
        #calculate corrected salinity
        st_cor = st.copy()
        sigma = np.linspace(-1,0,self.nz)
        if self.ch_gegs[key]['loc x=-L'][0] == 'j': 
            px_here = self.ch_outp[key]['px']-self.ch_outp[key]['px'][0]
            cor_dom = np.exp(-px_here/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(bb[0][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0)
            st_cor += self.soc_sca * cor_dom
    
        if self.ch_gegs[key]['loc x=0'][0] == 'j': 
            px_here = self.ch_outp[key]['px']
            cor_dom = np.exp(px_here/np.sqrt(self.Kh_tide/self.omega))[:,np.newaxis] *  np.sum(bb[1][:,np.newaxis] * np.cos(np.pi*np.arange(self.M)[:,np.newaxis]*sigma) , axis=0)
            st_cor += self.soc_sca * cor_dom
    
        # =============================================================================
        # Terms for in equation
        # =============================================================================
        flux = np.mean(-1/4 * np.real(self.ch_pars[key]['ut'][0]*np.conj(st) + np.conj(self.ch_pars[key]['ut'][0])*st) , axis=1) #part for dbdx
        flux_cor = np.mean(-1/4 * np.real(self.ch_pars[key]['ut'][0]*np.conj(st_cor) + np.conj(self.ch_pars[key]['ut'][0])*st_cor) , axis=1) #part for dbdx
        
        #remove sea parts 
        if self.ch_gegs[key]['loc x=-L'][0] == 's': 
            flux[:di_here[1]] = np.nan
            flux_cor[:di_here[1]] = np.nan
        if self.ch_gegs[key]['loc x=0'][0] == 's': 
            x_sea = np.linspace(0,self.ch_gegs[key]['L'][-1],self.ch_pars[key]['nxn'][-1])
            dsbdx_h = dsbdx[:,[di_here[-2]-1],:]
    
            flux[di_here[-2]:] = flux[di_here[-2]-1]/dsbdx_h * np.exp(-x_sea/self.ch_pars[key]['bn'][-1]) * dsbdx[0,di_here[-2]:,0]
            flux_cor[di_here[-2]:] = flux_cor[di_here[-2]-1]/dsbdx_h * np.exp(-x_sea/self.ch_pars[key]['bn'][-1]) * dsbdx[0,di_here[-2]:,0]
    
        return -flux_cor , st
       
    def func_vertterms(key,sn,dsndx,dsbdx,dsbdx2,bb): #we are not yet taking into account the boundary layer formulation
        # =============================================================================
        #  tidal velocities and salinity
        # =============================================================================
        di_here = self.ch_pars[key]['di']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            c2[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
            c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.g/(self.omega**2) \
                * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
        #coefficients for derivative
        dc2dx, dc3dx, dc4dx = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            dc2dx[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]*dsbdx[:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]*dsbdx2[:,di_here[dom]:di_here[dom+1]])
            dc3dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * (self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]*dsndx[:,di_here[dom]:di_here[dom+1]] + sn[:,di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]])
            dc4dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * self.g/(self.omega**2) * (dsndx[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]) \
                                                                                                                 + sn[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]))
        #tidal salinity    
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))         
        #derivative of tidal salinity
        dstidx = (dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc2dx 
            + (dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc3dx).sum(0) 
            + (dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc4dx).sum(0))[0]
 
        utb = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0]
        utp = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0]
        stb = np.mean(st,1)[:,np.newaxis] #kan in theorie analytisch
        stp = st-stb #kan in theorie analytisch
        
        dstibdx = np.mean(dstidx,1)[:,np.newaxis] #kan in theorie analytisch
        dstipdx = dstidx-dstibdx #kan in theorie analytisch
        dstdz = dsti_dz(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))

        nph = nn*np.pi/self.ch_gegs[key]['H']
        
        T1 = -1/4 * np.real(utb*np.conj(dstipdx) + np.conj(utb)*dstipdx) 
        T2 = -1/4 * np.real(utp*np.conj(dstibdx) + np.conj(utp)*dstibdx)
        T3 = -1/4 * np.real(utp*np.conj(dstipdx) + np.conj(utp)*dstipdx) 
        T4 = -np.mean(T3,axis = 1) #there is a minus sign in the equation!
        T5 = -1/4 * np.real(self.ch_pars[key]['wt'] *np.conj(dstdz) + np.conj(self.ch_pars[key]['wt'] )*dstdz) [0]
        
        # =============================================================================
        #         boundary layer correctoin
        # =============================================================================
        stc      = np.zeros(st.shape,dtype=complex)
        stc_dz   = np.zeros(st.shape,dtype=complex)
        stc_dx_b = np.zeros(st.shape,dtype=complex)
        stc_dx_p = np.zeros(st.shape,dtype=complex)

        eps = self.Kh_tide/(self.omega)
        if self.ch_gegs[key]['loc x=-L'][0] == 'j': 
            #calculate x
            x_all = np.arange(self.ch_pars[key]['di'][1])*self.ch_gegs[key]['dx'][0] 
            x_bnd = x_all[np.where(x_all<(-np.sqrt(eps)*np.log(self.tol)))[0][1:]]
            #calculate the derivative of st
            st_cor = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(bb[0] * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            stc[:len(st_cor)] = st_cor
 
            stc_dx_b[:len(st_cor)] += -1/np.sqrt(eps) * np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * bb[0][0] *self.soc_sca 
            stc_dx_p[:len(st_cor)] += -1/np.sqrt(eps) * np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(bb[0][1:] * np.cos(np.arange(1,self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            stc_dz[:len(st_cor)]   += np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(bb[0] * -np.arange(self.M)*np.pi/self.ch_gegs[key]['H'] * np.sin(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 

        if self.ch_gegs[key]['loc x=0'][0] == 'j': 
            #calculate x
            x_all = np.arange(self.ch_pars[key]['di'][-2]-self.ch_pars[key]['di'][-1]+1,1)*self.ch_gegs[key]['dx'][-1] 
            x_bnd = x_all[np.where(x_all>(np.sqrt(eps)*np.log(self.tol)))[0][:-1]]
            if -x_bnd[0] > self.ch_gegs[key]['L'][-1]: print('ERROR: boundary layer too large. Can be solved...')
            #calcualte the derivative of st
            st_cor = np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(bb[1] * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            stc[-len(st_cor):] = st_cor
            
            stc_dx_b[-len(st_cor):] += 1/np.sqrt(eps) * np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * bb[1][0]*self.soc_sca 
            stc_dx_p[-len(st_cor):] += 1/np.sqrt(eps) * np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(bb[1][1:] * np.cos(np.arange(1,self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            stc_dz[-len(st_cor):]   += np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(bb[1] * -np.arange(self.M)*np.pi/self.ch_gegs[key]['H'] * np.sin(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
        
        T1c = -1/4 * np.real(utb*np.conj(stc_dx_p) + np.conj(utb)*stc_dx_p) 
        T2c = -1/4 * np.real(utp*np.conj(stc_dx_b) + np.conj(utp)*stc_dx_b)
        T3c = -1/4 * np.real(utp*np.conj(stc_dx_p) + np.conj(utp)*stc_dx_p) 
        T4c = -np.mean(T3c,axis = 1)
        T5c = -1/4 * np.real(self.ch_pars[key]['wt'] *np.conj(stc_dz) + np.conj(self.ch_pars[key]['wt'] )*stc_dz) [0]
        
        return T1,T2,T3,T4,T5 , T1c,T2c,T3c,T4c,T5c
        #return T1, T2*0,T3*0,T4*0,T5 , T1c,T2c*0,T3c*0,T4c*0,T5c
       
    def func_solz_int_checks(key,sn,dsndx,dsbdx,dsbdx2): 
        # =============================================================================
        # Function to calculate the contribution of the tide to the subtidal stratification
        # =============================================================================
        di_here = self.ch_pars[key]['di']
        zlist = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]
        # ===========================================================================
        # calculate salinity and salt derivative
        # ===========================================================================        
        #coefficients for salinity calculation
        c1 = - self.ch_pars[key]['Kv_t']/(1j*self.omega)
        c2, c3, c4 = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            c2[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]] * dsbdx[:,di_here[dom]:di_here[dom+1]]
            c3[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]
            c4[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * sn[:,di_here[dom]:di_here[dom+1]] * self.g/(self.omega**2) \
                * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom])
        #coefficients for derivative
        dc2dx, dc3dx, dc4dx = np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex) , np.zeros((self.N,di_here[-1],self.nz),dtype=complex)
        for dom in range(len(self.ch_pars[key]['nxn'])): 
            dc2dx[:,di_here[dom]:di_here[dom+1]] = self.g/(self.omega**2) * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]*dsbdx[:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]*dsbdx2[:,di_here[dom]:di_here[dom+1]])
            dc3dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * (self.ch_pars[key]['eta'][:,di_here[dom]:di_here[dom+1]]*dsndx[:,di_here[dom]:di_here[dom+1]] + sn[:,di_here[dom]:di_here[dom+1]]*self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]])
            dc4dx[:,di_here[dom]:di_here[dom+1]] = - nn*np.pi/self.ch_gegs[key]['H'] * self.g/(self.omega**2) * (dsndx[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]) \
                                                                                                                 + sn[:,di_here[dom]:di_here[dom+1]] * (self.ch_pars[key]['detadx3'][:,di_here[dom]:di_here[dom+1]] + self.ch_pars[key]['detadx2'][:,di_here[dom]:di_here[dom+1]]/self.ch_pars[key]['bn'][dom]))
        #tidal salinity    
        st = sti(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))         
        #derivative of tidal salinity
        dstidx = (dsti_dc2(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc2dx 
            + (dsti_dc3(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc3dx).sum(0) 
            + (dsti_dc4(zlist,c1,(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn)) * dc4dx).sum(0))[0]
    
        utb = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0]
        utp = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0]
        stb = np.mean(st,1)[:,np.newaxis] #kan in theorie analytisch
        stp = st-stb #kan in theorie analytisch
        
        dstibdx = np.mean(dstidx,1)[:,np.newaxis] #kan in theorie analytisch
        dstipdx = dstidx-dstibdx #kan in theorie analytisch
        dstdz = dsti_dz(zlist,(c1,c2,c3,c4),(self.ch_pars[key]['deA'],self.ch_gegs[key]['H'],self.ch_pars[key]['B'],nn))
    
        term1 = self.Lsc/self.soc_sca * (-1/4*np.real(utp*np.conj(dstidx) + np.conj(utp)*dstidx)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)).mean(2)
        term2 = self.Lsc/self.soc_sca * (-1/4*np.real(utb*np.conj(dstipdx) + np.conj(utb)*dstipdx)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)).mean(2)
        term3 = self.Lsc/self.soc_sca * (-1/4*np.real(self.ch_pars[key]['wt'] * np.conj(dstdz) + np.conj(self.ch_pars[key]['wt']) * dstdz) * np.cos(nn*np.pi*z_nd)).mean(2)
    
        #remove sea parts 
        if self.ch_gegs[key]['loc x=-L'][0] == 's':  
            term1[:,:di_here[1]] = 0
            term2[:,:di_here[1]] = 0
            term3[:,:di_here[1]] = 0
        if self.ch_gegs[key]['loc x=0'][0] == 's':   
            term1[:,di_here[-2]:] = 0
            term2[:,di_here[-2]:] = 0
            term3[:,di_here[-2]:] = 0
            
        #maar eigenlijk willen we deze termen niet projecteren. 
        term1np = self.Lsc/self.soc_sca * (-1/4*np.real(utp*np.conj(dstidx) + np.conj(utp)*dstidx))
        term2np = self.Lsc/self.soc_sca * (-1/4*np.real(utb*np.conj(dstipdx) + np.conj(utb)*dstipdx))
        term3np = self.Lsc/self.soc_sca * (-1/4*np.real(self.ch_pars[key]['wt'] * np.conj(dstdz) + np.conj(self.ch_pars[key]['wt']) * dstdz)) 
    
        #remove sea parts 
        if self.ch_gegs[key]['loc x=-L'][0] == 's':  
            term1np[:,:di_here[1]] = 0
            term2np[:,:di_here[1]] = 0
            term3np[:,:di_here[1]] = 0
        if self.ch_gegs[key]['loc x=0'][0] == 's':   
            term1np[:,di_here[-2]:] = 0
            term2np[:,di_here[-2]:] = 0
            term3np[:,di_here[-2]:] = 0
        
        return term1np, term2np, term3np
        

    def func_solz_intbl_checks(key,bb,loc): 
        # =============================================================================
        # Function to calculate the contribution of the boundary layer correction to the inner domain
        # =============================================================================
        #prepare
        Bm = bb.copy()#[:self.M] + 1j*bb[self.M:]
        eps = self.Kh_tide/(self.omega)
        if np.isnan(bb[0]):
            return np.nan, np.nan, np.nan, np.nan
        
        
        if loc == 'loc x=-L':
            #calculate x 
            x_all = np.arange(self.ch_pars[key]['di'][1])*self.ch_gegs[key]['dx'][0] 
            x_bnd = x_all[np.where(x_all<(-np.sqrt(eps)*np.log(self.tol)))[0][1:]]
            #print(key, x_all, x_bnd)
            if x_bnd[-1] >= self.ch_gegs[key]['L'][0]: print('ERROR: boundary layer too large. Can be solved...')
            #calculate the salinity correctoin
            st_cor = np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            st_cor_x = st_cor * -1/np.sqrt(eps)
            
            #calculate the associated flux. 
            utb_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0][1:len(x_bnd)+1]
            utp_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0][1:len(x_bnd)+1]
            wt_here   = self.ch_pars[key]['wt'][0,1:len(x_bnd)+1]
            stp_cor_x = -1/np.sqrt(eps) * np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm[1:] * np.cos(np.arange(1,self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            st_cor_z= np.exp(-x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * -np.arange(self.M)*np.pi/self.ch_gegs[key]['H'] * np.sin(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            
            
            t1 = np.mean(-1/4 * np.real(utp_here*np.conj( st_cor_x) + np.conj(utp_here)* st_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
            t2 = np.mean(-1/4 * np.real(utb_here*np.conj(stp_cor_x) + np.conj(utb_here)*stp_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
            t3 = np.mean(-1/4 * np.real(wt_here*np.conj(st_cor_z  ) + np.conj(wt_here)*st_cor_z  )[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
       
            t1np = -1/4 * np.real(utp_here*np.conj( st_cor_x) + np.conj(utp_here)* st_cor_x) * self.Lsc/self.soc_sca
            t2np = -1/4 * np.real(utb_here*np.conj(stp_cor_x) + np.conj(utb_here)*stp_cor_x) * self.Lsc/self.soc_sca
            t3np = -1/4 * np.real(wt_here*np.conj(st_cor_z  ) + np.conj(wt_here)*st_cor_z  ) * self.Lsc/self.soc_sca
            t4np = -1/4 * np.real(utp_here*np.conj(stp_cor_x) + np.conj(utp_here)*stp_cor_x) * self.Lsc/self.soc_sca
        
        elif loc == 'loc x=0':
            #calculate x
            x_all = np.arange(self.ch_pars[key]['di'][-2]-self.ch_pars[key]['di'][-1]+1,1)*self.ch_gegs[key]['dx'][-1] 
            x_bnd = x_all[np.where(x_all>(np.sqrt(eps)*np.log(self.tol)))[0][:-1]]
            if -x_bnd[0] >= self.ch_gegs[key]['L'][-1]: print('ERROR: boundary layer too large. Can be solved...')
            #calculate the salinity correction
            st_cor = np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * np.cos(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            st_cor_x = st_cor * 1/np.sqrt(eps)
        
            #calculate the associated flux. 
            utb_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * (self.ch_pars[key]['B']/self.ch_pars[key]['deA']*np.sinh(self.ch_pars[key]['deA'])-1))[0,-len(x_bnd)-1:-1]
            utp_here = (self.g/(1j*self.omega) * self.ch_pars[key]['detadx'] * self.ch_pars[key]['B'] * (np.cosh(self.ch_pars[key]['deA']*z_nd) - np.sinh(self.ch_pars[key]['deA'])/self.ch_pars[key]['deA']))[0,-len(x_bnd)-1:-1]
            wt_here   = self.ch_pars[key]['wt'][0,-len(x_bnd)-1:-1]
            stp_cor_x = 1/np.sqrt(eps) * np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm[1:] * np.cos(np.arange(1,self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
            st_cor_z= np.exp(x_bnd[:,np.newaxis]/np.sqrt(eps)) * np.sum(Bm * -np.arange(self.M)*np.pi/self.ch_gegs[key]['H'] * np.sin(np.arange(self.M)*np.pi*np.linspace(-1,0,self.nz)[:,np.newaxis]) ,1)*self.soc_sca 
    
            t1 = np.mean(-1/4 * np.real(utp_here*np.conj( st_cor_x) + np.conj(utp_here)* st_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
            t2 = np.mean(-1/4 * np.real(utb_here*np.conj(stp_cor_x) + np.conj(utb_here)*stp_cor_x)[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
            t3 = np.mean(-1/4 * np.real(wt_here*np.conj(st_cor_z  ) + np.conj(wt_here)*st_cor_z  )[np.newaxis,:,:] * np.cos(nn*np.pi*z_nd)  , -1) * self.Lsc/self.soc_sca
                        
            t1np = -1/4 * np.real(utp_here*np.conj( st_cor_x) + np.conj(utp_here)* st_cor_x) * self.Lsc/self.soc_sca
            t2np = -1/4 * np.real(utb_here*np.conj(stp_cor_x) + np.conj(utb_here)*stp_cor_x) * self.Lsc/self.soc_sca
            t3np = -1/4 * np.real(wt_here*np.conj(st_cor_z  ) + np.conj(wt_here)*st_cor_z  ) * self.Lsc/self.soc_sca
            t4np = -1/4 * np.real(utp_here*np.conj(stp_cor_x) + np.conj(utp_here)*stp_cor_x) * self.Lsc/self.soc_sca

        else:  print('ERROR!')

        #return t1, t2, t3
        return t1np, t2np, t3np, t4np

       
    # =============================================================================
    # save this
    # =============================================================================
    self.ch_tide['sol_i'] = func_sol_int
    self.ch_tide['jac_i'] = func_jac_int
    self.ch_tide['sol_b'] = func_sol_bnd
    self.ch_tide['jac_b'] = func_jac_bnd
    self.ch_tide['sol_s'] = func_sol_sea
    self.ch_tide['jac_s'] = func_jac_sea

    self.ch_tide['solz_i'] = func_solz_int
    self.ch_tide['jacz_i'] = func_jacz_int

    #for boundar layer correction
    self.ch_tide['sol_bound'] = func_sol_boundarylayer
    self.ch_tide['jac_bound'] = func_jac_boundarylayer
    self.ch_tide['sol_i_cor'] = func_sol_int_cor
    self.ch_tide['jac_i_cor'] = func_jac_int_cor
    self.ch_tide['sol_b_cor'] = func_sol_bnd_cor
    self.ch_tide['jac_b_cor'] = func_jac_bnd_cor
    
    #calculations afterwards. 
    self.ch_tide['flux'] = func_flux
    self.ch_tide['vertterms'] = func_vertterms
    self.ch_tide['check_intti'] = func_solz_int_checks
    self.ch_tide['check_inttibl'] = func_solz_intbl_checks
        