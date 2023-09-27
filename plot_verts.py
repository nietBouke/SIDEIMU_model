import numpy as np
import matplotlib.pyplot as plt

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

def plot_proc_vt(self,key, xloc):

    #local variables
    sn,dsndx,dsbdx,dsbdx2 = conv_ans(self, key)
    u_bar = self.ch_pars[key]['Q'] * self.ch_pars[key]['bH_1']    
    nnp = np.arange(1,self.N+1)*np.pi
    dx = self.ch_pars[key]['dl']*self.Lsc
    g1,g2,g3,g4,g5 = self.ch_pars[key]['g1'] , self.ch_pars[key]['g2'] , self.ch_pars[key]['g3'] , self.ch_pars[key]['g4'], self.ch_pars[key]['g5']
    alf, bex  =  self.ch_pars[key]['alf'] , self.ch_pars[key]['bex']
    
    #print(u_bar.shape)
    
    term_build = {
        '1': lambda x, z : u_bar[x] * np.sum([dsndx[n,x,0] * np.cos(nnp[n]*z) for n in range(self.N)],0),
        '2': lambda x, z : (u_bar[x]*(g1 + g2*z**2) + alf*dsbdx[0,x,0]*(g3+g4*z**2+g5*z**3))  * np.sum([dsndx[n,x] * np.cos(nnp[n]*z) for n in range(self.N)],0),
        '3': lambda x, z : (u_bar[x]*(g1 + g2*z**2) + alf*dsbdx[0,x,0]*(g3+g4*z**2+g5*z**3) ) * dsbdx[0,x,0],
        '4': lambda x, z : -(bex[x]**-1*np.sum([sn[n,x,0]*alf*dsbdx[0,x,0] * (2*g4*np.cos(nnp[n])/(nnp[n]**2)+g5*((6*np.cos(nnp[n])-6)/(nnp[n]**4) -3*np.cos(nnp[n])/(nnp[n]**2))) for n in range(self.N)],0) 
                                    + np.sum([ sn[n,x,0] * alf*dsbdx2[0,x,0] *(2*g4*np.cos(nnp[n])/(nnp[n]**2)+g5*((6*np.cos(nnp[n])-6)/(nnp[n]**4) -3*np.cos(nnp[n])/(nnp[n]**2))) for n in range(self.N)],0)
                                    + np.sum([ dsndx[0,x,0] * (2*g2*u_bar[x] * np.cos(nnp[n])/nnp[n]**2 ) for n in range(self.N)],0)
                                    + np.sum([ dsndx[0,x,0] * alf*dsbdx[0,x,0]*(2*g4*np.cos(nnp[n])/(nnp[n]**2)+g5*((6*np.cos(nnp[n])-6)/(nnp[n]**4) -3*np.cos(nnp[n])/(nnp[n]**2))) for n in range(self.N)],0)),  
        '5': lambda x, z : -(alf*self.ch_gegs[key]['H']* ( dsbdx2[0,x,0] + bex[x]**-1*dsbdx[0,x,0] ) * (-g5/4*z**4-g4/3*z**3 -g3*z)) * np.sum([nnp[n]/self.ch_gegs[key]['H'] * sn[n,x,0]*np.sin(nnp[n]*z) for n in range(self.N)],0) , 
        '6': lambda x, z : self.ch_pars[key]['Kv'] * np.sum([nnp[n]**2/self.ch_gegs[key]['H']**2 * sn[n,x,0]*np.cos(nnp[n]*z) for n in range(self.N)],0) ,
        '7': lambda x, z : -2*bex[x]**(-1)*self.ch_pars[key]['Kh'][x]*np.sum([dsndx[n,x] * np.cos(nnp[n]*z) for n in range(self.N)],0) - self.ch_pars[key]['Kh'][x] * np.sum([(sn[n,x+1,0]-2*sn[n,x,0]+sn[n,x-1,0])/(dx[x]**2) * np.cos(nnp[n]*z) for n in range(self.N)],0) ,
        }    
    #z_nd = np.linspace(-self.ch_gegs[key]['H'],0,self.nz)[np.newaxis,np.newaxis,:]

    
    def plot_vert(xloc):
        T1,T2,T3 = term_build['1'](xloc, self.z_nd),  term_build['2'](xloc, self.z_nd),  term_build['3'](xloc, self.z_nd),  
        T4 =  term_build['4'](xloc, self.z_nd),
        T5,T6,T7 = term_build['5'](xloc, self.z_nd),  term_build['6'](xloc, self.z_nd),  term_build['7'](xloc, self.z_nd) 
        T8, T9, T10, T11, T12, T8c, T9c, T10c, T11c, T12c = self.ch_tide['vertterms'](key,*conv_ans(self, key),self.bbb_n[key])
        
        
        T8, T9, T10, T11, T12 = T8[xloc]+T8c[xloc], T9[xloc]+T9c[xloc], T10[xloc]+T10c[xloc], T11[xloc]+T11c[xloc], T12[xloc]+T12c[xloc]
        #T8, T9, T10, T11, T12 = T8[xloc]-T8c[xloc], T9[xloc]-T9c[xloc], T10[xloc]-T10c[xloc], T11[xloc]-T11c[xloc], T12[xloc]-T12c[xloc]
        #T8, T9, T10, T11, T12 = T8[xloc], T9[xloc], T10[xloc], T11[xloc], T12[xloc]
        T8c, T9c, T10c, T11c, T12c = T8c[xloc], T9c[xloc], T10c[xloc], T11c[xloc], T12c[xloc]
        
        #er gaat ergens iets fout met deze termen. 
        #plot
        plt.plot(T1,self.z_nd,label='T1')
        plt.plot(T2,self.z_nd,label='T2')
        plt.plot(T3,self.z_nd,label='T3')
        plt.plot(np.repeat(T4,len(self.z_nd)),self.z_nd,label='T4')
        plt.plot(T5,self.z_nd,label='T5')
        plt.plot(T6,self.z_nd,label='T6')
        plt.plot(T7,self.z_nd,label='T7')
        
        
        plt.plot(T8,self.z_nd,label='T8',ls=':')
        plt.plot(T9,self.z_nd,label='T9',ls=':')
        plt.plot(T10,self.z_nd,label='T10',ls=':')
        plt.plot(np.repeat(T11,len(self.z_nd)),self.z_nd,label='T11',ls=':')
        plt.plot(T12,self.z_nd,label='T12',ls=':')
        
        plt.plot(T1+T2+T3+T4+T5+T6+T7+T8+T9+T10+T11+T12,self.z_nd,c='black',lw=3,label='tot')
        #plt.plot(T1+T2+T3+T4+T5+T6+T7,self.z_nd,c='grey',lw=3,label='tot')
        #plt.title('Location = '+str(self.ch_outp[key]['px'][xloc]/1000)+' km')
        plt.ylabel('$z/H$')
        plt.grid()
        plt.legend()
        plt.show()
        
        #print(T12c)
        
    def plot_vert_ground(xloc):
        T1,T2,T3 = term_build['1'](xloc, self.z_nd),  term_build['2'](xloc, self.z_nd),  term_build['3'](xloc, self.z_nd),  
        T4 =  term_build['4'](xloc, self.z_nd),
        T5,T6,T7 = term_build['5'](xloc, self.z_nd),  term_build['6'](xloc, self.z_nd),  term_build['7'](xloc, self.z_nd) 
        #T8, T9, T10, T11, T12, T8c, T9c, T10c, T11c, T12c = self.ch_tide['vertterms'](key,*conv_ans(self, key),self.bbb_n[key])
        
        T10,T11,T12 = self.ch_tide['check_intti'](key,*conv_ans(self, key))
        T10c1,T11c1,T12c1,T13c1 = self.ch_tide['check_inttibl'](key,self.bbb_n[key][0],'loc x=-L')
        T10c2,T11c2,T12c2,T13c2 = self.ch_tide['check_inttibl'](key,self.bbb_n[key][1],'loc x=0')
        
        T10tot = T10.copy()
        if self.ch_gegs[key]['loc x=-L' ][0] =='j':  T10tot[1:len(T10c1)+1] += T10c1
        if self.ch_gegs[key]['loc x=0' ][0] =='j':  T10tot[-len(T10c2)-1:-1] += T10c2
        T10here = T10tot[xloc] * self.soc_sca/self.Lsc
        
        T11tot = T11.copy()
        if self.ch_gegs[key]['loc x=-L' ][0] =='j': T11tot[1:len(T11c1)+1] += T11c1-T13c1
        if self.ch_gegs[key]['loc x=0' ][0] =='j': T11tot[-1-len(T11c2):-1] += T11c2-T13c2
        T11here = T11tot[xloc] * self.soc_sca/self.Lsc
        
        T12tot = T12[0].copy()
        if self.ch_gegs[key]['loc x=-L' ][0] =='j': T12tot[1:len(T12c1)+1] += T12c1
        if self.ch_gegs[key]['loc x=0' ][0] =='j': T12tot[-len(T12c2)-1:-1]+= T12c2
        T12here = T12tot[xloc] * self.soc_sca/self.Lsc
        
        #plot
        plt.plot(T1,self.z_nd,label='T1')
        plt.plot(T2,self.z_nd,label='T2')
        plt.plot(T3,self.z_nd,label='T3')
        plt.plot(np.repeat(T4,len(self.z_nd)),self.z_nd,label='T4')
        plt.plot(T5,self.z_nd,label='T5')
        plt.plot(T6,self.z_nd,label='T6')
        plt.plot(T7,self.z_nd,label='T7')
        
       
        plt.plot(T10here,self.z_nd,label='T10',ls=':')
        plt.plot(T11here,self.z_nd,label='T11',ls=':')
        plt.plot(T12here,self.z_nd,label='T12',ls=':')

        plt.plot(T1+T2+T3+T4+T5+T6+T7+T10here+T11here+T12here,self.z_nd,c='black',lw=3,label='tot')
        #plt.title('Location = '+str(self.ch_outp[key]['px'][xloc]/1000)+' km')
        plt.ylabel('$z/H$')
        plt.grid()
        plt.legend()
        plt.show()
        
        #project on Fourier modes
        def proj_four(T):
            return np.mean(T[:,np.newaxis] * np.cos(np.arange(1,self.M)*np.pi * self.z_nd[:,np.newaxis]),0)
        
        T1p,T2p,T3p,T5p,T6p,T7p = proj_four(T1),proj_four(T2),proj_four(T3),proj_four(T5),proj_four(T6),proj_four(T7)
        T10p, T11p, T12p = proj_four(T10here),proj_four(T11here),proj_four(T12here)
        
        plt.title(key)
        plt.plot(T1p/ self.soc_sca*self.Lsc,label='T1')
        plt.plot(T2p/ self.soc_sca*self.Lsc,label='T2')
        plt.plot(T3p/ self.soc_sca*self.Lsc,label='T3')
        plt.plot(T5p/ self.soc_sca*self.Lsc,label='T5')
        plt.plot(T6p/ self.soc_sca*self.Lsc,label='T6')
        plt.plot(T7p/ self.soc_sca*self.Lsc,label='T7')
        plt.plot((T10p+T11p+T12p)/ self.soc_sca*self.Lsc,label='Tt')
        
        plt.plot((T1p+T2p+T3p+T5p+T6p+T7p+T10p+T11p+T12p)/ self.soc_sca*self.Lsc,lw=3,c='black',label='tot')
        plt.grid()#, plt.legend()
        plt.show()

        
        
    #plot_vert(xloc)
    plot_vert_ground(xloc)
    #self.ch_tide['vertterms'](key,*conv_ans(self, key),self.bbb_n[key])
    #self.ch_tide['sol_i_cor'](key,self.bbb_n[key].flatten(),'loc x=-L','dv')
   

#plot_proc_vt(delta, 'Nieuwe Maas 1 old', 74)
#plot_proc_vt(delta, 'Nieuwe Waterweg v2',1)
#plot_proc_vt(delta, 'C1',166)
#plot_proc_vt(delta, 'C2',1)
#plot_proc_vt(delta, 'C3',1)




# =============================================================================
# bende
# =============================================================================


        

'''





#checks
#for x=-L
bbs = np.concatenate([np.real(self.bbb_n[key][0]),np.imag(self.bbb_n[key][0])])
Tcor = self.ch_tide['sol_i_cor'](key,bbs,'loc x=-L','dv')
print(Tcor.shape)
#bbs = np.concatenate([np.real(self.bbb_n[key][1]),np.imag(self.bbb_n[key][1])])
#Tcor = self.ch_tide['sol_i_cor'](key,bbs,'loc x=0','dv')

plt.plot(Tcor[:,xloc])
plt.plot(proj_four(T10c1[xloc]+T11c1[xloc]+T12c1[xloc]))
#plt.plot(self.c_save2[:,xloc])
#plt.plot(self.c_save[7])
#plt.plot(self.c_save3[7][5:10])
#plt.plot((self.c_save3[7].reshape((5,11)))[:,xloc])

plt.show()

Tc9 = []
for x in range(11):
    Tc9.append(proj_four(T10c1[x]+T11c1[x]+T12c1[x]))
Tc9 = np.array(Tc9).T*self.soc_sca/self.Lsc



plt.plot(proj_four(T10c1[xloc]+T11c1[xloc]+T12c1[xloc]))
plt.plot(Tc9[:,xloc])
plt.show()

Tc9f = Tc9.flatten()[:10]        
plt.plot(Tc9f)
plt.plot(self.c_save[7])
plt.show()
        
print(np.where(Tc9==Tc9f[0]))
print(np.where(Tc9==Tc9f[1]))
print(np.where(Tc9==Tc9f[2]))
print(np.where(Tc9==Tc9f[3]))
print(np.where(Tc9==Tc9f[4]))
print(np.where(Tc9==Tc9f[5]))
print(np.where(Tc9==Tc9f[6]))
print(np.where(Tc9==Tc9f[7]))



#print( proj_four(T10c1[xloc]) / Tcor[:,xloc] )

print(T1p/self.c_save[0])
print(T2p/self.c_save[1])
print(T3p/self.c_save[2])
print(T5p/self.c_save[3])
print(T6p/self.c_save[4])
print(T7p/self.c_save[5])
print(proj_four((T10+T11+T12)[0,xloc])* self.soc_sca/self.Lsc/(self.c_save[6]))

print()
print(proj_four((T10c1[xloc-1]+T11c1[xloc-1]+T12c1[xloc-1]))* self.soc_sca/self.Lsc/self.c_save[7])

#print()
#print(self.bbb_n[key][0])
'''
