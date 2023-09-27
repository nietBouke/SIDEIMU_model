# =============================================================================
# Runfile general network in equilibrium
# =============================================================================
# Bouke, July 2023
import numpy as np
# =============================================================================
# import functions
# =============================================================================
from inputfile_v7 import input_network
from functions_all_v7 import network_funcs

# =============================================================================
# input
# =============================================================================
#gather the input at one place, that is here
#constants
g =9.81 #gravitation
Be=7.6e-4 #isohaline contractie
Sc=2.2 #Schmidt getal
cv=7.28e-5*2#*1.5 #empirische constante
ch = None #0.35/10 #empirische constante
ch2= 50#*2
CD = 0.001 #wind drag coefficient
r = 1.225/1000 #density air divided by density water

#gebruik ch2=325 en cv = 1.9e-4 voor nu

#tides
omega = 2*np.pi/44700
#only one of the coming two variables will actually be used
cv_t = None #0.0013 #0.00144 according to Kris
Av_t = 0.02
Kv_t = 0.02/2.2
#only one of the coming two variables will be used
sf_t = 0.002 #0.06555 according to Kris
r_t = None #0.262 #1/4

N, Lsc, nt = 5, 1000 , 121

inp_const = (g,Be,Sc,cv,ch,ch2,CD,r,omega,cv_t,Av_t,Kv_t,sf_t,r_t,N,Lsc,nt)

#options 
mod_opt_tide = True #built
mod_opt_wt   = True #does not work 
mod_opt_bl   = True #seems to work 
mod_opt_vert = True #built
mod_opt = (mod_opt_tide, mod_opt_wt, mod_opt_bl, mod_opt_vert)


#note that only partial slip is possible in this code. Some minor changes can be done to incorporate this
#geometry
#ch_gegs = input_network.inp_deltatest2_geo()
#ch_gegs = input_network.inp_HollandseIJssel1_geo()
ch_gegs = input_network.inp_RijnMaas0_highres_geo()
#ch_gegs = input_network.inp_hooghly3_geo()

#physcial parameters
#inp_phys = input_network.inp_deltatest4_phys()
#inp_phys = input_network.inp_HollandseIJssel1_phys()
inp_phys = input_network.inp_RijnMaas2_phys()
#inp_phys = input_network.inp_hooghly1_phys()

# =============================================================================
# Initialisation salt field
# =============================================================================
init_nt= np.array([None]) #none means no initialisation from outside, code starts with zeros
# =============================================================================
# calculations
# =============================================================================

#create object, and load all the input into the model
delta = network_funcs(ch_gegs, inp_const, inp_phys, mod_opt)

#print(delta.ch_gegs)

for key in sorted(delta.ch_keys):
    print(delta.ch_gegs[key]['Name'],
          ' & ',
          delta.ch_gegs[key]['H'] ,          
          ' & ',
          delta.ch_gegs[key]['L']/1000 ,          
          ' & ',
          delta.ch_gegs[key]['b'] ,  
          ' & ',
          delta.ch_gegs[key]['dx'] ,  
          r' & \\'
          )
print('\hline')


