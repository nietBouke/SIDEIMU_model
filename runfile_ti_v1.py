# =============================================================================
# Runfile general network in equilibrium
# =============================================================================
# Bouke, July 2023
import numpy as np
# =============================================================================
# import functions
# =============================================================================
from inputfile_v1 import input_network
from functions_all_v1 import network_funcs

# =============================================================================
# input
# =============================================================================
#gather the input at one place, that is here
#constants
g =9.81 #gravitation
Be=7.6e-4 #isohaline contractie
Sc=2.2 #Schmidt getal
cv=7.28e-5*1.5 #empirische constante
ch = None #0.35/10 #empirische constante
ch2= 50*5
CD = 0.001 #wind drag coefficient
r = 1.225/1000 #density air divided by density water

#gebruik ch2=325 en cv = 1.9e-4 voor nu

#tides
omega = 2*np.pi/44700
#only one of the coming two variables will actually be used
cv_t = 0.0013 #0.00144 according to Kris
Av_t = None
Kv_t = 0.02/2.2
Kh_t = 33
#only one of the coming two variables will be used
sf_t = 0.05 #0.06555 according to Kris
r_t = None #0.262 #1/4

N, Lsc, nt = 5, 1000 , 121

inp_const = (g,Be,Sc,cv,ch,ch2,CD,r,omega,cv_t,Av_t,Kv_t,sf_t,r_t,Kh_t,N,Lsc,nt)

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
delta = network_funcs(ch_gegs, inp_const, inp_phys)

#calculate river discharge distribution
delta.Qdist_calc()

#calculate tidal properties
delta.tide_calc()

#calculate salinity
init = delta.run_model_nt(init_nt)
hio = delta.run_model(init)

#convert raw output
delta.calc_output()

# =============================================================================
# Visualisation
# =============================================================================
#plot tide
#delta.plot_tide()

#delta.plot_tide_pointRM()
#delta.calc_tide_pointRM()
#plot salt and river discharge
delta.plot_Qs_simple()
#delta.plot_Qs_comp()
#delta.plot_proc_ch()
#delta.plot_proc_ch(channels = [ 'Nieuwe Maas 1 old','Nieuwe Maas 2 old','Oude Maas 2','Hartelkanaal v2', 'Nieuwe Waterweg v2', 'Breeddiep'])
#delta.plot_proc_ch(channels = [ 'Lek', 'Hollandse IJssel' ])
#delta.plot_proc_ch(channels = [ 'Spui' , 'Oude Maas 3'])
#delta.plot_junc_tid()
#animation of tidal behaviour
#delta.anim_tide_wl('hoho_v0')
#delta.anim_tide('hoho_v1')
#delta.plot_salt_ti_point()
delta.plot_contours(['Hartelkanaal v2'])

#plot details
#delta.plot_tide_timeseries()
delta.plot_new_compRM()
#delta.anim_new_compRM('RM_fm9')
#delta.plot_jinyang()

#do a calculation
#delta.max_min_salts()

#delta.plot_proc_vt('Nieuwe Waterweg v2',1)
#delta.plot_proc_vt('C1',-2)
#delta.plot_proc_vt('C2',1)
#delta.plot_proc_vt('C3',15)
#delta.plot_proc_vt('C3',-105)





