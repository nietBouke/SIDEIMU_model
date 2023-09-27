# =============================================================================
#  properties of the delta 
# =============================================================================
import numpy as np
from bs4 import BeautifulSoup
import os
import pyproj 

def kml_subtract_latlon(infile):
    with open(infile, 'r') as f:
        s = BeautifulSoup(f, 'xml')
        for coords in s.find_all('coordinates'):
            # Take coordinate string from KML and break it up into [Lat,Lon,Lat,Lon...] to get CSV row
            space_splits = coords.string.split(" ")
            row = []

            for split in space_splits[:-1]:
                # Note: because of the space between <coordinates>" "-80.123, we slice [1:]
                comma_split = split.split(',')
                # longitude,lattitude,
                row.append([comma_split[0],comma_split[1]])
    row[0][0] = row[0][0][5:]
    return row
    
def inp_NW_NM_OM_1_geo():

    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Nieuwe Waterweg'] = { 'Name' : 'Nieuwe Waterweg' ,
               'H' : 16 ,
               'L' : np.array([18000], dtype=float),
               'b' : np.array([600,600], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               } 

    ch_gegs['Nieuwe Maas'] = { 'Name' : 'Nieuwe Maas' ,
               'H' : 14-3,
               'L' : np.array([80000,20000], dtype=float),
               'b' : np.array([300,350,450], dtype=float), #one longer than L
               'dx' : np.array([1000,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['Oude Maas'] = { 'Name' : 'Oude Maas' ,
               'H' : 13 ,
               'L' : np.array([80000,20000], dtype=float),
               'b' : np.array([300,300,300], dtype=float), #one longer than L
               'dx' : np.array([1000,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r2'
               }
    
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['Nieuwe Waterweg']['plot x'] = np.linspace(0,-ch_gegs['Nieuwe Waterweg']['L'].sum(),101)
    ch_gegs['Nieuwe Waterweg']['plot y'] = np.zeros(101)
    ch_gegs['Nieuwe Waterweg']['plot color'] = 'r'

    ch_gegs['Nieuwe Maas']['plot x'] = np.linspace(ch_gegs['Nieuwe Maas']['L'].sum(),0,101)
    ch_gegs['Nieuwe Maas']['plot y'] = np.linspace(ch_gegs['Nieuwe Maas']['L'].sum(),0,101)
    ch_gegs['Nieuwe Maas']['plot color'] = 'b'

    ch_gegs['Oude Maas']['plot x'] = np.linspace(ch_gegs['Oude Maas']['L'].sum(),0,101)
    ch_gegs['Oude Maas']['plot y'] = -np.linspace(ch_gegs['Oude Maas']['L'].sum(),0,101)
    ch_gegs['Oude Maas']['plot color'] = 'black'

    return ch_gegs


# =============================================================================
# physical properties of the delta
# =============================================================================
def inp_NW_NM_OM_1_phys():
    Qriv = [500,1000] #mind the sign! this is the discharge at r1,r2,...
    Qweir = [] #mind the sign! this is the discharge at r1,r2,...
    Qhar = [] #mind the sign! this is the discharge at r1,r2,...
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [30] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [0.83] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [70/180*np.pi] #this is the phase of the tide at s1,s2, ...

    return Qriv,Qweir,Qhar, n_sea, soc, sri, swe, a_tide, p_tide
