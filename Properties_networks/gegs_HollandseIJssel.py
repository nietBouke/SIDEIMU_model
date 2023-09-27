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
    


def inp_HollandseIJssel1_geo():

    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Nieuwe Maas'] = { 'Name' : 'Nieuwe Maas' ,
               'H' : 11 ,
               'L' : np.array([19000], dtype=float),
               'b' : np.array([400,500], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               } 

    ch_gegs['Lek'] = { 'Name' : 'Lek' ,
               'H' : 8,
               'L' : np.array([47000], dtype=float),
               'b' : np.array([150,400], dtype=float), #one longer than L
               'dx' : np.array([200], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w1'
               }

    ch_gegs['Hollandse IJssel'] = { 'Name' : 'Hollandse IJssel' ,
               'H' : 5 ,
               'L' : np.array([19000], dtype=float),
               'b' : np.array([70,110], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w2'
               }
    
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['Nieuwe Maas']['plot x'] = np.linspace(0,-ch_gegs['Nieuwe Maas']['L'].sum(),101)
    ch_gegs['Nieuwe Maas']['plot y'] = np.zeros(101)
    ch_gegs['Nieuwe Maas']['plot color'] = 'r'

    ch_gegs['Lek']['plot x'] = np.linspace(ch_gegs['Lek']['L'].sum(),0,101)
    ch_gegs['Lek']['plot y'] = -np.linspace(ch_gegs['Lek']['L'].sum(),0,101)
    ch_gegs['Lek']['plot color'] = 'b'

    ch_gegs['Hollandse IJssel']['plot x'] = np.linspace(ch_gegs['Hollandse IJssel']['L'].sum(),0,101)
    ch_gegs['Hollandse IJssel']['plot y'] = np.linspace(ch_gegs['Hollandse IJssel']['L'].sum(),0,101)
    ch_gegs['Hollandse IJssel']['plot color'] = 'black'

    return ch_gegs

# =============================================================================
# physical properties of the delta
# =============================================================================

def inp_HollandseIJssel1_phys():
    Qriv = [] #mind the sign! this is the discharge at r1,r2,...
    Qweir= [300,15]
    Qhar = []
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [25] #this is salinity at s1,s2, ...
    sri = [] #this is the salinity of the river water at r1,r2, ...
    swe = [0,0]
    a_tide = [0.7]#this is the amplitude of the tide at s1,s2, ...
    p_tide = [0] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide

