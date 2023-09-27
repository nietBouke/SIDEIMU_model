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


def inp_Po2_geo():
    
    ch_gegs = {}
    ch_gegs['Po 1'] = { 'Name' : 'Po 1' ,
               'H' : 5 ,
               'L' : np.array([40000,10000], dtype=float),
               'b' : np.array([375,375,375], dtype=float), #one longer than L
               'dx' : np.array([1000,1000], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               } 

    ch_gegs['Po di Goro'] =  { 'Name' : 'Po di Goro' ,
               'H' : 4 , 
               'L' : np.array([48500], dtype=float),
               'b' : np.array([125,150], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Po 2'] =  { 'Name' : 'Po 2' ,
               'H' : 5 ,               
               'L' : np.array([29200], dtype=float),
               'b' : np.array([375,375], dtype=float), #one longer than L
               'dx' : np.array([292], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Po della Donzella'] =  { 'Name' : 'Po della Donzella' ,
               'H' : 4,
               'L' : np.array([21300], dtype=float),
               'b' : np.array([125,125], dtype=float), #one longer than L
               'dx' : np.array([710], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Po di Venezia 1'] =  { 'Name' : 'Po di Venezia 1' ,
               'H' : 5 ,
               'L' : np.array([2100], dtype=float),
               'b' : np.array([375,425], dtype=float), #one longer than L
               'dx' : np.array([210], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Po di Maistra'] =  { 'Name' : 'Po di Maistra' ,
               'H' : 3 ,
               'L' : np.array([17400], dtype=float),
               'b' : np.array([50,75], dtype=float), #one longer than L
               'dx' : np.array([580], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 's3' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Po di Venezia 2'] =  { 'Name' : 'Po di Venezia 2' ,
               'H' : 5 ,               
               'L' : np.array([8700], dtype=float),
               'b' : np.array([425,550], dtype=float), #one longer than L
               'dx' : np.array([435], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Po della Pila'] =  { 'Name' : 'Po della Pila' ,
               'H' : 4 ,
               'L' : np.array([9500], dtype=float),
               'b' : np.array([375,550], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 's4' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Po delle Tolle'] =  { 'Name' : 'Po delle Tolle' ,
               'H' : 3 ,               
               'L' : np.array([10900], dtype=float),
               'b' : np.array([250,275], dtype=float), #one longer than L
               'dx' : np.array([100], dtype=float), #same length as L
               'Ut' : 0.5 ,
               'loc x=0' : 's5' ,
               'loc x=-L': 'j4'
               }


    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================

    path_Po = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_Po/'
    list_channels = [f for f in os.listdir(path_Po) if f.endswith('.kml')]
    Po_coords = {}
    for i in range(len(list_channels)):
        Po_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_Po+list_channels[i]),dtype=float)

    cs = ['r','olive','b','black','c','orange','g','m','y']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = Po_coords[key][:,0]
        ch_gegs[key]['plot y'] = Po_coords[key][:,1]
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

# =============================================================================
# physical properties of the delta
# =============================================================================
def inp_Po2_phys():
    Qriv = [250] #mind the sign! this is the discharge at r1,r2,...
    n_sea = [0,0,0,0,0] #this is the water level at s1,s2,...
    soc = [35,35,35,35,35] #this is salinity at s1,s2, ...
    sri = [0] #this is the salinity of the river water at r1,r2, ...
    a_tide = [0.7,0.6,0.5,0.5,0.7] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,0,0,0,0] #this is the phase of the tide at s1,s2, ...
    
    return Qriv, n_sea, soc, sri, a_tide, p_tide

def inp_Po3_phys():
    Qriv = [250] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0,0,0,0] #this is the water level at s1,s2,...
    soc = [35,35,35,35,35] #this is salinity at s1,s2, ...
    sri = [0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [0.7,0.6,0.5,0.5,0.7] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,0,0,0,0] #this is the phase of the tide at s1,s2, ...
    
    return Qriv, Qweir, Qhar, n_sea, soc, sri, swe, a_tide, p_tide




