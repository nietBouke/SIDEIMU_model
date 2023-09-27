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
    
def inp_deltatest1_geo():

    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['C1'] = { 'Name' : 'C1' ,
               'H' : 10 ,
               'L' : np.array([50000], dtype=float),
               'b' : np.array([1000,1000], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               } 

    ch_gegs['C2'] = { 'Name' : 'C2' ,
               'H' : 11,
               'L' : np.array([80000,20000], dtype=float),
               'b' : np.array([800,800,1200], dtype=float), #one longer than L
               'dx' : np.array([1000,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['C3'] = { 'Name' : 'C3' ,
               'H' : 7 ,
               'L' : np.array([80000,20000], dtype=float),
               'b' : np.array([200,200,200], dtype=float), #one longer than L
               'dx' : np.array([1000,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r2'
               }
    
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['C1']['plot x'] = np.linspace(0,-ch_gegs['C1']['L'].sum(),101)
    ch_gegs['C1']['plot y'] = np.zeros(101)
    ch_gegs['C1']['plot color'] = 'r'

    ch_gegs['C2']['plot x'] = np.linspace(ch_gegs['C2']['L'].sum(),0,101)
    ch_gegs['C2']['plot y'] = np.linspace(ch_gegs['C2']['L'].sum(),0,101)
    ch_gegs['C2']['plot color'] = 'b'

    ch_gegs['C3']['plot x'] = np.linspace(ch_gegs['C3']['L'].sum(),0,101)
    ch_gegs['C3']['plot y'] = -np.linspace(ch_gegs['C3']['L'].sum(),0,101)
    ch_gegs['C3']['plot color'] = 'black'

    return ch_gegs

def inp_deltatest2_geo():
    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['C1'] = { 'Name' : 'C1' ,
               'H' : 10 ,
               'L' : np.array([100000,50000], dtype=float),
               'b' : np.array([1000,2000,2000], dtype=float), #one longer than L
               'dx' : np.array([4000,2000/4], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               } 

    ch_gegs['C2'] = { 'Name' : 'C2' ,
               'H' : 10,
               'L' : np.array([40000], dtype=float),
               'b' : np.array([1000,1000], dtype=float), #one longer than L
               'dx' : np.array([1000/4], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               }
    
    ch_gegs['C3'] = { 'Name' : 'C2' ,
               'H' : 10,
               'L' : np.array([40000], dtype=float),
               'b' : np.array([1000,1000], dtype=float), #one longer than L
               'dx' : np.array([1000/4], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j1'
               }
    '''
    ch_gegs['C3'] = { 'Name' : 'C3' ,
               'H' : 10 ,
               'L' : np.array([10000,12000,9000], dtype=float),
               'b' : np.array([500,500,500,500], dtype=float), #one longer than L
               'dx' : np.array([1000/5,1000/5,1000/5], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j1'
               }
    '''
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['C1']['plot x'] = np.linspace(-ch_gegs['C1']['L'].sum(),0,101)
    ch_gegs['C1']['plot y'] = np.zeros(101)
    ch_gegs['C1']['plot color'] = 'r'

    ch_gegs['C2']['plot x'] = np.linspace(100,ch_gegs['C2']['L'].sum()+100,101)
    ch_gegs['C2']['plot y'] = np.linspace(100,ch_gegs['C2']['L'].sum()+100,101)
    ch_gegs['C2']['plot color'] = 'b'

    ch_gegs['C3']['plot x'] = np.linspace(100,ch_gegs['C3']['L'].sum()+100,101)
    ch_gegs['C3']['plot y'] = -np.linspace(100,ch_gegs['C3']['L'].sum()+100,101)
    ch_gegs['C3']['plot color'] = 'black'

    #ch_gegs['C3']['plot x'] = np.flip(ch_gegs['C3']['plot x'])
    #ch_gegs['C3']['plot y'] = np.flip(ch_gegs['C3']['plot y'])

    return ch_gegs

def inp_deltatest3_geo():

    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['C1'] = { 'Name' : 'C1' ,
               'H' : 10 ,
               'L' : np.array([100000], dtype=float),
               'b' : np.array([1000,1000], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               } 

    ch_gegs['C2'] = { 'Name' : 'C2' ,
               'H' : 10 ,
               'L' : np.array([50000], dtype=float),
               'b' : np.array([500,500], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['C3'] = { 'Name' : 'C3' ,
               'H' : 10 ,
               'L' : np.array([50000], dtype=float),
               'b' : np.array([500,500], dtype=float), #one longer than L
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r2'
               }
   
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['C1']['plot x'] = np.linspace(-ch_gegs['C1']['L'].sum(),0,101)
    ch_gegs['C1']['plot y'] = np.zeros(101)
    ch_gegs['C1']['plot color'] = 'r'

    ch_gegs['C2']['plot x'] = np.linspace(0,ch_gegs['C2']['L'].sum(),101)
    ch_gegs['C2']['plot y'] = np.zeros(101)
    ch_gegs['C2']['plot color'] = 'b'

    ch_gegs['C3']['plot x'] = np.zeros(101)
    ch_gegs['C3']['plot y'] = -np.linspace(0,ch_gegs['C3']['L'].sum(),101)
    ch_gegs['C3']['plot color'] = 'black'

    #ch_gegs['C3']['plot x'] = np.flip(ch_gegs['C3']['plot x'])
    #ch_gegs['C3']['plot y'] = np.flip(ch_gegs['C3']['plot y'])

    return ch_gegs

def inp_deltatest4_geo():

    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['C1'] = { 'Name' : 'C1' ,
               'H' : 10 ,
               'L' : np.array([100000,25000], dtype=float),
               'b' : np.array([2000,2000,3000], dtype=float), #one longer than L
               'dx' : np.array([1000,500], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               } 

    ch_gegs['C2'] = { 'Name' : 'C2' ,
               'H' : 10 ,
               'L' : np.array([10000,20000], dtype=float),
               'b' : np.array([1000,2000,1000], dtype=float), #one longer than L
               'dx' : np.array([1000,500], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['C3'] = { 'Name' : 'C3' ,
               'H' : 10 ,
               'L' : np.array([20000,20000,15000], dtype=float),
               'b' : np.array([2000,4000,5000,5000], dtype=float), #one longer than L
               'dx' : np.array([1000,500,500], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               }
   
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['C1']['plot x'] = np.linspace(-ch_gegs['C1']['L'].sum(),0,101)
    ch_gegs['C1']['plot y'] = np.zeros(101)
    ch_gegs['C1']['plot color'] = 'r'

    ch_gegs['C2']['plot x'] = np.linspace(0,ch_gegs['C2']['L'].sum(),101)
    ch_gegs['C2']['plot y'] = np.zeros(101)
    ch_gegs['C2']['plot color'] = 'b'

    ch_gegs['C3']['plot x'] = np.zeros(101)
    ch_gegs['C3']['plot y'] = -np.linspace(0,ch_gegs['C3']['L'].sum(),101)
    ch_gegs['C3']['plot color'] = 'black'

    #ch_gegs['C3']['plot x'] = np.flip(ch_gegs['C3']['plot x'])
    #ch_gegs['C3']['plot y'] = np.flip(ch_gegs['C3']['plot y'])

    return ch_gegs

def inp_deltatest5_geo():

    # =============================================================================
    #     Input: geometry
    # =============================================================================
    ch_gegs = {}
    ch_gegs['C1'] = { 'Name' : 'C1' ,
               'H' : 11 ,
               'L' : np.array([55000,5000], dtype=float),
               'b' : np.array([500,500,500], dtype=float), #one longer than L
               'dx' : np.array([500,50], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'w1'
               } 

    ch_gegs['C2'] = { 'Name' : 'C2' ,
               'H' : 12 ,
               'L' : np.array([55000,5000], dtype=float),
               'b' : np.array([300,300,300], dtype=float), #one longer than L
               'dx' : np.array([500,50], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               }

    ch_gegs['C3'] = { 'Name' : 'C3' ,
               'H' : 10 ,
               'L' : np.array([5000,15000], dtype=float),
               'b' : np.array([600,600,600], dtype=float), #one longer than L
               'dx' : np.array([50,250], dtype=float), #same length as L
               'Ut' : 1. ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j1'
               }
   
    # =============================================================================
    # Input: plotting - maybe build seperate dictionary for this
    # =============================================================================
    ch_gegs['C1']['plot x'] = np.linspace(-ch_gegs['C1']['L'].sum(),0,101)
    ch_gegs['C1']['plot y'] = np.zeros(101)
    ch_gegs['C1']['plot color'] = 'r'

    ch_gegs['C2']['plot x'] = np.linspace(0,ch_gegs['C2']['L'].sum(),101)
    ch_gegs['C2']['plot y'] = np.zeros(101)
    ch_gegs['C2']['plot color'] = 'b'

    ch_gegs['C3']['plot x'] = np.zeros(101)
    ch_gegs['C3']['plot y'] = -np.linspace(0,ch_gegs['C3']['L'].sum(),101)
    ch_gegs['C3']['plot color'] = 'black'

    #ch_gegs['C3']['plot x'] = np.flip(ch_gegs['C3']['plot x'])
    #ch_gegs['C3']['plot y'] = np.flip(ch_gegs['C3']['plot y'])

    return ch_gegs

# =============================================================================
# physical properties of the delta
# =============================================================================
def inp_deltatest1_phys():
    Qriv = [200] #mind the sign! this is the discharge at r1,r2,...
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [35,35] #this is salinity at s1,s2, ...
    sri = [0] #this is the salinity of the river water at r1,r2, ...
    a_tide = [1,1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,0] #this is the phase of the tide at s1,s2, ...

    return Qriv, n_sea, soc, sri, a_tide, p_tide

def inp_deltatest2_phys():
    Qriv = [100] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [35,35] #this is salinity at s1,s2, ...
    sri = [0] #this is the salinity of the river water at r1,r2, ...
    a_tide = [1,1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,0] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, a_tide, p_tide

def inp_deltatest3_phys():
    Qriv = [125,125] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [35] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [2] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide

def inp_deltatest4_phys():
    Qriv = [200] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [35,35] #this is salinity at s1,s2, ...
    sri = [0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [1.5,1.5]#this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,1] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide

def inp_deltatest5_phys():
    Qriv = [200,150] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0] #this is the water level at s1,s2,...
    soc = [35] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [1.5] #this is the amplitude of the tide at s1,s2, ...
    #a_tide = [0.1,0.1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide

def inp_deltatest6_phys():
    Qriv = [200,300] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [35,35] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [1.5,1] #this is the amplitude of the tide at s1,s2, ...
    #a_tide = [0.1,0.1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,0] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide

