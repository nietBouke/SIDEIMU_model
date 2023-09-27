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


def inp_hooghly1_geo():
    # =============================================================================
    #     Input: geometry
    # sources depth: https://www.researchgate.net/publication/330023472_Study_on_Maintenance_Dredging_for_Navigable_Depth_Assurance_in_the_Macro-tidal_Hooghly_Estuary_Volume_2
    
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Mooriganga'] = { 'Name' : 'Mooriganga' ,
               'H' : 7 ,
               'L' : np.array([35600], dtype=float),
               'b' : np.array([3000,5000], dtype=float), #one longer than L
               'dx' : np.array([356], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Hooghly 1'] = { 'Name' : 'Hooghly 1' ,
               'H' : 8,
               'L' : np.array([30000], dtype=float),
               'b' : np.array([13000,27500], dtype=float), #one longer than L
               'dx' : np.array([300], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Hooghly 2'] = { 'Name' : 'Hooghly 2' ,
               'H' : 8 ,
               'L' : np.array([8600], dtype=float),
               'b' : np.array([5000,9000], dtype=float), #one longer than L
               'dx' : np.array([200], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Hooghly 3'] = { 'Name' : 'Hooghly 3' ,
               'H' : 8 ,
               'L' : np.array([18250], dtype=float),
               'b' : np.array([4500,5000], dtype=float), #one longer than L
               'dx' : np.array([250], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Hooghly 4'] = { 'Name' : 'Hooghly 4' ,
               'H' : 10 ,
               'L' : np.array([24000], dtype=float),
               'b' : np.array([2100,4500], dtype=float), #TODO: make this more sophisticated
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Hooghly Main'] = { 'Name' : 'Hooghly Main' ,
               'H' : 6 ,
               'L' : np.array([118000], dtype=float),
               'b' : np.array([500,2000], dtype=float), 
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               }
 
    ch_gegs['Rupnarayan River'] = { 'Name' : 'Rupnarayan River' ,
               'H' : 5 ,
               'L' : np.array([47000], dtype=float),
               'b' : np.array([400,1500], dtype=float), 
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r2'
               }
 
    ch_gegs['Haldia'] = { 'Name' : 'Haldia' ,
               'H' : 9 ,
               'L' : np.array([14500], dtype=float),
               'b' : np.array([1300,2250], dtype=float), 
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j5' ,
               'loc x=-L': 'j2'
               }
 
    ch_gegs['Haldi seaward'] = { 'Name' : 'Haldi seaward' ,
               'H' : 7 ,
               'L' : np.array([11000], dtype=float),
               'b' : np.array([2800,4200], dtype=float),
               'dx' : np.array([200], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j5'
               }
 
 
    ch_gegs['Haldi Main'] = { 'Name' : 'Haldi Main' ,
               'H' : 3 ,
               'L' : np.array([47000], dtype=float),
               'b' : np.array([170,1150], dtype=float), 
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j5' ,
               'loc x=-L': 'j6'
               }
 
 
    ch_gegs['Haldi North'] = { 'Name' : 'Haldi North' ,
               'H' : 2 ,
               'L' : np.array([21500], dtype=float),
               'b' : np.array([40,100], dtype=float), 
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j6' ,
               'loc x=-L': 'r3'
               }
 
    ch_gegs['Haldi South'] = { 'Name' : 'Haldi South' ,
               'H' : 2 ,
               'L' : np.array([20500], dtype=float),
               'b' : np.array([30,120], dtype=float), 
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j6' ,
               'loc x=-L': 'r4'
               }

    # =============================================================================
    # Input: plotting
    # =============================================================================
    path_RR = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_Hooghly/'
    list_channels = [f for f in os.listdir(path_RR) if f.endswith('.kml')]
    RR_coords = {}
    for i in range(len(list_channels)):
        RR_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RR+list_channels[i]),dtype=float)
        
    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
      'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RR_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RR_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def inp_hooghly2_geo():
    # =============================================================================
    #     Input: geometry
    # sources depth: https://www.researchgate.net/publication/330023472_Study_on_Maintenance_Dredging_for_Navigable_Depth_Assurance_in_the_Macro-tidal_Hooghly_Estuary_Volume_2
    
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Mooriganga'] = { 'Name' : 'Mooriganga' ,
               'H' : 7 ,
               'L' : np.array([35000], dtype=float),
               'b' : np.array([3000,5000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Hooghly 1'] = { 'Name' : 'Hooghly 1' ,
               'H' : 8,
               'L' : np.array([30000], dtype=float),
               'b' : np.array([13000,27500], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Hooghly 2'] = { 'Name' : 'Hooghly 2' ,
               'H' : 8 ,
               'L' : np.array([8600], dtype=float),
               'b' : np.array([5000,9000], dtype=float), #one longer than L
               'dx' : np.array([860], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Hooghly 3'] = { 'Name' : 'Hooghly 3' ,
               'H' : 8 ,
               'L' : np.array([18000], dtype=float),
               'b' : np.array([4500,5000], dtype=float), #one longer than L
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Hooghly 4'] = { 'Name' : 'Hooghly 4' ,
               'H' : 10 ,
               'L' : np.array([24000], dtype=float),
               'b' : np.array([2100,4500], dtype=float), #TODO: make this more sophisticated
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Hooghly Main'] = { 'Name' : 'Hooghly Main' ,
               'H' : 6 ,
               'L' : np.array([118000], dtype=float),
               'b' : np.array([500,2000], dtype=float), 
               'dx' : np.array([2000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               }
 
    ch_gegs['Rupnarayan River'] = { 'Name' : 'Rupnarayan River' ,
               'H' : 5 ,
               'L' : np.array([48000], dtype=float),
               'b' : np.array([400,1500], dtype=float), 
               'dx' : np.array([2000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r2'
               }
 
    ch_gegs['Haldia'] = { 'Name' : 'Haldia' ,
               'H' : 9 ,
               'L' : np.array([14000], dtype=float),
               'b' : np.array([1300,2250], dtype=float), 
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j5' ,
               'loc x=-L': 'j2'
               }
 
    ch_gegs['Haldi seaward'] = { 'Name' : 'Haldi seaward' ,
               'H' : 7 ,
               'L' : np.array([11000], dtype=float),
               'b' : np.array([2800,4200], dtype=float),
               'dx' : np.array([1000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j5'
               }
 
 
    ch_gegs['Haldi Main'] = { 'Name' : 'Haldi Main' ,
               'H' : 3 ,
               'L' : np.array([48000], dtype=float),
               'b' : np.array([170,1150], dtype=float), 
               'dx' : np.array([2000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j5' ,
               'loc x=-L': 'j6'
               }
 
 
    ch_gegs['Haldi North'] = { 'Name' : 'Haldi North' ,
               'H' : 2 ,
               'L' : np.array([22000], dtype=float),
               'b' : np.array([40,100], dtype=float), 
               'dx' : np.array([2000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j6' ,
               'loc x=-L': 'r3'
               }
 
    ch_gegs['Haldi South'] = { 'Name' : 'Haldi South' ,
               'H' : 2 ,
               'L' : np.array([20000], dtype=float),
               'b' : np.array([30,120], dtype=float), 
               'dx' : np.array([2000], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j6' ,
               'loc x=-L': 'r4'
               }

    # =============================================================================
    # Input: plotting
    # =============================================================================
    path_RR = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_Hooghly/'
    list_channels = [f for f in os.listdir(path_RR) if f.endswith('.kml')]
    RR_coords = {}
    for i in range(len(list_channels)):
        RR_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RR+list_channels[i]),dtype=float)
        
    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
      'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RR_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RR_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs

def inp_hooghly3_geo():
    # =============================================================================
    #     Input: geometry
    # sources depth: https://www.researchgate.net/publication/330023472_Study_on_Maintenance_Dredging_for_Navigable_Depth_Assurance_in_the_Macro-tidal_Hooghly_Estuary_Volume_2
    
    # =============================================================================
    ch_gegs = {}
    ch_gegs['Mooriganga'] = { 'Name' : 'Mooriganga' ,
               'H' : 7 ,
               'L' : np.array([4000,31600], dtype=float),
               'b' : np.array([3000,3000*(5/3)**(4000/35600),5000], dtype=float), #one longer than L
               'dx' : np.array([100,400], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's2' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Hooghly 1'] = { 'Name' : 'Hooghly 1' ,
               'H' : 8,
               'L' : np.array([30000], dtype=float),
               'b' : np.array([13000,27500], dtype=float), #one longer than L
               'dx' : np.array([300], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 's1' ,
               'loc x=-L': 'j4'
               }

    ch_gegs['Hooghly 2'] = { 'Name' : 'Hooghly 2' ,
               'H' : 8 ,
               'L' : np.array([1500,5600,1500], dtype=float),
               'b' : np.array([5000,5500,8000,9000], dtype=float), #one longer than L
               'dx' : np.array([50,100,50], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j3'
               }

    ch_gegs['Hooghly 3'] = { 'Name' : 'Hooghly 3' ,
               'H' : 8 ,
               'L' : np.array([2000,14250,2000], dtype=float),
               'b' : np.array([4500,4600,4900,5000], dtype=float), #one longer than L
               'dx' : np.array([100,250,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j3' ,
               'loc x=-L': 'j2'
               }

    ch_gegs['Hooghly 4'] = { 'Name' : 'Hooghly 4' ,
               'H' : 10 ,
               'L' : np.array([2000,20000,2000], dtype=float),
               'b' : np.array([2100,2200,4300,4500], dtype=float), #TODO: make this more sophisticated
               'dx' : np.array([200,500,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j2' ,
               'loc x=-L': 'j1'
               }

    ch_gegs['Hooghly Main'] = { 'Name' : 'Hooghly Main' ,
               'H' : 6 ,
               'L' : np.array([113000,5000], dtype=float),
               'b' : np.array([500,1800,2000], dtype=float), 
               'dx' : np.array([1000,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r1'
               }
 
    ch_gegs['Rupnarayan River'] = { 'Name' : 'Rupnarayan River' ,
               'H' : 5 ,
               'L' : np.array([43000,2000], dtype=float),
               'b' : np.array([400,1400,1500], dtype=float), 
               'dx' : np.array([500,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j1' ,
               'loc x=-L': 'r2'
               }
 
    ch_gegs['Haldia'] = { 'Name' : 'Haldia' ,
               'H' : 9 ,
               'L' : np.array([2000,14500,2000], dtype=float),
               'b' : np.array([1300,1500,2000,2250], dtype=float), 
               'dx' : np.array([100,500,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j5' ,
               'loc x=-L': 'j2'
               }
 
    ch_gegs['Haldi seaward'] = { 'Name' : 'Haldi seaward' ,
               'H' : 7 ,
               'L' : np.array([11000], dtype=float),
               'b' : np.array([2800,4200], dtype=float),
               'dx' : np.array([200], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j4' ,
               'loc x=-L': 'j5'
               }
 
 
    ch_gegs['Haldi Main'] = { 'Name' : 'Haldi Main' ,
               'H' : 3 ,
               'L' : np.array([44000,3000], dtype=float),
               'b' : np.array([170,1000,1150], dtype=float), 
               'dx' : np.array([500,100], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j5' ,
               'loc x=-L': 'j6'
               }
 
 
    ch_gegs['Haldi North'] = { 'Name' : 'Haldi North' ,
               'H' : 2 ,
               'L' : np.array([21500], dtype=float),
               'b' : np.array([40,100], dtype=float), 
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j6' ,
               'loc x=-L': 'r3'
               }
 
    ch_gegs['Haldi South'] = { 'Name' : 'Haldi South' ,
               'H' : 2 ,
               'L' : np.array([20500], dtype=float),
               'b' : np.array([30,120], dtype=float), 
               'dx' : np.array([500], dtype=float), #same length as L
               'Ut' : 1 ,
               'loc x=0' : 'j6' ,
               'loc x=-L': 'r4'
               }
    
    #for key in list(ch_gegs.keys()):
    #   ch_gegs[key]['H'] = 7
        
    # =============================================================================
    # Input: plotting
    # =============================================================================
    path_RR = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo_Hooghly/'
    list_channels = [f for f in os.listdir(path_RR) if f.endswith('.kml')]
    RR_coords = {}
    for i in range(len(list_channels)):
        RR_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_RR+list_channels[i]),dtype=float)
        
    cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru',\
      'beige','plum','silver','forestgreen','darkkhaki','rosybrown','teal','snow','aliceblue','mintcream']
    count=0
    for key in list(ch_gegs.keys()):
        ch_gegs[key]['plot x'] = np.flip(RR_coords[key][:,0])
        ch_gegs[key]['plot y'] = np.flip(RR_coords[key][:,1])
        ch_gegs[key]['plot color'] = cs[count]
        count+=1

    return ch_gegs


# =============================================================================
# physical properties of the delta
# =============================================================================

def inp_hooghly0_phys():
    #Qriv = [1500,300,100,100] #mind the sign! this is the discharge at r1,r2,...
    Qriv = [1500/3,300/3,100/3,100/3] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [30,30] #this is salinity at s1,s2, ...
    sri = [0,0,0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [1.388,1.388] # at Dadanpatra #this is the amplitude of the tide at s1,s2, ...
    #a_tide = [0.1,0.1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,20/180*np.pi] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide

def inp_hooghly1_phys():
    #Qriv = [1500,300,100,100] #mind the sign! this is the discharge at r1,r2,...
    Qriv = [1500,300,100,100] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0] #this is the water level at s1,s2,...
    soc = [30,30] #this is salinity at s1,s2, ...
    sri = [0,0,0,0] #this is the salinity of the river water at r1,r2, ...
    swe = []
    a_tide = [1.388,1.388] # at Dadanpatra #this is the amplitude of the tide at s1,s2, ...
    #a_tide = [0.1,0.1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,20/180*np.pi] #this is the phase of the tide at s1,s2, ...

    return Qriv, Qweir, Qhar,n_sea, soc, sri, swe, a_tide, p_tide
