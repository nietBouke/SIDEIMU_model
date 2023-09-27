# =============================================================================
#  properties of the Kapaus delta 
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


def inp_Kapaus1_geo():
     #TODO: add width, depth, tidal current in this system
     # =============================================================================
     #     Input: geometry
     # =============================================================================
     ch_gegs = {}
 
     #get the length later from kml
     #TODO: add the right junctions
     ch_gegs['Jungkat'] = { 'loc x=0' : 's1' , 'loc x=-L': 'j1' } 
     ch_gegs['Kupah'] = { 'loc x=0' : 's2' , 'loc x=-L': 'j1' } 
     ch_gegs['Danau Pacat r2'] = { 'loc x=0' : 'j1' , 'loc x=-L': 'j2' }  
     ch_gegs['Danau Pacat r1'] = { 'loc x=0' : 'j2' , 'loc x=-L': 'r1' }  
     ch_gegs['Rasau'] = { 'loc x=0' : 'j3' , 'loc x=-L': 'j2' }  
     ch_gegs['Kapaus r1'] = { 'loc x=0' : 'j3' , 'loc x=-L': 'r2' }  
     ch_gegs['Kapaus r2'] = { 'loc x=0' : 'j4' , 'loc x=-L': 'j3' }  
     ch_gegs['Kapaus r3'] = { 'loc x=0' : 'j6' , 'loc x=-L': 'j4' }  
     ch_gegs['Putih'] = { 'loc x=0' : 'j5' , 'loc x=-L': 'j4' }  
     ch_gegs['Tanjung'] = { 'loc x=0' : 's7' , 'loc x=-L': 'j5' }  
     ch_gegs['Kempet'] = { 'loc x=0' : 'j5' , 'loc x=-L': 'j6' }  
     ch_gegs['Kapaus r4'] = { 'loc x=0' : 'j7' , 'loc x=-L': 'j6' }  
     ch_gegs['Kapaus r5'] = { 'loc x=0' : 'j8' , 'loc x=-L': 'j7' }  
     ch_gegs['Sungaikakip'] = { 'loc x=0' : 's3' , 'loc x=-L': 'j7' }  
     ch_gegs['Masjid'] = { 'loc x=0' : 's6' , 'loc x=-L': 'j8' }   
     ch_gegs['Kapaus r6'] = { 'loc x=0' : 'j9' , 'loc x=-L': 'j8' }  
     ch_gegs['Kakap r1'] = { 'loc x=0' : 'j10' , 'loc x=-L': 'j9' }   
     ch_gegs['Rumah r1'] = { 'loc x=0' : 'j11' , 'loc x=-L': 'j9' }   
     ch_gegs['Shodiqin'] = { 'loc x=0' : 'j11' , 'loc x=-L': 'j10' }   
     ch_gegs['Rumah r2'] = { 'loc x=0' : 's4' , 'loc x=-L': 'j11' }  
     ch_gegs['Kakap r2'] = { 'loc x=0' : 's5' , 'loc x=-L': 'j10' }   
 
     
     #add the properties I do not know
     for key in list(ch_gegs.keys()):
         ch_gegs[key]['H'] = 5 
         ch_gegs[key]['b'] = np.array([1000,1000], dtype=float)
         ch_gegs[key]['dx']= np.array([500], dtype=float)
         ch_gegs[key]['Ut']= 1 
         
     # =============================================================================
     # Input: plotting - maybe build seperate dictionary for this
     # =============================================================================
     path_Ka = '/Users/biemo004/Documents/UU phd Saltisolutions/Databestanden/Geo Kapaus/'
     list_channels = [f for f in os.listdir(path_Ka) if f.endswith('.kml')]
     Ka_coords = {}
     for i in range(len(list_channels)):
         Ka_coords[list_channels[i][:-4]] = np.array(kml_subtract_latlon(path_Ka+list_channels[i]),dtype=float)
         
     cs = ['r','olive','b','dimgray','c','orange','g','m','y','gold','indigo','tan','skyblue','maroon','peru','beige','plum','silver','forestgreen','darkkhaki','rosybrown']
     count=0
     for key in list(ch_gegs.keys()):
         ch_gegs[key]['plot x'] = np.flip(Ka_coords[key][:,0])
         ch_gegs[key]['plot y'] = np.flip(Ka_coords[key][:,1])
         ch_gegs[key]['plot color'] = cs[count]
         count+=1
         
     # =============================================================================
     #     #try to use kml files for length
     # =============================================================================
     p = pyproj.Proj(proj='utm',zone=49,ellps='WGS84') #to convert degrees to km. for appropriate zone see https://proj.org/operations/projections/utm.html#cmdoption-arg-zone
     for key in list(ch_gegs.keys()):
         #length of the channel
         dist = np.zeros(len(ch_gegs[key]['plot x']))
         for i in range(1,len(ch_gegs[key]['plot x'])):
             x0,y0 = p(longitude = ch_gegs[key]['plot x'][i],latitude=ch_gegs[key]['plot y'][i])
             x1,y1 = p(longitude = ch_gegs[key]['plot x'][i-1],latitude=ch_gegs[key]['plot y'][i-1])
             dist[i] = dist[i-1]+ ((x0-x1)**2 + (y0-y1)**2)**0.5
         ch_gegs[key]['L'] = np.array([np.round(dist[-1],-3)]) 
         #Name parameter
         ch_gegs[key]['Name'] = key
     
     ch_gegs['Kapaus r5']['L'] = np.array([400.])
     ch_gegs['Kapaus r5']['dx'] = np.array([100.])
         
     return ch_gegs
 
# =============================================================================
# physical properties of the Kapaus delta
# =============================================================================
def inp_Kapaus1_phys():
    Qriv = [1000,500] #mind the sign! this is the discharge at r1,r2,...
    Qweir= []
    Qhar = []
    n_sea = [0,0,0,0,0,0,0] #this is the water level at s1,s2,...
    soc = [35,35,35,35,35,35,35] #this is salinity at s1,s2, ...
    sri = [0,0] #this is the salinity of the river water at r1,r2, ...
    a_tide = [1.7,1.5,1,1,1,1,1] #this is the amplitude of the tide at s1,s2, ...
    p_tide = [0,0,0,0,0,0,0] #this is the phase of the tide at s1,s2, ...
    
    return Qriv,Qweir,Qhar, n_sea, soc, sri, a_tide, p_tide
