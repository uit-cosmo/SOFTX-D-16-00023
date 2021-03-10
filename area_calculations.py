import numpy as  np 

def generate_area_array(): 
    area_lat = np.genfromtxt('EBM/output/area.dat', dtype=float)
    area_pole = area_lat[0] # area of pole
    area_lat = area_lat[1:-1] # area without north- and southpole
    area_wo_poles = np.repeat(area_lat, 128) # repeat the array 128 times because every latitude has 128 longitude cells
    
    return area_pole, area_wo_poles

def test_area_earth(): 
    area_pole, area_wo_poles = generate_area_array()
    if np.isclose((area_wo_poles.sum() + area_pole * 2), 1):
        print('Area is approx. 1. \nTest successfull!')
    else: 
        print('Area is not approx. 1. \nTest failed!')
    
    
    
def save_area_array_numpy(): 
    area_pole, area_wo_poles = generate_area_array()
    np.savez('EBM/output/area_numpy', area_pole =  area_pole, area_wo_poles = area_wo_poles)
save_area_array_numpy() 