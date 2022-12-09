import math
import heapq
import numpy as np

KNOWN = 0
BAND = 1
INSIDE = 2

MAX_VALUE = 1e6
EPS = 1e-6

# initializing flags and T-values
def _init(height, width, mask, radius):
    distance_map = np.full((height,width), 0.0, dtype=float)
    flags = np.full((height,width), 0, dtype=int)
    flags[np.nonzero(mask)] = INSIDE  
    # flags = mask.astype(int) * INSIDE (replace by line above, deals with case where some masked regions are not exactly 1) !!!!
    band = []

    mask_Y, mask_X = np.nonzero(mask) # save indices of non-zero values in mask, that is the region we want to inpaint
    for Y, X in zip(mask_Y,mask_X):
       # set T to a large value inside the region we want to inpaint
       distance_map[Y,X] = MAX_VALUE
       for i in (-1,1):
           neigbs = [(Y + i, X), (Y, X+i)]
           for nb_y, nb_x in neigbs:
                if nb_y < 0 or nb_y > height or nb_x < 0 or nb_x > width: # error handling
                    continue
                if flags[nb_y, nb_x] == BAND: # error handling
                    continue
                # if the neighbour of a point we want to inpaint is not in the region we want to inpaint then it is on the band (boundary), we update flag and T
                if mask[nb_y, nb_x] == 0: 
                    flags[nb_y, nb_x] = BAND 
                    distance_map[nb_y, nb_x] = 0.0
                    heapq.heappush(band, (distance_map[nb_y, nb_x], nb_y, nb_x)) ##  maybe it's redundant adding distance map value at that point !!!!
    
    #Might need to compute distances in future 
    return distance_map, flags, band






