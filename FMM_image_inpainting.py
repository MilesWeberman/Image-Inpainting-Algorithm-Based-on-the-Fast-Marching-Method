import math
import heapq
# from tkinter.tix import MAX
import numpy as np

KNOWN = 0
BAND = 1
INSIDE = 2

INF = 1e6
EPS = 1e-6

# initializing flags and T-values
def _init(height, width, mask, radius):
    distances = np.full((height,width), 0.0, dtype=float) 
    flags = mask.astype(int) * INSIDE
    band = []

    mask_Y, mask_X = mask.nonzero() # save indices of non-zero values in mask, that is the region we want to inpainting
    for Y, X in zip(mask_Y,mask_X):
       distances[Y,X] = INF
       # error handling
       for i in (-1,1):
           neigbs = [(Y + i, X), (Y, X+i)]
           for nb_y, nb_x in neigbs:
                if nb_y < 0 or nb_y > height or nb_x < 0 or nb_x > width:
                    continue
    
                if flags[nb_y, nb_x] == BAND: 
                    continue
                
                if mask[nb_y, nb_x] == 0:
                    flags[nb_y, nb_x] = BAND 
                    distances[nb_y, nb_x] = 0.0
                    heapq.heappush(band, (distances[nb_y, nb_x], nb_y, nb_x))
    
    #Might need to compute distances in future 
    return distances, flags, band






