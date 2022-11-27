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
                    heapq.heappush(band, (distances[nb_y, nb_x], nb_y, nb_x))   #BAND points stored on heap
    
    #Might need to compute distances in future 
    return distances, flags, band

#finds closest quadrant by solving step of eikonal equation | Solves T value
def eikonal(y1, x1, y2, x2, height, width, T_vals, flags):

    #check if points in image
    if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
        return INF

    if y2 < 0 or y2 >= height or x2 < 0 or x2 >= width:
        return INF

    #get flag of point 1 and point 2
    flag1 = flags[y1, x1] 
    flag2 = flags[y2, x2]

    # both pixels are known
    if flag1 == KNOWN:
        if flag2 == KNOWN:
            T1 = T_vals[y1, x1]
            T2 = T_vals[y2, x2]
            d = 2.0 - (T1 - T2) ** 2
            if d > 0.0:
                r = math.sqrt(d)
                s = (T1 + T2 - r) / 2.0
                if s >= T1 and s >= T2:
                    return s
                else:
                    s += r
                    if s >= T1 and s >= T2:
                        return s
                
                return INF
        else:
            #if only flag 1 = KNOWN
            T1 = T_vals[y1, x1]
            return 1.0 + T1

    #if only flag2 = KNOWN
    if flag2 == KNOWN:
        T2 = T_vals[y2, x2]
        return 1.0 + T2

    # neither pixel is known
    return INF



