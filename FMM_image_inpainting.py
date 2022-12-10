import math
import heapq
import numpy as np

KNOWN = 0
BAND = 1
INSIDE = 2

MAX_VALUE = 1e6
EPS = 1e-6 # TODO: idk why we need that

# initializing flags and T-values
def _init(mask, radius, height, width):
    distance_map = np.full((height,width), 0.0, dtype=float)
    flags = np.full((height,width), 0, dtype=int)
    flags[np.nonzero(mask)] = INSIDE  
    # TODO: flags = mask.astype(int) * INSIDE (replace by line above, deals with case where some masked regions are not exactly 1)
    band = [] # TODO: do we need to heapify

    mask_Y, mask_X = np.nonzero(mask) # save indices of non-zero values in mask, that is the region we want to inpaint
    for Y, X in zip(mask_Y,mask_X):
       # set T to a large value inside the region we want to inpaint
       distance_map[Y,X] = MAX_VALUE
       for i in (-1,1):
           neigbs = [(Y + i, X), (Y, X + i)]
           for nb_y, nb_x in neigbs:
                if nb_y < 0 or nb_y > height or nb_x < 0 or nb_x > width: # error handling
                    continue
                if flags[nb_y, nb_x] == BAND:
                    continue
                # if the neighbour of a point we want to inpaint is not in the region we want to inpaint then it is on the narrow band (boundary), we update flag and T
                if mask[nb_y, nb_x] == 0: 
                    flags[nb_y, nb_x] = BAND 
                    distance_map[nb_y, nb_x] = 0.0
                    heapq.heappush(band, (nb_y, nb_x)) # TODO: maybe it's redundant adding distance map value at that point !!!! distance_map[nb_y, nb_x]
    #Might need to compute distances in future 
    return distance_map, flags, band

def _FMM(distance_map, flags, band, height, width):
    while len(band) != 0:
        # extract the BAND point with the smallest T and mark it as KNOWN (step 1)
        y,x = heapq.heappop(band) 
        flags[y,x] = KNOWN
        # iterate through the popped point's neighbours 
        for i in (-1,1):
                neigbs = [(y + i, x), (y, x + i)]
                for nb_y, nb_x in neigbs:
                    if nb_y < 0 or nb_y > height or nb_x < 0 or nb_x > width: # error handling
                        continue
                    if flags[nb_y, nb_x] != KNOWN:
                        if flags[nb_y, nb_x] == INSIDE: # if that point is in the region to be inpainted
                            # march the boundary inward by adding a new point to it (step 2) and inpaint that point (step 3)
                            flags[nb_y, nb_x] = BAND
                            _inpaint_point(nb_y, nb_x)
                        # propagates the value T of point at [y,x] to its neighbors (step 4)
                        distance_map[nb_y, nb_x] = min(_solve_eikonal(...)) # TODO: figure out the right way !!!!!
                        # (re)insert point in the heap 
                        heapq.heappush(band, (nb_y, nb_x)) # TODO: do we need to add if not already in band ??????

def _inpaint_point(img, distance_map, flags, y, x):

    epsilon = 3 # TODO: fix value? parameter? function of unknown thickness (Telea 2.4)
    # find neighbourhood of point to inpaint point
    B = []
    # TODO: find neighbourhood: add if != INSIDE and check if in bound 
    # TODO: how to find neighbourhood (only along x and y axes, or go in circle), also figure out if to include points on band
    
    # calculate gradient of T at [y,x]
    gradT = np.gradient(distance_map)
    gradT_yx = np.array((gradT[0][y,x], gradT[1][y,x]))

    # calculate gradient of image intensity
    gradI = np.gradient(img)
    # TODO: check if it's ok to calculate here or do we need to calculate in loop? 
    # TODO: check pseudocode for Telea : what is the if conditional statement (is it if we're actly calculating gradient using central diff)

    # initialize inpainting value to 0 
    numerator = 0
    denominator = 0 

    for j,i in B:
        # calculate the weight function w
        vector = np.array((y-j, x-i)) # vector from neighbourhood point to point to inpaint
        norm_vector = np.linalg.norm(vector) # TODO: check if this is the right type of norm - see documentation (by default frobenius norm)
        dir = np.dot(gradT_yx,vector)/norm_vector
        dist = 1/norm_vector^2
        lev = 1/(1 + abs(distance_map[y,x] - distance_map[j,i]))
        w = dir * dist * lev
        # calculate inpainting value
        gradI_ji = np.array((gradI[0][j,i], gradI[1][j,i]))
        numerator += w * (img[j,i] + np.dot(gradI_ji,vector)) 
        denominator += w

    # update inpainting value
    img[y,x] = numerator/denominator

# solves a finite difference discretization of the Eikonal equation (eqn 3 in Telea's paper)
def _solve_eikonal(distance_map, y, x):  
    return

def inpaint(img, mask):
    # TODO: error handling: make sure both image have same size
    # TODO: figure out how ot work with 3 channels and where to separate grey one, look at Telea (change _inpaint_point accordingly)
    height,width = img.shape
    return




