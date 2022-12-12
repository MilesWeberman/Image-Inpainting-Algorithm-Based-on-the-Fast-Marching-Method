import math
import heapq
import numpy as np
import cv2 as cv

img = cv.imread('C:/Users/jonas/Desktop/mcgill/Fourth year/Comp 558/Final project/FMM_image_inpainting/Pic.png')
mask = cv.imread('C:/Users/jonas/Desktop/mcgill/Fourth year/Comp 558/Final project/FMM_image_inpainting/Mask.png',0)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

KNOWN = 0
BAND = 1
INSIDE = 2

MAX_VALUE = 1e6
EPS = 1e-6 # TODO: idk why we need that

# initializing flags and T-values
def _init(mask, height, width):
    distance_map = np.full((height,width), 0.0, dtype=float)
    flags = np.full((height,width), 0, dtype=int)
    flags[np.nonzero(mask)] = INSIDE  
    band = [] 

    mask_Y, mask_X = np.nonzero(mask) # save indices of non-zero values in mask, that is the region we want to inpaint
    for Y, X in zip(mask_Y,mask_X):
       # set T to a large value inside the region we want to inpaint
       distance_map[Y,X] = MAX_VALUE
       for i in range(-1,2):
           for j in range(-1,2):
            if i == 0 and j == 0:
                continue
            nb_y = Y + i
            nb_x = X + j
            if nb_y < 0 or nb_y > height or nb_x < 0 or nb_x > width: # error handling
                continue
            if flags[nb_y, nb_x] == BAND:
                continue
            # if the neighbour of a point we want to inpaint is not in the region we want to inpaint then it is on the narrow band (boundary), we update flag and T
            # note that points with flag BAND are already inpainted
            if mask[nb_y, nb_x] == 0: 
                flags[nb_y, nb_x] = BAND 
                distance_map[nb_y, nb_x] = 0.0
                heapq.heappush(band, (distance_map[nb_y, nb_x], nb_y, nb_x)) # TODO: maybe it's redundant adding distance map value at that point !!!! distance_map[nb_y, nb_x]
    # TODO: might need to compute distances in future 
    return distance_map, flags, band

def _FMM(img, distance_map, flags, band, height, width, epsilon):
    while len(band) != 0:
        # extract the BAND point with the smallest T and mark it as KNOWN (step 1)
        T,y,x = heapq.heappop(band) 
        flags[y,x] = KNOWN
        # iterate through the popped point's neighbours 
        for i in range(-1,2):
            for j in range(-1,2):
                if i == 0 and j == 0:
                    continue
                nb_y = y + i
                nb_x = x + j

                if nb_y < 0 or nb_y > height or nb_x < 0 or nb_x > width: # error handling
                    continue
                if flags[nb_y, nb_x] != KNOWN:
                    
                    if flags[nb_y, nb_x] == INSIDE: # if that point is in the region to be inpainted
                        # march the boundary inward by adding a new point to it (step 2) and inpaint that point (step 3)
                        #flags[nb_y, nb_x] = BAND
                        _inpaint_point(img, distance_map, flags, epsilon, nb_y, nb_x, height, width)
                    
                    # error handling
                    if nb_y-1 >= 0 and nb_x-1 >= 0:
                        sol1 = _solve_eikonal(nb_y-1, nb_x, nb_y, nb_x-1, height, width,distance_map, flags)

                    if nb_y-1 >= 0 and nb_x+1 < width: 
                        sol2 = _solve_eikonal(nb_y-1, nb_x, nb_y, nb_x+1, height, width, distance_map, flags)   

                    if nb_y+1 < height and nb_x-1 >= 0:
                        sol3 = _solve_eikonal(nb_y+1, nb_x, nb_y, nb_x-1, height, width, distance_map, flags)

                    if nb_y+1 < height and nb_x+1 < width:
                        sol4 = _solve_eikonal(nb_y+1, nb_x, nb_y, nb_x+1, height, width, distance_map, flags)

                    # propagates the value T of point at [y,x] to its neighbors (step 4)
                    distance_map[nb_y, nb_x] = min(sol1, sol2, sol3, sol4) 
                    # (re)insert point in the heap 
                    # TODO: if that point was in band remove it from the heap before reinserting - maybe do that before updating distance_map and so it's more efficient
                    # TODO: make sure we're adding to heap correctly (min is based of distance map)
                    
                    if flags[nb_y, nb_x] == BAND:
                        counter = 0
                        for k in band:
                            dist, cur_y, cur_x = k
                            if cur_y == nb_y and cur_x == nb_x:
                                band[counter] = (distance_map[nb_y, nb_x], nb_y, nb_x)
                                heapq.heapify(band)
                                break
                    else:
                        heapq.heappush(band, (distance_map[nb_y, nb_x], nb_y, nb_x)) # TODO: do we need to add if not already in band ??????
                        flags[nb_y, nb_x] = BAND

def _inpaint_point(img, distance_map, flags, epsilon, y, x, height, width):
    # TODO: fix value? parameter? function of unknown thickness (Telea 2.4)
    # find neighbourhood of point to inpaint point
    B = []
    for i in range(y-epsilon,y+epsilon+1):
        for j in range(x-epsilon,x+epsilon+1):
            if i < 0 or i > height or j < 0 or j > width: # error handling
                continue
            if flags[i, j] == KNOWN:
                # check if distance to point to inpaint is smaller or equal than epsilon 
                if np.linalg.norm(np.array((y-i,x-j))) <= epsilon: # TODO: check if this is the right type of norm - see documentation (by default frobenius norm)
                    B.append((i,j))
    
    # calculate gradient of T at [y,x]
    gradT = np.gradient(distance_map)
    #gradT = (distance_map[y + 1,x] - distance_map[y -1,x], distance_map[y, x + 1] - distance_map[y, x-1])
    gradT_yx = np.array((gradT[0][y,x], gradT[1][y,x]))

    # calculate gradient of image intensity
    #gradI = np.gradient(img)

    #TODO: Add error handeling for the gradient and also figure out what we are going to do if one of the values is out of bounds
    #Pyheal has some thing they do but I didn't fully understand it so I did not want to add it here
    gradI = (img[y + 1,x] - img[y -1,x], img[y, x + 1] - img[y, x-1])
    

    # initialize inpainting value to 0 
    numerator = 0
    denominator = 0 

    for i,j in B:
        # calculate the weight function w
        vector = np.array((y-i, x-j)) # vector from neighbourhood point to point to inpaint
        norm_vector = np.linalg.norm(vector,1) # TODO: check if this is the right type of norm - see documentation (by default frobenius norm)
        
        if gradT_yx[0] == 0 and gradT_yx[1] == 0:
            dir = 1
        else:
            dir = np.dot(gradT_yx,vector)/norm_vector
        dist = 1/norm_vector**2
        lev = 1/(1 + abs(distance_map[y,x] - distance_map[i,j]))
        w = abs(dir * dist * lev)
        # calculate inpainting value
        #gradI_ij = np.array((gradI[0][i,j], gradI[1][i,j]))
        numerator += w * (img[i,j]) #+ np.dot(gradI,vector)) 
        denominator += w

    #update inpainting value
    if numerator == 0:
        print("numerator " + str(numerator))

    if numerator/denominator >= 255:
        print("Too high")

        
    img[y,x] = numerator/denominator

def _solve_eikonal(y1, x1, y2, x2, height, width, T_vals, flags):
    #check if points in image
    if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
        return MAX_VALUE
        
    if y2 < 0 or y2 >= height or x2 < 0 or x2 >= width:
        return MAX_VALUE

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
                
                return MAX_VALUE
        else:
            #if only flag 1 = KNOWN
            T1 = T_vals[y1, x1]
            return 1.0 + T1

    #if only flag2 = KNOWN
    if flag2 == KNOWN:
        T2 = T_vals[y2, x2]
        return 1.0 + T2

    # neither pixel is known
    return MAX_VALUE

def inpaint(img, mask, epsilon):
    # error handling
    if img.shape != mask.shape:
        raise ValueError("input image and mask are not the same size")
    # TODO: figure out how ot work with 3 channels and where to separate grey one, look at Telea (change _inpaint_point accordingly)
    height,width = img.shape

    # initialization
    distance_map, flags, band = _init(mask, height, width)

    # solve inpainting distance
    _FMM(img, distance_map, flags, band, height, width, epsilon)

    return img

im = inpaint(img, mask, 3)

cv.imshow("Grayscale",im)
cv.waitKey(0) 
cv.destroyAllWindows()
