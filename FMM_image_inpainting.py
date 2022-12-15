import math
import heapq
import numpy as np

KNOWN = 0
BAND = 1
INSIDE = 2

MAX_VALUE = 1e6
EPS = 1e-6 

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
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width: # error handling
                continue
            if flags[nb_y, nb_x] == BAND:
                continue
            # if the neighbour of a point we want to inpaint is not in the region we want to inpaint then it is on the narrow band (boundary), we update flag and T
            # note that points with flag BAND are already inpainted
            if mask[nb_y, nb_x] == 0: 
                flags[nb_y, nb_x] = BAND 
                distance_map[nb_y, nb_x] = 0.0
                heapq.heappush(band, (distance_map[nb_y, nb_x], nb_y, nb_x)) 
    
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
                    distance_map[nb_y, nb_x] = min(sol1 + sol2 + sol3 + sol4) 
                    
                    # (re)insert point in the heap   
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
    return img

def _inpaint_point(img, distance_map, flags, epsilon, y, x, height, width):
    

    # find neighbourhood of point to inpaint point
    B = []
    for i in range(y-epsilon,y+epsilon+1):
        for j in range(x-epsilon,x+epsilon+1):
            if i < 0 or i >= height or j < 0 or j >= width: # error handling
                continue
            if i == epsilon and j == epsilon:
                continue
            if flags[i, j] == KNOWN:
                # check if distance to point to inpaint is smaller or equal than epsilon 
                if np.linalg.norm(np.array((y-i,x-j))) <= epsilon: 
                    B.append((i,j))
    
    # calculate gradient of T at [y,x]
    gradT = np.gradient(distance_map)
    #gradT = (distance_map[y + 1,x] - distance_map[y -1,x], distance_map[y, x + 1] - distance_map[y, x-1])
    gradT_yx = np.array((gradT[0][y,x], gradT[1][y,x]))

    # calculate gradient of image intensity
    #gradI = np.gradient(img)

    


    #This section code chunk is for using the method similar to openCV
    #Ended up using the version wihtout the gradient in the final algorithm
    # gradI = (img[y + 1,x] - img[y -1,x], img[y, x + 1] - img[y, x-1])
    # grad1 = [gradI[0][0], gradI[1][0]]
    # grad2 = [gradI[0][1], gradI[1][1]]
    # grad3 = [gradI[0][2], gradI[1][2]]

    # Ia = [0,0,0]
    # Jx = [0,0,0]
    # Jy = [0,0,0]



    # initialize inpainting value to 0 
    denominator = 0 
    numerators = [0,0,0]

    for i,j in B:
        # calculate the weight function w
        vector = np.array((y-i, x-j)) # vector from neighbourhood point to point to inpaint
        norm_vector = np.linalg.norm(vector,1) # TODO: check if this is the right type of norm - see documentation (by default frobenius norm)
        
        dir = np.dot(gradT_yx,vector)/norm_vector
        if dir == 0:
            dir = EPS
        dist = 1/norm_vector**2
        #print(abs(distance_map[y,x] - distance_map[i,j]))
        lev = 1/(1 + abs(distance_map[y,x] - distance_map[i,j]))
        #w = abs(dir * dist * lev)
        w = abs(dir*dist*lev)
        # calculate inpainting value
        numerators[0] += w * img[i,j,0] 
        numerators[1] += w * img[i,j,1]
        numerators[2] += w * img[i,j,2] 

        denominator += w

        #This section is also used for the open CV method
        # Ia[0] += w * img[i,j,0]
        # Ia[1] += w * img[i,j,1]
        # Ia[2] += w * img[i,j,2]

        # Jx[0] -= w * grad1[0]*vector[0]
        # Jx[1] -= w * grad2[0]*vector[0]
        # Jx[2] -= w * grad3[0]*vector[0]

        # Jy[0] -= w * grad1[1]*vector[1]
        # Jy[1] -= w * grad2[1]*vector[1]
        # Jy[2] -= w * grad3[1]*vector[1]

    
    values = [0,0,0]

    #This is the formual for the open CV method 
    #To calculate the value at each pixel
    # for i in range(0,3):
    #     values[i] = Ia[i]/denominator + (Jx[i] + Jy[i])/(math.sqrt(Jx[i]* Jx[i] + Jy[i]*Jy[i])+ 1e-20)+ 0.5

    values = numerators/ denominator

    img[y,x,0] = values[0]
    img[y,x,1] = values[1]
    img[y,x,2] = values[2]



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
    height,width,w = img.shape
    height1, width1 = mask.shape

    if height != height1 or width != width1:
        raise ValueError("input image and mask are not the same size")

    # initialization
    distance_map, flags, band = _init(mask, height, width)

    _FMM(img, distance_map, flags, band, height, width, epsilon)

    return img 

