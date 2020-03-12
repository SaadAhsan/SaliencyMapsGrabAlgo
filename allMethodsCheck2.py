'''
This is setting up all the methods and importing all the packages for the algorithm for final method to create saliency map

'''

#import Djik as D
import lm

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from IPython.display import Image, display
from glob import glob
   
from skimage import  color
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, dtype_limits
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

from scipy import signal, optimize
import scipy
from scipy.sparse.csgraph import dijkstra as dkstra

fastHist = True

try:
    from fast_histogram import histogram1d
except:
    fastHist = False


data_dir = 'MSRA10K_Imgs_GT/Imgs/'

listofImageNames = glob('MSRA10K_Imgs_GT/Imgs/*', recursive=True)

listOfImgName_X = glob('MSRA10K_Imgs_GT/Imgs/*.jpg', recursive=True)

listOfImgName_Y = listOfImgName_X.copy()

for i in range(len(listOfImgName_Y)) :
    name = listOfImgName_Y[i]
    name = name[0:-3]+'png'
    listOfImgName_Y[i] = name
    
F = lm.makeLMfilters()

bnds = []

for i in range(1600):
    bnds.append((0, 1))


def SuperPixels (image, n_segments) :
    super_pix = slic(image, n_segments = n_segments, sigma = 3, convert2lab=True)
    return super_pix
    
    
def makeAdjMatrix (super_pixels) :
    
    (height,width) = (super_pixels.shape)
    numInd = np.max(super_pixels) + 1 - np.min(super_pixels)
    
    Adjacency_ij = np.zeros((numInd,numInd))
    
    sm_0_1 = super_pixels[:,0:width-1]
    sm_1_w = super_pixels[:,1:width]
    c1 = (sm_0_1 != sm_1_w)
    
    for x, y in zip(sm_0_1[c1], sm_1_w[c1]) :
        Adjacency_ij[int(x),int(y)] =1
        Adjacency_ij[int(y),int(x)] =1
    
    sm_1_0 = super_pixels[0:height-1,:]
    sm_1_h = super_pixels[1:height,:]
    c2 = (sm_1_0 != sm_1_h)
    
    for x, y in zip(sm_1_0[c2], sm_1_h[c2]) :
        Adjacency_ij[int(x),int(y)] =1
        Adjacency_ij[int(y),int(x)] =1
        
    return Adjacency_ij, numInd
   

def makeBorderset(super_pixels) :      

    (height,width) = (super_pixels.shape)
    
    B_Top = list(set(super_pixels[0,0:width-1]))
    B_Bot = list(set(super_pixels[height-1,0:width-1]))
    B_Left = list(set(super_pixels[0:height-1,0]))
    B_Right = list(set(super_pixels[0:height-1,width-1]))
    
    BorderSet = {0 : B_Top , 1 : B_Bot, 2 :B_Left, 3 :B_Right}
    
    return BorderSet


def makeLabImage(image) :
    lab = color.rgb2lab(image)

    lab_Image = lab[:,:,0]*lab[:,:,1]*lab[:,:,2]

    minIm = np.min(np.min(lab_Image))
    maxIm = np.max(np.max(lab_Image))

    lab_Image = 1.0*(lab_Image - minIm)/(maxIm-minIm+0.000001)
    
    return lab_Image
    

def make_LM_Filter(image, F) :
    
    image_grey = color.rgb2gray(image)
    
    (height,width) = (image_grey.shape)

    filter_response = np.zeros((height,width,48))

    for i in range(48) :
        filter_response[:,:,i] = signal.fftconvolve(image_grey, F[:,:,i], mode='same')
        
    max_LM_filter = np.amax(filter_response,axis=2)

    minIm = np.min(max_LM_filter)
    maxIm = np.max(max_LM_filter)

    max_LM_filter = 1.0*(max_LM_filter - minIm)/(maxIm-minIm+0.000001)
    
    return max_LM_filter
    

def makeRegionalDescriptors(super_pixels, lab_Image, max_LM_filter, numInd) : 
    
    reg_desc_col = {i:lab_Image[(super_pixels == i)] for i in range(numInd)}
    reg_desc_tex = {i:max_LM_filter[(super_pixels == i)] for i in range(numInd)}
    reg_desc_col_ave = {i:sum(reg_desc_col [i])/len(reg_desc_col [i]) for i in range(numInd)}
        
    return reg_desc_col, reg_desc_tex, reg_desc_col_ave
    
    
def ChiSquaredDistance(array1, array2, K = 16) :
    #minA = min(min(array1),min(array2))
    #maxA = max(max(array1),max(array2))
    
    #print(minA, maxA, 1.0*(maxA-minA)/K)
    #if minA == maxA :
    #    return 0
    
    #interval = np.arange(start = 0, stop = 1, step = 1.0/K)
    
    if fastHist:
        hist1 = histogram1d(array1, bins=K, range=(0, 1))
        hist2 = histogram1d(array2, bins=K, range=(0, 1))  
    else: 
        hist1,_ = np.histogram(array1, bins=K, range=(0, 1))
        hist2,_ = np.histogram(array2, bins=K, range=(0, 1))
    
    hist1d = hist1 / len(array1)
    hist2d = hist2 / len(array2)
    
    den = hist1d + hist2d + 0.00001
    num = ((hist1d-hist2d)**2)
    sumi = 2*num/den
    
    #sumi = list(filter(lambda x: math.isnan(x) == False, sumi))
    
    return np.sum(sumi)
    
    
def GeodDist(desc_Col, desc_Tex, Ave_Col, Adjacency_ij, lamda1 = 0.25, lamda2 = 0.45, lamda3 =0.3, K=16) :
    numInd = len(Ave_Col)
    
    distance = np.zeros((numInd,numInd))
    
    x = np.ones((numInd,numInd))
    
    y = np.ones((numInd,numInd))

    x = np.cumsum(x,axis=1) - 1

    y = np.cumsum(y,axis=0) - 1

    for i, j in zip(x[Adjacency_ij==1], y[Adjacency_ij==1]) :
        i = int(i)
        j = int(j)
        distance[i,j] = lamda1 *np.abs(Ave_Col[i] - Ave_Col[j])
        distance[i,j] += lamda2 *ChiSquaredDistance(desc_Col[i], desc_Col[j], K=K)
        distance[i,j] += lamda3 *ChiSquaredDistance(desc_Tex[i], desc_Tex[j], K=K)
    
    distance[Adjacency_ij==0] = 999999     
    
    distance[np.eye(numInd)==1] = 0
    
    return distance


def makeEdgeGroups(Adjacency_ij, BorderSet, numInd) :

    Edge_Group_2 = np.zeros((numInd,numInd))
    Edge_Group_3 = np.zeros((numInd,numInd))
    
    a = []
    for i in BorderSet :
        a += BorderSet[i]

    B_All = list(set(a))
    
    TwoHops = np.matmul(Adjacency_ij,Adjacency_ij)
    
    TwoHops = (TwoHops>0)*1.0 - Adjacency_ij - np.eye(numInd)
    
    TwoHops = 1.0*(TwoHops==1)

    iB = [i for i in range(numInd) if i not in B_All]
    
    
    Edge_Group_2[iB,:] = 1
    Edge_Group_2[:,B_All] += 1
    
    Edge_Group_2 = (Edge_Group_2==2)*1.0
    
    Edge_Group_2 = Edge_Group_2 + Edge_Group_2.T
    
    Edge_Group_2 = Edge_Group_2 - Adjacency_ij - TwoHops
    
    Edge_Group_2 = 1.0*(Edge_Group_2==1)
        
    Edge_Group_3[B_All,:] =1
    Edge_Group_3[:,B_All] +=1
    
    Edge_Group_3 = (Edge_Group_3==2)*1.0
    
    Edge_Group_3 = Edge_Group_3 - Adjacency_ij - TwoHops - np.eye(numInd)
    
    Edge_Group_3 = 1.0*(Edge_Group_3==1)
                    
    return TwoHops, Edge_Group_2, Edge_Group_3, B_All
    
    
def makeWeightMatrix(numInd,Adjacency_ij,TwoHops,Edge_Group_2,Edge_Group_3,GeoDist,B_All, sigma2 = -0.1) :

    weight_matrix = np.zeros((numInd,numInd))

    mask_Edges = TwoHops + Adjacency_ij + 1.0*Edge_Group_2/len(B_All) + Edge_Group_3

    weight_matrix = np.exp(GeoDist / sigma2) * mask_Edges
    
    return weight_matrix


def chooseBorderExclude(BorderSet, reg_desc_col, reg_desc_tex) :
    
    reg_desc_col_Bound = { 0 : [ ], 1 : [ ], 2 : [ ], 3 : [ ]}
    reg_col_Ave_Bound = { 0 : 0, 1 : 0, 2 : 0, 3 : 0}
    reg_desc_tex_Bound = { 0 : [], 1 : [], 2 : [], 3 : []}

    for i in BorderSet:
        for k in BorderSet[i] :
            a = (reg_desc_col_Bound[i])
            a += list(reg_desc_col[k])
            reg_desc_col_Bound[i] = a
            a = reg_desc_tex_Bound[i]
            a += list(reg_desc_tex[k])
            reg_desc_tex_Bound[i] = a
        a = reg_desc_col_Bound[i]
        reg_col_Ave_Bound[i] = sum(a)/len(a)
        
    adj = np.ones((4,4)) - np.identity(4)
    
    BoundDist = GeodDist(reg_desc_col_Bound, reg_desc_tex_Bound, reg_col_Ave_Bound, adj)
    
    return(np.argmax(np.sum(BoundDist, axis=0)))
    
    
def createDiagonal(weight_matrix) :
    a = np.sum(weight_matrix, axis=0)
    D = np.identity(len(a))*a
    
    return D
    

def calculateSaliencyprob(Diag,weight_matrix, BorderSet, numInd, mew = 0.99, damp = 0.001) :
    #inverseMat = scipy.linalg.pinv(Diag - mew*weight_matrix)
    A = Diag - mew*weight_matrix
    F = np.zeros((numInd,3))
    j = -1
    for i in BorderSet :
        y = np.zeros((numInd,1))
        for k in BorderSet[i] :
            y[k] = 1
        f = scipy.sparse.linalg.lsmr(A,y,damp=damp)
        #print(f[0])
        f = f[0]
        f = (f-np.min(f))/(np.max(f)-np.min(f)+0.000001)
        j+=1
        F[:,j] = f.flatten()
    return F
   

def seeSaliencyMap (F_cross, super_pixels) :
    height,width = super_pixels.shape
    Im1 = np.zeros((height,width,3))
    for i in range(height):
        for j in range(width):
            Im1[i,j,0] = F_cross[super_pixels[i,j],0]
            Im1[i,j,1] = F_cross[super_pixels[i,j],1]
            Im1[i,j,2] = F_cross[super_pixels[i,j],2]
    
    Im2 = 1-Im1
    return Im1, Im2
  
    
def costFunction (F_cross, S_ave, weight_matrix) :
    
    numInd = len(F_cross)
    F_cross = F_cross.reshape((numInd,1))
    
    FB_i = (F_cross > S_ave)
    BB_i = 1 - FB_i
    FB_i = 1 - BB_i
    
    F_ij = np.zeros((numInd,numInd))
    F_ij = F_ij + F_cross
    #print(F_ij[1:10,1:10])
    F_ij = np.transpose(F_ij)
    F_ij = F_ij - F_cross
    #print(F_ij[1:10,1:10])
    F_ij = F_ij**2
    
    #print(F_cross)
    #print(F_ij)

    cost  = np.sum(FB_i*((F_cross-1)**2))
    cost += np.sum(BB_i*((F_cross)**2))
    cost += np.sum(np.sum((weight_matrix*F_ij[:,:])))
    
    return np.sum(cost)
    
def costJacFunction (F_cross, S_ave, weight_matrix) :
    numInd = len(F_cross)
    F_cross = F_cross.reshape((numInd,1))
    
    FB_i = (F_cross > S_ave)
    BB_i = 1 - FB_i
    FB_i = 1 - BB_i
    
    cost = 2*FB_i*(F_cross-1)
    cost += 2*BB_i*(F_cross)
    
    v = np.sum(weight_matrix,axis=1).reshape((numInd,1))
    
    cost += 4*(F_cross)*v
    
    weight_matrix = F_cross*weight_matrix
    
    cost -= 4*np.sum(weight_matrix,axis=0).reshape((numInd,1))
    
    return cost

def seeSaliencyMap2 (F_cross, super_pixels) :
    height,width = super_pixels.shape
    Im1 = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            Im1[i,j] = F_cross[super_pixels[i,j]]
    Im2 = 1-Im1
    return Im1, Im2
    
def calculateFinalSaliency(Diag,weight_matrix, numInd, F_y, mew = 0.99, thresh = 0.8, quant = 90, damp = 0.001) :
    inverseMat = np.linalg.pinv(Diag - mew*weight_matrix)
    f = np.matmul(inverseMat,F_y)
    m = np.percentile(f, quant)
    t = m*thresh
    g = (f>t)
    f[g] = t
    f = (f-np.min(f))/(np.max(f)-np.min(f)+0.000001)
    F = f.flatten()
    
    #A = Diag - mew*weight_matrix
    #f = scipy.sparse.linalg.lsmr(A,F_y,damp=damp)
    #f = f[0]
    #f = (f-np.min(f))/(np.max(f)-np.min(f)+0.000001)
    #F = f.flatten()
    return F

def getAggSaliencyMap(image,N_segments,Fi=F, b=bnds, l1=0.25, l2=0.45, l3=0.3, 
                      s2=-0.1, K=16, mew=0.99, damp1 =0.001,damp2 =0.001, thresh = 0.8, quant = 90):
    
    super_pixels = SuperPixels (image, N_segments)
    
    Adjacency_ij, numInd = makeAdjMatrix(super_pixels)
    BorderSet = makeBorderset(super_pixels)
    
    lab_Image = makeLabImage(image)
    max_LM_filter = make_LM_Filter(image, Fi)
    reg_desc_col, reg_desc_tex, reg_desc_col_ave = makeRegionalDescriptors(super_pixels, lab_Image, max_LM_filter, numInd)
    
    GeoDist = GeodDist(reg_desc_col, reg_desc_tex, reg_desc_col_ave, Adjacency_ij, lamda1 = l1, lamda2 = l2, lamda3 =l3, K=K)
    GeoDist = dkstra(GeoDist)
    
    TwoHops, Edge_Group_2, Edge_Group_3, B_All = makeEdgeGroups(Adjacency_ij, BorderSet, numInd)
    
    weight_matrix = makeWeightMatrix(numInd,Adjacency_ij,TwoHops,Edge_Group_2,Edge_Group_3,GeoDist,B_All, sigma2 = s2)
    
    BoundRemove = chooseBorderExclude(BorderSet, reg_desc_col, reg_desc_tex)
    BorderSet.pop(BoundRemove)
    
    Diag = createDiagonal(weight_matrix)
    
    F_cross = calculateSaliencyprob(Diag, weight_matrix, BorderSet, numInd, mew=mew, damp = damp1)
    
    S_orig = (1-F_cross[:,0])*(1-F_cross[:,1])*(1-F_cross[:,2])
    
    F_map,S_map = seeSaliencyMap (F_cross, super_pixels)
    
    S_ave = np.mean(S_orig)
    
    res = scipy.optimize.minimize(costFunction, S_orig, args=(S_ave, weight_matrix), 
                              jac=costJacFunction, method="L-BFGS-B",bounds=bnds[0:len(S_orig)])
    
    F_y = np.resize(res.x,(numInd,1))
    
    S_map1,F_map = seeSaliencyMap2 (F_y, super_pixels)
     
    S_star = calculateFinalSaliency(Diag,weight_matrix, numInd, F_y,mew = mew, thresh = thresh, quant =quant, damp = damp2)

    S_map2,F_map = seeSaliencyMap2 (S_star, super_pixels)
    
    return S_map2, S_map1, S_map

def getAggSaliencyMap2(image,N_segments,Fi=F, b=bnds, l1=0.25, l2=0.45, l3=0.3, 
                      s2=-0.1, K=16, mew=0.99, damp1 =0.001,damp2 =0.001, thresh = 0.8, quant = 90):
    
    super_pixels = SuperPixels (image, N_segments)
    
    Adjacency_ij, numInd = makeAdjMatrix(super_pixels)
    BorderSet = makeBorderset(super_pixels)
    
    lab_Image = makeLabImage(image)
    max_LM_filter = make_LM_Filter(image, Fi)
    reg_desc_col, reg_desc_tex, reg_desc_col_ave = makeRegionalDescriptors(super_pixels, lab_Image, max_LM_filter, numInd)
    
    GeoDist = GeodDist(reg_desc_col, reg_desc_tex, reg_desc_col_ave, Adjacency_ij, lamda1 = l1, lamda2 = l2, lamda3 =l3, K=K)
    GeoDist = dkstra(GeoDist)
    
    TwoHops, Edge_Group_2, Edge_Group_3, B_All = makeEdgeGroups(Adjacency_ij, BorderSet, numInd)
    
    weight_matrix = makeWeightMatrix(numInd,Adjacency_ij,TwoHops,Edge_Group_2,Edge_Group_3,GeoDist,B_All, sigma2 = s2)
    
    BoundRemove = chooseBorderExclude(BorderSet, reg_desc_col, reg_desc_tex)
    BorderSet.pop(BoundRemove)
    
    Diag = createDiagonal(weight_matrix)
    
    F_cross = calculateSaliencyprob(Diag, weight_matrix, BorderSet, numInd, mew=mew, damp = damp1)
    
    S_orig = (1-F_cross[:,0])*(1-F_cross[:,1])*(1-F_cross[:,2])
    
    F_map,S_map = seeSaliencyMap (F_cross, super_pixels)
    
    S_ave = np.mean(S_orig)
    
    res = scipy.optimize.minimize(costFunction, S_orig, args=(S_ave, weight_matrix), 
                              jac=costJacFunction, method="L-BFGS-B",bounds=bnds[0:len(S_orig)])
    
    F_y = np.resize(res.x,(numInd,1))
    
    S_map,F_map = seeSaliencyMap2 (F_y, super_pixels)
     
    S_star = calculateFinalSaliency(Diag,weight_matrix, numInd, F_y,mew = mew, thresh = thresh, quant =quant, damp = damp2)

    S_map,F_map = seeSaliencyMap2 (S_star, super_pixels)
    
    return S_map
    
def createFusionSaliency(image, N_segments, PyrL=3, Fi = F, b = bnds, l1 = 0.25, l2 = 0.45, l3 =0.3, s2 = -0.1,K =16, 
                         mew=0.99, damp1 =0.001,damp2 =0.001, thresh = 0.8, quant = 90):

    height, width, _ = image.shape

    #print(height, width)

    image_pyramid = {0:image}
    numSeg = {0:N_segments}
    
    Sal = getAggSaliencyMap2(image, N_segments, Fi = Fi, b = b, l1 = l1, l2 = l2, l3 =l3, s2 = s2,K=K, 
                             mew=mew,damp1 =damp1, damp2 = damp2, thresh = thresh, quant = quant)
    
    gaussian = F[22:27,22:27,37]

    for l in range(1,PyrL+1):
        numSeg[l] = math.ceil(numSeg[l-1]/(2**(2*(l-1))))

        img = image_pyramid[l-1]
        img[:,:,0] = scipy.signal.fftconvolve(img[:,:,0], gaussian, mode='same')
        img[:,:,1] = scipy.signal.fftconvolve(img[:,:,1], gaussian, mode='same')
        img[:,:,2] = scipy.signal.fftconvolve(img[:,:,2], gaussian, mode='same')

        img = img/np.max(img)

        image_pyramid[l] = resize(img, (img.shape[0] // 2, img.shape[1] // 2),
                           anti_aliasing=True)
        
    for l in range(1,PyrL+1):
        img = image_pyramid[l]
        NumSeg = numSeg[l]
        
        s = getAggSaliencyMap2(img, NumSeg, Fi = Fi, b = b, l1 = l1, l2 = l2, l3 =l3, s2 = s2,K=K, 
                               mew=mew,damp1 =damp1, damp2 = damp2, thresh = thresh, quant = quant)
        
        s = resize(s, (height, width), anti_aliasing=False)
        
        Sal += s
        
    Sal = Sal/(PyrL+1)
    
    Sal = Sal/(np.max(Sal)+0.000001)
    
    return Sal