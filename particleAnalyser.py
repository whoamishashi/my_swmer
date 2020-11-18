#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:25:18 2020

@author: shashikant
"""
import math
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 2
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from sklearn.cluster import OPTICS
from sklearn.neighbors import LocalOutlierFactor
import cv2
import random

def classifyShape(img):
    
    """ Get raw image """
    image = cv2.imread(img) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # 17*15=255, 17*8=136, 17*6=102
    
    """"Morphing"""
    kernel= np.ones((5,5),np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations =5)
    morph = cv2.dilate      (morph, kernel, iterations =1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations =5)
    morph = cv2.dilate      (morph, kernel, iterations =1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations =5)
    morph = cv2.dilate      (morph, kernel, iterations =1)
    morph = cv2.medianBlur  (morph, 5)
    
    """Detect shape"""
    #get contours
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contours_good = [c for c in contours if cv2.contourArea(c)>1000]
    contours_draw = np.zeros(morph.shape, np.uint8)
    cv2.drawContours(contours_draw, contours_good, -1, (255,255,255), -1)

    #get largest contour
    areas = [cv2.contourArea(c) for c in contours_good]
    largest_contour = contours_good[np.argmax(areas)]
    areas[np.argmax(areas)] = 0
    second_largest_contour = contours_good[np.argmax(areas)]

    #get centroids
    mask = np.zeros(morph.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], 0, (255,255,255), -1)
    M = cv2.moments(mask)
    centX_contour0, centY_contour0 = (int(M["m10"] / M["m00"]),  int(M["m01"] / M["m00"]))
    ellipse=cv2.fitEllipse(largest_contour)
    ((centx,centy), (MA, ma), angle) = ellipse
    centX_ellipse0, centY_ellipse0 = (int(centx), int(centy))
    cv2.ellipse(contours_draw,ellipse,(255,255,255),1)
    cv2.circle(contours_draw, (centX_ellipse0, centY_ellipse0), 7, (255, 0, 0) , 10)
    cv2.circle(contours_draw, (centX_contour0, centY_contour0), 7, (0, 255, 0), 10)
    
    #identify
    dist = math.sqrt( (centX_contour0-centX_ellipse0)**2 + (centY_contour0-centY_ellipse0)**2 )
    identifier = ''
    if dist < 25:
        largest_contour_area = cv2.contourArea(largest_contour)
        second_largest_contour_area = cv2.contourArea(second_largest_contour)
        if (largest_contour_area > 7*second_largest_contour_area): 
            identifier= 'disc'
        else:
            isNested= cv2.pointPolygonTest(largest_contour, tuple(second_largest_contour[0][0]), False)
            if isNested > 0:
                if (largest_contour_area > 5*second_largest_contour_area):
                    identifier= 'disc-like ring'
                else:
                    identifier= 'ring'
            else: 
                identifier= 'disc'
    elif dist>25 and dist <50:
        identifier = 'long arc'
    else:
        identifier= 'short arc'
    return identifier

def processCellid(file_npy):

    """ get data """
    cellid      = file_npy.split('cellid_')[-1].split('.dat')[0]
    v_BEcoordinates = np.load (file_npy)
    nParticles = len(v_BEcoordinates)

    """ KDE """
    v_BEcoordinates = v_BEcoordinates[v_BEcoordinates[:,2].argsort()]
    Z = np.array(v_BEcoordinates[:,2])
    sampleSize = 10000
    if nParticles > sampleSize:
        randomList = random.sample(range(nParticles), sampleSize)
        Z = Z[randomList]
    Z = Z.reshape(-1,1)

    kde = KernelDensity(kernel='gaussian', bandwidth=100).fit(Z)
    Z_plot = np.linspace(Z.min(),Z.max(), len(Z))[:, np.newaxis]
    dens = np.exp(kde.score_samples(Z_plot))
    min_ind = argrelextrema(dens, np.less)[0]
    max_ind = argrelextrema(dens, np.greater)[0]
    partition_idx = None
    if len(min_ind)>0 and len(max_ind)>1:
        partition_idx = int(min_ind[-1])
    
    sampleSize =400 
    percentage_weak = 50/100
    
    v_BE_reduced = []
    if nParticles > sampleSize:
        if partition_idx:
            nIndices = int(sampleSize*percentage_weak) if sampleSize*percentage_weak < partition_idx else partition_idx
            randomList1 = random.sample(range(0, partition_idx), nIndices)
            randomList2 = random.sample(range(partition_idx, nParticles), int(sampleSize-nIndices))
            randomList = np.concatenate([randomList1, randomList2])
            v_BE_reduced = v_BEcoordinates[randomList, :]
        else:
            randomList = random.sample(range(0, nParticles), int(sampleSize))
            v_BE_reduced = v_BEcoordinates[randomList, :]
    else:
        v_BE_reduced = v_BEcoordinates

    """ OPTICS """
    min_samples = int(sampleSize/10)
    model = OPTICS(min_samples= min_samples).fit(v_BE_reduced)
    values, counts = np.unique(model.labels_, return_counts=True)
    min_particles = int(sampleSize/100)
    bad_index = np.where(counts<min_particles)
    Nvalues = np.delete(values, bad_index)
    Npopulations = len(Nvalues)
    
    """ poster plot """
    fig = plt.figure()
    vpll = v_BE_reduced[:,2]
    vperp = v_BE_reduced[:,0]**2 + v_BE_reduced[:,1]**2
    vperp = np.sqrt(vperp)
    plt.scatter(vpll, vperp, c=model.labels_)
    plt.gca().set_xlim(-1000, 1000)
    plt.gca().set_ylim(0, 1000)
    if 'alphas' in cellid : plt_title = 'He++ : cellid = ' + cellid.split('_')[0]
    if 'protons' in cellid: plt_title = 'H+ : cellid = ' + cellid.split('_')[0]
    plt.title(plt_title)
    plt.xlabel('v_{parallel} [km/s]'); plt.ylabel('v_{perpendicular} [km/s]')
    # plt.savefig('/home/shashikant/Desktop/Esa/submitted_vpll_vperp/' + str(cellid)+'.png')

    """ classify """
    shapes = ['sphere']
    
    if Npopulations > 1:
        sampleSize =10000
        percentage_weak = 100/100
        v_BE_reduced = []
        if nParticles > sampleSize:
            if partition_idx:
                nIndices = int(sampleSize*percentage_weak) if sampleSize*percentage_weak < partition_idx else partition_idx
                randomList = random.sample(range(0, partition_idx), nIndices)
                v_BE_reduced = v_BEcoordinates[randomList, :]
            else:
                randomList = random.sample(range(0, nParticles), int(sampleSize))
                v_BE_reduced = v_BEcoordinates[randomList, :]
        else:
            v_BE_reduced = v_BEcoordinates
        
        """ LOR """      
        n_neighbors = int(sampleSize/10)
        modelLOR = LocalOutlierFactor(n_neighbors=n_neighbors) 
        modelLOR.fit_predict(v_BE_reduced)
        lof = modelLOR.negative_outlier_factor_ 
        thresh = np.quantile(lof, .1)
        index = np.where(lof<=thresh)
        
        """ image processing """
        v_BE_reduced = np.delete(v_BE_reduced, index, axis=0)
        fig = plt.figure(figsize = (10,10))
        plt.scatter(v_BE_reduced[:,0], v_BE_reduced[:,1])
        plt.axis('off')
        plt.subplots_adjust(bottom = 0)
        plt.subplots_adjust(top = 1)
        plt.subplots_adjust(right = 1)
        plt.subplots_adjust(left = 0)
        img = 'xy.png'
        plt.savefig(img)

        classifyShape_retval = classifyShape(img)
        if isinstance(classifyShape_retval, str):
            shapes.insert(0, classifyShape_retval)
    
    return (cellid, shapes)

def main():
    cellids_mercury = [6930198]
    root_mercury = r'/home/shashikant/Desktop/Esa/my_data/particle_velocity_space_files/vBE_particles_mercury_runset06run01/'
    
    for cellid in cellids_mercury:   
        (cellid, shapes) = processCellid(root_mercury + 'vBE_particles_cellid_'+str(cellid)+'_sw_alphas.dat.npy')
        # (cellid, shapes) = processCellid(root_mercury + 'vBE_particles_cellid_'+str(cellid)+'_sw_protons.dat.npy')
        print (cellid, shapes)

if __name__ == "__main__":
    main()
