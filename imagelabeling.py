#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:51:41 2020

@author: mahmoud
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np
# import time

def imshow_components(labels, img):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("img")
    
    plt.subplot(1,2,2)
    plt.imshow(labeled_img)
    plt.title("img_labeled")
    
    plt.show()

def con4(im, i, j):
    if i == 0:
        c0 = 0    
    else:
        c0 = im[i-1][j]
    
    if j == 0:
        c1 = 0
    else:    
        c1 = im[i][j - 1]
        
    # if j == im.shape[1]-1:
    #     c2 = 0
    # else:
    #     c2 = im[i][j+1]
        
    # if i == im.shape[0]-1:
    #     c3 = 0
    # else:
    #     c3 = im[i+1][j]
        
    la4 = np.array([c0, c1])
    
    return la4

def con8(im, i, j):
    
    if i == 0: c0 = 0
    else: c0 = im[i-1][j]
    
    if j == 0 or i == 0: c1 = 0
    else: c1 = im[i-1][j-1]

    if j == 0: c2 = 0
    else: c2 = im[i][j-1]
    
    # if j == 0 and i == im.shape[0] - 1: c3 = 0
    # else: c3 = im[i+1][j-1]
    
    # if j == im.shape[1] - 1: c4 = 0
    # else: c4 = im[i][j+1]
    
    # if i == im.shape[0] - 1 and j == im.shape[1] - 1: c5 = 0
    # else: c5 = im[i+1][j+1]
    
    # if i == im.shape[0] - 1: c6 = 0
    # else: c6 = im[i+1][j]
    
    if i == 0 or j == im.shape[1] - 1: c7 = 0
    else: c7 = im[i-1][j+1] 

    la8 = np.array([c0, c1, c2, c7])
    
    return la8

# tic = time.clock()



def img_labeled(img):
    
    img_temp = np.array(np.zeros(img.shape))
    labl = 0
    labl_list = np.zeros(100)
    img_result = np.array(np.zeros(img.shape))
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            
                 
            if img[i][j] == 1:
                ck = con8(img_temp, i, j)
                
                if len(ck) == 2:
                    cond_ck = ck[0] + ck[1]
                    
                else:
                    cond_ck = ck[0] + ck[1] + ck[2] + ck[3]                
                    
                if (cond_ck) == 0: 
                    labl            = labl + 1
                    img_temp[i][j]  = labl
                    labl_list[labl] = labl
                
                else:
                    
                    count_label = list(set(ck[np.nonzero(ck)]))
                    
                                    
                    if  len(count_label) == 1:
                        img_temp[i][j] = count_label[0]
    
                    else:                                          
                        # min_labl = min(count_label)
                        min_labl = len(labl_list)
                        for mm in count_label:
                            labl_list[int(mm)] = labl_list[int(labl_list[int(mm)])]
                            if labl_list[int(mm)] < min_labl:
                                min_labl = labl_list[int(mm)]
                            
                            
                        img_temp[i][j] = min_labl       
                        
                        
                        
                        for mm in count_label:
                            
                            labl_temp = labl_list[int(mm)]
                            
                            while labl_temp != min_labl:
                                
                                labl_temp2 = labl_list[int(labl_temp)]
                                labl_list[int(labl_temp)] = min_labl    
                                labl_temp = labl_list[int(labl_temp2)]
                                
                            labl_list[int(mm)] = min_labl
                            
                        # print(labl_list)

                        
    # print(labl_list)
    # print(img_temp)     
    
    labl2_list = np.zeros(labl + 1)
    labl2 = 0
    
    for k in range(1, labl + 1):
         
        if labl_list[k] < k :
            
            lablk         = labl_list[k]
            lablk_temp    = labl2_list[int(lablk)]
            labl2_list[k] = lablk_temp
            
        else:
            labl2         = labl2 + 1
            labl2_list[k] = labl2
            
    
    for i in range(0, img_temp.shape[0]):
        for j in range(0, img_temp.shape[1]):
            
            img_result[i][j] = labl2_list[int(img_temp[i][j])]
    return img_result            
        
def height_obj(img_labeled, obj_label):
    
    ymin = img_labeled.shape[0] + 2
    ymax = 0
    for index, row in enumerate(img_labeled):
        
        if obj_label in row:
            y = index
            # x = row.index(obj_label)
            if y < ymin:
                ymin = y
                
            if y > ymax:
                ymax = y
                
    h = ymax - ymin + 1
    return h
    

# toc = time.clock()
# print("time = ", toc - tic)
if __name__ == '__main__':
    
    add_img = "index.png"
    img = np.array(255, dtype = np.int32) -  cv2.imread(add_img, 0)
    img = img/255
    img[img < 0.5] = 0
    img[img > 0.5] = 1
    
    img_result = img_labeled(img)  
    print(img_result)
    imshow_components(img_result, img)    
    
    h = height_obj(img_result, 3)
    print(h)
