from skimage.data import camera
from skimage.filters import frangi
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import color, data, io
from sklearn.metrics import accuracy_score,balanced_accuracy_score, jaccard_score
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score

def color_filter(img,circles, red, green):
    img2 = img.copy()
    for w in range(img_width):
        for h in range(img_height):            
            if(in_circle(circles,w,h)): # tylko dla tych bedących pikselami oka
                #zwieksz lekko kolor czerwony
                img2[h,w][2]=min(255,int(img[h,w][2]*red))
                #zwieksz barwe zielona, proporcjonalnie do jej nasycenia
                img2[h,w][1]=min(int(img[h,w][1]*green),255)
    #cv2.imshow('green_filter',img)druga 
    #cv2.waitKey(0)
    return img2

def detect_circle(img): #poszukuje obrysu oka
     img2=img.copy()
     imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     min_length=min(img.shape[0],img.shape[1])
     max_length=max(img.shape[0],img.shape[1])
     minradius=int(0.4*min_length)
     #alg. HoughCircles znajduje koło na zdjęciu, na wejciu img w skali szarosci
     circles = cv2.HoughCircles(imgGray,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=minradius,maxRadius=int(max_length*0.5))
     circles[0][0][2] = circles[0][0][2] -1
     if circles is not None:
         circles = np.uint16(np.around(circles))#zaokragla war. do war. calkowitych
         x=int(circles[0][0][0])
         y=int(circles[0][0][1])
         r=int(circles[0][0][2])
         if x!=0 and y!=0 and r!=0:    
            for i in circles[0,:]:
                cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),1) # rysuje koło
            #cv2.imshow('circle',img2)
            #cv2.waitKey(0)
     return circles

#sprawdza, czy dany piksel nalezy do oczodolu (true), czy do tla(false)
def in_circle(circles,w,h,less=0):
    in_c=False
    if circles is not None:
        if (circles[0][0][0]-w)**2+(circles[0][0][1]-h)**2<(circles[0][0][2]-less)**2:
            in_c=True
    return in_c

def delete_circle(binaryImage,circles):
    binDel = binaryImage.copy()
    for w in range(img_width):
        for h in range(img_height):            
            if(not in_circle(circles,w,h)):
                binDel[h,w] = False
                binDel[h,w] = False
                binDel[h,w]= False
    return binDel

#def change_brightness(img, value=30):
    #imgBright = img.copy()
    #hsv = cv2.cvtColor(imgBright, cv2.COLOR_BGR2HSV)
    #h, s, v = cv2.split(hsv)
    #v = cv2.add(v,value)
    #v[v > 255] = 255
    #v[v < 0] = 0
    #final_hsv = cv2.merge((h, s, v))
    #imgBright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #return imgBright



def calculate_metrics(eyeBinary,goldPath,circles):
    goldMask = cv2.cvtColor(cv2.imread('C:\\Users\\darek\\Desktop\\Pyramid Games\\PythonApp\\PythonApp\\source_pics\\Image_05L_1stHO.png'), cv2.COLOR_BGR2GRAY) 
    goldBinary = goldMask>120
    goldBinary = delete_circle(goldBinary,circles)
    
    acc = accuracy_score(goldBinary.flatten(), eyeBinary.flatten())
    spec = specificity_score(goldBinary.flatten(), eyeBinary.flatten(), average='weighted')
    sen = sensitivity_score(goldBinary.flatten(), eyeBinary.flatten(), average='weighted')
    
    acc
    sen
    spec
    
    cv2.imshow('eyeFrangi', np.uint8(eyeBinary*255))
    cv2.imshow('eyeGold', np.uint8(goldBinary*255))
    cv2.waitKey(0)
    
    print('acc - ' + str(acc))
    print('spec - ' + str(spec))
    print('sen - ' + str(sen))
    return acc, spec ,sen

eyeBaseDir = 'C:\\Users\\darek\\Desktop\\Pyramid Games\\PythonApp\\PythonApp\\source_pics\\Image_05L.jpg'
image = cv2.imread(eyeBaseDir)
img_height=image.shape[0]
img_width=image.shape[1]

circles=detect_circle(image.copy())

#bright =change_brightness(image, 20)
greenImg = color_filter(image,circles,1.00,1.50)

frangiGreen = frangi(cv2.cvtColor(greenImg, cv2.COLOR_BGR2GRAY))

binary2 = (frangiGreen*100000 > 0.09) # fuknckaj tworzaca 0-1 czarno-biale zdjecie
binary2 = delete_circle(binary2,circles)

#cv2.imshow('great frangi',np.uint8(binary2*255))
#cv2.waitKey(0)

acc, sen, spec = calculate_metrics(binary2,"",circles)




