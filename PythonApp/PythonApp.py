from skimage.data import camera
from skimage.filters import frangi, hessian, sobel, gabor
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import color, data, io


def color_filter(img,circles, red, blue):
    img2 = img.copy()
    for w in range(img_width):
        for h in range(img_height):            
            if(in_circle(circles,w,h)): # tylko dla tych bedących pikselami oka
                #zwieksz lekko kolor czerwony
                img2[h,w][2]=min(255,int(img[h,w][2]*red))
                #zwieksz barwe zielona, proporcjonalnie do jej nasycenia
                img2[h,w][1]=min(int(img[h,w][1]*blue),255)
    #cv2.imshow('green_filter',img)
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
         print('is cricle')
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



#ustawia tło w usrednionym kolorze pikseli wewnatrz kola
def blur_background(img,circles):
    summ=0
    for w in range(img_width):
      for h in range(img_height):
          if in_circle(circles,w,h):
              summ+=img[h,w]
    
    meann=summ/(img_height*img_width)
    
    for w in range(img_width):
        for h in range(img_height):  
            if img[h,w]<0.15:
                img[h,w]=meann

    return img

def change_brightness(img, value=30):
    imgBright = img.copy()
    hsv = cv2.cvtColor(imgBright, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    imgBright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return imgBright

def delete_circle(binaryImage,circles):
    binDel = binaryImage.copy()
    for w in range(img_width):
        for h in range(img_height):            
            if(not in_circle(circles,w,h)):
                binDel[h,w] = False
                binDel[h,w] = False
                binDel[h,w]= False
    return binDel
def calculate_metrics(eyeBinary,goldPath,circles):
    goldMask = cv2.imread(eyeBaseDir +  'Image_05L_1stHO.png')

     for w in range(img_width):
        for h in range(img_height):   


eyeBaseDir = sys.path[1] + '\\EyeBase\\'


image = cv2.imread(eyeBaseDir +  'Image_05L.jpg')


img_height=image.shape[0]
img_width=image.shape[1]
image3 = cv2.imread(eyeBaseDir +  'Image_05L.jpg')

print(img_height)
print(img_width)

circles=detect_circle(image.copy())
#image=color_filter(image,circles,1.05,1.5)
#cv2.imshow('sharpen',sharpen(image)  )

bright =change_brightness(image, 20)
green = color_filter(bright,circles,1.00,1.50)


#frangiBright = frangi( cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY))
frangiGreen = frangi( cv2.cvtColor(green, cv2.COLOR_BGR2GRAY))

#cv2.imshow('brightfrangi',frangiBright*700000)
#cv2.imshow('greenfrangi',frangiGreen*700000)

#binary = (frangiBright*100000 > 0.066) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#binary = delete_circle(binary,circles)
#binary = np.uint8(binary)
binary2 = (frangiGreen*100000 > 0.063) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
binary2 = delete_circle(binary2,circles)
circles[0][0][2] = circles[0][0][2] +1

acc, sen, spec = calculate_metrics(binary2,"",circles)
binary2 = np.uint8(binary2)

#cv2.imshow('brightbinary',delete_circle(binary,circles) )
#cv2.imshow('sharpenbright',change_brightness(sharpen(image))   )
#cv2.waitKey(0)


fig, ax = plt.subplots(figsize=(6,6))

#cv2.imshow('image2', image2)
#cv2.imshow('image2 gray', frangi(image2))
#cv2.imshow('image2 frangi', image2frangi)
#cv2.waitKey(0)

#cv2.imshow('image3',image3)
#cv2.imshow('image3 gray', cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY))
#cv2.imshow('image3 frangi', image3frangi)
#cv2.waitKey(0)

#cv2.destroyAllWindows()
#ax=fig.add_subplot(3, 4, 1)
#ax.axis('off')
#ax.imshow(image2)

#ax=fig.add_subplot(3, 4, 2)
#ax.axis('off')
#ax.imshow(image3)

#img2 = color.rgb2gray(image2)
#img2=blur_background(img2, circles)

#filt_real, filt_imag = gabor(img2, frequency=1.2)

#plt.figure()            # doctest: +SKIP
#abc = ''
#duppa = frangi(filt_real)
#binary = (duppa*10000 > 0.005) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#binary = np.uint8(binary)
#cv2.imshow('testujemy',binary )
#cv2.waitKey(0)



#image1 = cv2.imread(eyeBaseDir +  'Image_05L.jpg') #czytam zdjec
#image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #filtr na odcienie szarosci
#after = frangi(image) #filtr sobela

#after*=100000
#duppa*=100000

#binary = (after > 0.25) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#binary = np.uint8(binary)
#binary2 = (duppa > 0.25) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#binary2 = np.uint8(binary)

#measure =0.1
#while measure >0.00:
#    binary = (after > measure) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#    binary = np.uint8(binary)
#    binary2 = (duppa > measure) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#    binary2 = np.uint8(binary)
#    cv2.imshow('afterBinary', binary)
#    cv2.imshow('betterOneBinary', binary2)
#    cv2.waitKey(0)
#    measure = measure - 0.01

#cv2.imshow('after', after)
#cv2.imshow('afterBinary', binary)

#cv2.imshow('betterOne', duppa)
#cv2.imshow('betterOneBinary', binary2)
#cv2.waitKey(0)

#after = cv2.normalize(after, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#binary = (after > 0.01) * 255 # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#binary = np.uint8(binary)
#cv2.imshow('kurwa', after)
#cv2.imshow('kurwa2', binary)
#cv2.waitKey(0)


##               TODO
##frangi jest w imshow ax, zostaje lekki obrys środkowego jasniejszego punktu - to też kwestia testów
##czyli ogólnie trzeba teraz: progowanie maski eksperckiej + progowanie(wynik frangiego + usunięta obramka oka)
##wykrozystanie gotowych funkcji z bibliotek(jest w jego dokumentacji na ekursy) do 3 pomiarów
## jak to będzie to mamy wszystko, zrobic test dla 3 lepszych wynikow 2 gorszych i odsyłamy


#ax=fig.add_subplot(3, 4, 3)
#ax.axis('off')
#ax.imshow(duppa)

#ax = fig.add_subplot(3, 4, 4)
#ax.axis('off')
#binary = (duppa > 1.5) * 255  # fuknckaj tworzaca 0-1 czarno-biale zdjecie
#binary = np.uint8(binary)
#ax.imshow(binary)


#plt.show()
##cv2.imshow('image',image)
##cv2.waitKey(0)



