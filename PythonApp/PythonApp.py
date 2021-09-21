from skimage.data import camera
from skimage.filters import frangi, hessian
import cv2
import matplotlib.pyplot as plt
import numpy as np


def color_filter(img,circles, red, blue):
    img2 = img.copy()
    for w in range(img_width):
        for h in range(img_height):            
            if(in_circle(circles,w,h)):
                #zwieksz lekko kolor czerwony
                img2[h,w][2]=min(255,int(img[h,w][2]*red))
                #zwieksz barwe zielona, proporcjonalnie do jej nasycenia
                img2[h,w][1]=min(int(img[h,w][1]*blue),255)
    #cv2.imshow('green_filter',img)
    #cv2.waitKey(0)
    return img2

def detect_circle(img):
     img2=img.copy()
     imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     min_length=min(img.shape[0],img.shape[1])
     max_length=max(img.shape[0],img.shape[1])
     minradius=int(0.4*min_length)
     #alg. HoughCircles znajduje koło na zdjęciu, na wejciu img w skali szarosci
     circles = cv2.HoughCircles(imgGray,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=minradius,maxRadius=int(max_length*0.5))

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

image = cv2.imread('C:\\Users\\dodzi\\Downloads\\CHASEDB1\\Image_01L.jpg')

img_height=image.shape[0]
img_width=image.shape[1]
image3 = cv2.imread('C:\\Users\\dodzi\\Downloads\\CHASEDB1\\Image_01L.jpg')

print(img_height)
print(img_width)

circles=detect_circle(image.copy())
image2=color_filter(image,circles,1.05,1.5)
#image2frangi=frangi(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
image3 = color_filter(image,circles,1.05,1.6)
#image3frangi=frangi(cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY))
fig, ax = plt.subplots(figsize=(6,6))

#cv2.imshow('image2', image2)
#cv2.imshow('image2 gray', frangi(image2))
#cv2.imshow('image2 frangi', image2frangi)
#cv2.waitKey(0)

#cv2.imshow('image3',image3)
#cv2.imshow('image3 gray', cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY))
#cv2.imshow('image3 frangi', image3frangi)
#cv2.waitKey(0)

cv2.destroyAllWindows()
ax=fig.add_subplot(3, 4, 1)
ax.axis('off')
ax.imshow(image2)

ax=fig.add_subplot(3, 4, 2)
ax.axis('off')
ax.imshow(image3)

ax=fig.add_subplot(3, 4, 3)
ax.axis('off')
ax.imshow(frangi(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)))

ax=fig.add_subplot(3, 4, 4)
ax.axis('off')
ax.imshow(frangi(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)))

ax=fig.add_subplot(3, 4, 4)
ax.axis('off')
ax.imshow(color_filter(image,circles,1.1,1.7))

plt.show()
#cv2.imshow('image',image)
#cv2.waitKey(0)


