import os
import numpy as np
import time
import cv2
import tensorflow

import threading

import neuralNetwork as neuralNetwork

minValue = 70

x0 = 200
y0 = 100
height = 200
width = 200

saveImg = False
guessGesture = False

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


binaryMode = True
backgroundSubMode = False
mask = 0
background = 0
counter = 0
mod = 0

numOfSamples = 301 # number of images per group
gestureGroupName = ""
path = ""

counter = 0

menu =  '''\nMenu
    1 OpenCV
    2 Model training with keras
    3 Quit	
    '''

def saveROIImg(img):
    global counter, gestureGroupName, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestureGroupName = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestureGroupName + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.05)



def binaryMask(frame, x0, y0, width, height, framecount):
    global guessGesture, mod, saveImg
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    #roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0+height, x0:x0+width]

    kernel = np.ones((10,10),np.uint8) # moje
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel) # moje
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 0:
        #ores = cv2.UMat.get(res)
        t = threading.Thread(target=neuralNetwork.guessGesture, args = [mod, res])
        t.start()

    return res

# Subtracting background method
def backgroundSubMask(frame, x0, y0, width, height, framecount):
    global guessGesture, takeBackgroundSubMask, mod, background, saveImg
        
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    #roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    #Take background image
    if takeBackgroundSubMask == True:
        background = roi
        takeBackgroundSubMask = False
        print("Refreshing background image for mask...")		

    
    #Take a difference between ROI and background
    diff = cv2.absdiff(roi, background)

    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        
    mask = cv2.GaussianBlur(diff, (3,3), 5)
    mask = cv2.erode(diff, skinkernel, iterations = 1)
    mask = cv2.dilate(diff, skinkernel, iterations = 1)
    res = cv2.bitwise_and(roi, roi, mask = mask)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 0:
        t = threading.Thread(target=neuralNetwork.guessGesture, args = [mod, res])
        t.start()
        #t.join()
    
    return res
	
	
def Main():
    global guessGesture, mod, binaryMode, backgroundSubMode, mask, takeBackgroundSubMask, x0, y0, width, height, saveImg, gestname, path

    font = cv2.FONT_HERSHEY_DUPLEX 
    
    size = 0.5
    fx = 10
    fy = 350
    fh = 18

        
    #Load neural network
    while True:
        answer = int(input(menu))
        if answer == 1:
            mod = neuralNetwork.loadCNN()
            break
        elif answer == 2:
            mod = neuralNetwork.loadCNN(True)
            neuralNetwork.trainModel(mod)
            input("Press any key to continue")
            break
        
        else:
            print("Quitting")
            return 0
        
    ## Camera input
    cap = cv2.VideoCapture('http://192.168.1.101:4747/mjpegfeed')
    cv2.namedWindow('VideoCapture', cv2.WINDOW_NORMAL)

    # set rt size as 640x480, being hold on as properties
    ret = cap.set(3,640)
    ret = cap.set(4,480)

    framecount = 0
    start = time.time()
    
    while(True):
        ret, frame = cap.read()
        max_area = 0
        
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (640,480))
                      
        if ret == True:
            if backgroundSubMode == True:
                roi = backgroundSubMask(frame, x0, y0, width, height, framecount)
            elif binaryMode == True:
                roi = binaryMask(frame, x0, y0, width, height, framecount)

            
            framecount = framecount + 1
            end  = time.time()
            timediff = (end - start)
            if( timediff >= 1):
                #timediff = end - start
                start = time.time()
                framecount = 0

        cv2.putText(frame,'b - Binary mask',(fx,fy), font, size,(0,255,0),1,1)
        cv2.putText(frame,'x - background subtraction mask',(fx,fy + fh), font, size,(0,255,0),1,1)		
        cv2.putText(frame,'p - prediction',(fx,fy + 2*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'n - Create a new image folder',(fx,fy + 3*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'s - Save new images for training',(fx,fy + 4*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'ESC to exit',(fx,fy + 5*fh), font, size,(0,255,0),1,1)
        
        cv2.imshow('VideoCapture',frame)
        cv2.imshow('ROI', roi)
        
        key = cv2.waitKey(5) & 0xff # get most significant byte
        
        # ESCAPE
        if key == 27:
            break
        
        # binary mask
        elif key == ord('b'):
            binaryMode = True
            backgroundSubMode = False
            print("Binary mask active")
        
	# background subtraction mask
        elif key == ord('x'): 
            takeBackgroundSubMask = True
            backgroundSubMode = True
            print("Backgroudn subtraction mask active")
        
	# prediction 	
        elif key == ord('p'):
            guessGesture = not guessGesture
            print("Predicting - {}".format(guessGesture))
        
        # Adjusting ROI window
        elif key == ord('l'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5

        # Saving images (can be paused/resumed if needed)
        elif key == ord('s'):
            saveImg = not saveImg
            
            if newFolderName != '':
                saveImg = True
            else:
                print("Press 'n' to add image group")
                saveImg = False
        
        # Creating folder for a new label
        elif key == ord('n'):
            gestureGroupName = input("Enter the gesture folder name: ")
            try:
                os.makedirs(gestureGroupName)
            except OSError as e:
                # if directory already present
                if e.errno != 17:
                    print('Cant create directory named ' + gestureGroupName)
            
            path = "./"+gestureGroupName+"/"
        

    #Camera release and destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()

