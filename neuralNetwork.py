import numpy as np
import os
import csv

from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam

from keras import backend as K
import tensorflow
K.set_image_data_format('channels_first')

import cv2
from matplotlib import pyplot as plt

# image size
img_rows, img_cols = 200, 200

# number of color channels, 1 for grayscale
img_channels = 1

# Training Batch size
batch_size = 32

# Number of output classes 
outputClassesNumber = 5

# Number of epochs
epochNumber = 2

# Total number of convolutional filters to use
filterNumber = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

# data
path = "./"

# Image folder
path2 = './imgfolder_b'

WeightFileName = []

# outputs
output = ["Five", "Nothing", "One", "Three", "Two"]

def modlistdir(path, pattern = None):
    pathList = os.listdir(path)
    fileList = []
    for name in pathList:
        #Ignore hidden folders/files
        if pattern == None:
            if name.startswith('.'):
                continue
            else:
                fileList.append(name)
        elif name.endswith(pattern):
            fileList.append(name)
            
    return fileList


# CNN Model builder
def loadCNN(bTraining = False):
    global get_output
    model = Sequential()
    
    
    model.add(Conv2D(filterNumber, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(filterNumber, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputClassesNumber))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    model.summary()
    model.get_config()
    
    
    if not bTraining :
        WeightFileName = modlistdir('.','.hdf5') # List all the weight files available
        if len(WeightFileName) == 0:
            print('No pretrained weight file found.')
        else:
            print('Found weight files - {}'.format(WeightFileName))
            #Load pretrained weights
            w = int(input("Which weight file to load (enter the index of file, starting from 0): "))
            fname = WeightFileName[int(w)]
            print("loading ", fname)
            model.load_weights(fname)

            layer = model.layers[-1]
            get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    
    
    return model

# Guess gestures with model and passed img (put on loop in trackgesture.py)
def guessGesture(model, img):
    global output, get_output
    #Load image to array and flatten it
    image = np.array(img).flatten()
    
    image = image.reshape(img_channels,img_rows,img_cols) 
    
    # float32
    image = image.astype('float32') 
    
    # normalize image to 0-1
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    # Get prediction
    
    probabilityArray = get_output([rimage, 0])[0]

    # print whole probability array if needed
    
    #print (probabilityArray) 
    
    d = {}
    i = 0
    for items in output:
        d[items] = probabilityArray[0][i] * 100
        i += 1
    
    # Get the output with maximum probability
    import operator
    
    guess = max(d.items(), key=operator.itemgetter(1))[0]
    probability  = d[guess]

    if probability > 65.0:
        print(guess + "  Probability: ", probability)
                
        return output.index(guess)

    else:
        # 'Nothing' should always be indexed as 1
        print("Nothing  Probability: ", probability)
        return 1

def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0])) # open one image to get size
    #image1 = np.array(Image.open(path2 +'/' + imlist[0] +'/' + imlist[0] + '1.png')) # open one image to get size
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')
    

    
    print(immatrix.shape)
    
    input("Press any key")
    
    #Label the images

    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = int(total_images / outputClassesNumber)
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(outputClassesNumber):
        label[int(s):int(r)] = classIndex
        s = r
        r = s + samples_per_class
    
    '''
    # for example: 0-301 img samples group 0, 302-603 group 1 and so on
    '''
    
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
     
    # Split X and y into training and testing sets
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train = X_train / 255
    X_test = X_test / 255
     
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, outputClassesNumber)
    Y_test = np_utils.to_categorical(y_test, outputClassesNumber)
    return X_train, X_test, Y_train, Y_test



def trainModel(model):

    # Initialize the data by labelling and splitting into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers() 

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochNumber,
                 verbose=1, validation_split=0.2)

    visualizeHis(hist)

    answer = input("Save the trained weights - y/n ?")
    if answer == 'y':
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname,overwrite=True)
    else:
        model.save_weights("newWeight.hdf5",overwrite=True) # saving only weights, no resuming needed

def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    range_of_epochs=range(epochNumber)

    plt.figure(1,figsize=(7,5))
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.plot(range_of_epochs,train_loss)
    plt.plot(range_of_epochs,val_loss)
    plt.title('Train loss vs value loss value')
    plt.grid(True)
    plt.legend(['train','val'])
    #plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.plot(range_of_epochs,train_acc)
    plt.plot(range_of_epochs,val_acc)
    plt.title('Train accuracy vs Validation accuracy')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()
