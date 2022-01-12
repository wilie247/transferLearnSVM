# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 00:08:44 2021

@author: Femi William
"""


import keras 

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

import os
import seaborn as sns

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D
from keras import models
from keras.models import Model
import random

import tensorflow as ts

import hdf5storage



labels = {}


SIZE = 256  #Resize images


#############################

subs = [6]


tEnd = 128

for x in subs:
    x = x -1
    labels[x] = hdf5storage.loadmat('Labels.mat')['Labels']['Face_AudioName'][0,x]


subs = [15, 16, 17]
def channelSector22():
    
    
    for x in subs:
        
        # Directory
        #x = subs[x]
        directory = 'S'+str(x)
      
       
        os.mkdir('imagesRV/' + directory)
        os.mkdir('imagesRV/' + directory + '/New')
        os.mkdir('imagesRV/' + directory + '/Test')
      
        x = x -1
        labels[x] = hdf5storage.loadmat('Labels.mat')['Labels']['Face_AudioName'][0,x]
        myEEG = hdf5storage.loadmat('EEG_Datasets.mat')['EEG']['Face_AudioName'][0,x]
        myEEGRecog = hdf5storage.loadmat('EEG_Datasets.mat')['EEG']['Face_AudioNameTesting'][0,x]
        baseLine = hdf5storage.loadmat('EEG_Datasets.mat')['EEG']['FOnlyBaseLine'][0,x]
        realChansBaseLine = np.mean(baseLine[3:17, 0:128, :], axis=2)
        
        for i in range(myEEG.shape[2]):
            myEEGreal = myEEG[3:17, 0:128, i]
            myEEGrealRecog = myEEGRecog[3:17, 0:128, i]
            
            #GET FOR AF Channels
            #myEEG[3+48:3+52, 0:128, i]
           
            counter, end = 0, 3
            img = []
            Allimg = np.zeros((14,40),dtype=np.float64())
            AllimgR = np.zeros((14,40),dtype=np.float64())
            for t in range(3, 70, 5):
                #get AF
                print(t)
                counter = counter + 1
                start = t+52
                end = start + 5
                img = myEEG[start:end, 0:tEnd, i]
                img = np.ma.masked_equal(img,0)
                realXIndexR = np.where(img.mask)[1]
                realXR = np.delete(img,realXIndexR, axis=1)
                
                realXR = realXR.ravel()
                
                print(realXR)
                print("===================================")
                
                
                Allimg[counter-1,:] = realXR
                    
                Rimg = myEEGRecog[start:end, 0:tEnd, i]
                
                Rimg = np.ma.masked_equal(Rimg,0)
                realXIndexR = np.where(Rimg.mask)[1]
                realXR = np.delete(Rimg,realXIndexR, axis=1)
                realXR = realXR.ravel()
                
                AllimgR[counter-1,:] = realXR
                
                print(realXR)
                print("===================================")
            
            plt.imshow(Allimg, aspect = 'auto', cmap="jet") 
            plt.axis('off')
            #plt.show() 
            plt.savefig('imagesRV/' + directory + '/New/' + str(i) +".png", bbox_inches='tight',pad_inches = 0)
           
            
            plt.imshow(AllimgR, aspect = 'auto', cmap="jet") 
            plt.axis('off')
            #plt.show() 
            
           

            plt.savefig('imagesRV/' + directory + '/Test/' + str(i) +".png", bbox_inches='tight',pad_inches = 0)

 
import math
import random

subs = [3, 5, 6, 7, 8, 10, 13]

def rem_forg():
    
    for x in subs:
        x = x -1
        labels[x] = hdf5storage.loadmat('Labels.mat')['Labels']['Face_AudioName'][0,x]

      
    for t in subs:
        print("Classification on " + str(t))
        labInuse = labels.get(t-1)[0]
        labFamil = [idx for idx, element in enumerate(labInuse) if element==1]
        lab_forg = [idx for idx, element in enumerate(labInuse) if element==0]
        
        total = len(labFamil) + len(lab_forg)
        
        indx = labFamil + lab_forg
        random.shuffle(indx)
        
        train = list(range(0, math.floor(0.9 * total)))
        train = [indx[k] for k in train]
        validate = list(range(len(train), len(train) + math.floor(0.1 * total)))
        validate = [indx[k] for k in validate]
        test = list(range(len(train), total))
        test = [indx[k] for k in test]
        
        #print(os.listdir("imagesRV/S" +str(t)+"/"))
        
        #print(train)
        #print(validate)
        #print(test)
        
        SIZE = 128
        
        train_images = []
        train_labels = [] 
        for directory_path in glob.glob("imagesRV/S"+str(t)+"/*"):
            label = directory_path.split("\\")[-1]
            if label == "New": #print(label)
                for img_path in glob.glob(os.path.join(directory_path, "*.png")):
                    fn = os.path.splitext(os.path.basename(img_path))[0]
                    if int(fn) in train:
                        # print(img_path)
                        # print("start")
                        # print(indx.index(int(fn)))
                        # print("after")
                        # print(labInuse[int(fn)])
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
                        img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        train_images.append(img)
                        train_labels.append(labInuse[int(fn)])
                  
        #print(len(train_images)) 
        #print(len(train_labels)) 
        #if label == "New":
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        
        
        
        # VALIDATE
        test_images = []
        test_labels = [] 
        for directory_path in glob.glob("imagesRV/S"+str(t)+"/*"):
            _label = directory_path.split("\\")[-1]
            #print(_label)
            if _label == "New":
                #print(_label)
                for img_path in glob.glob(os.path.join(directory_path, "*.png")):
                    fn = os.path.splitext(os.path.basename(img_path))[0]
                    if int(fn) in validate:
                        # print(img_path)
                        # print("start")
                        # print(indx.index(int(fn)))
                        # print("after")
                        # print(labInuse[int(fn)])
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        test_images.append(img)
                        test_labels.append(labInuse[int(fn)])
        
        #print(len(test_images)) 
        #print(len(test_labels))   
        #if _label == "Test":
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        
        
        #tEST
        Ttest_images = []
        Ttest_labels = [] 
        for directory_path in glob.glob("imagesRV/S"+str(t)+"/*"):
            _label = directory_path.split("\\")[-1]
            if _label == "New":
                for img_path in glob.glob(os.path.join(directory_path, "*.png")):
                    fn = os.path.splitext(os.path.basename(img_path))[0]
                    if int(fn) in test:
                        # print(img_path)
                        # print("start")
                        # print(indx.index(int(fn)))
                        # print("after")
                        # print(labInuse[int(fn)])
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        Ttest_images.append(img)
                        Ttest_labels.append(labInuse[int(fn)])
        
        #print(len(Ttest_images)) 
        #print(len(Ttest_labels))  
        #if _label == "Test":              
        Ttest_images = np.array(Ttest_images)
        Ttest_labels = np.array(Ttest_labels)
        
        
        #Encode labels from text to integers.
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(test_labels)
        test_labels_encoded = le.transform(test_labels)
        le.fit(Ttest_labels)
        Ttest_labels_encoded = le.transform(Ttest_labels)
        le.fit(train_labels)
        train_labels_encoded = le.transform(train_labels)
        
        #Split data into test and train datasets (already split but assigning to meaningful convention)
        x_train, y_train, x_test, y_test, xx_test, yy_test = train_images, train_labels_encoded, test_images, test_labels_encoded,Ttest_images, Ttest_labels_encoded
        
        ###################################################################
        # Normalize pixel values to between 0 and 1
       
        x_train, x_test, xx_test = x_train / 255.0, x_test / 255.0, xx_test/255.0
        
        #One hot encode y values for neural network. 
        from keras.utils import to_categorical
        y_train_one_hot = to_categorical(y_train)
        y_test_one_hot = to_categorical(y_test)
        yy_test_one_hot = to_categorical(yy_test)

        X_train = x_train #np.load(path + "X_train.npy")
        X_test = x_test #np.load(path + "X_test.npy")
        y_train = y_train #np.load(path + "y_train.npy")
        y_test = y_test #np.load(path + "y_test.npy")

        # example of loading the resnet50 model
        from keras.applications.resnet50 import ResNet50
        # load model
        #Res_model = ResNet50()
        # summarize the model
        #Res_model.summary()
        
        Res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
        
        activation = 'relu'
        #Add layers for deep learning prediction
        
        x = Flatten()(Res_model.output)
        #x = Dense(64, activation = activation, kernel_initializer = 'he_uniform')(x)
        prediction_layer = Dense(2, activation = 'sigmoid')(x)
        
        # Make a new model combining both feature extractor and x
        _model = Model(inputs=Res_model.input, outputs=prediction_layer)
        _model.compile(optimizer='rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        
        #Inc_model.summary()
       
        #Load model wothout classifier/fully connected layers
        #layer_name1 = "block1_pool"
        #mymodel = Model(inputs=VGG_model.input, outputs = VGG_model.get_layer(layer_name1).output)
        
        #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
        for layer in Res_model.layers:
        	layer.trainable = False
            
        #VGG_model.summary()  #Trainable parameters will be 0
        #Now, let us use features from Resnet convolutional network for SVM
        feature_extractor= Res_model.predict(x_train)
        

        features = feature_extractor.reshape(feature_extractor.shape[0], -1)   
        
        X_for_RF = features #This is our X input to RF
        
        print("CNN Results from validation")
        prediction_NN = _model.predict(x_test)
        prediction_NN = np.argmax(prediction_NN, axis=-1)
        prediction_NN = le.inverse_transform(prediction_NN)
        
        # #Confusion Matrix - verify accuracy of each class
        # #Confusion Matrix - verify accuracy of each class
        from sklearn.metrics import confusion_matrix
        from sklearn import metrics
        from sklearn.metrics import f1_score
        
        cm = confusion_matrix(test_labels, prediction_NN)
        print("printing ResNET CNN Confusion matrix on validation set")
        print(cm)
        print ("CNN Val Accuracy = ", sum(np.diagonal(cm))/sum(map(sum, cm)))
        print("f1 Score =", f1_score(test_labels_encoded, le.transform(prediction_NN)))
        sns.heatmap(cm, annot=True)
        
        # #Check results on a
        
        
        print("printing ResNET perf Test SET")
        prediction = _model.predict(xx_test)
        prediction = np.argmax(prediction, axis=-1)  #argmax to convert categorical back to original
        prediction = le.inverse_transform(prediction)  #Reverse the label encoder to original name
        
        cm2 = confusion_matrix(Ttest_labels, prediction)
        print("printing ResNET Confusion matrix on TEST set")
        print(cm2)
        print ("InceptionNET TEST Accuracy = ", sum(np.diagonal(cm2))/sum(map(sum, cm2)))
        print("f1 Score =", f1_score(Ttest_labels_encoded, le.transform(prediction)))
        sns.heatmap(cm2, annot=True)
        
        ################################
       
       
        
        print("SVM STARTS")
        from sklearn import svm
        feature_extractor= Res_model.predict(x_train)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
       
        X_for_svm = features #This is out X input to SVM
        svm_model = svm.SVC(kernel = "rbf")
        
        # Train the model on training data
        svm_model.fit(X_for_svm, y_train) #For sklearn no one hot encoding
        
        #Send test data through same feature extractor process
        feature_extractor= Res_model.predict(x_test)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
       
        X_test_feature = features
        #Now predict using the trained RF model. 
        prediction_RF = svm_model.predict(X_test_feature)
        #Inverse le transform to get original label back. 
        prediction_RF = le.inverse_transform(prediction_RF)
        
        print("printing On Validation perf")
        #Print overall accuracy
        from sklearn import metrics
        print ("SVM Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
        
        #Confusion Matrix - verify accuracy of each class
        cm = confusion_matrix(test_labels, prediction_RF)
        print(cm)
        sns.heatmap(cm, annot=True)
        
        #Check results on a few select images 
        #n=5 #dog park. RF works better than CNN
        print("printing SVM perf Test SET")
        feature_extractor= Res_model.predict(xx_test)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
        input_img_features=features
        
        prediction_RF = svm_model.predict(input_img_features)
        prediction_RF = le.inverse_transform(prediction_RF)  #Reverse the label encoder to original name
        print("SVM Accuracy = ", metrics.accuracy_score(Ttest_labels, prediction_RF))
        print("SVM CONFMAT")
        print(metrics.confusion_matrix(Ttest_labels, prediction_RF))


        
subs = [16, 17]
rem_forg()         
 
subs = [3]        
for x in subs:
    x = x -1
    labels[x] = hdf5storage.loadmat('Labels.mat')['Labels']['Face_AudioName'][0,x]

  
for t in subs:
    
    print("Classification for " + str(t))
    labInuse = labels.get(t-1)[0]
    labFamil = [idx for idx, element in enumerate(labInuse) if element==1]
    
    total = len(labInuse) #len(labFamil)
    
    indx = list(range(0, total))
    train = list(range(0, math.floor(0.8 * total)))
    train = [indx[k] for k in train]
    validate = list(range(len(train), len(train) + math.floor(0.1 * total)))
    validate = [indx[k] for k in validate]
    test = list(range(len(train), total))
    test = [indx[k] for k in test]
    
    #print(os.listdir("imagesRV/S" +str(t)+"/"))
    
    #print(train)
    #print(validate)
    #print(test)
    
    SIZE = 128
    
    train_images = []
    train_labels = [] 
    for directory_path in glob.glob("imagesRV/S"+str(t)+"/*"):
        label = directory_path.split("\\")[-1]
        #print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            fn = os.path.splitext(os.path.basename(img_path))[0]
            if int(fn) in train:
                #print(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
                img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                train_images.append(img)
                train_labels.append(label)
              
    #print(len(train_images)) 
    #print(len(train_labels))      
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    
    
    # VALIDATE
    test_images = []
    test_labels = [] 
    for directory_path in glob.glob("imagesRV/S"+str(t)+"/*"):
        _label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            fn = os.path.splitext(os.path.basename(img_path))[0]
            if int(fn) in validate:
                #print(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                test_images.append(img)
                test_labels.append(_label)
    
    #print(len(test_images)) 
    #print(len(test_labels))     
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    
    
    #tEST
    Ttest_images = []
    Ttest_labels = [] 
    for directory_path in glob.glob("imagesRV/S"+str(t)+"/*"):
        _label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            fn = os.path.splitext(os.path.basename(img_path))[0]
            if int(fn) in test:
                #print(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                Ttest_images.append(img)
                Ttest_labels.append(_label)
    
    #print(len(Ttest_images)) 
    #print(len(Ttest_labels))                
    Ttest_images = np.array(Ttest_images)
    Ttest_labels = np.array(Ttest_labels)
    
    
    #Encode labels from text to integers.
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    le.fit(Ttest_labels)
    Ttest_labels_encoded = le.transform(Ttest_labels)
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    
    #Split data into test and train datasets (already split but assigning to meaningful convention)
    x_train, y_train, x_test, y_test, xx_test, yy_test = train_images, train_labels_encoded, test_images, test_labels_encoded,Ttest_images, Ttest_labels_encoded
    
    ###################################################################
    # Normalize pixel values to between 0 and 1
    x_train, x_test, xx_test = x_train / 255.0, x_test / 255.0, xx_test/255.0
    
    #One hot encode y values for neural network. 
    from keras.utils import to_categorical
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    yy_test_one_hot = to_categorical(yy_test)

    X_train = x_train #np.load(path + "X_train.npy")
    X_test = x_test #np.load(path + "X_test.npy")
    y_train = y_train #np.load(path + "y_train.npy")
    y_test = y_test #np.load(path + "y_test.npy")

    # example of loading the resnet50 model
    from keras.applications.resnet50 import ResNet50
    # load model
    #Res_model = ResNet50()
    # summarize the model
    #Res_model.summary()
    
    Res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
    
    activation = 'relu'
    #Add layers for deep learning prediction
    
    x = Flatten()(Res_model.output)
    #x = Dense(64, activation = activation, kernel_initializer = 'he_uniform')(x)
    prediction_layer = Dense(2, activation = 'sigmoid')(x)
    
    # Make a new model combining both feature extractor and x
    _model = Model(inputs=Res_model.input, outputs=prediction_layer)
    _model.compile(optimizer='rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    
    #Inc_model.summary()
   
    #Load model wothout classifier/fully connected layers
    #layer_name1 = "block1_pool"
    #mymodel = Model(inputs=VGG_model.input, outputs = VGG_model.get_layer(layer_name1).output)
    
    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in Res_model.layers:
    	layer.trainable = False
        
    #VGG_model.summary()  #Trainable parameters will be 0
    #Now, let us use features from Resnet convolutional network for SVM
    feature_extractor= Res_model.predict(x_train)
    

    features = feature_extractor.reshape(feature_extractor.shape[0], -1)   
    
    X_for_RF = features #This is our X input to RF
    
    print("CNN Results from validation")
    prediction_NN = _model.predict(x_test)
    prediction_NN = np.argmax(prediction_NN, axis=-1)
    prediction_NN = le.inverse_transform(prediction_NN)
    
    # #Confusion Matrix - verify accuracy of each class
    # #Confusion Matrix - verify accuracy of each class
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    from sklearn.metrics import f1_score
    
    cm = confusion_matrix(test_labels, prediction_NN)
    print("printing ResNET CNN Confusion matrix on validation set")
    print(cm)
    print ("CNN Val Accuracy = ", sum(np.diagonal(cm))/sum(map(sum, cm)))
    print("f1 Score =", f1_score(test_labels_encoded, le.transform(prediction_NN)))
    sns.heatmap(cm, annot=True)
    
    # #Check results on a
    
    
    print("printing ResNET perf Test SET")
    prediction = _model.predict(xx_test)
    prediction = np.argmax(prediction, axis=-1)  #argmax to convert categorical back to original
    prediction = le.inverse_transform(prediction)  #Reverse the label encoder to original name
    
    cm2 = confusion_matrix(Ttest_labels, prediction)
    print("printing ResNET Confusion matrix on TEST set")
    print(cm2)
    print ("InceptionNET TEST Accuracy = ", sum(np.diagonal(cm2))/sum(map(sum, cm2)))
    print("f1 Score =", f1_score(Ttest_labels_encoded, le.transform(prediction)))
    sns.heatmap(cm2, annot=True)
    
    ################################
   
    
   
    
    
    print("SVM STARTS")
    from sklearn import svm
    feature_extractor= Res_model.predict(x_train)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
   
    X_for_svm = features #This is out X input to SVM
    svm_model = svm.SVC(kernel = "rbf")
    
    # Train the model on training data
    svm_model.fit(X_for_svm, y_train) #For sklearn no one hot encoding
    
    #Send test data through same feature extractor process
    feature_extractor= Res_model.predict(x_test)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
   
    X_test_feature = features
    #Now predict using the trained RF model. 
    prediction_RF = svm_model.predict(X_test_feature)
    #Inverse le transform to get original label back. 
    prediction_RF = le.inverse_transform(prediction_RF)
    
    print("printing On Validation perf")
    #Print overall accuracy
    from sklearn import metrics
    print ("SVM Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))
    print("f1 Score =", f1_score(test_labels_encoded, le.transform(prediction_RF)))
    #Confusion Matrix - verify accuracy of each class
    cm = confusion_matrix(test_labels, prediction_RF)
    print(cm)
    sns.heatmap(cm, annot=True)
    
    #Check results on a few select images 
    #n=5 #dog park. RF works better than CNN
    print("printing SVM perf Test SET")
    feature_extractor= Res_model.predict(xx_test)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)
    input_img_features=features
    
    prediction_RF = RF_model.predict(input_img_features)
    prediction_RF = le.inverse_transform(prediction_RF)  #Reverse the label encoder to original name
    print("SVM Accuracy = ", metrics.accuracy_score(Ttest_labels, prediction_RF))
    print("f1 Score =", f1_score(Ttest_labels_encoded, le.transform(prediction_RF)))
    print("SVM CONFMAT")
    print(metrics.confusion_matrix(Ttest_labels, prediction_RF))


    
    
    