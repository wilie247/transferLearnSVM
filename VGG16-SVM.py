# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 00:30:35 2021

@author: Femi William
"""


import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
import random

import hdf5storage



AF = [3, 8, 13, 19, 24, 29]


FR1 = [4, 15]
FR2 = [5, 14]
FC = [6, 13]
TM =[7, 12]
PA = [8, 11]
OC = [9, 10]
labels = {}


SIZE = 256  #Resize images


#############################

subs = [8]


tEnd = 128


def channelSector22():
    
    
    for x in subs:
            
        # Directory
        #x = subs[x]
        directory = 'S'+str(x)
      
       
        os.mkdir('imagesT/' + directory)
        os.mkdir('imagesT/' + directory + '/Face')
        os.mkdir('imagesT/' + directory + '/FaceNameVoiced')
      
        x = x -1
        labels[x] = hdf5storage.loadmat('Labels.mat')['Labels']['FaceOnly'][0,x]
        myEEG = hdf5storage.loadmat('EEG_Datasets.mat')['EEG']['FaceOnly'][0,x]
        myEEGRecog = hdf5storage.loadmat('EEG_Datasets.mat')['EEG']['Face_AudioName'][0,x]
        baseLine = hdf5storage.loadmat('EEG_Datasets.mat')['EEG']['FOnlyBaseLine'][0,x]
        realChansBaseLine = np.mean(baseLine[3:17, 0:128, :], axis=2)
        
        for i in range(myEEG.shape[2]):
            myEEGreal = myEEG[3:17, 0:128, i]
            myEEGrealRecog = myEEGRecog[3:17, 0:128, i]
            
          
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
                img = img[img != 0]
                
                print(img)
                print("===================================")
                
                
                Allimg[counter-1,:] = img
                    
                Rimg = myEEGRecog[start:end, 0:tEnd, i]
                Rimg = Rimg[Rimg != 0]
                AllimgR[counter-1,:] = Rimg
                
                print(Rimg)
                print("===================================")
            
            plt.imshow(Allimg, aspect = 'auto', cmap="jet") 
            plt.axis('off')
            #plt.show() 
            plt.savefig('imagesT/' + directory + '/Face/' + str(i) +".png", bbox_inches='tight',pad_inches = 0)
           
            
            plt.imshow(AllimgR, aspect = 'auto', cmap="jet") 
            plt.axis('off')
            #plt.show() 
            
           

            plt.savefig('imagesT/' + directory + '/FaceNameVoiced/' + str(i) +".png", bbox_inches='tight',pad_inches = 0)

 

subs = [3]

def callAnalizer():
    
    for x in subs:
        x = x -1
        labels[x] = hdf5storage.loadmat('Labels.mat')['Labels']['Face_AudioName'][0,x]
    
    
    import math
    
    for t in subs:
        
        print("subject " + str(t))
        labInuse = labels.get(t-1)[0]
        labFamil = [idx for idx, element in enumerate(labInuse) if element==0]
        
        total = len(labInuse)#)
        print(total)
        
        indx = list(range(0, total))
        train = list(range(0, math.floor(0.8 * total)))
        train = [indx[k] for k in train]
        validate = list(range(len(train), len(train) + math.floor(0.1 * total)))
        validate = [indx[k] for k in validate]
        test = list(range(len(train), total))
        test = [indx[k] for k in test]
        
       # print(os.listdir("imagesRV/S" +str(t)+"/"))
        
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
            fruit_label = directory_path.split("\\")[-1]
            for img_path in glob.glob(os.path.join(directory_path, "*.png")):
                fn = os.path.splitext(os.path.basename(img_path))[0]
                if int(fn) in validate:
                    #print(img_path)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    test_images.append(img)
                    test_labels.append(fruit_label)
        
        #print(len(test_images)) 
        #print(len(test_labels))     
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        
        
        #tEST
        Ttest_images = []
        Ttest_labels = [] 
        for directory_path in glob.glob("imagesRV/S"+str(t)+"/*"):
            fruit_label = directory_path.split("\\")[-1]
            for img_path in glob.glob(os.path.join(directory_path, "*.png")):
                fn = os.path.splitext(os.path.basename(img_path))[0]
                if int(fn) in test:
                    #print(img_path)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    Ttest_images.append(img)
                    Ttest_labels.append(fruit_label)
        
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
        
        #############################
        
        VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

        activation = 'relu'
        #Add layers for deep learning prediction
        
        x = Flatten()(VGG_model.output)
        #x = Dense(64, activation = activation, kernel_initializer = 'he_uniform')(x)
        prediction_layer = Dense(2, activation = 'sigmoid')(x)
        
        # Make a new model combining both feature extractor and x
        myVGG_model = Model(inputs=VGG_model.input, outputs=prediction_layer)
        myVGG_model.compile(optimizer='rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        
       
        #Load model wothout classifier/fully connected layers
        
        #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
        for layer in VGG_model.layers:
        	layer.trainable = False
            
        #VGG_model.summary()  #Trainable parameters will be 0



        #Now, let us use features from VGG convolutional network for RF
        feature_extractor=VGG_model.predict(x_train)
        

        features = feature_extractor.reshape(feature_extractor.shape[0], -1)   
        
        X_for_RF = features #This is our X input to RF
            
        
        # Make a new model combining both feature extractor and x
        #VGG_model = Model(inputs=features, outputs=prediction_layer)
        #VGG_model.compile(optimizer='rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])
        # Add top and fit
        
        
        
        print("printing VGG16 CNN summary")
        #print(VGG_model.summary()) 
        
        ##########################################
        #Train the VGG model
        #history = myVGG_model.fit(x_train, y_train_one_hot, epochs=10, validation_data = (x_test, y_test_one_hot))
        
        
        #plot the training and validation accuracy and loss at each epoch
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs = range(1, len(loss) + 1)
        # plt.plot(epochs, loss, 'y', label='Training loss')
        # plt.plot(epochs, val_loss, 'r', label='Validation loss')
        # plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()
        
        
        # acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # plt.plot(epochs, acc, 'y', label='Training acc')
        # plt.plot(epochs, val_acc, 'r', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()
        
        
        print("CNN Results from validation")
        prediction_NN = myVGG_model.predict(x_test)
        prediction_NN = np.argmax(prediction_NN, axis=-1)
        prediction_NN = le.inverse_transform(prediction_NN)
        
        # #Confusion Matrix - verify accuracy of each class
        # #Confusion Matrix - verify accuracy of each class
        from sklearn.metrics import confusion_matrix
        from sklearn import metrics
        from sklearn.metrics import f1_score
        
        cm = confusion_matrix(test_labels, prediction_NN)
        print("printing VGG16 CNN Confusion matrix on validation set")
        print(cm)
        print ("CNN Val Accuracy = ", sum(np.diagonal(cm))/sum(map(sum, cm)))
        print("f1 Score =", f1_score(test_labels_encoded, le.transform(prediction_NN)))
        sns.heatmap(cm, annot=True)
        
        # #Check results on a
        
        
        print("printing VGG16CNN perf Test SET")
        prediction = myVGG_model.predict(xx_test)
        prediction = np.argmax(prediction, axis=-1)  #argmax to convert categorical back to original
        prediction = le.inverse_transform(prediction)  #Reverse the label encoder to original name
        
        cm2 = confusion_matrix(Ttest_labels, prediction)
        print("printing VGG16CNN Confusion matrix on TEST set")
        print(cm2)
        print ("VGG16CNN TEST Accuracy = ", sum(np.diagonal(cm2))/sum(map(sum, cm2)))
        print("f1 Score =", f1_score(Ttest_labels_encoded, le.transform(prediction)))
        sns.heatmap(cm2, annot=True)
        
        ################################
       
        #RANDOM FOREST
        
        print("SVM STARTS")
        from sklearn import svm
        feature_extractor=VGG_model.predict(x_train)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
       
        X_for_svm = features #This is out X input to SVM
        svm_model = svm.SVC(kernel = "rbf", gamma = 2)
        
        # Train the model on training data
        svm_model.fit(X_for_svm, y_train) #For sklearn no one hot encoding
        
        #Send test data through same feature extractor process
        feature_extractor=VGG_model.predict(x_test)
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
        print("CONFMAT")
        #Confusion Matrix - verify accuracy of each class
        cm = confusion_matrix(test_labels, prediction_RF)
        print(cm)
        sns.heatmap(cm, annot=True)
        
        #Check results on a few select images
        #n=5 #dog park. RF works better than CNN
        print("printing SVM perf Test SET")
        feature_extractor=VGG_model.predict(xx_test)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
        input_img_features=features
        
        prediction_RF = svm_model.predict(input_img_features)
        prediction_RF = le.inverse_transform(prediction_RF)  #Reverse the label encoder to original name
        print("SVM Accuracy = ", metrics.accuracy_score(Ttest_labels, prediction_RF))
        print("f1 Score =", f1_score(Ttest_labels_encoded, le.transform(prediction_RF)))
        print("RF CONFMAT")
        print(metrics.confusion_matrix(Ttest_labels, prediction_RF))
    

import math
import random
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
        
        #############################
        
        VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

        activation = 'relu'
        #Add layers for deep learning prediction
        
        x = Flatten()(VGG_model.output)
        #x = Dense(64, activation = activation, kernel_initializer = 'he_uniform')(x)
        prediction_layer = Dense(2, activation = 'sigmoid')(x)
        
        # Make a new model combining both feature extractor and x
        myVGG_model = Model(inputs=VGG_model.input, outputs=prediction_layer)
        myVGG_model.compile(optimizer='rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        
       
        #Load model wothout classifier/fully connected layers
        
        #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
        for layer in VGG_model.layers:
        	layer.trainable = False
            
        #VGG_model.summary()  #Trainable parameters will be 0



        #Now, let us use features from VGG convolutional network for RF
        feature_extractor=VGG_model.predict(x_train)
        

        features = feature_extractor.reshape(feature_extractor.shape[0], -1)   
        
        X_for_RF = features #This is our X input to RF
            
        
        # Make a new model combining both feature extractor and x
        #VGG_model = Model(inputs=features, outputs=prediction_layer)
        #VGG_model.compile(optimizer='rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])
        # Add top and fit
        
        
        
        print("printing VGG16 CNN summary")
        #print(VGG_model.summary()) 
        
        ##########################################
        #Train the VGG model
        #history = myVGG_model.fit(x_train, y_train_one_hot, epochs=10, validation_data = (x_test, y_test_one_hot))
        
        
        #plot the training and validation accuracy and loss at each epoch
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # epochs = range(1, len(loss) + 1)
        # plt.plot(epochs, loss, 'y', label='Training loss')
        # plt.plot(epochs, val_loss, 'r', label='Validation loss')
        # plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()
        
        
        # acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # plt.plot(epochs, acc, 'y', label='Training acc')
        # plt.plot(epochs, val_acc, 'r', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()
        
        
        print("CNN Results from validation")
        prediction_NN = myVGG_model.predict(x_test)
        prediction_NN = np.argmax(prediction_NN, axis=-1)
        prediction_NN = le.inverse_transform(prediction_NN)
        
        # #Confusion Matrix - verify accuracy of each class
        # #Confusion Matrix - verify accuracy of each class
        from sklearn.metrics import confusion_matrix
        from sklearn import metrics
        from sklearn.metrics import f1_score
        
        cm = confusion_matrix(test_labels, prediction_NN)
        print("printing VGG16 CNN Confusion matrix on validation set")
        print(cm)
        print ("CNN Val Accuracy = ", sum(np.diagonal(cm))/sum(map(sum, cm)))
        print("f1 Score =", f1_score(test_labels_encoded, le.transform(prediction_NN)))
        sns.heatmap(cm, annot=True)
        
        # #Check results on a
        
        
        print("printing VGG16CNN perf Test SET")
        prediction = myVGG_model.predict(xx_test)
        prediction = np.argmax(prediction, axis=-1)  #argmax to convert categorical back to original
        prediction = le.inverse_transform(prediction)  #Reverse the label encoder to original name
        
        cm2 = confusion_matrix(Ttest_labels, prediction)
        print("printing VGG16CNN Confusion matrix on TEST set")
        print(cm2)
        print ("VGG16CNN TEST Accuracy = ", sum(np.diagonal(cm2))/sum(map(sum, cm2)))
        print("f1 Score =", f1_score(Ttest_labels_encoded, le.transform(prediction)))
        sns.heatmap(cm2, annot=True)
        
        ################################
       
        
        
        print("SVM STARTS")
        from sklearn import svm
        feature_extractor=VGG_model.predict(x_train)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
       
        X_for_svm = features #This is out X input to SVM
        svm_model = svm.SVC(kernel = "rbf", gamma = 2)
        
        # Train the model on training data
        svm_model.fit(X_for_svm, y_train) #For sklearn no one hot encoding
        
        #Send test data through same feature extractor process
        feature_extractor=VGG_model.predict(x_test)
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
        print("CONFMAT")
        #Confusion Matrix - verify accuracy of each class
        cm = confusion_matrix(test_labels, prediction_RF)
        print(cm)
        sns.heatmap(cm, annot=True)
        
        #Check results on a few select images
        #n=5 #dog park. RF works better than CNN
        print("printing SVM perf Test SET")
        feature_extractor=VGG_model.predict(xx_test)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
        input_img_features=features
        
        prediction_RF = svm_model.predict(input_img_features)
        prediction_RF = le.inverse_transform(prediction_RF)  #Reverse the label encoder to original name
        print("SVM Accuracy = ", metrics.accuracy_score(Ttest_labels, prediction_RF))
        print("f1 Score =", f1_score(Ttest_labels_encoded, le.transform(prediction_RF)))
        print("RF CONFMAT")
        print(metrics.confusion_matrix(Ttest_labels, prediction_RF))
    
    
subs = [5, 6, 7, 8]
rem_forg()  


subs = [17]               
 
channelSector22()
 

callAnalizer()    
     


