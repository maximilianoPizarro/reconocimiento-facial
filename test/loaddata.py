# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:41:50 2018

@author: Max
"""
import os
import cv2.cv2
from sklearn.svm import SVC  #esto para importar el clasificador vectorial
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float,data

from sklearn.datasets import load_files
from sklearn.linear_model import LinearRegression


def prepare_training_data(data_folder_path):
         
        #------STEP-1--------
        #get the directories (one directory for each subject) in data folder
        dirs = os.listdir(data_folder_path)
        #print(dirs)
         
        #list to hold all subject faces
        faces = []
        #list to hold labels for all subjects
        labels = []
         
        #let's go through each directory and read images within it
        for dir_name in dirs:
             
            #our subject directories start with letter 's' so
            #ignore any non-relevant directories if any
            if not dir_name.startswith("s"):
                continue;
              
            #------STEP-2--------
            #extract label number of subject from dir_name
            #format of dir name = slabel
            #, so removing letter 's' from dir_name will give us label
            label = int(dir_name.replace("s", ""))
             
            #build path of directory containing images for current subject subject
            #sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dir_name
             
            #get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path) 
            #------STEP-3--------
            #go through each image name, read image, 
            #detect face and add face to list of faces
            for image_name in subject_images_names:
             
                #ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue;
                 
                #build image path
                #sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name
                
                image = img_as_float(cv2.imread(image_path))                
                
                #read image
           #     info = np.iinfo(image_path.dtype) # Get the information of the incoming image type
            #    print(info)
                
                #image = cv2.imread(img_as_float(image_path))
                
                 
                #display an image window to show the image 
                cv2.imshow("Training on image...", image)
                cv2.waitKey(100)
                #detect face
               # face= detect_face(image)
                face= image    
                #------STEP-4--------
                #for the purpose of this tutorial
                #we will ignore faces that are not detected
                if face is not None:
                    #add face to list of faces
                    faces.append(face)
                    #add label for this face
                    #labels.append(label)
                     
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    cv2.destroyAllWindows()
         
        return faces
    
def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area    
    x, y, w, h = faces[0]
    #return only the face part of the image
    return faces[0]    
    






def entrenamiento_evaluacion():
    X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target,test_size=0.25, random_state=0)    
    #print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

def imprimir_rostros(images, target, top_n):
    # configuramos el tamanio de las imagenes por pulgadas
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # graficamos las imagenes en una matriz de 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.gray())
        # etiquetamos las imagenes con el valor objetivo (target value)
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
        
svc_3 = SVC(kernel='linear')        
faces=load_files('..\\resources\\') 
faces.data=prepare_training_data('..\\resources\\')      

x_entrenamiento,x_evaluacion, y_entrenamiento,y_evaluacion=entrenamiento_evaluacion()

imprimir_rostros(faces.data,faces.target,7)

M=[]
M.append(x_entrenamiento)
M.append(np.asarray(x_entrenamiento).shape)
N=[]
N.append(y_entrenamiento)
N.append(np.asarray(y_entrenamiento))

print(np.asarray(x_entrenamiento).shape)
print(np.asarray(y_entrenamiento).shape)

model = LinearRegression()
#model.fit(M,N)
    
        

#svc_3.fit(np.asarray(x_entrenamiento,dtype=np.float64),np.asarray(y_entrenamiento,dtype=np.float64))
#svc_3.score(x_entrenamiento, y_evaluacion)

#y_pred = svc_3.predict(x_entrenamiento)
#print(x_evaluacion.shape)

#eval_faces = [np.reshape(a, (64, 64)) for a in x_evaluacion]
#imprimir_rostros(eval_faces, y_pred, 10)

plt.show() #esto es para abrir un frame donde se pegan las imagenes     
    
