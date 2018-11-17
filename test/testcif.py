# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:23:34 2018

@author: Max
"""
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.svm import SVC  #esto para importar el clasificador vectorial
import numpy as np

fig = plt.figure(figsize=(8, 6))
# plot several images
svc_3 = SVC(kernel='linear')  #esto es clasificador o Classifier cuyo modelos es un hiperplano que separa instancias (puntos) de una clase del resto


def entrenamiento_evaluacion():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target, random_state=0)    
    #print(X_train.shape, X_test.shape)
    return X_train, y_train,X_test,y_test

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
    
faces = datasets.fetch_olivetti_faces()
#faces.data.shape
print(faces.DESCR)
print(faces.keys())
print(faces.images.shape)
print(faces.data.shape)
print(faces.target.shape)

x_entrenamiento, y_entrenamiento,x_evaluacion,y_evaluacion=entrenamiento_evaluacion()

svc_3.fit(x_entrenamiento,y_entrenamiento)
y_pred = svc_3.predict(x_entrenamiento)
#print(x_evaluacion.shape)

eval_faces = [np.reshape(a, (64, 64)) for a in x_evaluacion]
imprimir_rostros(eval_faces, y_pred, 10)

plt.show() #esto es para abrir un frame donde se pegan las imagenes

def mostrarImagenes():
    for i in range(15):
        ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(faces.images[i], cmap=plt.cm.bone)

