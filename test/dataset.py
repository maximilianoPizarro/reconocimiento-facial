import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.svm import SVC  #esto para importar el clasificador vectorial
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.datasets.base import Bunch

from sklearn.datasets import load_files

faces=load_files('..\\resources\\')


svc_3 = SVC(kernel='linear')


def entrenamiento_evaluacion():
    X_train, X_test, y_train, y_test = train_test_split(faces.data,faces.target, random_state=0)    
    #print(X_train.shape, X_test.shape)
    return X_train, y_train,X_test

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
        
x_entrenamiento, y_evaluacion,x_evaluacion=entrenamiento_evaluacion()

svc_3.fit(x_entrenamiento,y_evaluacion)
y_pred = svc_3.predict(x_entrenamiento)
#print(x_evaluacion.shape)

eval_faces = [np.reshape(a, (64, 64)) for a in x_evaluacion]
imprimir_rostros(eval_faces, y_pred, 10)

plt.show() #esto es para abrir un frame donde se pegan las imagenes        