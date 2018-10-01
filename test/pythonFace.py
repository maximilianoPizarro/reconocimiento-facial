#fuente http://ezequielaguilar.com.pa/data-science/reconocimiento-facial-utilizando-python-scikit-learn/

#testeado con python 2.7.9
#complementar el gestor de dependencias https://pip.pypa.io/en/stable/installing/
#			curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#			python get-pip.py
#Descargar las siguientes librerias
#python -m pip install scipy
#python -m pip install scikit-learn   #actualizar si es necesario python -m pip install -U scikit-learn
#python -m pip install matplotlib
#python -m pip install numpy
#python -m pip install IPython

import IPython
import sklearn as sk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC  #esto para importar el clasificador vectorial
from sklearn.datasets import fetch_olivetti_faces   #de aca saco el dataset de imagenes del a libreria
from sklearn.cross_validation import train_test_split #esto es para el test de comparacion
from sklearn import metrics

print('IPython version:', IPython.__version__)
print('numpy version:', np.__version__)
print('scikit-learn version:', sk.__version__)
print('matplotlib version:', matplotlib.__version__)

#indices de imagenes de personas con lentes
glasses = [(10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
           (69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
           (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
           (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
           (330, 339), (358, 359), (360, 369)]

#Pre: images coleccion de imagenes, target elemento imagen, top_n cantidad de elementos	
#post: imprime las imagenes por pantalla

def print_faces(images, target, top_n):
    # configuramos el tamanio de las imagenes por pulgadas
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # graficamos las imagenes en una matriz de 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm)
        # etiquetamos las imagenes con el valor objetivo (target value)
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))

#Pre: clf es el clasificador vectorial de la libreria, x_train,x_test,y_train,y_test la clasificacion del conjunto
#post: imprime el reporte de clasificacion vectorial

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print("Exactitud training set:")
    print(clf.score(X_train, y_train))
    print("Exactitud testing set:")
    print(clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    print("Reporte de Classificador:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

#Pre: array completo
#post: array segmentado

def create_target(segments):
    #creamos un nuevo array "y"
    y = np.zeros(faces.target.shape[0])
    
    # put 1 in the specified segments
    for (start, end) in segments:
        y[start:end + 1] = 1
    return y	


###MAIN###
	
#Importamos el dataset de rostros
faces = fetch_olivetti_faces()

#imprimimos propiedades del dataset faces.data contiene el puntero de la lista y faces.target la lista de imagenes en cuestion
print(faces.DESCR)
print(faces.keys())
print(faces.images.shape)
print(faces.data.shape)
print(faces.target.shape)

#No tenemos que escalar los atributos, porque ya se encuentran normalizados.
print(np.max(faces.data))
print(np.min(faces.data))
print(np.mean(faces.data))		
		
#print_faces(faces.images, faces.target, 20)	#esto es para mostrar el dataset sin clasificar

svc_3 = SVC(kernel='linear')  #esto es clasificador o Classifier cuyo modelos es un hiperplano que separa instancias (puntos) de una clase del resto

#X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0) #Creamos los conjuntos train y test

target_glasses = create_target(glasses)#cargo el conjunto con lentes
X_train, X_test, y_train, y_test = train_test_split(faces.data, target_glasses, test_size=0.25, random_state=0)#Idem con el conjunto con lentes

train_and_evaluate(svc_3, X_train, X_test, y_train, y_test)
y_pred = svc_3.predict(X_test)

eval_faces = [np.reshape(a, (64, 64)) for a in X_test]
print_faces(eval_faces, y_pred, 10)

plt.show() #esto es para abrir un frame donde se pegan las imagenes


		