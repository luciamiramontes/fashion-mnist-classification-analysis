"""
Fashion MNIST Classification Analysis
This project performs exploratory data analysis and machine learning on the
Fashion-MNIST dataset. It includes binary classification using K-Nearest
Neighbors (KNN) and multiclass classification using Decision Trees, along
with model evaluation through cross-validation and confusion matrices.
"""

#El codigo se encuentra dividido en 3 secciones principales. En la primera parte se encuentra las funciones definidas para evitar
#repeticion de codigo. En la segunda parte incluye todo lo que esta relacionado con el analisis exploratorio de datos. 
#Por ultimo, la tercera parte incluye el armado de modelos predictivos que se divide en dos secciones. Una seccion para la 
#clasificacion binaria y otra donde estan los modelos de clasificacion multiclase.


#%% Importamos las bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
#%%

#Importamos el archivo a analizar
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Fashion-MNIST.csv"
file = pd.read_csv(DATA_PATH, index_col=0)

#%%============================================================================
#FUNCIONES
#%%============================================================================
def probar_knn_con_pixeles(pixeles, X_train, X_test, y_train, y_test, nombre_clases):
    
    """Crea el modelo knn con los pixeles que se le dan y prueba con distintos valores de k-vecinos"""
    """Tambien calcula la exactitud y grafica la matriz de confusion del modelo"""

    #Seleccionamos distintos valores de k-vecinos mas cercanos para probar en los modelos
    ks = [5,10,20,35,50]
    train = X_train[pixeles]
    test = X_test[pixeles]

    print(f"\n===== Resultados con píxeles: {pixeles} =====\n")
    
    for k in ks:
        clasificador = KNeighborsClassifier(n_neighbors=k)
        clasificador.fit(train, y_train)
        prediccion = clasificador.predict(test)
        
        exactitud = accuracy_score(y_test, prediccion)
        matriz_confusion = confusion_matrix(y_test, prediccion)
        
        print(f"Exactitud con {k} vecinos: {exactitud}")
        print(f"Matriz de confusión con {k} vecinos:\n{matriz_confusion}\n")
        graficar_matriz_confusion_knn(matriz_confusion, nombre_clases, k)

def graficar_matriz_confusion_knn(matriz, nombre_clases, k):
    
    """Grafica matriz de confusion"""
    
    plt.figure()
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", 
                xticklabels=nombre_clases, yticklabels=nombre_clases)
    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title(f"Matriz de confusión con {k} vecinos")
    plt.tight_layout()
    plt.show()
    
def graficar_clase(clase, X, y):
    
    """Grafica 8 ejemplos de una clase"""
    
    plt.figure(figsize=(8,3))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(X[y == clase].iloc[i].values.reshape(28, 28), cmap='gray')
        plt.title(f"Clase {clase}")
        plt.axis('off')

    plt.show()
    
def graficar_dos_clases(clase1, clase2, X, y):
    
    """Grafica 5 ejemplos de 2 clases"""
    
    plt.figure(figsize=(6,3))
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(X[y == clase1].iloc[i].values.reshape(28, 28), cmap='gray')
        plt.title(f"Clase {clase1}")
        plt.axis('off')

    for i in range(5):
        plt.subplot(2, 5, i+6)
        plt.imshow(X[y == clase2].iloc[i].values.reshape(28, 28), cmap='gray')
        plt.title(f"Clase {clase2}")
        plt.axis('off')

    plt.show()
#%%============================================================================
#ANALISIS EXPLORATORIO DE DATOS
#%%============================================================================

#Informacion general del Dataset
file.head()

file.describe()
#%%

# Obtenemos la cantidades de clases de ropa distinta (label)
file['label'].unique() # Vemos que hay del 0 al 9 
#%%

#  Cantidad de ejemplos por etiqueta
file['label'].value_counts().sort_index() # Hay 7000 de cada uno
#%%

# Vemos el tipo de dato de las columnas
file.dtypes 
#%%

# Extraer los píxeles sin la etiqueta
X = file.drop('label', axis=1)

# Extraer etiquetas
y = file['label']
#%%

#Ploteamos una imagen de ejemplo
img = np.array(X.iloc[7]).reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(f"Clase {y.iloc[7]}")
plt.axis('off')
plt.show()
#%%

#Graficamos una prenda de cada clase para ver si identificamos la prenda
plt.figure(figsize=(15, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X[y == i].iloc[0].values.reshape(28, 28), cmap='gray')
    plt.title(f"Clase {i}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(
    BASE_DIR / "outputs" / "figures" / "fashion_mnist_classes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
  
# Vemos que:
    # 0: Remera
    # 1: Pantalón
    # 2: Sueter
    # 3: Vestido
    # 4: Campera
    # 5: Sandalia
    # 6: Camisa
    # 7: Zapatilla
    # 8: Cartera
    # 9: Bota
#%% 

#Analizamos diferencias entre clase 1 y 2

graficar_dos_clases(1, 2, X, y)
#%% 

#Analizamos diferencias entre clase 2 y 6

graficar_dos_clases(2, 6, X, y)
#%% 

#Analizamos varias imagenes de la clase 8

graficar_clase(8, X, y)
#%% 

#Analizamos varias imagenes de la clase 9

graficar_clase(9, X, y)
#%%

#Graficamos un promedio de cada clase para ver la forma que tienen y usarlo en el armado del modelo
clases_prom = file.groupby('label').mean()
X_prom = clases_prom


plt.figure(figsize=(20,8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_prom.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Clase {i}")
    plt.axis()
    plt.grid()

plt.show()
#%%============================================================================
#ARMADO DE MODELO (CLASIFICACION BINARIA)
#%%============================================================================

#Guaardamos los nombres de las clases para los graficos
nombre_clases = ["Remera", "Cartera"] 

#Tomamos solo las clases 0 y 8 
clases_0_y_8 = file[(file['label'] == 0) | (file['label'] == 8)]

clases_0_y_8['label'].value_counts()
#%%

#Separamos los datos en variables explicativas y la variable objetivo

# Extraer los píxeles sin la etiqueta
X_0_y_8 = clases_0_y_8.drop('label', axis=1)

# Extraer etiquetas
y_0_y_8 = clases_0_y_8['label']
#%%

# Separamos los datos en conjuntos de train y test (20 % para el test y 80% para el train)

X_train, X_test, y_train, y_test = train_test_split(X_0_y_8, y_0_y_8, test_size=0.2)
#%%

#Usamos nuestros graficos de promedio para comparar las clases 0 y 8

cero_y_ocho = [0, 8]
plt.figure(figsize=(10, 5))

for i, clase in enumerate(cero_y_ocho):
    plt.subplot(1, 2, i + 1)  
    plt.imshow(X_prom.iloc[clase].values.reshape(28, 28), cmap='gray')
    plt.title(f"Clase {clase}")
    plt.axis()
    plt.grid()

plt.tight_layout()
plt.show()
#%% 

#Buscamos 3 pixeles muy diferentes entre las clases de "clases_prom" (565, 98, 120)

#Graficamos la ubicacion de esos pixeles en el grafico promedio
cero_y_ocho = [0, 8]
plt.figure(figsize=(10, 5))

for i, clase in enumerate(cero_y_ocho):
    plt.subplot(1, 2, i + 1)  # Cambiar a 1 fila, 2 columnas
    plt.imshow(X_prom.iloc[clase].values.reshape(28, 28), cmap='gray')
    plt.scatter(565 % 28, 565 // 28, color='red', s=100, marker='o')
    plt.scatter(98 % 28, 98 // 28, color='red', s=100, marker='o')
    plt.scatter(120 % 28, 120 // 28, color='red', s=100, marker='o')
    plt.title(f"Clase {clase}")
    plt.axis()
    plt.grid()

plt.tight_layout()
plt.show()
#%%

#Hacemos el modelo con los pixeles sleccionados
pixeles = ["pixel565", "pixel98", "pixel120"]
probar_knn_con_pixeles(pixeles, X_train, X_test, y_train, y_test, nombre_clases)
#%% 

#Probamos otra estrategia de seleccion de atributos

#Hacemos la suma total de los pixeles por clase y luego la diferencia entre estas
cant_clase_0 = clases_0_y_8[clases_0_y_8['label'] == 0].sum()
cant_clase_8 = clases_0_y_8[clases_0_y_8['label'] == 8].sum()
diferencia = abs(cant_clase_0 - cant_clase_8)

#Ordenamos de mayor a menor
diferencia.sort_values(ascending = False) 
#Mientras mas sea la diferencia, mayor diferencia de color en ese pixel entre las clases

#%%        
        
#1era prueba con 3 pixeles - 554, 526, 538 (top 3 de diferencia)
pixeles = diferencia.sort_values(ascending=False).head(3).index.tolist()
probar_knn_con_pixeles(pixeles, X_train, X_test, y_train, y_test, nombre_clases)
#%% 

#2da prueba con 3 pixeles - 39, 610, 369 (tienen mucho valor pero estan separados en el grafico)
pixeles = ["pixel39", "pixel610", "pixel369"]
probar_knn_con_pixeles(pixeles, X_train, X_test, y_train, y_test, nombre_clases)
#%% 

#3era prueba 3 pixeles - 0, 1, 27 (top 3 de menos valor)
#Queremos ver que la exactitud sea baja

pixeles = diferencia.sort_values(ascending=True).head(3).index.tolist()
probar_knn_con_pixeles(pixeles, X_train, X_test, y_train, y_test, nombre_clases)
#%% 

#1era prueba con 10 pixeles (10 con mas diferencia)
pixeles = diferencia.sort_values(ascending=False).head(10).index.tolist()
probar_knn_con_pixeles(pixeles, X_train, X_test, y_train, y_test, nombre_clases)
#%% 

#1era prueba con 100 - pixeles 
pixeles = diferencia.sort_values(ascending=False).head(100).index.tolist()
probar_knn_con_pixeles(pixeles, X_train, X_test, y_train, y_test, nombre_clases)
#%%============================================================================
#ARMADO DE MODELO (CLASIFICACION MULTICLASE)
#%%============================================================================

#Separamos los datos en conjunto de desarrollo (dev) y valdiacion (eval)
X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=0.1) #90% desarrollo y 10% validacion
#%%

#Probamos un modelo de arbol simple con distintas alturas
alturas = [1,3,5,10]
for h in alturas:
    modelo = DecisionTreeClassifier(max_depth = h, criterion="entropy") # Es un poquito mejor que gini
    modelo.fit(X_dev, y_dev)
    prediccion = modelo.predict(X_dev)
    accuracy = accuracy_score(y_dev, prediccion)
    print(f"Exactitud para altura maxima = {h}: {accuracy:.2f}") 
#%%
########## PUEDE TARDAR VARIOS MINUTOS EN CORRER ##########
# Este bloque realiza validación cruzada y puede tardar varios minutos, dependiendo del equipo.
# Ahora entrenamos un modelo usando validación cruzada con k-folding
nsplits = 5
kf = KFold(n_splits=nsplits)

resultados = np.zeros((nsplits, len(alturas)))
# Una fila por cada fold, una columna por cada modelo

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas):
        
        arbol = DecisionTreeClassifier(max_depth = hmax, criterion = "entropy", min_samples_split=8, random_state=10)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        score = accuracy_score(kf_y_test,pred)
        
        resultados[i, j] = score
        
# Promedio scores sobre los folds
scores_promedio = resultados.mean(axis = 0)

for i,e in enumerate(alturas):
    print(f'Score promedio del modelo con hmax = {e}: {scores_promedio[i]:.4f}')

#%%
###############################################################
# ESTE BLOQUE REALIZA LA BÚSQUEDA COMPLETA DE HIPERPARÁMETROS.
# Puede tardar aproximadamente 30 minutos (o más, según el equipo).
###############################################################
RUN_LONG_ANALYSIS = False

if RUN_LONG_ANALYSIS:
    #Probamos con mas hiperparametros para ver
    alturas = [5,10]
    min_samples_splits = [2, 8, 10, 20]
    min_samples_leafs = [1, 5, 10]
    max_features_list = [None, 'log2', 'sqrt']
    
    nsplits = 5
    kf = KFold(n_splits=nsplits)
    
    mejor_score = 0
    mejor_config = None
    
    # Guardar resultados
    resultados = []
    
    #Hacemos un GridSearch para buscar el mejor modelo
    for h in alturas:
        for mss in min_samples_splits:
            for msl in min_samples_leafs:
                for mf in max_features_list:
    
                    scores_fold = []
    
                    for train_idx, test_idx in kf.split(X_dev):
                        X_train, X_test = X_dev.iloc[train_idx], X_dev.iloc[test_idx]
                        y_train, y_test = y_dev.iloc[train_idx], y_dev.iloc[test_idx]
    
                        modelo = DecisionTreeClassifier(
                            max_depth=h,
                            min_samples_split=mss,
                            min_samples_leaf=msl,
                            max_features=mf,
                            criterion="entropy",
                            random_state=0
                        )
                        modelo.fit(X_train, y_train)
                        pred = modelo.predict(X_test)
                        acc = accuracy_score(y_test, pred)
                        scores_fold.append(acc)
    
                    score_promedio = np.mean(scores_fold)
                    resultados.append(((h, mss, msl, mf), score_promedio))
    
                    print(f"Profundidad={h}, min_split={mss}, min_leaf={msl}, max_feat={mf}: acc={score_promedio:.4f}")
    
                    if score_promedio > mejor_score:
                        mejor_score = score_promedio
                        mejor_config = (h, mss, msl, mf)
    
    print("\n Mejor configuración:")
    print(f"max_depth={mejor_config[0]}, min_samples_split={mejor_config[1]}, min_samples_leaf={mejor_config[2]}, max_features={mejor_config[3]}")
    print(f"Score promedio en CV: {mejor_score:.4f}")
    
    # Mejor configuración:
    # max_depth=10, min_samples_split=8, min_samples_leaf=1, max_features=None
    # Score promedio en CV: 0.8084
#%%

# Entreno el mejor modelo elegido en el conjunto dev entero
arbol_elegido = DecisionTreeClassifier(max_depth = 10, criterion = "entropy", min_samples_split=8, min_samples_leaf=1, max_features=None, random_state=10)
arbol_elegido.fit(X_dev, y_dev)
y_pred = arbol_elegido.predict(X_dev)

score_arbol_elegido_dev = accuracy_score(y_dev, y_pred)
print(f"El score del arbol elegido es: {score_arbol_elegido_dev}")

# Pruebo el modelo elegid y entrenado en el conjunto eval
y_pred_eval = arbol_elegido.predict(X_eval)       
score_arbol_elegido_eval = accuracy_score(y_eval, y_pred_eval)
print(f"El score de la evaluacion es: {score_arbol_elegido_eval}")
matriz_de_confusion = confusion_matrix(y_eval, y_pred_eval)
print(f"La matriz de confusion es: \n {matriz_de_confusion}")
#%%

#Graficamos la matriz de confusion
nombre_clases = ["Remera", "Pantalón", "Sueter", "Vestido", "Campera", "Sandalia", "Camisa", "Zapatilla", "Cartera", "Bota"]
   

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_de_confusion, annot=True, fmt="d", cmap="Reds", 
            xticklabels=nombre_clases, yticklabels=nombre_clases)
plt.title("Matriz de Confusión - Fashion MNIST (Árbol de Decisión)")
plt.ylabel("Clase real")
plt.xlabel("Clase predicha")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(
    BASE_DIR / "outputs" / "figures" / "decision_tree_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()