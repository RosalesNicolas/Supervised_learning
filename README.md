# Supervised_learning
Repositorio de la materia Aprendizaje Supervisado

Repositorio de respuestas de la materia: Aprendizaje Supervisado (2022), en el marco de la Diplomatura en Ciencia de Datos - Universidad Nacional de Córdoba.

# **DIPLOMATURA 2022**

# Aprendizaje Supervisado

## GRUPO Nº24

## INTEGRANTES:
   - [x] Nico Rosales 
   - [x] Daniel Rubio
   - [x] Diana Fonnegra
   - [x] Clarisa Manzone

----   
En este repositorio se encuentran las entregas correspondientes a la asignatura de _Aprendizaje Supervisado_.

## **Requerimientos - Librerías necesarias**:
   - [x] matplotlib.pyplot
   - [x] pandas
   - [x] seaborn
   - [x] numpy
   - [x] sklearn
   - [x] random
   - [x] missingno
   - [x] scipy
   - [x] XGBoost
   - [x] hyperopt

# Documentación:

   - [x] En este trabajo practico se desarrollaron modelos de clasificación en base al set de datos entregado por la cátedra.
   - [x] Se trata de un dataset derivado de la competencia Spaceship Titanic de Kaggle. Los datos fueron tomados originalmente de la competencia y se generaron los datasets de entrenamiento y evaluación.
   - [x] En la etapa 1 se realiza en proceso de EDA.
   - [x] En la etapa 2 se completa la fase EDA y se procede al curado del dataset.
   - [x] En la etapa 3 se evaluan diferentes tipos de clasificadores. Se registra la precisión de cada clasificador seleccionando Random Forest para su optimización.
   - [x] Se aplica búsqueda aleatoria de hiperparámetros. 
   - [x] Sobre el set de hiperparámetros obtenidos se reajustan los mismos aplicando GridSearch evaluando la sensibilidad en la precisión frente a las variaciones propuestas en la grilla de busqueda.
   - [x] Se obtiene un modelo "óptimo" que predice con 0.8+ de precisión según la evaluación reportada en la competencia de Kaggle.
   - [x] En la etapa 4 se aborda el problema de clasificación mediante una red neuronal.
   - [x] Se optimiza la misma ajustando el layerDropout de forma manual e iterativa, se prueban distintas capas densas para definir el modelo final. (0.80804 de precisión Kaggle).
   - [x] En la etapa 5 se repite el mismo esquema de trabajo aplicado para RandomForest (etapa 3) utilizando XGBoost como clasifificador y la librería Hyperopt como optimizador de hiperparámetros.

 
