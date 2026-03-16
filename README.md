# deep_learning_portafolio_implementacion

Detección de Fake News utiliando un modelo entrenado de deep learning.


## Módulo 2 Implementación de un modelo de deep learning. (Portafolio Implementación)

Entregable: Implementación de un modelo de deep learning.

1. Crea un repositorio de github para este proyecto.
2. Selecciona un problema y consigue un dataset para resolver dicho problema. 
3. Implementa una arquitectura de deep learning para solucionar el problema. Lo que se busca es que apliques correcta y efectivamente las técnicas vistas en el módulo.
4. Analiza los resultados de tu modelo set de pruebas y validación.
5. Mejora tu modelo usando técnicas de regularización, ajustando hiper parámetros, modificando la arquitectura de tu modelo o buscando otro modelo. 
6. Documenta y explica cuáles son los cambios que funcionaron y por qué funcionaron. 
7. Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.
8. Después de la entrega intermedia se te darán correcciones que puedes incluir en tu entrega final.


## Contenido

- Notebook de limpieza de datos -> etl_final.ipynb
- Notebook de entrenamiento -> ent_final.ipynb
- Notebook de predicciones -> pred_final.ipynb
- Reporte -> EvidenciaMod2_PaulinaLopezCuevas_A01701095.pdf
- Archivo de modelo guardado -> modelo_bueno.keras
- Dataset para pruebas -> test_dataset-20251107T153823Z-1-001.zip
- Archivo del vobulario para el modelo -> vocab.txt

## Objetivo
- Realizar un proceso completo de ETL de datos textuales.
- Desarrollar modelos de clasificación binaria con Deep Learning.
- Aplicar técnicas de procesamiento de lenguaje natural (NLP).
- Optimizar la arquitectura mediante regularización e hiperparámetros.
- Evaluar la estabilidad y generalización del modelo con diferentes datasets.
- Generar un sistema de predicción funcional para nuevos textos.

## Dataset
### Versión final utilizada:
- Fuente: Kaggle – Fake News & True News datasets
- Total de muestras: 42,834
- Clases balanceadas:
  - Fake news → 0
  -   True news → 1
- Atributo utilizado: Título de la noticia
- Idioma: Inglés

## Implementación Técnica
### Procesamiento de datos
- Eliminación de columnas irrelevantes
- Limpieza de caracteres especiales y normalización a minúsculas
- Tokenización + Vectorización con TextVectorization (Keras)
- División 70/15/15 (train/validation/test)
- Balanceo de clases por muestreo
- Shuffle para evitar sesgos de entrenamiento

### Arquitectura del modelo (Versión Final)
- Capa Embedding
- 2 capas Bidirectional LSTM
- SpatialDropout1D, Dropout
- GlobalMaxPooling1D
- BatchNormalization
- Regularización L2
- Penalización por clase (class weights) para mejorar detección de Fake News

### Métricas utilizadas
- Binary Cross-Entropy
- Accuracy
- Precision, Recall
- AUC
- Matriz de Confusión para análisis de errores

## Conclusiones
El modelo final logró una capacidad de detección sólida y estable, demostrando que los títulos pueden ser un buen indicador para filtrar noticias potencialmente falsas. Sin embargo, la credibilidad real de una noticia no depende únicamente del título: medio, autor, fecha, contexto y fuentes adicionales también influyen.

Este sistema puede utilizarse como:
- Herramienta de pre-filtrado
- No sustituye verificación periodística completa

Posibles extensiones:
- Uso combinado de título + texto + metadatos
- Modelos más avanzados de NLP como Transformers
- Deployment en una API o aplicación web interactiva

## Requisitos Técnicos
- Python 3.8+
- TensorFlow / TensorFlow Text
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

Guadalupe Paulina López Cuevas

A01701095
