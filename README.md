# Birdify - Identificador de Aves

**Birdify** es una aplicación web que utiliza un modelo de aprendizaje automático para identificar especies de aves a partir de imágenes. Con esta aplicación, los usuarios pueden cargar una imagen de un ave y obtener una predicción de la especie, junto con el nivel de confianza de la predicción.

La aplicación está construida utilizando Flask como framework web y Keras/TensorFlow para el modelo de clasificación de imágenes. Cuando un usuario sube una imagen de un ave, la aplicación procesa la imagen y predice la especie utilizando un modelo previamente entrenado. El resultado incluye el nombre de la especie y el porcentaje de coincidencia con la predicción.

## Características

- **Subida de imágenes**: Los usuarios pueden cargar imágenes de aves desde su dispositivo.
- **Clasificación de especies**: El modelo predice la especie del ave a partir de la imagen proporcionada.
- **Interfaz simple**: La interfaz de usuario es intuitiva y fácil de usar, con un diseño limpio y atractivo.
- **Respuestas claras**: Después de la predicción, el usuario recibe un mensaje con el nombre de la especie y el porcentaje de coincidencia.

## Dependencias

El proyecto requiere las siguientes dependencias:

- **Flask**: Framework web para construir la API.
- **Flask-Cors**: Soporte para solicitudes CORS (Cross-Origin Resource Sharing).
- **OpenCV**: Para el procesamiento de imágenes.
- **NumPy**: Biblioteca para manipulación de matrices y arrays.
- **TensorFlow**: Framework de aprendizaje automático para construir y ejecutar el modelo de clasificación.
- **Keras**: API de alto nivel para la construcción de redes neuronales en TensorFlow.
- **Pillow**: Biblioteca para abrir, manipular y guardar imágenes.
- **Gunicorn**: Servidor WSGI para desplegar la aplicación en producción.
- **Matplotlib**: (opcional) Para la visualización de imágenes y gráficos.

## Instrucciones de instalación