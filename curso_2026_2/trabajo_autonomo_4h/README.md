# Trabajo autónomo: visión por computador en cuatro horas

Dos notebooks guiados de **120 minutos cada uno**, en español y autocontenidos.
Parten de la teoría matemática ya vista; incluyen una introducción a cómo ejecutar
y leer las celdas sin experiencia previa programando redes. No es necesario recorrer
los códigos de 04 en adelante para realizar esta actividad. Se concentra en visión;
generativos, secuencias y refuerzo quedan como material de consulta posterior.

| Orden | Notebook | Qué aprenderás |
|---|---|---|
| 1 | [Pillow, OpenCV y CNN](01_pillow_opencv_cnn.ipynb) | Abrir fotos, recortar, interpolar, guardar, convertir RGB/BGR/HSV, suavizar, detectar bordes, crear máscaras y entrenar una CNN |
| 2 | [Transferencia y red personalizada](02_transfer_inception_resnet152_personalizadas.ipynb) | Comparar InceptionV3 y ResNet152 congeladas; implementar ramas multiescala, proyecciones 1×1 y conexiones residuales |

## Abrir y ejecutar

### Alternativa con celular e internet, sin programar

El [taller con celular y papel](TALLER_CELULAR_SIN_PROGRAMAR.md) dura **cuatro horas**
y tiene una dificultad ligeramente mayor: exige calcular, auditar errores y diseñar
bajo restricciones. Incluye observación de fotografías reales, filtros a mano,
conteo de parámetros, un bloque Inception-Residual, transferencia y evaluación sin fuga.
No requiere ejecutar los notebooks ni instalar aplicaciones. Se entrega con fotos
de las respuestas o un documento escrito en el celular.

Tiene ejemplos guiados con números distintos a los ejercicios y referencias
oficiales. Puede usarse como alternativa a la ruta programada o como
actividad adicional de cuatro horas. Prepara habilidades de análisis y diseño similares;
la ejecución y depuración se practican posteriormente con los notebooks.

### Ruta con notebooks

1. Abre cada notebook y pulsa su distintivo **Abrir en Colab**. El enlace apunta a
   `master` y requiere que estos cambios estén publicados. Para usar los archivos
   locales de inmediato: [Colab](https://colab.research.google.com/) → Archivo → Subir notebook.
2. Inicia sesión en Colab con tu cuenta de Google y usa un entorno Python estándar. Puedes solicitar GPU T4 gratuita en
   Entorno de ejecución → Cambiar tipo de entorno. No se necesitan cuentas de Kaggle,
   Drive, credenciales ni suscripción.
3. Ejecuta en orden. La configuración detecta GPU; sin GPU activa un subconjunto
   reducido automáticamente. El segundo notebook empieza desde cero y no depende del primero.
4. Lee, predice lo que ocurrirá y ejecuta. Realiza **un reto obligatorio por notebook**.
   Las extensiones y la repetición con otro dataset quedan fuera de las cuatro horas.
5. Guarda copias con salidas y descarga los archivos de resultados desde Archivos.

## Datos y presupuesto de recursos

La ruta principal usa **Flower Photos**, 3.670 fotografías de cinco clases publicadas
con el tutorial de TensorFlow. La descarga es de unos 218 MB y las atribuciones CC-BY
se conservan en `LICENSE.txt` dentro de los datos. El segundo notebook también ejecuta
**CIFAR-10**, fotos de objetos de 32×32 y diez clases, como alternativa (~120 MB).
Se descarga solo train en Parquet desde la [distribución pública uoft-cs](https://huggingface.co/datasets/uoft-cs/cifar10),
con verificación SHA-256. Colab incluye el lector PyArrow; el entorno local lo declara
en `requirements-tensorflow-2026.txt`. No hace falta una cuenta de Hugging Face.
Las muestras pequeñas se seleccionan después de descargar: no reducen la transferencia inicial.

| Configuración automática | CPU reducido | GPU |
|---|---:|---:|
| Fotos por clase | 20 | 100 |
| Train / validación / test por clase | 12 / 4 / 4 | 60 / 20 / 20 |
| Resolución para las redes | 96×96 | 128×128 |
| Tamaño de lote | 8 | 16 |
| Épocas CNN / red propia | 3 | 5 |
| Máximo de épocas de cabeza transferida | 10 | 10 |

Las bases descargan además unos 85 MB (InceptionV3) y 230 MB (ResNet152), sin la
cabeza ImageNet. Se extraen características una vez, se elimina cada base antes
de cargar la siguiente y se entrenan cabezas pequeñas. La red propia tiene menos
de 200.000 parámetros. No se entrena ResNet152 completa ni se hace fine-tuning en
la ruta obligatoria. El modo reducido es didáctico: no produce una clasificación
fiable de arquitecturas con tan pocos datos.

Las cuatro horas son una **planificación pedagógica**, no una garantía de tiempo
de cómputo. [Colab tiene disponibilidad y límites variables](https://research.google.com/colaboratory/faq.html).
Si hay lentitud o memoria insuficiente, activa `MODO_MINIMO=True` y vuelve a ejecutar
desde arriba. Reduce `BATCH` a 4 si hace falta; no reduzcas `LADO` por debajo de 75
para InceptionV3. Ante error de descarga, revisa conexión y reintenta la celda;
no se sustituyen los datos reales por datos artificiales.

## Entrega

- Mosaico de transformaciones y explicación de RGB/BGR, recorte y filtros.
- CNN, curva de pérdida y lectura de errores en la matriz de confusión.
- Comparación InceptionV3/ResNet152, parámetros totales frente a entrenados.
- Red propia y experimento de una sola modificación, con hipótesis y evidencia.
- Particiones sin fuga, conclusiones y limitaciones; no se exige accuracy mínima.

Entrega ambos `.ipynb` con salidas, `metricas_cnn.json`, `comparacion_transfer.csv`,
`configuracion.json` y los dos `particiones.json` (renómbralos por notebook al descargarlos).
Las celdas de retos tienen tablas para registrar observaciones; no bloquean “Ejecutar todo”.
La selección se hace con validación. El test se consulta al final y no se usa para afinar modelos.

## Referencias y mantenimiento

### Dificultad y alcance de la revisión

Consulta la [revisión de dificultad, cobertura y comprobaciones](VERIFICACION.md).

La ruta programada empieza por ejemplos completos y deja dos modificaciones puntuales:
cambiar `FILTROS_INICIALES` y `USAR_ATAJO`. No exige implementar bucles de entrenamiento
desde cero. Ejecuta hasta los retos, realiza la comparación con validación y después
continúa al test final. La ruta con celular tiene ejemplos resueltos con otros números
y conserva como obligatorios los conceptos de todos los temas; cálculos repetitivos
y ampliaciones de memoria quedan opcionales.

Las cuatro horas por ruta son una estimación para estudiantes con la teoría matemática
ya estudiada, incluyendo lectura y autocorrección; no una medición con estudiantes.
No se espera hacer ambas rutas, todas las extensiones y ambos datasets en cuatro horas.
La preparación conceptual del taller es comparable; practicar código requiere los notebooks.

- [Pillow: Image](https://pillow.readthedocs.io/en/stable/reference/Image.html).
- [OpenCV: espacios de color](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html).
- [TensorFlow: imágenes y atribución de flores](https://www.tensorflow.org/tutorials/load_data/images).
- [CIFAR-10: fuente y autores](https://www.cs.toronto.edu/~kriz/cifar.html).
- [Keras: transferencia y fine-tuning](https://keras.io/guides/transfer_learning/).
- [Keras Applications](https://keras.io/api/applications/).

Para regenerar estos dos notebooks: `python tools/build_autonomous_vision.py` desde
la raíz. Validación de estructura y sintaxis de todo el curso:
`python tools/validate_notebooks.py`. La ejecución con descargas y TensorFlow requiere
un entorno compatible, como Colab; la validación estática no certifica entrenamiento.
