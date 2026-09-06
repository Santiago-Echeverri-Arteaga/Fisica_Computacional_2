# Plan de actualización del repositorio y del curso

> **Estado:** colección inicial completa. Los 29 notebooks planeados están
> implementados; las siguientes iteraciones deben incorporar retroalimentación de
> clase, presentaciones del docente y datasets elegidos para los proyectos.

## 1. Decisiones curriculares

El sílabo de Física Computacional 2 es la fuente principal: el aprendizaje
profundo constituye el núcleo del curso. El sílabo de Física Computacional 1 se
usa como puente para recuperar aprendizaje de máquina clásico. El acta de
concertación determina las fechas y porcentajes de evaluación cuando exista una
diferencia con el sílabo genérico.

Distribución orientativa del tiempo restante:

- 30–35 %: nivelación de aprendizaje de máquina clásico y evaluación;
- 60–65 %: fundamentos y arquitecturas de redes neuronales;
- 5–10 %: panorama de transformers, modelos de lenguaje y temas actuales.

La bibliografía de Ovidiu Calin se utiliza como columna matemática, no como
orden obligatorio de exposición. Las derivaciones se redactan de forma propia y
se conectan con experimentos computacionales reproducibles.

## 2. Cronograma operativo (martes y miércoles, 2 h)

| Fecha | Sesión | Producto de código |
|---|---|---|
| 1–2 sep. | EDA; regresión y regularización; logística y árboles (realizado) | Material existente por revisar |
| 8 sep. | KNN: distancia, escala y sesgo-varianza | Notebook 11, primera mitad |
| 9 sep. | SVM: margen, kernels y búsqueda de hiperparámetros | Notebook 11, segunda mitad |
| 15 sep. | Árboles, bosques aleatorios y bagging | Notebook 12 |
| 16 sep. | Boosting y stacking | Notebook 13 |
| 22 sep. | Clases desbalanceadas y métricas | Notebook 14 |
| 23 sep. | Explicación global/local y modelos sustitutos | Notebook 15 |
| 29 sep. | Parcial 1 | Evaluación concertada |
| 30 sep. | DBSCAN y Mean-Shift | Notebook 20 |
| 6 oct. | K-means, mezclas gaussianas y selección | Notebook 21 |
| 7 oct. | Agrupamiento jerárquico | Notebook 22 |
| 13 oct. | Neurona, capas densas y activaciones | Notebooks 30–31 |
| 14 oct. | Descenso de gradiente y retropropagación | Notebook 32 |
| 20 oct. | Parcial 2 | Evaluación concertada |
| 21 oct. | Pérdidas, optimizadores y regularización | Notebook 33 |
| 27 oct. | Red feedforward en TensorFlow y PyTorch | Notebooks 34–35 |
| 28 oct. | CNN: convolución, LeNet-5 y AlexNet | Notebooks 40–41 |
| 3 nov. | VGG, ResNet, Inception, FractalNet y transferencia | Notebooks 42–43 |
| 4 nov. | RNN, LSTM, GRU y Seq2Seq | Notebooks 60–61 |
| 10–18 nov. | Sustentaciones; síntesis de AE/GAN, RL y transformers | Notebooks 50–51, 62, 70–71 |

Las dos clases de recuperación grabadas se reservan para (1) laboratorio de
activaciones/retropropagación y (2) autoencoders/GAN. Esto evita sacrificar la
franja de sustentaciones. El alcance de RL y LLM es introductorio: fundamentos,
un experimento pequeño y límites; no entrenamiento de un modelo fundacional.

## 3. Mapa de notebooks

### 00 — Fundamentos comunes (2)

- `00_protocolo_evaluacion`: train/validation/test, validación cruzada, métricas,
  búsqueda y fuga de información.
- `01_pipelines_reproducibilidad`: semillas, transformaciones, persistencia y
  trazabilidad del experimento.

### 01 — Nivelación de ML supervisado (6)

- `10_regresion_regularizacion` (migración del material ya dictado).
- `11_knn_svm`.
- `12_arboles_bosques_bagging`.
- `13_boosting_stacking`.
- `14_clases_desbalanceadas`.
- `15_explicabilidad_local_global`.

### 02 — Aprendizaje no supervisado (3)

- `20_dbscan_mean_shift`.
- `21_kmeans_gmm`.
- `22_clustering_jerarquico`.

### 03 — Fundamentos de redes neuronales (6)

- `30_neurona_y_feedforward_desde_cero`.
- `31_activaciones_diseno_y_comparacion`: incluye activaciones creadas por el
  estudiante, derivadas, estabilidad numérica y ablación controlada.
- `32_gradiente_y_backpropagation`.
- `33_perdidas_optimizadores_regularizacion`.
- `34_mlp_tensorflow`.
- `35_mlp_pytorch`.

### 04 — Visión profunda (4)

- `40_convolucion_desde_cero`.
- `41_lenet_alexnet`.
- `42_vgg_resnet_inception_fractalnet`.
- `43_transfer_learning_fisica`.

### 05 — Representaciones generativas (2)

- `50_autoencoders`.
- `51_gan`.

### 06 — Secuencias (3)

- `60_rnn_lstm_gru`.
- `61_seq2seq_atencion`.
- `62_series_temporales_fisicas`.

### 07 — Temas actuales (2)

- `70_aprendizaje_por_refuerzo`.
- `71_transformers_y_llm`: tokenización, atención, preentrenamiento,
  alineamiento, inferencia y un transformer diminuto; incluye qué partes de los
  modelos comerciales son públicas y cuáles no.

### 08 — Proyecto (1)

- `80_plantilla_proyecto_final`: problema, fuente de datos, línea base,
  protocolo, ablaciones, incertidumbre, ética, reproducibilidad y presentación.

## 4. Arquitectura técnica

- Los notebooks de clase son autocontenidos para Colab.
- Los datos pequeños se generan o distribuyen con `scikit-learn`; los datos
  grandes se descargan con una celda verificable, caché y licencia documentada.
- No se guardan datasets, entornos virtuales ni pesos grandes en Git.
- TensorFlow y PyTorch viven en notebooks distintos cuando se enseña la API; la
  comparación usa la misma arquitectura, inicialización y partición.
- La búsqueda de hiperparámetros se realiza sólo sobre desarrollo. El conjunto
  de prueba se consulta una única vez al final.

## 5. Criterios de terminado por notebook

Un notebook no se considera listo hasta que:

- se ejecute de principio a fin en un runtime limpio de Colab;
- no contenga salidas de ejecución ni rutas locales;
- fije semillas cuando la operación lo permita;
- termine en menos de 10 minutos con CPU, salvo que se marque como GPU;
- incluya tiempo estimado, métricas apropiadas y respuestas esperadas para el
  docente en un archivo separado;
- identifique fuente, licencia y versión de cada dataset externo;
- pase validación de sintaxis y ejecución automática.

## 6. Fases de migración

1. **Completado — Base:** estructura, entorno, protocolo común y KNN/SVM.
2. **Completado — ML clásico:** ensambles, desbalance, explicabilidad y clustering.
3. **Completado — Núcleo matemático:** activaciones, gradiente, pérdidas y MLP desde cero.
4. **Completado — Frameworks:** TensorFlow, PyTorch, CNN y secuencias.
5. **Completado — Cierre:** generativos, RL, transformers/LLM y plantilla de proyecto.
6. **Pendiente de decisión — Saneamiento histórico:** inventariar, migrar o archivar material antiguo;
   retirar `envUN` sólo después de resolver los cambios locales existentes.

## 7. Material adicional útil

Para ajustar profundidad, ejemplos y notación serán especialmente útiles:

- las presentaciones ya utilizadas en clase;
- las rúbricas de los dos parciales y del proyecto;
- la edición o capítulos concretos de los libros guía disponibles legalmente;
- información sobre conocimientos previos reales del grupo y número de
  estudiantes.

No son un bloqueo para continuar con los notebooks base.
