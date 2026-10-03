# Índice de notebooks 2026-2

Todos son autocontenidos y tienen distintivo de Colab. La ruta del martes es sólo el **30**; el miércoles se elige una
[alternativa de taller](03_redes_fundamentos/TALLER_FEEDFORWARD.md). Los módulos 04–08 son posteriores.
La [alternativa B sin programación](03_redes_fundamentos/TALLER_SIN_COMPUTADOR.md)
se lee desde el celular y se resuelve en papel; no requiere notebook.
CPU indica que se pueden
ejecutar sin acelerador; GPU opcional reduce espera, pero no cambia los objetivos.

| Código | Notebook | Tiempo | Entorno | Producto principal |
|---|---|---:|---|---|
| 00 | [Protocolo de evaluación](00_fundamentos/00_protocolo_evaluacion.ipynb) | 2 h | CPU | búsqueda sin fuga y test final |
| 01 | [Pipelines y reproducibilidad](00_fundamentos/01_pipelines_reproducibilidad.ipynb) | 2 h | CPU | pipeline recuperable y registro JSON |
| 10 | [Regresión y regularización](01_nivelacion_ml/10_regresion_regularizacion.ipynb) | 4 h | CPU | Linear/Ridge/Lasso en proyectil con arrastre |
| 11 | [KNN y SVM](01_nivelacion_ml/11_knn_svm.ipynb) | 4 h | CPU | regímenes de oscilador amortiguado |
| 12 | [Árboles, bosques y bagging](01_nivelacion_ml/12_arboles_bosques_bagging.ipynb) | 4 h | CPU | comparación y varianza de ensambles |
| 13 | [Boosting y stacking](01_nivelacion_ml/13_boosting_stacking.ipynb) | 4 h | CPU | modelos secuenciales y metamodelo |
| 14 | [Clases desbalanceadas](01_nivelacion_ml/14_clases_desbalanceadas.ipynb) | 4 h | CPU | pesos, oversampling, SMOTE y PR |
| 15 | [Explicabilidad](01_nivelacion_ml/15_explicabilidad_local_global.ipynb) | 4 h | CPU | permutación y sustitutos local/global |
| 20 | [DBSCAN y Mean-Shift](02_no_supervisado/20_dbscan_mean_shift.ipynb) | 4 h | CPU | agrupamiento por densidad y modos |
| 21 | [K-means y GMM](02_no_supervisado/21_kmeans_gmm.ipynb) | 4 h | CPU | silhouette, BIC y pertenencia probabilística |
| 22 | [Jerárquico](02_no_supervisado/22_clustering_jerarquico.ipynb) | 4 h | CPU | dendrograma y comparación de linkages |
| 30 | [Sesión integrada: TensorFlow, Keras y PyTorch](03_redes_fundamentos/30_neurona_y_feedforward_desde_cero.ipynb) | 2 h | TF + PyTorch/CPU/Internet | Fashion-MNIST, ciclos, activaciones y test |
| 32 | [Gradiente y backpropagation](03_redes_fundamentos/32_gradiente_y_backpropagation.ipynb) | 4 h | NumPy/CPU | derivación y red entrenada desde cero |
| 33 | [Pérdidas, optimizadores y regularización](03_redes_fundamentos/33_perdidas_optimizadores_regularizacion.ipynb) | 4 h | NumPy/CPU | SGD, momentum, RMSProp y Adam |
| 36 | [Alternativa A: dos redes y una temperatura](03_redes_fundamentos/36_taller_120_minutos.ipynb) | 2 h | CPU/Internet | datos UCI preparados, activación propia y error en kelvin |
| 40 | [Convolución desde cero](04_vision/40_convolucion_desde_cero.ipynb) | 4 h | NumPy/CPU | filtros, tamaños y pooling |
| 41 | [LeNet-5 y AlexNet](04_vision/41_lenet_alexnet.ipynb) | 4 h | PyTorch/CPU/Internet | LeNet en Fashion-MNIST y AlexNet inspeccionada |
| 42 | [VGG, ResNet, Inception y FractalNet](04_vision/42_vgg_resnet_inception_fractalnet.ipynb) | 4 h | TensorFlow/CPU | bloques y conteos de parámetros |
| 43 | [Transfer learning](04_vision/43_transfer_learning_fisica.ipynb) | 4 h | TF/GPU/Internet | base ImageNet congelada y fine-tuning |
| 50 | [Autoencoders](05_generativos/50_autoencoders.ipynb) | 4 h | TF/CPU/Internet | reconstrucción de Fashion-MNIST y anomalías |
| 51 | [GAN](05_generativos/51_gan.ipynb) | 4 h | PyTorch/GPU opcional | generador de dígitos y diagnóstico |
| 60 | [RNN, LSTM y GRU](06_secuencias/60_rnn_lstm_gru.ipynb) | 4 h | TensorFlow/CPU | comparación en señales amortiguadas |
| 61 | [Seq2Seq con atención](06_secuencias/61_seq2seq_atencion.ipynb) | 4 h | PyTorch/CPU | inversión de secuencias y mapa de atención |
| 62 | [Serie de Lorenz](06_secuencias/62_series_temporales_fisicas.ipynb) | 4 h | TensorFlow/CPU | split temporal, Ridge y LSTM |
| 70 | [Q-learning](07_temas_actuales/70_aprendizaje_por_refuerzo.ipynb) | 4 h | NumPy/CPU | política y función de valor |
| 71 | [Transformer y mini-GPT](07_temas_actuales/71_transformers_y_llm.ipynb) | 6 h | PyTorch/GPU opcional | atención causal y generación autoregresiva |
| 80 | [Proyecto final](08_proyecto/80_plantilla_proyecto_final.ipynb) | 2 h | según proyecto | diseño, evidencia y lista de control |
