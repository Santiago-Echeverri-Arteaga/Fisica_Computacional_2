# Física Computacional 2 — edición 2026-2

Esta carpeta contiene la actualización progresiva del curso. Los notebooks son
autocontenidos, legibles de arriba hacia abajo y diseñados para ejecutarse en
Google Colab sin descargas manuales ni configuración de rutas locales.

## Martes y miércoles: redes feedforward

- **Martes, 120 minutos:** abrir únicamente [30 — TensorFlow, Keras y PyTorch desde la base](03_redes_fundamentos/30_neurona_y_feedforward_desde_cero.ipynb).
  La teoría feedforward es prerrequisito; el énfasis está en programación, Fashion-MNIST,
  activaciones propias y comparación con una línea base de ML.
- **Miércoles:** elegir una de las [dos alternativas de taller](03_redes_fundamentos/TALLER_FEEDFORWARD.md).
  Ambas duran 120 minutos. La A entrena dos redes con datos de superconductividad
  desde el [notebook 36](03_redes_fundamentos/36_taller_120_minutos.ipynb), con datos ya preparados.
  La B es [sin programación, con celular y papel](03_redes_fundamentos/TALLER_SIN_COMPUTADOR.md).
- **Consulta posterior:** 32–33 profundizan en backpropagation, optimización y
  regularización. No es necesario recorrerlos antes del taller.
- **Desde 04:** visión, generativos y secuencias son unidades posteriores.
  Sus guías distinguen ejemplos de mecanismo, benchmarks y aplicaciones físicas.

## Principios de diseño

- **Matemática antes de la API:** la teoría ya estudiada se conecta con formas de tensores,
  pérdidas y decisiones de programación.
- **Una sola idea nueva a la vez:** primero se implementa o visualiza la idea;
  después se usa la implementación industrial de la biblioteca.
- **Evaluación honesta:** separación de prueba antes de explorar, `Pipeline`
  para prevenir fuga de información y validación cruzada para seleccionar
  hiperparámetros.
- **Conexión con física:** se combinan benchmarks reconocidos con datos físicos abiertos.
  Las simulaciones pequeñas se reservan para aislar mecanismos y verificar resultados.
- **Dos frameworks con propósito:** TensorFlow y PyTorch se comparan sobre
  el mismo problema; los proyectos extensos escogen uno, no ambos.

## Contenido completo

La edición contiene 27 notebooks:

| Módulo | Notebooks | Enfoque |
|---|---:|---|
| 00 Fundamentos | 2 | evaluación, pipelines y reproducibilidad |
| 01 ML supervisado | 6 | regresión, KNN/SVM, ensambles, desbalance y explicación |
| 02 No supervisado | 3 | densidad, centroides, mezclas y jerarquías |
| 03 Redes: fundamentos | 4 | sesión integrada, activaciones, apoyo y taller |
| 04 Visión | 4 | convolución, LeNet, AlexNet, VGG, ResNet, Inception, FractalNet y transferencia |
| 05 Generativos | 2 | autoencoders y GAN |
| 06 Secuencias | 3 | RNN/LSTM/GRU, Seq2Seq-atención y Lorenz |
| 07 Temas actuales | 2 | Q-learning y transformer/GPT diminuto |
| 08 Proyecto | 1 | plantilla reproducible y lista de control |

El [índice de notebooks](INDICE_NOTEBOOKS.md) incluye enlaces directos, duración,
framework y producto esperado.

El plan completo de migración, el cronograma y los criterios de terminado están
en [`PLAN_ACTUALIZACION.md`](PLAN_ACTUALIZACION.md).
La [guía de sesiones](GUIA_DOCENTE.md) propone el uso de las dos horas de cada
sesión. La selección razonada de textos guía está en
[`BIBLIOGRAFIA_COMENTADA.md`](BIBLIOGRAFIA_COMENTADA.md).
Las fuentes, tamaños y condiciones de descarga están en
[`DATOS_ABIERTOS.md`](DATOS_ABIERTOS.md).

## Ejecución

### Google Colab

Abra el notebook desde GitHub y pulse **Abrir en Colab**. Cada notebook declara
si requiere TensorFlow, PyTorch, GPU o Internet. Los notebooks 30, 36, 41,
43 y 50 descargan datos. La primera ejecución requiere conexión; las siguientes
pueden usar caché. No hace falta clonar el repositorio para ejecutar el 30.
En Colab use un runtime Python estándar; GPU es opcional para el 30. Los enlaces
de Colab apuntan a la rama `master`: los cambios locales deben estar publicados
allí para abrir esa versión mediante el distintivo.

### Entorno local

Se recomienda Python 3.11–3.13 para mantener compatibilidad con TensorFlow y
PyTorch durante el resto del curso.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-2026.txt
jupyter lab
```

Para la sesión integrada 30 instale ambos frameworks:

```bash
python -m pip install -r requirements-feedforward-2026.txt
```

Para otros bloques neuronales puede instalar sólo el entorno necesario:

```bash
python -m pip install -r requirements-tensorflow-2026.txt
# o
python -m pip install -r requirements-pytorch-2026.txt
```

El archivo histórico `requirements.txt` se conserva para no alterar de forma
silenciosa los notebooks antiguos. No debe usarse como entorno nuevo del curso.

## Convenciones para cada notebook

Cada unidad nueva debe contener:

1. objetivos observables y conocimientos previos;
2. intuición física y formulación matemática;
3. implementación mínima visible;
4. implementación con la biblioteca apropiada;
5. partición de datos, `Pipeline`, métricas y validación;
6. errores frecuentes y preguntas de comprobación;
7. ejercicios con niveles básico, intermedio y reto;
8. referencias y procedencia de los datos.
