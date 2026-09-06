# Física Computacional 2 — edición 2026-2

Esta carpeta contiene la actualización progresiva del curso. Los notebooks son
autocontenidos, legibles de arriba hacia abajo y diseñados para ejecutarse en
Google Colab sin descargar datos ni configurar rutas locales.

## Principios de diseño

- **Matemática antes de la API:** cada método comienza con su objetivo,
  hipótesis y ecuaciones esenciales.
- **Una sola idea nueva a la vez:** primero se implementa o visualiza la idea;
  después se usa la implementación industrial de la biblioteca.
- **Evaluación honesta:** separación de prueba antes de explorar, `Pipeline`
  para prevenir fuga de información y validación cruzada para seleccionar
  hiperparámetros.
- **Conexión con física:** se priorizan datos físicos abiertos o simulaciones
  pequeñas cuyo mecanismo generador pueda entenderse.
- **Dos frameworks con propósito:** TensorFlow y PyTorch se compararán en
  ejemplos pequeños; los proyectos extensos escogerán uno, no ambos.

## Contenido completo

La edición contiene 29 notebooks:

| Módulo | Notebooks | Enfoque |
|---|---:|---|
| 00 Fundamentos | 2 | evaluación, pipelines y reproducibilidad |
| 01 ML supervisado | 6 | regresión, KNN/SVM, ensambles, desbalance y explicación |
| 02 No supervisado | 3 | densidad, centroides, mezclas y jerarquías |
| 03 Redes: fundamentos | 6 | feedforward, activaciones propias, backprop, TF y PyTorch |
| 04 Visión | 4 | convolución, LeNet, AlexNet, VGG, ResNet, Inception, FractalNet y transferencia |
| 05 Generativos | 2 | autoencoders y GAN |
| 06 Secuencias | 3 | RNN/LSTM/GRU, Seq2Seq-atención y Lorenz |
| 07 Temas actuales | 2 | Q-learning y transformer/GPT diminuto |
| 08 Proyecto | 1 | plantilla reproducible y lista de control |

El [índice de notebooks](INDICE_NOTEBOOKS.md) incluye enlaces directos, duración,
framework y producto esperado.

El plan completo de migración, el cronograma y los criterios de terminado están
en [`PLAN_ACTUALIZACION.md`](PLAN_ACTUALIZACION.md).
La [guía docente](GUIA_DOCENTE.md) propone el uso de las dos horas de cada
sesión. La selección razonada de textos guía está en
[`BIBLIOGRAFIA_COMENTADA.md`](BIBLIOGRAFIA_COMENTADA.md).
Las extensiones con datos científicos reales están catalogadas en
[`DATOS_ABIERTOS.md`](DATOS_ABIERTOS.md).
La selección razonada de textos para una futura edición está en
[`BIBLIOGRAFIA_COMENTADA.md`](BIBLIOGRAFIA_COMENTADA.md).

## Ejecución

### Google Colab

Abra el notebook desde GitHub y pulse **Abrir en Colab**. Cada notebook declara
si requiere TensorFlow, PyTorch, GPU o Internet. Salvo transferencia, los datos
son incluidos o generados en memoria.

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

Para los bloques neuronales instale **uno** de estos entornos:

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
