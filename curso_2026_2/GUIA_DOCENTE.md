# Guía de sesiones y prácticas

## Martes: un notebook, 120 minutos

Abrir [30 — Feedforward en la práctica](03_redes_fundamentos/30_neurona_y_feedforward_desde_cero.ipynb).
La teoría de redes feedforward ya es prerrequisito. La sesión se concentra en
traducirla a código y entender qué hace cada API. No es necesario abrir otros notebooks.

| Minutos | Actividad | Comprobación |
|---|---|---|
| 0–12 | runtime, Fashion-MNIST, particiones y EDA | forma, tipo y rango de entradas |
| 12–25 | tensores y gradientes en TF/PyTorch | derivada escalar igual a −15.2 |
| 25–45 | TF: variables, forward, pérdida y actualización | formas y pérdidas por época |
| 45–60 | Keras: Input, Dense, compile y fit | 50 890 parámetros |
| 60–80 | PyTorch: Module, DataLoader y ciclo | zero_grad, backward, step, eval |
| 80–105 | activaciones propias y ablación | derivadas, gradientes y validación |
| 105–115 | test y errores frente a logística | tabla, curvas y confusiones |
| 115–120 | conexión con medición y regresión física | salida, pérdida y unidades |

El modo `clase` usa 12 000 imágenes de entrenamiento, 3 000 de validación y
cuatro épocas por red. El modo `completo` usa 48 000/12 000 y diez épocas;
se reserva para exploración posterior. Los tiempos de entrenamiento dependen
del runtime y no son la duración pedagógica de la sesión. Ejecutar previamente
desde un runtime limpio permite comprobar la descarga y tener las gráficas listas.

Si hay demora, reducir `EPOCAS` a 2 antes de ejecutar los entrenamientos y mantener
ese presupuesto en todas las comparaciones. Conservar el ciclo explícito,
la activación propia y la evaluación. Usar la espera para predecir formas y
discutir errores, no para añadir arquitecturas nuevas.

## Miércoles: elegir una alternativa

La [guía del taller](03_redes_fundamentos/TALLER_FEEDFORWARD.md) contiene instrucciones,
entregables y criterios. Las dos alternativas se excluyen entre sí.

- **A, programación guiada de 120 minutos:** datos de superconductividad ya
  preparados en el notebook 36; dos redes densas con igual arquitectura e
  inicialización, 15 épocas y dos activaciones. Se retiran el análisis exploratorio,
  los modelos clásicos, las múltiples semillas y el informe adicional.
- **B, celular y papel durante 120 minutos:** [guía sin computador](03_redes_fundamentos/TALLER_SIN_COMPUTADOR.md),
  con cálculo de una predicción y una actualización, análisis de activaciones,
  lectura de fragmentos de Keras/TensorFlow/PyTorch, consulta dirigida de
  documentación oficial y evaluación de resultados hipotéticos suministrados.
  No requiere escribir ni ejecutar código. Se aceptan fotos de hojas numeradas.

Las dos alternativas trabajan los mismos conceptos; B exige demostrar los
cálculos y justificar la interpretación de las herramientas. En A se elige un
único framework y se comparan las redes por error absoluto medio de validación
en kelvin. La evaluación de ambas se reparte en cuatro criterios de igual peso:
red, activación, entrenamiento/herramientas y evaluación física. No se requieren
arquitecturas posteriores ni superar una puntuación prefijada.

## Material de apoyo y continuidad

| Notebooks | Uso |
|---|---|
| 30 | sesión integrada: frameworks, activaciones y clasificación |
| 36 | preparación del taller de regresión física con datos UCI |
| 32 | comprobación de backpropagation con un problema controlado |
| 33 | profundización posterior en optimización y regularización |
| 40–43 | localidad, CNN y transferencia; continuar con benchmarks de imágenes |
| 50–51 | reconstrucción y generación; distinguir error de reconstrucción y fidelidad |
| 60–62 | secuencias y dinámica física; comparación con persistencia y MLP |
| 70–71 | panorama de decisiones y atención, con alcance introductorio |
| 80 | proyecto con evidencia reproducible y comparación con modelos simples |

En las unidades posteriores, las simulaciones siguen sirviendo para comprobar
un mecanismo o una ley conocida. Los benchmarks permiten medir rendimiento en
observaciones reales. Ninguna de las dos fuentes sustituye a la otra. El proyecto
debe identificar expresamente si los datos son observados, simulados o derivados.

## Entorno y mantenimiento

Colab debe disponer de TensorFlow, PyTorch y las bibliotecas científicas usuales.
El notebook 30 imprime versiones. En local, usar Python 3.11–3.13 e instalar
`requirements-feedforward-2026.txt`. No instalar el `requirements.txt` histórico.
Los notebooks de estudiantes no dependen de módulos privados del repositorio.

Los generadores en `tools/` son la fuente editable del contenido. Para regenerar
y validar desde la raíz:

```bash
python tools/build_all_notebooks.py
python -m pytest
python tools/validate_notebooks.py
python tools/validate_notebooks.py --execute --tiers tensorflow-pytorch-network --timeout 1200
```

La última orden ejecuta el 30 y necesita ambos frameworks e Internet. La validación
estática comprueba estructura y sintaxis, pero no sustituye la ejecución. Los
notebooks del repositorio se guardan sin salidas; los entregables del taller sí
deben conservar sus resultados. El notebook 36 contiene espacios para completar;
ejecutar su carga no equivale a resolver el taller.
