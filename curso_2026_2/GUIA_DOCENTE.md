# Guía docente de uso

## Estructura sugerida de una sesión de 120 minutos

| Minutos | Actividad |
|---:|---|
| 0–10 | pregunta física, predicción individual y recuperación de la clase anterior |
| 10–35 | fundamento matemático en tablero; formas y supuestos |
| 35–60 | implementación mínima o cálculo manual |
| 60–70 | pausa |
| 70–95 | biblioteca, pipeline y experimento controlado |
| 95–110 | lectura de métricas, fallos y límites físicos |
| 110–120 | pregunta de salida y asignación del ejercicio |

Los notebooks marcados con 4 h están diseñados para dos sesiones: la primera
termina normalmente antes de `GridSearchCV` o del entrenamiento con framework;
la segunda comienza recuperando la predicción matemática de lo que debería pasar.

## Evaluación formativa

- **Predicción previa:** el estudiante anticipa el efecto de un hiperparámetro.
- **Comprobación mínima:** explica una ecuación o forma de tensor antes de ejecutar.
- **Cambio controlado:** modifica una variable y mantiene las demás constantes.
- **Salida:** distingue resultado estadístico, supuesto matemático y significado físico.

No se califica por obtener la mayor métrica. Se valora justificar el protocolo,
detectar fuga, reconocer incertidumbre y explicar un fallo.

## Uso de TensorFlow y PyTorch

Los notebooks 34 y 35 resuelven problemas equivalentes para separar concepto de
API. Después de esa comparación no conviene duplicar cada práctica en ambos
frameworks: visión/generativos/secuencias alternan frameworks. Cada grupo escoge
uno para el proyecto y debe poder explicar el ciclo de entrenamiento del otro.

## Sesiones grabadas de recuperación

1. Notebook 31 completo: activaciones, derivadas y ablación con semillas.
2. Notebooks 50–51 en versión guiada: representación latente y juego adversarial.

Incluya pausas explícitas antes de cada bloque de ejercicios y una breve solución
en video después de la fecha de entrega, no dentro del notebook del estudiante.

## Proyecto final

La plantilla 80 operacionaliza el acuerdo: grupos de tres, modelo neural no visto
directamente en clase, fundamento matemático e implementación. Debe existir una
línea base, selección usando desarrollo, test final, ablaciones, incertidumbre,
análisis de errores, licencia de datos y ejecución desde un runtime limpio.

## Adaptación al ritmo real

Si el grupo requiere más tiempo, preserve en este orden:

1. evaluación sin fuga y líneas base;
2. activaciones, gradiente, backpropagation, pérdidas y regularización;
3. una CNN y una arquitectura secuencial completas;
4. transferencia y proyecto;
5. panorama generativo, RL y LLM.

Las listas históricas de arquitecturas deben usarse para comparar ideas, no para
memorizar cronologías ni entrenar todos los modelos.
