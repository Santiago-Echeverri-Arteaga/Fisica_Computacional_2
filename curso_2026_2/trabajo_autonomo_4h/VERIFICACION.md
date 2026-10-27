# Revisión del material

Revisión del 26 de octubre de 2026. Destinatarios: estudiantes que ya recibieron
la teoría matemática, pero no han programado estos modelos.

## Dificultad y dedicación

La ruta de notebooks proporciona todo el código; primero se observa una entrada,
una operación y una salida. Una introducción explica celdas, asignaciones, funciones,
dimensiones y errores de orden de ejecución. Los retos esenciales cambian un valor
de filtros o un indicador de conexión residual; no exigen reconstruir la arquitectura.
Las celdas imprimen las métricas que deben registrarse y la selección precede al test.

El taller sin computador tiene ejemplos resueltos de geometría, filtros, parámetros,
compatibilidad de ramas, transferencia y métricas. La dificultad adicional está en
reparar diseños y justificar decisiones. El recálculo de una arquitectura más ancha,
el cálculo de memoria y una auditoría más extensa son ampliaciones opcionales.
Se retiraron las rúbricas y los tiempos asignados por sección.

**Juicio de revisión:** la ruta esencial es razonable para trabajo autónomo con la
teoría matemática ya estudiada y los apoyos incluidos. Las cuatro horas por ruta son
una estimación, no el resultado de un piloto con estudiantes. No incluyen realizar
las dos rutas juntas, todas las ampliaciones ni repetir con ambos datasets.
El taller desarrolla razonamiento comparable; no certifica destreza de programación.

## Cobertura

| Tema solicitado | Evidencia en el material |
|---|---|
| Pillow con imágenes reales | Leer, recortar, redimensionar, rotar, cambiar brillo, guardar |
| OpenCV | BGR/RGB, grises, HSV, suavizado, Canny, máscara y morfología |
| CNN y retos | CNN pequeña entrenada, formas, filtros, pérdida, errores y cambio de capacidad |
| Bases reales | Flower Photos en la ruta principal; CIFAR-10 como alternativa ejecutable |
| Transfer learning | InceptionV3 y ResNet152 con pesos ImageNet, preprocesamiento específico y cabezas nuevas |
| Redes diseñadas a mano | Ramas multiescala, proyección 1×1, concatenación, suma residual y ablación |
| Recursos gratuitos | CPU reducido automático, lotes pequeños, imágenes 96×96 y bases congeladas secuenciales |
| Aprendizaje autónomo | Instrucciones de ejecución, ejemplos, pistas, retos y solucionario del taller |

## Alcance técnico de las comprobaciones

**Resultado: las tres ejecuciones terminaron sin errores.**

| Prueba local en CPU, modo reducido | Resultado | Tiempo observado |
|---|---|---:|
| Notebook 01, flores, incluyendo entrenamiento del reto con 8 filtros | Correcto | 48 s |
| Notebook 02, flores, ambas bases ImageNet y red propia | Correcto | 126 s |
| Notebook 02, CIFAR-10, ambas bases y variantes de la red propia | Correcto | 139 s |

La primera prueba de flores sin caché había tardado 89 s; la repetición de 48 s
aprovechó datos descargados. La prueba de 126 s descargó los pesos; la de CIFAR-10
reutilizó esos pesos y descargó su archivo Parquet. Son tiempos del equipo local,
con distintos estados de caché, no una comparación de rendimiento entre datasets.
La estructura y sintaxis de los 29 notebooks y las tres pruebas del repositorio
también pasaron. Se comprobaron los enlaces internos, las cuentas del solucionario
y la ausencia de rúbrica y tiempos por sección en el taller.

La ejecución se verifica localmente en CPU con Python 3.13.5 y TensorFlow 2.21.0,
usando descargas reales y pesos ImageNet. El programa `tools/check_autonomous_vision.py`
ejecuta todas las celdas de código en orden, valida el formato del notebook y guarda
registros en `curso_2026_2/.artifacts/` (no publicados). No simula datos ni sustituye
las bases grandes por otras más pequeñas.

Esta comprobación no es una ejecución dentro del servicio Colab. Los notebooks usan
las bibliotecas y API disponibles en un entorno Python estándar, sin servicios de pago,
sin montar Drive y sin requerir GPU. Los límites y la disponibilidad del servicio
son variables: [FAQ oficial de Colab](https://research.google.com/colaboratory/faq.html).
Los tiempos locales no predicen los de Colab ni el tiempo de aprendizaje.

Para reproducir en un entorno compatible, desde la raíz del repositorio:

```bash
python -m pip install -r requirements-tensorflow-2026.txt
python tools/check_autonomous_vision.py 01 --check-challenges
python tools/check_autonomous_vision.py 02
python tools/check_autonomous_vision.py 02 --dataset cifar10 --check-challenges
python tools/validate_notebooks.py
python -m pytest -q
```

Las variantes de la red personalizada se comprueban con una pasada de predicción
y un paso real de actualización de pesos; esto verifica compatibilidad de formas
y gradientes, no la convergencia de cada variante. El reto de filtros de la primera
CNN sí repite el entrenamiento corto completo.
