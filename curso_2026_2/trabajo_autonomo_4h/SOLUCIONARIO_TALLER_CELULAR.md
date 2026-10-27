# Solucionario razonado: taller con celular y papel

Consulta después de intentar el [taller](TALLER_CELULAR_SIN_PROGRAMAR.md).
Las respuestas abiertas admiten varias soluciones. Los números de ejercicios y
tablas son ejemplos didácticos, no resultados de entrenamientos reales.

## A. Imágenes

**A1:** una respuesta válida identifica rasgos observables en las fotos elegidas,
por ejemplo contorno de pétalos, distribución de colores y textura. El fondo verde
puede correlacionarse con una clase sin ser un rasgo de la flor. Pasar a gris pierde
distinciones cromáticas; recortar puede eliminar el objeto. No se exige diagnosticar
la especie a partir de la imagen ni medir píxeles desde la pantalla.

**A2:** factores 128/1200 ≈ 0.1067 y 128/800 = 0.16: no son iguales y deforman.
Caja centrada (200,0,1000,800). Se elimina 320 000 de 960 000 píxeles: un tercio.
Al ajustar proporcionalmente, queda un contenido ideal de 128×85.33 y relleno
arriba y abajo de unos 21.33 píxeles por lado, antes del redondeo. Rellenar conserva
los extremos, aunque reduce el tamaño útil del objeto e introduce franjas; recortar
con `fit` los perdería. Se acepta otra elección si explica su costo.

**A3:** BGR = (50,100,200); mostrarlo como RGB intercambia rojo y azul. Solo P pasa
los tres intervalos: Q falla por saturación, R por tono y S por brillo. El fondo
puede compartir color y las sombras pueden sacar pétalos del rango. Canny responde
a cambios locales de intensidad; esos cambios no contienen por sí solos una etiqueta
de especie. No debe confundirse una máscara de color con segmentación semántica.

## B. CNN

**B1:** antes de ReLU:

```text
0   2  0
0   0  0
0  -2  0
```

La salida superior central es 1×1 + 0×(−1) + 1×1 + 0×(−1) = 2.
La inferior central es 0×1 + 1×(−1) + 0×1 + 1×(−1) = −2.
ReLU convierte −2 en cero. Pooling produce:

```text
2  2
0  0
```

El filtro responde positivamente a intensidad mayor a la izquierda que a la derecha,
en una frontera vertical. ReLU pierde la respuesta negativa a la transición opuesta.
El suavizado usa un filtro prefijado para atenuar fluctuaciones; una CNN ajusta sus
filtros mediante datos y gradientes de una pérdida.

**B2:**

| Etapa | Salida | Parámetros |
|---|---|---:|
| Conv | 6×6×2 | (3×3×1+1)×2 = 20 |
| ReLU | 6×6×2 | 0 |
| Pool | 3×3×2 | 0 |
| GAP | 2 | 0 |
| Dense | 3 | (2+1)×3 = 9 |

Total: **29**. Flatten produce 18 entradas; Dense tiene 57 parámetros, total **77**.
Más parámetros pueden aumentar el riesgo de sobreajuste, pero no lo prueban por sí
solos. Lote: entrada (4,8,8,1), salida (4,3). La dimensión de lote no multiplica
los parámetros del modelo, que se comparten entre ejemplos.

## C. Ramas y residuales

**C1:** cada rama termina en 24×24×8. La concatenación produce 24×24×32;
la proyección produce 24×24×16, compatible con el atajo original.

| Parte | Parámetros con f=8 |
|---|---:|
| A | (16+1)×8 = 136 |
| B | 136 + (9×8+1)×8 = 720 |
| C | 136 + 584 + 584 = 1304 |
| D | 136 |
| Proyección | (32+1)×16 = 528 |
| Total | **2824** |

Cumple el límite de 3000. No hace falta proyectar el atajo original.

**C2:** 32 canales y 16 canales no admiten esa suma elemento a elemento. Se puede
proyectar la concatenación a 16, como en C1, o proyectar el atajo de 16 a 32 y sumar
con la concatenación; en el segundo caso la salida tendría 32 canales. La segunda
alternativa tiene otro presupuesto y no se da por aprobada automáticamente.

**C3:** f=12 da ramas con 12 canales y concatenación con 48. Conteos: A=204,
B=1512, C=2820, D=204, proyección=784. Total **5524**: no cumple.
Una ablación válida elimina solo la suma residual, conserva las demás capas y
compara macro-F1 de validación con la misma semilla, datos y épocas. La hipótesis
“el atajo mejora en este presupuesto” no queda apoyada si empata o empeora; una
sola corrida tampoco resuelve la variabilidad. No se requiere predecir un ganador.

## D. Transferencia

**D1:** InceptionV3 ≈ (0.569, −0.216, −0.608). ResNet152 =
(−53.939, −16.779, 76.320), en orden BGR tras restar medias.
Dividir primero por 255 cambia la escala esperada. Inception recibiría números
transformados casi todos cerca de −1; ResNet restaría medias grandes a valores
muy pequeños. Las formas siguen siendo válidas: el error es de significado y distribución.

**D2:** cabeza: (2048+1)×5 = **10 245** parámetros entrenables en ambos casos.
Los pesos congelados siguen existiendo, pero no se actualizan. Características:
500×2048×4 = **4 096 000 bytes = 3.90625 MiB**. Esto no es la RAM total de entrenamiento.
Si la base cambia, cambian sus características. Una cabeza entrenada sobre vectores
fijos no transmite gradientes a la base que los generó: para afinarla se necesitan
las imágenes y el paso por la base actual dentro del entrenamiento.

**D3:** desde imágenes, Inception cuesta 7 minutos, ResNet152 16 y la red propia 12:
solo Inception satisface el límite. Con características de ambas bases ya guardadas,
las dos cabezas cuestan 1 minuto y se elige ResNet152 por la métrica estipulada.
La diferencia 0.02 no prueba superioridad general; hacen falta incertidumbre,
repeticiones y considerar datos/resolución/preentrenamiento. Para fine-tuning:
learning rate pequeño, recompilar después de cambiar qué capas son entrenables y
controlar BatchNorm, manteniéndola en inferencia en esta estrategia.

## E. Auditoría

**E1:** las fotos de una planta no son ejemplos independientes para el objetivo
de plantas nuevas. Repartir por archivo puede enseñar la misma planta al modelo
y luego evaluarla como si fuera nueva.

| Partición | Plantas por clase | Fotos por clase | Plantas totales | Fotos totales |
|---|---:|---:|---:|---:|
| Train | 12 | 60 | 60 | 300 |
| Validación | 4 | 20 | 20 | 100 |
| Test | 4 | 20 | 20 | 100 |

Crear aumentos solo para train, después de separar grupos; sus derivados permanecen
en train. Para jardines nuevos, agrupar por jardín: separar plantas del mismo jardín
no bastaría. La proporción exacta podría no ser alcanzable con pocos jardines y
habría que documentar esa limitación sin dividirlos para forzar porcentajes.

**E2:** conservar época 3. Las épocas 4 y 5 no mejoran; paciencia 2 detiene al
terminar la 5, restaurando los pesos de la 3 si esa opción está activa. Train sigue
bajando mientras validación sube: patrón compatible con sobreajuste. Son válidas
propuestas como reducir capacidad o añadir aumento, cambiando un factor a la vez.

**E3:** accuracy = 22/30 ≈ **0.7333**. Recall B = 6/10 = **0.60**.

| Clase | TP | FP (columna fuera de diagonal) | FN (fila fuera de diagonal) | F1 |
|---|---:|---:|---:|---:|
| A | 8 | 1 | 2 | 16/19 ≈ 0.8421 |
| B | 6 | 4 | 4 | 12/20 = 0.6000 |
| C | 8 | 3 | 2 | 16/21 ≈ 0.7619 |

Macro-F1 ≈ **0.7347**. Una corrección aumenta accuracy en 1/30, es decir,
**3.33 puntos porcentuales**. La confusión B→C es la más numerosa (3); se pueden
inspeccionar iluminación, fondo, encuadre y etiquetas. Se aceptan otras prioridades
justificadas por el costo del error. Inspeccionar test al final sirve para describir
limitaciones; si se rediseña usando esos hallazgos, ese test ya no es independiente
para evaluar la nueva versión: se necesita otra evaluación reservada.

**E4:** fallos y reparaciones:

1. Derivados de una foto en varias particiones: separar primero por fuente/grupo,
   aumentar solo train para evitar fuga.
2. BGR/255 para ResNet152: conservar RGB 0–255 y aplicar su preprocesamiento específico
   una vez; lo contrario desplaza canales y distribución.
3. Selección repetida con test: usar validación y reservar test hasta el cierre.
4. Descongelar después de compilar sin recompilar: recompilar para aplicar el cambio
   de entrenabilidad; revisar tasa de aprendizaje y BatchNorm.
5. Afinar la base con vectores antiguos: volver a introducir imágenes y la base actual
   en el grafo de entrenamiento; los vectores fijos no permiten actualizarla.
6. Elegir menor pérdida de train: elegir con la métrica de validación acordada;
   una pérdida de train menor puede coincidir con peor generalización.
7. Atribuir toda diferencia a la arquitectura: reconocer el distinto preentrenamiento,
   presupuesto y datos. La comparación de sistemas no aísla un efecto causal del diseño.

## F. Criterios de una defensa sólida

El protocolo debe ser internamente coherente: jardines separados si se exige jardín
nuevo; transformación motivada por encuadre; preprocesamiento que corresponda a la
base elegida; presupuesto que incluya las etapas necesarias; un factor por comparación;
selección por validación y test al final. Una elección diferente de Inception es válida
si el protocolo declara otras restricciones y las justifica; no es válida si ignora
el presupuesto que dice cumplir.

La conclusión debe reconocer que saber calcular formas y detectar fuga prepara
para implementar, pero no demuestra que el programa ejecute ni que el tiempo o la
accuracy real coincidan con tablas inventadas. Se deben verificar instalación,
descarga, memoria, ejecución y métricas cuando se disponga de un entorno de cómputo.
