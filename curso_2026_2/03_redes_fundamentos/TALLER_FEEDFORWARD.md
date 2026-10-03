# Taller: comprender y comparar redes feedforward

**Miércoles. Elija una sola alternativa. Duración: 120 minutos.** Puede trabajar
individualmente o en pareja. Las dos alternativas estudian capas densas,
activaciones, entrenamiento y evaluación, con formas distintas de demostrar lo aprendido.

- **A — Con computador:** modificar y entrenar dos redes sobre datos de superconductividad.
  Use el [notebook 36](36_taller_120_minutos.ipynb).
- **B — Sin programar:** calcular una red a mano, interpretar fragmentos de código,
  consultar documentación desde el celular y analizar un experimento suministrado.
  Abra la [guía completa para celular y papel](TALLER_SIN_COMPUTADOR.md).
  No requiere computador, Colab, instalaciones ni descarga de bases de datos.

## Alternativa A — Dos redes para predecir una temperatura

**Tiempo total: dos horas.** Se entrega un único notebook. No se solicita análisis
exploratorio de datos ni comparación con modelos clásicos de aprendizaje de máquina.
La descarga, la separación de los datos y su cambio de escala ya están preparados.
El trabajo consiste en programar y comparar **dos entrenamientos**, con una sola semilla.

### Qué problema se va a resolver

Cada fila describe un material mediante 81 características numéricas. El objetivo
es predecir su temperatura crítica de superconductividad, llamada **Tc**, en
**kelvin (K)**. Se utilizan los 21 263 registros de
[Superconductivty Data, UCI 464](https://archive.ics.uci.edu/dataset/464/superconductivty+data).
Fuente: Hamidieh (2018), DOI 10.24432/C53P47, licencia CC BY 4.0.

Una **red feedforward** transmite información desde la entrada hacia la salida,
sin conexiones de retorno. Una **red densa** conecta cada neurona de una capa con
todas las de la siguiente. Este tipo de red también se denomina **perceptrón
multicapa**, o **MLP**, por su nombre en inglés *multilayer perceptron*.

Use Keras con TensorFlow **o** PyTorch; no ambos. Puede adaptar el código del
notebook 30; el punto de partida de la entrega es el 36. Este último explica
cómo cambiar la salida y la pérdida de clasificación a regresión.

### 1. Ejecutar la preparación e identificar las entradas — 0 a 15 minutos

Abra el notebook 36 en Colab y ejecute las celdas de preparación. Ya descargan los
datos, separan materiales y calculan las escalas usando sólo entrenamiento.
No cambie estas celdas ni construya nuevas particiones.

| Variable preparada | Significado | Uso |
|---|---|---|
| `xt`, `yt` | características y temperaturas de entrenamiento, estandarizadas | ajustar los pesos |
| `xv`, `yv` | características y temperaturas de validación, estandarizadas | comparar las dos redes |
| `xs` | características del conjunto de prueba, estandarizadas | evaluación final |
| `y[val]`, `y[test]` | temperaturas reales de validación y prueba, en K | calcular errores en unidades físicas |
| `kelvin(pred)` | función que devuelve predicciones a la escala original | convertir salidas de la red a K |

**Estandarizar** significa restar la media de entrenamiento y dividir por su
desviación estándar. La red aprende en esa escala; el resultado físico se presenta
en kelvin. **Validación** sirve para elegir entre configuraciones; **prueba**
(*test*) sirve para evaluar la elección una vez terminada.

**Deje escrito:** cuántas filas hay en cada conjunto, cuántas entradas recibe la
red y por qué un mismo material no debe aparecer a la vez en entrenamiento y
prueba. Basta con tres frases. No se requieren histogramas, correlaciones ni
un informe de limpieza de datos.

### 2. Construir y entrenar la primera red — 15 a 50 minutos

Complete una función llamada `crear_red(activacion)` que devuelva una red nueva.
Debe tener **81 entradas → 32 neuronas → 16 neuronas → 1 salida**. Las dos capas
ocultas usan la activación recibida; la salida es lineal, es decir, no aplica una
activación adicional. Incluya un sesgo en cada capa densa.

Para la primera red use **ReLU**, la unidad lineal rectificada: `max(0, z)`.
Configure estos valores, sin buscar otros:

- optimizador **Adam**, el método que actualiza los pesos, con tasa de aprendizaje `0.001`;
- **15 épocas**: cada época es un recorrido por todos los ejemplos de entrenamiento;
- **lotes de 256**: se calcula una actualización por cada grupo de hasta 256 ejemplos;
- pérdida **MSE**, *mean squared error* o error cuadrático medio: promedio de
  `(predicción − objetivo)**2` sobre las temperaturas estandarizadas;
- semilla `42`, para controlar la generación de números aleatorios.

Antes de entrenar, guarde una **copia de los pesos iniciales**. La segunda red
partirá de esos mismos valores. Cada red debe tener su propio optimizador nuevo.
En Keras puede usar `fit`; en PyTorch puede adaptar el ciclo del 30 usando `nn.MSELoss()`. Entrene las
15 épocas completas, sin parada anticipada, y registre la pérdida media de
entrenamiento y validación de cada época. Nunca actualice pesos con validación.

**Compruebe y muestre:** la red tiene **3 169 parámetros** y, para un lote de
256 materiales, produce 256 números, con forma `(256, 1)`. Las temperaturas
objetivo también deben tener forma `(256, 1)` para calcular la pérdida correctamente.

### 3. Cambiar sólo la activación y repetir — 50 a 80 minutos

Defina su propia función con esta regla: devuelve `z` si `z >= 0` y devuelve
`0.1*z` si `z < 0`. Es una variante de ReLU que conserva una pendiente pequeña
para valores negativos. Escríbala con operaciones de TensorFlow o PyTorch;
no use la capa de activación ya preparada por la biblioteca.

Construya otra red con `crear_red`, usando esa función en las dos capas ocultas.
Cargue los pesos iniciales guardados antes del primer entrenamiento. Conserve
las mismas filas, el orden de los lotes, la tasa, las 15 épocas y el tamaño de lote.
Reinicie también el generador que baraja los lotes. Puede reutilizar la lógica
de semillas y copia de pesos del notebook 30.

**Muestre:** una gráfica de ReLU y de la función propia entre −3 y 3; escriba la
pendiente de cada función cuando `z < 0` y cuando `z > 0`. No se pide la derivada
en cero, donde las pendientes laterales no coinciden. Entrene la segunda red.
No haga más variantes ni repita con otras semillas.

### 4. Elegir con validación y evaluar en kelvin — 80 a 105 minutos

Use los pesos obtenidos **al terminar la época 15** de cada red. Siga este orden:

1. Obtenga las predicciones de cada red para `xv` y conviértalas a kelvin con
   `kelvin(pred)`. Compare con las temperaturas reales `y[val]`.
2. Calcule el **MAE**, *mean absolute error* o error absoluto medio: el promedio
   de `abs(temperatura_predicha − temperatura_real)`. Si resulta 9 K, significa
   que el error absoluto es de 9 kelvin en promedio sobre ese conjunto;
   no significa que todos los errores sean menores de 9 K.
3. Llene dos filas: nombre de la activación y MAE de validación en K.
   Elija la red con menor MAE. Si hay empate al redondear a dos decimales, elija ReLU.
   Escriba esa decisión **antes** de usar `xs`.
4. Sólo para la red elegida, prediga con `xs`, convierta con `kelvin` y calcule
   el MAE frente a `y[test]`. Agregue ese único resultado como evaluación final.
   No entrene de nuevo ni cambie la decisión después de verlo.
5. Grafique juntas las pérdidas de entrenamiento y validación, por época,
   de la red elegida. Esas pérdidas están en escala estandarizada, no en K.
   Muestre también una tabla con cinco temperaturas reales de prueba, sus
   predicciones y sus errores absolutos en K; use las primeras cinco filas.

No se pide comparar con Ridge, árboles ni ensambles; tampoco calcular otras
métricas, guardar un archivo CSV o buscar los peores casos.

### 5. Explicar y entregar — 105 a 120 minutos

Responda en dos o tres frases cada pregunta:

1. ¿Qué activación eligió y qué valores de validación sustentan la decisión?
2. ¿Qué indican las curvas sobre la diferencia entre aprender entrenamiento
   y predecir ejemplos nuevos? Use sus curvas, aunque no haya sobreajuste visible.
3. ¿Qué significa el MAE final en kelvin y qué no permite concluir sobre un
   material de una familia química que no aparece en los datos?

**Entrega:** un `.ipynb` con los dos entrenamientos, comprobación de formas y
parámetros, gráfica de activaciones, tabla de validación, un MAE de prueba,
curvas de la red elegida, cinco predicciones y las tres respuestas.
Indique nombres, framework y versión. No hace falta otro informe.

**Control del tiempo:** al minuto 80 cierre los entrenamientos. Si el equipo
es lento, acuerde desde el inicio ocho épocas para **ambas** redes y registre
ese cambio; use la última época de ese presupuesto para comparar. No reduzca
el presupuesto de una sola red. No se evalúa por obtener un error prefijado.

## Alternativa B — Celular y papel, sin programación

La [guía completa](TALLER_SIN_COMPUTADOR.md) contiene los datos, las ecuaciones,
los fragmentos de código que se deben **leer**, las preguntas y las fuentes.
En dos horas se entregan cálculos propios, una correspondencia entre matemáticas
y bibliotecas y decisiones justificadas sobre un experimento. No se solicita
escribir, corregir ni ejecutar programas. La consulta se limita a secciones
concretas de documentación oficial; no hace falta leer un artículo completo.

## Evaluación de cada alternativa

| Criterio | A: programación | B: celular y papel |
|---|---:|---:|
| Construcción y comprensión de la red | 25 % | 25 % |
| Activación y efecto en el aprendizaje | 25 % | 25 % |
| Entrenamiento y uso de las herramientas | 25 % | 25 % |
| Evaluación e interpretación física | 25 % | 25 % |

En A se valora el código y sus resultados; en B, el procedimiento de cálculo,
la interpretación precisa del código suministrado y el uso razonado de fuentes.
Copiar definiciones sin aplicarlas al caso no satisface B. Los pesos pertenecen
al taller y no cambian los porcentajes globales del curso.
