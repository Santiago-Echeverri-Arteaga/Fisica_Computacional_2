# Alternativa B — Redes neuronales con celular y papel

**Duración: 120 minutos. Sin programar.** Necesita celular con navegador, papel,
lápiz y calculadora básica del celular. Puede entregar fotos legibles de sus hojas
numeradas o un documento redactado desde el celular. No necesita computador,
cuenta de Colab, instalación de bibliotecas ni ejecución de ningún fragmento.

El objetivo es demostrar que comprende lo que hace un programa de entrenamiento:
seguir sus operaciones, calcular un paso, relacionarlo con la documentación y
reconocer una evaluación incorrecta. Se requiere procedimiento y justificación;
una recopilación de definiciones no es suficiente.

## 1. Reconocer la red y sus herramientas — 0 a 20 minutos

Una **red feedforward** lleva información de entrada a salida sin ciclos. Una
**capa densa** conecta cada entrada con cada neurona de esa capa. Una red con
capas ocultas también recibe el nombre de **perceptrón multicapa**, **MLP**
(*multilayer perceptron*). Keras permite describirla mediante capas; aquí se usa
Keras con TensorFlow. PyTorch ofrece sus propias capas y herramientas para el
mismo tipo de cálculo.

**ReLU** significa unidad lineal rectificada (*rectified linear unit*) y devuelve
`max(0, z)`: conserva los valores positivos y convierte los negativos en cero.

Lea estos fragmentos como descripciones de **la misma arquitectura**. No los copie
en un editor ni los ejecute; no incluyen datos ni pretenden ser programas completos.

**Keras:**

```python
red = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(1)
])
red.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
```

**PyTorch:**

```python
red = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)
criterio = nn.MSELoss()
optimizador = torch.optim.SGD(red.parameters(), lr=0.1)
```

`SGD` significa descenso de gradiente estocástico: un método para actualizar
parámetros usando gradientes calculados con ejemplos o lotes. `mse` y `MSELoss`
indican **error cuadrático medio**, o **MSE** (*mean squared error*).

Consulte únicamente los apartados **Arguments** e **Input/Output shape** de
[Dense, documentación oficial de Keras](https://keras.io/api/layers/core_layers/dense/)
y **Hyperparameters** de
[Optimización, tutorial oficial de PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
Puede usar la traducción del navegador. Los enlaces son lecturas, no tareas de instalación.

**Responda:**

1. Dibuje las dos entradas, las dos neuronas ocultas y la salida. Marque los sesgos.
   Cuente por separado pesos y sesgos de cada capa, y luego el total. Recuerde:
   una capa con `n` entradas y `m` salidas tiene `n*m` pesos y `m` sesgos.
2. Para un lote de cuatro ejemplos, escriba la forma de la entrada, de la salida
   oculta y de la salida final. Explique qué significa cada dimensión.
3. ¿Por qué la última capa no tiene ReLU si se quiere predecir una temperatura
   **estandarizada**, que puede ser negativa? ¿Un número estandarizado negativo
   implica una temperatura negativa en kelvin?

Anote al final de este bloque qué apartado de la documentación sustentó una
respuesta y aplíquelo a uno de los números de esta red. No basta con pegar el enlace.

## 2. Hacer una predicción y una actualización — 20 a 45 minutos

Imagine que las dos entradas son propiedades estandarizadas de un material.
La salida representa su temperatura crítica estandarizada. Los números son
**un ejemplo didáctico inventado**, no mediciones de UCI ni pesos de una red entrenada.

Use `x1 = 1`, `x2 = 2` y objetivo `y = 1.3`. La red tiene estos pesos y sesgos:

- primera neurona: `z1 = 1*x1 + 0.5*x2 − 0.5`;
- segunda neurona: `z2 = −1*x1 + 0.25*x2 + 0`;
- activación: `a1 = max(0, z1)`, `a2 = max(0, z2)`;
- salida: `ŷ = v1*a1 + v2*a2 + c`, con `v1 = 0.4`, `v2 = −0.2`, `c = 0.1`;
- pérdida de este único ejemplo: `L = (ŷ − y)²`.

**Haga a mano, mostrando sustituciones:**

1. Calcule `z1`, `z2`, `a1`, `a2`, `ŷ` y `L`.
2. Calcule las tres derivadas de la capa de salida usando:
   `∂L/∂v1 = 2*(ŷ−y)*a1`, `∂L/∂v2 = 2*(ŷ−y)*a2` y `∂L/∂c = 2*(ŷ−y)`.
3. Actualice **simultáneamente sólo** `v1`, `v2` y `c`, con tasa `η = 0.1` y regla
   `parámetro_nuevo = parámetro_anterior − η*derivada`. Todas las derivadas se
   calculan con los valores anteriores. Mantenga fija la capa oculta en este paso.
4. Recalcule la predicción y la pérdida con los tres parámetros nuevos.
   ¿Disminuyó la pérdida? ¿Este resultado de un ejemplo garantiza mejorar
   en otros materiales? Justifique ambas respuestas.

Redondee a cuatro decimales al presentar resultados, no en pasos intermedios.

## 3. Estudiar una activación propia — 45 a 65 minutos

Regrese a **todos los pesos originales**, anteriores a la actualización.
Sustituya ReLU por la función `φ(z) = z` cuando `z >= 0` y `φ(z) = 0.1*z`
cuando `z < 0`. Es una variante con pendiente pequeña en el lado negativo.

1. Haga una tabla de ReLU y `φ` para `z = −2, −1, 0, 1, 2`, y dibuje las
   dos gráficas sobre los mismos ejes. Escriba sus pendientes para `z < 0`
   y `z > 0`. Explique por qué no hay una única derivada ordinaria en cero.
2. Recalcule `a1`, `a2`, `ŷ` y `L` de la red con `φ`. No actualice pesos esta vez.
3. Calcule `∂ŷ/∂z2 = v2*φ′(z2)` para cada activación en el `z2` obtenido.
   Explique cómo cambia el paso del gradiente por esa neurona.
4. ¿Permite esta sola predicción afirmar que la función propia es mejor?
   Nombre dos condiciones que mantendría iguales al entrenar ambas redes.

## 4. Leer un paso de entrenamiento — 65 a 95 minutos

Los siguientes pasos suponen que `red`, datos y optimizador ya existen. `xb`
contiene un lote de entradas y `yb` sus objetivos; tienen formas compatibles.
**No escriba código ni lo ejecute.** Identifique qué hace cada línea.

**TensorFlow, líneas T1–T5:**

```python
# T1
with tf.GradientTape() as cinta:
    pred = red(xb)                              # T2
    perdida = tf.reduce_mean((pred - yb)**2)     # T3
gradientes = cinta.gradient(perdida, red.trainable_variables)  # T4
optimizador.apply_gradients(zip(gradientes, red.trainable_variables))  # T5
```

**PyTorch, líneas P1–P5:**

```python
optimizador.zero_grad()            # P1
pred = red(xb)                     # P2
perdida = criterio(pred, yb)       # P3
perdida.backward()                # P4
optimizador.step()                # P5
```

Consulte **Gradient tapes** en
[Autodiferenciación de TensorFlow](https://www.tensorflow.org/guide/autodiff)
y **Optimization Loop** en
[Optimización de PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
Use sólo esas secciones y los fragmentos anteriores. El tutorial llama
`test_loop` al ciclo de evaluación; en este taller, las decisiones se toman
con validación y la prueba se reserva para el final.

**Entregue:**

1. Cuatro pares de líneas T/P que correspondan a predecir, medir pérdida,
   calcular gradientes y actualizar pesos. Explique cada operación con una frase
   y relacione una con un cálculo concreto de la actividad 2.
2. Explique el papel de T1 y de P1: ¿por qué en PyTorch se borran gradientes
   antes del siguiente lote? ¿`backward()` cambia por sí solo los pesos?
3. Un compañero afirma: «Si pongo `red.eval()`, PyTorch ya no calcula gradientes».
   Consulte los comentarios de `test_loop`, en **Full Implementation**,
   sobre `eval()` y `torch.no_grad()` en el tutorial de PyTorch. Indique si tiene razón y explique la diferencia.
4. Otro compañero selecciona la activación que obtiene menor error en **prueba**
   y luego presenta ese mismo error como evaluación independiente. Explique el
   problema y describa en palabras el orden correcto de entrenamiento,
   validación y prueba. No escriba un programa corregido.

Para los puntos 2 y 3, anote enlace y nombre del apartado consultado, y explique
cómo la lectura sostiene su respuesta. No se exige transcribir párrafos.

## 5. Decidir con resultados y recuperar unidades — 95 a 115 minutos

Esta tabla es **un experimento hipotético suministrado para análisis**. Son dos
redes con iguales datos, pesos iniciales y presupuesto de 15 épocas. Una época
es un recorrido por entrenamiento. Se registraron resultados en las épocas 5 y 15.
El acuerdo previo es comparar los pesos **al terminar la época 15** por error
absoluto medio de validación; no seleccionar una época anterior después de ver la tabla.

**MAE** (*mean absolute error*) es el **error absoluto medio**:
`promedio de abs(predicción − valor real)`. Aquí se expresa en kelvin.
Las pérdidas MSE se calcularon con temperaturas estandarizadas y no están en K.

| Activación / época | MSE entrenamiento | MSE validación | MAE validación [K] |
|---|---:|---:|---:|
| ReLU / 5 | 0.28 | 0.31 | 11.0 |
| ReLU / 15 | 0.12 | 0.27 | 9.4 |
| Propia / 5 | 0.30 | 0.29 | 10.7 |
| Propia / 15 | 0.10 | 0.34 | 10.1 |

1. Elija una red siguiendo el acuerdo. Explique por qué elegir sólo la menor
   pérdida de entrenamiento llevaría a una decisión diferente. Señale qué
   evolución sugiere sobreajuste: aprender entrenamiento sin mejorar validación.
2. Para la red elegida se suministran tres salidas de prueba estandarizadas:
   `−0.65`, `0.80` y `1.10`. En entrenamiento se había usado media `30 K` y
   desviación estándar `20 K`. Conviértalas mediante `Tc_predicha = 30 + 20*salida`.
   Las temperaturas reales correspondientes son `20 K`, `40 K` y `60 K`.
   Calcule tres errores absolutos y su promedio. Muestre las operaciones.
3. ¿Por qué este promedio de tres casos no estima con precisión el error para
   todos los superconductores? ¿Por qué separar por material es mejor que repetir
   el mismo material en entrenamiento y prueba? Una frase razonada para cada punto.

## Entrega y revisión — 115 a 120 minutos

Revise signos, unidades y que haya vuelto a los pesos originales en la actividad 3.
Entregue **3–5 páginas o fotos numeradas**, con nombre y las cinco actividades
identificadas. Se aceptan gráficas a mano y cuentas con calculadora. No se exige
archivo de código, notebook, PDF, video ni capturas de una biblioteca funcionando.

Se evalúan cuatro aspectos, cada uno con 25 %: comprensión de la red
(actividad 1), activaciones (3), entrenamiento y herramientas (2 y 4), y evaluación
física (5). Se valoran operaciones, explicaciones y referencias aplicadas al caso,
no extensión ni diseño de las hojas. Esta alternativa cubre los mismos conceptos
que la programación mediante una comprobación manual y argumentada.

[Volver a las dos alternativas](TALLER_FEEDFORWARD.md)
