# Taller: auditar y diseñar un sistema de visión con celular y papel

**Dedicación estimada: cuatro horas para la ruta esencial.** La teoría matemática
ya vista es suficiente; no se exige experiencia programando. La dificultad es
ligeramente mayor que seguir los ejemplos porque debes justificar y corregir decisiones.
Necesitas celular con navegador, calculadora, papel y lápiz; puedes escribir tus
respuestas en una aplicación de notas. No necesitas computador, Colab, instalaciones,
cuentas ni ejecutar código. Se puede realizar como alternativa a los dos notebooks
o como taller adicional después de ellos. Las cuatro horas corresponden a este taller.

La preparación buscada es similar en interpretación de imágenes, seguimiento de
operaciones, diseño de CNN y evaluación de transferencia. Programar, depurar y medir
tiempos reales requiere posteriormente práctica con los notebooks. Aquí la dificultad
adicional está en justificar decisiones, detectar errores y resolver restricciones,
no en memorizar nombres de funciones.

## Ruta y forma de trabajo

| Actividad | Evidencia |
|---|---|
| Leer la ficha y abrir las referencias | Glosario de cinco términos |
| A. Fotografías, geometría y color | Comparación visual y cálculos |
| B. Seguir una CNN a mano | Mapas, formas y parámetros |
| C. Diseñar ramas y atajos | Diagrama con presupuesto |
| D. Transferencia bajo restricciones | Preprocesamiento y decisión |
| E. Auditar un experimento | Particiones, métricas y correcciones |
| F. Defender una propuesta | Protocolo y conclusión |

Resuelve cada ejercicio mostrando tu procedimiento y justificando tus decisiones.
Si te atascas, anota qué te falta, usa el ejemplo o una pista y continúa.
Al corregir, conserva el intento inicial y explica el cambio. Internet sirve para
consultar documentación y observar fotografías; copiar definiciones no resuelve los casos.

**Ruta esencial:** resuelve A–F, salvo los fragmentos marcados como ampliación opcional.
Las ampliaciones quedan fuera de las cuatro horas. Usa respuestas breves y operaciones
legibles, no ensayos por pregunta. La estimación incluye lectura, solución y autocorrección;
no ha sido cronometrada con estudiantes. Si una operación se repite, muestra una vez el
procedimiento y luego los resultados. Las fórmulas necesarias están en esta guía.

## Ficha de consulta: lo necesario para comenzar

- **Imagen:** arreglo de alto × ancho × canales. Pillow informa tamaño como
  (ancho, alto); una imagen RGB tiene tres canales. Un lote añade una dimensión inicial.
- **Filtro:** conjunto pequeño de pesos que se desplaza sobre la imagen. En las
  cuentas de este taller se multiplica elemento a elemento y se suma **sin invertir
  el filtro**, como la correlación cruzada usada habitualmente por las CNN.
- **ReLU:** reemplaza cada número negativo por cero. **Max pooling:** conserva el
  máximo de cada ventana. **GAP:** promedio espacial de cada canal.
- **Convolución:** lado de salida = piso((lado de entrada + 2p − k)/s) + 1,
  con padding p, kernel k y stride s. Con `same` y stride 1 conserva alto y ancho.
  Parámetros con sesgo = (k × k × canales de entrada + 1) × filtros.
- **Dense:** parámetros = (entradas + 1) × salidas. ReLU, pooling, concatenación,
  suma y GAP no tienen parámetros entrenables en estos ejemplos.
- **Inception:** procesa ramas en paralelo y concatena sus canales. **Residual:**
  suma una transformación y un atajo de igual forma. Una convolución 1×1 puede ajustar canales.
- **Transferencia:** aprovechar pesos aprendidos previamente. Congelar evita
  actualizar esos pesos; la cabeza nueva aprende las clases actuales. Fine-tuning
  permite actualizar parte de la base con una tasa de aprendizaje pequeña.
- **Evaluación:** train ajusta pesos; validación elige diseños; test estima el
  resultado final tras cerrar decisiones. Una misma fuente de imágenes puede crear
  dependencia aunque los archivos tengan nombres distintos.

En tus palabras define canal, filtro, parámetro, base congelada y fuga de información.
La ficha permanece disponible durante todo el taller.

## A. Fotografías, geometría y color

**Ejemplo resuelto con otras dimensiones:** para ajustar una foto de 600×400 dentro
de 120×120 sin deformar, el factor es min(120/600,120/400)=0.2. El contenido mide
120×80; quedan 40 píxeles de relleno vertical, 20 arriba y 20 abajo. El recorte a un
cuadrado central sería (100,0,500,400). Aplica el mismo razonamiento a A2.

Abre el [tutorial de clasificación de flores de TensorFlow](https://www.tensorflow.org/tutorials/images/classification)
y busca las fotografías de rosas y tulipanes. No ejecutes las celdas. Elige una foto
de cada grupo; identifícala mediante el enlace y su posición en la página.
Si no cargan las imágenes, toma dos fotos propias de flores u objetos con la cámara
del celular. En ese caso describe tus objetos y no les atribuyas las etiquetas del dataset.

**A1. Observar antes de transformar.** Dibuja de manera esquemática objeto y fondo.
Para cada imagen anota: qué rasgo puede ayudar a clasificar, qué rasgo del fondo
podría convertirse en una pista engañosa, y qué información perderías al convertir
a grises. Propón un cambio de brillo y un recorte que podrían dificultar reconocerla.
Si el editor nativo permite recortar, haz una comparación; el dibujo es suficiente.
No tienes que instalar un editor ni estimar colores numéricos desde la pantalla.

**A2. Elegir una transformación.** Ahora considera una fotografía hipotética de
1200 píxeles de ancho y 800 de alto. Debe entrar a una red de 128×128.

1. Calcula el factor horizontal y vertical si se deforma directamente a 128×128.
2. Para recortar primero un cuadrado central de 800×800, escribe la caja
   (izquierda, arriba, derecha, abajo). ¿Qué fracción del área original se pierde?
3. Para conservar todo el encuadre, explica dónde aparecería el relleno al ajustar
   proporcionalmente dentro de 128×128. No necesitas redondear a píxeles enteros.
4. Si los pétalos útiles están en los extremos izquierdo y derecho, elige entre
   deformar, recortar o rellenar. Justifica tu elección y una desventaja.

Consulta los apartados `fit` y `pad` de [Pillow ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html).
Anota qué función corresponde a dos de tus propuestas y explica su efecto en esta foto.

**A3. Detectar dos errores silenciosos.** Un píxel RGB es (200, 100, 50). Escribe
la terna BGR equivalente y explica qué pasaría si se mostrara como RGB sin convertir.
Una máscara de amarillo acepta H entre 20 y 40, S entre 70 y 255 y V entre 50 y 255
en HSV de OpenCV de 8 bits. Decide si pasa cada píxel HSV:

| Píxel | H | S | V |
|---|---:|---:|---:|
| P | 30 | 150 | 200 |
| Q | 30 | 30 | 200 |
| R | 60 | 150 | 200 |
| S | 30 | 150 | 20 |

¿Por qué una máscara correcta puede incluir fondo y excluir partes de una flor?
Explica por qué Canny puede detectar bordes sin reconocer la especie.
Consulta la nota de rangos de [OpenCV: espacios de color](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html).

## B. Seguir una CNN a mano

**Ejemplo de una salida:** si una ventana es `[[2,0],[1,3]]` y el filtro
`[[1,-1],[1,-1]]`, la respuesta es 2−0+1−3=0. Desplazar la ventana una posición
genera la salida siguiente. Para contar parámetros, una Conv 3×3 con tres canales
de entrada y cuatro filtros tiene (3×3×3+1)×4=112. La resolución espacial no cambia
este conteo porque el filtro se comparte en todas las posiciones.

Los números siguientes son **datos didácticos inventados**, no píxeles extraídos
de las fotografías anteriores. Muestran un mecanismo con cuentas pequeñas.

**B1. Filtrar y reducir.** Usa esta imagen de un canal y este filtro:

```text
Imagen I             Filtro K
1  1  0  0           1  -1
1  1  0  0           1  -1
0  0  1  1
0  0  1  1
```

Aplica K con stride 1, sin padding y sesgo cero. Calcula las nueve salidas y muestra
al menos dos sumas completas. Aplica ReLU y después max pooling 2×2 con stride 1.
¿Qué orientación de transición responde positivamente? ¿Qué información descarta ReLU?
Describe la diferencia entre suavizar primero para Canny y aprender los pesos del filtro.

**B2. Auditar otra red.** Esta red es independiente de la matriz anterior:

```text
Entrada 8×8×1
→ Conv 3×3, 2 filtros, stride 1, valid, con sesgo
→ ReLU
→ MaxPool 2×2, stride 2, valid
→ GAP
→ Dense de 3 salidas, con sesgo y softmax
```

Completa en papel una tabla: etapa, forma de salida, parámetros. Da el total.
Un compañero cambia GAP por Flatten: calcula los nuevos parámetros totales y
explica qué relación tiene ese cambio con el riesgo de sobreajuste en pocas fotos.
Escribe también la forma de entrada y salida final para un lote de cuatro imágenes.

**Pista:** GAP conserva un número por canal; Flatten conserva cada posición.

## C. Diseñar una red de ramas y atajos

**Ejemplo de compatibilidad:** concatenar dos mapas 10×10×4 produce 10×10×8;
sumarlos produce 10×10×4. Para sumar 10×10×8 a un atajo 10×10×4 puedes proyectar
el primero con una Conv 1×1 de cuatro filtros: (8+1)×4=36 parámetros.

Construye en papel un bloque inspirado en Inception y ResNet. No tiene pesos
preentrenados y no pretende reproducir InceptionResNetV2 completa.
La entrada es 24×24×16. Todas las convoluciones usan stride 1, `same` y sesgo.
Las cuatro ramas parten de esa misma entrada:

| Rama | Operaciones sucesivas |
|---|---|
| A | Conv 1×1, f filtros |
| B | Conv 1×1, f filtros → Conv 3×3, f filtros |
| C | Conv 1×1, f filtros → Conv 3×3, f filtros → Conv 3×3, f filtros |
| D | MaxPool 3×3, stride 1, same → Conv 1×1, f filtros |

Concatena las ramas por canales y aplica una Conv 1×1 de **16 filtros**. Suma su
salida con la entrada original y aplica ReLU. Considera ReLU tras las convoluciones
internas de las ramas; no añade parámetros. La proyección final es lineal antes de sumar.

**C1. Diseñar con un límite.** El presupuesto es **3000 parámetros** para el bloque.
Para f=8, dibuja el flujo y anota la forma después de cada rama, concatenación,
proyección y suma. Cuenta parámetros por rama y de la proyección. ¿Cumple el límite?

**C2. Detectar un diseño inválido.** Un compañero suma directamente la concatenación
con la entrada, omitiendo la proyección final. Explica por qué falla. Propón una
reparación que conserve 16 canales de salida. **Ampliación opcional:** propone otra
que conserve los canales concatenados. La primera ya fue contada en C1.

**C3. Decidir sin ejecutar.** Diseña una comparación con/sin atajo que mantenga iguales ramas, datos y épocas.
Indica qué métrica mirarías y qué resultado refutaría tu hipótesis. No asumas que
la conexión residual siempre gana en un experimento pequeño.

**Ampliación opcional:** recalcula el total para f=12 manteniendo 16 canales después
de la proyección y decide si cumple el presupuesto. No necesitas esta cuenta para C3.

## D. Transferencia bajo restricciones

**Ejemplo resuelto:** una base congelada entrega 64 características y una Dense
predice tres clases. Solo se entrenan (64+1)×3=195 parámetros. Congelar no borra
los pesos anteriores. Para una intensidad 127.5, la regla de Inception da
127.5/127.5−1=0; esta operación no es la normalización a [0,1].

**D1. Preparar el mismo píxel para dos bases.** Parte del RGB (200,100,50), float32
en escala 0–255. Para este cálculo usa estas reglas oficiales de preprocesamiento:

- InceptionV3: cada canal se transforma como x/127.5 − 1.
- ResNet152: invierte RGB a BGR y resta, en ese orden, (103.939,116.779,123.680).

Calcula las dos salidas con tres decimales. Un equipo divide primero por 255 y
después aplica esas reglas. Explica por qué el código podría ejecutarse sin error
y aun así producir características inadecuadas. No confundas ResNet152 con ResNet152V2.
Consulta [InceptionV3](https://keras.io/api/applications/inceptionv3/) y
[ResNet](https://keras.io/api/applications/resnet/); si la página muestra solo el
índice, usa los enlaces a la función o a su código fuente. Las fórmulas necesarias ya están aquí.

**D2. Qué se entrena.** Ambas bases entregan aquí vectores de 2048 características;
añadimos directamente una Dense de cinco clases con sesgo. Calcula los parámetros
entrenables cuando la base está congelada. **Ampliación opcional:** si almacenas 500 vectores float32 de
2048 valores, calcula sus bytes y MiB (1 MiB = 1 048 576 bytes), sin contar etiquetas
ni estructura del archivo. **Para todos:** explica por qué descongelar la base vuelve obsoletas
las características almacenadas para entrenar esa base.

**D3. Elegir con una restricción explícita.** Esta tabla es un **escenario inventado**,
no mediciones del repositorio ni promesas sobre Colab. Los tiempos se midieron en
un mismo entorno hipotético y excluyen descargas. Ya incluyen todas las épocas.

| Sistema | Macro-F1 validación | Extracción (min) | Ajuste (min) |
|---|---:|---:|---:|
| InceptionV3 congelada + cabeza | 0.80 | 6 | 1 |
| ResNet152 congelada + cabeza | 0.82 | 15 | 1 |
| Red propia desde cero | 0.68 | 0 | 12 |

Debes repetir un sistema desde las imágenes con un presupuesto de **10 minutos**,
sin características almacenadas. Elige el sistema factible con mayor macro-F1.
¿Cambia la decisión si ya tienes las características de ambas bases y solo debes
reentrenar sus cabezas? ¿Puedes concluir que ResNet152 siempre es mejor por 0.02?

Lee solo los apartados sobre congelación y fine-tuning de la
[guía de transferencia](https://keras.io/guides/transfer_learning/).
Escribe tres precauciones para una futura fase de fine-tuning: tasa de aprendizaje,
recompilación y BatchNorm. No tienes que ejecutar esa fase.

**Apoyo:** BatchNorm conserva estadísticas aprendidas de activaciones. Con pocos
datos conviene evitar cambiarlas indiscriminadamente. En esta estrategia se mantiene
en inferencia; al cambiar las capas entrenables se recompila y se usa una tasa pequeña.
Explica el propósito de estas medidas; no se pide conocer su sintaxis.

## E. Auditar un experimento

**Ejemplo de métrica:** para una clase con TP=4, FP=1 y FN=3,
precision=4/(4+1)=0.8, recall=4/(4+3)≈0.571 y F1=8/(8+1+3)≈0.667.
TP son aciertos de esa clase; FP son otros objetos predichos como ella; FN son
objetos de esa clase predichos como otra. En una matriz, mira columna para FP y fila para FN.

**E1. Diseñar particiones sin fuga.** Tienes cinco clases de flores. Por clase hay
20 plantas distintas y cinco fotos por planta, tomadas en una misma sesión.
El objetivo es reconocer **plantas nuevas**, no nuevas fotos de plantas conocidas.

Un equipo mezcla las 500 fotos y reparte 60/20/20 al azar por archivo. Explica el
problema y propone un reparto 60/20/20 **por planta dentro de cada clase**. Da los
números de plantas y fotos por clase y los totales de cada partición. ¿Cuándo deben
generarse los giros y recortes aumentados? ¿Cómo agruparías las fotos si el objetivo
cambiara a generalizar a jardines nuevos y varias plantas compartieran jardín?

**E2. Interpretar un registro.** Son resultados didácticos inventados:

| Época | Pérdida train | Pérdida validación |
|---|---:|---:|
| 1 | 1.40 | 1.30 |
| 2 | 1.00 | 1.05 |
| 3 | 0.75 | 0.90 |
| 4 | 0.55 | 0.93 |
| 5 | 0.40 | 1.08 |

Si el criterio de selección es menor pérdida de validación, ¿qué pesos conservarías?
Con paciencia de dos épocas consecutivas sin mejora, ¿cuándo pararías? ¿Qué patrón
sugiere sobreajuste? Propón un cambio y una comparación que pueda evaluar su efecto.

**E3. Calcular y cuestionar métricas.** Tras cerrar decisiones se obtuvo esta matriz
didáctica de test para **tres clases**, una tarea independiente del caso de cinco clases.
Las filas son reales; las columnas, predichas:

| Real / predicha | A | B | C |
|---|---:|---:|---:|
| A | 8 | 2 | 0 |
| B | 1 | 6 | 3 |
| C | 0 | 2 | 8 |

Calcula accuracy, recall de B y macro-F1. Para cada clase,
F1 = 2TP/(2TP+FP+FN); macro-F1 es el promedio de los tres F1. Muestra de dónde
salen FP y FN. ¿Qué error investigarías primero y qué evidencia visual buscarías?
Si corriges una sola predicción equivocada, ¿cuánto aumenta accuracy en puntos
porcentuales? Explica por qué un test pequeño no permite afirmar mejoras definitivas.

**E4. Detectar fallos en un plan escrito.** Marca y repara al menos tres.
**Ampliación opcional:** encuentra cinco o más.

> Primero duplicamos cada foto mediante giros y luego repartimos archivos al azar.
> Pasamos imágenes BGR/255 directamente a ResNet152 porque todas las redes usan lo
> mismo. Elegimos veinte arquitecturas por su accuracy de test. Para afinar la base,
> cambiamos trainable a verdadero después de compilar y seguimos entrenando la cabeza
> sobre vectores que extrajimos ayer. Conservamos la época de menor pérdida de train.
> La mayor accuracy demuestra que la arquitectura propia es inferior, aunque solo
> las otras dos recibieron preentrenamiento en ImageNet.

No es suficiente escribir “está mal”: explica la consecuencia y una corrección.

## F. Defensa de una propuesta

Entrega un protocolo de 150–200 palabras para clasificar fotos de flores de un
jardín nuevo. Debe indicar: unidad de partición, transformación geométrica, orden
de canales y normalización, sistema inicial, presupuesto, experimento con una sola
modificación, criterio de validación y momento de usar test. Incluye dos limitaciones.

Responde además en dos frases: ¿qué parte podrías implementar con seguridad al
abrir los notebooks y qué necesitarías comprobar mediante ejecución real?

## Entrega y autocorrección

Entrega fotos legibles de hojas numeradas o un documento escrito desde el celular.
Identifica A–F, incluye operaciones, diagramas y enlaces consultados. No se pide
enviar archivos a servicios externos ni entrenar un modelo. Puedes adjuntar fotos
propias o señalar las fotografías de referencia sin redistribuirlas.

Antes de terminar, comprueba que cada operación muestre entrada y salida, que las
sumas residuales tengan formas compatibles y que ninguna decisión use test para
seleccionar. Justifica tu elección de arquitectura y reconoce las limitaciones.

Revisa al final tus operaciones con la ficha y los ejemplos, y escribe tres errores
que ahora sabrías detectar. Esta revisión forma parte del
aprendizaje; si necesitas más tiempo, regístralo en vez de omitir la justificación.
