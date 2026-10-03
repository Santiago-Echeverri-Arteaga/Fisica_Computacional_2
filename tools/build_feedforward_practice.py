"""Sesión integrada y prácticas con datos abiertos; fuentes de los notebooks 30 y 36."""

from build_initial_notebooks import COLAB_ROOT, code, md, write_notebook


def portada(path, title, duration, environment):
    return md(f'''[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_ROOT}/{path})

# {title}

**Duración:** {duration}. **Entorno:** {environment}.
Ejecute de arriba hacia abajo en un runtime limpio. Las descargas se guardan en caché.
''')


def build_integrated():
    path = '03_redes_fundamentos/30_neurona_y_feedforward_desde_cero.ipynb'
    cells = [portada(path, 'Feedforward en la práctica: TensorFlow, Keras y PyTorch',
                     '120 minutos', 'Colab, CPU suficiente, Internet, TensorFlow y PyTorch'), md(r'''
## Ruta del martes

La teoría de las redes feedforward es el punto de partida. Hoy traducimos
$a=\phi(XW+b)$ a código, entrenamos con imágenes reales y discutimos qué se gana
y qué se pierde al cambiar de nivel de abstracción.

| Minutos | Trabajo |
|---|---|
| 0–12 | runtime, Fashion-MNIST, particiones y EDA |
| 12–25 | tensores, formas y gradientes en ambos frameworks |
| 25–45 | red y entrenamiento con TensorFlow desde variables |
| 45–60 | la misma arquitectura con Keras: capas, compile, fit |
| 60–80 | PyTorch: Module, DataLoader y ciclo de entrenamiento |
| 80–105 | activaciones propias: derivadas, gradientes y ablación |
| 105–115 | evaluación final, errores y comparación con ML |
| 115–120 | interpretación física y preparación del taller |

**Al terminar:** identificar cada tensor del entrenamiento; implementar un MLP
en los dos frameworks; definir una activación diferenciable; separar selección
y evaluación. No se necesitan CNN, RNN ni técnicas de cursos posteriores.

Keras es una API de alto nivel que admite varios backends. Aquí se usa
`tf.keras` con TensorFlow. TensorFlow y PyTorch proporcionan tensores,
diferenciación automática y ejecución en dispositivos. `fit` automatiza el mismo
ciclo que luego escribimos de forma explícita; no sustituye la elección de datos,
pérdida y protocolo.
'''), code('''
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import copy
import time
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

SEMILLA = 42
MODO = 'clase'  # 'completo': 48 000 imágenes de entrenamiento y más épocas
assert MODO in {'clase', 'completo'}
EPOCAS = 4 if MODO == 'clase' else 10
BATCH = 256
keras.utils.set_random_seed(SEMILLA)
torch.manual_seed(SEMILLA)
torch.set_num_threads(2)
print({'Python': platform.python_version(), 'TensorFlow': tf.__version__,
       'Keras': keras.__version__, 'PyTorch': torch.__version__})
print('GPU TF:', tf.config.list_physical_devices('GPU'))
# CPU en los tres casos para facilitar la comparación; GPU no es obligatoria.
dispositivo = torch.device('cpu')
'''), md('''
## 1. Datos, escala y línea base

Fashion-MNIST contiene 60 000 imágenes de desarrollo y 10 000 de prueba,
28×28 píxeles, diez clases de prendas. Conservamos la prueba oficial cerrada.
En modo clase usamos 12 000 ejemplos de entrenamiento y 3 000 de validación,
muestreados con estratificación. El modo completo usa 48 000/12 000.
Dividir por 255 es una transformación fija del rango del sensor: no estima
estadísticos con el test. Aplanar convierte cada imagen en 784 entradas de una
red densa; no estamos usando convoluciones.

Fuente y licencia: [Keras / Fashion-MNIST](https://keras.io/api/datasets/fashion_mnist/),
Zalando SE, MIT. La primera ejecución necesita Internet. Si falla la descarga,
restablezca la conexión y ejecute de nuevo; no sustituya silenciosamente los datos.
'''), code('''
(imagenes, etiquetas), (imagenes_test, y_test) = keras.datasets.fashion_mnist.load_data()
ids = np.arange(len(etiquetas))
tr, va = train_test_split(ids, test_size=.2, stratify=etiquetas, random_state=SEMILLA)
if MODO == 'clase':
    tr, _ = train_test_split(tr, train_size=12000, stratify=etiquetas[tr], random_state=SEMILLA)
    va, _ = train_test_split(va, train_size=3000, stratify=etiquetas[va], random_state=SEMILLA)
X_train = imagenes[tr].reshape(-1, 784).astype('float32') / 255
X_val = imagenes[va].reshape(-1, 784).astype('float32') / 255
y_train, y_val = etiquetas[tr].astype('int64'), etiquetas[va].astype('int64')
assert not set(tr) & set(va)
assert X_train.shape[1] == 784 and np.isfinite(X_train).all()
nombres = ['camiseta', 'pantalón', 'suéter', 'vestido', 'abrigo',
           'sandalia', 'camisa', 'zapatilla', 'bolso', 'botín']
display(pd.Series(y_train).value_counts().sort_index().rename('n_train').to_frame())
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, i in zip(axes.flat, range(10)):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(nombres[y_train[i]]); ax.axis('off')
plt.tight_layout(); plt.show()
print('Formas:', X_train.shape, X_val.shape, 'rango:', X_train.min(), X_train.max())

t0 = time.perf_counter()
lineal = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30,
                       tol=1e-3, random_state=SEMILLA)
lineal.fit(X_train, y_train)
tiempo_lineal = time.perf_counter() - t0
print('Accuracy de validación, logística SGD:', lineal.score(X_val, y_val))
'''), md(r'''
## 2. Tensores y autodiferenciación

Un tensor tiene forma, tipo y dispositivo. `@` es producto matricial; `*` es
producto elemento a elemento. El sesgo `(64,)` se distribuye sobre el lote.
Para 784→64→10 hay $(784+1)64+(64+1)10=50\,890$ parámetros.
Las etiquetas son enteros, pero las entradas y pesos son flotantes.

Primero comprobamos $L=(wx+b-y)^2$ con $x=2,y=5,w=0.7,b=-0.2$:
$\partial L/\partial w=-15.2$. Autodiferenciación aplica la regla de la cadena,
no diferencias finitas. TensorFlow registra operaciones dentro de `GradientTape`;
PyTorch construye el grafo cuando intervienen tensores con `requires_grad=True`.
'''), code('''
w_tf = tf.Variable(0.7)
with tf.GradientTape() as tape:
    perdida_tf = (w_tf * 2.0 - 0.2 - 5.0)**2
g_tf = tape.gradient(perdida_tf, w_tf)
w_pt = torch.tensor(0.7, requires_grad=True)
perdida_pt = (w_pt * 2.0 - 0.2 - 5.0)**2
perdida_pt.backward()
print('TF:', g_tf.numpy(), 'PyTorch:', w_pt.grad.item())
np.testing.assert_allclose(g_tf.numpy(), -15.2, rtol=1e-6)
np.testing.assert_allclose(w_pt.grad.numpy(), -15.2, rtol=1e-6)
print(tf.constant(X_train[:8]).shape, torch.from_numpy(X_train[:8]).dtype)
'''), md('''
## 3. TensorFlow desde variables: forward → pérdida → gradiente → actualización

La salida contiene diez **logits**, números reales sin normalizar. La entropía
cruzada calcula internamente una versión estable de softmax. No aplicar softmax
antes de una pérdida configurada con `from_logits=True`.

La inicialización usa varianza 2/n en la capa ReLU. `tf.data` prepara minibatches;
el último puede tener menos ejemplos. Ponderamos las pérdidas por su tamaño.
Guardamos los pesos de la época con menor pérdida de validación en los tres
frameworks. El test no interviene. `tf.function` compila el paso: la primera
llamada puede tardar más por el trazado del grafo.
'''), code('''
rng = np.random.default_rng(SEMILLA)
iniciales = [rng.normal(0, np.sqrt(2/784), (784, 64)).astype('float32'),
             np.zeros(64, 'float32'),
             rng.normal(0, np.sqrt(1/64), (64, 10)).astype('float32'),
             np.zeros(10, 'float32')]
W1, b1, W2, b2 = [tf.Variable(a) for a in iniciales]
parametros = [W1, b1, W2, b2]
def forward_tf(x):
    return tf.nn.relu(x @ W1 + b1) @ W2 + b2
criterio_tf = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt_tf = keras.optimizers.Adam(learning_rate=1e-3)

def datos_tf(X, y, barajar=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if barajar:
        ds = ds.shuffle(len(X), seed=SEMILLA, reshuffle_each_iteration=True)
    # Un pool pequeño también funciona en runtimes con pocos recursos.
    opciones = tf.data.Options()
    opciones.threading.private_threadpool_size = 2
    return ds.batch(BATCH).with_options(opciones)

@tf.function
def paso_tf(xb, yb):
    with tf.GradientTape() as tape:
        perdida = criterio_tf(yb, forward_tf(xb))
    gradientes = tape.gradient(perdida, parametros)
    opt_tf.apply_gradients(zip(gradientes, parametros))
    return perdida

train_tf, val_tf = datos_tf(X_train, y_train, True), datos_tf(X_val, y_val)
hist_tf = {'train': [], 'val': []}
mejor = np.inf
t0 = time.perf_counter()
for epoca in range(EPOCAS):
    total = 0.0
    for xb, yb in train_tf:
        total += float(paso_tf(xb, yb)) * len(xb)
    v = sum(float(criterio_tf(yb, forward_tf(xb))) * len(xb) for xb, yb in val_tf) / len(y_val)
    hist_tf['train'].append(total / len(y_train)); hist_tf['val'].append(v)
    if v < mejor:
        mejor = v; mejores_tf = [p.numpy().copy() for p in parametros]
    print(epoca + 1, 'train/val:', hist_tf['train'][-1], v)
for p, valor in zip(parametros, mejores_tf):
    p.assign(valor)
tiempo_tf = time.perf_counter() - t0
'''), md('''
## 4. Keras: capas, modelo, compile y fit

`Input` declara la forma de un ejemplo; `Dense` conserva pesos entrenables.
`compile` elige optimizador, pérdida y métricas; `fit` organiza el entrenamiento;
`predict` hace inferencia. Usamos los mismos pesos iniciales que en TensorFlow
para comparar abstracciones. Esto no garantiza trayectorias idénticas: el orden
de lotes, las implementaciones del optimizador y el dispositivo importan.
'''), code('''
def crear_keras(activacion='relu'):
    modelo = keras.Sequential([
        keras.Input(shape=(784,)),
        keras.layers.Dense(64, activation=activacion),
        keras.layers.Dense(10),
    ])
    modelo.set_weights([a.copy() for a in iniciales])
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    return modelo

modelo_k = crear_keras()
modelo_k.summary()
assert modelo_k.count_params() == 50890
t0 = time.perf_counter()
hist_k = modelo_k.fit(datos_tf(X_train, y_train, True), validation_data=val_tf,
                     epochs=EPOCAS, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(
                         monitor='val_loss', patience=EPOCAS, restore_best_weights=True)])
tiempo_k = time.perf_counter() - t0
'''), md('''
## 5. PyTorch desde la base hasta Module

`nn.Linear` guarda su matriz como `(salidas, entradas)`: transponemos los pesos
para empezar desde la misma función. `TensorDataset` empareja datos y etiquetas;
`DataLoader` produce lotes. `zero_grad` borra gradientes acumulados, `backward`
calcula derivadas y `step` actualiza parámetros. `train()`/`eval()` cambian el modo
de capas que lo necesitan; `no_grad()` evita construir el grafo en inferencia.
Son acciones diferentes. Aquí no hay dropout ni batch normalization.
'''), code('''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.oculta = nn.Linear(784, 64)
        self.salida = nn.Linear(64, 10)
    def forward(self, x):
        return self.salida(torch.relu(self.oculta(x)))

modelo_pt = MLP().to(dispositivo)
with torch.no_grad():
    modelo_pt.oculta.weight.copy_(torch.from_numpy(iniciales[0].T))
    modelo_pt.oculta.bias.copy_(torch.from_numpy(iniciales[1]))
    modelo_pt.salida.weight.copy_(torch.from_numpy(iniciales[2].T))
    modelo_pt.salida.bias.copy_(torch.from_numpy(iniciales[3]))
loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                    batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEMILLA))
xv = torch.from_numpy(X_val).to(dispositivo)
yv = torch.from_numpy(y_val).to(dispositivo)
opt_pt = torch.optim.Adam(modelo_pt.parameters(), lr=1e-3)
criterio_pt = nn.CrossEntropyLoss()
hist_pt = {'train': [], 'val': []}
mejor = np.inf
t0 = time.perf_counter()
for epoca in range(EPOCAS):
    modelo_pt.train(); total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(dispositivo), yb.to(dispositivo)
        opt_pt.zero_grad()
        perdida = criterio_pt(modelo_pt(xb), yb)
        perdida.backward()
        opt_pt.step()
        total += perdida.item() * len(xb)
    modelo_pt.eval()
    with torch.no_grad():
        v = criterio_pt(modelo_pt(xv), yv).item()
    hist_pt['train'].append(total / len(y_train)); hist_pt['val'].append(v)
    if v < mejor:
        mejor = v; mejor_estado = copy.deepcopy(modelo_pt.state_dict())
    print(epoca + 1, 'train/val:', hist_pt['train'][-1], v)
modelo_pt.load_state_dict(mejor_estado)
tiempo_pt = time.perf_counter() - t0
'''), md(r'''
## 6. Activaciones: función, derivada y efecto dentro de la red

Sigmoid se satura en ambos extremos; tanh además centra sus salidas. ReLU
conserva derivada 1 en la región positiva, pero puede dejar neuronas inactivas.
Diseñamos $\phi(z)=\tanh(z)+0.1\sin(2z)$, con
$\phi'(z)=1-\tanh^2(z)+0.2\cos(2z)$.
El término oscilante recuerda una respuesta periódica: es una hipótesis de
diseño, no una ley física ni garantía de mejorar. Su derivada puede ser negativa.

No usar NumPy, `.numpy()` ni `.detach()` dentro de la activación: romperían la
dependencia que necesita autodiferenciación. Usamos operaciones del framework.
'''), code('''
@keras.utils.register_keras_serializable(package='FC2')
def onda_tf(z):
    return tf.math.tanh(z) + 0.1 * tf.math.sin(2*z)

class Onda(nn.Module):
    def forward(self, z):
        return torch.tanh(z) + 0.1 * torch.sin(2*z)

z = np.linspace(-6, 6, 501).astype('float32')
zt = tf.constant(z)
with tf.GradientTape() as tape:
    tape.watch(zt)
    f = onda_tf(zt)
df_tf = tape.gradient(f, zt).numpy()
zp = torch.tensor(z, requires_grad=True)
Onda()(zp).sum().backward()
analitica = 1 - np.tanh(z)**2 + .2*np.cos(2*z)
# float32 y kernels distintos pueden diferir unas pocas unidades de redondeo.
print('Error máximo TF / PyTorch:', np.max(np.abs(df_tf-analitica)),
      np.max(np.abs(zp.grad.numpy()-analitica)))
np.testing.assert_allclose(df_tf, analitica, atol=1e-6, rtol=1e-5)
np.testing.assert_allclose(zp.grad.numpy(), analitica, atol=1e-6, rtol=1e-5)
# Diferencias centrales en float64, lejos del error de redondeo de float32.
zd = z.astype('float64'); h = 1e-5
onda_np = lambda x: np.tanh(x) + .1*np.sin(2*x)
np.testing.assert_allclose((onda_np(zd+h)-onda_np(zd-h))/(2*h),
                           1-np.tanh(zd)**2+.2*np.cos(2*zd), atol=1e-8)
funciones = {'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh,
             'ReLU': tf.nn.relu, 'onda': onda_tf}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for nombre, funcion in funciones.items():
    with tf.GradientTape() as tape:
        tape.watch(zt); valores = funcion(zt)
    axes[0].plot(z, valores.numpy(), label=nombre)
    axes[1].plot(z, tape.gradient(valores, zt).numpy(), label=nombre)
for ax in axes:
    ax.set_xlabel('preactivación z'); ax.legend(); ax.grid(alpha=.2)
axes[0].set_ylabel('activación'); axes[1].set_ylabel('derivada')
plt.show()
'''), md('''
### Ablación controlada y diagnóstico de gradientes

Entrenamos cuatro redes con los mismos pesos iniciales, lotes, presupuesto y
optimizador; sólo cambia la activación. Elegimos por pérdida de **validación**.
La inicialización fija aísla ese cambio, aunque no sea óptima para todas las
activaciones. Una segunda investigación podría ajustar la inicialización.
Una semilla sirve para explorar; no prueba superioridad universal.

La norma del gradiente muestra si la señal de aprendizaje llega a cada matriz;
una derivada casi nula en muchos ejemplos sugiere saturación. No confundir
norma pequeña con error: también puede indicar cercanía a un mínimo.
'''), code('''
candidatos, filas_act = {}, []
for nombre, funcion in funciones.items():
    keras.utils.set_random_seed(SEMILLA)
    modelo = crear_keras(funcion)
    t0 = time.perf_counter()
    historia = modelo.fit(datos_tf(X_train, y_train, True), validation_data=val_tf,
                          epochs=EPOCAS, verbose=0,
                          callbacks=[keras.callbacks.EarlyStopping(
                              monitor='val_loss', patience=EPOCAS, restore_best_weights=True)])
    segundos = time.perf_counter() - t0
    xb = tf.constant(X_train[:BATCH]); yb = y_train[:BATCH]
    with tf.GradientTape() as tape:
        perdida = criterio_tf(yb, modelo(xb))
    gradientes = tape.gradient(perdida, modelo.trainable_variables)
    pre = xb @ modelo.layers[0].kernel + modelo.layers[0].bias
    with tf.GradientTape() as tape:
        tape.watch(pre); respuesta = funcion(pre)
    derivadas = tape.gradient(respuesta, pre)
    fila = {'activación': nombre, 'val_loss': min(historia.history['val_loss']),
            'segundos': segundos, 'fracción_derivada_casi_cero':
            float(tf.reduce_mean(tf.cast(tf.abs(derivadas) < .01, tf.float32))),
            'norma_grad_W1': float(tf.linalg.norm(gradientes[0])),
            'norma_grad_W2': float(tf.linalg.norm(gradientes[2]))}
    filas_act.append(fila); candidatos[nombre] = modelo
tabla_act = pd.DataFrame(filas_act).sort_values('val_loss')
display(tabla_act)
ganadora = tabla_act.iloc[0]['activación']
print('Elección cerrada antes de abrir test:', ganadora)
'''), md('''
## 7. Test final, curvas y errores

La comparación entre APIs es descriptiva, no una competición de velocidad:
el tiempo incluye costes distintos de compilación. No elegimos framework con
este test ni seguimos ajustando al verlo. La fila de activación corresponde a
una única elección hecha antes con validación.
'''), code('''
X_test = imagenes_test.reshape(-1, 784).astype('float32') / 255
predicciones = {
    'Logística SGD': lineal.predict(X_test),
    'TF variables': tf.argmax(forward_tf(X_test), axis=1).numpy(),
    'Keras ReLU': np.argmax(modelo_k.predict(X_test, batch_size=BATCH, verbose=0), axis=1),
    'Keras activación elegida': np.argmax(candidatos[ganadora].predict(
        X_test, batch_size=BATCH, verbose=0), axis=1),
}
modelo_pt.eval()
with torch.no_grad():
    predicciones['PyTorch'] = modelo_pt(torch.from_numpy(X_test).to(dispositivo)).argmax(1).cpu().numpy()
tiempos = {'Logística SGD': tiempo_lineal, 'TF variables': tiempo_tf,
           'Keras ReLU': tiempo_k, 'PyTorch': tiempo_pt,
           'Keras activación elegida': float(tabla_act.iloc[0]['segundos'])}
tabla = pd.DataFrame([{'modelo': nombre, 'accuracy_test': accuracy_score(y_test, pred),
                      'f1_macro_test': f1_score(y_test, pred, average='macro'),
                      'segundos_train': tiempos[nombre]}
                     for nombre, pred in predicciones.items()])
display(tabla)
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for ax, hist, nombre in zip(axes, [hist_tf, {'train': hist_k.history['loss'],
                                          'val': hist_k.history['val_loss']}, hist_pt],
                            ['TF variables', 'Keras', 'PyTorch']):
    ax.plot(hist['train'], label='train'); ax.plot(hist['val'], label='val')
    ax.set(title=nombre, xlabel='época (desde 0)', ylabel='entropía cruzada'); ax.legend()
plt.tight_layout(); plt.show()
pred = predicciones['Keras activación elegida']
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=nombres,
                                        xticks_rotation=90, normalize='true')
plt.show()
errores = np.flatnonzero(pred != y_test)[:8]
fig, axes = plt.subplots(1, 8, figsize=(15, 3))
for ax in axes: ax.axis('off')
for ax, i in zip(axes, errores):
    ax.imshow(imagenes_test[i], cmap='gray')
    ax.set_title(f'{nombres[y_test[i]]} →\\n{nombres[pred[i]]}', fontsize=8)
plt.show()
tabla.to_csv('resultados_feedforward.csv', index=False)
'''), md(r'''
## 8. Del píxel a una medición física

Cada píxel es una intensidad medida. Una red densa puede combinar todos los
canales, pero aplanar no incorpora localidad ni simetrías espaciales. Una alta
accuracy tampoco garantiza robustez frente a cambios de iluminación o de sensor.
Para predecir una temperatura crítica a partir de propiedades de materiales,
cambiamos 784 por el número de características, diez logits por una salida
lineal y entropía cruzada por MSE. MAE y RMSE se reportan en kelvin.

El notebook **36** prepara ese caso con 21 263 superconductores de UCI para
la alternativa A del taller. La descarga y la separación por material ya están
resueltas; la tarea consiste en comparar dos activaciones en una red de regresión.

**Comprobación:** ¿por qué no se aplica softmax antes de CrossEntropyLoss?
¿Qué ocurriría sin `zero_grad`? ¿Por qué la activación propia no debe llamar a
NumPy? ¿Qué diferencia hay entre medir la derivada de una activación y medir
el gradiente de la pérdida respecto a un peso? ¿Ganó el MLP a la línea base y
a qué coste? Una respuesta negativa también es un resultado.

**Errores frecuentes:** etiquetas flotantes en PyTorch; tensores en dispositivos
distintos; optimizador reutilizado entre modelos; evaluación con pesos de una
época que no fue la seleccionada; escoger activación mirando test.

**Miércoles:** [dos alternativas de taller](TALLER_FEEDFORWARD.md).
La A entrena dos redes durante dos horas; la B se realiza sin programar, con
celular y papel. No es necesario completar ambas ni usar los dos frameworks.

### Referencias

- [Autodiferenciación en TensorFlow](https://www.tensorflow.org/guide/autodiff).
- [Ciclo explícito en TensorFlow/Keras](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch).
- [PyTorch: fundamentos](https://docs.pytorch.org/tutorials/beginner/basics/intro.html).
- [Fashion-MNIST: datos y clases](https://keras.io/api/datasets/fashion_mnist/).
''')]
    write_notebook(path, cells, tier='tensorflow-pytorch-network')


UCI_LOAD = '''
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEMILLA = 42
URL = 'https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip'
cache = Path.home() / '.cache' / 'fc2' / 'superconductivity.zip'
cache.parent.mkdir(parents=True, exist_ok=True)
if not cache.exists():
    with urlopen(URL, timeout=90) as respuesta:
        contenido = respuesta.read()
    # Validar el ZIP antes de dejarlo en caché.
    with ZipFile(io.BytesIO(contenido)) as z:
        assert z.testzip() is None
    cache.write_bytes(contenido)
print('SHA256:', hashlib.sha256(cache.read_bytes()).hexdigest())
with ZipFile(cache) as z:
    datos = pd.read_csv(z.open('train.csv'))
    composicion = pd.read_csv(z.open('unique_m.csv'))
assert len(datos) == len(composicion) == 21263
np.testing.assert_allclose(datos.critical_temp, composicion.critical_temp)
grupos = composicion['material'].astype(str).to_numpy()
X = datos.drop(columns='critical_temp')
y = datos.critical_temp.to_numpy(dtype='float32')
assert X.shape[1] == 81 and np.isfinite(X.to_numpy()).all()
dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEMILLA).split(X, y, grupos))
tr_local, va_local = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEMILLA).split(
    X.iloc[dev], y[dev], grupos[dev]))
train, val = dev[tr_local], dev[va_local]
assert not set(grupos[train]) & set(grupos[val])
assert not set(grupos[dev]) & set(grupos[test])
print('Filas train/val/test:', len(train), len(val), len(test))
# EDA exclusivamente sobre train.
display(X.iloc[train].describe().T.head(10))
print('Ausentes train:', X.iloc[train].isna().sum().sum())
plt.hist(y[train], bins=40); plt.xlabel('Temperatura crítica [K]'); plt.ylabel('n'); plt.show()
sx = StandardScaler().fit(X.iloc[train])
sy = StandardScaler().fit(y[train, None])
xt, xv, xs = [sx.transform(X.iloc[idx]).astype('float32') for idx in (train, val, test)]
yt, yv = [sy.transform(y[idx, None]).astype('float32') for idx in (train, val)]
def kelvin(pred):
    return sy.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel()
def metricas(real, pred):
    return {'MAE_K': mean_absolute_error(real, pred),
            'RMSE_K': np.sqrt(mean_squared_error(real, pred)), 'R2': r2_score(real, pred)}
baselines = {'Ridge': Ridge(alpha=10),
             'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, random_state=SEMILLA)}
for nombre, modelo in baselines.items():
    modelo.fit(xt, y[train])
    print(nombre, 'validación:', metricas(y[val], modelo.predict(xv)))
'''


def build_workshop():
    path = '03_redes_fundamentos/36_taller_120_minutos.ipynb'
    # Reutilizar sólo la preparación UCI: el taller no exige EDA ni modelos clásicos.
    preparacion = UCI_LOAD.split("def metricas")[0]
    preparacion = preparacion.replace(
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        "from sklearn.metrics import mean_absolute_error")
    for linea in [
        'from sklearn.linear_model import Ridge\n',
        'from sklearn.ensemble import HistGradientBoostingRegressor\n',
        '# EDA exclusivamente sobre train.\n',
        'display(X.iloc[train].describe().T.head(10))\n',
        "print('Ausentes train:', X.iloc[train].isna().sum().sum())\n",
        "plt.hist(y[train], bins=40); plt.xlabel('Temperatura crítica [K]'); plt.ylabel('n'); plt.show()\n",
    ]:
        preparacion = preparacion.replace(linea, '')
    cells = [portada(path, 'Alternativa A: dos redes para predecir una temperatura',
                     '120 minutos', 'Colab con Internet; Keras/TensorFlow o PyTorch'), md('''
## Instrucciones y alcance

Siga la [alternativa A de la guía](TALLER_FEEDFORWARD.md). Un perceptrón multicapa
(**MLP**, *multilayer perceptron*) es la red densa feedforward que se entrena aquí.
La preparación ya está resuelta. Complete las celdas indicadas con ayuda del
notebook 30. No se pide análisis exploratorio ni modelos clásicos.

**Dos redes, misma arquitectura:** 81→32→16→1; ReLU frente a una función propia;
semilla 42; Adam 0.001; lote 256; 15 épocas sin parada anticipada.
Seleccione por error absoluto medio de validación en kelvin al finalizar la última
época. Use prueba sólo para la red elegida. Este notebook contiene espacios de
trabajo y no incluye la solución del entrenamiento.

**Sin computador:** la [alternativa B](TALLER_SIN_COMPUTADOR.md) se realiza
leyendo desde el celular y trabajando en papel. No utiliza este notebook.

## 1. Preparación suministrada — 0 a 15 minutos

Ejecute la siguiente celda sin modificarla. Descarga Superconductivty Data de
UCI y separa los materiales para que una misma fórmula no se repita entre conjuntos.
La escala se estima sólo en entrenamiento. La primera ejecución necesita conexión.

Fuente: [Hamidieh (2018), UCI 464](https://archive.ics.uci.edu/dataset/464/superconductivty+data),
21 263 registros, 81 características, licencia CC BY 4.0.
'''), code(preparacion), code('''
SEMILLA = 42
EPOCAS = 15
BATCH = 256
TASA = 0.001
print('Entrenamiento:', xt.shape, yt.shape)
print('Validación:', xv.shape, yv.shape)
print('Prueba reservada:', xs.shape)
assert xt.shape[1] == 81 and yt.shape == (len(xt), 1)
'''), md('''
### ¿Qué variable se usa en cada paso?

- `xt`, `yt`: entradas y objetivos para ajustar los pesos; están estandarizados.
- `xv`, `yv`: entradas y objetivos de validación para comparar; están estandarizados.
- `xs`: entradas reservadas para la evaluación final.
- `kelvin(pred)`: devuelve las salidas a kelvin; compare con `y[val]` o `y[test]`.

**Respuesta 1:** escriba las cantidades de filas que imprimió la celda, las 81
entradas que recibe cada ejemplo y por qué separar por material evita una
comparación engañosa. No tiene que analizar distribuciones ni limpiar la base.

## 2. Red con ReLU — 15 a 50 minutos

Complete `crear_red(activacion)` y el entrenamiento. Las capas ocultas tienen
32 y 16 neuronas; ambas aplican la activación recibida. La salida tiene una neurona
lineal. La pérdida es **MSE**, error cuadrático medio de objetivos estandarizados.
El optimizador Adam usa `TASA` como tasa de aprendizaje.

Puede adaptar las capas y el ciclo del 30: cambie a 81→32→16→1 y use `EPOCAS`
épocas fijas, sin parada anticipada ni selección de la mejor época.
Para pasar de clasificación a regresión, en Keras use `loss='mse'` y quite la
métrica `accuracy`; en PyTorch use `nn.MSELoss()` en lugar de `nn.CrossEntropyLoss()`.
Los objetivos son flotantes con forma `(N, 1)`. La salida es lineal, sin softmax
ni `argmax`: se conserva el número predicho y se convierte con `kelvin`.
Guarde pérdidas medias de entrenamiento y validación de cada época. En PyTorch,
pondere la pérdida de cada lote por su cantidad de ejemplos antes de promediar.
Compruebe 3 169 parámetros y formas `(256, 1)` para predicciones y objetivos.

**Antes de entrenar:** guarde una copia de los pesos iniciales, no una referencia
que siga cambiando. Keras: copie cada arreglo de `get_weights()`. PyTorch:
use `copy.deepcopy(red.state_dict())`. La segunda red debe cargar esa copia
con `set_weights` o `load_state_dict`, respectivamente.
'''), code('''
# Complete crear_red(activacion), la copia inicial y el primer entrenamiento.
# Guarde red_relu e historia_relu (pérdidas train/val por época).
# Registre el framework elegido y su versión.
'''), md('''
## 3. Función propia y segundo entrenamiento — 50 a 80 minutos

Defina una función que devuelva `z` cuando `z >= 0` y `0.1*z` cuando `z < 0`.
Use operaciones del framework (por ejemplo, su operación `where`), no una capa
predefinida de activación. Grafique esta función y ReLU entre −3 y 3, y escriba
sus pendientes a cada lado de cero. No se pide derivada en cero.

Cree una segunda red desde los pesos iniciales guardados y un optimizador **nuevo**.
Mantenga arquitectura, datos y presupuesto. Para repetir el orden de los lotes,
cree otra vez el dataset/loader con la misma semilla: no continúe usando un
barajador cuyo estado ya avanzó durante el primer entrenamiento.
Puede adaptar `datos_tf` del 30 o un nuevo `torch.Generator().manual_seed(42)`.
Entrene y conserve red_propia e historia_propia.
'''), code('''
# Complete activación, gráfica, pendientes y segundo entrenamiento.
'''), md('''
## 4. Elegir con validación; después abrir prueba — 80 a 105 minutos

Con los pesos de la última época, prediga para `xv` con ambas redes. Convierta
las predicciones usando `kelvin` y calcule **MAE**, error absoluto medio, frente
a `y[val]`. Puede usar `mean_absolute_error`, ya importada.

Llene la tabla de dos filas y registre la elección por menor MAE de validación.
Si empatan al redondear a dos decimales, elija ReLU. No use `xs` en este paso.
'''), code('''
# Tabla: activación | MAE_validación_K.
# Escriba qué red elige y por qué, antes de ejecutar la siguiente sección.
'''), md('''
### Evaluación final

Prediga con la red elegida sobre `xs` y convierta a kelvin. Calcule un único
MAE de prueba comparando con `y[test]`. Grafique las pérdidas train/val de la red
elegida por época. Muestre las primeras cinco temperaturas reales, predicciones
y errores absolutos de prueba en K. No modifique los pesos ni la elección después.
'''), code('''
# Complete MAE final, curvas y tabla de cinco predicciones.
'''), md('''
## 5. Respuestas y entrega — 105 a 120 minutos

Responda las tres preguntas de la guía: qué activación eligió y con qué evidencia,
qué muestran sus curvas y qué significa el MAE físico con sus limitaciones.
Entregue este notebook con código y resultados. No hace falta un CSV ni otro informe.

Si el equipo exige menos tiempo, fije ocho épocas **para ambas redes antes de
entrenar** y documente el cambio. No se exige alcanzar un valor mínimo de error.
''')]
    write_notebook(path, cells, tier='tensorflow-network')


if __name__ == '__main__':
    build_integrated()
    build_workshop()
