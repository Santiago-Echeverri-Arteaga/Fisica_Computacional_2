"""Construye los seis notebooks de fundamentos de redes neuronales."""

from __future__ import annotations

from build_classical_notebooks import header
from build_initial_notebooks import code, md, write_notebook


def build_neuron_feedforward() -> None:
    path = "03_redes_fundamentos/30_neurona_y_feedforward_desde_cero.ipynb"
    cells = [
        header(path, "Neurona y propagación hacia adelante desde cero", "¿Qué calcula realmente una red densa?", 4),
        md(
            r"""
            ## De una neurona a una red

            Una capa aplica una transformación afín y una no linealidad:

            $$z^{(\ell)}=a^{(\ell-1)}W^{(\ell)}+b^{(\ell)},\qquad
            a^{(\ell)}=\phi\!\left(z^{(\ell)}\right).$$

            Para un lote de $N$ observaciones, $a^{(\ell-1)}$ es una matriz
            $N\times n_{\ell-1}$ y $W^{(\ell)}$ tiene forma
            $n_{\ell-1}\times n_\ell$. La no linealidad es indispensable: sin ella,
            componer capas afines sigue produciendo una sola transformación afín.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            SEMILLA = 42
            rng = np.random.default_rng(SEMILLA)

            def relu(z):
                return np.maximum(0.0, z)

            def tanh(z):
                return np.tanh(z)

            class CapaDensa:
                def __init__(self, entradas, salidas, rng, activación=lambda z: z):
                    # Inicialización de Xavier: varianza ~ 1/entradas.
                    self.W = rng.normal(0, np.sqrt(1 / entradas), size=(entradas, salidas))
                    self.b = np.zeros((1, salidas))
                    self.activación = activación

                def __call__(self, a):
                    self.z = a @ self.W + self.b
                    self.a = self.activación(self.z)
                    return self.a

            X = rng.normal(size=(5, 3))
            capa_1 = CapaDensa(3, 4, rng, relu)
            capa_2 = CapaDensa(4, 1, rng)
            salida = capa_2(capa_1(X))

            print("X:", X.shape)
            print("W1:", capa_1.W.shape, "a1:", capa_1.a.shape)
            print("W2:", capa_2.W.shape, "salida:", salida.shape)
            """
        ),
        md(
            r"""
            ## Conteo de parámetros

            Una capa con $n_{in}$ entradas y $n_{out}$ salidas contiene
            $n_{in}n_{out}+n_{out}$ parámetros. Para 3→4→1 hay
            $(3\times4+4)+(4\times1+1)=21$. Verificar formas y conteos antes de
            entrenar previene muchos errores silenciosos.
            """
        ),
        code(
            """
            def contar_parámetros(*capas):
                return sum(capa.W.size + capa.b.size for capa in capas)

            print("Parámetros:", contar_parámetros(capa_1, capa_2))
            assert contar_parámetros(capa_1, capa_2) == 21
            """
        ),
        md(
            r"""
            ## Aproximación con características neuronales

            Para visualizar potencia expresiva sin mezclar todavía backpropagation,
            fijamos neuronas ocultas aleatorias $\tanh(w_jx+b_j)$ y calculamos por
            mínimos cuadrados sólo los pesos de salida. Aumentar neuronas amplía el
            espacio de funciones disponible, pero no garantiza generalización.
            """
        ),
        code(
            """
            x = np.linspace(-3, 3, 220)[:, None]
            y = np.sin(2.2 * x[:, 0]) * np.exp(-0.12 * x[:, 0] ** 2)
            índice = rng.permutation(len(x))
            train, test = índice[:150], índice[150:]

            fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=True)
            for ancho, ax in zip([3, 15, 80], axes):
                local_rng = np.random.default_rng(SEMILLA)
                W = local_rng.normal(size=(1, ancho))
                b = local_rng.uniform(-2, 2, size=(1, ancho))
                H_train = np.c_[np.ones(len(train)), np.tanh(x[train] @ W + b)]
                beta = np.linalg.lstsq(H_train, y[train], rcond=None)[0]
                H = np.c_[np.ones(len(x)), np.tanh(x @ W + b)]
                pred = H @ beta
                rmse = np.sqrt(np.mean((y[test] - pred[test]) ** 2))
                ax.plot(x, y, "k--", label="función")
                ax.plot(x, pred, label="red")
                ax.scatter(x[test], y[test], s=8, alpha=0.3)
                ax.set_title(f"{ancho} neuronas | RMSE={rmse:.3f}")
                ax.legend()
            plt.show()
            """
        ),
        md(
            """
            **Ejercicios:** demuestre algebraicamente por qué dos capas lineales se
            reducen a una; cambie `tanh` por ReLU; calcule parámetros de una red
            80→128→64→3; investigue qué ocurre al extrapolar fuera de $[-3,3]$.
            """
        ),
    ]
    write_notebook(path, cells)


def build_activations() -> None:
    path = "03_redes_fundamentos/31_activaciones_diseno_y_comparacion.ipynb"
    cells = [
        header(path, "Funciones de activación: diseño y comparación", "¿Podemos proponer una activación y evaluarla científicamente?", 4),
        md(
            r"""
            ## Qué debe aportar una activación

            Una activación introduce no linealidad y controla el flujo de gradientes.
            Analizaremos rango, derivada, saturación, suavidad, costo y media de las
            salidas. No existe una función universalmente mejor: la comparación debe
            mantener arquitectura, datos, inicialización y presupuesto constantes.

            Diseñaremos
            $$\phi_{onda}(x)=\tanh(x)+0.1\sin(2x),\qquad
            \phi'_{onda}(x)=1-\tanh^2(x)+0.2\cos(2x).$$
            Es una hipótesis del estudiante, no una mejora asegurada.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import make_moons
            from sklearn.metrics import accuracy_score, log_loss
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            SEMILLA = 42

            def sigmoid(x):
                # Forma estable frente a overflow.
                positiva = x >= 0
                salida = np.empty_like(x, dtype=float)
                salida[positiva] = 1 / (1 + np.exp(-x[positiva]))
                exp_x = np.exp(x[~positiva])
                salida[~positiva] = exp_x / (1 + exp_x)
                return salida

            def relu(x): return np.maximum(0.0, x)
            def d_relu(x): return (x > 0).astype(float)
            def tanh(x): return np.tanh(x)
            def d_tanh(x): return 1 - np.tanh(x) ** 2
            def swish(x): return x * sigmoid(x)
            def d_swish(x):
                s = sigmoid(x)
                return s + x * s * (1 - s)
            def onda(x): return np.tanh(x) + 0.1 * np.sin(2 * x)
            def d_onda(x): return 1 - np.tanh(x) ** 2 + 0.2 * np.cos(2 * x)

            activaciones = {
                "ReLU": (relu, d_relu),
                "tanh": (tanh, d_tanh),
                "swish": (swish, d_swish),
                "onda_propia": (onda, d_onda),
            }
            """
        ),
        code(
            r"""
            z = np.linspace(-6, 6, 600)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for nombre, (f, df) in activaciones.items():
                axes[0].plot(z, f(z), label=nombre)
                axes[1].plot(z, df(z), label=nombre)
            axes[0].set(title="activaciones", xlabel="z", ylabel="$\\phi(z)$")
            axes[1].set(title="derivadas", xlabel="z", ylabel="$\\phi'(z)$")
            for ax in axes: ax.legend()
            plt.show()
            """
        ),
        md(
            """
            ## Verificación numérica de la derivada

            Antes de entrenar una activación propia comprobamos su derivada con
            diferencias centrales. El error debe decrecer hasta que el redondeo de
            punto flotante domina.
            """
        ),
        code(
            """
            puntos = np.linspace(-3, 3, 101)
            h = 1e-5
            derivada_numérica = (onda(puntos + h) - onda(puntos - h)) / (2 * h)
            error = np.max(np.abs(derivada_numérica - d_onda(puntos)))
            print(f"Error máximo de la derivada propia: {error:.2e}")
            assert error < 1e-7
            """
        ),
        md(
            r"""
            ## Ablación controlada en una red NumPy

            Entrenamos una red 2→16→1 sobre las mismas lunas. Reiniciamos exactamente
            los pesos para cada activación; sólo cambia $\phi$. La salida usa sigmoid
            y entropía cruzada binaria.
            """
        ),
        code(
            """
            X, y = make_moons(n_samples=900, noise=0.22, random_state=SEMILLA)
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y[:, None], test_size=0.25, stratify=y, random_state=SEMILLA
            )
            escalador = StandardScaler().fit(X_dev)
            X_dev, X_test = escalador.transform(X_dev), escalador.transform(X_test)

            def entrenar(activación, derivada, épocas=900, lr=0.05):
                rng = np.random.default_rng(SEMILLA)
                W1 = rng.normal(0, np.sqrt(1 / 2), (2, 16)); b1 = np.zeros((1, 16))
                W2 = rng.normal(0, np.sqrt(1 / 16), (16, 1)); b2 = np.zeros((1, 1))
                historial = []
                n = len(X_dev)
                for época in range(épocas):
                    z1 = X_dev @ W1 + b1
                    a1 = activación(z1)
                    prob = sigmoid(a1 @ W2 + b2)
                    dz2 = (prob - y_dev) / n
                    dW2 = a1.T @ dz2; db2 = dz2.sum(axis=0, keepdims=True)
                    dz1 = (dz2 @ W2.T) * derivada(z1)
                    dW1 = X_dev.T @ dz1; db1 = dz1.sum(axis=0, keepdims=True)
                    W1 -= lr * dW1; b1 -= lr * db1
                    W2 -= lr * dW2; b2 -= lr * db2
                    if época % 25 == 0:
                        eps = 1e-9
                        pérdida = -np.mean(y_dev*np.log(prob+eps)+(1-y_dev)*np.log(1-prob+eps))
                        historial.append((época, pérdida))
                prob_test = sigmoid(activación(X_test @ W1 + b1) @ W2 + b2)
                return np.asarray(historial), accuracy_score(y_test, prob_test >= 0.5), log_loss(y_test, prob_test)

            filas = []
            plt.figure(figsize=(8, 4))
            for nombre, (f, df) in activaciones.items():
                hist, acc, pérdida = entrenar(f, df)
                filas.append({"activación": nombre, "accuracy_test": acc, "log_loss_test": pérdida})
                plt.plot(hist[:, 0], hist[:, 1], label=nombre)
            plt.yscale("log")
            plt.xlabel("época"); plt.ylabel("entropía cruzada train"); plt.legend(); plt.show()
            display(pd.DataFrame(filas).set_index("activación"))
            """
        ),
        md(
            r"""
            Una sola corrida no basta para declarar ganadora. Repita semillas y
            reporte media, dispersión, velocidad y estabilidad. Examine también la
            distribución de preactivaciones y gradientes por capa.

            **Proyecto de activación:** proponga $\phi$ y su derivada; verifíquela
            numéricamente; justifique inicialización; compare al menos cinco semillas;
            incluya un caso donde falle. Una función más complicada sólo merece
            conservarse si ofrece evidencia reproducible.
            """
        ),
    ]
    write_notebook(path, cells)


def build_gradient_backprop() -> None:
    path = "03_redes_fundamentos/32_gradiente_y_backpropagation.ipynb"
    cells = [
        header(path, "Descenso de gradiente y backpropagation", "¿Cómo atribuye la regla de la cadena cada error a cada parámetro?", 4),
        md(
            r"""
            ## Gradiente

            Para parámetros $\theta$, descenso de gradiente actualiza
            $\theta_{t+1}=\theta_t-\eta\nabla_\theta L$. Backpropagation no es un
            optimizador: es una forma eficiente de calcular productos de derivadas
            desde la salida hacia las entradas mediante la regla de la cadena.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            # Grafo escalar: L=(wx+b-y)^2
            x, y, w, b = 2.0, 5.0, 0.7, -0.2
            pred = w * x + b
            pérdida = (pred - y) ** 2
            dL_dpred = 2 * (pred - y)
            dL_dw = dL_dpred * x
            dL_db = dL_dpred
            print({"pred": pred, "L": pérdida, "dL/dw": dL_dw, "dL/db": dL_db})

            def L(w_, b_): return (w_ * x + b_ - y) ** 2
            h = 1e-6
            num_w = (L(w + h, b) - L(w - h, b)) / (2 * h)
            num_b = (L(w, b + h) - L(w, b - h)) / (2 * h)
            print("error gradiente:", abs(num_w-dL_dw), abs(num_b-dL_db))
            """
        ),
        md(
            r"""
            ## Backpropagation matricial

            Para una red de regresión 1→20→1 con `tanh` y MSE:
            $\delta^{(2)}=2(\hat y-y)/N$,
            $\nabla W^{(2)}=(a^{(1)})^T\delta^{(2)}$ y
            $\delta^{(1)}=(\delta^{(2)}(W^{(2)})^T)\odot(1-\tanh^2 z^{(1)})$.
            Guardar $z$ y $a$ durante forward evita recalcularlos.
            """
        ),
        code(
            """
            SEMILLA = 42
            rng = np.random.default_rng(SEMILLA)
            X = np.linspace(-2.5, 2.5, 240)[:, None]
            y = np.sin(2 * X) + 0.15 * X
            W1 = rng.normal(0, np.sqrt(1), (1, 20)); b1 = np.zeros((1, 20))
            W2 = rng.normal(0, np.sqrt(1/20), (20, 1)); b2 = np.zeros((1, 1))

            def forward(X, W1, b1, W2, b2):
                z1 = X @ W1 + b1
                a1 = np.tanh(z1)
                pred = a1 @ W2 + b2
                return z1, a1, pred

            def gradientes(X, y, W1, b1, W2, b2):
                z1, a1, pred = forward(X, W1, b1, W2, b2)
                n = len(X)
                d_pred = 2 * (pred - y) / n
                dW2 = a1.T @ d_pred
                db2 = d_pred.sum(axis=0, keepdims=True)
                dz1 = (d_pred @ W2.T) * (1 - np.tanh(z1) ** 2)
                dW1 = X.T @ dz1
                db1 = dz1.sum(axis=0, keepdims=True)
                return (dW1, db1, dW2, db2), np.mean((pred-y)**2)
            """
        ),
        code(
            """
            # Gradient check de cinco entradas aleatorias de W1.
            analíticos, _ = gradientes(X, y, W1, b1, W2, b2)
            dW1 = analíticos[0]
            h = 1e-5
            for j in [0, 3, 7, 12, 19]:
                original = W1[0, j]
                W1[0, j] = original + h
                L_plus = np.mean((forward(X, W1, b1, W2, b2)[2] - y) ** 2)
                W1[0, j] = original - h
                L_minus = np.mean((forward(X, W1, b1, W2, b2)[2] - y) ** 2)
                W1[0, j] = original
                numérico = (L_plus - L_minus) / (2*h)
                print(j, "analítico=", dW1[0,j], "numérico=", numérico)
            """
        ),
        code(
            """
            historial = []
            lr = 0.03
            for época in range(2_000):
                (dW1, db1, dW2, db2), mse = gradientes(X, y, W1, b1, W2, b2)
                W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
                if época % 50 == 0: historial.append(mse)

            pred = forward(X, W1, b1, W2, b2)[2]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(np.arange(len(historial))*50, historial)
            axes[0].set(yscale="log", xlabel="época", ylabel="MSE")
            axes[1].plot(X, y, "k--", label="objetivo")
            axes[1].plot(X, pred, label="red")
            axes[1].legend()
            plt.show()
            """
        ),
        md(
            """
            **Ejercicios:** derive gradientes de $b_1$; pruebe tasas 0.001, 0.03 y
            0.5; grafique normas de gradiente; sustituya `tanh` por su activación del
            notebook anterior; explique la diferencia entre derivación automática y
            diferenciación numérica.
            """
        ),
    ]
    write_notebook(path, cells)


def build_losses_optimizers_regularization() -> None:
    path = "03_redes_fundamentos/33_perdidas_optimizadores_regularizacion.ipynb"
    cells = [
        header(path, "Pérdidas, optimizadores y regularización", "¿Qué objetivo optimizamos y qué sesgo introducimos?", 4),
        md(
            r"""
            ## La pérdida expresa una hipótesis de ruido

            MSE corresponde a una observación gaussiana con varianza fija; MAE se
            relaciona con Laplace y es menos sensible a valores extremos; Huber es
            cuadrática cerca de cero y lineal lejos. Para clasificación, la
            entropía cruzada es la log-verosimilitud negativa.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            error = np.linspace(-4, 4, 500)
            delta = 1.0
            pérdidas = {
                "MSE/2": 0.5*error**2,
                "MAE": np.abs(error),
                "Huber": np.where(np.abs(error)<=delta, 0.5*error**2, delta*(np.abs(error)-0.5*delta)),
            }
            for nombre, valores in pérdidas.items(): plt.plot(error, valores, label=nombre)
            plt.xlabel("residuo"); plt.ylabel("pérdida"); plt.legend(); plt.show()

            def bce_desde_logits(y, logits):
                # softplus(logit) - y*logit: estable incluso para logits grandes.
                return np.maximum(logits, 0) - y*logits + np.log1p(np.exp(-np.abs(logits)))

            print(bce_desde_logits(np.array([0, 1]), np.array([-1000.0, 1000.0])))
            """
        ),
        md(
            r"""
            ## Optimizadores desde cero

            Momentum acumula velocidad; RMSProp normaliza por una media de cuadrados;
            Adam combina ambos y corrige el sesgo inicial. Los compararemos en la
            función de Rosenbrock
            $f(x,y)=(1-x)^2+100(y-x^2)^2$, cuyo valle curvo expone diferencias.
            """
        ),
        code(
            """
            def rosenbrock(theta):
                x, y = theta
                return (1-x)**2 + 100*(y-x**2)**2

            def grad_rosenbrock(theta):
                x, y = theta
                return np.array([-2*(1-x)-400*x*(y-x**2), 200*(y-x**2)])

            def optimizar(método, pasos=5_000):
                theta = np.array([-1.4, 1.5], dtype=float)
                m = np.zeros(2); v = np.zeros(2); trayectoria = [theta.copy()]
                for t in range(1, pasos+1):
                    g = np.clip(grad_rosenbrock(theta), -1e3, 1e3)
                    if método == "SGD":
                        theta -= 0.001*g
                    elif método == "Momentum":
                        m = 0.9*m + g
                        theta -= 0.0002*m
                    elif método == "RMSProp":
                        v = 0.99*v + 0.01*g*g
                        theta -= 0.002*g/(np.sqrt(v)+1e-8)
                    else:  # Adam
                        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*g*g
                        mh = m/(1-0.9**t); vh = v/(1-0.999**t)
                        theta -= 0.003*mh/(np.sqrt(vh)+1e-8)
                    if t % 20 == 0: trayectoria.append(theta.copy())
                return np.asarray(trayectoria), rosenbrock(theta)

            filas = []
            plt.figure(figsize=(7, 5))
            for método in ["SGD", "Momentum", "RMSProp", "Adam"]:
                ruta, final = optimizar(método)
                filas.append({"optimizador": método, "pérdida_final": final, "x": ruta[-1,0], "y": ruta[-1,1]})
                plt.plot(ruta[:,0], ruta[:,1], label=método, alpha=0.8)
            plt.scatter([1], [1], marker="*", s=180, c="gold", edgecolor="k", label="mínimo")
            plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
            display(pd.DataFrame(filas).set_index("optimizador"))
            """
        ),
        md(
            r"""
            ## Regularización

            - L2 añade $\lambda\|W\|_2^2$ y favorece pesos pequeños.
            - L1 añade $\lambda\|W\|_1$ y favorece esparsidad.
            - Dropout multiplica activaciones por una máscara Bernoulli durante
              entrenamiento y usa escalado invertido $m/(1-p)$.
            - Early stopping limita implícitamente cuánto se adapta el modelo.
            """
        ),
        code(
            """
            rng = np.random.default_rng(42)
            activaciones = rng.normal(size=(10_000, 20))
            for p in [0.0, 0.2, 0.5, 0.8]:
                máscara = rng.binomial(1, 1-p, size=activaciones.shape)
                salida = activaciones if p == 0 else activaciones*máscara/(1-p)
                print(f"p={p:.1f} | media={salida.mean():+.3f} | std={salida.std():.3f}")
            """
        ),
        md(
            """
            **Ejercicios:** derive gradiente de L2; muestre por qué el escalado de
            dropout conserva la esperanza; compare optimizadores con presupuesto
            idéntico; diseñe una pérdida asimétrica para un sensor donde subestimar
            sea más costoso que sobreestimar.
            """
        ),
    ]
    write_notebook(path, cells)


def build_tensorflow_mlp() -> None:
    path = "03_redes_fundamentos/34_mlp_tensorflow.ipynb"
    cells = [
        header(path, "Red feedforward con TensorFlow/Keras", "¿Cómo se traduce la derivación a una API entrenable?", 4),
        md(
            """
            **Requiere:** runtime estándar de Colab con TensorFlow. Usaremos un
            problema de clasificación no lineal, una partición fija de test y
            validación para escoger anchura y tasa. `EarlyStopping` restaura los
            pesos con menor pérdida de validación.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from sklearn.datasets import make_moons
            from sklearn.metrics import accuracy_score, log_loss
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            SEMILLA = 42
            tf.keras.utils.set_random_seed(SEMILLA)
            X, y = make_moons(n_samples=1_600, noise=0.24, random_state=SEMILLA)
            X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
            X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=SEMILLA)
            scaler = StandardScaler().fit(X_train)
            X_train, X_val, X_test = map(scaler.transform, [X_train, X_val, X_test])
            print("TensorFlow", tf.__version__, X_train.shape, X_val.shape, X_test.shape)
            """
        ),
        code(
            """
            def crear_mlp(unidades=32, tasa=1e-3):
                modelo = tf.keras.Sequential(
                    [
                        tf.keras.layers.Input(shape=(2,)),
                        tf.keras.layers.Dense(unidades, activation="relu"),
                        tf.keras.layers.Dense(unidades, activation="relu"),
                        tf.keras.layers.Dense(1),  # logits, no probabilidades
                    ]
                )
                modelo.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=tasa),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],
                )
                return modelo

            modelo_demo = crear_mlp()
            modelo_demo.summary()
            """
        ),
        code(
            """
            configuraciones = [(16, 1e-3), (32, 1e-3), (64, 3e-4)]
            filas, candidatos = [], []
            for unidades, tasa in configuraciones:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEMILLA)
                modelo = crear_mlp(unidades, tasa)
                parada = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=12, restore_best_weights=True
                )
                historia = modelo.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=150, batch_size=64, verbose=0, callbacks=[parada],
                )
                mejor_val = min(historia.history["val_loss"])
                filas.append({"unidades": unidades, "tasa": tasa, "val_loss": mejor_val, "épocas": len(historia.history["loss"])})
                candidatos.append((mejor_val, modelo, historia))
            display(pd.DataFrame(filas).sort_values("val_loss"))
            """
        ),
        code(
            """
            _, mejor_modelo, historia = min(candidatos, key=lambda item: item[0])
            logits = mejor_modelo.predict(X_test, verbose=0).ravel()
            prob = tf.sigmoid(logits).numpy()
            print("accuracy test:", accuracy_score(y_test, prob >= 0.5))
            print("log loss test:", log_loss(y_test, prob))

            plt.plot(historia.history["loss"], label="train")
            plt.plot(historia.history["val_loss"], label="validation")
            plt.xlabel("época"); plt.ylabel("BCE"); plt.legend(); plt.show()
            """
        ),
        md(
            """
            ## Un paso explícito con `GradientTape`

            `fit` organiza el ciclo, pero la diferenciación puede verse directamente.
            No ejecute este paso sobre el modelo final: es una demostración separada.
            """
        ),
        code(
            """
            demo = crear_mlp(16, 1e-3)
            optimizador = demo.optimizer
            x_batch = tf.convert_to_tensor(X_train[:64], dtype=tf.float32)
            y_batch = tf.cast(y_train[:64, None], tf.float32)
            with tf.GradientTape() as tape:
                logits = demo(x_batch, training=True)
                pérdida = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=logits))
            grads = tape.gradient(pérdida, demo.trainable_variables)
            optimizador.apply_gradients(zip(grads, demo.trainable_variables))
            print("pérdida:", float(pérdida), "normas:", [float(tf.norm(g)) for g in grads])
            """
        ),
        md(
            """
            **Ejercicios:** reproduzca con `tanh`; añada L2 y dropout por separado;
            compare parámetros y tiempo; construya una activación propia de forma
            vectorizada y verifique que TensorFlow pueda derivarla automáticamente.
            """
        ),
    ]
    write_notebook(path, cells, tier="tensorflow")


def build_pytorch_mlp() -> None:
    path = "03_redes_fundamentos/35_mlp_pytorch.ipynb"
    cells = [
        header(path, "Red feedforward con PyTorch", "¿Cómo controlamos explícitamente el ciclo de entrenamiento?", 4),
        md(
            """
            **Requiere:** runtime estándar de Colab con PyTorch. Usaremos el mismo
            tipo de problema que en TensorFlow, pero escribiremos `Dataset`, lotes,
            forward, backward, actualización, validación y early stopping.
            """
        ),
        code(
            """
            import copy
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            from sklearn.datasets import make_moons
            from sklearn.metrics import accuracy_score, log_loss
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            SEMILLA = 42
            torch.manual_seed(SEMILLA)
            dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X, y = make_moons(n_samples=1_600, noise=0.24, random_state=SEMILLA)
            X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
            X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=SEMILLA)
            scaler = StandardScaler().fit(X_train)
            X_train, X_val, X_test = map(scaler.transform, [X_train, X_val, X_test])

            def tensor_dataset(X, y):
                return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y[:,None], dtype=torch.float32))

            generador = torch.Generator().manual_seed(SEMILLA)
            train_loader = DataLoader(tensor_dataset(X_train, y_train), batch_size=64, shuffle=True, generator=generador)
            val_loader = DataLoader(tensor_dataset(X_val, y_val), batch_size=256)
            print("PyTorch", torch.__version__, "dispositivo", dispositivo)
            """
        ),
        code(
            """
            class MLP(nn.Module):
                def __init__(self, unidades=32):
                    super().__init__()
                    self.red = nn.Sequential(
                        nn.Linear(2, unidades), nn.ReLU(),
                        nn.Linear(unidades, unidades), nn.ReLU(),
                        nn.Linear(unidades, 1),
                    )
                def forward(self, x):
                    return self.red(x)

            modelo = MLP().to(dispositivo)
            print(modelo)
            print("parámetros:", sum(p.numel() for p in modelo.parameters() if p.requires_grad))
            """
        ),
        code(
            """
            def pérdida_media(modelo, loader, criterio):
                modelo.eval()
                total, n = 0.0, 0
                with torch.no_grad():
                    for xb, yb in loader:
                        xb, yb = xb.to(dispositivo), yb.to(dispositivo)
                        total += criterio(modelo(xb), yb).item() * len(xb)
                        n += len(xb)
                return total / n

            criterio = nn.BCEWithLogitsLoss()
            optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
            mejor_estado, mejor_val, paciencia = None, np.inf, 0
            historia = {"train": [], "val": []}

            for época in range(150):
                modelo.train()
                acumulada = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(dispositivo), yb.to(dispositivo)
                    optimizador.zero_grad()
                    logits = modelo(xb)
                    pérdida = criterio(logits, yb)
                    pérdida.backward()
                    optimizador.step()
                    acumulada += pérdida.item() * len(xb)
                train_loss = acumulada / len(train_loader.dataset)
                val_loss = pérdida_media(modelo, val_loader, criterio)
                historia["train"].append(train_loss); historia["val"].append(val_loss)
                if val_loss < mejor_val - 1e-5:
                    mejor_val = val_loss; mejor_estado = copy.deepcopy(modelo.state_dict()); paciencia = 0
                else:
                    paciencia += 1
                if paciencia >= 15: break

            modelo.load_state_dict(mejor_estado)
            print("épocas:", len(historia["train"]), "mejor val:", mejor_val)
            """
        ),
        code(
            """
            modelo.eval()
            with torch.no_grad():
                logits = modelo(torch.tensor(X_test, dtype=torch.float32, device=dispositivo))
                prob = torch.sigmoid(logits).cpu().numpy().ravel()
            print("accuracy test:", accuracy_score(y_test, prob >= 0.5))
            print("log loss test:", log_loss(y_test, prob))
            plt.plot(historia["train"], label="train")
            plt.plot(historia["val"], label="validation")
            plt.xlabel("época"); plt.ylabel("BCE"); plt.legend(); plt.show()
            """
        ),
        md(
            """
            **Lectura del ciclo:** `zero_grad` evita acumular gradientes; `backward`
            aplica diferenciación automática; `step` actualiza; `eval` cambia el
            comportamiento de dropout/batch normalization; `no_grad` evita construir
            un grafo en inferencia.

            **Ejercicios:** omita deliberadamente `zero_grad`; añada dropout; registre
            normas de gradiente; compare con TensorFlow usando misma arquitectura,
            partición, épocas máximas y criterio de selección.
            """
        ),
    ]
    write_notebook(path, cells, tier="pytorch")


if __name__ == "__main__":
    build_neuron_feedforward()
    build_activations()
    build_gradient_backprop()
    build_losses_optimizers_regularization()
    build_tensorflow_mlp()
    build_pytorch_mlp()
    print("Notebooks de fundamentos neuronales generados.")
