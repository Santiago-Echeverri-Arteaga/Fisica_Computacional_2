"""Construye la sesión integrada y los dos notebooks de profundización neural."""

from __future__ import annotations

from build_classical_notebooks import header
from build_initial_notebooks import code, md, write_notebook


def build_neuron_feedforward() -> None:
    from build_feedforward_practice import build_integrated
    build_integrated()


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


if __name__ == "__main__":
    build_neuron_feedforward()
    build_gradient_backprop()
    build_losses_optimizers_regularization()
    print("Notebooks de fundamentos neuronales generados.")
