"""Construye los notebooks iniciales de la edición 2026-2.

Este archivo hace revisable en Git la estructura de los notebooks y permite
regenerarlos sin conservar salidas de ejecución. No es material obligatorio
para los estudiantes.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "curso_2026_2"
COLAB_ROOT = (
    "https://colab.research.google.com/github/"
    "Santiago-Echeverri-Arteaga/Fisica_Computacional_2/blob/master/curso_2026_2"
)


def _source(value: str) -> list[str]:
    text = textwrap.dedent(value).strip() + "\n"
    return text.splitlines(keepends=True)


def md(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


def notebook(cells: list[dict], tier: str = "base", accelerator: str = "cpu") -> dict:
    for number, cell in enumerate(cells, start=1):
        cell["id"] = f"fc2-{number:03d}"
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "fc2": {"tier": tier, "accelerator": accelerator},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(
    relative_path: str,
    cells: list[dict],
    tier: str = "base",
    accelerator: str = "cpu",
) -> None:
    destination = COURSE / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook(cells, tier=tier, accelerator=accelerator), ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )


def build_evaluation_notebook() -> None:
    badge = f"{COLAB_ROOT}/00_fundamentos/00_protocolo_evaluacion.ipynb"
    cells = [
        md(
            f"""
            <a href="{badge}" target="_parent">
              <img src="https://colab.research.google.com/assets/colab-badge.svg"
                   alt="Abrir en Colab"/>
            </a>

            # Protocolo de evaluación sin fuga de información

            **Duración sugerida:** 2 horas<br>
            **Idea central:** una puntuación sólo es creíble si los datos usados para
            tomar decisiones no reaparecen disfrazados en la evaluación final.

            Al terminar podrá:

            1. distinguir entrenamiento, validación y prueba;
            2. explicar por qué el escalado debe aprenderse dentro de un `Pipeline`;
            3. usar validación cruzada y `GridSearchCV` sin tocar el test;
            4. comparar modelos con métricas acordes al problema.
            """
        ),
        md(
            r"""
            ## 1. El contrato experimental

            Separamos primero un conjunto de **prueba** que permanecerá cerrado.
            El resto es el conjunto de **desarrollo**. Dentro de desarrollo podemos
            usar una validación fija o, preferiblemente cuando hay pocos datos,
            validación cruzada.

            $$
            \text{datos}\longrightarrow
            \begin{cases}
            \text{desarrollo: entrenar y seleccionar}\\
            \text{prueba: estimar una sola vez el desempeño final}
            \end{cases}
            $$

            En clasificación estratificamos para conservar aproximadamente la
            proporción de clases. La semilla hace repetible la partición; no elimina
            la incertidumbre estadística.
            """
        ),
        code(
            """
            import platform

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import sklearn
            from sklearn.datasets import load_breast_cancer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                balanced_accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.model_selection import (
                GridSearchCV,
                StratifiedKFold,
                cross_validate,
                train_test_split,
            )
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)

            print(f"Python: {platform.python_version()}")
            print(f"NumPy: {np.__version__}")
            print(f"scikit-learn: {sklearn.__version__}")
            """
        ),
        md(
            """
            ## 2. Datos y pregunta

            Usaremos el conjunto pequeño de diagnóstico de cáncer de mama incluido
            en `scikit-learn`. No es una aplicación física: su función aquí es
            aislar el protocolo de evaluación antes de introducir modelos nuevos.

            La variable objetivo vale 0 para maligno y 1 para benigno. En una
            aplicación real deberíamos discutir procedencia, población, sesgos y
            consecuencias de cada tipo de error; aquí nos concentramos en la
            mecánica experimental.
            """
        ),
        code(
            """
            data = load_breast_cancer(as_frame=True)
            X = data.data
            y = data.target

            resumen = pd.DataFrame(
                {
                    "observaciones": [len(X)],
                    "variables": [X.shape[1]],
                    "fracción_clase_positiva": [y.mean()],
                }
            )
            display(resumen)
            display(X.head(3))
            """
        ),
        md(
            """
            ## 3. Tres particiones explícitas

            Esta partición sirve para entender los papeles. Reservamos 20 % para
            prueba; del 80 % restante usamos 25 % como validación. El resultado es
            60 % entrenamiento, 20 % validación y 20 % prueba.
            """
        ),
        code(
            """
            X_dev, X_test, y_dev, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                stratify=y,
                random_state=RANDOM_STATE,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_dev,
                y_dev,
                test_size=0.25,
                stratify=y_dev,
                random_state=RANDOM_STATE,
            )

            particiones = pd.DataFrame(
                {
                    "n": [len(X_train), len(X_val), len(X_test)],
                    "fracción positiva": [y_train.mean(), y_val.mean(), y_test.mean()],
                },
                index=["train", "validation", "test"],
            )
            display(particiones)
            """
        ),
        md(
            r"""
            ## 4. Línea base y métricas

            La exactitud $\mathrm{accuracy}=(TP+TN)/N$ puede ocultar fallos en una
            clase minoritaria. Por eso observaremos además:

            $$
            \mathrm{precision}=\frac{TP}{TP+FP},\qquad
            \mathrm{recall}=\frac{TP}{TP+FN},\qquad
            F_1=2\frac{\mathrm{precision}\,\mathrm{recall}}
            {\mathrm{precision}+\mathrm{recall}}.
            $$

            `StandardScaler` y regresión logística se encapsulan en un `Pipeline`.
            Al ejecutar `fit`, el escalador aprende media y desviación **sólo** de
            los datos que el estimador recibe en esa llamada.
            """
        ),
        code(
            """
            def metricas_binarias(y_true, y_pred, y_prob):
                '''Devuelve métricas con nombres legibles para comparar experimentos.'''
                return pd.Series(
                    {
                        "accuracy": accuracy_score(y_true, y_pred),
                        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, zero_division=0),
                        "recall": recall_score(y_true, y_pred, zero_division=0),
                        "f1": f1_score(y_true, y_pred, zero_division=0),
                        "roc_auc": roc_auc_score(y_true, y_prob),
                    }
                )


            baseline = Pipeline(
                steps=[
                    ("escala", StandardScaler()),
                    ("modelo", LogisticRegression(max_iter=5_000, random_state=RANDOM_STATE)),
                ]
            )
            baseline.fit(X_train, y_train)

            pred_val = baseline.predict(X_val)
            prob_val = baseline.predict_proba(X_val)[:, 1]
            display(metricas_binarias(y_val, pred_val, prob_val).to_frame("validación"))

            ConfusionMatrixDisplay.from_predictions(y_val, pred_val, cmap="Blues")
            plt.title("Matriz de confusión — validación")
            plt.show()
            """
        ),
        md(
            """
            ### Antipatrón: fuga de información

            Esto es incorrecto:

            ```python
            X_escalado = StandardScaler().fit_transform(X)  # vio validation y test
            X_train, X_test = train_test_split(X_escalado)  # demasiado tarde
            ```

            También hay fuga si imputamos, seleccionamos variables o reducimos
            dimensión antes de separar. Toda transformación que **aprende parámetros**
            pertenece al `Pipeline`.
            """
        ),
        md(
            """
            ## 5. Validación cruzada

            Una validación fija depende bastante de una sola partición. En
            validación cruzada de cinco pliegues, cada quinta parte de desarrollo
            actúa una vez como validación. El test continúa cerrado.
            """
        ),
        code(
            """
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scoring = {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1": "f1",
                "roc_auc": "roc_auc",
            }

            scores = cross_validate(
                baseline,
                X_dev,
                y_dev,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
            )
            resumen_cv = pd.DataFrame(
                {
                    metrica.removeprefix("test_"): [scores[metrica].mean(), scores[metrica].std()]
                    for metrica in scores
                    if metrica.startswith("test_")
                },
                index=["media", "desviación estándar"],
            ).T
            display(resumen_cv)
            """
        ),
        md(
            """
            ## 6. Ajuste de hiperparámetros

            `C` controla la fuerza de regularización de la regresión logística:
            valores pequeños regularizan más. Cada combinación se evalúa dentro de
            los pliegues; el escalador vuelve a ajustarse en cada entrenamiento.
            Elegimos con AUC y conservamos otras métricas para diagnóstico.
            """
        ),
        code(
            """
            parametros = {
                "modelo__C": np.logspace(-3, 3, 7),
                "modelo__class_weight": [None, "balanced"],
            }
            busqueda = GridSearchCV(
                estimator=baseline,
                param_grid=parametros,
                scoring=scoring,
                refit="roc_auc",
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
            )
            busqueda.fit(X_dev, y_dev)

            print("Mejores hiperparámetros:", busqueda.best_params_)
            print(f"AUC CV media: {busqueda.best_score_:.3f}")

            resultados = pd.DataFrame(busqueda.cv_results_)
            columnas = [
                "param_modelo__C",
                "param_modelo__class_weight",
                "mean_test_roc_auc",
                "std_test_roc_auc",
                "mean_test_f1",
                "rank_test_roc_auc",
            ]
            display(resultados[columnas].sort_values("rank_test_roc_auc").head(8))
            """
        ),
        md(
            """
            ## 7. Abrimos el test una sola vez

            `GridSearchCV(refit=...)` ya reentrenó la mejor configuración con todo
            desarrollo. Ahora y sólo ahora medimos generalización. Si cambiamos el
            modelo después de ver este resultado, el test pasa a ser otra validación
            y necesitamos un nuevo conjunto de prueba.
            """
        ),
        code(
            """
            mejor_modelo = busqueda.best_estimator_
            pred_test = mejor_modelo.predict(X_test)
            prob_test = mejor_modelo.predict_proba(X_test)[:, 1]

            informe_final = metricas_binarias(y_test, pred_test, prob_test)
            display(informe_final.to_frame("test final"))

            ConfusionMatrixDisplay.from_predictions(y_test, pred_test, cmap="Purples")
            plt.title("Matriz de confusión — test final")
            plt.show()
            """
        ),
        md(
            """
            ## Comprobación y ejercicios

            **Antes de continuar, explique sin código:**

            1. ¿Por qué el test no participa en `GridSearchCV`?
            2. ¿Qué información del conjunto de entrenamiento guarda el escalador?
            3. ¿Qué error sería más costoso en este ejemplo y qué métrica lo refleja?

            **Ejercicios**

            - Básico: cambie la semilla y cuantifique cuánto varía el test.
            - Intermedio: use `RepeatedStratifiedKFold` y compare la incertidumbre.
            - Reto: elija el umbral de decisión usando sólo predicciones de validación
              y evalúelo una vez en test.

            **Regla para el resto del curso:** partición, transformaciones, selección
            y métricas se deciden antes de mirar el resultado final.
            """
        ),
    ]
    write_notebook("00_fundamentos/00_protocolo_evaluacion.ipynb", cells)


def build_knn_svm_notebook() -> None:
    badge = f"{COLAB_ROOT}/01_nivelacion_ml/11_knn_svm.ipynb"
    cells = [
        md(
            f"""
            <a href="{badge}" target="_parent">
              <img src="https://colab.research.google.com/assets/colab-badge.svg"
                   alt="Abrir en Colab"/>
            </a>

            # K vecinos cercanos y máquinas de soporte vectorial

            **Aplicación:** identificar el régimen de un oscilador amortiguado a
            partir de su trayectoria ruidosa.<br>
            **Duración sugerida:** dos sesiones de 2 horas.

            Al terminar podrá:

            1. implementar la predicción KNN a partir de una distancia;
            2. interpretar margen, vectores de soporte y pérdida hinge;
            3. explicar por qué ambos métodos son sensibles a la escala;
            4. ajustar sus hiperparámetros dentro de `Pipeline`;
            5. compararlos con validación cruzada y un test intacto.
            """
        ),
        code(
            """
            import platform

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import sklearn
            import seaborn as sns
            from sklearn.datasets import make_blobs
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                accuracy_score,
                balanced_accuracy_score,
                classification_report,
                f1_score,
            )
            from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            RANDOM_STATE = 42
            rng = np.random.default_rng(RANDOM_STATE)
            sns.set_theme(style="whitegrid", context="notebook")

            print(f"Python: {platform.python_version()}")
            print(f"NumPy: {np.__version__}")
            print(f"scikit-learn: {sklearn.__version__}")
            """
        ),
        md(
            r"""
            ## 1. KNN: aprender conservando los ejemplos

            Para un punto nuevo $\mathbf{x}$, calculamos su distancia a los puntos
            de entrenamiento y hacemos votar a los $k$ más cercanos. La distancia
            de Minkowski es

            $$
            d_p(\mathbf{x},\mathbf{z})=
            \left(\sum_{j=1}^{m}|x_j-z_j|^p\right)^{1/p}.
            $$

            $p=1$ produce Manhattan y $p=2$ Euclídea. Si una variable se mide en
            miles y otra entre 0 y 1, la primera dominará la distancia: **escalar no
            es opcional**.

            KNN no ajusta una fórmula paramétrica durante `fit`; almacena los datos.
            Un $k$ pequeño da fronteras flexibles y alta varianza. Un $k$ grande
            suaviza la frontera y puede introducir sesgo.
            """
        ),
        code(
            """
            def predecir_knn_un_punto(x_nuevo, X_train, y_train, k=5, p=2):
                '''KNN mínimo: distancia, selección de vecinos y voto mayoritario.'''
                distancias = np.linalg.norm(X_train - x_nuevo, ord=p, axis=1)
                indices_vecinos = np.argsort(distancias)[:k]
                clases, conteos = np.unique(y_train[indices_vecinos], return_counts=True)
                prediccion = clases[np.argmax(conteos)]
                return prediccion, indices_vecinos, distancias[indices_vecinos]


            X_pequeno = np.array([[0.0, 0.2], [0.3, 0.1], [0.8, 0.9], [1.0, 0.7]])
            y_pequeno = np.array([0, 0, 1, 1])
            x_nuevo = np.array([0.65, 0.60])

            pred, vecinos, distancias = predecir_knn_un_punto(
                x_nuevo, X_pequeno, y_pequeno, k=3, p=2
            )
            print("Vecinos:", vecinos)
            print("Distancias:", np.round(distancias, 3))
            print("Predicción:", pred)
            """
        ),
        md(
            r"""
            ## 2. SVM: buscar una separación robusta

            Para un clasificador lineal $f(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b$,
            la frontera satisface $f(\mathbf{x})=0$. SVM busca un margen grande y
            penaliza puntos que caen dentro del margen o al lado incorrecto:

            $$
            \min_{\mathbf{w},b}\;\frac{1}{2}\|\mathbf{w}\|^2
            +C\sum_i\max\left(0,1-y_i f(\mathbf{x}_i)\right).
            $$

            El segundo término es la **pérdida hinge**. $C$ grande castiga con fuerza
            los errores; $C$ pequeño tolera más violaciones y regulariza. Un kernel
            reemplaza productos internos por similitudes. El RBF,

            $$K(\mathbf{x},\mathbf{z})=\exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2),$$

            permite fronteras curvas. `gamma` grande hace que cada punto influya en
            una región pequeña y aumenta la flexibilidad.
            """
        ),
        code(
            """
            def perdida_hinge(y_signed, score):
                '''Pérdida por observación para etiquetas codificadas como -1 y +1.'''
                return np.maximum(0.0, 1.0 - y_signed * score)


            ejemplo_margen = pd.DataFrame(
                {
                    "y": [1, 1, -1, -1],
                    "score": [2.0, 0.4, -1.4, 0.2],
                }
            )
            ejemplo_margen["hinge"] = perdida_hinge(
                ejemplo_margen["y"], ejemplo_margen["score"]
            )
            display(ejemplo_margen)
            """
        ),
        code(
            """
            # Geometría del margen en dos dimensiones (ejemplo sólo ilustrativo).
            X_2d, y_2d = make_blobs(
                n_samples=90,
                centers=[(-1.2, -0.8), (1.2, 0.8)],
                cluster_std=0.65,
                random_state=RANDOM_STATE,
            )
            svm_2d = SVC(kernel="linear", C=1.0).fit(X_2d, y_2d)

            xx, yy = np.meshgrid(
                np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 250),
                np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 250),
            )
            decision = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap="coolwarm", s=35)
            ax.contour(xx, yy, decision, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
            ax.scatter(
                svm_2d.support_vectors_[:, 0],
                svm_2d.support_vectors_[:, 1],
                s=130,
                facecolors="none",
                edgecolors="gold",
                linewidths=2,
                label="vectores de soporte",
            )
            ax.set(title="Frontera, márgenes y vectores de soporte", xlabel="$x_1$", ylabel="$x_2$")
            ax.legend()
            plt.show()
            """
        ),
        md(
            r"""
            ## 3. Sistema físico: oscilador amortiguado

            Simularemos la ecuación

            $$\ddot{x}+2\zeta\omega_0\dot{x}+\omega_0^2x=0,$$

            donde $\zeta$ es la razón de amortiguamiento. Queremos inferir, a partir
            de 80 mediciones ruidosas de $x(t)$, uno de tres regímenes:

            - **subamortiguado** ($0<\zeta<1$): oscila y decae;
            - **crítico** ($\zeta=1$): vuelve al equilibrio con rapidez sin oscilar;
            - **sobreamortiguado** ($\zeta>1$): suma de dos decaimientos.

            Variamos amplitud, frecuencia natural y ruido para que el algoritmo no
            memorice una sola curva ideal. Es un dataset **sintético inspirado en
            física**, no datos experimentales.
            """
        ),
        code(
            """
            def trayectoria_oscilador(t, zeta, omega0, amplitud):
                '''Solución con x(0)=amplitud y velocidad inicial cero.'''
                if zeta < 1.0:
                    raiz = np.sqrt(1.0 - zeta**2)
                    omega_d = omega0 * raiz
                    return amplitud * np.exp(-zeta * omega0 * t) * (
                        np.cos(omega_d * t) + zeta / raiz * np.sin(omega_d * t)
                    )
                if zeta == 1.0:
                    return amplitud * (1.0 + omega0 * t) * np.exp(-omega0 * t)

                raiz = np.sqrt(zeta**2 - 1.0)
                r1 = -omega0 * (zeta - raiz)
                r2 = -omega0 * (zeta + raiz)
                c1 = -amplitud * r2 / (r1 - r2)
                c2 = amplitud * r1 / (r1 - r2)
                return c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)


            def crear_dataset_osciladores(n_por_clase=180, n_tiempos=80, semilla=42):
                generador = np.random.default_rng(semilla)
                tiempos = np.linspace(0.0, 8.0, n_tiempos)
                curvas, etiquetas = [], []

                for clase in range(3):
                    for _ in range(n_por_clase):
                        omega0 = generador.uniform(0.8, 1.2)
                        amplitud = generador.uniform(0.8, 1.2)
                        if clase == 0:
                            zeta = generador.uniform(0.08, 0.80)
                        elif clase == 1:
                            zeta = 1.0
                        else:
                            zeta = generador.uniform(1.20, 2.00)

                        x = trayectoria_oscilador(tiempos, zeta, omega0, amplitud)
                        ruido = generador.normal(0.0, 0.025, size=n_tiempos)
                        curvas.append(x + ruido)
                        etiquetas.append(clase)

                orden = generador.permutation(len(etiquetas))
                X = np.asarray(curvas)[orden]
                y = np.asarray(etiquetas)[orden]
                columnas = [f"x(t={t:.2f})" for t in tiempos]
                return pd.DataFrame(X, columns=columnas), pd.Series(y, name="régimen"), tiempos


            nombres = {0: "subamortiguado", 1: "crítico", 2: "sobreamortiguado"}
            X, y, tiempos = crear_dataset_osciladores()
            print("Forma de X:", X.shape)
            display(y.value_counts().sort_index().rename(index=nombres).to_frame("observaciones"))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
            for clase, ax in enumerate(axes):
                indices = np.flatnonzero(y.to_numpy() == clase)[:8]
                for indice in indices:
                    ax.plot(tiempos, X.iloc[indice], alpha=0.55)
                ax.set(title=nombres[clase], xlabel="tiempo")
            axes[0].set_ylabel("desplazamiento $x(t)$")
            fig.suptitle("Muestras ruidosas: ocho trayectorias por régimen")
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 4. División y pipelines

            Apartamos 20 % como test estratificado. `GridSearchCV` dividirá el 80 %
            de desarrollo en cinco pliegues: en cada iteración cuatro entrenan y uno
            valida. Así, los pliegues cumplen el papel de validación sin desperdiciar
            una partición fija.

            El escalado ocurre dentro de cada pipeline. Aunque todas las columnas
            tienen la misma unidad, su dispersión cambia mucho con el tiempo debido
            al decaimiento; estandarizar evita ponderarlas accidentalmente sólo por
            su varianza.
            """
        ),
        code(
            """
            X_dev, X_test, y_dev, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                stratify=y,
                random_state=RANDOM_STATE,
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            knn = Pipeline(
                [("escala", StandardScaler()), ("modelo", KNeighborsClassifier())]
            )
            svm = Pipeline(
                [
                    ("escala", StandardScaler()),
                    ("modelo", SVC(random_state=RANDOM_STATE)),
                ]
            )

            print(f"Desarrollo: {len(X_dev)} | test intacto: {len(X_test)}")
            """
        ),
        md(
            """
            ## 5. Ajuste de KNN

            Exploramos número de vecinos, voto uniforme o ponderado por distancia y
            dos métricas de Minkowski. Seleccionamos por $F_1$ macro: calcula $F_1$
            por clase y les da el mismo peso.
            """
        ),
        code(
            """
            grilla_knn = {
                "modelo__n_neighbors": [3, 7, 15, 31],
                "modelo__weights": ["uniform", "distance"],
                "modelo__p": [1, 2],
            }
            busqueda_knn = GridSearchCV(
                knn,
                grilla_knn,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
            )
            busqueda_knn.fit(X_dev, y_dev)

            print("KNN elegido:", busqueda_knn.best_params_)
            print(f"F1-macro CV: {busqueda_knn.best_score_:.3f}")

            resultados_knn = pd.DataFrame(busqueda_knn.cv_results_)
            vista_knn = resultados_knn[
                [
                    "param_modelo__n_neighbors",
                    "param_modelo__weights",
                    "param_modelo__p",
                    "mean_train_score",
                    "mean_test_score",
                ]
            ].sort_values("mean_test_score", ascending=False)
            display(vista_knn.head(8))
            """
        ),
        md(
            """
            ## 6. Ajuste de SVM

            Usamos dos grillas separadas: `gamma` sólo tiene sentido para RBF. Esta
            forma evita combinaciones irrelevantes y enseña que una búsqueda debe
            representar decisiones científicas, no sólo producir muchas corridas.
            """
        ),
        code(
            """
            grilla_svm = [
                {
                    "modelo__kernel": ["linear"],
                    "modelo__C": [0.1, 1.0, 10.0],
                },
                {
                    "modelo__kernel": ["rbf"],
                    "modelo__C": [0.1, 1.0, 10.0],
                    "modelo__gamma": ["scale", 0.01, 0.1, 1.0],
                },
            ]
            busqueda_svm = GridSearchCV(
                svm,
                grilla_svm,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
            )
            busqueda_svm.fit(X_dev, y_dev)

            print("SVM elegida:", busqueda_svm.best_params_)
            print(f"F1-macro CV: {busqueda_svm.best_score_:.3f}")

            resultados_svm = pd.DataFrame(busqueda_svm.cv_results_)
            columnas_svm = [
                "param_modelo__kernel",
                "param_modelo__C",
                "param_modelo__gamma",
                "mean_train_score",
                "mean_test_score",
            ]
            display(resultados_svm[columnas_svm].sort_values("mean_test_score", ascending=False).head(8))
            """
        ),
        md(
            """
            ## 7. Comparación y apertura del test

            Primero comparamos la media de validación cruzada. Después evaluamos una
            sola vez ambas configuraciones elegidas sobre las mismas observaciones
            de prueba. Además de exactitud usamos exactitud balanceada, $F_1$ macro y
            $F_1$ macro. Las probabilidades de una SVM requieren un paso adicional
            de calibración y se estudiarán junto con incertidumbre de predicción.
            """
        ),
        code(
            """
            def evaluar_modelo(nombre, busqueda, X_test, y_test):
                modelo = busqueda.best_estimator_
                pred = modelo.predict(X_test)
                return {
                    "modelo": nombre,
                    "f1_macro_cv": busqueda.best_score_,
                    "accuracy_test": accuracy_score(y_test, pred),
                    "balanced_accuracy_test": balanced_accuracy_score(y_test, pred),
                    "f1_macro_test": f1_score(y_test, pred, average="macro"),
                }


            comparacion = pd.DataFrame(
                [
                    evaluar_modelo("KNN", busqueda_knn, X_test, y_test),
                    evaluar_modelo("SVM", busqueda_svm, X_test, y_test),
                ]
            ).set_index("modelo")
            display(comparacion.style.highlight_max(subset=["f1_macro_test"], color="#c6efce"))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for nombre, busqueda, ax in [
                ("KNN", busqueda_knn, axes[0]),
                ("SVM", busqueda_svm, axes[1]),
            ]:
                pred = busqueda.best_estimator_.predict(X_test)
                ConfusionMatrixDisplay.from_predictions(
                    y_test,
                    pred,
                    display_labels=[nombres[i] for i in range(3)],
                    cmap="Blues",
                    colorbar=False,
                    ax=ax,
                )
                ax.set_title(nombre)
                ax.tick_params(axis="x", rotation=25)
            fig.suptitle("Matrices de confusión sobre el mismo test")
            fig.tight_layout()
            plt.show()

            mejor_busqueda = max([busqueda_knn, busqueda_svm], key=lambda b: b.best_score_)
            pred_mejor = mejor_busqueda.best_estimator_.predict(X_test)
            print(
                classification_report(
                    y_test,
                    pred_mejor,
                    target_names=[nombres[i] for i in range(3)],
                    zero_division=0,
                )
            )
            """
        ),
        md(
            """
            ## 8. Interpretación y límites

            - KNN debe almacenar desarrollo y comparar cada nueva curva con muchos
              ejemplos. Es transparente localmente, pero su costo de inferencia y la
              maldición de la dimensionalidad pueden crecer.
            - SVM depende sobre todo de los vectores de soporte. RBF puede capturar
              similitudes no lineales, pero `C` y `gamma` interactúan y las
              puntuaciones de margen no son probabilidades; calibrarlas es un modelo
              adicional y debe hacerse usando sólo desarrollo.
            - Una puntuación alta aquí no demuestra éxito experimental: las curvas
              provienen exactamente de las ecuaciones que asumimos y el ruido es
              gaussiano simple. Datos reales introducen deriva, respuesta del sensor,
              valores ausentes y mecanismos no modelados.

            ## Preguntas y ejercicios

            1. ¿Qué le ocurre a KNN si duplicamos numéricamente sólo una variable?
            2. Compare la brecha `mean_train_score - mean_test_score`. ¿Dónde ve
               sobreajuste?
            3. Quite `StandardScaler` de ambos pipelines y cuantifique el cambio.
            4. **Reto físico:** entrene con ruido estándar 0.025 y pruebe con 0.075.
               ¿Sigue siendo válida la estimación del test original?
            5. **Reto matemático:** grafique la pérdida hinge para $y=+1$ y explique
               por qué deja de premiar predicciones más allá del margen.

            **Salida para el informe:** tabla de comparación, matriz de confusión,
            mejor configuración y un párrafo que diferencie error estadístico de
            error de modelado físico.
            """
        ),
    ]
    write_notebook("01_nivelacion_ml/11_knn_svm.ipynb", cells)


if __name__ == "__main__":
    build_evaluation_notebook()
    build_knn_svm_notebook()
    print("Notebooks iniciales generados sin salidas de ejecución.")
