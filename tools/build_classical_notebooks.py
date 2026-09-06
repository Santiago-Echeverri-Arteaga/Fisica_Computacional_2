"""Construye los notebooks de fundamentos, ML clásico y clustering."""

from __future__ import annotations

from build_initial_notebooks import COLAB_ROOT, code, md, write_notebook


def header(path: str, title: str, question: str, hours: int = 2) -> dict:
    return md(
        f"""
        <a href="{COLAB_ROOT}/{path}" target="_parent">
          <img src="https://colab.research.google.com/assets/colab-badge.svg"
               alt="Abrir en Colab"/>
        </a>

        # {title}

        **Pregunta guía:** {question}<br>
        **Duración sugerida:** {hours} horas.<br>
        **Entorno:** CPU; datos incluidos o generados en memoria.

        El orden de trabajo es siempre: problema → matemática → implementación
        mínima → biblioteca → evaluación → interpretación física.
        """
    )


def build_reproducibility() -> None:
    path = "00_fundamentos/01_pipelines_reproducibilidad.ipynb"
    cells = [
        header(path, "Pipelines y reproducibilidad", "¿Qué hace repetible y auditable un experimento?"),
        md(
            r"""
            ## Objetivos

            1. Diferenciar aleatoriedad controlada de determinismo absoluto.
            2. Encapsular imputación, escala y modelo en un `Pipeline`.
            3. registrar configuración, versiones y resultados sin copiar celdas;
            4. guardar y recuperar el pipeline completo.

            Un resultado reproducible especifica datos, código, dependencias,
            semilla, partición y métrica. Si $\hat\mu$ y $\hat\sigma$ se calculan
            con entrenamiento, una medida nueva se transforma como
            $z=(x-\hat\mu)/\hat\sigma$ usando **esos mismos parámetros**.
            """
        ),
        code(
            """
            import json
            import platform
            import tempfile
            from pathlib import Path

            import joblib
            import numpy as np
            import pandas as pd
            import sklearn
            from sklearn.datasets import make_regression
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_absolute_error, root_mean_squared_error
            from sklearn.model_selection import GridSearchCV, KFold, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            SEMILLA = 2026
            rng = np.random.default_rng(SEMILLA)
            print({"python": platform.python_version(), "sklearn": sklearn.__version__})
            """
        ),
        md(
            """
            ## Un experimento pequeño

            Simulamos cinco canales de un detector y una energía objetivo. Insertamos
            valores ausentes sólo para mostrar que la imputación también debe estar
            dentro del pipeline.
            """
        ),
        code(
            """
            X_array, y_array = make_regression(
                n_samples=500,
                n_features=5,
                n_informative=4,
                noise=12.0,
                random_state=SEMILLA,
            )
            X = pd.DataFrame(X_array, columns=[f"canal_{i}" for i in range(5)])
            y = pd.Series(y_array, name="energía")
            mascara = rng.random(X.shape) < 0.03
            X = X.mask(mascara)

            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEMILLA
            )
            pipe = Pipeline(
                [
                    ("imputación", SimpleImputer(strategy="median")),
                    ("escala", StandardScaler()),
                    ("modelo", Ridge()),
                ]
            )
            cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
            búsqueda = GridSearchCV(
                pipe,
                {"modelo__alpha": np.logspace(-3, 3, 13)},
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
            ).fit(X_dev, y_dev)

            pred = búsqueda.best_estimator_.predict(X_test)
            resultado = {
                "semilla": SEMILLA,
                "mejor_alpha": búsqueda.best_params_["modelo__alpha"],
                "rmse_test": root_mean_squared_error(y_test, pred),
                "mae_test": mean_absolute_error(y_test, pred),
            }
            print(json.dumps(resultado, indent=2))
            """
        ),
        md(
            """
            ## Persistir el objeto correcto

            Guardar sólo el estimador perdería la mediana y la escala aprendidas.
            Guardamos el pipeline entero y comprobamos que las predicciones sean
            idénticas. El archivo temporal se elimina automáticamente.
            """
        ),
        code(
            """
            with tempfile.TemporaryDirectory() as carpeta:
                ruta = Path(carpeta) / "pipeline.joblib"
                joblib.dump(búsqueda.best_estimator_, ruta)
                recuperado = joblib.load(ruta)
                diferencia = np.max(np.abs(pred - recuperado.predict(X_test)))

            print(f"Diferencia máxima después de recuperar: {diferencia:.2e}")
            assert diferencia == 0.0
            """
        ),
        md(
            """
            ## Lista de control y ejercicios

            - ¿El test se separó antes de seleccionar hiperparámetros?
            - ¿Todas las transformaciones que aprenden están en el pipeline?
            - ¿Se registran versiones, semilla, configuración y métrica?
            - ¿El artefacto guardado recibe datos crudos en el mismo esquema?

            **Ejercicios:** cambie la imputación por media; repita con cinco semillas;
            guarde en JSON media y desviación de RMSE; explique qué parte sigue sin
            ser determinista si se usa una GPU.
            """
        ),
    ]
    write_notebook(path, cells)


def build_regression() -> None:
    path = "01_nivelacion_ml/10_regresion_regularizacion.ipynb"
    cells = [
        header(path, "Regresión multilineal, Ridge y Lasso", "¿Cómo regularizamos una ley física aproximada?", 4),
        md(
            r"""
            ## Modelo y regularización

            En mínimos cuadrados buscamos
            $\min_{\boldsymbol\beta}\|\mathbf y-X\boldsymbol\beta\|_2^2$.
            Ridge añade $\alpha\|\boldsymbol\beta\|_2^2$ y contrae coeficientes
            correlacionados; Lasso añade $\alpha\|\boldsymbol\beta\|_1$ y puede
            volver algunos exactamente cero. La penalización depende de la escala,
            por lo que estandarizamos dentro del pipeline.

            Simularemos el alcance de un proyectil con una corrección suave por
            arrastre. No afirmamos que la corrección sea una solución exacta: es un
            laboratorio de identificación de un modelo sustituto.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.linear_model import Lasso, LinearRegression, Ridge
            from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
            from sklearn.model_selection import GridSearchCV, KFold, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import PolynomialFeatures, StandardScaler

            SEMILLA = 42
            rng = np.random.default_rng(SEMILLA)
            n = 800
            v0 = rng.uniform(10, 80, n)
            theta = rng.uniform(np.deg2rad(10), np.deg2rad(80), n)
            masa = rng.uniform(0.05, 2.0, n)
            arrastre = rng.uniform(0.0, 0.06, n)
            g = 9.81
            alcance_vacío = v0**2 * np.sin(2 * theta) / g
            factor = 1 / (1 + 5.0 * arrastre * v0 / masa)
            alcance = alcance_vacío * factor + rng.normal(0, 2.0, n)

            X = pd.DataFrame(
                {"v0": v0, "theta_rad": theta, "masa": masa, "arrastre": arrastre}
            )
            y = pd.Series(alcance, name="alcance_m")
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEMILLA
            )
            display(X.head())
            """
        ),
        md(
            r"""
            ## Modelos comparables

            Los tres reciben las mismas variables polinomiales de grado 2, escala,
            pliegues y test. `PolynomialFeatures` permite productos como
            $v_0\,c/m$; sigue siendo una regresión lineal respecto a sus parámetros.
            """
        ),
        code(
            """
            cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)

            def pipeline(modelo):
                return Pipeline(
                    [
                        ("polinomio", PolynomialFeatures(degree=2, include_bias=False)),
                        ("escala", StandardScaler()),
                        ("modelo", modelo),
                    ]
                )

            configuraciones = {
                "lineal": (pipeline(LinearRegression()), {}),
                "ridge": (pipeline(Ridge()), {"modelo__alpha": np.logspace(-4, 4, 17)}),
                "lasso": (
                    pipeline(Lasso(max_iter=30_000)),
                    {"modelo__alpha": np.logspace(-4, 1, 16)},
                ),
            }

            búsquedas = {}
            filas = []
            for nombre, (estimador, grilla) in configuraciones.items():
                búsqueda = GridSearchCV(
                    estimador,
                    grilla,
                    scoring="neg_root_mean_squared_error",
                    cv=cv,
                    n_jobs=-1,
                ).fit(X_dev, y_dev)
                búsquedas[nombre] = búsqueda
                pred = búsqueda.predict(X_test)
                filas.append(
                    {
                        "modelo": nombre,
                        "RMSE_CV": -búsqueda.best_score_,
                        "RMSE_test": root_mean_squared_error(y_test, pred),
                        "MAE_test": mean_absolute_error(y_test, pred),
                        "R2_test": r2_score(y_test, pred),
                        "hiperparámetros": búsqueda.best_params_,
                    }
                )

            comparación = pd.DataFrame(filas).set_index("modelo")
            display(comparación)
            """
        ),
        code(
            """
            mejor_nombre = comparación["RMSE_CV"].idxmin()
            mejor = búsquedas[mejor_nombre].best_estimator_
            pred = mejor.predict(X_test)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].scatter(y_test, pred, alpha=0.55)
            límites = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
            axes[0].plot(límites, límites, "k--")
            axes[0].set(xlabel="alcance real", ylabel="predicción", title=mejor_nombre)
            axes[1].scatter(pred, y_test - pred, alpha=0.55)
            axes[1].axhline(0, color="k", linestyle="--")
            axes[1].set(xlabel="predicción", ylabel="residuo", title="Diagnóstico de residuos")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Lectura física y ejercicios

            Un buen $R^2$ no valida la ley de arrastre: sólo indica que el sustituto
            predice bien dentro de la distribución simulada. Inspeccione residuos
            contra velocidad y arrastre para buscar estructura omitida.

            - Básico: use sólo la fórmula de vacío como predictor y mida el error.
            - Intermedio: grafique norma de coeficientes contra $\alpha$.
            - Reto: entrene con velocidades hasta 60 m/s y pruebe entre 60–80 m/s;
              discuta interpolación frente a extrapolación.
            """
        ),
    ]
    write_notebook(path, cells)


def _classification_imports() -> dict:
    return code(
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
        )
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

        SEMILLA = 42
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
        """
    )


def build_trees_forests_bagging() -> None:
    path = "01_nivelacion_ml/12_arboles_bosques_bagging.ipynb"
    cells = [
        header(path, "Árboles, bosques y bagging", "¿Cómo reduce varianza un conjunto de árboles?", 4),
        md(
            r"""
            ## Matemática esencial

            Un árbol elige cortes que reducen impureza. Para clases con proporciones
            $p_k$, Gini es $G=1-\sum_k p_k^2$. Un árbol profundo tiene poco sesgo y
            alta varianza. Bagging entrena estimadores sobre muestras bootstrap y
            promedia sus predicciones. Si los errores tienen correlación $\rho$, la
            varianza del promedio de $B$ modelos se aproxima por
            $\sigma^2[\rho+(1-\rho)/B]$. Random Forest también submuestrea variables
            para reducir $\rho$.
            """
        ),
        _classification_imports(),
        code(
            """
            from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier, plot_tree

            X, y = make_classification(
                n_samples=1_200,
                n_features=12,
                n_informative=6,
                n_redundant=3,
                class_sep=1.0,
                flip_y=0.04,
                random_state=SEMILLA,
            )
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEMILLA
            )
            modelos = {
                "árbol": (
                    DecisionTreeClassifier(random_state=SEMILLA),
                    {"max_depth": [2, 4, 8, None], "min_samples_leaf": [1, 5, 15]},
                ),
                "bagging": (
                    BaggingClassifier(
                        estimator=DecisionTreeClassifier(random_state=SEMILLA),
                        random_state=SEMILLA,
                        n_jobs=-1,
                    ),
                    {"n_estimators": [30, 100], "max_samples": [0.6, 1.0]},
                ),
                "random_forest": (
                    RandomForestClassifier(random_state=SEMILLA, n_jobs=-1),
                    {
                        "n_estimators": [100, 250],
                        "max_depth": [4, 10, None],
                        "max_features": ["sqrt", 0.7],
                        "min_samples_leaf": [1, 5],
                    },
                ),
            }

            filas, búsquedas = [], {}
            for nombre, (modelo, grilla) in modelos.items():
                búsqueda = GridSearchCV(
                    modelo, grilla, scoring="f1", cv=cv, n_jobs=-1, return_train_score=True
                ).fit(X_dev, y_dev)
                pred = búsqueda.predict(X_test)
                búsquedas[nombre] = búsqueda
                filas.append(
                    {
                        "modelo": nombre,
                        "F1_CV": búsqueda.best_score_,
                        "F1_test": f1_score(y_test, pred),
                        "accuracy_test": accuracy_score(y_test, pred),
                        "mejor": búsqueda.best_params_,
                    }
                )
            display(pd.DataFrame(filas).set_index("modelo"))
            """
        ),
        code(
            """
            árbol_pequeño = DecisionTreeClassifier(max_depth=3, random_state=SEMILLA).fit(
                X_dev, y_dev
            )
            plt.figure(figsize=(16, 6))
            plot_tree(árbol_pequeño, max_depth=2, filled=True, fontsize=8)
            plt.title("Primeras decisiones de un árbol limitado")
            plt.show()

            bosque = búsquedas["random_forest"].best_estimator_
            importancia = pd.Series(bosque.feature_importances_).sort_values(ascending=False)
            importancia.head(10).sort_values().plot.barh(title="Importancia por impureza")
            plt.xlabel("reducción media de impureza")
            plt.show()
            """
        ),
        md(
            """
            **Advertencia:** la importancia por impureza puede favorecer variables
            continuas o con muchas categorías; se contrastará con permutación en el
            notebook 15.

            **Ejercicios:** observe la brecha train–CV al variar profundidad; mida
            cuánto cambia el bosque con cinco semillas; explique por qué aumentar
            árboles reduce varianza pero no corrige un sesgo sistemático.
            """
        ),
    ]
    write_notebook(path, cells)


def build_boosting_stacking() -> None:
    path = "01_nivelacion_ml/13_boosting_stacking.ipynb"
    cells = [
        header(path, "Boosting y stacking", "¿Cómo combinamos modelos que corrigen errores distintos?", 4),
        md(
            r"""
            ## Tres ideas que no deben confundirse

            - **Bagging:** modelos en paralelo sobre bootstrap; reduce varianza.
            - **Boosting:** modelos secuenciales corrigen residuos o ejemplos
              difíciles. En gradient boosting,
              $F_m(x)=F_{m-1}(x)+\eta h_m(x)$.
            - **Stacking:** un metamodelo aprende a combinar predicciones *fuera de
              pliegue* de modelos distintos.

            “Blagging” no es un método estándar en este contexto; normalmente se
            quiere decir **bagging** o **boosting**.
            """
        ),
        _classification_imports(),
        code(
            """
            from sklearn.ensemble import (
                AdaBoostClassifier,
                GradientBoostingClassifier,
                HistGradientBoostingClassifier,
                StackingClassifier,
            )
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.tree import DecisionTreeClassifier

            X, y = make_classification(
                n_samples=1_400,
                n_features=16,
                n_informative=7,
                n_redundant=4,
                class_sep=0.9,
                flip_y=0.06,
                random_state=SEMILLA,
            )
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEMILLA
            )

            candidatos = {
                "AdaBoost": (
                    AdaBoostClassifier(random_state=SEMILLA),
                    {"n_estimators": [50, 150], "learning_rate": [0.03, 0.1, 0.5]},
                ),
                "GradientBoosting": (
                    GradientBoostingClassifier(random_state=SEMILLA),
                    {
                        "n_estimators": [80, 160],
                        "learning_rate": [0.03, 0.1],
                        "max_depth": [1, 2],
                    },
                ),
                "HistGradientBoosting": (
                    HistGradientBoostingClassifier(random_state=SEMILLA),
                    {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31]},
                ),
            }
            filas, búsquedas = [], {}
            for nombre, (modelo, grilla) in candidatos.items():
                búsqueda = GridSearchCV(modelo, grilla, scoring="f1", cv=cv, n_jobs=-1).fit(
                    X_dev, y_dev
                )
                búsquedas[nombre] = búsqueda
                filas.append(
                    {
                        "modelo": nombre,
                        "F1_CV": búsqueda.best_score_,
                        "F1_test": f1_score(y_test, búsqueda.predict(X_test)),
                        "mejor": búsqueda.best_params_,
                    }
                )
            display(pd.DataFrame(filas).set_index("modelo"))
            """
        ),
        code(
            """
            base = [
                (
                    "logística",
                    Pipeline(
                        [("escala", StandardScaler()), ("modelo", LogisticRegression(max_iter=3000))]
                    ),
                ),
                (
                    "knn",
                    Pipeline(
                        [("escala", StandardScaler()), ("modelo", KNeighborsClassifier(15))]
                    ),
                ),
                ("árbol", DecisionTreeClassifier(max_depth=5, random_state=SEMILLA)),
            ]
            stacking = StackingClassifier(
                estimators=base,
                final_estimator=LogisticRegression(max_iter=3000),
                cv=cv,
                n_jobs=-1,
            )
            stacking.fit(X_dev, y_dev)
            pred_stack = stacking.predict(X_test)
            print(f"F1 test stacking: {f1_score(y_test, pred_stack):.3f}")
            """
        ),
        md(
            """
            En stacking, entrenar el metamodelo con predicciones hechas sobre los
            mismos datos usados para ajustar los modelos base produciría fuga. La
            implementación genera predicciones fuera de pliegue internamente.

            **Ejercicios:** trace error contra número de estimadores; compare tiempos;
            cambie el metamodelo; determine si el stacking mejora de forma estable en
            diez particiones o sólo en la semilla mostrada.
            """
        ),
    ]
    write_notebook(path, cells)


def build_imbalance() -> None:
    path = "01_nivelacion_ml/14_clases_desbalanceadas.ipynb"
    cells = [
        header(path, "Clases desbalanceadas", "¿Cómo detectar eventos raros sin engañarnos con accuracy?", 4),
        md(
            r"""
            ## El problema

            Simulamos un detector donde sólo 2 % son eventos señal. Predecir siempre
            “fondo” logra cerca de 98 % de accuracy y utilidad científica nula.
            Precision–recall responde preguntas distintas: de las alarmas, ¿cuántas
            son reales?, y de las señales, ¿cuántas recuperamos?

            El remuestreo debe ocurrir **dentro de cada pliegue de entrenamiento**.
            Hacer SMOTE antes de separar replica información hacia validación.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from imblearn.over_sampling import RandomOverSampler, SMOTE
            from imblearn.pipeline import Pipeline
            from sklearn.datasets import make_classification
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                average_precision_score,
                balanced_accuracy_score,
                classification_report,
                precision_recall_curve,
            )
            from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
            from sklearn.preprocessing import StandardScaler

            SEMILLA = 42
            X, y = make_classification(
                n_samples=4_000,
                n_features=15,
                n_informative=7,
                n_redundant=4,
                weights=[0.98, 0.02],
                class_sep=1.0,
                random_state=SEMILLA,
            )
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEMILLA
            )
            print(pd.Series(y).value_counts(normalize=True).rename("fracción"))
            """
        ),
        code(
            """
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
            estrategias = {
                "dummy": Pipeline([("modelo", DummyClassifier(strategy="most_frequent"))]),
                "pesos": Pipeline(
                    [
                        ("escala", StandardScaler()),
                        ("modelo", LogisticRegression(class_weight="balanced", max_iter=3000)),
                    ]
                ),
                "oversampling": Pipeline(
                    [
                        ("escala", StandardScaler()),
                        ("muestreo", RandomOverSampler(random_state=SEMILLA)),
                        ("modelo", LogisticRegression(max_iter=3000)),
                    ]
                ),
                "SMOTE": Pipeline(
                    [
                        ("escala", StandardScaler()),
                        ("muestreo", SMOTE(random_state=SEMILLA)),
                        ("modelo", LogisticRegression(max_iter=3000)),
                    ]
                ),
            }
            filas, ajustados = [], {}
            for nombre, modelo in estrategias.items():
                grilla = {"modelo__C": [0.1, 1, 10]} if nombre != "dummy" else {}
                búsqueda = GridSearchCV(
                    modelo, grilla, scoring="average_precision", cv=cv, n_jobs=-1
                ).fit(X_dev, y_dev)
                ajustados[nombre] = búsqueda.best_estimator_
                prob = búsqueda.predict_proba(X_test)[:, 1]
                pred = búsqueda.predict(X_test)
                filas.append(
                    {
                        "estrategia": nombre,
                        "AP_test": average_precision_score(y_test, prob),
                        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                    }
                )
            display(pd.DataFrame(filas).set_index("estrategia"))
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(7, 5))
            for nombre, modelo in ajustados.items():
                prob = modelo.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, prob)
                ax.plot(recall, precision, label=nombre)
            ax.axhline(y_test.mean(), color="k", linestyle="--", label="prevalencia")
            ax.set(xlabel="recall", ylabel="precision", title="Curvas precision–recall")
            ax.legend()
            plt.show()

            mejor = ajustados["pesos"]
            prob = mejor.predict_proba(X_test)[:, 1]
            umbrales = np.linspace(0.05, 0.95, 19)
            tabla = []
            for u in umbrales:
                pred = (prob >= u).astype(int)
                tabla.append({"umbral": u, "alarmas": pred.sum(), "recall": ((pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum()})
            display(pd.DataFrame(tabla))
            """
        ),
        md(
            """
            El umbral se escoge en validación según el costo científico de falsos
            positivos y falsos negativos; la tabla de test es sólo demostrativa.

            **Ejercicios:** imponga recall mínimo 0.90 usando predicciones fuera de
            pliegue; compare calibración; explique cuándo SMOTE sería físicamente
            absurdo porque interpola entre eventos que no admiten estados intermedios.
            """
        ),
    ]
    write_notebook(path, cells)


def build_explainability() -> None:
    path = "01_nivelacion_ml/15_explicabilidad_local_global.ipynb"
    cells = [
        header(path, "Explicabilidad local y global", "¿Qué aprendió un modelo y cuándo no debemos creer la explicación?", 4),
        md(
            r"""
            ## Tres aproximaciones

            1. **Importancia por permutación:** se rompe una variable y se mide la
               caída de desempeño. Es global y refleja dependencia predictiva, no
               causalidad.
            2. **Sustituto global:** un modelo interpretable $g(x)$ aproxima las
               predicciones de una caja negra $f(x)$ en toda la población.
            3. **Sustituto local:** se perturba alrededor de $x_0$ y se ajusta un
               modelo simple ponderado por proximidad. La fidelidad es local.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import make_friedman1
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import PartialDependenceDisplay, permutation_importance
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score, root_mean_squared_error
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.tree import DecisionTreeRegressor, plot_tree

            SEMILLA = 42
            X_array, y = make_friedman1(n_samples=1_500, n_features=10, noise=1.0, random_state=SEMILLA)
            nombres = [f"sensor_{i}" for i in range(X_array.shape[1])]
            X = pd.DataFrame(X_array, columns=nombres)
            X_dev, X_test, y_dev, y_test = train_test_split(
                X, y, test_size=0.25, random_state=SEMILLA
            )
            caja_negra = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, random_state=SEMILLA, n_jobs=-1
            ).fit(X_dev, y_dev)
            pred = caja_negra.predict(X_test)
            print(f"RMSE test: {root_mean_squared_error(y_test, pred):.3f}")
            """
        ),
        code(
            """
            perm = permutation_importance(
                caja_negra,
                X_test,
                y_test,
                scoring="neg_root_mean_squared_error",
                n_repeats=15,
                random_state=SEMILLA,
                n_jobs=-1,
            )
            importancia = pd.DataFrame(
                {"variable": nombres, "caída_media": perm.importances_mean, "std": perm.importances_std}
            ).sort_values("caída_media", ascending=False)
            display(importancia)

            variables = importancia.head(3)["variable"].tolist()
            PartialDependenceDisplay.from_estimator(caja_negra, X_test, variables)
            plt.suptitle("Dependencia parcial: promedio global condicionado por el modelo")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # Sustituto global: aprende a reproducir f(X), no las etiquetas originales.
            y_caja_dev = caja_negra.predict(X_dev)
            sustituto_global = DecisionTreeRegressor(max_depth=3, random_state=SEMILLA).fit(
                X_dev, y_caja_dev
            )
            fidelidad_global = r2_score(pred, sustituto_global.predict(X_test))
            print(f"Fidelidad global R²(g(X), f(X)): {fidelidad_global:.3f}")

            plt.figure(figsize=(16, 6))
            plot_tree(sustituto_global, feature_names=nombres, filled=True, fontsize=8)
            plt.title("Árbol sustituto global")
            plt.show()
            """
        ),
        code(
            """
            def sustituto_local(modelo, x0, X_referencia, n=1500, ancho=0.75, semilla=42):
                rng = np.random.default_rng(semilla)
                escala = X_referencia.std(axis=0).to_numpy()
                vecinos = x0.to_numpy() + rng.normal(size=(n, len(x0))) * escala * ancho
                vecinos = pd.DataFrame(vecinos, columns=X_referencia.columns)
                dist2 = np.sum(((vecinos - x0) / escala) ** 2, axis=1)
                pesos = np.exp(-dist2 / (2 * ancho**2))
                objetivo_local = modelo.predict(vecinos)
                local = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                local.fit(vecinos, objetivo_local, ridge__sample_weight=pesos)
                coef = local.named_steps["ridge"].coef_
                fidelidad = r2_score(objetivo_local, local.predict(vecinos), sample_weight=pesos)
                return pd.Series(coef, index=X_referencia.columns), fidelidad


            x0 = X_test.iloc[0]
            coeficientes, fidelidad_local = sustituto_local(caja_negra, x0, X_dev)
            print(f"Predicción explicada: {caja_negra.predict(x0.to_frame().T)[0]:.3f}")
            print(f"Fidelidad local ponderada: {fidelidad_local:.3f}")
            display(coeficientes.reindex(coeficientes.abs().sort_values(ascending=False).index).head(6))
            """
        ),
        md(
            """
            Una explicación no convierte correlación en mecanismo físico. Debe
            reportarse junto con su fidelidad, región de validez, variables
            correlacionadas y sensibilidad a la semilla/ancho local.

            **Ejercicios:** explique tres observaciones distintas; cambie el ancho;
            duplique un sensor correlacionado y observe la importancia; compare el
            sustituto con la ecuación conocida de `make_friedman1`.
            """
        ),
    ]
    write_notebook(path, cells)


def build_dbscan_meanshift() -> None:
    path = "02_no_supervisado/20_dbscan_mean_shift.ipynb"
    cells = [
        header(path, "DBSCAN y Mean-Shift", "¿Cómo hallamos estructuras sin fijar el número de grupos?", 4),
        md(
            r"""
            ## Densidad y modos

            DBSCAN declara núcleo a un punto con al menos `min_samples` dentro de una
            bola de radio $\varepsilon$; conecta núcleos y marca como ruido lo no
            alcanzable. Mean-Shift asciende hacia modos de una densidad kernel:
            $m(x)=\sum_i K_h(x_i-x)x_i/\sum_i K_h(x_i-x)-x$.

            Ambos dependen de escala. DBSCAN detecta formas no convexas y ruido;
            Mean-Shift encuentra modos pero puede ser costoso y muy sensible al
            ancho de banda.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
            from sklearn.datasets import make_moons
            from sklearn.metrics import adjusted_rand_score, silhouette_score
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            SEMILLA = 42
            X, y_real = make_moons(n_samples=700, noise=0.085, random_state=SEMILLA)
            rng = np.random.default_rng(SEMILLA)
            ruido = rng.uniform(low=[-1.5, -1.0], high=[2.5, 1.5], size=(55, 2))
            X = np.vstack([X, ruido])
            y_real = np.r_[y_real, np.full(len(ruido), -1)]
            Xs = StandardScaler().fit_transform(X)

            vecinos = NearestNeighbors(n_neighbors=6).fit(Xs)
            distancias, _ = vecinos.kneighbors(Xs)
            kdist = np.sort(distancias[:, -1])
            plt.plot(kdist)
            plt.ylabel("distancia al 6.º vecino")
            plt.xlabel("puntos ordenados")
            plt.title("Ayuda visual para elegir epsilon")
            plt.show()
            """
        ),
        code(
            """
            filas = []
            for eps in np.linspace(0.10, 0.40, 13):
                etiquetas = DBSCAN(eps=eps, min_samples=6).fit_predict(Xs)
                máscara = etiquetas != -1
                n_grupos = len(set(etiquetas)) - (-1 in etiquetas)
                sil = silhouette_score(Xs[máscara], etiquetas[máscara]) if n_grupos > 1 else np.nan
                filas.append(
                    {
                        "eps": eps,
                        "grupos": n_grupos,
                        "ruido": (~máscara).mean(),
                        "silhouette_sin_ruido": sil,
                    }
                )
            tabla = pd.DataFrame(filas)
            display(tabla)
            mejor_eps = tabla.query("grupos >= 2").sort_values("silhouette_sin_ruido").iloc[-1]["eps"]
            etiquetas_db = DBSCAN(eps=mejor_eps, min_samples=6).fit_predict(Xs)
            """
        ),
        code(
            """
            ancho = estimate_bandwidth(Xs, quantile=0.18, n_samples=500, random_state=SEMILLA)
            etiquetas_ms = MeanShift(bandwidth=ancho, bin_seeding=True).fit_predict(Xs)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, etiquetas, título in [
                (axes[0], y_real, "generación conocida (sólo diagnóstico)"),
                (axes[1], etiquetas_db, f"DBSCAN eps={mejor_eps:.2f}"),
                (axes[2], etiquetas_ms, f"Mean-Shift h={ancho:.2f}"),
            ]:
                ax.scatter(Xs[:, 0], Xs[:, 1], c=etiquetas, cmap="tab10", s=14)
                ax.set_title(título)
            plt.tight_layout()
            plt.show()
            print("ARI DBSCAN:", adjusted_rand_score(y_real, etiquetas_db))
            print("ARI Mean-Shift:", adjusted_rand_score(y_real, etiquetas_ms))
            """
        ),
        md(
            """
            ARI usa las etiquetas generadoras y sólo está disponible porque es una
            simulación; silhouette no conoce la verdad, pero tampoco decide qué
            agrupación tiene significado físico.

            **Ejercicios:** cambie unidades de un eje; aumente ruido; estudie la
            estabilidad de cada grupo; explique por qué un único `eps` falla cuando
            la densidad cambia mucho entre regiones.
            """
        ),
    ]
    write_notebook(path, cells)


def build_kmeans_gmm() -> None:
    path = "02_no_supervisado/21_kmeans_gmm.ipynb"
    cells = [
        header(path, "K-means y mezclas gaussianas", "¿Cuándo conviene una asignación dura o probabilística?", 4),
        md(
            r"""
            K-means minimiza $\sum_i\|x_i-\mu_{z_i}\|^2$: supone grupos aproximadamente
            esféricos de varianza similar y produce una etiqueta dura. Una mezcla
            gaussiana modela $p(x)=\sum_k\pi_k\mathcal N(x\mid\mu_k,\Sigma_k)$ y EM
            alterna responsabilidades y parámetros; entrega probabilidades.

            Simularemos tres poblaciones estelares con covarianzas distintas.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score, silhouette_score
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler

            SEMILLA = 42
            rng = np.random.default_rng(SEMILLA)
            parámetros = [
                ([-2, 0], [[0.20, 0.10], [0.10, 1.10]], 260),
                ([1.4, 1.2], [[1.00, -0.55], [-0.55, 0.50]], 330),
                ([2.5, -1.5], [[0.25, 0.0], [0.0, 0.25]], 180),
            ]
            grupos = [rng.multivariate_normal(m, c, n) for m, c, n in parámetros]
            X = np.vstack(grupos)
            y_real = np.concatenate([np.full(len(g), i) for i, g in enumerate(grupos)])
            Xs = StandardScaler().fit_transform(X)
            """
        ),
        code(
            """
            diagnóstico = []
            for k in range(1, 8):
                km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Xs)
                diagnóstico.append(
                    {
                        "k": k,
                        "inercia": km.inertia_,
                        "silhouette": silhouette_score(Xs, km.labels_) if k > 1 else np.nan,
                    }
                )
            diagnóstico = pd.DataFrame(diagnóstico)
            display(diagnóstico)
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].plot(diagnóstico.k, diagnóstico.inercia, "o-")
            axes[0].set(title="Codo", xlabel="k", ylabel="inercia")
            axes[1].plot(diagnóstico.k, diagnóstico.silhouette, "o-")
            axes[1].set(title="Silhouette", xlabel="k", ylabel="score")
            plt.show()
            """
        ),
        code(
            """
            modelos_gmm, criterios = {}, []
            for cov in ["spherical", "diag", "tied", "full"]:
                for k in range(1, 7):
                    gmm = GaussianMixture(
                        n_components=k, covariance_type=cov, n_init=5, random_state=SEMILLA
                    ).fit(Xs)
                    modelos_gmm[(cov, k)] = gmm
                    criterios.append({"covarianza": cov, "k": k, "BIC": gmm.bic(Xs), "AIC": gmm.aic(Xs)})
            criterios = pd.DataFrame(criterios)
            display(criterios.sort_values("BIC").head(8))

            km = KMeans(n_clusters=3, n_init=30, random_state=SEMILLA).fit(Xs)
            clave = criterios.sort_values("BIC").iloc[0][["covarianza", "k"]]
            gmm = modelos_gmm[(clave["covarianza"], int(clave["k"]))]
            prob = gmm.predict_proba(Xs)
            incertidumbre = 1 - prob.max(axis=1)
            print("ARI K-means:", adjusted_rand_score(y_real, km.labels_))
            print("ARI GMM:", adjusted_rand_score(y_real, gmm.predict(Xs)))
            print("Mayor incertidumbre de pertenencia:", incertidumbre.max())
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].scatter(*Xs.T, c=y_real, s=12, cmap="tab10")
            axes[0].set_title("poblaciones simuladas")
            axes[1].scatter(*Xs.T, c=km.labels_, s=12, cmap="tab10")
            axes[1].set_title("K-means")
            im = axes[2].scatter(*Xs.T, c=incertidumbre, s=12, cmap="magma")
            axes[2].set_title("incertidumbre GMM")
            fig.colorbar(im, ax=axes[2])
            plt.show()
            """
        ),
        md(
            """
            **Ejercicios:** compare `full` y `spherical`; transforme una variable a
            otra unidad sin escalar; inyecte una cuarta población pequeña; explique
            por qué BIC, silhouette y significado astrofísico pueden sugerir números
            de grupos distintos.
            """
        ),
    ]
    write_notebook(path, cells)


def build_hierarchical() -> None:
    path = "02_no_supervisado/22_clustering_jerarquico.ipynb"
    cells = [
        header(path, "Clustering aglomerativo jerárquico", "¿Qué estructura multiescala revela un dendrograma?", 4),
        md(
            r"""
            Cada observación empieza como un grupo. En cada paso se fusionan los dos
            grupos más próximos. `single` usa la pareja mínima, `complete` la máxima,
            `average` el promedio y Ward minimiza el aumento de suma de cuadrados.
            Ward requiere distancia euclídea. El dendrograma muestra una **secuencia
            de fusiones**, no una verdad automática sobre cuántos grupos existen.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from scipy.cluster.hierarchy import dendrogram, linkage
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.datasets import make_blobs
            from sklearn.metrics import adjusted_rand_score, silhouette_score
            from sklearn.preprocessing import StandardScaler

            SEMILLA = 42
            X, y_real = make_blobs(
                n_samples=450,
                centers=[(-3, -1), (0, 2), (2.5, -1), (3.5, 2.8)],
                cluster_std=[0.55, 0.9, 0.6, 0.45],
                random_state=SEMILLA,
            )
            Xs = StandardScaler().fit_transform(X)

            muestra = Xs[np.random.default_rng(SEMILLA).choice(len(Xs), 100, replace=False)]
            Z = linkage(muestra, method="ward")
            plt.figure(figsize=(12, 4))
            dendrogram(Z, truncate_mode="lastp", p=25, show_contracted=True)
            plt.ylabel("incremento de distancia Ward")
            plt.title("Dendrograma truncado")
            plt.show()
            """
        ),
        code(
            """
            filas, modelos = [], {}
            for enlace in ["single", "complete", "average", "ward"]:
                for k in range(2, 8):
                    modelo = AgglomerativeClustering(n_clusters=k, linkage=enlace)
                    etiquetas = modelo.fit_predict(Xs)
                    modelos[(enlace, k)] = etiquetas
                    filas.append(
                        {
                            "linkage": enlace,
                            "k": k,
                            "silhouette": silhouette_score(Xs, etiquetas),
                            "ARI_diagnóstico": adjusted_rand_score(y_real, etiquetas),
                        }
                    )
            resultados = pd.DataFrame(filas)
            display(resultados.sort_values("silhouette", ascending=False).head(10))

            mejor = resultados.sort_values("silhouette").iloc[-1]
            etiquetas = modelos[(mejor.linkage, int(mejor.k))]
            plt.scatter(*Xs.T, c=etiquetas, s=18, cmap="tab10")
            plt.title(f"Mejor silhouette: {mejor.linkage}, k={int(mejor.k)}")
            plt.show()
            """
        ),
        md(
            """
            **Ejercicios:** construya a mano las tres primeras fusiones de seis
            puntos; compare el efecto cadena de `single`; use distancia coseno con
            `average`; aplique bootstrap y mida qué pares de observaciones permanecen
            juntos. Esa matriz de coasociación es más informativa que un único corte.
            """
        ),
    ]
    write_notebook(path, cells)


if __name__ == "__main__":
    build_reproducibility()
    build_regression()
    build_trees_forests_bagging()
    build_boosting_stacking()
    build_imbalance()
    build_explainability()
    build_dbscan_meanshift()
    build_kmeans_gmm()
    build_hierarchical()
    print("Notebooks clásicos generados.")
