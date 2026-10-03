# Datos abiertos para extensiones y proyectos

El bloque feedforward usa datos descargables desde las APIs o archivos oficiales.
Las simulaciones se conservan para verificar mecanismos y para dinámica física.
Las descargas requieren Internet en la primera ejecución y no requieren cuenta.

## Datos usados en las prácticas

| Dataset | Tamaño y tarea | Carga y uso | Partición y licencia |
|---|---|---|---|
| [Fashion-MNIST](https://keras.io/api/datasets/fashion_mnist/) | 60 000 + 10 000 imágenes, 28×28, diez clases | `keras.datasets.fashion_mnist.load_data()` en 30 y 50; `torchvision.datasets.FashionMNIST` en 41 | test oficial; validación estratificada dentro de desarrollo; Zalando SE, MIT |
| [Superconductivty Data, UCI 464](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | 21 263 filas, 81 características, regresión de Tc; ZIP ~8 MB | archivo oficial con `train.csv` y `unique_m.csv`, notebook 36 | agrupar por fórmula exacta; Hamidieh (2018), DOI 10.24432/C53P47, CC BY 4.0 |

El 30 tiene modos `clase` y `completo`; ambos preservan el test oficial. El 36
prepara los datos UCI de superconductividad para la alternativa A. La alternativa B
no descarga bases: usa ejemplos y resultados hipotéticos claramente identificados. No comparar puntuaciones entre
notebooks como si hubieran usado idéntico presupuesto. En UCI se imprime SHA256
del ZIP y se comprueba alineación de las temperaturas entre ambos archivos.
Separar por fórmula exacta reduce fuga por material repetido, pero no garantiza
independencia entre familias químicas similares.

## Extensiones científicas

| Fuente | Aplicación posible | Condiciones de uso |
|---|---|---|
| [CERN Open Data](https://opendata.cern.ch/) | clasificación de eventos, jets, Higgs y tracking | empezar por muestras preparadas para ML; distinguir simulación de datos reales |
| [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/competitions/higgs-boson) | desbalance, boosting, calibración y significancia | Kaggle puede requerir cuenta; documentar la métrica física AMS |
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | clustering planetario, regresión y candidatos | el archivo cambia; guardar fecha y consulta exacta |
| [GWOSC](https://gwosc.org/) | series temporales, denoising, CNN/RNN | separar por evento/segmento, no por ventanas solapadas |

## Lista de control

1. URL estable y fecha de consulta.
2. Licencia o términos de uso explícitos.
3. Diccionario de variables, unidades y valores ausentes.
4. Hash o versión del archivo descargado.
5. Descripción de filtros y exclusiones.
6. Partición por objeto, corrida o experimento cuando corresponda.
7. Dataset pequeño de demostración para pruebas; descarga grande bajo demanda.

CERN ofrece conjuntos derivados específicamente para aprendizaje de máquina,
incluidos problemas de etiquetado de eventos con decaimientos de Higgs. Para un
primer proyecto suelen ser más manejables que los formatos completos del detector.
