# Datos abiertos para extensiones y proyectos

Los notebooks base evitan descargas para que la clase sea estable. Los proyectos
pueden sustituir las simulaciones por datos abiertos después de comprobar licencia,
tamaño, esquema y unidad correcta de partición.

| Fuente | Aplicación posible | Precaución docente |
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
