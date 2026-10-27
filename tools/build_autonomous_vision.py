"""Genera los dos notebooks autocontenidos del taller de visión (sin dependencias)."""
import json
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "curso_2026_2/trabajo_autonomo_4h"
OUT.mkdir(parents=True, exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": dedent(text).strip() + "\n"}


def save(name, cells):
    cells = deepcopy(cells)
    url = ("https://colab.research.google.com/github/Santiago-Echeverri-Arteaga/"
           "Fisica_Computacional_2/blob/master/curso_2026_2/trabajo_autonomo_4h/" + name)
    cells.insert(1, md(f"[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n\n"
                       "Si esta versión aún no está publicada, descarga el archivo y usa **Colab → Archivo → Subir notebook**."))
    for i, cell in enumerate(cells):
        cell["id"] = f"autonomo-{i:03d}"
        cell["source"] = cell["source"].splitlines(keepends=True)
    document = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"},
                             "colab": {"name": name}, "fc2": {"tier": "tensorflow-network"}}}
    (OUT / name).write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


setup = code('''
    import gc, json, time, platform
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from PIL import Image, ImageOps, ImageEnhance
    from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

    SEED = 42
    keras.utils.set_random_seed(SEED)
    GPU = bool(tf.config.list_physical_devices("GPU"))
    # Cambia a True si tienes poca memoria o quieres una ejecución de demostración.
    MODO_MINIMO = not GPU
    BATCH = 8 if MODO_MINIMO else 16
    LADO = 96 if MODO_MINIMO else 128
    N_CLASE = 20 if MODO_MINIMO else 100
    EPOCAS = 3 if MODO_MINIMO else 5
    print({"Python": platform.python_version(), "TensorFlow": tf.__version__,
           "GPU": GPU, "modo_minimo": MODO_MINIMO, "lado": LADO,
           "imagenes_por_clase": N_CLASE, "batch": BATCH})
''')

data = code('''
    # Ambos datasets son fotografías reales. La ruta guiada usa flores.
    DATASET = "flores"  # Alternativa del notebook 02: "cifar10"
    if DATASET == "flores":
        archivo = keras.utils.get_file(
            "flower_photos.tgz",
            origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
            extract=True)
        # Keras puede devolver la carpeta extraída o la ruta del archivo según versión.
        base = Path(archivo)
        candidatos = [base / "flower_photos", base,
                      base.parent / "flower_photos", base.parent / "flower_photos_extracted" / "flower_photos"]
        carpeta = next((p for p in candidatos if (p / "daisy").is_dir()), None)
        if carpeta is None:
            raise FileNotFoundError(f"No se encontró flower_photos junto a {base}. Revisa la descarga.")
        nombres = sorted(p.name for p in carpeta.iterdir() if p.is_dir())
        rng = np.random.default_rng(SEED)
        bloques, etiquetas, ids = [], [], []
        for k, nombre in enumerate(nombres):
            rutas = sorted((carpeta / nombre).glob("*.jpg"))
            elegidas = rng.choice(len(rutas), N_CLASE, replace=False)
            for j in elegidas:
                with Image.open(rutas[j]) as im:
                    bloques.append(np.asarray(ImageOps.fit(im.convert("RGB"), (LADO, LADO))))
                etiquetas.append(k)
                ids.append(str(rutas[j].relative_to(carpeta)))
        X = np.stack(bloques)
        y = np.array(etiquetas, dtype="int32")
        print("Créditos de fotografías:", carpeta / "LICENSE.txt")
    elif DATASET == "cifar10":
        # No usamos el test oficial para seleccionar hiperparámetros.
        # Distribución pública de uoft-cs: solo train, sin descargar el test oficial.
        # Evita depender de la velocidad del servidor del archivo original de Toronto.
        from io import BytesIO
        archivo_cifar = keras.utils.get_file(
            "cifar10-train.parquet",
            origin="https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
            file_hash="8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
            hash_algorithm="sha256")
        tabla_cifar = pd.read_parquet(archivo_cifar, engine="pyarrow")
        y_oficial = tabla_cifar["label"].to_numpy(dtype="int32")
        assert len(tabla_cifar) == 50000
        nombres = ["avión", "automóvil", "pájaro", "gato", "ciervo",
                   "perro", "rana", "caballo", "barco", "camión"]
        rng = np.random.default_rng(SEED)
        seleccion = np.concatenate([rng.choice(np.flatnonzero(y_oficial == k), N_CLASE,
                                               replace=False) for k in range(10)])
        # Se conservan las fotos originales 32x32; ampliar no recupera detalle.
        imagenes = []
        for j in seleccion:
            with Image.open(BytesIO(tabla_cifar.iloc[j]["img"]["bytes"])) as im:
                imagenes.append(np.asarray(im.convert("RGB")))
        X, y = np.stack(imagenes), y_oficial[seleccion]
        ids = [f"cifar10/uoft-cs-parquet/train/{i}" for i in seleccion]
        del tabla_cifar, y_oficial
    else:
        raise ValueError("DATASET debe ser 'flores' o 'cifar10'")

    # Separar ANTES de explorar. Muestreo estratificado y sin imágenes compartidas.
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(y))
    idx_train, resto = train_test_split(indices, test_size=0.4, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(resto, test_size=0.5, stratify=y[resto], random_state=SEED)
    assert not (set(idx_train) & set(idx_val) or set(idx_train) & set(idx_test) or set(idx_val) & set(idx_test))
    K = len(nombres)
    print("Forma, tipo, rango:", X.shape, X.dtype, (X.min(), X.max()))
    print(pd.DataFrame({nombre: np.bincount(y[idx], minlength=K)
                        for nombre, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]}, index=nombres))
    # Registro para comprobar que las comparaciones usan exactamente los mismos datos.
    Path("particiones.json").write_text(json.dumps(
        {n: [ids[i] for i in ix] for n, ix in [("train", idx_train), ("val", idx_val), ("test", idx_test)]},
        indent=2), encoding="utf-8")
''')

pipeline = code('''
    def dataset(indices, entrenamiento=False):
        ds = tf.data.Dataset.from_tensor_slices((X[indices], y[indices]))
        if entrenamiento:
            ds = ds.shuffle(len(indices), seed=SEED)
        # Se redimensiona por lote, sin guardar un dataset ampliado completo en RAM.
        return ds.batch(BATCH).map(
            lambda x, y: (tf.image.resize(tf.cast(x, tf.float32), (LADO, LADO)), y),
            num_parallel_calls=1).prefetch(1)

    train_ds = dataset(idx_train, True)
    val_ds = dataset(idx_val)
    test_ds = dataset(idx_test)

    def compilar(modelo):
        modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def parada():
        return keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
''')

intro = '''
    **Requisitos:** teoría matemática de redes ya vista; no necesitas escribir un programa desde cero.
    Empieza ejecutando los ejemplos completos y cambia únicamente los parámetros señalados.
    **Cómo leer el código:** `=` asigna un valor; `#` inicia un comentario; `nombre(...)` llama
    una función; `objeto.metodo(...)` aplica una operación a un objeto. `X.shape` informa
    dimensiones; `for` repite operaciones y `def` define una función que se usará después.
    Por ejemplo, `LADO = 96` elige resolución y `foto.convert("RGB")` devuelve una imagen RGB.
    No necesitas memorizar las API: relaciona entrada, operación y salida con la teoría.

    **Primer uso de Colab:** pulsa el triángulo de cada celda de código, espera a que termine
    (inicia sesión con tu cuenta de Google para conectar el entorno)
    y continúa hacia abajo. El texto explicativo no se ejecuta. Si aparece `NameError`,
    ejecuta las celdas anteriores; después de desconectarte usa Ejecutar todo para reconstruir
    las variables. Guarda una copia desde el menú Archivo si quieres conservar tus cambios.
    No hace falta montar Drive en el código. Las tablas y figuras aparecen debajo de las celdas.
    En tu primera práctica detente en el reto, compara con validación y solo entonces
    continúa al test final. “Ejecutar todo” comprueba el ejemplo completo, pero no sustituye
    esa comparación ni autoriza a elegir cambios mirando resultados de test.
    Usa Colab con Python; GPU T4 es opcional (Entorno de ejecución → Cambiar tipo de entorno).
    No se necesita Drive, Kaggle, claves ni pago. Colab incluye las bibliotecas importadas;
    no actualices TensorFlow/NumPy durante la sesión. Si falta una biblioteca, vuelve a un entorno Colab estándar.

    **Presupuesto:** descarga de flores ≈218 MB; CIFAR-10 alternativo ≈120 MB (train en Parquet).
    El subconjunto reduce entrenamiento y RAM, no el tamaño de la descarga inicial.
    La separación por archivos es un benchmark didáctico: no tenemos identificadores de planta
    o jardín para garantizar independencia por origen. No extrapoles sus métricas a jardines nuevos.
    El modo CPU usa 20 imágenes por clase (12/4/4 en train/val/test): sirve para aprender el flujo,
    no para concluir qué arquitectura es mejor. GPU usa 100 (60/20/20).
    Los tiempos de la guía incluyen lectura y retos; no son tiempos de ejecución medidos.
    Colab gratuito no garantiza GPU ni duración. Si la sesión se corta, guarda tu notebook y vuelve
    a ejecutar desde el inicio; las salidas guardadas no restauran variables.
'''

save("01_pillow_opencv_cnn.ipynb", [
    md('''
    # 01 · De la fotografía a una CNN: Pillow y OpenCV
    **Trabajo autónomo: 120 minutos.** Al terminar podrás leer una foto, explicar sus canales,
    aplicar transformaciones y filtros, entrenar una CNN pequeña y analizar errores.

    | Minutos | Actividad |
    |---|---|
    | 0–15 | Entorno, descarga y particiones |
    | 15–40 | Pillow: recortar, redimensionar, rotar y guardar |
    | 40–65 | OpenCV: color, filtros, bordes y segmentación |
    | 65–95 | Entrenar y evaluar la CNN |
    | 95–120 | Un reto, conclusiones y entrega |
    ''') , md(intro), setup,
    md('''
    ## 1. Datos: fotografías de cinco clases de flores
    Usamos la colección pública del tutorial de TensorFlow. Cada carpeta aporta su etiqueta.
    El recorte centrado conserva proporciones pero puede cortar partes de la flor.
    `X` contiene píxeles RGB `uint8` de 0 a 255; `y` contiene números de clase.
    Train ajusta pesos, validación orienta decisiones y test se reserva para el resultado final.
    ''') , data,
    md('''
    ## 2. Pillow: imagen, arreglo y coordenadas
    Una imagen Pillow informa `(ancho, alto)`; NumPy usa `(alto, ancho, canales)`.
    `crop` recibe `(izquierda, arriba, derecha, abajo)`, con origen arriba a la izquierda.
    Antes de ejecutar, predice el tamaño del recorte. Cambiar brillo modifica intensidades;
    redimensionar interpola posiciones. Ninguna operación crea información que no estaba en la foto.
    ''') , code('''
    import cv2
    # Abrimos una foto original de entrenamiento, no una imagen de test.
    if DATASET == "flores":
        with Image.open(carpeta / ids[idx_train[0]]) as original:
            foto = ImageOps.exif_transpose(original).convert("RGB")
    else:
        foto = Image.fromarray(X[idx_train[0]])
    ancho, alto = foto.size
    recorte = foto.crop((ancho//4, alto//4, 3*ancho//4, 3*alto//4))
    pequena = ImageOps.fit(foto, (160, 160), method=Image.Resampling.LANCZOS)
    giro = pequena.rotate(25, resample=Image.Resampling.BICUBIC)
    brillo = ImageEnhance.Brightness(pequena).enhance(1.4)
    pequena.save("flor_rgb.png")
    print("Pillow:", foto.size, foto.mode, "NumPy:", np.asarray(foto).shape)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, im, titulo in zip(axes, [foto, recorte, pequena, giro, brillo],
                             ["Original", "Recorte", "160 × 160", "Giro 25°", "Brillo ×1.4"]):
        ax.imshow(im); ax.set_title(titulo); ax.axis("off")
    plt.tight_layout(); plt.show()
    assert np.asarray(pequena).shape == (160, 160, 3)
    ''') , md('''
    **Comprueba tu comprensión:** ¿por qué aparecen esquinas negras al girar? ¿Qué diferencia
    hay entre `resize` (puede deformar) e `ImageOps.fit` (recorta)? Prueba `ImageOps.pad` para
    conservar toda la fotografía. Predice primero qué cambiará y anota lo observado.

    ## 3. OpenCV: RGB ≠ BGR
    `cv2.imread` devuelve BGR; Matplotlib espera RGB. Una conversión incorrecta intercambia
    rojo y azul sin lanzar un error. En HSV, OpenCV con `uint8` usa H entre 0 y 179, S/V entre 0 y 255.
    El filtro gaussiano suaviza ruido; Canny localiza cambios fuertes de intensidad.
    Una máscara de color es una regla manual, no una red entrenada ni una segmentación perfecta.
    ''') , code('''
    bgr = cv2.imread("flor_rgb.png")
    assert bgr is not None, "Ejecuta la celda que guarda flor_rgb.png"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(suavizada, 60, 140)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array([20, 70, 50]), np.array([40, 255, 255]))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    segmento = cv2.bitwise_and(rgb, rgb, mask=mascara)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, im, titulo in zip(axes.flat, [bgr, rgb, gris, bordes, mascara, segmento],
                             ["BGR mal visualizado", "RGB correcto", "Grises", "Canny", "Máscara amarilla", "Selección"]):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(titulo); ax.axis("off")
    plt.tight_layout(); plt.show()
    assert np.array_equal(rgb, np.asarray(pequena))
    print("Fracción seleccionada:", (mascara > 0).mean())
    ''') , md('''
    **Pausa:** si la máscara está vacía, comprueba si tu foto contiene amarillo. Un resultado
    vacío puede ser correcto. Cambia umbrales, prueba otra foto de train y describe dos fallos
    de esta regla (iluminación, fondo, sombras…). No elijas umbrales mirando test.

    ## 4. CNN: filtros aprendidos
    OpenCV aplicó filtros definidos por nosotros. `Conv2D` aprende sus coeficientes por gradiente.
    Para stride 1 y padding `same`, una convolución conserva alto y ancho; pooling los reduce.
    Una capa con C entradas y F filtros 3×3 tiene `(3×3×C + 1)×F` parámetros con sesgo.
    GlobalAveragePooling promedia cada mapa; Dense produce probabilidades de clase.
    La normalización a [0,1] está dentro del modelo; no la repitas en el dataset.
    ''') , pipeline, code('''
    keras.utils.set_random_seed(SEED)
    FILTROS_INICIALES = 16  # Reto: cambia solo este 16 por 8 y reejecuta esta celda.
    train_ds = dataset(idx_train, True)  # Reinicia también el orden de mezcla para el reto.
    cnn = keras.Sequential([
        layers.Input((LADO, LADO, 3)),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal", seed=SEED),
        layers.Conv2D(FILTROS_INICIALES, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2), layers.Dense(K, activation="softmax")], name="cnn_pequena")
    cnn.summary()
    compilar(cnn)
    inicio = time.perf_counter()
    historia = cnn.fit(train_ds, validation_data=val_ds, epochs=EPOCAS,
                       callbacks=[parada()], verbose=2)
    print("Segundos de entrenamiento:", round(time.perf_counter() - inicio, 1))
    pd.DataFrame(historia.history)[["loss", "val_loss"]].plot(title="Pérdida: train y validación")
    plt.show()
    print("Referencia de accuracy por clase mayoritaria:", np.bincount(y[idx_train]).max()/len(idx_train))
    print("Para tu tabla: filtros =", FILTROS_INICIALES,
          "val_accuracy restaurada =", cnn.evaluate(val_ds, verbose=0)[1],
          "parámetros =", cnn.count_params())
    ''') , md('''
    ## 5. Reto obligatorio: cambia una sola decisión (15 min)
    Elige **antes del test**: (A) comparar 8 y 16 filtros en la primera capa, o (B) comparar
    aumento horizontal activado/desactivado. Conserva particiones, semilla y épocas.
    Reejecuta la celda de construcción para entrenar desde cero. Registra ambos resultados
    de validación y parámetros. Deja en `cnn` la versión elegida por validación.
    No se exige una accuracy mínima: explica por qué el resultado mejora, empeora o es inconcluso.
    **Ejemplo de registro (inventado):** si la salida dice filtros=16, val_accuracy=0.35,
    parámetros=24000, escribe esos datos en la fila base. Repite con 8 y registra la nueva salida.
    Sustituye `None` por tus números, sin comillas. No copies los números del ejemplo.

    **Retos extra fuera de las cuatro horas:** comparar imágenes grises replicadas en tres
    canales; probar `cv2.filter2D` con Sobel; sustituir el recorte por padding y estudiar los errores.
    ''') , code('''
    # Completa manualmente tus observaciones sin inventar métricas.
    registro_reto = [
        {"variante": "base", "val_accuracy": None, "parametros": None},
        {"variante": "mi_cambio", "val_accuracy": None, "parametros": None},
    ]
    display(pd.DataFrame(registro_reto))
    ''') , md('''
    ## 6. Test final y errores (10 min)
    Ejecuta cuando hayas terminado de elegir. Macro-F1 promedia F1 dando el mismo peso a cada
    clase. Con cuatro fotos de test por clase, una sola predicción cambia mucho las métricas.
    Analiza errores sin volver a ajustar el modelo con ese test.
    ''') , code('''
    probabilidades = cnn.predict(test_ds, verbose=0)
    pred = probabilidades.argmax(axis=1)
    metricas = {"accuracy": float(accuracy_score(y[idx_test], pred)),
                "macro_f1": float(f1_score(y[idx_test], pred, average="macro", zero_division=0)),
                "parametros": cnn.count_params(), "dataset": DATASET, "semilla": SEED}
    print(metricas)
    ConfusionMatrixDisplay.from_predictions(y[idx_test], pred, labels=np.arange(K),
                                            display_labels=nombres, xticks_rotation=45)
    plt.show()
    errores = np.flatnonzero(pred != y[idx_test])[:6]
    if len(errores):
        fig, axes = plt.subplots(1, len(errores), figsize=(3*len(errores), 3), squeeze=False)
        for ax, j in zip(axes.flat, errores):
            ax.imshow(X[idx_test[j]])
            ax.set_title(f"Real: {nombres[y[idx_test[j]]]}\\nPred: {nombres[pred[j]]}"); ax.axis("off")
        plt.tight_layout(); plt.show()
    Path("metricas_cnn.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    ''') , md('''
    ## Entrega y autoevaluación
    Guarda una copia del notebook con salidas. Incluye el mosaico Pillow/OpenCV, la tabla del reto,
    curva de pérdida y matriz de confusión. Escribe cinco líneas: RGB/BGR, efecto del filtro,
    diferencia train/validación/test, resultado del reto y una limitación del conjunto pequeño.
    Descarga `metricas_cnn.json` y `particiones.json` desde el panel Archivos de Colab.


    Fuentes: [Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html),
    [OpenCV](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html),
    [flores y atribución CC-BY](https://www.tensorflow.org/tutorials/load_data/images),
    [Colab gratuito](https://research.google.com/colaboratory/faq.html).
    ''')
])

save("02_transfer_inception_resnet152_personalizadas.ipynb", [
    md('''
    # 02 · Transfer learning y una red Inception-Residual construida a mano
    **Trabajo autónomo: 120 minutos.** Notebook independiente: no requiere variables del 01.
    Compararás InceptionV3 y **ResNet152** preentrenadas en ImageNet y construirás una red pequeña
    inspirada en ramas multiescala y conexiones residuales. Esta última se entrena desde cero;
    no es una reproducción de InceptionResNetV2 ni tiene sus pesos.

    | Minutos | Actividad |
    |---|---|
    | 0–15 | Preparar datos y entender transferencia |
    | 15–55 | Extraer características con InceptionV3 y ResNet152 |
    | 55–85 | Construir y entrenar la red personalizada |
    | 85–110 | Reto controlado y evaluación final |
    | 110–120 | Interpretación y entrega |
    ''') , md(intro + '''
    Los pesos de InceptionV3 y ResNet152 sin cabeza descargan aproximadamente 85 y 230 MB,
    además del dataset. Solo hay una base grande viva a la vez; lote 8/16, imágenes 96/128,
    extracción una vez por modelo, y cabezas pequeñas entrenadas sobre características guardadas.
    No ejecutamos fine-tuning de ResNet152 en la ruta de cuatro horas.
    Si faltan recursos, activa `MODO_MINIMO=True` en la configuración y reinicia desde arriba.
    ''') , setup,
    md('''
    ## 1. El mismo problema, distintas representaciones
    Deja `DATASET="flores"` para la ruta de cuatro horas. Como extensión, cambia a `"cifar10"`
    y ejecuta de nuevo todo el notebook: son fotos reales de objetos de 32×32, diez clases.
    La segunda base es una alternativa ejecutable, no una segunda corrida obligatoria.
    No compares accuracies entre bases como si midieran la misma dificultad.

    Una base congelada transforma imagen → vector de características. Solo la nueva capa Dense
    aprende las clases actuales. Usamos `include_top=False` y promedio global; así admitimos
    tamaños menores que 299 (Inception) y 224 (ResNet), a costa de perder detalle.
    InceptionV3 admite mínimo 75×75, por lo que 96×96 es válido.
    ''') , data, pipeline,
    md('''
    ## 2. Preprocesamiento: una diferencia decisiva
    InceptionV3 espera valores transformados a [-1,1]. ResNet152 espera RGB convertido a BGR
    y centrado con medias de ImageNet. Las funciones oficiales hacen esas operaciones.
    Alimentamos ambas con RGB float32 de 0 a 255: **no dividir entre 255 antes**.
    Una cabeza es un clasificador lineal softmax sobre características no lineales aprendidas.
    Guardamos las características para entrenar varias épocas sin repetir la base grande.
    ''') , code('''
    especificaciones = {
        "InceptionV3": (keras.applications.InceptionV3, keras.applications.inception_v3.preprocess_input),
        "ResNet152": (keras.applications.ResNet152, keras.applications.resnet.preprocess_input),
    }
    resultados = []
    predicciones_test = {}  # Se consultan solo en la evaluación final.
    for nombre, (constructor, preprocesar) in especificaciones.items():
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        inicio = time.perf_counter()
        base = constructor(weights="imagenet", include_top=False,
                           input_shape=(LADO, LADO, 3), pooling="avg")
        base.trainable = False

        def extraer(indices):
            partes = []
            for imagenes, _ in dataset(indices):
                partes.append(base(preprocesar(imagenes), training=False).numpy())
            return np.concatenate(partes)

        f_train, f_val, f_test = (extraer(ix) for ix in (idx_train, idx_val, idx_test))
        parametros_base = base.count_params()
        segundos_extraccion = time.perf_counter() - inicio
        print(nombre, "características:", f_train.shape, "segundos:", round(segundos_extraccion, 1))
        del base
        gc.collect()
        keras.backend.clear_session()
        cabeza = keras.Sequential([layers.Input((f_train.shape[1],)), layers.Dense(K, activation="softmax")])
        compilar(cabeza)
        inicio_cabeza = time.perf_counter()
        h = cabeza.fit(f_train, y[idx_train], validation_data=(f_val, y[idx_val]),
                        batch_size=BATCH, epochs=10, callbacks=[parada()], verbose=2)
        p_val = cabeza.predict(f_val, verbose=0).argmax(axis=1)
        predicciones_test[nombre] = cabeza.predict(f_test, verbose=0).argmax(axis=1)
        resultados.append({"modelo": nombre, "val_accuracy": accuracy_score(y[idx_val], p_val),
                           "val_macro_f1": f1_score(y[idx_val], p_val, average="macro", zero_division=0),
                           "parametros_totales": parametros_base + cabeza.count_params(),
                           "parametros_entrenados": cabeza.count_params(),
                           "segundos_extraccion": segundos_extraccion,
                           "segundos_ajuste": time.perf_counter() - inicio_cabeza,
                           "epocas": len(h.history["loss"])})
        del cabeza, f_train, f_val, f_test
        gc.collect()
    display(pd.DataFrame(resultados))
    ''') , md('''
    **Interpreta:** ¿por qué ResNet152 tiene millones de parámetros pero entrenamos pocos miles?
    ¿Cómo cambia la comparación si cuentas tiempo de descarga? El tiempo de extracción incluye
    construcción y descarga de pesos si no estaban en caché; no es un benchmark puro.
    La mayor arquitectura no tiene por qué ganar con pocos datos y resolución reducida.

    ## 3. Diseñar a mano: ramas + conexión residual
    Inception procesa varias escalas en paralelo y concatena canales. Una proyección 1×1 reduce
    canales antes de filtros costosos. ResNet suma una transformación a un atajo: `relu(x + F(x))`.
    Para sumar, ambas formas deben coincidir. Proyectamos el atajo con 1×1 si cambia el número de canales.
    Aquí usamos cuatro ramas (1×1, 3×3, dos 3×3 y pooling), una proyección y una suma residual.
    Dos filtros 3×3 consecutivos amplían el campo receptivo e introducen una no linealidad intermedia.
    ''') , code('''
    def bloque_inception_residual(x, filtros=8, residual=True):
        rama1 = layers.Conv2D(filtros, 1, padding="same", activation="relu")(x)
        rama3 = layers.Conv2D(filtros, 1, padding="same", activation="relu")(x)
        rama3 = layers.Conv2D(filtros, 3, padding="same", activation="relu")(rama3)
        rama5 = layers.Conv2D(filtros, 1, padding="same", activation="relu")(x)
        rama5 = layers.Conv2D(filtros, 3, padding="same", activation="relu")(rama5)
        rama5 = layers.Conv2D(filtros, 3, padding="same", activation="relu")(rama5)
        rama_pool = layers.MaxPooling2D(3, strides=1, padding="same")(x)
        rama_pool = layers.Conv2D(filtros, 1, padding="same", activation="relu")(rama_pool)
        union = layers.Concatenate()([rama1, rama3, rama5, rama_pool])
        salida = layers.Conv2D(2*filtros, 1, padding="same")(union)
        if residual:
            atajo = x if int(x.shape[-1]) == 2*filtros else layers.Conv2D(2*filtros, 1)(x)
            salida = layers.Add()([atajo, salida])
        return layers.Activation("relu")(salida)

    def crear_personalizada(filtros=8, residual=True):
        entrada = keras.Input((LADO, LADO, 3))
        x = layers.Rescaling(1./255)(entrada)
        x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
        x = bloque_inception_residual(x, filtros, residual)
        x = layers.MaxPooling2D()(x)
        x = bloque_inception_residual(x, 2*filtros, residual)
        x = layers.GlobalAveragePooling2D()(x)
        salida = layers.Dense(K, activation="softmax")(x)
        return keras.Model(entrada, salida, name="mini_inception_residual")

    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    USAR_ATAJO = True  # Reto: cambia solo True por False y reejecuta esta celda.
    train_ds = dataset(idx_train, True)
    personalizada = crear_personalizada(filtros=8, residual=USAR_ATAJO)
    personalizada.summary()
    assert personalizada.output_shape == (None, K)
    assert personalizada.count_params() < 200_000
    compilar(personalizada)
    inicio = time.perf_counter()
    h_custom = personalizada.fit(train_ds, validation_data=val_ds, epochs=EPOCAS,
                                  callbacks=[parada()], verbose=2)
    tiempo_custom = time.perf_counter() - inicio
    pd.DataFrame(h_custom.history)[["loss", "val_loss"]].plot(title="Red propia: pérdida")
    plt.show()
    print("Para tu tabla: atajo =", USAR_ATAJO,
          "val_accuracy restaurada =", personalizada.evaluate(val_ds, verbose=0)[1],
          "parámetros =", personalizada.count_params())
    ''') , md('''
    ## 4. Reto de arquitectura: experimento controlado (25 min)
    Formula una hipótesis sobre el atajo y compara `residual=True` con `False`, manteniendo
    filtros, particiones, épocas y semilla. Cambia el argumento en la celda anterior, reconstruye
    el modelo y registra ambas métricas de validación. Elige con validación; deja el modelo elegido
    en `personalizada`. Explica por qué quitar el atajo no implica necesariamente empeorar en 3–5 épocas.
    **Paso concreto:** ejecuta con `USAR_ATAJO=True`, anota la línea final “Para tu tabla”, cambia
    a `False` y ejecuta de nuevo solo la celda de construcción/entrenamiento. Rellena los `None`
    de la tabla con esos números. Si eliges la primera versión, vuelve a entrenarla con `True`
    antes de ejecutar el test. No hace falta editar la función que define las ramas.

    Alternativa: conserva el atajo y cambia `filtros=8` a `12`. Antes de ejecutar predice cómo
    cambiarán parámetros, tiempo y riesgo de sobreajuste. No cambies ambos factores simultáneamente.
    ''') , code('''
    _, val_acc_custom = personalizada.evaluate(val_ds, verbose=0)
    print("Validación de la variante actual:", val_acc_custom,
          "Parámetros:", personalizada.count_params())
    # Completa con las mediciones de tus dos corridas; ninguna respuesta está prefijada.
    tabla_reto = pd.DataFrame([
        {"variante": "con atajo", "val_accuracy": None, "parametros": None},
        {"variante": "sin atajo", "val_accuracy": None, "parametros": None}])
    display(tabla_reto)
    ''') , md('''
    ## 5. Comparación final: abrir test una sola vez
    Congela tus decisiones. Reportamos los tres sistemas, seleccionando el candidato por
    **macro-F1 de validación** antes de mostrar test. Las predicciones de test de las bases se
    calcularon sin entrenar con ellas. No las uses después para modificar hiperparámetros.
    La red propia no recibió ImageNet: la tabla compara sistemas con distinta información previa,
    no demuestra la superioridad causal de una arquitectura. El test pequeño tiene gran incertidumbre.
    ''') , code('''
    p_val_custom = personalizada.predict(val_ds, verbose=0).argmax(axis=1)
    predicciones_test["Personalizada"] = personalizada.predict(test_ds, verbose=0).argmax(axis=1)
    filas = resultados + [{"modelo": "Personalizada",
        "val_accuracy": accuracy_score(y[idx_val], p_val_custom),
        "val_macro_f1": f1_score(y[idx_val], p_val_custom, average="macro", zero_division=0),
        "parametros_totales": personalizada.count_params(),
        "parametros_entrenados": personalizada.count_params(),
        "segundos_extraccion": 0., "segundos_ajuste": tiempo_custom,
        "epocas": len(h_custom.history["loss"])}]
    tabla = pd.DataFrame(filas).set_index("modelo")
    elegido = tabla["val_macro_f1"].idxmax()
    print("Elegido por validación (empates: primero de la tabla):", elegido)
    for nombre, pred in predicciones_test.items():
        tabla.loc[nombre, "test_accuracy"] = accuracy_score(y[idx_test], pred)
        tabla.loc[nombre, "test_macro_f1"] = f1_score(y[idx_test], pred, average="macro", zero_division=0)
    display(tabla)
    ConfusionMatrixDisplay.from_predictions(y[idx_test], predicciones_test[elegido],
                                            labels=np.arange(K), display_labels=nombres, xticks_rotation=45)
    plt.show()
    tabla.to_csv("comparacion_transfer.csv")
    Path("configuracion.json").write_text(json.dumps({"dataset": DATASET, "seed": SEED,
        "lado": LADO, "n_clase": N_CLASE, "gpu": GPU, "tensorflow": tf.__version__,
        "seleccion_validacion": elegido}, indent=2), encoding="utf-8")
    ''') , md('''
    ## Entrega y autoevaluación (10 min)
    Entrega el notebook con salidas, `comparacion_transfer.csv`, `configuracion.json` y
    `particiones.json`. Incluye tu hipótesis, tabla del reto y 150 palabras de conclusión:
    diferencia entre pesos totales y entrenados; concatenación frente a suma; selección sin fuga;
    precisión frente a costo; limitaciones del subconjunto y del preentrenamiento.


    **Extensiones fuera de las cuatro horas:** repetir con CIFAR-10; estudiar variación con tres
    semillas; implementar un bloque de reducción; explorar fine-tuning de las últimas capas de
    Inception con learning rate bajo. Al descongelar hay que recompilar; mantener BatchNorm en
    inferencia. No actives estas extensiones para completar la ruta mínima.

    Fuentes: [guía oficial de transferencia](https://keras.io/guides/transfer_learning/),
    [aplicaciones y arquitecturas](https://keras.io/api/applications/),
    [ResNet152 y preprocesamiento](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152),
    [CIFAR-10 y autores](https://www.cs.toronto.edu/~kriz/cifar.html),
    [distribución CIFAR-10 utilizada](https://huggingface.co/datasets/uoft-cs/cifar10),
    [flores y licencias](https://www.tensorflow.org/tutorials/load_data/images),
    [límites variables de Colab](https://research.google.com/colaboratory/faq.html).
    ''')
])

print("Generados dos notebooks en", OUT)
