"""Orientación específica de los notebooks de apoyo y de las unidades posteriores."""

ROUTES = {
    '32': ('Apoyo: verificar el entrenamiento', 'La teoría feedforward ya es prerrequisito. '
           'Use este notebook para contrastar la derivada manual con GradientTape y autograd del 30. '
           'Producto: tabla de errores entre derivadas y explicación de las formas matriciales. '
           'La señal sintética permite conocer la respuesta exacta; el trabajo con datos reales '
           'se realiza en 30 y 36. No es otra sesión obligatoria antes del taller.'),
    '33': ('Profundización posterior al taller', 'SGD y Adam ya aparecen operativamente en el 30. '
           'Aquí se estudia por qué se comportan de manera diferente. Después del ejemplo controlado, '
           'compare dos tasas en el caso UCI del 36 con particiones fijas y MAE de validación. '
           'Producto: curvas y coste de cuatro entrenamientos como máximo. Dropout y penalizaciones '
           'son extensión posterior; no son requisitos del miércoles.'),
    '40': ('Unidad posterior: de MLP a filtros', 'Comience recuperando los errores entre prendas '
           'del notebook 30. Los dígitos pequeños permiten verificar una correlación a mano. '
           'Después use una imagen Fashion-MNIST del 41 y compare Sobel y Laplaciano. '
           'Producto: tamaños de salida, mapas y relación entre el Laplaciano discreto y un campo '
           'escalar. Esta unidad se estudia después del taller feedforward.'),
    '41': ('Unidad posterior: benchmark con imágenes', 'Use Fashion-MNIST para continuar el problema '
           'del 30. Para comparar MLP y CNN vuelva a entrenar ambas con los mismos índices y '
           'presupuesto; no compare directamente puntuaciones de muestras diferentes. Registre '
           'accuracy, F1 macro, parámetros, tiempo y confusiones camisa/camiseta. La localidad '
           'de una imagen de detector motiva filtros compartidos, pero no convierte estas prendas '
           'en datos de un experimento físico.'),
    '42': ('Unidad posterior: arquitecturas y coste', 'Esta unidad inspecciona bloques; no entrena '
           'ImageNet. Producto: tabla de formas, parámetros y una explicación del camino del '
           'gradiente. Si se hace una comparación empírica, elija sólo un bloque y añádalo al '
           'clasificador Fashion-MNIST del 41 con el mismo protocolo. No es necesario entrenar '
           'todas las arquitecturas para entender sus diferencias.'),
    '43': ('Unidad posterior: transferencia', 'CIFAR-10 es el benchmark descargable del ejemplo. '
           'Separe el efecto de congelar la base del efecto de cambiar resolución o presupuesto. '
           'Producto: validación antes/después del ajuste fino, test final y tiempo. La transferencia '
           'a imágenes científicas exige comprobar canales, unidades, licencia y separación por '
           'objeto físico; no basta con cambiar las etiquetas de CIFAR-10.'),
    '50': ('Unidad posterior: representación', 'Fashion-MNIST permite pasar de clasificación a '
           'reconstrucción con las mismas observaciones. Compare error de validación para dos '
           'dimensiones latentes, después abra test. Producto: reconstrucciones y error por clase. '
           'Una anomalía de reconstrucción es una desviación estadística, no evidencia suficiente '
           'de una anomalía física del sensor.'),
    '51': ('Unidad posterior: dinámica generativa', 'El conjunto pequeño de dígitos mantiene visible '
           'la dinámica adversarial. Una extensión opcional usa Fashion-MNIST: cambie entrada/salida '
           'de 64 a 784, visualización de 8×8 a 28×28 y empiece con 6 000 imágenes y 10 épocas. '
           'Producto: muestras a ruido fijo y evidencia de diversidad o colapso. No interprete '
           'una pérdida decreciente como garantía de fidelidad; generar datos sintéticos tampoco '
           'aumenta por sí solo la información experimental.'),
    '60': ('Unidad posterior: orden temporal', 'Las señales amortiguadas mantienen un mecanismo '
           'físico conocido. Seleccione arquitectura por validación. Añada como referencias '
           'persistencia y un MLP sobre la ventana aplanada; así se mide qué aporta la recurrencia. '
           'Producto: MSE y error frente al horizonte. Para series reales de UCI, divida '
           'cronológicamente antes de generar ventanas; no mezcle ventanas vecinas al azar.'),
    '61': ('Unidad posterior: comprobar atención', 'La inversión de secuencias es una tarea de '
           'control con alineamiento conocido. Producto: exactitud de secuencia completa y mapa '
           'de atención en ejemplos no vistos. Esta simulación prueba el mecanismo, no su '
           'rendimiento sobre textos reales. Sólo después de dominarla tiene sentido trabajar '
           'con pares de secuencias de mediciones; respete la separación por experimento.'),
    '62': ('Unidad posterior: predicción física', 'Lorenz conserva el análisis de dinámica y '
           'sensibilidad a condiciones iniciales. Compare persistencia, Ridge, MLP y LSTM con '
           'idéntica ventana y partición temporal. Producto: error de un paso y de rollout; '
           'no confunda interpolación local con predicción fiable a largo plazo. La simulación '
           'tiene una función pedagógica explícita y no se presenta como observación real.'),
    '70': ('Panorama posterior: decisiones', 'Q-learning introduce interacción con un entorno; '
           'no es requisito de los talleres de datos supervisados. Producto: política, retorno '
           'y comparación con decisiones aleatorias usando semillas distintas de entrenamiento. '
           'El entorno controlado permite conocer recompensas y transiciones; no hace falta '
           'importar una tabla UCI para una tarea de decisión secuencial.'),
    '71': ('Panorama posterior: atención causal', 'El corpus diminuto permite inspeccionar formas '
           'y máscara causal. Producto: pérdida en texto reservado y ejemplos de generación con '
           'sus límites; no interpretar memorización como comprensión física. El entrenamiento '
           'de un modelo grande no forma parte de esta unidad ni del taller feedforward.'),
    '80': ('Proyecto: datos y evidencia', 'Priorice una fuente pública con descarga reproducible '
           'y un presupuesto que permita iterar. UCI superconductividad ofrece un punto de partida '
           'físico; Fashion-MNIST y CIFAR-10 permiten comparar representaciones. Distinga datos '
           'observados, simulados y derivados. Incluya una línea base clásica y una MLP antes '
           'de justificar una arquitectura más avanzada; no se exige que lo más complejo gane.'),
}


def route_for(path):
    from pathlib import PurePosixPath
    return ROUTES.get(PurePosixPath(path).name[:2])
