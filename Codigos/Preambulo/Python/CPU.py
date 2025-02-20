import numpy as np

#No al HARD CODING
RUTA = 'C:/Users/hola-/OneDrive/Documentos/U/Trabajo/UQ/Análisis de datos/Codigos/Clase_3/ejemplo.dat'
#IMPRIMIR DATOS
tipo_dato = [('A', (np.str_, 10)), ('B', np.float64), ('C', np.int32)]
N = np.array([('Ana', 33.3, 1), ('Bob', 44.4, 5), ('Cairne', 66.6, 2),('Dana', 88.8, 20)], dtype=tipo_dato)
np.savetxt(RUTA, N, fmt=['%s' , '%.2f' , '%d'], header="ESTE ES UN ENCABEZADO", footer="ESTO ES UN FOOTER")
print("DONE!")

#LEER DATOS
Variable = np.genfromtxt(RUTA, dtype=tipo_dato, skip_header=1, skip_footer=1)

#Imprimir para corroborar
print(Variable)
print(Variable[0][0])
print(type(Variable[0][0]))