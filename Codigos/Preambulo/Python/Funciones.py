from math import sqrt
def distancia(dimensiones:int,*coordenadas, metrica="pitagoras", **config):
    '''Programa que recibe coordenadas de dos vectores de dimension arbitraria y calcula la distancia por
    alguna metrica seleccionada

    dimensiones: int
    Cantidad de dimensiones que tienen todos los vectores. Todos deben tener la misma dimensión
    metrica: str
    Metrica usada en el programa
    *coordenadas
    Coordenadas escritas como flotantes uno después de otro de forma. Deben darse dimensiones*2 datos
    **config
    Criterios de configuracion adicionales
    
    >>>distancia(2,werqwqihfihqe)
    12
    '''
    #Verifica que el dato dimensiones sea un entero positivo
    try:
        dimensiones=int(dimensiones)
    except:
        raise Exception("Tipo de dato no válido, se requiere un entero positivo")
    else:
        if dimensiones<0:
            raise Exception("Tipo de dato no válido, se requiere un entero positivo")

    #Verifica que se tengan la cantidad correcta de coordenadas
    if len(coordenadas)%dimensiones:
        raise Exception("Coordenadas insuficientes: {} coordenadas representando N vectores en \
un espacio de dimensión {}".format(len(coordenadas),dimensiones))
    else:
        print("Comprobación de errores completada con éxito") #este else está para remover

    #Se selecciona la métrica
    if metrica.lower()=="pitagoras":
        #Si es métrica de pitagoras
        suma=0
        for i in range(dimensiones):
            suma+=(coordenadas[i]-coordenadas[i+dimensiones])**2
        return(sqrt(suma))
    else:
        raise Exception("Esa métrica todavía no está implementada")

    #TODO: Implementar más métricas
    #TODO: Añadir opciones de configuración

print(distancia(2,10,20,-1,2,metrica="PITAGORAS"))