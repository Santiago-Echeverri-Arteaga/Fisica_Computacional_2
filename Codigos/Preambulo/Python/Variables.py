def funcion():
    """DOCSTRING"""
    variable_2=2
    global variable_1
    variable_1 += 3 #Qué sucede si quito la línea anterior? Y si además quito el +?
    print("EN FUNCION\n\nVariable 1={1:.2f}\nVariable 2={0:.2f}".format(variable_2,variable_1))


variable_1 = 1
funcion()
print("\n\nAFUERA\n\nVariable 1={0:.2f}".format(variable_1))