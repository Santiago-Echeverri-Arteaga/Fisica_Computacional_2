class Book:
    """ Clase de prueba para una libreria en donde se debe ingresar el
    título del libro como cadena, una lista de autores y opcionalmente editorial, ISBN y precio.
    Se puede preguntar si dos libros son iguales y ...."""
    def num_authors(self):
        return len(self.authors)
    def __init__(self, T, authors, publisher='Not found', isbn='None', price='Not found'):
        self.title = T
        self.authors = authors[:]
        self.publisher = publisher
        self.isbn = isbn
        self.price = price
        if len(authors) < 3:
            self.autores = 1
        elif len(authors) < 5:
            self.autores = 2
        else:
            self.autores = self.num_authors()/2
    def __str__(self):
        cadena = "El libro seleccionado tiene como título ``{0}'' y su ISBN es: {1}".format(self.title,self.isbn)
        return cadena
    def __eq__(self, other):
        if self.isbn == other.isbn and self.title == other.title:
            return(1)
        elif self.isbn == other.isbn and self.title != other.title:
            return(2)
        else:
            return(0)
   
#####################################    
##  Llamando a la clase Book      ###
#####################################


dato1 = Book("Invierno en Lisboa", ["Ana", "Bob", "Charles", "Dante"],
             publisher='Norma',price=13, isbn=45254)

dato2 = Book("Invierno en Lisboa",["A","B"], "Norma", isbn=5255)


dato3 = Book("Invierno en Lisboa 2", ["Ana", "Bob", "Charles", "Dante", "Edwin", "Fry"],
             publisher='Norma',price=15, isbn=45256)

print(dato1.num_authors())
print(dato3)
#print(dato3.title)
#print(dato3.authors)
#print(dato3.publisher)
#print(dato3.ISBN)
#print(dato3.price)
#print(dato2.autores)
#print(dato3.num_authors())
#x_copy = copy(dato1)

#print(x_copy)
print(dato1==dato2)

class Padre:
    def __init__(self,variable_a):
        self.a = variable_a
    def metodo_1(self):
        return(self.a*2)
    def metodo_2(self):
        return(self.a+'!!!')


class Hijo(Padre):
    def __init__(self,variable_a, variable_b):
        self.a = variable_a
        self.b = variable_b
    def metodo_1(self):
        return(super().metodo_2()+super().metodo_1())
    def metodo_3(self):
        return(self.a+self.b)


# p = Padre('Hi')
# c = Hijo('Hi','bye')
# print("Metodo padre 1: ",p.metodo_1())
# print("Metodo padre 2: ",p.metodo_2())
# print("Metodo hijo 1: ",c.metodo_1())
# print("Metodo hijo 2: ",c.metodo_2())
# print("Metodo hijo 3: ",c.metodo_3())        
#print(c.__dict__)
#x = Padre(5)
#y = Hijo(21,15)
#print(y.metodo_2())


class Building:
     def __init__(self, floors):
         self._floors = floors
     def __setitem__(self, floor_number, data):
         try:
             self._floors[floor_number] = data
         except IndexError:
             self._floors.append(data)
     def __getitem__(self, floor_number):
         if floor_number == "Primero":
             return self._floors[0]
         elif floor_number == "Segundo":
             return self._floors[1]
         elif floor_number == "Tercero":
             return self._floors[2]
         else:
             return self._floors[3]
     def __str__(self):
         return "["+' '.join(self._floors)+"]"

#r = Building(["Sotano","Recepción","Oficinas","Hotel"])
#r[4] = "Vacío"
#print(r["Primero"])
#print(r)
#print(r.__dict__)


#####################################    
##  Clase Building Alterna       ###
#####################################
#class Building(object):
#     def __init__(self, floors):
#         self._floors = ["Vacío"]*floors
#     def occupy(self, floor_number, data):
#          self._floors[floor_number] = data
#     def get_floor_data(self, floor_number):
#          return self._floors[floor_number]
#     def __str__(self):
#         return "["+', '.join(self._floors)+"]"
      
#building1 = Building(6) # Construct a building with 4 floors
#building1.occupy(0,"Recepción")
#building1.occupy(1,"Oficinas")
#building1.occupy(2,"Bodega")

#i = 1
#print("El piso número "+str(i)+" tiene: ",building1.get_floor_data(i))
#print(building1)
#building1.occupy(0, 'Reception')
#building1.occupy(1, 'ABC Corp')
#building1.occupy(2, 'DEF Inc')
#print( building1.get_floor_data(2) )


#building1 = Building(6) # Construct a building with 4 floors
#building1[0] = 'Recepción'
#building1[1] = 'Oficinas'
#building1[2] = 'Bodega'
#print( building1)


class Longitud:
    __metric = {"mm" : 0.001, "cm" : 0.01, "m" : 1, "km" : 1000,
                "in" : 0.0254, "ft" : 0.3048, "yd" : 0.9144,
                "mi" : 1609.344 }   
    def __init__(self, valor, unidad = "m" ):
        self.value = valor
        self.unit = unidad
    
    def Converse2Metres(self):
        return self.value * Longitud.__metric[self.unit]
    
    def Conversor(self,unidad_destino):
        self.value = self.value * Longitud.__metric[self.unit]/Longitud.__metric[unidad_destino]
        self.unit = unidad_destino
    def Comparar(self, otro):
        if self.Converse2Metres()>otro.Converse2Metres():
            print("El elemento mayor es ",self.value,self.unit)
        elif self.Converse2Metres()<otro.Converse2Metres():
            print("El elemento mayor es ",otro.value,otro.unit)
        else:
            print("Son iguales")
    def __add__(self, other):
        if type(other) == int or type(other) == float:
            l = self.Converse2Metres() + other
        else:
            l = self.Converse2Metres() + other.Converse2Metres()
        return Longitud(l/Longitud.__metric[self.unit],self.unit)
    
    
    def __str__(self):
        if self.value != None:
            return str(self.Converse2Metres()) + "m"
        else:
            return "Dato invalido"
        

#####################################    
##  Llamando a la clase Longitud    ###
#####################################
# x = Longitud(3.15,"mm")
# y = Longitud(8, "ft")
# x.Comparar(y)
# x.Conversor("ft")
# z=x+y+5+8
# print(z)
# print(x+y)