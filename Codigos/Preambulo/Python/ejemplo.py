# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 18:20:20 2022

@author: DELL
"""
def contador(max):
    print("Dentro contador")
    n=0
    while n<max:
        print("N=",n)
        yield(n)
        print("Despues de YIELD")
        n=n+1
    print("Terminando")

print("Inicio")
mycont=contador(3)
print("Instanciado")
print(mycont)
for i in mycont:
    print("Valor iterador",i)
print("Listo")