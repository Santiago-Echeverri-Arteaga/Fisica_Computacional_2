import numpy as np
import pylab as plt



# arr = np.array(['1.2', '2.5', '3.5', '4', '5'], dtype=np.string_)
# print("Tipo: ",arr.dtype)
# float_arr = arr.astype(np.float64)
# print(float_arr)
# print("Tipo de datos de arr",arr.dtype)
# print("Tipo de datos de float_arr",float_arr.dtype)

# A = [[1,2,3],[3,4,6]]


# A = np.array(A)
# print('Elemento 1,1 de A:\n{0} y shape de A={1}'.format(A[1,1],A.shape))

# B = np.eye(8)
# print('B=\n{}'.format(B))

# C = np.linspace(5,10,21)
# D = np.logspace(-3,0,12)

# print('C=\n{}'.format(C))
# print('D=\n{}'.format(D))

# arr= np.arange(10)
# print(arr)
# arr[5:8]=12
# print(arr)

# mat = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# print('mat=\n{}'.format(mat))
# mat = mat[1:3,1:]
# print('Submatriz de mat=\n{}'.format(mat))

data = np.random.randint(0,6,size=(4,10))
# print('data=\n{}'.format(data))

# a = np.array([[1,2,3],[4,5,6],[1,8,9]])
# b = np.array([4,2,0])
# print("a:\n",a)
# print("|a|:\n",np.linalg.det(a))

# l,s,v = np.linalg.svd(a)
# print("l:\n",l)
# print("s:\n",s)
# print("v:\n",v)
# print(np.var(a))

# arr= np.arange(10)
# arr[5:]=0
# print('arr={0}'.format(arr))

# lista=np.array([[1,-3,-1],[1,-4,-1],[-1,3,2]])
# lista_inv = np.linalg.inv(lista)

# print('lista_inv@lista=\n{0}'.format(lista_inv@lista))
# print('sublista=\n{0}'.format(lista[:2,1:]))
 
# a = np.array([[1,2,3],[6,2,0],[2,5,7]])
# print("a:",a)
# b= np.array([4,0,6])
# print("b:",b)
# q,t = np.linalg.qr(a)
# print(t)

names = np.array(["Ana","Bob","Charles","Bob"])
cond = names == "Bob" 
# print('Condicion:',cond)
# xarr = np.array([1.1,1.2,1.3,1.4,1.5])
# yarr = np.array([2.1,2.2,2.3,2.4,2.5])
# result = np.where(cond, xarr, yarr)
# print('Resultado:',result)
# data[data<3]=0
# print(data)

# print(np.arange(32).reshape((2,4,4)))
# arr = np.random.randn(4,4)
# a = np.where(arr<0, 0, arr)
# print("arr:",arr)
# print("a:",a)
data = np.random.randint(0,6,size=(4,4))
print(data)
print("sum:",data.sum())
print("sum:",data.sum(axis=1))
print("sum:",data.sum(axis=0))
    
# a = np.arange(1,17).reshape((4,4))
# b = 5*a-8
# l,u = np.linalg.eig(a)
# print(l)
# print(u)
        
