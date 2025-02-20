# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:09:56 2021

@author: hola-
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.image as mpimg

#Función lambda para poder trabajar en cm
cm2inch = lambda x:x/2.54 

#Definición de figuras y ajuste de ejes
fig_1 = plt.figure(constrained_layout=True,figsize=(cm2inch(13),cm2inch(13)))
fig_2 = plt.figure(constrained_layout=False,figsize=(cm2inch(20),cm2inch(20)))
fig_2.subplots_adjust(left=0.05, bottom=0, right=0.974, top=0.99, wspace=0.23, hspace=0.0)
fig_3 = plt.figure(constrained_layout=True)
fig_4 = plt.figure(constrained_layout=True)

#Definición de rejillas para las figuras 1 y 2
gs_1 = fig_1.add_gridspec(4,3)
gs_2 = fig_2.add_gridspec(2,2)
gs_4 = fig_4.add_gridspec(2,2)

#########################################
##
##              PANELES
##
#########################################
#Definición de paneles figura 1
ax1 = fig_1.add_subplot(gs_1[0, :])
ax2 = fig_1.add_subplot(gs_1[1, -1])
ax3 = fig_1.add_subplot(gs_1[2:, -2:])
ax4 = fig_1.add_subplot(gs_1[-2:, 0])
ax5 = fig_1.add_subplot(gs_1[1, :2])

#Duplico eje ax1
ax1_duplicado = ax1.twinx()

#Definición de paneles figura 2
panel1 = fig_2.add_subplot(gs_2[0, 0])
panel2 = fig_2.add_subplot(gs_2[0, 1])
panel3 = fig_2.add_subplot(gs_2[1, 0])
#panel4 = fig_2.add_subplot(gs_2[1, 1])

#Definición de paneles figura 3
axes_1 = fig_3.add_axes([0.5, 0.1, 0.4, 0.3],facecolor='#666600')
axes_2 = fig_3.add_axes([0.2, 0.5, 0.3, 0.4])
axes_3 = fig_3.add_axes([0.55, 0.5, 0.02, 0.3])

#Definición de paneles figura 4
primero = fig_4.add_subplot(gs_4[0, 0], projection='polar')
segundo = fig_4.add_subplot(gs_4[0, 1], projection='3d')
tercero = fig_4.add_subplot(gs_4[1, 0], projection='3d')
cuarto = fig_4.add_subplot(gs_4[1, 1])

#Inser a axes1
ax1_inset = zoomed_inset_axes(axes_1, zoom=3, loc='upper right')


#########################################
##
##          FUNCIONES A GRAFICAR
##
#########################################
#Construcción de funciones a graficar

#axes1 y su inset
x =  np.linspace(-np.pi,np.pi,100)
x_gauss =  np.linspace(-10,10,100)
y_gauss = np.exp(-x**2/0.5)

#ax1
y_sin = np.sin(x)
y_cos = np.cos(2*x)

#ax2
mu= 100
sigma = 15
x_hist = mu + sigma*np.random.randn(10000)

#ax3
x_decay_teo = np.arange(0.1, 4, 0.01)
x_decay_exp = np.arange(0.1, 4, 0.5)
y_decay_teo = np.exp(-x_decay_teo+0.3)
y_decay_exp = np.exp(-x_decay_exp)

yerr = np.abs(0.2 - 0.06*x_decay_exp)

#ax4
bar_x = ["Ana","Bob","Charles"]
heigh_x = [52, 20, 87]
heigh2_x = [0, 10, 30]

#ax5
Imagen_ruta = 'C:/Users/DELL/OneDrive/Documentos/U/Trabajo/UQ/Análisis de datos/Codigos/Clase_4/piton.jpg' 
#Panel 1
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0.2, 0)

#Panel 2
panel2_w = 3
panel2_x = np.linspace(-4,4,150)
panel2_y = np.linspace(-4,5,150)
panel2_X,panel2_Y = np.meshgrid(panel2_x, panel2_y)
panel2_U = -1 - panel2_X**2 + panel2_Y
panel2_V = 1 + panel2_X - panel2_Y**2
speed = np.sqrt(panel2_U**2 + panel2_V**2)
panel2_lw = 5*speed / speed.max()

#axes2
axes2_x = np.linspace(-3, 3, 256)
axes2_y = np.linspace(-3, 3, 256)
axes2_X, axes2_Y = np.meshgrid(axes2_x, axes2_y)
axes2_Z = np.sinc(np.sqrt(axes2_X ** 2 + axes2_Y ** 2))

#Primero
#Espiral de Arquímedes
arq_r = np.linspace(0, 3, 300)
arq_theta = np.linspace(0, 4, 300)*np.pi
#Caracol de Pascal
pas_theta = np.linspace(-np.pi, 2*np.pi, 100)
pas_r = 1+2*np.cos(pas_theta)
#Lemniscata
lem_theta = np.linspace(-np.pi, np.pi, 100)
lem_r = 2*np.sqrt((np.sin(2*lem_theta))*((np.sin(2*lem_theta))>0))
#Rosa
ros_theta = np.linspace(-np.pi, np.pi, 800)
ros_r = 3*np.sin(5*ros_theta)

#Segundo
segundo_x = np.linspace(-3, 3, 256)
segundo_y = np.linspace(-3, 3, 256)
segundo_X, segundo_Y = np.meshgrid(segundo_x, segundo_y)

#Función sinc
#segundo_Z = np.sinc(np.sqrt(4*segundo_X ** 2 + segundo_Y ** 2))
#segundo_y2 = np.sinc(np.sqrt(4*segundo_x ** 2))
#segundo_y3 = np.sinc(np.sqrt(segundo_y ** 2))

#Campana
segundo_Z = np.exp(-(4*segundo_X ** 2 + segundo_Y ** 2))
segundo_y2 = np.exp(-(4*segundo_x ** 2))
segundo_y3 = np.exp(-(segundo_x ** 2))

#Tercero
tercero_y = np.linspace(1, 8, 5)
tercero_x = np.linspace(0, 5, 16)
tercero_T, tercero_A = np.meshgrid(tercero_x, tercero_y)
tercero_data = np.exp(-tercero_T * (1. / tercero_A))

tercero_Xi = tercero_T.flatten()
tercero_Yi = tercero_A.flatten()
tercero_Zi = np.zeros(tercero_data.size)
tercero_dx = .25 * np.ones(tercero_data.size)
tercero_dy = .6 * np.ones(tercero_data.size)
tercero_dz = tercero_data.flatten()

#Cuarto
cuarto_x = np.linspace(-4,4,100)
cuarto_z2 = 0.2/((cuarto_x-1)**2+0.2**2)+0.5/((cuarto_x+1)**2+0.5**2)+0.05/((cuarto_x-0)**2+0.05**2)
cuarto_z3 = 0.2/((cuarto_x-1)**2+0.2**2)
cuarto_z4 = 0.5/((cuarto_x+1)**2+0.5**2)
cuarto_z5 = 0.05/((cuarto_x-0)**2+0.05**2)

#########################################
##
##              ESTILO EN AX1
##
#########################################
#Graficas en ax1
axes = [ax1.plot,ax1.plot,ax1_duplicado.plot]
ax1_color=['#994C00','#00FF00','r']
ax1_lw=[1.0,2.0,2.0]
ax1_ls=['-','-','-']
ax1_zorder=[2,2,2]
ax1_estilo = ['r-', 'b-']
ax1_label = [r'sin($x$)',r'cos($x$)',r'$x^2$']
ax1_funcion = [y_sin,y_cos,x**2]

#########################################
##
##              GRAFICAS
##
#########################################
#Graficas en ax1
for i in enumerate(axes):
        i[1](x,ax1_funcion[i[0]],color=ax1_color[i[0]],lw=ax1_lw[i[0]],zorder=ax1_zorder[i[0]],ls=ax1_ls[i[0]],label=ax1_label[i[0]])

#ax1.plot(x,ax1_funcion[0],color=ax1_color[0],lw=ax1_lw[0],zorder=ax1_zorder[0],ls=ax1_ls[0],label=ax1_label[0])
#ax1.plot(x,ax1_funcion[1],color=ax1_color[1],lw=ax1_lw[0],zorder=ax1_zorder[1],ls=ax1_ls[1], label=ax1_label[0])
#ax1_duplicado.plot(x,ax1_funcion[2],color=ax1_color[2],lw=ax1_lw[0],zorder=ax1_zorder[2],ls=ax1_ls[2], label=ax1_label[0])

#Graficas en ax2
ax2.hist(x_hist, edgecolor='r', facecolor='b', rwidth=0.5,
        align = 'right', orientation = 'horizontal')

#Graficas en ax3
ax3.errorbar(x_decay_exp, y_decay_exp, yerr=yerr, color = '#666666', ecolor='b',
             elinewidth=0.5, capsize=3, marker='s', lw = 0.0, label='Experimento')
ax3.plot(x_decay_teo,y_decay_teo, color='r', label='Teoría')

#Graficas en ax4
ax4.barh(bar_x, heigh_x, facecolor='r', edgecolor='b', height=0.5,
        left=[0,10,30])
ax4.barh(bar_x, heigh2_x, facecolor='k', edgecolor='b', height=0.5,
        left=[0,0,0])

#Graficas en ax5 y panel3
img = mpimg.imread(Imagen_ruta)
panel3.imshow(img[:,:,1],cmap='nipy_spectral', interpolation='spline36')
ax5.hist(img[:,:,0].ravel(), bins=25,  fc='k', ec='r')

#Graficas en panel 1
panel1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

#Graficas en panel2
strm = panel2.streamplot(panel2_X, panel2_Y, panel2_U, panel2_V, color=speed,
 linewidth=panel2_lw, cmap='bwr', density=[0.9, 0.2])
fig_2.colorbar(strm.lines,ax=panel2)


#Graficas en axes1 y su inset
col=np.abs(np.linspace(-1,1,len(x_gauss)))
axes_1.plot(x_gauss,y_gauss, c='k')
axes_1.scatter(x_gauss,y_gauss, c = col, s = 10, cmap = 'hot', label='Experimento')
ax1_inset.scatter(x_gauss,y_gauss, c = col, s = 10, cmap = 'hot', label='Experimento')


#Graficas en axes2
cb1 = axes_2.contour(axes2_X,axes2_Y,axes2_Z,levels=350,cmap='inferno',antialiased=False)
fig_3.colorbar(cb1,cax=axes_3,shrink=0.8,ticks=[0.,0.7,1.0],
             orientation='vertical',format='%.2f',drawedges=False)

#Graficas en primero
primero.plot(arq_theta, arq_r, color='r')
primero.plot(pas_theta+(pas_r<0)*np.pi, np.abs(pas_r),color='b',ls='--', lw=2.0)
primero.plot(lem_theta+(lem_r<0)*np.pi, np.abs(lem_r))
primero.plot(ros_theta+(ros_r<0)*np.pi, np.abs(ros_r),color='k')


#Graficas en segundo
# segundo.plot_wireframe(segundo_X, segundo_Y, segundo_Z, cstride=12, rstride=13, color='k', lw=1)
segundo.plot_surface(segundo_X,segundo_Y,segundo_Z,antialiased=True,cmap='inferno')
segundo.plot(segundo_x, segundo_y2, zs=3, zdir='y', lw = 2, color = 'g',ls='--')
segundo.plot(segundo_x, segundo_y3, zs=-3, zdir='x', lw = 0.5, color = 'k')

#Grafica en tercero
tercero.bar3d([0,1,2,3,0,1,2,3,0,1,2,3], [0,0,0,0,1,1,1,1,2,2,2,2], [0,0,0,0,0,0,0,0,0,0,0,0],
         0.5, 0.8, [0.4,0.0,0.0,-0.4, 0.4,0.2,0.5,0.4, -0.4,-0.3,0.3,0.4],
         alpha=0.8, edgecolor='r', color='c')

#Graficas en cuarto
cuarto.plot(cuarto_x,cuarto_z2,color='r')
cuarto.plot(cuarto_x,cuarto_z3,color='b')
plt.fill_between(cuarto_x,cuarto_z2,cuarto_z3, color='g', alpha=0.3)


#########################################
##
##              DECORACIÓN
##
#########################################
#Decoración en las gráficas
ax1.grid(True, 'major', axis='x', lw=2.0)
ax1.xaxis.set_major_locator(MultipleLocator(np.pi))
ax1.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
ax1.set_xticks([-np.pi,0,np.pi])
ax1.set_xticklabels([r'$-\pi$',r'$0$',r'$\pi$'])

ax1.text(-2, 0.5, "a)",fontsize=20,color='b')

ax1.annotate("Texto",xy=(0.01,-1),xytext=(1,0.4),
             arrowprops=dict(width=1.0, facecolor='k', edgecolor= 'k', headwidth=5))

ax1.arrow(0,-1, 3.14-0.5, 0, head_width=0.3, head_length=0.5, zorder=3)

ax3.set_yscale('log')

ax3.axhline(y=0.1, xmin= 0, xmax=2.5, ls='--', lw=1.5, alpha=0.5)
ax3.axvline(x=2.5, ymin=0.01, ymax=1, color='k', ls='--', lw=1.5, alpha=0.5)


ax1_inset.set_ylim(0.9,1.05)
ax1_inset.set_xlim(-1,1)
mark_inset(axes_1, ax1_inset, 3, 2)

ax3.legend(loc=1,fontsize=8)
ax1.set_xlabel(r'$\mathcal{L}(\rho^\hat{O})$',color='r',fontsize=16)
ax1.set_ylabel(r'$\mathcal{L}(\rho^\hat{O})$',color='b',fontsize=16)


plt.show()
