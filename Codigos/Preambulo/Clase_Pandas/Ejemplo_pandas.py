# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:22:41 2022

@author: DELL
"""
from IPython.display import display
import pandas as pd
import numpy as np

frame1 = pd.DataFrame( {'id':['ball','pencil','pen','mug','ashtray'],
                        'color': ['white','red','red','black','green'],
                        'brand': ['OMG','ABC','ABC','POD','POD']})
frame2 = pd.DataFrame( {'id':['pencil','pencil','ball','pen'],
                        'brand': ['OMG','ABC',np.nan,'POD']})
out = pd.merge(frame1,frame2,on=['id','brand'],how='outer')
left = pd.merge(frame1,frame2,on=['id','brand'],how='left')
right= pd.merge(frame1,frame2,on=['id','brand'],how='right')
print("FRAME 1")
display(frame1)
print("FRAME 2")
display(frame2)
print("OUT")
display(out)
print("LEFT")
display(left)
print("RIGHT")
display(right)
frame2.columns = ['brand2','id2']
print("JOIN")
display(frame1.join(frame2))
##
##
##
frame1 = pd.DataFrame(np.random.rand(9).reshape(3,3), index=[1,2,3],
columns=['A','B','C'])
frame2 = pd.DataFrame(np.random.rand(9).reshape(3,3), index=[4,5,6],
columns=['A','B','C'])
print("FRAME 1")
display(frame1)
print("FRAME 2")
display(frame2)
print("CONCAT")
display(pd.concat([frame1, frame2]))
##
##
##

frame1 = pd.DataFrame(np.arange(9).reshape(3,3), index=['white','black','red'],
                      columns=['ball','pen','pencil'])
print("FRAME 1")
display(frame1)
ser5 = frame1.stack()
print("STACK")
display(ser5)
ser5.unstack()
print("UNSTACK")
display(ser5)
##
##
##
longframe = pd.DataFrame({ 'color':['white','white','white','red','red','red',
                                    'black','black','black'],
                          'item':['ball','pen','mug','ball','pen','mug','ball',
                                  'pen','mug'], 'value': np.random.rand(9)})
print("longframe")
display(longframe)
print("WIDE")
display(longframe.pivot('color','item'))
##
##
##
frame = pd.DataFrame({ 'item':['ball','mug','pen','pencil','ashtray'],
                      'color':['white','rosso','verde','black','yellow'],
                      'price':[5.56,4.20,1.30,0.56,2.75]})
newcolors = {'rosso': 'red', 'verde': 'green'}
frame=frame.replace(newcolors)
print("MAP")
display(frame)
prices = {
    'ball' : 5.56,'mug' : 4.20,'bottle' : 1.30,'scissors' : 3.41,'pen' : 1.30,
    'pencil' : 0.56,'ashtray' : 2.75}
reindex = {0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth'}
frame.rename(reindex)
print("MAP")
display(frame)
recolumn = {'item':'object','price': 'value'}
frame.rename(index=reindex, columns=recolumn)
print("MAP")
display(frame)