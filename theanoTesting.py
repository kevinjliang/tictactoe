# -*- coding: utf-8 -*-
"""
Theano testing
Created on Sat Jul  9 16:07:26 2016

@author: kevin_000
"""

import theano
import theano.tensor as T
from theano import pp

x = T.dscalar('x')
y = T.dscalar('y')
z = x**2 + 3*x*y

dzdx = T.grad(z,x)
pp(dzdx)

f = theano.function([x,y],dzdx)
print(f(2,4))
print(f(7,9))