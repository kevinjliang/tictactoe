# -*- coding: utf-8 -*-
"""
Theano testing
Created on Sat Jul  9 16:07:26 2016

@author: kevin_000
"""

import tictactoe as ttt
import theano
import theano.tensor as T
from theano import pp

agent = ttt.deepAI()

x = T.matrix('x') 
y = T.ivector('y')

x = x.reshape((1,1,125,125))

p_move, move = agent.net.forward(x,(1,1,125,125))

game = ttt.tttGrid()

image = game.getImage()
image = image.reshape((1,1,125,125))

makeMove = theano.function(
    x,
    move,
)

out = makeMove(image)
print(out)