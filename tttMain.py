## Tic-Tac-Toe Deep RL Main 
##
## Run tests and other stuff

import tttTest
import tictactoe as ttt
import numpy as np
#import theano
#import theano.tensor as T
#from theano.sandbox.rng_mrg import MRG_RandomStreams
#import matplotlib.pyplot as plt


#rng = np.random.RandomState(1337)

#tttTest.testGameXwins()

#deepAItest = tttTest.testDeepAI()
#deepAItest.playNMoves(500)


trainer = ttt.trainDeepAI()
trainer.train(moveLimit=100000)

#deepAItest = tttTest.testDeepAI()
#game = ttt.tttGrid()
#aiX = ttt.optimalAI(game.X,game,difficulty=.5)
#aiO = ttt.optimalAI(game.O,game,difficulty=.7) 
#deepAItest.generateImages(aiX,aiO,1000)


#pyx = T.vector('pyx')
#srng = MRG_RandomStreams()
#
#rv_m = srng.multinomial(n=1,pvals=pyx)
#
#f = theano.function([pyx],rv_m)
#
#p = np.array([[0.6,0.4],[0.3,0.7]])
#print(f(p))

