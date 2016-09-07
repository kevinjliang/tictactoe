## Tic-Tac-Toe Deep RL Main 
##
## Run tests and other stuff

import tttTest
import tictactoe as ttt
import numpy as np
#import matplotlib.pyplot as plt


rng = np.random.RandomState(1337)

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
