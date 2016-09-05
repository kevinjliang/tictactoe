## Tic-Tac-Toe Deep RL Main 
##
## Run tests and other stuff

import tttTest
import tictactoe as ttt
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.RandomState(1337)

#tttTest.testGameXwins()
#
#tttTest.testGameOwins()

#optAItest = tttTest.testOptAI()
#optAItest.testAll()
#optAItest.simulate2OptAIs()

#deepAItest = tttTest.testDeepAI()
#deepAItest.playNGames(1)

deepAItest = tttTest.testDeepAI2()
deepAItest.performOneBatchUpdate()
deepAItest.makeMove()
