## Tic-Tac-Toe Deep RL Main 
##
## Run tests and other stuff

import tttTest
import tictactoe as ttt
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.RandomState(1337)


#deepAItest = tttTest.testDeepAI()
#deepAItest.playNMoves(500)


trainer = ttt.trainDeepAI()
trainer.train(moveLimit=100000)

