## Tic-Tac-Toe Deep RL Main 
##
## Run tests and other stuff

#import tttTest
import tictactoe as ttt
import numpy as np
#import theano
#import theano.tensor as T
#from theano.sandbox.rng_mrg import MRG_RandomStreams
#import matplotlib.pyplot as plt


#rng = np.random.RandomState(1337)


trainer = ttt.trainDeepAI()
avgRecord = trainer.trainNTimes(10)
np.savetxt('avgRecord.txt',avgRecord,fmt='%d')

##trainer.loadDeepAIParams('netParams.p')
#images,actions,outcomes,duration,who,record = trainer.playNGames(50)
#np.savetxt('images.txt',images.reshape((64,-1)))     # Maybe save states instead?
#np.savetxt('actions.txt',actions,fmt='%d')
#np.savetxt('outcomes.txt',outcomes,fmt='%d')
#np.savetxt('duration.txt',duration,fmt='%d')
#np.savetxt('who.txt',who,fmt='%d')
#np.savetxt('record.txt',record)
#
#trainer.train(gameLimit=15000)



