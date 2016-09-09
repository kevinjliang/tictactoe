## Tic-Tac-Toe Deep RL Main 
##
## Run tests and other stuff

import tttTest
import tictactoe as ttt
import numpy as np
import theano
import theano.tensor as T
#from theano.sandbox.rng_mrg import MRG_RandomStreams
#import matplotlib.pyplot as plt


#rng = np.random.RandomState(1337)


trainer = ttt.trainDeepAI()
#trainer.loadDeepAIParams('netParams.p')
images,actions,outcomes,duration,who,record = trainer.playNGames(50)
np.savetxt('images.txt',images.reshape((64,-1)))     # Maybe save states instead?
np.savetxt('actions.txt',actions,fmt='%d')
np.savetxt('outcomes.txt',outcomes,fmt='%d')
np.savetxt('duration.txt',duration,fmt='%d')
np.savetxt('who.txt',who,fmt='%d')
np.savetxt('record.txt',record)

trainer.train(gameLimit=1000000)

#x = T.vector('x').astype("int32")
#M = T.matrix('M').astype(theano.config.floatX)
#
#v = M[T.arange(x.shape[0]),x]
#
#f = theano.function([x,M],v)
#
#xval = np.array((1,2,3,4))
#Mval = np.random.rand(5,6)
#print(f(xval,Mval))

#pyx = T.vector('pyx')
#srng = MRG_RandomStreams()
#
#rv_m = srng.multinomial(n=1,pvals=pyx)
#
#f = theano.function([pyx],rv_m)
#
#p = np.array([[0.6,0.4],[0.3,0.7]])
#print(f(p))

