## Tic-Tac-Toe Deep RL Main 
import tttGrid as ttt
import numpy as np
import matplotlib.pyplot as plt

game = ttt.tttGrid()

im = game.image
anX = game.Xs[:,:,1]
im = im + anX

plt.imshow(im)

print('Done')