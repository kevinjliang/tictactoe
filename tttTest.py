# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:19:07 2016

@author: Kevin Liang
"""

## Tic-Tac-Toe Deep RL tests
import tttGrid as ttt
import matplotlib.pyplot as plt
from time import sleep

def testGameXwins():
    game = ttt.tttGrid()
    
    
    winner = game.move(1,1)
    print("Winner after move 1: {0}".format(winner))
    sleep(1)
    plt.imshow(game.getImage())
    
    winner = game.move(2,2)
    print("Winner after move 2: {0}".format(winner))
    sleep(1)
    plt.imshow(game.getImage())
    
    winner = game.move(1,5)
    print("Winner after move 3: {0}".format(winner))
    sleep(1)    
    plt.imshow(game.getImage())
    
    winner = game.move(2,9)
    print("Winner after move 4: {0}".format(winner))
    sleep(1)
    plt.imshow(game.getImage())    
    
    winner = game.move(1,7)
    print("Winner after move 5: {0}".format(winner))
    sleep(1)
    plt.imshow(game.getImage())
    
    winner = game.move(2,3)
    print("Winner after move 6: {0}".format(winner))
    sleep(1)    
    plt.imshow(game.getImage())
    
    winner = game.move(1,4)
    print("Winner after move 7: {0}".format(winner))
    sleep(1)
    plt.imshow(game.getImage())
    
    print('Game Over')