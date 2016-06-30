# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:19:07 2016

@author: Kevin Liang
"""

## Tic-Tac-Toe Deep RL tests
import tictactoe as ttt
import matplotlib.pyplot as plt

## Grid tests
def testGameXwins():
    game = ttt.tttGrid()
        
    winner = game.move(game.X,1)
    print("Winner after move 1: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.O,2)
    print("Winner after move 2: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.X,5)
    print("Winner after move 3: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.O,9)
    print("Winner after move 4: {0}".format(winner))
    plt.imshow(game.getImage())    
    
    winner = game.move(game.X,7)
    print("Winner after move 5: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.O,3)
    print("Winner after move 6: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.X,4)
    print("Winner after move 7: {0}".format(winner))
    plt.imshow(game.getImage())
    
    print('Game Over')
    

def testGameOwins():
    game = ttt.tttGrid()
    
    winner = game.move(game.X,1)
    print("Winner after move 1: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.O,5)
    print("Winner after move 2: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.X,2)
    print("Winner after move 3: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.O,3)
    print("Winner after move 4: {0}".format(winner))
    plt.imshow(game.getImage())    
    
    winner = game.move(game.X,7)
    print("Winner after move 5: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.O,4)
    print("Winner after move 6: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.X,8)
    print("Winner after move 7: {0}".format(winner))
    plt.imshow(game.getImage())
    
    winner = game.move(game.O,6)
    print("Winner after move 8: {0}".format(winner))
    plt.imshow(game.getImage())
    
    print('Game Over')

## Optimal AI tests
def testOptAI():
    game = ttt.tttGrid()
    aiX = ttt.optimalAI(game.X,game)
    aiX.setDifficulty(0.5)
    aiX.setAnnounce(True)
    aiO = ttt.optimalAI(game.O,game)    
    aiO.setDifficulty(1)    
    aiO.setAnnounce(True)    
    
    aiXMove = aiX.ply(game)
    winner = game.move(aiX.identity,aiXMove)
    print("Winner after move 1: {0}".format(winner))
    plt.imshow(game.getImage())
    
    aiOMove = aiO.ply(game)
    winner = game.move(aiO.identity,aiOMove)
    print("Winner after move 2: {0}".format(winner))
    plt.imshow(game.getImage())
    
    aiXMove = aiX.ply(game)
    winner = game.move(aiX.identity,aiXMove)
    print("Winner after move 3: {0}".format(winner))
    plt.imshow(game.getImage())
    
    aiOMove = aiO.ply(game)
    winner = game.move(aiO.identity,aiOMove)
    print("Winner after move 4: {0}".format(winner))
    plt.imshow(game.getImage())
    
    aiXMove = aiX.ply(game)
    winner = game.move(aiX.identity,aiXMove)
    print("Winner after move 5: {0}".format(winner))
    plt.imshow(game.getImage())
    
    aiOMove = aiO.ply(game)
    winner = game.move(aiO.identity,aiOMove)
    print("Winner after move 6: {0}".format(winner))
    plt.imshow(game.getImage())
    