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
class testOptAI:
    
    def __init__(self):
        self.testWinResult = []
        self.testBlockWinResult = []
        self.testForkResult = []
        self.testBlockForkResult = []
        
    def simulate2OptAIs():
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
    
    def testAll(self):
        '''
        Run all tests below
        '''
        self.testWinResult = self.testWin()
        self.testBlockWinResult = self.testBlockWin()
        self.testForkResult = self.testFork()
        self.testBlockForkResult = self.testBlockFork()
        
        print("Win test result: {0}".format(self.testWinResult))
        print("Block Win test result: {0}".format(self.testBlockWinResult))
        print("Fork test results: {0}".format(self.testForkResult))
        print("Block fork test results: {0}".format(self.testBlockForkResult))
    
    def testWin(self):
        game = ttt.tttGrid()
        
        aiX = ttt.optimalAI(game.X,game,1)
        aiO = ttt.optimalAI(game.O,game,1)
        
        game.move(game.X,1)
        game.move(game.O,3)
        game.move(game.X,5)
        game.move(game.O,6)
                
        for i in range(100):
            moveX = aiX.ply(game)
            moveO = aiO.ply(game)
            
            if moveX != 9:
                print("AI X didn't go for win at iteration {0}".format(i))
                return False
            if moveO != 9:
                print("AI O didn't go for win at iteration {0}".format(i))
                return False
        
        game.newGame()
        
        game.move(game.X,1)
        game.move(game.O,9)
        game.move(game.X,2)
        game.move(game.O,6)
        game.move(game.X,4)
        game.move(game.O,8)
        
        Xmoves = set()
        Omoves = set()
        
        wins = (3,7)
        
        for i in range(100):
            moveX = aiX.ply(game)
            moveO = aiO.ply(game)
            
            Xmoves.add(moveX)
            Omoves.add(moveO)
            
            if moveX not in wins:
                print("AI X didn't go for one of the wins at iteration {0}".format(i))
                return False
            if moveO not in wins:
                print("AI O didn't go for one of the wins at iteration {0}".format(i))
                return False
        
        if len(Xmoves)+len(Omoves) != 4:
            print("Someone isn't picking their wins randomly")
            return False
        
        return True
        
    def testBlockWin(self):
        game = ttt.tttGrid()
        
        aiX = ttt.optimalAI(game.X,game,1)
        aiO = ttt.optimalAI(game.O,game,1)
        
        game.move(game.X,1)
        game.move(game.O,9)
        game.move(game.X,4)
                
        for i in range(100):
            moveO = aiO.ply(game)
            
            if moveO != 7:
                print("AI O didn't block X's win at iteration {0}".format(i))
                return False
        
        game.move(game.O,7)
        for i in range(100):
            moveX = aiX.ply(game)
            
            if moveX != 8:
                print("AI X didn't block O's win at iteration {0}".format(i))
                return False
        
        return True
        
    def testFork(self):
        game = ttt.tttGrid()
        
        aiX = ttt.optimalAI(game.X,game,1)
        
        game.move(game.X,1)
        game.move(game.O,9)
        game.move(game.X,5)
        game.move(game.O,2)
        
        forkMoves = (4,7)
        
        for i in range(100):
            moveX = aiX.ply(game)
            
            if moveX not in forkMoves:
                print("AI X didn't make fork at iteration {0}".format(i))
                return False
        
        return True
        
    def testBlockFork(self):
        game = ttt.tttGrid()
        
        aiO = ttt.optimalAI(game.O,game,1)
        
        game.move(game.X,1)
        game.move(game.O,9)
        game.move(game.X,5)
        
        goodForkBlocks = (3,7) 
        badForkBlocks = (2,4)
        
        Omoves = set()
        
        for i in range(100):
            moveO = aiO.ply(game)
            Omoves.add(moveO)
            
            if moveO in badForkBlocks:
                print("AI O made a bad fork block at iteration {0}".format(i))
                return False
            elif moveO in goodForkBlocks:
                continue
            else:
                print("AI O didn't block fork at iteration {0}".format(i))
                return False
                
        if len(Omoves) != 2:
            print("AI O isn't picking fork blocks randomly")
            print(Omoves)
            return False
        
        return True