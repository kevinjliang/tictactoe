# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:19:07 2016

@author: Kevin Liang
"""

## Tic-Tac-Toe Deep RL tests
import tictactoe as ttt
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import layers
import theano
import theano.tensor as T

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
        
    def simulate2OptAIs(self):
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
        plt.figure(0)
        plt.imshow(game.getImage())
        
        aiOMove = aiO.ply(game)
        winner = game.move(aiO.identity,aiOMove)
        print("Winner after move 2: {0}".format(winner))
        plt.figure(1)        
        plt.imshow(game.getImage())
        
        aiXMove = aiX.ply(game)
        winner = game.move(aiX.identity,aiXMove)
        print("Winner after move 3: {0}".format(winner))
        plt.figure(2)        
        plt.imshow(game.getImage())
        
        aiOMove = aiO.ply(game)
        winner = game.move(aiO.identity,aiOMove)
        print("Winner after move 4: {0}".format(winner))
        plt.figure(3)        
        plt.imshow(game.getImage())
        
        aiXMove = aiX.ply(game)
        winner = game.move(aiX.identity,aiXMove)
        print("Winner after move 5: {0}".format(winner))
        plt.figure(4)
        plt.imshow(game.getImage())
        
        aiOMove = aiO.ply(game)
        winner = game.move(aiO.identity,aiOMove)
        print("Winner after move 6: {0}".format(winner))
        plt.figure(5)
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
        
        
        
## Deep AI tests
class testDeepAI:
    def __init__(self):
        self.game = ttt.tttGrid()
        self.deepAI = ttt.deepAI()
        self.aiX = ttt.optimalAI(game.X,game,.1)
        self.aiO = ttt.optimalAI(game.O,game,.1)        
        
    
    def playNGames(self,N):
        updateRate = N
        gameImages = np.empty((self.game.image.shape[0],self.game.image.shape[1],5*updateRate))*np.nan
        gameActions = np.empty(5*updateRate)*np.nan
        gameOutcomes = np.empty(5*updateRate)*np.nan

        wins = 0
        draws = 0
        losses = 0
        broken = 0
        
        # Iterator within each batch for indexing into gameImages, gameActions, gameOutcomes
        i = 0     
        
        for n in range(N):
            self.game.newGame()
            
            # Randomly assign player and opponent identities
            playerIdentity = np.random.choice([self.game.X,self.game.O])
            if playerIdentity == self.game.X:
                self.deepAI.setIdentity(self.game.X,self.game.O)
                playerX = self.deepAI
                playerO = self.aiO
                
                print("DeepAI is {0}".format(self.game.X))
            else:
                self.deepAI.setIdentity(self.game.O,self.game.X)
                playerO = self.deepAI
                playerX = self.aiX
                
                print("DeepAI is {0}".format(self.game.O))
            
            winner = 0
            gameStart = i
            playerToGo = playerX
            
            # Play a game to completion
            while winner==0:
                print("Player to go: {0}".format(playerToGo.identity))
                
                # Save image before move was made
                gameImages[:,:,i] = self.game.getImage()
                
                # Have player make move
                move = playerToGo.ply(self.game)
                gameActions[i] = move
                winner = self.game.move(playerToGo.identity,move)
                
                # If gameover
                if winner==self.game.X:          # X won
                    if playerToGo.identity!=self.game.X:
                        # Make sure something weird didn't just happen
                        # TODO: Turn into exception
                        print("Someone won, and it wasn't the player that just went...")
                        return
                    
                    if self.deepAI.identity==self.game.X:
                        gameOutcomes[gameStart:(i+1)] = 1
                        wins = wins+1
                    else:
                        gameOutcomes[gameStart:(i+1)] = -1
                        losses = losses+1
                elif winner==self.game.O:        # O won
                    if playerToGo.identity!=self.game.O:
                        # Make sure something weird didn't just happen
                        # TODO: Turn into exception
                        print("Someone won, and it wasn't the player that just went...")
                        return
                    
                    if self.deepAI.identity==self.game.O:
                        gameOutcomes[gameStart:(i+1)] = 1
                        wins = wins+1
                    else:
                        gameOutcomes[gameStart:(i+1)] = -1
                        losses = losses+1
                elif winner==self.game.DRAW:     # Game ended in draw
                    gameOutcomes[gameStart:(i+1)] = 0
                    draws = draws+1
                elif winner==-1:            # Someone messed up (rule broken)
                    gameOutcomes[gameStart:(i+1)] = -2
                    broken = broken+1
                    
                # The other player's turn to go next
                if playerToGo.identity == playerX.identity:
                    
                    playerToGo = playerO
                else:
                    playerToGo = playerX        

        
                # Increment batch counter
                i = i + 1
                
        print("Wins: {0} \nDraws: {1} \nLosses: {2} \nBroken: {3}".format(wins,draws,losses,broken))
        
        
        
class testDeepAI2:
    def __init__(self):
        self.game = ttt.tttGrid()
#        self.deepAI = ttt.deepAI()
    
    def makeMove(self):
        tdeepAI = ttt.deepAI(epsilon=0)             # turn off exploration
        tdeepAI.setAnnounce(True)
        move = tdeepAI.ply(self.game)
        return move
        
    def performOneBatchUpdate(self):
        # For simplicity generate images to test on with two optimal AI  
        print("Creating AIs")
        aiX = ttt.optimalAI(self.game.X,self.game,difficulty=.5)
        aiO = ttt.optimalAI(self.game.O,self.game,difficulty=.7)  
        
        # Train on 100 images 
        batch_size = 500 
        
        # Generate images if not already generated
        if os.path.isfile('Data/batchData.p'):
            print("Data found. Unpickling")            
            
            f = open('Data/batchData.p','rb')
            images,actions,labels = pickle.load(f)
            f.close()
        else:
            print("Data not found. Generating...")
            images,actions,labels = self.generateImages(aiX,aiO,batch_size) 
            
            f = open('Data/batchData.p','wb')
            pickle.dump([images,actions,labels],f)
            f.close()
        
        print('Creating Deep AI')
        tdeepAI = ttt.deepAI() 
        print('Training one batch')
        loss = tdeepAI.trainModel(images,actions-1,labels)
        print(loss)
        
        print('Finished')
        
    def generateImages(self,aiX,aiO,batch_size):
        images = np.zeros((batch_size,1,125,125),dtype=np.int32)
        actions = np.zeros(batch_size,dtype=np.int32)
        labels = np.zeros(batch_size,dtype=np.int32)
        
        i = 0
        
        while(True):
            # X goes first
            playerToGo = aiX
            
            # No winner yet
            self.game.newGame()            
            winner = 0
            gameStart = i
            
            # Go until someone wins
            while(winner==0):
                # Save image before move was made
                images[i,0,:,:] = self.game.getImage()
                
                # Have player make move
                move = playerToGo.ply(self.game)
                actions[i] = move
                winner = self.game.move(playerToGo.identity,move)          
                
                # If gameover
                if winner==self.game.X:          # X won
                    if playerToGo.identity!=self.game.X:
                        # Make sure something weird didn't just happen
                        # TODO: Turn into exception
                        print("Someone won, and it wasn't the player that just went...")
                        return
                    
                    if playerToGo.identity==self.game.X:
                        labels[gameStart:(i+1)] = 0
                    else:
                        labels[gameStart:(i+1)] = 2
                elif winner==self.game.O:        # O won
                    if playerToGo.identity!=self.game.O:
                        # Make sure something weird didn't just happen
                        # TODO: Turn into exception
                        print("Someone won, and it wasn't the player that just went...")
                        return
                    
                    if playerToGo.identity==self.game.O:
                        labels[gameStart:(i+1)] = 0
                    else:
                        labels[gameStart:(i+1)] = 2
                elif winner==self.game.DRAW:     # Game ended in draw
                    labels[gameStart:(i+1)] = 1
                elif winner==-1:            # Someone messed up (rule broken)
                    labels[gameStart:(i+1)] = 3
                    
                # The other player's turn to go next
                if playerToGo.identity == aiX.identity:
                    
                    playerToGo = aiO
                else:
                    playerToGo = aiX        

                # Increment batch counter
                i = i + 1   
                if i==batch_size:
                    return images,actions,labels
            
            print(winner)
            
        
        
    def makeNet(self):
        x = T.tensor4('x')                  # images
    #    rng = np.random.RandomState(1337)
        batch_size = T.iscalar('batch_size')   
        
    #    nFilters = (20,50)
    
        layer0_input = x.reshape((batch_size, 1, 125, 125))         
    
        print(T.shape(layer0_input))
    #    layer0 = layers.LeNetConvPoolLayer(
    #        rng,
    #        input=layer0_input,
    #        image_shape=(batch_size, 1, layer0_input.shape[2], layer0_input.shape[3]),
    #        filter_shape=(nFilters[0], 1, 7, 7),
    #        poolsize=(2,2)
    #    )

    def theanoLogicalIndexing(self):
        x = T.tensor4('x')
        l = T.ivector('l')
        l_r = T.iscalar('l_r')
        
        Xval = np.random.randint(0,10,(5,1,3,3))
        Lval = np.array([1,1,0,-1,-2])
        L_r = 1
        
        z = x[T.eq(l,l_r).nonzero(),:,:,:][0]
#        Z = Xval[Lval==1,:,:,:]
#        print(Z)
        
        evalz = theano.function([x,l,l_r],z)
        
        print(Xval)
        print("*********************")
#        print(Lval)
        zEval = evalz(Xval,Lval,L_r)
        print(zEval)
        print(zEval.shape)
    
    def theanoTensorIndexing(self):
        r = T.vector('r')        

        rewards = np.array([1,-0.05,-1,-5])
        
        l = T.ivector('l')
        
        Lval = np.array([1,1,2,3,0,2,0,1,3])
        
        z = r[l]
        evalz = theano.function([l],z,givens={r:rewards})
        
        print(evalz(Lval))
