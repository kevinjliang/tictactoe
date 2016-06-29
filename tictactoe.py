## Tic-Tac-Toe module
## Kevin Liang


###############################################################################
## tttGrid - the tic-tac-toe grid and display mechanisms
##
## Grid position labelled as the following:
## 
##   1  |  2  |  3
##   -------------
##   4  |  5  |  6
##   -------------
##   7  |  8  |  9

import numpy as np

class tttGrid:
    # Player IDs
    X = 1
    O = 2
    GRIDSIZE = 3;
    
    # X/O shapes in all 9 locations, for the image
    Xs = np.zeros((125,125,9))
    Os = np.zeros((125,125,9))
    
    def __init__(self):
        '''
        Initialize an empty playing grid
        '''
        self.grid = np.zeros((self.GRIDSIZE, self.GRIDSIZE))
        self.image = np.zeros((125,125))
        
        # Draw in tic-tac-toe board
        self.image[39:43,:] = 1
        self.image[82:86,:] = 1
        self.image[:,39:43] = 1
        self.image[:,82:86] = 1
        
        # Load the images of all X/O shapes in all 9 locations
        for i in range(1,10):
            self.Xs[:,:,i-1] = eval('np.loadtxt(\'shapes\X{0}.txt\'.format(i))')
            self.Os[:,:,i-1] = eval('np.loadtxt(\'shapes\O{0}.txt\'.format(i))')
        
    def getImage(self):
        return self.image
        
    def getGrid(self):
        return self.grid
        
    def move(self,player,position):
        '''
        Player "player" makes a move to the position "position"
        
        return 0 -> no winner yet
        return 1 -> X has won
        return 2 -> O has won
        
        return -1 -> error
        '''
        row = (position-1)//self.GRIDSIZE
        col = (position%self.GRIDSIZE)-1
        
        if self.grid[row, col]!= 0:
            # Invalid move, symbol already there
            return -1
        
        # Mark the spot with player's symbol
        self.grid[row, col] = player
        
        self.updateImage(player,position)
        
        return self.checkWin()
    
    def updateImage(self,player,position):
        '''
        Adds an X/O symbol at the proper location of the image of the grid
        '''
        if player==1:
            self.image = self.image + self.Xs[:,:,position-1]
        elif player==2:
            self.image = self.image + self.Os[:,:,position-1]
        
    def checkWin(self):
        '''
        Check if a player has won
        
        return 0 -> no winner yet
        return 1 -> X has won
        return 2 -> O has won
        
        return -1 -> error
        '''

        # Check if X won
        x_row = any(np.all(self.grid==self.X,axis=1))       # rows
        x_col = any(np.all(self.grid==self.X,axis=0))       # columns
        x_diag = np.all(list(map(lambda spot: spot==self.X,[r[c] for c,r in enumerate(self.grid)]))) or \
            np.all(list(map(lambda spot: spot==self.X,[r[-c-1] for c,r in enumerate(self.grid)]))) # diagonals
        
        x_wins = x_row or x_col or x_diag
        
        # Check if O won
        o_row = any(np.all(self.grid==self.O,axis=1))       # rows
        o_col = any(np.all(self.grid==self.O,axis=0))       # columns
        o_diag = np.all(list(map(lambda spot: spot==self.O,[r[c] for c,r in enumerate(self.grid)]))) or \
            np.all(list(map(lambda spot: spot==self.O,[r[-c-1] for c,r in enumerate(self.grid)]))) # diagonals
        
        o_wins = o_row or o_col or o_diag        
        
        if x_wins and o_wins:
            return -1       #something went wrong  
        elif x_wins:
            return 1
        elif o_wins:
            return 2

        return 0            
        
        
    def newGame(self):
        '''
        Go back to an empty playing grid
        '''
        self.grid = np.zeros((self.GRIDSIZE, self.GRIDSIZE))
        self.image = np.zeros((125,125))
        
        # Draw in tic-tac-toe board
        self.image[39:43,:] = 1
        self.image[82:86,:] = 1
        self.image[:,39:43] = 1
        self.image[:,82:86] = 1
        
###############################################################################
## tttGame - tic-tac-toe game between DeepRL and optimumAI
##
        
#class tttGame:        
    
        
        
###############################################################################
## tttGame - tic-tac-toe AI: Newell and Simon
##
## Optimal strategey pre-coded. DeepRL will train against it.
## Takes in the actual grid of tttGrid as input  
        
class optimumAI:        
    def __init__(self,identity):
        # X: 1, O: 2
        self.identity = identity
        
        # Control how often the optimum AI deviates from optimum
        self.difficulty = 0.9
    
        
        
        
        
        
        
        
        