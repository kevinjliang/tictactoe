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
    O = 10
    GRIDSIZE = 3;
    
    # X/O shapes in all 9 locations, for the image
    Xs = np.zeros((125,125,9))
    Os = np.zeros((125,125,9))
    
    # Diagonals
    DIAG1 = [(0, 0), (1, 1), (2, 2)]
    DIAG2 = [(0, 2), (1, 1), (2, 0)]
    
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
        
        # Win conditions
        self.rowWin = np.zeros(3)
        self.colWin = np.zeros(3)
        self.diag1Win = 0
        self.diag2Win = 0
        
    def getImage(self):
        return self.image
        
    def getGrid(self):
        return self.grid
        
    def move(self,player,position):
        '''
        Player "player" makes a move to the position "position"
        
        return 0 -> no winner yet
        return self.X -> X has won
        return self.O -> O has won
        
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
        
        self.updateWinCheck(player,row,col)
        
        return self.checkWin()
    
    def updateImage(self,player,position):
        '''
        Adds an X/O symbol at the proper location of the image of the grid
        '''
        if player==self.X:
            self.image = self.image + self.Xs[:,:,position-1]
        elif player==self.O:
            self.image = self.image + self.Os[:,:,position-1]
    
    def updateWinCheck(self,player,row,col):
        '''
        Update win check conditions
        '''
        self.rowWin[row] += player
        self.colWin[col] += player
        
        if row==col:
            self.diag1Win += player
        if row==(2-col):
            self.diag2Win += player
        
    def checkWin(self):
        '''
        Check if a player has won
        
        return 0 -> no winner yet
        return self.X -> X has won
        return self.O -> O has won
        
        return -1 -> error
        '''
        
        # Check rows
        if 3*self.X in self.rowWin:
            return self.X
        elif 3*self.O in self.rowWin:
            return self.O

        # Check columnss
        if 3*self.X in self.colWin:
            return self.X
        elif 3*self.O in self.colWin:
            return self.O

        # Check diagonals
        if 3*self.X==self.diag1Win or 3*self.X==self.diag2Win:
            return self.X
        elif 3*self.O==self.diag1Win or 3*self.O==self.diag2Win:
            return self.O

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
        
        # Reset win conditions
        self.rowWin = np.zeros(3)
        self.colWin = np.zeros(3)
        self.diag1Win = 0
        self.diag2Win = 0
        
###############################################################################
## tttGame - tic-tac-toe game between DeepRL and optimalAI
##
        
#class tttGame:        
    
        
        
###############################################################################
## tttGame - tic-tac-toe AI: Newell and Simon
##
## Optimal strategey pre-coded. DeepRL will train against it.
## Takes in the actual grid of tttGrid as input  
        
class optimalAI:        
    def __init__(self,identity,tttgrid):
        # X: 1, O: 10
        self.identity = identity
        self.opponent = [x for x in [tttgrid.X,tttgrid.O] if x!=identity][0]
        
        # Control how often the optimal AI deviates from optimum; default to makinga random move 10% of the time
        self.difficulty = 0.9
        
        # Flag to have AI announce rule being followed; default to off
        self.announce = False
    
    def setDifficulty(self,difficulty):
        self.difficulty = difficulty
        
    def setAnnounce(self,announceSetting):
        self.announce = announceSetting
    
    def ply(self,tttgrid):
        '''
        Make either optimal or random move, depending on the AI's difficulty
        '''
        if np.random.uniform()<self.difficulty:
            return self.optimalMove(tttgrid)
        else:
            moveOptions = [rc2pos(i,j) for i in range(3) for j in range(3) if tttgrid.grid[i,j]==0]
            move = np.random.choice(moveOptions)            
            if self.announce:
                print("###{0} made a random move!: {1}".format(self.identity,move))
            return move
    
    def optimalMove(self,tttgrid):
        '''
        Follow rules in order, according to Newell, Simon
        '''
       
        # Rule 1: Check if AI can win
        move = self.potentialWin(self.identity,tttgrid)
        if move !=0:
            if self.announce:
                print("***{0} is going for the win!: {1}".format(self.identity,move))
            return move
        
        # Rule 2: Block potential opponent win
        move = self.potentialWin(self.opponent,tttgrid)        
        if move !=0:
            if self.announce:
                print("***{0} is blocking {1}'s win!: {2}".format(self.identity,self.opponent,move))
            return move
            
        # Rule 3: Make a potential fork
        forks = self.potentialFork(self.identity,tttgrid)
        if forks:
            if self.announce:
                print("***{0} is making a fork!: {1}".format(self.identity,move))
            return np.random.choice(forks)
        
        # Rule 4: Block an opponent's fork, with a 2-in-a-row if possible
        oppForks = self.potentialFork(self.opponent,tttgrid)
        if oppForks:
            move = self.twoInARow(self.identity,tttgrid,oppForks)
            if move !=0:
                if self.announce:
                    print("***{0} is blocking {1}'s fork with a 2-in-a-row!: {2}".format(self.identity,self.opponent,move))
                return move
            else:
                if self.announce:
                    print("***{0} is blocking {1}'s fork!: {2}".format(self.identity,self.opponent,move))
                return np.random.choice(oppForks)
                
        # Rule 5: Take the center
        move = self.takeCenter(tttgrid)
        if move !=0:
            if self.announce:
                print("***{0} is taking the center!: {1}".format(self.identity,move))
            return move
            
        # Rule 6: Take corner opposite of opponent
        move = self.oppositeCorner(tttgrid)
        if move !=0:
            if self.announce:
                print("***{0} is taking the opposite corner!: {1}".format(self.identity,move))
            return move
            
        # Rule 7: Take a corner
        move = self.takeCorner(tttgrid)
        if move !=0:
            if self.announce:
                print("***{0} is taking a corner!: {1}".format(self.identity,move))
            return move

        # Rule 8: Take a side
        move = self.takeSide(tttgrid)
        if move !=0:
            if self.announce:
                print("***{0} is a side! {1}".format(self.identity,move))
            return move
        
        # This should never run:
        return -1
        
    def potentialWin(self,player,tttgrid):
        '''
        Rule 1/2: Take the move that wins; if not, block move that allows opponent to win
        Returns position of move that allows "player" to win, or 0 if no such move exists
        '''        
        
        # Check rows
        if 2*player in tttgrid.rowWin:
            row = np.where(tttgrid.rowWin==2*player)[0][0]
            col = np.where(tttgrid.grid[row,:]==0)[0][0]
            return rc2pos(row,col)

        # Check columnss
        if 2*player in tttgrid.colWin:
            col = np.where(tttgrid.colWin==2*player)[0][0]
            row = np.where(tttgrid.grid[:,col]==0)[0][0]
            return rc2pos(row,col)

        # Check diagonals
        if 2*player==tttgrid.diag1Win:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG1[i]] == 0:
                    return rc2pos(tttgrid.DIAG1[i][0],tttgrid.DIAG1[i][1])  
        if 2*player==tttgrid.diag2Win:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG2[i]] == 0:
                    return rc2pos(tttgrid.DIAG2[i][0],tttgrid.DIAG2[i][1])
                    
        return 0
    
    def potentialFork(self,player,tttgrid):
        '''
        Rule 3/4: Take the move that creates a fork; if not, block move that allows opponent to fork
        Returns position of move that allows "player" to fork, or 0 if no such move exists
        '''  
        
        # Fork potential - list of places that would generate a two-in-a-row
        # If a location shows up a second time, it is a fork potential
        potential = []
        forks = []
        
        # Check rows
        for i in range(3):
            if tttgrid.rowWin[i] == player:
                for j in range(3):
                    if tttgrid.grid[i][j] == 0:
                        potential.append(rc2pos(i,j))
                        
        # Check columns
        for j in range(3):
            if tttgrid.colWin[j] == player:
                for i in range(3):
                    if tttgrid.grid[i][j] == 0:
                        pos = rc2pos(i,j)                        
                        if pos in potential:
                            forks.append(pos)
                        else:
                            potential.append(pos)
        
        # Check diagonals
        if tttgrid.diag1Win == player:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG1[i]] == 0:
                    pos = rc2pos(tttgrid.DIAG1[i][0],tttgrid.DIAG1[i][1])         
                    if pos in potential:
                        forks.append(pos)
                    else:
                        potential.append(pos)           
        if tttgrid.diag2Win == player:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG2[i]] == 0:
                    pos = rc2pos(tttgrid.DIAG2[i][0],tttgrid.DIAG2[i][1])         
                    if pos in potential:
                        forks.append(pos)
                    else:
                        potential.append(pos)           
                    
        return forks
    
    def twoInARow(self,player,tttgrid,oppForks):
        '''
        Rule 4: Block an opponent's fork with a 2-in-row if possible
        '''
        potential = []
        
        # Check rows
        for i in range(3):
            if tttgrid.rowWin[i] == player:
                for j in range(3):
                    if tttgrid.grid[i][j] == 0:
                        if rc2pos(i,j) in oppForks:
                            return rc2pos(i,j)
                        else:
                            potential.append(rc2pos(i,j))
                        
        # Check columns
        for j in range(3):
            if tttgrid.colWin[j] == player:
                for i in range(3):
                    if tttgrid.grid[i][j] == 0:
                        if rc2pos(i,j) in oppForks:
                            return rc2pos(i,j)
                        else:
                            potential.append(rc2pos(i,j))
        
        # Check diagonals
        if tttgrid.diag1Win == player:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG1[i]] == 0:
                    pos = rc2pos(tttgrid.DIAG1[i][0],tttgrid.DIAG1[i][1])                
                    if pos in oppForks:
                        return pos
                    else:
                        potential.append(pos)            
        if tttgrid.diag2Win == player:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG2[i]] == 0:
                    pos = rc2pos(tttgrid.DIAG2[i][0],tttgrid.DIAG2[i][1])              
                    if pos in oppForks:
                        return pos
                    else:
                        potential.append(pos)
        
        if not potential:
            return 0
        else:
            return np.random.choice(potential)
            
    def takeCenter(self,tttgrid):
        '''
        Rule 5: Take the center, if available        
        '''
        if tttgrid.grid[1,1]==0:
            return 5
        else:
            return 0
                    
    def oppositeCorner(self,tttgrid):
        '''
        Rule 6: Take the corner opposite from the opponent
        '''   
        availableCorners = []
        
        if tttgrid.grid[0][0] == self.opponent and tttgrid.grid[2][2] == 0:
            availableCorners.append(9)
        if tttgrid.grid[0][2] == self.opponent and tttgrid.grid[2][0] == 0:
            availableCorners.append(7)
        if tttgrid.grid[2][0] == self.opponent and tttgrid.grid[0][2] == 0:
            availableCorners.append(3)        
        if tttgrid.grid[2][2] == self.opponent and tttgrid.grid[0][0] == 0:
            availableCorners.append(1)
        
        if not availableCorners:
            return 0
        else:
            return np.random.choice(availableCorners)
    
    def takeCorner(self,tttgrid):
        '''
        Rule 7: Take an empty corner
        '''   
        availableCorners = []
        
        if tttgrid.grid[2][2] == 0:
            availableCorners.append(9)
        if tttgrid.grid[2][0] == 0:
            availableCorners.append(7)
        if tttgrid.grid[0][2] == 0:
            availableCorners.append(3)        
        if tttgrid.grid[0][0] == 0:
            availableCorners.append(1)
        
        if not availableCorners:
            return 0
        else:
            return np.random.choice(availableCorners)
            
    def takeSide(self,tttgrid):
        '''
        Rule 8: Take an empty side
        '''   
        availableSides = []
        
        if tttgrid.grid[0][1] == 0:
            availableSides.append(2)
        if tttgrid.grid[1][0] == 0:
            availableSides.append(4)
        if tttgrid.grid[1][2] == 0:
            availableSides.append(6)        
        if tttgrid.grid[2][1] == 0:
            availableSides.append(8)
        
        if not availableSides:
            return 0
        else:
            return np.random.choice(availableSides)
                    
                    
# Convert row and column coordinates (row,col) to position number (1-9)                    
def rc2pos(row,col):
    return 3*row + col + 1          
    
# Convert position number (1-9) to row and column coordinates (row,col) 
def pos2rc(pos):
    row = (pos-1)//3
    col = (pos%3)-1
    return row,col
                    
                    
                    
                    
                    
                    
        
        
        