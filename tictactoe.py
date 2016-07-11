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
import convolutionalNeuralNet as cnn
import theano
import theano.tensor as T
import pickle

class tttGrid:
    # Player IDs
    X = 1
    O = 10
    GRIDSIZE = 3;
    DRAW = 42
    
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
        self.movesTaken = 0
        
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
        return self.DRAW -> game ended in a draw
        
        return -1 -> error
        '''
        self.movesTaken = self.movesTaken+1       
        
        row = (position-1)//self.GRIDSIZE
        col = (position%self.GRIDSIZE)-1
        
        # Mark the spot with player's symbol if it's open
        if self.grid[row, col] == 0:
            self.grid[row, col] = player
        else:
        # Invalid move, symbol already there
            return -1
        
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
        return self.DRAW -> game ended in a draw
        
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
            
        # If no winner, and 9 moves have been taken, board is filled up and game ends in a draw
        if self.movesTaken==9:
            return self.DRAW            
        
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
        self.movesTaken = 0
    
        
        
###############################################################################
## tttGame - tic-tac-toe AI: Newell and Simon
##
## Optimal strategy pre-coded. DeepRL will train against it.
## Takes in the actual grid of tttGrid as input  
        
class optimalAI:        
    def __init__(self,identity,tttgrid,difficulty=0.9):
        # X: 1, O: 10
        self.identity = identity
        self.opponent = [x for x in [tttgrid.X,tttgrid.O] if x!=identity][0]
        
        # Control how often the optimal AI deviates from optimum; default to makinga random move 10% of the time
        self.difficulty = difficulty
        
        # Flag to have AI announce rule being followed; default to off
        self.announce = False
    
    def setDifficulty(self,difficulty):
        self.difficulty = difficulty
        
    def setAnnounce(self,announceSetting):
        self.announce = announceSetting
    
    def ply(self,tttgrid):
        '''
        Make either optimal or random move, depending on the AI's difficulty
        
        tttgrid: tttGrid object defined in this file
        '''
        if np.random.uniform()<self.difficulty:
            move = self.optimalMove(tttgrid)
            return move
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
        wins = []        
        
        # Check rows
        if 2*player in tttgrid.rowWin:
            row = np.where(tttgrid.rowWin==2*player)[0][0]
            col = np.where(tttgrid.grid[row,:]==0)[0][0]
            wins.append(rc2pos(row,col))

        # Check columnss
        if 2*player in tttgrid.colWin:
            col = np.where(tttgrid.colWin==2*player)[0][0]
            row = np.where(tttgrid.grid[:,col]==0)[0][0]
            wins.append(rc2pos(row,col))

        # Check diagonals
        if 2*player==tttgrid.diag1Win:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG1[i]] == 0:
                    wins.append(rc2pos(tttgrid.DIAG1[i][0],tttgrid.DIAG1[i][1]))  
        if 2*player==tttgrid.diag2Win:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG2[i]] == 0:
                    wins.append(rc2pos(tttgrid.DIAG2[i][0],tttgrid.DIAG2[i][1]))
        
        if not wins:    # no wins possible
            return 0
        else:
            return np.random.choice(wins)            
    
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
        twos = []
        
        # Check rows
        for i in range(3):
            if tttgrid.rowWin[i] == player:
                for j in range(3):
                    if tttgrid.grid[i][j] == 0:
                        if rc2pos(i,j) in oppForks:
                            twos.append(rc2pos(i,j))
                        else:
                            potential.append(rc2pos(i,j))
                        
        # Check columns
        for j in range(3):
            if tttgrid.colWin[j] == player:
                for i in range(3):
                    if tttgrid.grid[i][j] == 0:
                        if rc2pos(i,j) in oppForks:
                            twos.append(rc2pos(i,j))
                        else:
                            potential.append(rc2pos(i,j))
        
        # Check diagonals
        if tttgrid.diag1Win == player:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG1[i]] == 0:
                    pos = rc2pos(tttgrid.DIAG1[i][0],tttgrid.DIAG1[i][1])                
                    if pos in oppForks:
                        twos.append(pos)
                    else:
                        potential.append(pos)            
        if tttgrid.diag2Win == player:
            for i in range(3):
                if tttgrid.grid[tttgrid.DIAG2[i]] == 0:
                    pos = rc2pos(tttgrid.DIAG2[i][0],tttgrid.DIAG2[i][1])              
                    if pos in oppForks:
                        twos.append(pos)
                    else:
                        potential.append(pos)
        
        if not twos:
            return 0
        else:
            return np.random.choice(twos)
            
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
                                       

                    
###############################################################################
## tttGame - tic-tac-toe AI: Using deep RL policy gradients
##
## Starts with no prior knowledge of rules or strategy to win, and learns by playing many games
## Takes in 125x125 image of tttGrid as input

class deepAI:
    def __init__(self,alpha=1e-4,gamma=0.95,epsilon=0.02):       
        self.identity = []
        self.opponent = []
        
        # Hyperparameters
        self.alpha = alpha                       # Learning rate
        self.gamma = gamma                       # Discount rate
        self.epsilon = epsilon                   # Exploration rate
        
        # Rewards for various outcomes
        self.r_w = 1                             # win
        self.r_d = -0.05                         # draw
        self.r_l = -1                            # loss
        self.r_b = -5                            # broken rule
        
        
        # Convolutional neural net used as function approximator
        rng = np.random.RandomState(1337)
        self.net = self.tttCNN(rng)              # <---- This might not be quite right          
        
        # Flag to have AI announce move; default to off
        self.announce = False

    
    class tttCNN:
        '''
        Define the convolutional neural network architecture
        
        Currently:
        CNN w/ maxPool -> CNN w/ maxPool -> fully connected layer -> logistic
        '''
        def __init__(self, rng):
            self.x = T.matrix('x')   # the data is presented as rasterized images
            self.y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels            
            
            ## Network Architecture
            # Construct the first convolutional pooling layer:
            self.layer0 = cnn.LeNetConvPoolLayer(
                rng,
                filter_shape=(20, 1, 5, 5),
                poolsize=(2, 2)
            )
            
            # Construct the second convolutional pooling layer
            self.layer1 = cnn.LeNetConvPoolLayer(
                rng,
                filter_shape=(40, 20, 5, 5),
                poolsize=(2, 2)
            )

            # Fully connected hidden layer
            self.layer2 = cnn.HiddenLayer(
                rng,
                n_in= 40 * 4 * 4,
                n_out=500,
                activation=T.tanh
            )

            # Logistic regression with softmax
            self.layer3 = cnn.LogisticRegression(n_in=500, n_out=10)
            
            
          
        def cost(self,images,actions):          
            # Cost function (last layer)
            layer3_out, move = self.forward(images,T.shape(images))
            return self.layer3.negative_log_likelihood(layer3_out,actions)
        
        def forward(self,input,image_shape):
            '''
            Perform a forward pass
            
            TODO: need to figure out how to reformat input into 4D tensor
            TODO: looking into removing image_shape as an input
            '''
            layer0_out = self.layer0.forward(input,image_shape)
            layer1_out = self.layer1.forward(layer0_out,T.shape(layer0_out))
            layer2_out = self.layer2.forward(layer1_out.flatten(2))
            layer3_out, move = self.layer3.forward(layer2_out)
            return layer3_out, move
            

    def setIdentity(self,identity,opponentIdentity):
        # X: 1, O: 10
        # Note: we set the identity here, but the agent does not use this 
        # information when evaluating where to go next when presented a board.
        # Rather, this label is for identifying to the tttGrid which player is 
        # making the move.
        self.identity = identity
        self.opponent = opponentIdentity
        
    def setAnnounce(self,announceSetting):
        self.announce = announceSetting
    
    def loadDeepNet(self,filename):
        '''
        Load the parameters/architecture of the deep net from a file
        '''
        self.net = pickle.load(open(filename,'rb'))
        
    def saveDeepNet(self,filename):
        '''
        Save the deep net to a file
        '''
        pickle.dump(self.net,open(filename,'wb'))
        
    def ply(self,tttGrid):
        '''
        Make either move from deep net (exploitation) or random move (exploration)
        
        Pass in grid, but only use image information
        '''
        image = tttGrid.getImage()
        if np.random.uniform()>self.epsilon:
            # Exploitation: Pick move based on net
            layer3_out, move = self.net.forward(image.reshape(1,1,image.shape[0],image.shape[1]),(1,1,image.shape[0],image.shape[1]))
            
            if self.announce:
                print("***Deep Agent is exploiting with move {0}".format(move))
        else:
            # Exploration: Pick a totally random move
            move = np.random.randint(low=1,high=10)
            
            if self.announce:
                print("***Deep Agent is exploring with move {0}".format(move))
            return move
            
    def updateNetParams(self,images,outcomes,actions):
        '''
        Update parameters with backpropagation
        
        Individually find the (4) gradients that encourage actions that led
        to games that were won (w), drawn (d), lost (l), and ended because 
        of a broken rule (b). Weight gradients by their respective rewards
        
        images:     volume of 125x125 images that the deep agent was presented with
        outcomes:   the eventual outcome of the game (1,0,-1,-2) -> deep agent (won,drawn,loss,broke rule)
        actions:    the action that the deep agent took when presented with each frame in images
        '''
        images_w = images[:,:,outcomes==1]
        images_d = images[:,:,outcomes==0]
        images_l = images[:,:,outcomes==-1]
        images_b = images[:,:,outcomes==-2]
        
        actions_w = actions[outcomes==1]
        actions_d = actions[outcomes==0]
        actions_l = actions[outcomes==-1]
        actions_b = actions[outcomes==-2]
        
        params = self.net.layer3.params + self.net.layer2.params + self.net.layer1.params + self.net.layer0.params        
        
        x_w = T.matrix('x_w')   # the data is presented as rasterized images
        x_d = T.matrix('x_d')
        x_l = T.matrix('x_l')
        x_b = T.matrix('x_b')
        y_w = T.ivector('y_w')  # the labels are presented as 1D vector of
        y_d = T.ivector('y_d')  # [int] labels
        y_l = T.ivector('y_l')  
        y_b = T.ivector('y_b')  
        
        cost = self.net.cost(x_w,y_w)*self.r_w + self.net.cost(x_d,y_d)*self.r_d + self.net.cost(x_l,y_l)*self.r_l + self.net.cost(x_b,y_b)*self.r_b 
        grads = T.grad(cost,params)        
        
#        grad_w = self.gradient(images_w,actions_w,params)
#        grad_d = self.gradient(images_d,actions_d,params)
#        grad_l = self.gradient(images_l,actions_l,params)
#        grad_b = self.gradient(images_b,actions_b,params)
#        
#        grad = grad_w*self.r_w + grad_d*self.r_d + grad_l*self.r_l + grad_b*self.r_b
                
        updates = [
            (param_i, param_i - self.alpha * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]
            
        train_model = theano.function(
            [],
            cost,
            updates=updates,
            givens={
                x_w: images_w,
                y_w: actions_w,
                x_d: images_d,
                y_d: actions_d,
                x_l: images_l,
                y_l: actions_l,
                x_b: images_b,
                y_b: actions_b}
        )
        
        
        cost = train_model()     
        

#    def gradient(self,images,actions,params):
#        '''
#        Find gradient that nudges parameters in direction encouraging the
#        '''       
#        # the cost we minimize during training is the NLL of the model
#        cost = self.net.cost(images,actions)
#        
#        return T.grad(cost, params)

class DeepRL_PolicyGradient:
    '''
    Trains a class "deepAI" agent to play tic-tac-toe on tttGrid by playing
    against a class "optimalAI" agent
    '''
    def __init__(self,deepAI,alpha=1e-4,gamma=0.95,epsilon=0.02):
        self.player = deepAI
        
        # TODO: Figure if this should be placed here or in deepAI
        # Hyperparameters
        self.alpha = alpha                       # Learning rate
        self.gamma = gamma                       # Discount rate
        self.epsilon = epsilon                   # Exploration rate
    
    def train(self,totalGames,updateRate=500,saveRate=500):
        # Set up a tictactoe grid/game
        game = tttGrid()
        
        # Initialize an opponent who plays X and an opponent who plays O
        OpponentAs_X = optimalAI(game.X,game,difficulty=0.7)
        OpponentAs_O = optimalAI(game.O,game,difficulty=0.7)
        
        # Pre-allocate space for maximum number of moves (each an image frame) over games within an update batch
        # The maximum possible number of images/moves occurs when the player is X and every game ends in a tie (5 moves)         
        gameImages = np.empty((game.image.shape[0],game.image.shape[1],5*updateRate))*np.nan
        # Actions taken after each image in gameImages was presented
        gameActions = np.empty(5*updateRate)*np.nan
        # Indicate the eventual winner of each frame
        # (1,0,-1) for player's (win,tie,loss); -2 if rule broken
        gameOutcomes = np.empty(5*updateRate)*np.nan

        
        # Iterator within each batch for indexing into gameImages, gameActions, gameOutcomes
        i = 0     
        
        for n in range(totalGames):
            # Wipe the board for a new game
            game.newGame()
            
            # Randomly assign player and opponent identities
            playerIdentity = np.random.choice([game.X,game.O])
            if playerIdentity == game.X:
                self.player.setIdentity(game.X,game.O)
                playerX = self.player
                playerO = OpponentAs_O
            else:
                self.player.setIdentity(game.O,game.X)
                playerO = self.player
                playerX = OpponentAs_X
            
            winner = 0
            gameStart = i
            playerToGo = playerX
            
            # Play a game to completion
            while winner==0:
                # Save image before move was made
                gameImages[:,:,i] = game.getImage()
                
                # Have player make move
                move = playerToGo.ply(game)
                gameActions[i] = move
                winner = game.move(playerToGo.identity,move)
                
                # If gameover
                if winner==game.X:          # X won
                    if playerToGo.identity!=game.X:
                        # Make sure something weird didn't just happen
                        # TODO: Turn into exception
                        print("Someone won, and it wasn't the player that just went...")
                        return
                    
                    if self.player.identity==game.X:
                        gameOutcomes[gameStart:(i+1)] = 1
                    else:
                        gameOutcomes[gameStart:(i+1)] = -1
                elif winner==game.O:        # O won
                    if playerToGo.identity!=game.O:
                        # Make sure something weird didn't just happen
                        # TODO: Turn into exception
                        print("Someone won, and it wasn't the player that just went...")
                        return
                    
                    if self.player.identity==game.O:
                        gameOutcomes[gameStart:(i+1)] = 1
                    else:
                        gameOutcomes[gameStart:(i+1)] = -1
                elif winner==game.DRAW:     # Game ended in draw
                    gameOutcomes[gameStart:(i+1)] = 0
                elif winner==-1:            # Someone messed up (rule broken)
                    gameOutcomes[gameStart:(i+1)] = -2
                    
                # The other player's turn to go next
                if playerToGo.identity == playerX.identity:
                    playerToGo = playerO
                else:
                    playerToGo = playerX        
                
                # Increment batch counter
                i = i + 1
            
            # Perform update on deep net parameters every "updateRate" number of games                
            if n % updateRate == (updateRate-1):
                imageBatch = gameImages[~np.isnan(gameImages)]
                outcomeBatch = gameOutcomes[~np.isnan(gameOutcomes)]
                actionBatch = gameActions[~np.isnan(gameActions)]
                

###############################################################################
## Miscellaneous helper functions
###############################################################################        
        
 # Convert row and column coordinates (row,col) to position number (1-9)                    
def rc2pos(row,col):
    return 3*row + col + 1          
    
# Convert position number (1-9) to row and column coordinates (row,col) 
def pos2rc(pos):
    row = (pos-1)//3
    col = (pos%3)-1
    return row,col       