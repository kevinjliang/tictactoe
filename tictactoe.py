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
import layers
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
        self.image = np.zeros((125,125),dtype=np.int32)
        
        # Draw in tic-tac-toe board
        self.image[39:43,:] = 1
        self.image[82:86,:] = 1
        self.image[:,39:43] = 1
        self.image[:,82:86] = 1
        
        # Load the images of all X/O shapes in all 9 locations
        for i in range(1,10):
            self.Xs[:,:,i-1] = eval('np.loadtxt(\'Data\X{0}.txt\'.format(i))')
            self.Os[:,:,i-1] = eval('np.loadtxt(\'Data\O{0}.txt\'.format(i))')
        
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
            self.image = self.image + np.squeeze(self.Xs[:,:,position-1])
        elif player==self.O:
            self.image = self.image + np.squeeze(self.Os[:,:,position-1])
    
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
        
        # Control how often the optimal AI deviates from optimum; default to making a random move 10% of the time
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
        
        self.N = 500                             # Mini-batch size
        
        # Rewards for various outcomes
        self.r_w = 1                             # win
        self.r_d = -0.05                         # draw
        self.r_l = -1                            # loss
        self.r_b = -5                            # broken rule
        
        rewards = np.array([self.r_w, self.r_d, self.r_l, self.r_b])
        
        self.x = T.tensor4('x')                  # images
        self.a = T.ivector('a')                  # actions
        self.l = T.ivector('l')                  # labels (game outcome)
        rng = np.random.RandomState(1337)
        
        # Convolutional neural net used as function approximator
        # Network for training
        self.trainNet = self.tttCNN(self.x, rng, self.N)
        
        # Network for testing (playing games)
        self.testNet = self.tttCNN(self.x, rng, 1)
        
        # Define loss function,gradients,and update method for training
        self.loss,self.gradients,self.updateParams = self.createGradientFunctions(rewards)
        
        # Flag to have AI announce rule being followed; default to off
        self.announce = False
        
        
#        self.x = T.tensor4('x')                  # images
#        self.a = T.ivector('a')                  # actions
#        
#        self.l = T.ivector('l')                  # labels (game outcome)
##        self.x_r = T.tensor4('x_r')              # images corresponding to label r
##        self.a_r = T.ivector('a_r')              # actions corresponding to label r
#        rng = np.random.RandomState(1337)
#        
#        # Network for training
#        self.trainNet = self.tttCNN(self.x, rng, 500)
#        
#        # Network for testing (playing games)
#        self.testNet = self.tttCNN(self.x, rng, 1)
#        
#        # Symbolic expression of the cost (weighted by rewards) as a function of the actions and outcomes
#        self.cost = self.createCostExpression(self.x,self.a,self.l,rewards)
#        
#        # Function to update parameters
#        self.backprop = self.createUpdateFunction(self.x,self.a,self.l)
#        
#        # Flag to have AI announce move; default to off
#        self.announce = False

    
    class tttCNN:
        '''
        Define the convolutional neural network architecture
        
        Currently:
        CNN w/ maxPool -> CNN w/ maxPool -> fully connected layer -> logistic
        '''
        def __init__(self, input, rng, batch_size):
            '''
            :type input: theano.tensor.dtensor4
            :param input: symbolic image tensor, of shape image_shape
            
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type batch_size: theano.tensor.iscalar
            :param: size of the mini-batch being passed through the network
            
#            :type results: theano.tensor.ivector
#            :param: end result of each game for each image 
#            
#            :type rewards: tuple or list of length 4
#            :param: rewards for (W)in, (D)raw, (L)oss, or (B)roken rule
            '''
            ## Network Architecture
            # Convolutional filters per layer
            nFilters = (15,30,30)
        
            # Construct the first convolutional pooling layer:            
            layer0_input = input.reshape((batch_size, 1, 125, 125))         
            
            # Input to layer 0 is (125,125) image
            # Filtering and then maxpooling results in output size of:
            # ((125-8+1),(125-8+1)/2) = (59,59)
            self.layer0 = layers.LeNetConvPoolLayer(
                rng,
                input=layer0_input,
                image_shape=(batch_size, 1, 125, 125),
                filter_shape=(nFilters[0], 1, 8, 8),
                poolsize=(2,2)
            )
            
            # Construct the second convolutional pooling layer
            # Input to layer 1 is (59,59) image
            # Filtering and then maxpooling results in output size of:
            # ((59-8+1),(59-8+1)/2) = (26,26)
            self.layer1 = layers.LeNetConvPoolLayer(
                rng,
                input=self.layer0.output,
                image_shape=(batch_size, nFilters[0], 59, 59),
                filter_shape=(nFilters[1], nFilters[0], 8, 8),
                poolsize=(2,2)
            )
            
            # Construct the third convolutional pooling layer
            # Input to layer 2 is (26,26) image
            # Filtering and then maxpooling results in output size of:
            # ((26-9+1),(26-9+1)/2) = (9,9)
            self.layer2 = layers.LeNetConvPoolLayer(
                rng,
                input=self.layer1.output,
                image_shape=(batch_size, nFilters[1], 26, 26),
                filter_shape=(nFilters[2], nFilters[1], 9, 9),
                poolsize=(2,2)
            )
            
            # Fully connected hidden layer
            layer3_input = self.layer2.output.flatten(2)            
            
            self.layer3 = layers.HiddenLayer(
                rng,
                input=layer3_input,
                n_in=nFilters[1] * 9*9,
                n_out=500,
                activation=layers.relu
            )

            # Logistic regression with softmax
            self.layer4 = layers.LogisticRegression(input=self.layer3.output, n_in=500, n_out=9)
            
            self.params = self.layer4.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
            
            # Perform forward pass on one image
            self.forward = theano.function([input],self.layer4.y_pred)        
            

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


    def createGradientFunctions(self,rewards):
        ## TODO: rewards is unhappy indexing with a tensor?
        r_vector = T.vector("r_vector")
        
        loss = -T.mean(self.trainNet.layer4.p_y_given_x[T.arange(self.N),self.a] * r_vector[self.l])
        gradients = T.grad(loss,self.trainNet.params)

        updates = [
                (param_i, param_i - self.alpha* grad_i)
                for param_i, grad_i in zip(self.trainNet.params, gradients)
         ]
         
        givens = {r_vector: rewards}

        updateParams = theano.function([self.x,self.a,self.l],loss,updates=updates,givens=givens)
        return loss,gradients,updateParams

        
    def trainModel(self,images,actions,outcomes):
        loss = self.updateParams(images,actions,outcomes)
        
        self.testNet.params = self.trainNet.params

        return loss        




#    def createCostExpression(self,images,actions,outcomes,rewards):
#        '''
#        Creates a symbolic expression (theano tensor) representing the total 
#        cost, weighted by the outcome rewards
#        
#        images:     volume of 125x125 images that the deep agent was presented with
#        outcomes:   the eventual outcome of the game (1,0,-1,-2) -> deep agent (won,drawn,loss,broke rule)
#        actions:    the action that the deep agent took when presented with each frame in images
#        rewards:    reward function for each possible game outcome
#        '''
#        a_r = T.ivector('a_r')
#        l_r = T.iscalar('l_r')
#        
#        costFunction = theano.function(
#                            [l_r],
#                            self.trainNet.layer4.negative_log_likelihood(a_r),
#                            givens={
#                                self.x: images[T.eq(outcomes,l_r).nonzero(),:,:,:][0],
#                                a_r: actions[T.eq(outcomes,l_r).nonzero()]
#                            }
#                        )
#        
#        totalCost = rewards[0]*costFunction(1)+rewards[1]*costFunction(0)+rewards[2]*costFunction(-1)+rewards[3]*costFunction(-2)
#        
#        return totalCost
#     
#    def createUpdateFunction(self,images,actions,outcomes):
#        grads = T.grad(self.cost,self.trainNet.params)
#         
#        updates = [
#                (param_i, param_i - self.alpha* grad_i)
#                for param_i, grad_i in zip(self.trainNet.params, grads)
#         ]
#         
#        backprop = theano.function(
#            [images,actions,outcomes],
#            self.cost,
#            updates=updates    
#        )
#        
#        return backprop
#    
#    def updateNetParams(self,images,actions,outcomes):
#        iterationCost = self.backprop(images,actions,outcomes)
#        print(iterationCost)
#        
#        self.testNet.params = self.trainNet.params
        
#    def createGradientFunction(self,images,actions,outcomes):
#        '''
#        Creates a symbolic expression (theano tensor) representing the gradient
#        of the cost function with respect to the network parameters
#        
#        images:     volume of 125x125 images that the deep agent was presented with
#        outcomes:   the eventual outcome of the game (1,0,-1,-2) -> deep agent (won,drawn,loss,broke rule)
#        actions:    the action that the deep agent took when presented with each frame in images
#        '''
#        
#        grads = T.grad(self.cost(actions,outcomes),self.net.params)
#        
#        updates = [
#                (param_i, param_i - self.alpha* grad_i)
#                for param_i, grad_i in zip(self.net.params, grads)
#        ]
#        
#        backprop = theano.function(
#            [images,actions,outcomes],
#            self.net.layer4.errors(actions),
#            updates=updates    
#        )
#        
#        return backprop
    
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
            move = self.testNet.forward(image.reshape(1,1,image.shape[0],image.shape[1]))
            
            if self.announce:
                print("***Deep Agent is exploiting with move {0}".format(move))
        else:
            # Exploration: Pick a totally random move
            move = np.random.randint(low=1,high=10)
            
            if self.announce:
                print("***Deep Agent is exploring with move {0}".format(move))
        
        return move
            

            
            
            
            
            
            


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