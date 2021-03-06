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
    Xs = np.zeros((64,64,9))
    Os = np.zeros((64,64,9))
    
    # Diagonals
    DIAG1 = [(0, 0), (1, 1), (2, 2)]
    DIAG2 = [(0, 2), (1, 1), (2, 0)]
    
    def __init__(self):
        '''
        Initialize an empty playing grid
        '''
        self.grid = np.zeros((self.GRIDSIZE, self.GRIDSIZE))
        self.image = np.zeros((64,64),dtype=np.int32)
        
        # Draw in tic-tac-toe board
        self.image[20:22,:] = 1
        self.image[42:44,:] = 1
        self.image[:,20:22] = 1
        self.image[:,42:44] = 1
        
        # Load the images of all X/O shapes in all 9 locations
        for i in range(1,10):
            self.Xs[:,:,i-1] = eval('np.loadtxt(\'Data/X{0}.txt\'.format(i))')
            self.Os[:,:,i-1] = eval('np.loadtxt(\'Data/O{0}.txt\'.format(i))')
        
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
        self.image = np.zeros((64,64))
        
        # Draw in tic-tac-toe board
        self.image[20:22,:] = 1
        self.image[42:44,:] = 1
        self.image[:,20:22] = 1
        self.image[:,42:44] = 1
        
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
    def __init__(self,alpha=1e-3,gamma=0.97,epsilon=0.0):       
        self.identity = []
        
        # Hyperparameters
        self.alpha = alpha                       # Learning rate
        self.gamma = gamma                       # Discount rate
        self.epsilon = epsilon                   # Exploration rate
        
        # Rewards for various outcomes
        self.r_w = 1                             # win
        self.r_d = 0                             # draw
        self.r_l = -1                            # loss
        self.r_b = -10                           # broken rule
        
        rewards = np.array([self.r_w, self.r_d, self.r_l, self.r_b],dtype=np.int32)
        
        self.x = T.tensor4('x')                  # images
        self.a = T.ivector('a')                  # actions
        self.l = T.ivector('l')                  # labels (game outcome)
        self.d = T.ivector('d')                  # duration (for discount factor)
        self.w = T.ivector('w')                  # who (player identity of move)
        rng = np.random.RandomState(1337)
        
        # Convolutional neural net used as function approximator
        self.trainNet = self.tttCNN(self.x, rng)
        
        # Define loss function,gradients,and update method for training
#        self.loss,self.gradients,self.updateParams = self.createGradientFunctions(rewards)
        self.loss,self.updateParams = self.createGradientFunctions(rewards)
        
        # Flag to have AI announce rule being followed; default to off
        self.announce = False

    
    class tttCNN:
        '''
        Define the convolutional neural network architecture
        
        Currently:
        CNN w/ maxPool -> CNN w/ maxPool -> fully connected layer -> logistic
        '''
        def __init__(self, input, rng):
            '''
            :type input: theano.tensor.dtensor4
            :param input: symbolic image tensor, of shape image_shape
            
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            '''
            ## Network Architecture            
            # Layer 0: Convolutional group (2 cnn layers)
            self.layer0 = layers.convGroup(
                rng,
                input=input,
                filter_shapes=((8,1,3,3),(16,8,3,3)),
                finalpoolsize=(2,2)            
            )
            
            # Layer 1: Convolutional group (2 cnn layers)
            self.layer1 = layers.convGroup(
                rng,
                input=self.layer0.output,
                filter_shapes=((16,16,3,3),(16,16,3,3)),
                finalpoolsize=(2,2)            
            )
            
            # Layer 2: Convolutional group (2 cnn layers)
            self.layer2 = layers.convGroup(
                rng,
                input=self.layer1.output,
                filter_shapes=((16,16,3,3),(16,16,3,3)),
                finalpoolsize=(2,2)            
            )
            
            # Fully connected hidden layer
            layer3_input = self.layer2.output.flatten(2)            
            
            self.layer3 = layers.HiddenLayer(
                rng,
                input=layer3_input,
                n_in=16 * 8*8,
                n_out=30,
                activation=layers.relu
            )

            # Logistic regression with softmax
            self.layer4 = layers.LogisticRegression(rng,input=self.layer3.output, n_in=30, n_out=9)
            
            self.params = self.layer4.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
            
            # Perform forward pass on one image
            self.forward = theano.function([input],self.layer4.p_y_given_x)    
            
#            trng = T.shared_randomstreams.RandomStreams()
#            move = trng.choice(a=T.arange(1,10),p=self.layer4.p_y_given_x)
#            self.forward = theano.function([input],move) 
            

    def setIdentity(self,identity):
        # X: 1, O: 10
        # Note: we set the identity here, but the agent does not use this 
        # information when evaluating where to go next when presented a board.
        # Rather, this label is for identifying to the tttGrid which player is 
        # making the move.
        self.identity = identity
        
    def setAnnounce(self,announceSetting):
        self.announce = announceSetting


    def createGradientFunctions(self,rewards):
        r_vector = T.ivector("r_vector")
        
        # Loss expression for deep agent's moves
        player_loss = -T.mean(self.trainNet.layer4.p_y_given_x[T.arange(self.a.shape[0]),self.a] * r_vector[self.l] * T.pow(self.gamma,self.d) * self.w)
        # Loss for opponent's moves  
        # Opponent breaking rule should not be rewarded
        opponent_loss = -T.mean(self.trainNet.layer4.p_y_given_x[T.arange(self.a.shape[0]),self.a] * r_vector[self.l] * T.neq(self.l,3) * T.pow(self.gamma,self.d) * (1-self.w))
        
        # Total loss is sum of loss of the player minus the loss of the opponent (zero-sum game)
        loss = player_loss - opponent_loss        
#        gradients = T.grad(loss,self.trainNet.params)
#
#        updates = [
#                (param_i, param_i - self.alpha* grad_i)
#                for param_i, grad_i in zip(self.trainNet.params, gradients)
#         ]
        
        updates = self.adam(loss=loss,all_params=self.trainNet.params)
         
        givens = {r_vector: rewards}

        updateParams = theano.function([self.x,self.a,self.l,self.d,self.w],loss,updates=updates,givens=givens)
#        return loss,gradients,updateParams
        return loss,updateParams
    
    def adam(self,loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
        """
        ADAM update rules
        Default values are taken from [Kingma2014]
        References:
        [Kingma2014] Kingma, Diederik, and Jimmy Ba.
        "Adam: A Method for Stochastic Optimization."
        arXiv preprint arXiv:1412.6980 (2014).
        http://arxiv.org/pdf/1412.6980v4.pdf
        """
        updates = []
        all_grads = T.grad(loss, all_params)
        alpha = learning_rate
        t = theano.shared(np.float32(1))
        b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)
    
        for theta_previous, g in zip(all_params, all_grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                                dtype=theano.config.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                                dtype=theano.config.floatX))
    
            m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
            v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
            m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
            v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
            theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
    
            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((theta_previous, theta) )
        updates.append((t, t + 1.))
        return updates
        
    def trainModel(self,images,actions,outcomes,duration,who):
        loss = self.updateParams(images,actions,outcomes,duration,who)

        return loss        

    
    def loadDeepNet(self,filename):
        '''
        Load the parameters/architecture of the deep net from a file
        '''
        self.trainNet = pickle.load(open(filename,'rb'))
        
    def saveDeepNet(self,filename):
        '''
        Save the deep net to a file
        '''
        pickle.dump(self.trainNet,open(filename,'wb'))
        
    def ply(self,tttGrid):
        '''
        Make either move from deep net (exploitation) or random move (exploration)
        
        Pass in grid, but only use image information
        '''
        image = tttGrid.getImage()
        if np.random.uniform()>self.epsilon:
            # Exploitation: Pick move based on net
            p_move = np.squeeze(self.trainNet.forward(image.reshape(1,1,image.shape[0],image.shape[1])))
            move = np.random.choice(a=np.arange(1,10),p=p_move)
            
            if self.announce:
                print("***Deep Agent is exploiting with move {0}".format(move))
        else:
            # Exploration: Pick a totally random move
            move = np.random.randint(low=1,high=10)
            
            if self.announce:
                print("***Deep Agent is exploring with move {0}".format(move))
        
        return move
            
#    def visualizeLayer(self,layer,input_size):
#        # Input that maximizes filter response
#        ## TODO: Need to figure out how to find a number of input images equal to number of filters
#        maxInput = np.asarray(
#            np.random.normal(size=input_size),
#            dtype=theano.config.floatX
#        )
#
#        maxInput = theano.shared(value=maxInput, name='maxInput', borrow=True)                
#        
#        # Energy in output
#        visualLoss = -(layer.output ** 2).sum()   
#        
#        # Update Function
#        updates = self.adam(visualLoss,maxInput)
#        updateFunction = theano.function([],visualLoss,updates=updates)
#        
#        prevLoss = 99999
#                
#        while(True):
#            currentLoss = updateFunction()
#            
#            if abs(prevLoss-currentLoss)<1e-2:
#                return maxInput.get_value
#            
            
            
            
###############################################################################
## Train Protocol
###############################################################################             
class trainDeepAI:
    '''
    Trains a class "deepAI" agent to play tic-tac-toe on tttGrid by playing
    against a class "optimalAI" agent
    '''
    def __init__(self):
        self.deepAI = deepAI()
        self.game = tttGrid()
        self.aiX = optimalAI(self.game.X,self.game,.5)
        self.aiO = optimalAI(self.game.O,self.game,.6)  
        
    def loadDeepAIParams(self,filename):
        '''
        Load the deep net's parameters from a file from a previous run
        Allows for restarting a training in case of an interruption
        '''
        self.deepAI.loadDeepNet(filename)
    
    def trainNTimes(self,N,gameLimit=15000,updateRate=250,saveRate=5000):
        '''
        Run the below train() method N times and average the records
        '''
        avgRecords = np.zeros((4,gameLimit//updateRate))
        
        for i in range(N):
            print('**************** Initializing AI #{0} *****************'.format(i))
            self.deepAI = deepAI()
            records = self.train(gameLimit=gameLimit,updateRate=updateRate,saveRate=saveRate)
            np.savetxt('records{0}.txt'.format(i),records,fmt='%d')
            
            avgRecords = avgRecords + records
        
        return avgRecords/N
        
    def train(self,gameLimit=15000,updateRate=250,saveRate=5000):
        '''
        Train the deepAI agent to play tictactoe
        
        Play games in batches of updateRate, take the game data, and train the agent. 
        '''
        gamesElapsed = 0
        allRecords = np.zeros((4,gameLimit//updateRate))
        recIndex = 0
        
        while(gamesElapsed<gameLimit):
            images,actions,outcomes,duration,who,record = self.playNGames(updateRate)
            allRecords[:,recIndex] = record
            recIndex = recIndex + 1
            print("*****Games Played: {0}".format(gamesElapsed))
            print("Wins: {0} \tDraws: {1} \tLosses: {2} \tBroken: {3}".format(record[0],record[1],record[2],record[3]))
            
            gamesElapsed = gamesElapsed + updateRate
            
            loss = self.deepAI.trainModel(images,actions-1,outcomes,duration,who)        
            print('Loss: {0}\n'.format(loss))
            
            if gamesElapsed % saveRate == 0:
                print("Saving deep net")
                self.deepAI.saveDeepNet('netParams.p')
                
#        np.savetxt('allRecords.txt',allRecords,fmt='%d')
        return allRecords

    def playNGames(self,N,save=False):
        '''
        Play N number of games to completion (or a rule being broken)
        Results in i number of moves (depending on how long each game takes)
        
        save flag allows for recording of game states and vectors of actions, 
        outcomes, duration, who
        
        Returns:
        images (i,1,64,64): Images of the game state before each move
        actions (i)       : The move taken after seeing the aforementioned image
        outcomes (i)      : The eventual result of the game that the image was part of
                                - (0,1,2,3) -> (W,D,L,B)
        duration (i)      : How many moves the game took to finish (for discounting)
        who (i)           : The identity of the player making this move (1 -> deepAI)
        record (4)        : Number of each game outcome within the N games (W,L,D,B)
        '''
        images = np.zeros((N*9,1,64,64),dtype=np.int32)
        actions = np.zeros(N*9,dtype=np.int32)
        outcomes = np.zeros(N*9,dtype=np.int32)
        duration = np.zeros(N*9,dtype=np.int32)
        who = np.zeros(N*9,dtype=np.int32)
        
        if save:
            states = np.zeros((N*9,3,3),dtype=np.int32)

        wins = 0
        draws = 0
        losses = 0
        broken = 0
        
        # Iterator within each batch for indexing into gameImages, gameActions, gameOutcomes
        i = 0   
        gamesCompleted = 0
        
        while(True):
            self.game.newGame()
            
            # Randomly assign player and opponent identities
            playerX,playerO = self.assignPlayerIdentities()
            
            winner = 0
            gameStart = i
            playerToGo = playerX
            
            # Play a game to completion
            while winner==0:
#                print("Player to go: {0}".format(playerToGo.identity))
                
                # Save image before move was made
                images[i,0,:,:] = self.game.getImage()
                if save:
                    states[i,:,:] = self.game.getGrid()
                
                # Have player make move
                move = playerToGo.ply(self.game)
                actions[i] = move
                if playerToGo.identity==self.deepAI.identity:
                    who[i] = 1
                else:           # Must do this in case of rewrite from broken
                    who[i] = 0
                winner = self.game.move(playerToGo.identity,move)
                
                
                # If gameover
                if winner==self.game.X:          # X won                   
                    if self.deepAI.identity==self.game.X:
                        outcomes[gameStart:(i+1)] = 0
                        wins = wins+1
                    else:
                        outcomes[gameStart:(i+1)] = 2
                        losses = losses+1
                elif winner==self.game.O:        # O won
                    if self.deepAI.identity==self.game.O:
                        outcomes[gameStart:(i+1)] = 0
                        wins = wins+1
                    else:
                        outcomes[gameStart:(i+1)] = 2
                        losses = losses+1
                elif winner==self.game.DRAW:     # Game ended in draw
                    outcomes[gameStart:(i+1)] = 1
                    draws = draws+1
                elif winner==-1:            # Someone messed up (rule broken)
                    # Discard images leading up to broken move and reset counter
                    images[gameStart,0,:,:] = images[i,0,:,:]
                    actions[gameStart] = actions[i]
                    who[gameStart] = who[i] 
                    outcomes[gameStart] = 3
                    
                    i = gameStart
                    broken = broken+1
                    
                # Increment batch counter
#                print("{0}: Move {1} made by {2}, resulting in {3}".format(i,actions[i],playerToGo.identity,outcomes[i]))
                i = i + 1
                
                # The other player's turn to go next
                if playerToGo.identity == playerX.identity:
                    playerToGo = playerO
                else:
                    playerToGo = playerX    
                    
            gamesCompleted = gamesCompleted+1
            duration[gameStart:(i)] = i-gameStart
            if gamesCompleted==N:
#                    print("Wins: {0} \nDraws: {1} \nLosses: {2} \nBroken: {3}".format(wins,draws,losses,broken))
                record = [wins,draws,losses,broken]                  
#                    pickle.dump([images,actions,outcomes,duration,record,aiMove],open('stack.p','wb'))
                images = images[0:i]  
                actions = actions[0:i]
                outcomes = outcomes[0:i]
                duration = duration[0:i]
                who = who[0:i]
                
                if save:
                    states = states[0:i,:,:]
                    data = np.vstack((actions,outcomes,duration,who))
                    np.savetxt('states.txt',states.reshape((-1,9)),fmt='%d')
                    np.savetxt('data.txt',data,fmt='%d')
                return images,actions,outcomes,duration,who,record

    def assignPlayerIdentities(self):
        '''
        Randomly assign player and opponent identities
        '''
        playerIdentity = np.random.choice([self.game.X,self.game.O])
        if playerIdentity == self.game.X:
            self.deepAI.setIdentity(self.game.X)
            playerX = self.deepAI
            playerO = self.aiO
            
#                print("DeepAI is {0}".format(self.game.X))
        else:
            self.deepAI.setIdentity(self.game.O)
            playerO = self.deepAI
            playerX = self.aiX
            
        return playerX,playerO
    
 

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
