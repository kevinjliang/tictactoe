# tictactoe

My first explorations into deep learning (and Theano). A summary of the project can be found [here](https://github.com/kevinjliang/tictactoe/blob/master/Presentations/2016_09_16/2016_09_16_KevinLiang_TicTacToe.pdf).

This repo contains an implementation of the game of Tic-tac-toe, as well as a computer player that plays the game according to the Newell and Simon algorithm (https://github.com/WesleyyC/Tic-Tac-Toe, https://en.wikipedia.org/wiki/Tic-tac-toe#Strategy). Also included is my deep learning agent, which learns to play the game through reinforcement learning policy gradients. Importantly, the deep learning agent only receives game information in the form of a 64x64 image of the board, as opposed to the 3x3 state. As such, this agent needs to simultaneously learn how to "see" and interpret the board, as well as game strategy. This is achieved by playing many practice rounds against the Newell-Simon agent.

My code is still in its developmental state at the moment. It runs, but isn't particularly organized. Apologies. I'll get around to it eventually. 

Please email any questions to kevin.liang@duke.edu
