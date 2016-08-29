# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:29:21 2016

@author: kevin_000

Layer testing

"""

import theano
from theano import tensor as T
import layers
import numpy as np
import time

startTime = time.time()

datasets = layers.load_data('mnist.pkl.gz')


train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

rng = np.random.RandomState(1337)

classifier = layers.MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=500,
        n_out=10
    )
    
L1_reg=0.00
L2_reg=0.0001

cost = (
    classifier.negative_log_likelihood(y)
    + L1_reg * classifier.L1
    + L2_reg * classifier.L2_sqr
)

gparams = [T.grad(cost, param) for param in classifier.params]

print("Gradients calculated")
print(time.time()-startTime)

learning_rate = 0.01
updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(classifier.params, gparams)
]

print("Starting predictions")
print(time.time()-startTime)
y_predictions = classifier.forward(train_set_x.get_value())   
print("Predictions made")
print(time.time()-startTime)

train_model = theano.function(
    inputs=[],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x.get_value()[0:20],
        y: train_set_y[0:20]
    }
)

print("Training...")
print(time.time()-startTime)
for it in range(25):
    iterCost = train_model()
    
    print(iterCost)
    print(time.time()-startTime)









