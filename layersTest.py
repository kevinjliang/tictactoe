# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:29:21 2016

@author: kevin_000

Layer testing

"""

import theano
from theano import tensor as T
import layers


datasets = layers.load_data('mnist.pkl.gz')


train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x = T.matrix('x')  # data, presented as rasterized images
y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

classifier = layers.LogisticRegression(input=x,n_in=28 * 28, n_out=10)

cost = classifier.negative_log_likelihood(y)

g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

learning_rate = 0.1
updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]
           
           
y_predictions = classifier.forward(train_set_x.get_value())   

train_model = theano.function(
    inputs=[],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x.get_value(),
        y: train_set_y
    }
)

for it in range(25):
    iterCost = train_model()
    
    print(iterCost)









