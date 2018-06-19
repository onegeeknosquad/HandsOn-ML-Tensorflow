#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:08:23 2018

@author: mrpotatohead

XOR Classification Problem

Hands on Machine Learning with Scikit Learn and Tensorflow 
Page 262
Figure 10-6
"""
import tensorflow as tf
import numpy as np

X = [[0,0],[0,1],[1,0],[1,1]]
y = [[0],[1],[1],[0]]

n_steps = 50000
#n_epoch = 10000
n_training = len(X)

n_input_nodes = 2
n_hidden_nodes = 5
n_output_nodes = 1
learning_rate = 0.05

X_ = tf.placeholder(tf.float32, shape=[n_training, n_input_nodes], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[n_training, n_output_nodes],name="y-input")

w1 = tf.Variable(tf.random_uniform([n_input_nodes, n_hidden_nodes],-1,1),name="w1")
w2 = tf.Variable(tf.random_uniform([n_hidden_nodes, n_output_nodes],-1,1),name="w2")

bias1 = tf.Variable(tf.zeros([n_hidden_nodes]), name="bias1")
bias2 = tf.Variable(tf.zeros([n_output_nodes]), name="bias2")

layer1 = tf.sigmoid(tf.matmul(X_, w1) + bias1)
output = tf.sigmoid(tf.matmul(layer1, w2) + bias2)

cost = tf.reduce_mean(tf.square(y - output))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for _ in range(n_steps):
    sess.run(train_step, feed_dict={X_: X, y_: y})
    if _ % 1000 == 0:
        print("Batch: ", _)
        print("Inference ", sess.run(output, feed_dict={X_: X, y_: y}))
        print("Cost ", sess.run(cost, feed_dict={X_: X, y_: y}))
        
#Evaluate the Network
test_X = [[0,.17],[1,1],[.9,.1],[.83,.17]] # 0, 0, 1, 1
print(output.eval(feed_dict={X_:test_X}, session=sess))