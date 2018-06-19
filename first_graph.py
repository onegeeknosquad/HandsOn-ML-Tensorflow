#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:55:35 2018

@author: mrpotatohead

Getting started with Tensorflow
"""

import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

#sess = tf.Session()
#sess.run(x.initializer)
#sess.run(y.initializer)
#result = sess.run(f)
#print(result)
#
#
##Better Way:
#with tf.Session() as sess:
#    x.initializer.run()
#    y.initializer.run()
#    result = f.eval()
#    print(result)
    
#Even Better Way!
init = tf.global_variables_initializer() # prepare an init node

with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()
    print(result)
    
tf.reset_default_graph()