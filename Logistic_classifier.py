# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:21:56 2017

@author: lenovo
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#downloading MNIST data from tensorflow
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#creating placeholders for parameters that are to be calculated
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#placeholder for input labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#defining our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#implemeting cross entropy function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#applying optimization algorithm to reduce the loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#launch the model
sess = tf.InteractiveSession()

#initialize variables
tf.global_variables_initializer().run()

#training our model
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#calculating how many predictions are correct
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#calculating the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#acuracy on our test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))